"""Classes and methods related to candidate spaces."""

import abc
import bisect
import copy
import csv
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from more_itertools import one

from .constants import (
    ESTIMATE,
    METHOD_SCHEME,
    NEXT_METHOD,
    PREDECESSOR_MODEL,
    PREVIOUS_METHODS,
    TYPE_PATH,
    VIRTUAL_INITIAL_MODEL_METHODS,
    Criterion,
    Method,
)
from .handlers import TYPE_LIMIT, LimitHandler
from .model import (
    VIRTUAL_INITIAL_MODEL,
    VIRTUAL_INITIAL_MODEL_HASH,
    Model,
    ModelHash,
    default_compare,
)
from .models import Models
from .petab import get_petab_parameters

__all__ = [
    "BackwardCandidateSpace",
    "BruteForceCandidateSpace",
    "CandidateSpace",
    "FamosCandidateSpace",
    "ForwardCandidateSpace",
    "LateralCandidateSpace",
]


class CandidateSpace(abc.ABC):
    """A base class for collecting candidate models.

    The intended use of subclasses is to identify suitable models in a model
    space, that will be provided to a model selection method for selection.

    Attributes:
        criterion:
            The criterion by which models are compared.
        distances:
            The distances of all candidate models from the initial model.
        predecessor_model:
            The model used for comparison, e.g. for stepwise methods.
        previous_predecessor_model:
            The previous predecessor model.
        excluded_hashes:
            A list of model hashes that will not be accepted into the candidate
            space. The hashes of accepted models are added to :attr:``excluded_hashes``.
        governing_method:
            Used to store the search method that governs the choice of method during
            a search. In some cases, this is always the same as the method attribute.
            An example of a difference is in the bidirectional method, where ``governing_method``
            stores the bidirectional method, whereas `method` may also store the forward or
            backward methods.
        iteration:
            The iteration of model selection.
        limit:
            A handler to limit the number of accepted models.
        models:
            The current set of candidate models.
        method:
            The model selection method of the candidate space.
        predecessor_model:
            The model used for comparison, e.g. for stepwise methods.
        previous_predecessor_model:
            The previous predecessor model.
        retry_model_space_search_if_no_models:
            Whether a search with a candidate space should be repeated upon failure.
            Useful for the :class:`BidirectionalCandidateSpace`, which switches directions
            upon failure.
        summary_tsv:
            A string or :class:`pathlib.Path`. A summary of the model selection progress
            will be written to this file.
        calibrated_models:
            All models that have been calibrated across all iterations of model
            selection.
        latest_iteration_calibrated_models:
            The calibrated models from the most recent iteration.
    """

    """
    FIXME(dilpath)
    - rename previous_predecessor_model to latest_predecessor_model
    - distances: change list to int? Is storage of more than one value useful?
    """

    def __init__(
        self,
        method: Method,
        criterion: Criterion,
        # TODO add MODEL_TYPE = Union[str, Model], str for VIRTUAL_INITIAL_MODEL
        predecessor_model: Model | None = None,
        excluded_hashes: list[ModelHash] | None = None,
        limit: TYPE_LIMIT = np.inf,
        summary_tsv: TYPE_PATH = None,
        previous_predecessor_model: Model | None = None,
        calibrated_models: Models | None = None,
        iteration: int = 0,
    ):
        """See class attributes for arguments."""
        self.method = method

        self.limit = LimitHandler(
            current=self.n_accepted,
            limit=limit,
        )
        self.reset(
            predecessor_model=predecessor_model,
            excluded_hashes=excluded_hashes,
        )

        self.summary_tsv = summary_tsv
        if self.summary_tsv is not None:
            self.summary_tsv = Path(self.summary_tsv)
            self._setup_summary_tsv()

        self.previous_predecessor_model = previous_predecessor_model
        if self.previous_predecessor_model is None:
            self.previous_predecessor_model = self.predecessor_model

        self.set_iteration_user_calibrated_models(Models())
        self.criterion = criterion
        self.calibrated_models = calibrated_models or Models()
        self.latest_iteration_calibrated_models = Models()
        self.iteration = iteration

    def set_iteration_user_calibrated_models(
        self, user_calibrated_models: Models | None
    ) -> None:
        """Hide previously-calibrated models from the calibration tool.

        This allows previously-calibrated model results, e.g. from a previous
        model selection job, to be re-used in this job. Calibrated models are
        stored here between model selection iterations, while the calibration
        tool calibrates the uncalibrated models of the iteration. The models
        are then combined as the full model calibration results for the
        iteration, with :func:`get_iteration_calibrated_models`.

        Args:
            user_calibrated_models:
                The previously-calibrated models.
        """
        if not user_calibrated_models:
            self.iteration_user_calibrated_models = Models()
            return

        iteration_uncalibrated_models = Models()
        iteration_user_calibrated_models = Models()
        for model in self.models:
            if (
                (user_model := user_calibrated_models.get(model.hash, None))
                is not None
            ) and (
                user_model.get_criterion(
                    self.criterion, raise_on_failure=False
                )
                is not None
            ):
                logging.info(f"Using user-supplied result for: {model.hash}")
                user_model_copy = copy.deepcopy(user_model)
                user_model_copy.predecessor_model_hash = (
                    self.predecessor_model.hash
                )
                iteration_user_calibrated_models[user_model_copy.hash] = (
                    user_model_copy
                )
            else:
                iteration_uncalibrated_models.append(model)
        self.iteration_user_calibrated_models = (
            iteration_user_calibrated_models
        )
        self.models = iteration_uncalibrated_models

    def get_iteration_calibrated_models(
        self,
        calibrated_models: Models,
        reset: bool = False,
    ) -> Models:
        """Get all calibrated models for the current iteration.

        The full list of models identified for calibration in an iteration of
        model selection may include models for which calibration results are
        already available. This combines the calibration results of the
        uncalibrated models, with the models that were already calibrated, to
        produce the full list of models that were identified for calibration
        in the current iteration.

        Args:
            calibrated_models:
                The calibration results for the uncalibrated models of this
                iteration. Keys are model hashes, values are models.
            reset:
                Whether to remove the previously calibrated models from the
                candidate space, after they are used to produce the full list
                of calibrated models.
            iteration:
                If provided, the iteration attribute of each model will be set
                to this.

        Returns:
            All calibrated models for the current iteration.
        """
        combined_calibrated_models = (
            self.iteration_user_calibrated_models + calibrated_models
        )
        if reset:
            self.set_iteration_user_calibrated_models(
                user_calibrated_models=Models()
            )
        for model in combined_calibrated_models:
            model.iteration = self.iteration

        return combined_calibrated_models

    def write_summary_tsv(self, row: list[Any]):
        """Write the summary of the last iteration to a TSV file.

        The destination is defined in ``self.summary_tsv``.

        Args:
            row:
                The row that will be written to disk.
        """
        if self.summary_tsv is None:
            return

        # Format single values to be valid rows
        if not isinstance(row, list):
            row = [
                row,
                *([""] * 5),
            ]

        with open(self.summary_tsv, "a", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(row)

    def _setup_summary_tsv(self):
        """Setup the summary TSV file columns."""
        self.summary_tsv.resolve().parent.mkdir(parents=True, exist_ok=True)

        if not self.summary_tsv.exists():
            self.write_summary_tsv(
                [
                    "method",
                    "# candidates",
                    "predecessor change",
                    "current model criterion",
                    "current model",
                    "candidate changes",
                ]
            )

    @classmethod
    def read_arguments_from_yaml_dict(
        cls,
        yaml_dict: dict[str, str],
    ) -> dict[str, str | Model]:
        """Parse settings that were stored in YAML.

        Args:
            yaml_dict:
                The information that was read from the YAML file. Keys are
                class attributes, values are the corresponding values.

        Returns:
            The settings, parsed into PEtab Select objects where possible.
        """
        kwargs = copy.deepcopy(yaml_dict)

        predecessor_model = None
        if (
            predecessor_model_yaml := kwargs.pop(PREDECESSOR_MODEL, None)
        ) is not None:
            predecessor_model = Model.from_yaml(predecessor_model_yaml)

        return {
            **kwargs,
            PREDECESSOR_MODEL: predecessor_model,
        }

    def is_plausible(self, model: Model) -> bool:
        """Determine whether a candidate model is plausible.

        A plausible model is one that could possibly be chosen by
        model selection method, in the absence of information about other
        models in the model space.

        For example, given a forward selection method that starts with an
        initial model ``self.predecessor_model`` that has no estimated
        parameters, then only models with one or more estimated parameters are
        plausible.

        Args:
            model:
                The candidate model.

        Returns:
            ``True`` if ``model`` is plausible, else ``False``.
        """
        return True

    def distance(self, model: Model) -> None | float | int:
        """Compute the distance between two models that are neighbors.

        Args:
            model:
                The candidate model.

        Returns:
            The distance from ``self.predecessor_model`` to ``model``, or
            ``None`` if the distance should not be computed.
        """
        return None

    def accept(
        self,
        model: Model,
        distance: None | float | int,
    ) -> None:
        """Add a candidate model to the candidate space.

        Args:
            model:
                The model that will be added.
            distance:
                The distance of the model from the predecessor model.
        """
        model.predecessor_model_hash = self.predecessor_model.hash
        self.models.append(model)
        self.distances.append(distance)
        self.set_excluded_hashes(model, extend=True)

    def n_accepted(self) -> int:
        """Get the current number of accepted models.

        Returns:
            The number of models.
        """
        return len(self.models)

    def excluded(
        self,
        model_hash: Model | ModelHash,
    ) -> bool:
        """Check whether a model is excluded.

        Args:
            model:
                The model.

        Returns:
            ``True`` if the ``model`` is excluded, otherwise ``False``.
        """
        if isinstance(model_hash, Model):
            model_hash = model_hash.hash
        return model_hash in self.get_excluded_hashes()

    @abc.abstractmethod
    def _consider_method(self, model: Model) -> bool:
        """Consider whether a model should be accepted, according to a method.

        Args:
            model:
                The candidate model.

        Returns:
            Whether a model should be accepted.
        """
        return True

    def consider(self, model: Model | None) -> bool:
        """Add a candidate model, if it should be added.

        Args:
            model:
                The candidate model. This value may be ``None`` if the :class:`ModelSubspace`
                decided to exclude the model that would have been sent.

        Returns:
            Whether it is OK to send additional models to the candidate space. For
            example, if the limit of the number of accepted models has been reached,
            then no further models should be sent.
        """
        # FIXME(dilpath)
        # TODO change to return whether the model was accepted, and instead add
        # `self.continue` to determine whether additional models should be sent.
        # Model was excluded by the `ModelSubspace` that called this method, so can be
        # skipped.
        if model is None:
            # TODO use a different code than `True`?
            return True
        if self.limit.reached():
            return False
        if self.excluded(model):
            warnings.warn(
                f"Model `{model.hash}` has been previously excluded "
                "from the candidate space so is skipped here.",
                RuntimeWarning,
                stacklevel=2,
            )
            return True
        if not self.is_plausible(model):
            return True
        if not self._consider_method(model):
            return True
        self.accept(model, distance=self.distance(model))
        return not self.limit.reached()

    def reset_accepted(self) -> None:
        """Reset the accepted models."""
        self.models = Models()
        self.distances = []

    def set_predecessor_model(self, predecessor_model: Model | None):
        """Set the predecessor model.

        See class attributes for arguments.
        """
        if predecessor_model is None:
            predecessor_model = VIRTUAL_INITIAL_MODEL
        self.predecessor_model = predecessor_model

    def get_predecessor_model(self) -> str | Model:
        """Get the predecessor model."""
        return self.predecessor_model

    def set_excluded_hashes(
        self,
        hashes: Model | ModelHash | list[Model | ModelHash],
        extend: bool = False,
    ) -> None:
        """Set the excluded hashes.

        Args:
            hashes:
                The model hashes that will be excluded.
            extend:
                Whether to replace or extend the current excluded hashes.
        """
        # FIXME refactor to use `Models` and rename `set_excluded_models`?
        if isinstance(hashes, Model | ModelHash):
            hashes = [hashes]
        excluded_hashes = set()
        for potential_hash in hashes:
            if isinstance(potential_hash, Model):
                potential_hash = potential_hash.hash
            excluded_hashes.add(potential_hash)

        if extend:
            self.excluded_hashes = self.get_excluded_hashes() | excluded_hashes
        else:
            self.excluded_hashes = set(excluded_hashes)

    def get_excluded_hashes(self) -> set[ModelHash]:
        """Get the excluded hashes.

        Returns:
            The hashes of excluded models.
        """
        try:
            return self.excluded_hashes
        except AttributeError:
            self.excluded_hashes = set()
            return self.get_excluded_hashes()

    def set_limit(self, limit: TYPE_LIMIT = None) -> None:
        """Set the limit on the number of accepted models.

        Args:
            limit:
                The limit.
        """
        if limit is not None:
            self.limit.set_limit(limit)

    def get_limit(self) -> TYPE_LIMIT:
        """Get the limit on the number of accepted models."""
        return self.limit.get_limit()

    def wrap_search_subspaces(
        self,
        search_subspaces: Callable[[], None],
    ) -> Callable:
        """Decorate the subspace searches of a model space.

        Used by candidate spaces to perform changes that alter the search.
        See :class:`BidirectionalCandidateSpace` for an example, where it's
        used to switch directions.

        Args:
            search_subspaces:
                The method that searches the subspaces of a model space.

        Returns:
            The wrapped ``search_subspaces``.
        """

        def wrapper():
            search_subspaces()

        return wrapper

    def reset(
        self,
        predecessor_model: Model | None = None,
        # FIXME change `Any` to some `TYPE_MODEL_HASH` (e.g. union of str/int/float)
        excluded_hashes: list[ModelHash] | None = None,
        limit: TYPE_LIMIT = None,
    ) -> None:
        """Reset the candidate models, optionally reinitialize with a model.

        Args:
            predecessor_model:
                The initial model.
            excluded_hashes:
                These hashes will extend the current excluded hashes.
            limit:
                The new upper limit of the number of models in this candidate
                space. Defaults to the previous limit.
        """
        if excluded_hashes is None:
            excluded_hashes = self.get_excluded_hashes()
        if limit is None:
            limit = self.limit.get_limit()

        self.set_predecessor_model(predecessor_model)
        self.reset_accepted()
        self.set_excluded_hashes(
            excluded_hashes,
            extend=False,  # FIXME True?
        )
        self.set_limit(limit)

    def distances_in_estimated_parameters(
        self,
        model: Model,
        predecessor_model: Model | None = None,
    ) -> dict[str, float | int]:
        """Distance between two models in model space, using different metrics.

        All metrics are in terms of estimated parameters.

        Metrics:
            l1:
                The L_1 distance between two models.
            size:
                The difference in the number of estimated parameters between two
                models.

        Args:
            model:
                The candidate model.
            predecessor_model:
                See class attributes.

        Returns:
            The distances between the models, as a dictionary, where a key is
            the name of the metric, and the value is the corresponding
            distance.
        """
        model0 = predecessor_model
        if model0 is None:
            model0 = self.predecessor_model
        model1 = model

        if (
            model0.hash != VIRTUAL_INITIAL_MODEL_HASH
            and not model1.model_subspace_petab_yaml.samefile(
                model0.model_subspace_petab_yaml
            )
        ):
            # FIXME
            raise NotImplementedError(
                "Computing distances between models that have different "
                "model subspace PEtab problems is currently not supported. "
                "This check is based on the PEtab YAML file location."
            )

        # All parameters from the PEtab problem are used in the computation.
        if model0.hash == VIRTUAL_INITIAL_MODEL_HASH:
            parameter_ids = list(
                get_petab_parameters(model1._model_subspace_petab_problem)
            )
            if self.method == Method.FORWARD:
                parameters0 = np.array([0 for _ in parameter_ids])
            elif self.method == Method.BACKWARD:
                parameters0 = np.array([ESTIMATE for _ in parameter_ids])
            else:
                raise NotImplementedError(
                    "Distances for the virtual initial model have not yet been "
                    f'implemented for the method "{self.method}". Please notify the'
                    "developers."
                )
        else:
            parameter_ids = list(
                get_petab_parameters(model0._model_subspace_petab_problem)
            )
            parameters0 = np.array(
                model0.get_parameter_values(parameter_ids=parameter_ids)
            )
        parameters1 = np.array(
            model1.get_parameter_values(parameter_ids=parameter_ids)
        )

        # TODO change to some numpy elementwise operation
        estimated0 = np.array([p == ESTIMATE for p in parameters0]).astype(int)
        estimated1 = np.array([p == ESTIMATE for p in parameters1]).astype(int)

        difference = estimated1 - estimated0

        # Changes to(/from) estimated from(/to) not estimated
        l1 = np.abs(difference).sum()

        # Change in the number of estimated parameters.
        size = np.sum(difference)

        # TODO constants? e.g. Distance.L1 and Distance.Size
        distances = {
            "l1": l1,
            "size": size,
        }

        return distances

    def update_after_calibration(
        self,
        *args,
        iteration_calibrated_models: Models,
        **kwargs,
    ):
        """Do work in the candidate space after calibration.

        For example, this is used by the :class:`FamosCandidateSpace` to switch
        methods.

        Different candidate spaces require different arguments. All arguments
        are here, to ensure candidate spaces can be switched easily and still
        receive sufficient arguments.
        """
        self.calibrated_models += iteration_calibrated_models
        self.latest_iteration_calibrated_models = iteration_calibrated_models
        self.set_excluded_hashes(
            self.latest_iteration_calibrated_models,
            extend=False,  # FIXME True?
        )


class ForwardCandidateSpace(CandidateSpace):
    """The forward method class.

    Attributes:
        direction:
            ``1`` for the forward method, ``-1`` for the backward method.
        max_steps:
            Maximum number of steps forward in a single iteration of forward selection.
            Defaults to no maximum (``None``).
    """

    direction = 1

    def __init__(
        self,
        *args,
        predecessor_model: Model | str | None = None,
        max_steps: int = None,
        **kwargs,
    ):
        # Although `VIRTUAL_INITIAL_MODEL` is `str` and can be used as a default
        # argument, `None` may be passed by other packages, so the default value
        # is handled here instead.
        self.max_steps = max_steps
        if predecessor_model is None:
            predecessor_model = VIRTUAL_INITIAL_MODEL
        super().__init__(
            *args,
            method=Method.FORWARD if self.direction == 1 else Method.BACKWARD,
            predecessor_model=predecessor_model,
            **kwargs,
        )

    def is_plausible(self, model: Model) -> bool:
        distances = self.distances_in_estimated_parameters(model)
        n_steps = self.direction * distances["size"]

        if self.max_steps is not None and n_steps > self.max_steps:
            raise StopIteration(
                f"Maximal number of steps for method {self.method} exceeded. Stop sending candidate models."
            )

        # A model is plausible if the number of estimated parameters strictly
        # increases (or decreases, if `self.direction == -1`), and no
        # previously estimated parameters become fixed.
        if self.predecessor_model.hash == VIRTUAL_INITIAL_MODEL.hash or (
            n_steps > 0 and distances["l1"] == n_steps
        ):
            return True
        return False

    def distance(self, model: Model) -> int:
        # TODO calculated here and `is_plausible`. Rewrite to only calculate
        #      once?
        distances = self.distances_in_estimated_parameters(model)
        return distances["l1"]

    def _consider_method(self, model) -> bool:
        """See :meth:`CandidateSpace._consider_method`."""
        distance = self.distance(model)

        # Get the distance of the current "best" plausible model(s)
        distance0 = np.inf
        if self.distances:
            distance0 = one(set(self.distances))
            # TODO store each or just one?
            # distance0 = one(self.distances)
        # breakpoint()

        # Only keep the best model(s).
        if distance > distance0:
            return False
        if distance < distance0:
            self.reset_accepted()
        return True


class BackwardCandidateSpace(ForwardCandidateSpace):
    """The backward method class."""

    direction = -1


def forward_super_and_inner(
    candidate_space: CandidateSpace, method_name: str
) -> None:
    """Decorator to call the method of both the `super()` and `inner` space.

    Useful in the :class:`FamosCandidateSpace`, where e.g. excluded hashes
    should be applied to both the governing `FamosCandidateSpace`, and the
    active `FamosCandidateSpace.inner_candidate_space`.

    If the method is an instance member, `candidate_space` will be used instead
    of `super()`.

    Args:
        candidate_space:
            A candidate space that contains an `inner_candidate_space`.
        method_name:
            The method of the `candidate_space` to decorate.

    Returns:
        A tuple with the output from the method called with
            1. `super()` if possible else `candidate_space`, and
            2. `candidate_space.inner_candidate_space`.
    """

    # TODO check if docs look OK for wrapped methods; try functools.wraps
    # @wraps(getattr(candidate_space, method_name))
    def wrapped_method(*args, **kwargs):
        try:
            super_object = super(type(candidate_space), candidate_space)
            getattr(super_object, method_name)
        except AttributeError:
            super_object = candidate_space
        inner_object = candidate_space.inner_candidate_space

        return (
            getattr(super_object, method_name)(*args, **kwargs),
            getattr(inner_object, method_name)(*args, **kwargs),
        )

    return wrapped_method


def forward_inner(candidate_space: CandidateSpace, method_name: str) -> None:
    """Decorator to call the method of the `inner` space.

    See :func:`super_and_inner` for more details.

    Returns:
        The output from the method called with
        `candidate_space.inner_candidate_space`.
    """

    # TODO check if docs look OK for wrapped methods; try functools.wraps
    # @wraps(getattr(candidate_space, method_name))
    def wrapped_method(*args, **kwargs):
        inner_object = candidate_space.inner_candidate_space
        return getattr(inner_object, method_name)(*args, **kwargs)

    return wrapped_method


class FamosCandidateSpace(CandidateSpace):
    """The FAMoS method class.

    This candidate space implements and extends the original FAMoS
    algorithm (doi: 10.1371/journal.pcbi.1007230).

    Attributes:
        critical_parameter_sets:
            A list of lists, where each inner list contains parameter IDs.
            All models must estimate at least 1 parameter from each critical
            parameter set.
        swap_parameter_sets:
            A list of lists, where each inner list contains parameter IDs.
            The lateral moves in FAMoS are constrained to be between parameters that
            exist in the same swap parameter set.
        method_scheme:
            A dictionary that specifies how to switch between methods when
            the current method doesn't produce a better model.
            Keys are `n`-tuples that described a pattern of length `n`
            methods. Values are methods. If the previous methods match the
            pattern in the key, then the method in the value will be used next.
            The order of the dictionary is important: only the first matched
            pattern will be used.
            Defaults to the method scheme described in the original FAMoS
            publication.
        n_reattempts:
            Integer. The total number of times that a jump-to-most-distance action
            will be performed, triggered whenever the model selection would
            normally terminate. Defaults to no reattempts (``0``).
        consecutive_laterals:
            Boolean. If ``True``, the method will continue performing lateral moves
            while they produce better models. Otherwise, the method scheme will
            be applied after one lateral move.
    """

    default_method_scheme = {
        (Method.BACKWARD, Method.FORWARD): Method.LATERAL,
        (Method.FORWARD, Method.BACKWARD): Method.LATERAL,
        (Method.BACKWARD, Method.LATERAL): None,
        (Method.FORWARD, Method.LATERAL): None,
        (Method.FORWARD,): Method.BACKWARD,
        (Method.BACKWARD,): Method.FORWARD,
        (Method.LATERAL,): Method.FORWARD,
        (Method.MOST_DISTANT,): Method.FORWARD,
        None: Method.FORWARD,
    }

    forwarded_inner = [
        "_consider_method",
    ]
    _consider_method = None
    forwarded_super_and_inner = [
        "reset_accepted",
        "set_predecessor_model",
        "set_excluded_hashes",
        "set_limit",
    ]

    def __init__(
        self,
        *args,
        predecessor_model: Model | None = None,
        critical_parameter_sets: list = [],
        swap_parameter_sets: list = [],
        method_scheme: dict[tuple, str] = None,
        n_reattempts: int = 0,
        consecutive_laterals: bool = False,
        **kwargs,
    ):
        for method_name in FamosCandidateSpace.forwarded_inner:
            setattr(self, method_name, forward_inner(self, method_name))
        for method_name in FamosCandidateSpace.forwarded_super_and_inner:
            setattr(
                self, method_name, forward_super_and_inner(self, method_name)
            )

        self.critical_parameter_sets = critical_parameter_sets
        self.swap_parameter_sets = swap_parameter_sets

        self.method_scheme = method_scheme
        if method_scheme is None:
            self.method_scheme = FamosCandidateSpace.default_method_scheme

        # FIXME remove and use `self.method` everywhere in the constructor
        #       instead -- not required in other methods anymore
        self.initial_method = self.method_scheme[None]
        self.method = self.initial_method
        self.method_history = [self.initial_method]

        if predecessor_model is None:
            predecessor_model = VIRTUAL_INITIAL_MODEL

        if (
            predecessor_model.hash == VIRTUAL_INITIAL_MODEL.hash
            and critical_parameter_sets
        ) or (
            predecessor_model.hash != VIRTUAL_INITIAL_MODEL.hash
            and not self.check_critical(predecessor_model)
        ):
            raise ValueError(
                f"Provided predecessor model {predecessor_model.parameters} does not contain necessary critical parameters {self.critical_parameter_sets}. Provide a valid predecessor model."
            )

        if (
            predecessor_model.hash == VIRTUAL_INITIAL_MODEL.hash
            and self.initial_method not in VIRTUAL_INITIAL_MODEL_METHODS
        ):
            raise ValueError(
                f"The initial method {self.initial_method} does not support the `VIRTUAL_INITIAL_MODEL` as its predecessor_model."
            )

        # FIXME remove `None` from the resulting `inner_methods` set?
        inner_methods = set.union(
            *[
                {
                    *(
                        method_pattern
                        if method_pattern is not None
                        else (None,)
                    ),
                    next_method,
                }
                for method_pattern, next_method in self.method_scheme.items()
            ]
        )
        if Method.LATERAL in inner_methods and not self.swap_parameter_sets:
            raise ValueError(
                "Use of the lateral method with FAMoS requires `swap_parameter_sets`."
            )

        for method in inner_methods:
            if method is not None and method not in [
                Method.FORWARD,
                Method.LATERAL,
                Method.BACKWARD,
                Method.MOST_DISTANT,
            ]:
                raise NotImplementedError(
                    f"Methods FAMoS can swap to are `Method.FORWARD`, `Method.BACKWARD` and `Method.LATERAL`, not {method}. \
                    Check if the method_scheme scheme provided is correct."
                )

        self.inner_candidate_spaces = {
            Method.FORWARD: ForwardCandidateSpace(
                *args,
                predecessor_model=predecessor_model,
                **kwargs,
            ),
            Method.BACKWARD: BackwardCandidateSpace(
                *args,
                predecessor_model=predecessor_model,
                **kwargs,
            ),
            Method.LATERAL: LateralCandidateSpace(
                *args,
                predecessor_model=predecessor_model,
                max_steps=1,
                **kwargs,
            ),
        }
        self.inner_candidate_space = self.inner_candidate_spaces[
            self.initial_method
        ]

        super().__init__(
            *args,
            method=self.method,
            predecessor_model=predecessor_model,
            **kwargs,
        )

        self.n_reattempts = n_reattempts

        self.consecutive_laterals = consecutive_laterals
        if (
            not self.consecutive_laterals
            and (Method.LATERAL,) not in self.method_scheme
        ):
            raise ValueError(
                "Please provide a method to switch to after a lateral search, "
                "if not enabling the `consecutive_laterals` option."
            )

        if self.n_reattempts:
            # TODO make so max_number can be specified? It cannot in original FAMoS.
            self.most_distant_max_number = 100
        else:
            self.most_distant_max_number = 1

        # TODO update to new `Models` type. Currently problematic because the
        # order of `best_models` matters. This would be fine for `Models` to
        # handle, except that `Models._update` will use a pre-existing index
        # if there is a model with the same hash. FIXME regenerate the expected
        # FAMoS models when `best_models` never contains duplicate models...
        # Also add a `sort` method to `Models` to sort by criterion.
        self.best_models = []
        self.best_model_of_current_run = predecessor_model

        self.jumped_to_most_distant = False
        self.swap_done_successfully = False

    @classmethod
    def read_arguments_from_yaml_dict(cls, yaml_dict) -> dict:
        kwargs = copy.deepcopy(yaml_dict)

        if (method_scheme_raw := kwargs.pop(METHOD_SCHEME, None)) is not None:
            method_scheme = {
                (
                    tuple(
                        [
                            Method(method_str)
                            for method_str in definition[PREVIOUS_METHODS]
                        ]
                    )
                    if definition[PREVIOUS_METHODS] is not None
                    else None
                ): definition[NEXT_METHOD]
                for definition in method_scheme_raw
            }
            kwargs[METHOD_SCHEME] = method_scheme

        return super().read_arguments_from_yaml_dict(yaml_dict=kwargs)

    def update_after_calibration(
        self,
        *args,
        iteration_calibrated_models: Models,
        **kwargs,
    ) -> None:
        """See `CandidateSpace.update_after_calibration`."""
        super().update_after_calibration(
            *args,
            iteration_calibrated_models=iteration_calibrated_models,
            **kwargs,
        )

        # In case we jumped to most distant in the last iteration,
        # there's no need for an update, so we reset the jumped variable
        # to False and continue to candidate generation
        if self.jumped_to_most_distant:
            self.jumped_to_most_distant = False
            jumped_to_model = one(iteration_calibrated_models)
            self.set_predecessor_model(jumped_to_model)
            self.previous_predecessor_model = jumped_to_model
            self.best_model_of_current_run = jumped_to_model
            return False

        if self.update_from_iteration_calibrated_models(
            iteration_calibrated_models=iteration_calibrated_models,
        ):
            logging.info("Switching method")
            self.switch_method()
            self.switch_inner_candidate_space(
                excluded_hashes=self.calibrated_models,
            )
            logging.info(
                "Method switched to ", self.inner_candidate_space.method
            )

        self.method_history.append(self.method)

    def update_from_iteration_calibrated_models(
        self,
        iteration_calibrated_models: Models,
    ) -> bool:
        """Update ``self.best_models`` with the latest ``iteration_calibrated_models``
        and determine if there was a new best model. If so, return
        ``False``. ``True`` otherwise.
        """
        go_into_switch_method = True
        for model in iteration_calibrated_models:
            if (
                self.best_model_of_current_run.hash
                == VIRTUAL_INITIAL_MODEL_HASH
                or default_compare(
                    model0=self.best_model_of_current_run,
                    model1=model,
                    criterion=self.criterion,
                )
            ):
                go_into_switch_method = False
                self.best_model_of_current_run = model

            if len(
                self.best_models
            ) < self.most_distant_max_number or default_compare(
                model0=self.best_models[self.most_distant_max_number - 1],
                model1=model,
                criterion=self.criterion,
            ):
                self.insert_model_into_best_models(
                    model_to_insert=model,
                )

        self.best_models = self.best_models[: self.most_distant_max_number]

        # When we switch to LATERAL method, we will do only one iteration with this
        # method. So if we do it successfully (i.e. that we found a new best model), we
        # want to switch method. This is why we put go_into_switch_method to True, so
        # we go into the method switching pipeline
        if self.method == Method.LATERAL and not self.consecutive_laterals:
            self.swap_done_successfully = True
            go_into_switch_method = True
        return go_into_switch_method

    def insert_model_into_best_models(self, model_to_insert: Model) -> None:
        """Inserts a model into the list of best_models which are sorted
        w.r.t. the criterion specified.
        """
        insert_index = bisect.bisect_left(
            [
                model.get_criterion(self.criterion)
                for model in self.best_models
            ],
            model_to_insert.get_criterion(self.criterion),
        )
        self.best_models.insert(insert_index, model_to_insert)

    def consider(self, model: Model | None) -> bool:
        """Re-define ``consider`` of FAMoS to be the ``consider`` method
        of the ``inner_candidate_space``. Update all the attributes
        changed in the ``consider`` method.
        """
        if self.limit.reached():
            return False

        # Check if model contains necessary critical parameters and, if
        # the current method is swap, check the swap move is contained
        # in a swap set.

        if self.check_critical(model) and (
            not self.method == Method.LATERAL or self.check_swap(model)
        ):
            return_value = self.inner_candidate_space.consider(model)

            # update the attributes
            self.models = self.inner_candidate_space.models
            self.distances = self.inner_candidate_space.distances
            self.set_excluded_hashes(
                self.inner_candidate_space.get_excluded_hashes()
            )

            return return_value

        return True

    def is_plausible(self, model: Model) -> bool:
        if self.check_critical(model):
            return self.inner_candidate_space.is_plausible(model)
        return False

    def check_swap(self, model: Model) -> bool:
        """Check if parameters that are swapped are contained in the
        same swap parameter set.
        """
        if self.method != Method.LATERAL:
            return True

        predecessor_estimated_parameters_ids = set(
            self.predecessor_model.get_estimated_parameter_ids()
        )
        estimated_parameters_ids = set(model.get_estimated_parameter_ids())

        swapped_parameters_ids = estimated_parameters_ids.symmetric_difference(
            predecessor_estimated_parameters_ids
        )

        for swap_set in self.swap_parameter_sets:
            if swapped_parameters_ids.issubset(set(swap_set)):
                return True
        return False

    def check_critical(self, model: Model) -> bool:
        """Check if the model contains all necessary critical parameters"""
        estimated_parameters_ids = set(model.get_estimated_parameter_ids())
        for critical_set in self.critical_parameter_sets:
            if not estimated_parameters_ids.intersection(set(critical_set)):
                return False
        return True

    def switch_method(
        self,
    ) -> None:
        """Switch to the next method with respect to the history
        of methods used and the switching scheme in ``self.method_scheme``.
        """
        previous_method = self.method
        next_method = previous_method
        logging.info("SWITCHING", self.method_history)
        # If last method was LATERAL and we have made a succesfull swap
        # (a better model was found) then go back to FORWARD. Else do the
        # usual method swapping scheme.
        if self.swap_done_successfully:
            next_method = self.method_scheme[(Method.LATERAL,)]
        else:
            # iterate through the method_scheme dictionary to see which method to switch to
            for previous_methods in self.method_scheme:
                if previous_methods is not None and previous_methods == tuple(
                    self.method_history[-len(previous_methods) :]
                ):
                    next_method = self.method_scheme[previous_methods]
                    # if found a switch just break (choosing first good switch)
                    break

        # raise error if the method didn't change
        if next_method == previous_method:
            raise ValueError(
                "Method didn't switch when it had to. "
                "The `method_scheme` provided is not sufficient. "
                "Please provide a correct method_scheme scheme. "
                f"Method history: `{self.method_history}`. "
            )

        # Terminate if next method is `None`
        if next_method is None:
            if self.n_reattempts:
                self.jump_to_most_distant()
                return
            raise StopIteration(
                f"The next method is {next_method}. The search is terminating."
            )

        # If we try to switch to SWAP method but it's not available (no crit or swap parameter groups)
        if (
            next_method == Method.LATERAL
            and not [
                critical_set
                for critical_set in self.critical_parameter_sets
                if len(critical_set) > 1
            ]
            and not self.swap_parameter_sets
        ):
            if self.n_reattempts:
                self.jump_to_most_distant()
                return
            raise ValueError(
                f"The next method is {next_method}, but there are no critical or swap parameters sets. Terminating."
            )
        if previous_method == Method.LATERAL:
            self.swap_done_successfully = False
        self.update_method(method=next_method)

    def update_method(self, method: Method):
        """Update ``self.method`` to ``method``."""
        self.method = method

    def switch_inner_candidate_space(
        self,
        excluded_hashes: list[ModelHash],
    ):
        """Switch the inner candidate space to match the current method.

        Args:
            excluded_hashes:
                Hashes of excluded models.
        """
        # if self.method != Method.MOST_DISTANT:
        self.inner_candidate_space = self.inner_candidate_spaces[self.method]
        # reset the next inner candidate space with the current history of all
        # calibrated models
        self.inner_candidate_space.reset(
            predecessor_model=self.predecessor_model,
            excluded_hashes=excluded_hashes,
        )

    def jump_to_most_distant(
        self,
    ):
        """Jump to most distant model with respect to the history of all
        calibrated models.
        """
        predecessor_model = self.get_most_distant()

        logging.info("JUMPING: ", predecessor_model.parameters)

        # if model not appropriate make it so by adding the first
        # critical parameter from each critical parameter set
        if not self.check_critical(predecessor_model):
            for critical_set in self.critical_parameter_sets:
                # FIXME is this a good idea? probably better to request
                #       the model from the model subspace, rather than editing
                #       the parameters...
                predecessor_model.parameters[critical_set[0]] = ESTIMATE

        # self.update_method(self.initial_method)
        self.update_method(Method.MOST_DISTANT)

        self.n_reattempts -= 1
        self.jumped_to_most_distant = True

        # FIXME rename here `predecessor_model` to `most_distant_model`
        # self.predecessor_model = None
        self.set_predecessor_model(None)
        self.best_model_of_current_run = None
        self.models = Models([predecessor_model])

        self.write_summary_tsv("Jumped to the most distant model.")
        self.update_method(self.method_scheme[(Method.MOST_DISTANT,)])

    # TODO Fix for non-famos model subspaces. FAMOS easy because of only 0;ESTIMATE
    def get_most_distant(
        self,
    ) -> Model:
        """
        Get most distant model to all the checked models. We take models from the
        sorted list of best models (``self.best_models``) and construct complements of
        these models. For all these complements we compute the distance in number of
        different estimated parameters to all models from history. For each complement
        we take the minimum of these distances as it's distance to history. Then we
        choose the complement model with the maximal distance to history.

        TODO:
        Next we check if this model is contained in any subspace. If so we choose it.
        If not we choose the model in a subspace that has least distance to this
        complement model.
        """
        most_distance = 0
        most_distant_indices = []

        # FIXME for multiple PEtab problems?
        parameter_ids = list(
            get_petab_parameters(
                self.best_models[0]._model_subspace_petab_problem
            )
        )

        for model in self.best_models:
            model_estimated_parameters = np.array(
                [
                    p == ESTIMATE
                    for p in model.get_parameter_values(
                        parameter_ids=parameter_ids
                    )
                ]
            ).astype(int)
            complement_parameters = 1 - model_estimated_parameters
            # initialize the least distance to the maximal possible value of it
            complement_least_distance = len(complement_parameters)
            # get the complement least distance
            for calibrated_model in self.calibrated_models:
                calibrated_model_estimated_parameters = np.array(
                    [
                        p == ESTIMATE
                        for p in calibrated_model.get_parameter_values(
                            parameter_ids=parameter_ids
                        )
                    ]
                ).astype(int)

                difference = (
                    calibrated_model_estimated_parameters
                    - complement_parameters
                )
                l1_distance = np.abs(difference).sum()

                if l1_distance < complement_least_distance:
                    complement_least_distance = l1_distance

            # if the complement is further away than the current best one
            # then save it as the new most distant away one
            if complement_least_distance > most_distance:
                most_distance = complement_least_distance
                most_distant_indices = complement_parameters
        if len(most_distant_indices) == 0:
            raise StopIteration("No most_distant model found. Terminating")

        most_distant_parameter_values = [
            str(index).replace("1", ESTIMATE) for index in most_distant_indices
        ]

        most_distant_parameters = dict(
            zip(parameter_ids, most_distant_parameter_values, strict=True)
        )

        most_distant_model = Model(
            model_subspace_petab_yaml=model.model_subspace_petab_yaml,
            model_subspace_id=model.model_subspace_id,
            model_subspace_indices=most_distant_indices,
            parameters=most_distant_parameters,
        )

        return most_distant_model

    def wrap_search_subspaces(self, search_subspaces):
        def wrapper():
            search_subspaces(only_one_subspace=True)

        return wrapper


class LateralCandidateSpace(CandidateSpace):
    """Find models with the same number of estimated parameters."""

    def __init__(
        self,
        *args,
        max_steps: int = None,
        **kwargs,
    ):
        """
        Additional args:
            max_number_of_steps:
                Maximal allowed number of swap moves. If 0 then there is no maximum.
        """
        super().__init__(
            *args,
            method=Method.LATERAL,
            **kwargs,
        )
        self.max_steps = max_steps

    def is_plausible(self, model: Model) -> bool:
        if self.predecessor_model is None:
            raise ValueError(
                "The predecessor_model is still None. Provide an appropriate predecessor_model"
            )

        distances = self.distances_in_estimated_parameters(model)

        # If max_number_of_steps is non-zero and the number of steps made is
        # larger then move is not plausible.
        if self.max_steps is not None and distances["l1"] > 2 * self.max_steps:
            raise StopIteration(
                f"Maximal number of steps for method {self.method} exceeded. Stop sending candidate models."
            )

        # A model is plausible if the number of estimated parameters remains
        # the same, but some estimated parameters have become fixed and vice
        # versa.
        if (
            distances["size"] == 0
            and
            # distances['size'] == 0 implies L1 % 2 == 0.
            # FIXME here and elsewhere, deal with models that are equal
            #       except for the values of their fixed parameters.
            distances["l1"] > 0
        ):
            return True
        return False

    def _consider_method(self, model):
        return True


class BruteForceCandidateSpace(CandidateSpace):
    """The brute-force method class."""

    def __init__(self, *args, **kwargs):
        # if args or kwargs:
        #    # FIXME remove?
        #    # FIXME at least support limit
        #    warnings.warn(
        #        'Arguments were provided but will be ignored, because of the '
        #        'brute force candidate space.'
        #    )
        super().__init__(
            *args,
            method=Method.BRUTE_FORCE,
            **kwargs,
        )

    def _consider_method(self, model):
        return True


candidate_space_classes = {
    Method.FORWARD: ForwardCandidateSpace,
    Method.BACKWARD: BackwardCandidateSpace,
    Method.LATERAL: LateralCandidateSpace,
    Method.BRUTE_FORCE: BruteForceCandidateSpace,
    Method.FAMOS: FamosCandidateSpace,
}


def method_to_candidate_space_class(method: Method) -> type[CandidateSpace]:
    """Get a candidate space class, given its method name.

    Args:
        method:
            The name of the method corresponding to one of the implemented
            candidate spaces.

    Returns:
        The candidate space.
    """
    candidate_space_class = candidate_space_classes.get(method, None)
    if candidate_space_class is None:
        raise NotImplementedError(
            f"The provided method `{method}` does not correspond to an "
            "implemented candidate space."
        )
    return candidate_space_class


'''
def distances_in_estimated_parameters(
    model: Model,
    predecessor_model: Model,
    #method: str = None,
) -> Dict[str, Union[float, int]]:
    """Distance between two models in model space, using different metrics.

    All metrics are in terms of estimated parameters.

    Metrics:
        l1:
            The L_1 distance between two models.
        size:
            The difference in the number of estimated parameters between two
            models.

    Args:
        model:
            The candidate model.
        predecessor_model:
            The initial model.
        method:
            The search method in use. Necessary if the predecessor model is the
            virtual initial model.

    Returns:
        The distances between the models, as a dictionary, where a key is the
        name of the metric, and the value is the corresponding distance.
    """
    model0 = predecessor_model
    model1 = model

    if not model1.petab_yaml.samefile(model0.petab_yaml):
        raise NotImplementedError(
            'Computation of distances between different PEtab problems is '
            'currently not supported. This error is also raised if the same '
            'PEtab problem is read from YAML files in different locations.'
        )

    # All parameters from the PEtab problem are used in the computation.
    parameter_ids = list(model0.petab_parameters)

    parameters0 = np.array(model0.get_parameter_values(parameter_ids=parameter_ids))
    parameters1 = np.array(model1.get_parameter_values(parameter_ids=parameter_ids))

    estimated0 = (parameters0 == ESTIMATE).astype(int)
    estimated1 = (parameters1 == ESTIMATE).astype(int)

    difference = estimated1 - estimated0

    # Changes to(/from) estimated from(/to) not estimated
    l1 = np.abs(difference).sum()

    # Change in the number of estimated parameters.
    size = np.sum(difference)

    # TODO constants?
    distances = {
        'l1': l1,
        'size': size,
    }

    return distances
'''
