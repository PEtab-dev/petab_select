"""Classes and methods related to candidate spaces."""
import abc
import bisect
import copy
import csv
import logging
import os.path
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from more_itertools import one

from .constants import (
    ESTIMATE,
    METHOD,
    METHOD_SCHEME,
    MODELS,
    NEXT_METHOD,
    PREDECESSOR_MODEL,
    PREVIOUS_METHODS,
    TYPE_PATH,
    VIRTUAL_INITIAL_MODEL,
    VIRTUAL_INITIAL_MODEL_METHODS,
    Criterion,
    Method,
)
from .handlers import TYPE_LIMIT, LimitHandler
from .misc import snake_case_to_camel_case
from .model import Model, default_compare


class CandidateSpace(abc.ABC):
    """A base class for collecting candidate models.

    The intended use of subclasses is to identify suitable models in a model
    space, that will be provided to a model selection method for selection.

    Attributes:
        distances:
            The distances of all candidate models from the initial model.
            FIXME change list to int? Is storage of more than one value
            useful?
        predecessor_model:
            The model used for comparison, e.g. for stepwise methods.
        models:
            The current set of candidate models.
        exclusions:
            A list of model hashes. Models that match a hash in `exclusions` will not
            be accepted into the candidate space. The hashes of models that are accepted
            are added to `exclusions`.
        limit:
            A handler to limit the number of accepted models.
        method:
            The model selection method of the candidate space.
        governing_method:
            Used to store the search method that governs the choice of method during
            a search. In some cases, this is always the same as the method attribute.
            An example of a difference is in the bidirectional method, where `governing_method`
            stores the bidirectional method, whereas `method` may also store the forward or
            backward methods.
        retry_model_space_search_if_no_models:
            Whether a search with a candidate space should be repeated upon failure.
            Useful for the `BidirectionalCandidateSpace`, which switches directions
            upon failure.
        summary_tsv:
            A string or `pathlib.Path`. A summary of the model selection progress
            will be written to this file.
        #limited:
        #    A descriptor that handles the limit on the number of accepted models.
        #limit:
        #    Models will fail `self.consider` if `len(self.models) >= limit`.
    """

    method: Method = None
    retry_model_space_search_if_no_models: bool = False

    def __init__(
        self,
        # TODO add MODEL_TYPE = Union[str, Model], str for VIRTUAL_INITIAL_MODEL
        predecessor_model: Optional[Model] = None,
        exclusions: Optional[List[Any]] = None,
        limit: TYPE_LIMIT = np.inf,
        summary_tsv: TYPE_PATH = None,
    ):
        self.limit = LimitHandler(
            current=self.n_accepted,
            limit=limit,
        )
        # Each candidate class specifies this as a class attribute.
        self.governing_method = self.method
        self.reset(predecessor_model=predecessor_model, exclusions=exclusions)

        self.summary_tsv = summary_tsv
        if self.summary_tsv is not None:
            self.summary_tsv = Path(self.summary_tsv)
            self._setup_summary_tsv()

    def write_summary_tsv(self, row):
        if self.summary_tsv is None:
            return

        # Format single values to be valid rows
        if not isinstance(row, list):
            row = [
                row,
                *([''] * 5),
            ]

        with open(self.summary_tsv, 'a', encoding="utf-8") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(row)

    def _setup_summary_tsv(self):
        self.summary_tsv.resolve().parent.mkdir(parents=True, exist_ok=True)

        if self.summary_tsv.exists():
            self.write_summary_tsv(
                'Continuing summary file with new candidate space.'
            )
        else:
            self.write_summary_tsv(
                [
                    'method',
                    '# candidates',
                    'predecessor change',
                    'current model criterion',
                    'current model',
                    'candidate changes',
                ]
            )

    @classmethod
    def read_arguments_from_yaml_dict(cls, yaml_dict):
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
        initial model `self.predecessor_model` that has no estimated
        parameters, then only models with one or more estimated parameters are
        plausible.

        Args:
            model:
                The candidate model.

        Returns:
            `True` if `model` is plausible, else `False`.
        """
        return True

    def distance(self, model: Model) -> Union[None, float, int]:
        """Compute the distance between two models that are neighbors.

        Args:
            model:
                The candidate model.
            predecessor_model:
                The initial model.

        Returns:
            The distance from `predecessor_model` to `model`, or `None` if the
            distance should not be computed.
        """
        return None

    def accept(
        self,
        model: Model,
        distance: Union[None, float, int],
        # keep_others: bool = True,
    ) -> None:
        """Add a candidate model to the candidate space.

        Args:
            model:
                The model that will be added.
            distance:
                The distance of the model from the predecessor model.
            #keep_others:
            #    Whether to keep other models that were previously added to the
            #    candidate space.
        """
        model.predecessor_model_hash = (
            self.predecessor_model.get_hash()
            if isinstance(self.predecessor_model, Model)
            else self.predecessor_model
        )
        self.models.append(model)
        self.distances.append(distance)
        self.exclude(model)

    def n_accepted(self) -> TYPE_LIMIT:
        """Get the current number of accepted models."""
        return len(self.models)

    def exclude(
        self,
        model: Union[Model, List[Model]],
    ) -> None:
        """Exclude model(s) from future consideration.

        Args:
            model:
                The model(s) that will be excluded.
        """
        if isinstance(model, list):
            for _model in model:
                self.exclusions.append(_model.get_hash())
        else:
            self.exclusions.append(model.get_hash())

    def excluded(
        self,
        model: Model,
    ) -> bool:
        """Whether a model is excluded."""
        return model.get_hash() in self.exclusions

    @abc.abstractmethod
    def _consider_method(self, model) -> bool:
        """Consider whether a model should be accepted, according to a method.

        Args:
            model:
                The candidate model.

        Returns:
            Whether a model should be accepted.
        """
        return True

    def consider(self, model: Union[Model, None]) -> bool:
        """Add a candidate model, if it should be added.

        Args:
            model:
                The candidate model. This value may be `None` if the `ModelSubspace`
                decided to exclude the model that would have been sent.

        Returns:
            Whether it is OK to send additional models to the candidate space. For
            example, if the limit of the number of accepted models has been reached,
            then no further models should be sent.
            TODO change to return whether the model was accepted, and instead add
                 `self.continue` to determine whether additional models should be sent.
        """
        # Model was excluded by the `ModelSubspace` that called this method, so can be
        # skipped.
        if model is None:
            # TODO use a different code than `True`?
            return True
        if self.limit.reached():
            return False
        if self.excluded(model):
            warnings.warn(
                f'Model has been previously excluded from the candidate space so is skipped here. Model subspace ID: {model.model_subspace_id}. Parameterization: {model.parameters}',
                RuntimeWarning,
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
        self.models = []
        self.distances = []

    def set_predecessor_model(
        self, predecessor_model: Union[Model, str, None]
    ):
        self.predecessor_model = predecessor_model
        if (
            self.predecessor_model == VIRTUAL_INITIAL_MODEL
            and self.method not in VIRTUAL_INITIAL_MODEL_METHODS
        ):
            raise ValueError(
                f'A virtual initial model was requested for a method ({self.method}) that does not support them.'
            )

    def get_predecessor_model(self):
        return self.predecessor_model

    def set_exclusions(self, exclusions: Union[List[str], None]):
        # TODO change to List[str] for hashes?
        self.exclusions = exclusions
        if self.exclusions is None:
            self.exclusions = []

    def get_exclusions(self):
        return self.exclusions

    def set_limit(self, limit: TYPE_LIMIT = None):
        if limit is not None:
            self.limit.set_limit(limit)

    def get_limit(self):
        return self.limit.get_limit()

    def wrap_search_subspaces(self, search_subspaces: Callable[[], None]):
        """Decorate the subspace searches of a model space.

        Used by candidate spaces to perform changes that alter the search.
        See `BidirectionalCandidateSpace` for an example, where it's used to switch directions.

        Args:
            search_subspaces:
                The method that searches the subspaces of a model space.
        """

        def wrapper():
            search_subspaces()

        return wrapper

    def reset(
        self,
        predecessor_model: Optional[Union[Model, str, None]] = None,
        # FIXME change `Any` to some `TYPE_MODEL_HASH` (e.g. union of str/int/float)
        exclusions: Optional[Union[List[str], None]] = None,
        limit: TYPE_LIMIT = None,
    ) -> None:
        """Reset the candidate models, optionally reinitialize with a model.

        Args:
            predecessor_model:
                The initial model.
            exclusions:
                Whether to reset model exclusions.
            limit:
                The new upper limit of the number of models in this candidate space.
        """
        self.set_predecessor_model(predecessor_model)
        self.reset_accepted()
        self.set_exclusions(exclusions)
        self.set_limit(limit)

    def distances_in_estimated_parameters(
        self,
        model: Model,
        predecessor_model: Optional[Model] = None,
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

        Returns:
            The distances between the models, as a dictionary, where a key is the
            name of the metric, and the value is the corresponding distance.
        """
        model0 = predecessor_model
        if model0 is None:
            model0 = self.predecessor_model
        model1 = model

        if model0 != VIRTUAL_INITIAL_MODEL and not model1.petab_yaml.samefile(
            model0.petab_yaml
        ):
            raise NotImplementedError(
                'Computation of distances between different PEtab problems is '
                'currently not supported. This error is also raised if the same '
                'PEtab problem is read from YAML files in different locations.'
            )

        # All parameters from the PEtab problem are used in the computation.
        if model0 == VIRTUAL_INITIAL_MODEL:
            parameter_ids = list(model1.petab_parameters)
            # FIXME need to take superset of all parameters amongst all PEtab problems
            # in all model subspaces to get an accurate comparable distance. Currently
            # only reasonable when working with a single PEtab problem for all models
            # in all subspaces.
            if self.method == Method.FORWARD:
                parameters0 = np.array([0 for _ in parameter_ids])
            elif self.method == Method.BACKWARD:
                parameters0 = np.array([ESTIMATE for _ in parameter_ids])
            else:
                raise NotImplementedError(
                    'Distances for the virtual initial model have not yet been '
                    f'implemented for the method "{self.method}". Please notify the'
                    'developers.'
                )
        else:
            parameter_ids = list(model0.petab_parameters)
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

        # TODO constants?
        distances = {
            'l1': l1,
            'size': size,
        }

        return distances

    def update_after_calibration(self, history, local_history, criterion):
        """Inner candidate space update, only for FAMoSCandidateSpace.

        In each call of the Method_caller check wheter the FAMoS method
        should change methods by comparing local_history of the last
        search to the best max_nmb best models of the whole search. If
        no new best model is found we change methods using the
        provided or default method scheme.

        If we change methods, to construct the new candidate space we
        need the whole history of visited models to include them into
        excluded models so the new method does not visit them anymore.

        Non-dependent on method-changing, we update the method_history
        with the current method that's being used.

        Args:
            history:
                Whole history of models of the model selection run
            local_history:
                new models from last step with their scores (AIC/BIX)

        Return:
            jumped_to_most_distant:
                Boolean value of whether we have jumped to most distant
                in this iteration.
        """

        return False


class ForwardCandidateSpace(CandidateSpace):
    """The forward method class.

    Attributes:
        direction:
            `1` for the forward method, `-1` for the backward method.
        max_steps:
            Maximum number of steps forward in a single iteration of forward selection.
            Defaults to no maximum (`None`).
    """

    method = Method.FORWARD
    direction = 1

    def __init__(
        self,
        *args,
        predecessor_model: Optional[Union[Model, str]] = None,
        max_steps: int = None,
        **kwargs,
    ):
        # Although `VIRTUAL_INITIAL_MODEL` is `str` and can be used as a default
        # argument, `None` may be passed by other packages, so the default value
        # is handled here instead.
        self.max_number_of_steps = max_number_of_steps
        if predecessor_model is None:
            predecessor_model = VIRTUAL_INITIAL_MODEL
        super().__init__(*args, predecessor_model=predecessor_model, **kwargs)

    def is_plausible(self, model: Model) -> bool:
        distances = self.distances_in_estimated_parameters(model)
        unsigned_size = self.direction * distances['size']

        # If max_number_of_steps is non-zero and the number of steps made is
        # larger then move is not plausible.
        if (
            self.max_number_of_steps
            and unsigned_size > self.max_number_of_steps
        ):
            raise StopIteration(
                f"Maximal number of steps for method {self.method} exceeded. Stop sending candidate models."
            )

        # A model is plausible if the number of estimated parameters strictly
        # increases (or decreases, if `self.direction == -1`), and no
        # previously estimated parameters become fixed.
        if self.predecessor_model == VIRTUAL_INITIAL_MODEL or (
            unsigned_size > 0 and distances['l1'] == unsigned_size
        ):
            return True
        return False

    def distance(self, model: Model) -> int:
        # TODO calculated here and `is_plausible`. Rewrite to only calculate
        #      once?
        distances = self.distances_in_estimated_parameters(model)
        return distances['l1']

    def _consider_method(self, model) -> bool:
        """See `CandidateSpace._consider_method`."""
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

    method = Method.BACKWARD
    direction = -1


class BidirectionalCandidateSpace(ForwardCandidateSpace):
    """The bidirectional method class.

    Attributes:
        method_history:
            The history of models that were found at each search.
            A list of dictionaries, where each dictionary contains keys for the `METHOD`
            and the list of `MODELS`.
    """

    method = Method.BIDIRECTIONAL
    retry_model_space_search_if_no_models = True

    def __init__(
        self,
        *args,
        initial_method: Method = Method.FORWARD,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # FIXME cannot access from CLI
        self.initial_method = initial_method

        self.history: List[Dict[str, Union[Method, List[Model]]]] = []

    def update_method(self, method: Method):
        if method == Method.FORWARD:
            self.direction = 1
        elif method == Method.BACKWARD:
            self.direction = -1
        else:
            raise NotImplementedError(
                f'Bidirectional direction must be either `Method.FORWARD` or `Method.BACKWARD`, not {method}.'
            )

        self.method = method

    def switch_method(self):
        if self.method == Method.FORWARD:
            method = Method.BACKWARD
        elif self.method == Method.BACKWARD:
            method = Method.FORWARD

        self.update_method(method=method)

    def setup_before_model_subspaces_search(self):
        # If previous search found no models, then switch method.
        previous_search = None if not self.history else self.history[-1]
        if previous_search is None:
            self.update_method(self.initial_method)
            return

        self.update_method(previous_search[METHOD])
        if not previous_search[MODELS]:
            self.switch_method()
            self.retry_model_space_search_if_no_models = False

    def setup_after_model_subspaces_search(self):
        current_search = {
            METHOD: self.method,
            MODELS: self.models,
        }
        self.history.append(current_search)
        self.method = self.governing_method

    def wrap_search_subspaces(self, search_subspaces):
        def wrapper():
            def iterate():
                self.setup_before_model_subspaces_search()
                search_subspaces()
                self.setup_after_model_subspaces_search()

            # Repeat until models are found or switching doesn't help.
            iterate()
            while (
                not self.models and self.retry_model_space_search_if_no_models
            ):
                iterate()

            # Reset flag for next time.
            self.retry_model_space_search_if_no_models = True

        return wrapper


# TODO:
# - specifiy initial model (Implement)
# - make it work, without swap, compare results (similar)
# later : General constraints (critical, swap, something else...)


class FamosCandidateSpace(CandidateSpace):
    """The FAMoS method class.

    Attributes:
        critical_parameter_sets:
            A list of lists which represent the critical parameter sets.
            The initial predecessor model as well as every next model accepted
            have to contain at least 1 parameter from each critical parameter set
            non-fixed ('estimate').
        swap_parameter_sets:
            A list of lists which represent the swap parameter sets.
            The Lateral method for the FAMoS algorithm can make a swap move, where
            one non-fixed parameter is fixed and another fixed is un-fixed, only if
            both parameters are contained in the same swap parameter set.
        method_scheme:
            A dictionary of the method scheme used to switch to a different
            method when the current does not provide a better model. The keys of the
            dictionary are tuples of the previous methods of arbitrary size and values
            are the methods to which the method should swap if the current previous
            methods coincide with the key. FAMoS will iterate through the dictionary to
            find the first key that coincides with the current previous methods. The
            method of this key will be chosen, and other keys will be ignored.
        number_of_reattempts:
            Integer. If grater or equal 1 then at the point at which we would usually
            terminate, FAMoS will find a most_distant model to jump to and start the
            model selection again, excluding the already considered models. The integer
            value determines how many times it will reattempt. If 0 then will not reattempt.
        swap_only_once:
            Boolean, if True then the LATERAL method will switch to FORWARD method after
            one successful lateral move. Otherwise, the LATERAL method will continue
            searching for better models until no such models can be found.
    """

    method = Method.FAMOS
    default_method_scheme = {
        (Method.BACKWARD, Method.FORWARD): Method.LATERAL,
        (Method.FORWARD, Method.BACKWARD): Method.LATERAL,
        (Method.BACKWARD, Method.LATERAL): None,
        (Method.FORWARD, Method.LATERAL): None,
        (Method.FORWARD,): Method.BACKWARD,
        (Method.BACKWARD,): Method.FORWARD,
        (Method.LATERAL,): Method.FORWARD,
        None: Method.FORWARD,
    }

    def __init__(
        self,
        *args,
        predecessor_model: Optional[Union[Model, str, None]] = None,
        critical_parameter_sets: List = [],
        swap_parameter_sets: List = [],
        method_scheme: Dict[tuple, str] = None,
        number_of_reattempts: int = 0,
        swap_only_once: bool = True,
        **kwargs,
    ):
        self.critical_parameter_sets = critical_parameter_sets
        self.swap_parameter_sets = swap_parameter_sets

        self.method_scheme = method_scheme
        if method_scheme is None:
            self.method_scheme = self.default_method_scheme

        self.initial_method = self.method_scheme[None]
        self.method = self.initial_method
        self.method_history = [self.initial_method]

        if predecessor_model is None:
            predecessor_model = VIRTUAL_INITIAL_MODEL

        if (
            predecessor_model == VIRTUAL_INITIAL_MODEL
            and critical_parameter_sets
        ) or (
            # FIXME should virtual initial model raise error if critical sets
            #       are specified? i.e. should users be expected to supply a valid initial model
            #       if critical sets are specified? ideally the first iteration with the virtual
            #       initial model would find compatible models that satisfy the critical sets
            predecessor_model != VIRTUAL_INITIAL_MODEL
            and not self.check_critical(predecessor_model)
        ):
            raise ValueError(
                f'Provided predecessor model {predecessor_model.parameters} does not contain necessary critical parameters {self.critical_parameter_sets}. Provide a valid predecessor model.'
            )

        if (
            predecessor_model == VIRTUAL_INITIAL_MODEL
            and self.initial_method not in VIRTUAL_INITIAL_MODEL_METHODS
        ):
            raise ValueError(
                f"The initial method {self.initial_method} does not support the `VIRTUAL_INITIAL_MODEL` as its predecessor_model."
            )

        # FIXME remove `None` from the resulting `inner_methods` set?
        inner_methods = set.union(
            *[
                set(
                    [
                        *(
                            method_pattern
                            if method_pattern is not None
                            else (None,)
                        ),
                        next_method,
                    ]
                )
                for method_pattern, next_method in self.method_scheme.items()
            ]
        )
        if Method.LATERAL in inner_methods and not self.swap_parameter_sets:
            raise ValueError(
                f"Use of the lateral method with FAMoS requires `swap_parameter_sets`."
            )

        self.inner_candidate_spaces = {
            Method.FORWARD: ForwardCandidateSpace(
                *args,
                predecessor_model=predecessor_model,
                max_number_of_steps=0,
                **kwargs,
            ),
            Method.BACKWARD: BackwardCandidateSpace(
                *args,
                predecessor_model=predecessor_model,
                max_number_of_steps=0,
                **kwargs,
            ),
            Method.LATERAL: LateralCandidateSpace(
                *args,
                predecessor_model=(
                    predecessor_model
                    if predecessor_model != VIRTUAL_INITIAL_MODEL
                    else None
                ),
                max_number_of_steps=1,
                **kwargs,
            ),
        }
        self.inner_candidate_space = self.inner_candidate_spaces[
            self.initial_method
        ]

        super().__init__(*args, predecessor_model=predecessor_model, **kwargs)

        self.history: List[Dict[str, Union[Method, List[Model]]]] = []

        self.number_of_reattempts = number_of_reattempts

        self.swap_only_once = swap_only_once
        if self.swap_only_once and (Method.LATERAL,) not in self.method_scheme:
            raise ValueError(
                "Please provide a method to switch to after a lateral search, if enabling the `swap_only_once` option."
            )

        if self.number_of_reattempts:
            # TODO make so max_number can be specified? It cannot in original FAMoS.
            self.most_distant_max_number = 100
        else:
            self.most_distant_max_number = 1

        self.best_models = []
        self.best_model_of_current_run = predecessor_model

        self.found_new_best = True

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
        history: Dict[str, Model],
        local_history: Dict[str, Model],
        criterion: Criterion,
    ) -> None:
        """See `CandidateSpace.update_after_calibration`."""
        # In case we jumped to most distant in the last iteration,
        # here we reset the jumped variable to False
        self.jumped_to_most_distant = False

        self.history = history

        if not self.update_from_local_history(
            local_history=local_history, criterion=criterion
        ):
            logging.info("Switching method")
            self.switch_method()
            self.switch_inner_candidate_space(history)
            logging.info(
                "Method switched to ", self.inner_candidate_space.method
            )

        self.method_history.append(self.method)
        return self.jumped_to_most_distant

    def update_from_local_history(
        self, local_history: Dict[str, Model], criterion: Criterion
    ) -> bool:
        """Update the self.best_models with the latest local_history
        and determine if there was a new best model. If so, return
        True. False otherwise."""

        found_new_best = False
        for model_id in local_history:
            if (
                self.best_model_of_current_run == VIRTUAL_INITIAL_MODEL
                or default_compare(
                    self.best_model_of_current_run,
                    local_history[model_id],
                    criterion,
                )
            ):
                found_new_best = True
                self.best_model_of_current_run = local_history[model_id]

            if len(self.best_models) < self.most_distant_max_number:
                self.insert_model_into_best_models(
                    model_to_insert=local_history[model_id],
                    criterion=criterion,
                )
            elif default_compare(
                self.best_models[self.most_distant_max_number - 1],
                local_history[model_id],
                criterion,
            ):
                self.insert_model_into_best_models(
                    model_to_insert=local_history[model_id],
                    criterion=criterion,
                )

        self.best_models = self.best_models[: self.most_distant_max_number]

        # When we switch to LATERAL method, we will do only one iteration with this
        # method. So if we do it succesfully (i.e. that we found_new_best), we still
        # want to switch method. This is why we put found_new_best to False, so we go
        # into the method switching pipeline
        if (
            found_new_best
            and self.method == Method.LATERAL
            and self.swap_only_once
        ):
            self.swap_done_successfully = True
            found_new_best = False
        return found_new_best

    def insert_model_into_best_models(
        self, model_to_insert: Model, criterion: Criterion
    ) -> None:
        """Inserts a model into the list of best_models which are sorted
        w.r.t. the criterion specified."""
        insert_index = bisect.bisect_left(
            [model.get_criterion(criterion) for model in self.best_models],
            model_to_insert.get_criterion(criterion),
        )
        self.best_models.insert(insert_index, model_to_insert)

    def consider(self, model: Union[Model, None]) -> bool:
        """Re-define consider of FAMoS to be the consider method
        of the inner_candidate_space. Update all the attributes
        changed in the condsider method."""

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
            self.exclusions = self.inner_candidate_space.exclusions

            return return_value

        return True

    def _consider_method(self, model) -> bool:
        """See `CandidateSpace._consider_method`."""
        return self.inner_candidate_space._consider_method(model)

    def reset_accepted(self) -> None:
        """Changing the reset_accepted to reset the
        inner_candidate_space as well."""
        self.models = []
        self.distances = []
        self.inner_candidate_space.reset_accepted()

    def set_predecessor_model(
        self, predecessor_model: Union[Model, str, None]
    ):
        """Setting the predecessor model for the
        inner_candidate_space as well."""
        super().set_predecessor_model(predecessor_model=predecessor_model)
        self.inner_candidate_space.set_predecessor_model(
            predecessor_model=predecessor_model
        )

    def set_exclusions(self, exclusions: Union[List[str], None]):
        """Setting the exclusions for the
        inner_candidate_space as well."""
        self.exclusions = exclusions
        self.inner_candidate_space.exclusions = exclusions
        if self.exclusions is None:
            self.exclusions = []
            self.inner_candidate_space.exclusions = []

    def set_limit(self, limit: TYPE_LIMIT = None):
        """Setting the limit for the
        inner_candidate_space as well."""
        if limit is not None:
            self.limit.set_limit(limit)
            self.inner_candidate_space.limit.set_limit(limit)

    def is_plausible(self, model: Model) -> bool:
        if self.check_critical(model):
            return self.inner_candidate_space.is_plausible(model)
        return False

    def check_swap(self, model: Model) -> bool:
        """Check if parameters that are swapped are contained in the
        same swap parameter set."""
        if self.method != Method.LATERAL:
            return True

        predecessor_estimated_parameters_ids = set(
            self.predecessor_model.get_estimated_parameter_ids_all()
        )
        estimated_parameters_ids = set(model.get_estimated_parameter_ids_all())

        swapped_parameters_ids = estimated_parameters_ids.symmetric_difference(
            predecessor_estimated_parameters_ids
        )

        for swap_set in self.swap_parameter_sets:
            if swapped_parameters_ids.issubset(set(swap_set)):
                return True
        return False

    def check_critical(self, model: Model) -> bool:
        """Check if the model contains all necessary critical parameters"""

        estimated_parameters_ids = set(model.get_estimated_parameter_ids_all())
        for critical_set in self.critical_parameter_sets:
            if not estimated_parameters_ids.intersection(set(critical_set)):
                return False
        return True

    def switch_method(self) -> None:
        """Switch to the next method with respect to the history
        of methods used and the switching scheme in self.method_scheme"""

        previous_method = self.method
        next_method = previous_method
        logging.info("SWITCHING", self.method_history)
        # breakpoint()
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

        # if the next method is None (in default case if SWAP
        # method didn't find any better models) then terminate
        if not next_method:
            if self.number_of_reattempts:
                self.jump_to_most_distant()
                return
            else:
                raise StopIteration(
                    f"The next method provided is None. The search is terminating."
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
            if self.number_of_reattempts:
                self.jump_to_most_distant()
                return
            else:
                raise StopIteration(
                    "The next chosen method is Method.LATERAL, but there are no crit or swap parameters provided. Terminating"
                )
        if previous_method == Method.LATERAL:
            self.swap_done_successfully = False
        self.update_method(method=next_method)

    def update_method(self, method: Method):
        """Update self.method to the method."""
        if method not in [Method.FORWARD, Method.LATERAL, Method.BACKWARD]:
            raise NotImplementedError(
                f'FAMoS direction must be either `Method.FORWARD`, `Method.BACKWARD` or `Method.LATERAL`, not {method}. \
                Check if the method_scheme scheme provided is correct.'
            )

        self.method = method

    def switch_inner_candidate_space(self, history):
        """Switch self.inner_candidate_space to the candidate space of
        the current self.method."""

        self.inner_candidate_space = self.inner_candidate_spaces[self.method]
        # reset the next inner candidate space with the current history of all checked models
        self.inner_candidate_space.reset(
            predecessor_model=self.predecessor_model, exclusions=history.keys
        )
        # FIXME Check that problem.calibrated_models is excluded.

    def jump_to_most_distant(self):
        """Jump to most distant model with respect to the history of all
        calibrated models."""

        new_init_model = self.get_most_distant()

        logging.info("JUMPING: ", new_init_model.parameters)
        # breakpoint()

        # if model not appropriate make it so by adding the first
        # critical parameter from each critical parameter set
        if not self.check_critical(new_init_model):
            for critical_set in self.critical_parameter_sets:
                new_init_model.parameters[critical_set[0]] = ESTIMATE

        self.update_method(self.initial_method)

        self.number_of_reattempts -= 1
        self.jumped_to_most_distant = True

        self.predecessor_model = new_init_model
        self.best_model_of_current_run = new_init_model
        self.write_summary_tsv("Jumped to the most distant model.")

    # TODO Fix for non-famos model subspaces. FAMOS easy beacuse of only 0;ESTIMATE
    def get_most_distant(self) -> Model:
        """
        Get most distant model to all the checked models. We take models from the
        sorted list of best models (self.best_models) and construct complements of
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

        for model in self.best_models:
            model_parameter_ids = list(model.petab_parameters)
            model_parameters = model.get_parameter_values(
                parameter_ids=model_parameter_ids
            )

            model_estimated_parameters = np.array(
                [p == ESTIMATE for p in model_parameters]
            ).astype(int)
            complement_parameters = abs(model_estimated_parameters - 1)
            # initialize the least distance to the maximal possible value of it
            complement_least_distance = len(complement_parameters)
            # get the complement least distance
            for history_model_id in self.history:
                history_model = self.history[history_model_id]
                history_model_parameter_ids = list(
                    history_model.petab_parameters
                )
                history_model_parameters = history_model.get_parameter_values(
                    parameter_ids=history_model_parameter_ids
                )

                history_model_estimated_parameters = np.array(
                    [p == ESTIMATE for p in history_model_parameters]
                ).astype(int)

                difference = (
                    history_model_estimated_parameters - complement_parameters
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
            str(index).replace('1', ESTIMATE) for index in most_distant_indices
        ]

        most_distant_parameters = {
            parameter_id: index
            for parameter_id, index in zip(
                model.parameters, most_distant_parameter_values
            )
        }

        most_distant_model = Model(
            petab_yaml=model.petab_yaml,
            model_subspace_id=model.model_subspace_id,
            model_subspace_indices=most_distant_indices,
            parameters=most_distant_parameters,
        )

        return most_distant_model

    def wrap_search_subspaces(self, search_subspaces):
        def wrapper():
            search_subspaces(only_one_subspace=True)

        return wrapper


# TODO rewrite so BidirectionalCandidateSpace inherits from ForwardAndBackwardCandidateSpace
#      instead
class ForwardAndBackwardCandidateSpace(BidirectionalCandidateSpace):
    method = Method.FORWARD_AND_BACKWARD
    governing_method = Method.FORWARD_AND_BACKWARD
    retry_model_space_search_if_no_models = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, initial_method=None)

    def wrap_search_subspaces(self, search_subspaces):
        def wrapper():
            for method in [Method.FORWARD, Method.BACKWARD]:
                self.update_method(method=method)
                search_subspaces()
                self.setup_after_model_subspaces_search()

        return wrapper

    # Disable unused interface
    setup_before_model_subspaces_search = None
    switch_method = None

    def setup_after_model_space_search(self):
        pass


class LateralCandidateSpace(CandidateSpace):
    """Find models with the same number of estimated parameters."""

    method = Method.LATERAL

    def __init__(
        self,
        *args,
        predecessor_model: Union[Model, None],
        max_number_of_steps: int = 0,
        **kwargs,
    ):
        """
        Additional args:
            max_number_of_steps:
                Maximal allowed number of swap moves. If 0 then there is no maximum.
        """
        super().__init__(
            *args,
            predecessor_model=predecessor_model,
            **kwargs,
        )
        self.max_number_of_steps = max_number_of_steps

    def is_plausible(self, model: Model) -> bool:
        if self.predecessor_model is None:
            raise ValueError(
                f"The predecessor_model is still None. Provide an appropriate predecessor_model"
            )

        distances = self.distances_in_estimated_parameters(model)

        # If max_number_of_steps is non-zero and the number of steps made is
        # larger then move is not plausible.
        if (
            self.max_number_of_steps
            and distances['l1'] > 2 * self.max_number_of_steps
        ):
            raise StopIteration(
                f"Maximal number of steps for method {self.method} exceeded. Stop sending candidate models."
            )

        # A model is plausible if the number of estimated parameters remains
        # the same, but some estimated parameters have become fixed and vice
        # versa.
        if (
            distances['size'] == 0
            and
            # distances['size'] == 0 implies L1 % 2 == 0.
            # FIXME here and elsewhere, deal with models that are equal
            #       except for the values of their fixed parameters.
            distances['l1'] > 0
        ):
            return True
        return False

    # TODO does Lateral need this?
    def distance(self, model: Model) -> int:
        # TODO calculated here and `is_plausible`. Rewrite to only calculate
        #      once?
        distances = self.distances_in_estimated_parameters(model)
        return distances['l1']

    def _consider_method(self, model):
        return True


class BruteForceCandidateSpace(CandidateSpace):
    """The brute-force method class."""

    method = Method.BRUTE_FORCE

    def __init__(self, *args, **kwargs):
        # if args or kwargs:
        #    # FIXME remove?
        #    # FIXME at least support limit
        #    warnings.warn(
        #        'Arguments were provided but will be ignored, because of the '
        #        'brute force candidate space.'
        #    )
        super().__init__(*args, **kwargs)

    def _consider_method(self, model):
        return True


candidate_space_classes = [
    ForwardCandidateSpace,
    BackwardCandidateSpace,
    BidirectionalCandidateSpace,
    LateralCandidateSpace,
    BruteForceCandidateSpace,
    ForwardAndBackwardCandidateSpace,
    FamosCandidateSpace,
]


def method_to_candidate_space_class(method: Method) -> str:
    """Instantiate a candidate space given its method name.

    Args:
        method:
            The name of the method corresponding to one of the implemented candidate
            spaces.

    Returns:
        The candidate space.
    """
    for candidate_space_class in candidate_space_classes:
        if candidate_space_class.method == method:
            return candidate_space_class
    raise NotImplementedError(
        f'The provided method name {method} does not correspond to an implemented candidate space.'
    )


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
