import math
import warnings
from collections.abc import Iterable, Iterator
from itertools import product
from os.path import relpath
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import petab.v1 as petab
from more_itertools import powerset

from .candidate_space import CandidateSpace
from .constants import (
    ESTIMATE,
    MODEL_SUBSPACE_ID,
    MODEL_SUBSPACE_PETAB_YAML,
    PARAMETER_VALUE_DELIMITER,
    STEPWISE_METHODS,
    TYPE_PARAMETER_DICT,
    TYPE_PARAMETER_OPTIONS,
    TYPE_PARAMETER_OPTIONS_DICT,
    TYPE_PATH,
    Method,
)
from .misc import parameter_string_to_value
from .model import VIRTUAL_INITIAL_MODEL, Model
from .petab import get_petab_parameters

__all__ = [
    "ModelSubspace",
]


class ModelSubspace:
    """Efficient representation of exponentially large model subspaces.

    Attributes:
        model_subspace_id:
            The ID of the model subspace.
        petab_yaml:
            The location of the PEtab problem YAML file.
        parameters:
            The key is the ID of the parameter. The value is a list of values
            that the parameter can take (including ``ESTIMATE``).
        exclusions:
            Hashes of models that have been previously submitted to a candidate space
            for consideration (:meth:`CandidateSpace.consider`).
    """

    def __init__(
        self,
        model_subspace_id: str,
        petab_yaml: str | Path,
        parameters: TYPE_PARAMETER_OPTIONS_DICT,
        exclusions: list[Any] | None | None = None,
    ):
        self.model_subspace_id = model_subspace_id
        self.petab_yaml = Path(petab_yaml)
        self.parameters = parameters

        self.exclusions = set()
        if exclusions is not None:
            self.exclusions = set(exclusions)

        self.petab_problem = petab.Problem.from_yaml(self.petab_yaml)

        for parameter_id, parameter_value in self.parameters.items():
            if not parameter_value:
                raise ValueError(
                    f"The parameter `{parameter_id}` is in the definition "
                    "of this model subspace. However, its value is empty. "
                    f"Please specify either its fixed value or `'{ESTIMATE}'` "
                    "(e.g. in the model space table)."
                )

    def check_compatibility_stepwise_method(
        self,
        candidate_space: CandidateSpace,
    ) -> bool:
        """Check whether a candidate space is compatible with this subspace.

        Directional methods (e.g. forward, backward) are not supported
        among different PEtab problems.

        Args:
            candidate_space:
                The candidate space with a stepwise method.

        Returns:
            Whether the candidate space is compatible.
        """
        if candidate_space.method not in STEPWISE_METHODS:
            return True
        if (
            candidate_space.predecessor_model.hash
            != VIRTUAL_INITIAL_MODEL.hash
            and (
                str(
                    candidate_space.predecessor_model.model_subspace_petab_yaml.resolve()
                )
                != str(self.petab_yaml.resolve())
            )
        ):
            warnings.warn(
                "The supplied candidate space is initialized with a model "
                "that has a different PEtab YAML to this model subspace. "
                "This is currently not supported for stepwise methods "
                "(e.g. forward or backward). "
                f"This model subspace: `{self.model_subspace_id}`. "
                f"This model subspace PEtab YAML: `{self.petab_yaml}`. "
                "The candidate space PEtab YAML: "
                f"`{candidate_space.predecessor_model.model_subspace_petab_yaml}`.",
                stacklevel=2,
            )
            return False
        return True

    def get_models(self, estimated_parameters: list[str]) -> Iterator[Model]:
        """Get models in the subspace by estimated parameters.

        All models that have the provided ``estimated_parameters`` are returned.

        Args:
            estimated_parameters:
                The IDs of parameters that are estimated in the model. All
                other parameters will be fixed. Note that these parameters are
                in the subset of PEtab parameters that exist in the model
                subspace definition. Parameters in the PEtab problem but not
                the model subspace definition should not be included here.

                FIXME(dilpath)
                TODO support the full set of PEtab parameters? Then would need
                to turn off estimation for parameters that are not
                provided in `estimated_parameters` -- maybe unexpected for
                users.

        Returns:
            A list of models.
        """
        if set(estimated_parameters).difference(self.parameters):
            raise ValueError(
                "Some parameter IDs were provided that are not in the model "
                "subspace definition. NB: parameters that are only in the "
                "PEtab parameters table should not be included here. "
                f"IDs: {set(estimated_parameters).difference(self.parameters)}"
            )
        fixed_parameter_ids = [
            parameter_id
            for parameter_id in self.parameters
            if parameter_id not in estimated_parameters
        ]
        parameters_cannot_be_fixed_error = [
            parameter_id
            for parameter_id in fixed_parameter_ids
            if parameter_id not in self.can_fix
        ]
        if parameters_cannot_be_fixed_error:
            raise ValueError(
                "Models with the following fixed parameters were requested; "
                "however, there is no such model in this subspace: "
                f"{parameters_cannot_be_fixed_error}."
            )
        # Identify possible values for each of the fixed parameters.
        fixed_options = [
            [
                parameter_value
                for parameter_value in self.parameters[parameter_id]
                if parameter_value != ESTIMATE
            ]
            for parameter_id in fixed_parameter_ids
        ]
        # Generate models
        for fixed_parameter_values in product(*fixed_options):
            fixed_parameters = dict(
                zip(
                    fixed_parameter_ids,
                    fixed_parameter_values,
                    strict=False,
                )
            )
            parameters = {
                **fixed_parameters,
                **{id: ESTIMATE for id in estimated_parameters},
            }
            model = self.parameters_to_model(parameters)
            # Skip models that are excluded.
            if model is None:
                continue
            yield model

    def search(
        self,
        candidate_space: CandidateSpace,
        limit: int = np.inf,
    ):
        """Search for candidate models in this model subspace.

        Nothing is returned, as the result is managed by the
        ``candidate_space``.

        Args:
            candidate_space:
                The candidate space.
            limit:
                Limit the number of models.
        """

        def continue_searching(
            continue_sending: bool,
            # FIXME refactor to use LimitHandler
            limit: int = limit,
        ) -> bool:
            """Increment the model counter, and check whether to continue searching.

            Args:
                continue_sending:
                    Whether to continue sending models to the candidate space for
                    consideration.
                limit:
                    The maximum number of models to send to the candidate space.

            Returns:
                Whether to continue considering models.
            """
            try:
                continue_searching.counter += 1
            except AttributeError:
                continue_searching.counter = 1
            if continue_searching.counter >= limit:
                return False
            if not continue_sending:
                return False
            return True

        if not self.check_compatibility_stepwise_method(candidate_space):
            return

        # TODO check inside `continue_searching` too? or move
        # all limit handling to candidate space
        if candidate_space.limit.reached():
            return

        # Compute parameter sets that are useful for finding minimal forward or backward
        # moves in the subspace.
        # Parameters that are currently estimated in the predecessor model.
        if (
            candidate_space.predecessor_model.hash
            == VIRTUAL_INITIAL_MODEL.hash
        ):
            if candidate_space.method == Method.FORWARD:
                old_estimated_all = self.must_estimate_all
                old_fixed_all = self.can_fix_all
            elif candidate_space.method == Method.BACKWARD:
                old_estimated_all = self.can_estimate_all
                old_fixed_all = self.must_fix_all
            elif candidate_space.method == Method.BRUTE_FORCE:
                # doesn't matter what these are set to
                old_estimated_all = self.must_estimate_all
                old_fixed_all = self.must_fix_all
            else:
                # Should already be handled elsewhere (e.g.
                # `self.check_compatibility_stepwise_method`).
                raise NotImplementedError(
                    "The virtual initial model and method "
                    f"{candidate_space.method} is not implemented. "
                    "Please report at https://github.com/PEtab-dev/petab_select/issues if this is desired."
                )
        else:
            old_estimated_all = (
                candidate_space.predecessor_model.get_estimated_parameter_ids()
            )
            old_fixed_all = [
                parameter_id
                for parameter_id in self.parameters_all
                if parameter_id not in old_estimated_all
            ]

        # Parameters that are fixed in the candidate space
        # predecessor model but are necessarily estimated in this subspace.
        new_must_estimate_all = set(self.must_estimate_all).difference(
            old_estimated_all
        )
        new_can_fix_all = set(old_estimated_all).difference(
            self.must_estimate_all
        )
        new_must_fix_all = set(old_estimated_all).difference(
            self.can_estimate_all
        )
        new_can_estimate_all = set(self.can_estimate_all).difference(
            old_estimated_all
        )

        # Parameters related to minimal changes compared to the predecessor model.
        old_estimated = set(old_estimated_all).intersection(self.can_estimate)
        old_fixed = set(old_fixed_all).intersection(self.can_fix)
        new_must_estimate = set(new_must_estimate_all).intersection(
            self.parameters
        )
        # TODO remove this block...
        if not (
            set(self.must_estimate).difference(old_estimated)
            == new_must_estimate
        ):
            raise ValueError(
                "Unexpected error (sets that should be equal are not)."
            )
        new_can_estimate_optional = (
            set(self.can_estimate)
            .difference(self.must_estimate)
            .difference(old_estimated)
        )
        new_can_fix_optional = (
            set(new_can_fix_all)
            .intersection(self.can_fix)
            .difference(self.must_fix)
        )

        if candidate_space.method == Method.FORWARD:
            # There are no parameters that could become estimated in this subspace, so
            # there are no valid "forward" moves.
            if (
                not new_can_estimate_all
                and candidate_space.predecessor_model.hash
                != VIRTUAL_INITIAL_MODEL.hash
            ):
                return
            # There are estimated parameters in the predecessor model that
            # must be fixed in this subspace, so there are no valid "forward" moves.
            if new_must_fix_all:
                return
            # Smallest possible "forward" moves involve necessarily estimated
            # parameters.
            if (
                new_must_estimate_all
                or candidate_space.predecessor_model.hash
                == VIRTUAL_INITIAL_MODEL.hash
            ):
                # Consider minimal models that have all necessarily-estimated
                # parameters.
                estimated_parameters = {
                    parameter_id: ESTIMATE
                    for parameter_id in (
                        set(self.parameters)
                        .difference(self.can_fix)
                        .union(old_estimated)
                    )
                }
                models = self.get_models(
                    estimated_parameters=estimated_parameters,
                )
                previous_number_of_candidates = len(candidate_space.models)
                for model in models:
                    continue_sending = self.send_model_to_candidate_space(
                        model=model,
                        candidate_space=candidate_space,
                    )
                    if not continue_searching(continue_sending):
                        return

                # No need to consider other models, as they will necessarily
                # be worse than the current set of models in the candidate
                # space, if suitable candidates models have already been
                # identified.
                if len(candidate_space.models) > previous_number_of_candidates:
                    return

            # Keep track of the number of additional parameters estimated.
            # Stop considering models once all parameter sets with the minimal number of
            # extra estimated parameters are considered.
            n_estimated_extra = np.inf
            previous_number_of_candidates = len(candidate_space.models)
            # The powerset should be in ascending order by number of elements.
            for parameter_set in powerset(new_can_estimate_optional):
                try:
                    # The case of a "minimal" model in the subspace being a valid candidate
                    # in this case should have been handled above already with
                    # `new_must_estimate_all`
                    if not parameter_set:
                        continue
                    # If a model has been accepted by the candidate space, only
                    # consider models of the same size (same minimal increase in the number
                    # of extra estimated parameters), then stop.
                    if len(parameter_set) > n_estimated_extra:
                        break
                    estimated_parameters = (
                        set(old_estimated)
                        .union(new_must_estimate)
                        .union(parameter_set)
                    )
                    models = self.get_models(
                        estimated_parameters=list(estimated_parameters),
                    )
                    for model in models:
                        continue_sending = self.send_model_to_candidate_space(
                            model=model,
                            candidate_space=candidate_space,
                        )
                        if not continue_searching(continue_sending):
                            return
                    # If model accepted set the maximal number of extra parameters to
                    # current number of extra parameters
                    if (
                        len(candidate_space.models)
                        > previous_number_of_candidates
                    ):
                        n_estimated_extra = len(parameter_set)
                except StopIteration:
                    break

        elif candidate_space.method == Method.BACKWARD:
            # There are no parameters that could become fixed in this subspace, so there
            # are no valid "backward" moves.
            if (
                not new_can_fix_all
                and candidate_space.predecessor_model.hash
                != VIRTUAL_INITIAL_MODEL.hash
            ):
                return
            # There are fixed parameters in the predecessor model that must be estimated
            # in this subspace, so there are no valid "backward" moves.
            if new_must_estimate_all:
                return
            # Smallest possible "backward" moves involve necessarily fixed
            # parameters.
            if (
                new_must_fix_all
                or candidate_space.predecessor_model.hash
                == VIRTUAL_INITIAL_MODEL.hash
            ):
                # Consider minimal models that have all necessarily-fixed
                # parameters.
                estimated_parameters = {
                    parameter_id: ESTIMATE
                    for parameter_id in (
                        set(self.parameters)
                        .difference(self.must_fix)
                        .difference(old_fixed)
                    )
                }
                models = self.get_models(
                    estimated_parameters=estimated_parameters,
                )
                previous_number_of_candidates = len(candidate_space.models)
                for model in models:
                    continue_sending = self.send_model_to_candidate_space(
                        model=model,
                        candidate_space=candidate_space,
                    )
                    if not continue_searching(continue_sending):
                        return

                # No need to consider other models, as they will necessarily
                # be worse than the current set of models in the candidate
                # space, if suitable candidates models have already been
                # identified.
                if len(candidate_space.models) > previous_number_of_candidates:
                    return

            # Keep track of the number of new fixed parameters.
            # Stop considering models once all parameter sets with the minimal number of
            # new fixed parameters are considered.
            n_new_fixed = np.inf
            previous_number_of_candidates = len(candidate_space.models)
            # The powerset should be in ascending order by number of elements.
            for parameter_set in powerset(new_can_fix_optional):
                try:
                    # The case of a "minimal" model in the subspace being a valid candidate
                    # in this case should have been handled above already with
                    # `new_must_estimate_all`
                    if not parameter_set:
                        continue
                    # If a model has been accepted by the candidate space, only
                    # consider models of the same size (same minimal increase
                    # in the number of new fixed parameters), then stop.
                    if len(parameter_set) > n_new_fixed:
                        break
                    estimated_parameters = (
                        set(old_estimated)
                        .union(new_must_estimate)
                        .difference(parameter_set)
                    )
                    models = self.get_models(
                        estimated_parameters=list(estimated_parameters),
                    )
                    for model in models:
                        continue_sending = self.send_model_to_candidate_space(
                            model=model,
                            candidate_space=candidate_space,
                        )
                        if not continue_searching(continue_sending):
                            return
                    # If model accepted set the number of new fixed parameters to
                    # current number of new fixed parameters
                    if (
                        len(candidate_space.models)
                        > previous_number_of_candidates
                    ):
                        n_new_fixed = len(parameter_set)
                except StopIteration:
                    break

        elif candidate_space.method == Method.BRUTE_FORCE:
            # TODO remove list?
            for parameterization in list(product(*self.parameters.values())):
                parameters = dict(
                    zip(self.parameters, parameterization, strict=False)
                )
                model = self.parameters_to_model(parameters)
                # Skip models that are excluded.
                if model is None:
                    continue
                continue_sending = self.send_model_to_candidate_space(
                    model=model,
                    candidate_space=candidate_space,
                )
                if not continue_searching(continue_sending):
                    return

        elif candidate_space.method == Method.LATERAL:
            # There is an equal number of new necessarily estimated and fixed
            # parameters.
            if len(new_must_estimate_all) != len(new_must_fix_all):
                return

            if (
                # `and` is redundant with the "equal number" check above.
                (new_must_estimate_all and new_must_fix_all)
                or candidate_space.predecessor_model.hash
                == VIRTUAL_INITIAL_MODEL.hash
            ):
                # Consider all models that have the required estimated and
                # fixed parameters.
                estimated_parameters = {
                    parameter_id: ESTIMATE
                    for parameter_id in [*old_estimated, *new_must_estimate]
                    if parameter_id not in new_must_fix_all
                }
                models = self.get_models(
                    estimated_parameters=estimated_parameters,
                )
                previous_number_of_candidates = len(candidate_space.models)
                for model in models:
                    continue_sending = self.send_model_to_candidate_space(
                        model=model,
                        candidate_space=candidate_space,
                    )
                    if not continue_searching(continue_sending):
                        return

                # No need to consider other models, as they will necessarily
                # be worse than the current set of models in the candidate
                # space, if suitable candidates models have already been
                # identified.
                if len(candidate_space.models) > previous_number_of_candidates:
                    return

            # Keep track of the number of lateral moves performed.
            # Stop considering models once all parameter sets with the smallest
            # lateral move size are considered.
            n_lateral_moves = np.inf
            previous_number_of_candidates = len(candidate_space.models)
            # The powerset should be in ascending order by size of lateral
            # move.
            for parameter_set_estimate, parameter_set_fix in product(
                powerset(new_can_estimate_optional),
                powerset(new_can_fix_optional),
            ):
                try:
                    # At least some parameters must change.
                    if not parameter_set_estimate or not parameter_set_fix:
                        continue
                    # The same number of parameters must be fixed and estimated.
                    if len(parameter_set_estimate) != len(parameter_set_fix):
                        continue
                    # If a model has been accepted by the candidate space, only
                    # consider models of the same step size (same minimal
                    # number of steps in the lateral move), then stop.
                    if len(parameter_set_estimate) > n_lateral_moves:
                        break
                    estimated_parameters = (
                        set(old_estimated)
                        .union(new_must_estimate)
                        .union(parameter_set_estimate)
                        .difference(parameter_set_fix)
                    )
                    models = self.get_models(
                        estimated_parameters=list(estimated_parameters),
                    )
                    for model in models:
                        continue_sending = self.send_model_to_candidate_space(
                            model=model,
                            candidate_space=candidate_space,
                        )
                        if not continue_searching(continue_sending):
                            return
                    # If model accepted set the number of lateral moves to
                    # current number of lateral moves
                    if (
                        len(candidate_space.models)
                        > previous_number_of_candidates
                    ):
                        n_lateral_moves = len(parameter_set_estimate)
                except StopIteration:
                    break

        else:
            raise NotImplementedError(
                "The requested method is not yet implemented in the model "
                f"subspace interface: `{candidate_space.method}`."
            )

    def send_model_to_candidate_space(
        self,
        model: Model,
        candidate_space: CandidateSpace,
        exclude: bool | None = False,
        # use_exclusions: Optional[bool] = True,
    ) -> bool:
        """Send a model to a candidate space for consideration.

        Args:
            model:
                The model.
            candidate_space:
                The candidate space.
            exclude:
                Whether to add the model to the exclusions.

        Returns:
            Whether it is OK to send additional models to the candidate space. For
            example, if `len(candidate_space.models) == candidate_space.limit`, then
            no further models should be sent.
        """
        # TODO if different sources of `Model` are possible (not just
        # `ModelSubspace.indices_to_model`), then would need to manage exclusions there
        # or here.
        # if use_exclusions and hash(model) in self.exclusions:
        #    return True

        if exclude:
            self.exclude_model(model)
        # `result` is whether it is OK to send additional models to the candidate space.
        continue_sending = candidate_space.consider(model)
        return continue_sending

    def exclude_model_hash(self, model_hash: str) -> None:
        """Exclude a model hash from the model subspace.

        Args:
            model_hash:
                The model hash.
        """
        self.exclusions.add(model_hash)

    def exclude_model_hashes(self, model_hashes: Iterable[str]) -> None:
        """Exclude model hashes from the model subspace.

        Args:
            model_hashes:
                The model hashes.
        """
        for model_hash in model_hashes:
            self.exclude_model_hash(model_hash=model_hash)

    def exclude_model(self, model: Model) -> None:
        """Exclude a model from the model subspace.

        Models are excluded in `ModelSubspace.indices_to_model`, which contains the
        only call to `Model.__init__` in the `ModelSubspace` class.

        Args:
            model:
                The model that will be excluded.
        """
        self.exclude_model_hash(model_hash=model.hash)

    def exclude_models(self, models: Iterable[Model]) -> None:
        """Exclude models from the model subspace.

        Models are excluded in :meth:`ModelSubspace.indices_to_model`, which contains the
        only call to :meth:`Model.__init__` in the :class:`ModelSubspace` class.

        Args:
            models:
                The models that will be excluded.
        """
        for model in models:
            self.exclude_model(model)

    def excluded(
        self,
        model: Model,
    ) -> bool:
        """Whether a model is excluded."""
        return model.hash in self.exclusions

    def reset_exclusions(
        self,
        # TODO change typing with `List[Any]` to some `List[TYPE_MODEL_HASH]`
        exclusions: list[Any] | None | None = None,
    ):
        self.exclusions = set()
        if exclusions is not None:
            self.exclusions = set(exclusions)

    def reset(
        self,
        exclusions: list[Any] | None | None = None,
        limit: int | None = None,
    ):
        self.reset_exclusions(exclusions=exclusions)
        if limit is not None:
            self.set_limit(limit)

    @staticmethod
    def from_definition(
        definition: dict[str, str] | pd.Series,
        root_path: TYPE_PATH = None,
    ) -> "ModelSubspace":
        """Create a :class:`ModelSubspace` from a definition.

        Args:
            definition:
                A description of the model subspace. Keys are properties of the
                model subspace, including parameters that can take different
                values.
            root_path:
                Any paths will be resolved relative to this path.

        Returns:
            The model subspace.
        """
        model_subspace_id = definition.pop(MODEL_SUBSPACE_ID)
        if "petab_yaml" in definition:
            petab_yaml = definition.pop("petab_yaml")
            warnings.warn(
                "Change the `petab_yaml` column to "
                "`model_subspace_petab_yaml`, in the model space TSV.",
                DeprecationWarning,
                stacklevel=1,
            )
        else:
            petab_yaml = definition.pop(MODEL_SUBSPACE_PETAB_YAML)
        parameters = {
            column_id: decompress_parameter_values(value)
            for column_id, value in definition.items()
        }
        if root_path is not None:
            petab_yaml = Path(root_path) / petab_yaml
        return ModelSubspace(
            model_subspace_id=model_subspace_id,
            petab_yaml=petab_yaml,
            parameters=parameters,
        )

    def to_definition(self, root_path: TYPE_PATH | None = None) -> pd.Series:
        """Get the definition of the model subspace.

        Args:
            root_path:
                If provided, the ``model_subspace_petab_yaml`` will be made
                relative to this path.

        Returns:
            The definition.
        """
        petab_yaml = self.petab_yaml
        if root_path:
            petab_yaml = relpath(petab_yaml, start=root_path)
        return pd.Series(
            {
                MODEL_SUBSPACE_ID: self.model_subspace_id,
                MODEL_SUBSPACE_PETAB_YAML: petab_yaml,
                **{
                    parameter_id: PARAMETER_VALUE_DELIMITER.join(
                        str(v) for v in values
                    )
                    for parameter_id, values in self.parameters.items()
                },
            }
        )

    def indices_to_model(self, indices: list[int]) -> Model | None:
        """Get a model from the subspace, by indices of possible parameter values.

        Model exclusions are handled here.

        Args:
            indices:
                The indices of the lists in the values of the ``ModelSubspace.parameters``
                dictionary, ordered by the keys of this dictionary.

        Returns:
            A model with the PEtab problem of this subspace and the parameterization
            that corresponds to the indices.
            ``None``, if the model is excluded from the subspace.
        """
        model = Model(
            model_subspace_id=self.model_subspace_id,
            model_subspace_indices=indices,
            model_subspace_petab_yaml=self.petab_yaml,
            parameters=self.indices_to_parameters(indices),
            _model_subspace_petab_problem=self.petab_problem,
        )
        if self.excluded(model):
            return None
        return model

    def indices_to_parameters(
        self,
        indices: list[int],
    ) -> TYPE_PARAMETER_DICT:
        """Convert parameter indices to values.

        Args:
            indices:
                See :meth:`ModelSubspace.indices_to_model`.

        Returns:
            The parameterization that corresponds to the indices.
        """
        parameters = {
            parameter_id: self.parameters[parameter_id][index]
            for parameter_id, index in zip(
                self.parameters, indices, strict=False
            )
        }
        return parameters

    def parameters_to_indices(self, parameters: TYPE_PARAMETER_DICT):
        """Convert parameter values to indices.

        Args:
            parameters:
                Keys are parameter IDs, values are parameter values.

        Returns:
            The indices of the subspace that correspond to the parameterization.
        """
        if set(self.parameters).symmetric_difference(parameters):
            raise ValueError(
                "Parameter IDs differ between the stored and provided "
                "values: "
                f"{set(self.parameters).symmetric_difference(parameters)}"
            )
        indices = []
        for parameter_id, parameter_values in self.parameters.items():
            try:
                index = parameter_values.index(parameters[parameter_id])
            except ValueError:
                raise ValueError(
                    f"The following value for the parameter {parameter_id} is "
                    f"not in the model subspace: {parameters[parameter_id]}."
                )
            indices.append(index)
        return indices

    def parameters_to_model(
        self,
        parameters: TYPE_PARAMETER_DICT,
    ) -> Model | None:
        """Convert parameter values to a model.

        Args:
            parameters:
                Keys are parameter IDs, values are parameter values.

        Returns:
            A model with the PEtab problem of this subspace and the parameterization.
            ``None``, if the model is excluded from the subspace.
        """
        indices = self.parameters_to_indices(parameters)
        model = self.indices_to_model(indices)
        return model

    @property
    def parameters_all(self) -> TYPE_PARAMETER_DICT:
        """Get all parameters, including those only in the PEtab problem.

        Parameter values in the PEtab problem are overwritten by the
        model subspace values.
        """
        return {
            **get_petab_parameters(self.petab_problem, as_lists=True),
            **self.parameters,
        }

    @property
    def can_fix(self) -> list[str]:
        """Parameters that can be fixed, according to the subspace.

        Parameters that are fixed as part of the PEtab problem are not
        considered.
        """
        return [
            parameter_id
            for parameter_id, parameter_values in self.parameters.items()
            if parameter_values != [ESTIMATE]
        ]

    @property
    def can_fix_all(self) -> list[str]:
        """All arameters that can be fixed, according to the subspace."""
        return [
            parameter_id
            for parameter_id, parameter_values in self.parameters_all.items()
            if parameter_values != [ESTIMATE]
        ]

    @property
    def can_estimate(self) -> list[str]:
        """Parameters that can be estimated, according to the subspace.

        Parameters that are estimated as part of the PEtab problem are not
        considered.
        """
        return [
            parameter_id
            for parameter_id in self.parameters
            if ESTIMATE in self.parameters[parameter_id]
        ]

    @property
    def can_estimate_all(self) -> list[str]:
        """All parameters than can be estimated in this subspace."""
        return [
            parameter_id
            for parameter_id, parameter_values in self.parameters_all.items()
            if ESTIMATE in parameter_values
        ]

    @property
    def must_fix(self) -> list[str]:
        """Subspace parameters that must be fixed.

        Parameters that are fixed as part of the PEtab problem are not
        considered.
        """
        return [
            parameter_id
            for parameter_id in self.parameters
            if parameter_id not in self.can_estimate_all
        ]

    @property
    def must_fix_all(self) -> list[str]:
        """All parameters that must be fixed in this subspace."""
        return [
            parameter_id
            for parameter_id in self.parameters_all
            if parameter_id not in self.can_estimate_all
        ]

    @property
    def must_estimate(self) -> list[str]:
        """Subspace parameters that must be estimated.

        Does not include parameters that are estimated in the PEtab
        parameters table.
        """
        return [
            parameter_id
            for parameter_id, parameter_values in self.parameters.items()
            if parameter_values == [ESTIMATE]
        ]

    @property
    def must_estimate_all(self) -> list[str]:
        """All parameters that must be estimated in this subspace."""
        must_estimate_petab = [
            parameter_id
            for parameter_id in self.petab_problem.x_free_ids
            if parameter_id not in self.parameters
        ]
        return [*must_estimate_petab, *self.must_estimate]

    def get_estimated(
        self,
        additional_parameters: TYPE_PARAMETER_DICT | None = None,
    ) -> list[str]:
        """Get the IDs of parameters that are estimated.

        Args:
            additional_parameters:
                A specific parameterization that will take priority when
                determining estimated parameters.

        Returns:
            The parameter IDs.
        """
        raise NotImplementedError

        # parameters = []
        # for parameter_id, parameter_value in self.parameters_all.items():
        #     if additional_parameters.get(parameter_id, None) == ESTIMATE:
        #         parameters.append(parameter_id)
        #         continue
        #
        #     if parameter_id in additional_parameters:
        #         # Presumably not estimated.
        #         continue
        #
        # old_estimated_all = {
        #     parameter_id
        #     for parameter_id, parameter_values in self.parameters_all.items()
        #     if
        #     (
        #         # Predecessor model sets the parameter to be estimated
        #         (
        #             candidate_space.predecessor_model.parameters.get(
        #                 parameter_id, None
        #             )
        #             == ESTIMATE
        #         )
        #         or (
        #             # Predecessor model takes the default PEtab parameter
        #             parameter_id
        #             not in candidate_space.predecessor_model.parameters
        #             and
        #             # And the default PEtab parameter is estimated
        #             # The PEtab problem of this subspace and the
        #             # `candidate_space` is the same, as verified earlier with
        #             # `self.check_compatibility_stepwise_method`.
        #             self.petab_parameters[parameter_id] == [ESTIMATE]
        #         )
        #     )
        # }

    def __len__(self) -> int:
        """Get the number of models in this subspace."""
        factors = [len(p) for p in self.parameters.values()]
        combinations = math.prod(factors)
        return combinations


def decompress_parameter_values(
    values: float | int | str,
) -> TYPE_PARAMETER_OPTIONS:
    """Decompress parameter values.

    TODO refactor to only allow `str` here (i.e. improve file parsing)?

    Args:
        values:
            Parameter values in the compressed format.

    Returns:
        Parameter values, decompressed into a list.
    """
    if isinstance(values, float | int):
        return [values]

    parameter_strings = list(values.split(PARAMETER_VALUE_DELIMITER))
    values_decompressed = []
    for parameter_string in parameter_strings:
        values_decompressed.append(
            parameter_string_to_value(
                parameter_string=parameter_string,
                passthrough_estimate=True,
            )
        )
    return values_decompressed
