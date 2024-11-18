"""The `Model` class."""

from __future__ import annotations

import warnings
from os.path import relpath
from pathlib import Path
from typing import TYPE_CHECKING, Any

import petab.v1 as petab
import yaml
from more_itertools import one
from petab.v1.C import ESTIMATE, NOMINAL_VALUE

from .constants import (
    CRITERIA,
    ESTIMATED_PARAMETERS,
    MODEL_HASH,
    MODEL_HASH_DELIMITER,
    MODEL_ID,
    MODEL_SUBSPACE_ID,
    MODEL_SUBSPACE_INDICES,
    MODEL_SUBSPACE_INDICES_HASH_DELIMITER,
    MODEL_SUBSPACE_INDICES_HASH_MAP,
    PARAMETERS,
    PETAB_ESTIMATE_TRUE,
    PETAB_PROBLEM,
    PETAB_YAML,
    PREDECESSOR_MODEL_HASH,
    TYPE_CRITERION,
    TYPE_PARAMETER,
    TYPE_PATH,
    VIRTUAL_INITIAL_MODEL,
    Criterion,
)
from .criteria import CriterionComputer
from .misc import (
    parameter_string_to_value,
)
from .petab import PetabMixin

if TYPE_CHECKING:
    from .problem import Problem

__all__ = [
    "Model",
    "default_compare",
    "models_from_yaml_list",
    "models_to_yaml_list",
    "ModelHash",
]


class Model(PetabMixin):
    """A (possibly uncalibrated) model.

    NB: some of these attribute names correspond to constants defined in the
    `constants.py` file, to facilitate loading models from/saving models to
    disk (see the `saved_attributes` attribute).

    Attributes:
        converters_load:
            Functions to convert attributes from YAML to :class:`Model`.
        converters_save:
            Functions to convert attributes from :class:`Model` to YAML.
        criteria:
            The criteria values of the calibrated model (e.g. AIC).
        model_id:
            The model ID.
        petab_yaml:
            The path to the PEtab problem YAML file.
        parameters:
            Parameter values that will overwrite the PEtab problem definition,
            or change parameters to be estimated.
        estimated_parameters:
            Parameter estimates from a model calibration tool, for parameters
            that are specified as estimated in the PEtab problem or PEtab
            Select model YAML. These are untransformed values (i.e., not on
            log scale).
        saved_attributes:
            Attributes that will be saved to disk by the :meth:`Model.to_yaml`
            method.
    """

    saved_attributes = (
        MODEL_ID,
        MODEL_SUBSPACE_ID,
        MODEL_SUBSPACE_INDICES,
        MODEL_HASH,
        PREDECESSOR_MODEL_HASH,
        PETAB_YAML,
        PARAMETERS,
        ESTIMATED_PARAMETERS,
        CRITERIA,
    )
    converters_load = {
        MODEL_ID: lambda x: x,
        MODEL_SUBSPACE_ID: lambda x: x,
        MODEL_SUBSPACE_INDICES: lambda x: [] if not x else x,
        MODEL_HASH: lambda x: x,
        PREDECESSOR_MODEL_HASH: lambda x: x,
        PETAB_YAML: lambda x: x,
        PARAMETERS: lambda x: x,
        ESTIMATED_PARAMETERS: lambda x: x,
        CRITERIA: lambda x: {
            # `criterion_id_value` is the ID of the criterion in the enum `Criterion`.
            Criterion(criterion_id_value): float(criterion_value)
            for criterion_id_value, criterion_value in x.items()
        },
    }
    converters_save = {
        MODEL_ID: lambda x: str(x),
        MODEL_SUBSPACE_ID: lambda x: str(x),
        MODEL_SUBSPACE_INDICES: lambda x: [int(xi) for xi in x],
        MODEL_HASH: lambda x: str(x),
        PREDECESSOR_MODEL_HASH: lambda x: str(x) if x is not None else x,
        PETAB_YAML: lambda x: str(x),
        PARAMETERS: lambda x: {str(k): v for k, v in x.items()},
        # FIXME handle with a `set_estimated_parameters` method instead?
        # to avoid `float` cast here. Reason for cast is because e.g. pyPESTO
        # can provide type `np.float64`, which causes issues when writing to
        # YAML.
        # ESTIMATED_PARAMETERS: lambda x: x,
        ESTIMATED_PARAMETERS: lambda x: {
            str(id): float(value) for id, value in x.items()
        },
        CRITERIA: lambda x: {
            criterion_id.value: float(criterion_value)
            for criterion_id, criterion_value in x.items()
        },
    }

    def __init__(
        self,
        petab_yaml: TYPE_PATH,
        model_subspace_id: str = None,
        model_id: str = None,
        model_subspace_indices: list[int] = None,
        predecessor_model_hash: str = None,
        parameters: dict[str, int | float] = None,
        estimated_parameters: dict[str, int | float] = None,
        criteria: dict[str, float] = None,
        # Optionally provided to reduce repeated parsing of `petab_yaml`.
        petab_problem: petab.Problem | None = None,
        model_hash: Any | None = None,
    ):
        self.model_id = model_id
        self.model_subspace_id = model_subspace_id
        self.model_subspace_indices = model_subspace_indices
        # TODO clean parameters, ensure single float or str (`ESTIMATE`) type
        self.parameters = parameters
        self.estimated_parameters = estimated_parameters
        self.criteria = criteria

        self.predecessor_model_hash = predecessor_model_hash
        if self.predecessor_model_hash is not None:
            self.predecessor_model_hash = ModelHash.from_hash(
                self.predecessor_model_hash
            )

        if self.parameters is None:
            self.parameters = {}
        if self.estimated_parameters is None:
            self.estimated_parameters = {}
        if self.criteria is None:
            self.criteria = {}

        super().__init__(petab_yaml=petab_yaml, petab_problem=petab_problem)

        self.model_hash = None
        self.get_hash()
        if model_hash is not None:
            model_hash = ModelHash.from_hash(model_hash)
            if self.model_hash != model_hash:
                raise ValueError(
                    "The supplied model hash does not match the computed "
                    "model hash."
                )
        if self.model_id is None:
            self.model_id = self.get_hash()

        self.criterion_computer = CriterionComputer(self)

    def set_criterion(self, criterion: Criterion, value: float) -> None:
        """Set a criterion value for the model.

        Args:
            criterion:
                The criterion (e.g. ``petab_select.constants.Criterion.AIC``).
            value:
                The criterion value for the (presumably calibrated) model.
        """
        if criterion in self.criteria:
            warnings.warn(
                "Overwriting saved criterion value. "
                f"Criterion: {criterion}. Value: {self.get_criterion(criterion)}.",
                stacklevel=2,
            )
            # FIXME debug why value is overwritten during test case 0002.
            if False:
                print(
                    "Overwriting saved criterion value. "
                    f"Criterion: {criterion}. Value: {self.get_criterion(criterion)}."
                )
                breakpoint()
        self.criteria[criterion] = value

    def has_criterion(self, criterion: Criterion) -> bool:
        """Check whether the model provides a value for a criterion.

        Args:
            criterion:
                The criterion (e.g. `petab_select.constants.Criterion.AIC`).
        """
        # TODO also `and self.criteria[id] is not None`?
        return criterion in self.criteria

    def get_criterion(
        self,
        criterion: Criterion,
        compute: bool = True,
        raise_on_failure: bool = True,
    ) -> TYPE_CRITERION | None:
        """Get a criterion value for the model.

        Args:
            criterion:
                The ID of the criterion (e.g. ``petab_select.constants.Criterion.AIC``).
            compute:
                Whether to try to compute the criterion value based on other model
                attributes. For example, if the ``'AIC'`` criterion is requested, this
                can be computed from a predetermined model likelihood and its
                number of estimated parameters.
            raise_on_failure:
                Whether to raise a `ValueError` if the criterion could not be
                computed. If `False`, `None` is returned.

        Returns:
            The criterion value, or `None` if it is not available.
            TODO check for previous use of this method before `.get` was used
        """
        if criterion not in self.criteria and compute:
            self.compute_criterion(
                criterion=criterion,
                raise_on_failure=raise_on_failure,
            )
            # value = self.criterion_computer(criterion=id)
            # self.set_criterion(id=id, value=value)

        return self.criteria.get(criterion, None)

    def compute_criterion(
        self,
        criterion: Criterion,
        raise_on_failure: bool = True,
    ) -> TYPE_CRITERION:
        """Compute a criterion value for the model.

        The value will also be stored, which will overwrite any previously stored value
        for the criterion.

        Args:
            criterion:
                The ID of the criterion
                (e.g. :obj:`petab_select.constants.Criterion.AIC`).
            raise_on_failure:
                Whether to raise a `ValueError` if the criterion could not be
                computed. If `False`, `None` is returned.

        Returns:
            The criterion value.
        """
        try:
            criterion_value = self.criterion_computer(criterion)
            self.set_criterion(criterion, criterion_value)
            result = criterion_value
        except ValueError as err:
            if raise_on_failure:
                raise ValueError(
                    f"Insufficient information to compute criterion `{criterion}`."
                ) from err
            result = None
        return result

    def set_estimated_parameters(
        self,
        estimated_parameters: dict[str, float],
        scaled: bool = False,
    ) -> None:
        """Set the estimated parameters.

        Args:
            estimated_parameters:
                The estimated parameters.
            scaled:
                Whether the ``estimated_parameters`` values are on the scale
                defined in the PEtab problem (``True``), or untransformed
                (``False``).
        """
        if scaled:
            estimated_parameters = self.petab_problem.unscale_parameters(
                estimated_parameters
            )
        self.estimated_parameters = estimated_parameters

    @staticmethod
    def from_dict(
        model_dict: dict[str, Any],
        base_path: TYPE_PATH = None,
        petab_problem: petab.Problem = None,
    ) -> Model:
        """Generate a model from a dictionary of attributes.

        Args:
            model_dict:
                A dictionary of attributes. The keys are attribute
                names, the values are the corresponding attribute values for
                the model. Required attributes are the required arguments of
                the :meth:`Model.__init__` method.
            base_path:
                The path that any relative paths in the model are relative to
                (e.g. the path to the PEtab problem YAML file
                :meth:`Model.petab_yaml` may be relative).
            petab_problem:
                Optionally provide the PEtab problem, to avoid loading it multiple
                times.
                NB: This may cause issues if multiple models write to the same PEtab
                problem in memory.

        Returns:
            A model instance, initialized with the provided attributes.
        """
        unknown_attributes = set(model_dict).difference(Model.converters_load)
        if unknown_attributes:
            warnings.warn(
                "Ignoring unknown attributes: "
                + ", ".join(unknown_attributes),
                stacklevel=2,
            )

        if base_path is not None:
            model_dict[PETAB_YAML] = base_path / model_dict[PETAB_YAML]

        model_dict = {
            attribute: Model.converters_load[attribute](value)
            for attribute, value in model_dict.items()
            if attribute in Model.converters_load
        }
        model_dict[PETAB_PROBLEM] = petab_problem
        return Model(**model_dict)

    @staticmethod
    def from_yaml(model_yaml: TYPE_PATH) -> Model:
        """Generate a model from a PEtab Select model YAML file.

        Args:
            model_yaml:
                The path to the PEtab Select model YAML file.

        Returns:
            A model instance, initialized with the provided attributes.
        """
        with open(str(model_yaml)) as f:
            model_dict = yaml.safe_load(f)
        # TODO check that the hash is reproducible
        if isinstance(model_dict, list):
            try:
                model_dict = one(model_dict)
            except ValueError:
                if len(model_dict) <= 1:
                    raise
                raise ValueError(
                    "The provided YAML file contains a list with greater than "
                    "one element. Use the `models_from_yaml_list` method or "
                    "provide a PEtab Select model YAML file with only one "
                    "model specified."
                )

        return Model.from_dict(model_dict, base_path=Path(model_yaml).parent)

    def to_dict(
        self,
        resolve_paths: bool = True,
        paths_relative_to: str | Path = None,
    ) -> dict[str, Any]:
        """Generate a dictionary from the attributes of a :class:`Model` instance.

        Args:
            resolve_paths:
                Whether to resolve relative paths into absolute paths.
            paths_relative_to:
                If not ``None``, paths will be converted to be relative to this path.
                Takes priority over ``resolve_paths``.

        Returns:
            A dictionary of attributes. The keys are attribute
            names, the values are the corresponding attribute values for
            the model. Required attributes are the required arguments of
            the :meth:`Model.__init__` method.
        """
        model_dict = {}
        for attribute in self.saved_attributes:
            model_dict[attribute] = self.converters_save[attribute](
                getattr(self, attribute)
            )
        # TODO test
        if resolve_paths:
            if model_dict[PETAB_YAML]:
                model_dict[PETAB_YAML] = str(
                    Path(model_dict[PETAB_YAML]).resolve()
                )
        if paths_relative_to is not None:
            if model_dict[PETAB_YAML]:
                model_dict[PETAB_YAML] = relpath(
                    Path(model_dict[PETAB_YAML]).resolve(),
                    Path(paths_relative_to).resolve(),
                )
        return model_dict

    def to_yaml(self, petab_yaml: TYPE_PATH, *args, **kwargs) -> None:
        """Generate a PEtab Select model YAML file from a :class:`Model` instance.

        Parameters:
            petab_yaml:
                The location where the PEtab Select model YAML file will be
                saved.
            args, kwargs:
                Additional arguments are passed to ``self.to_dict``.
        """
        # FIXME change `getattr(self, PETAB_YAML)` to be relative to
        # destination?
        # kind of fixed, as the path will be resolved in `to_dict`.
        with open(petab_yaml, "w") as f:
            yaml.dump(self.to_dict(*args, **kwargs), f)
        # yaml.dump(self.to_dict(), str(petab_yaml))

    def to_petab(
        self,
        output_path: TYPE_PATH = None,
        set_estimated_parameters: bool | None = None,
    ) -> dict[str, petab.Problem | TYPE_PATH]:
        """Generate a PEtab problem.

        Args:
            output_path:
                The directory where PEtab files will be written to disk. If not
                specified, the PEtab files will not be written to disk.
            set_estimated_parameters:
                Whether to set the nominal value of estimated parameters to their
                estimates. If parameter estimates are available, this
                will default to `True`.

        Returns:
            A 2-tuple. The first value is a PEtab problem that can be used
            with a PEtab-compatible tool for calibration of this model. If
            ``output_path`` is not ``None``, the second value is the path to a
            PEtab YAML file that can be used to load the PEtab problem (the
            first value) into any PEtab-compatible tool.
        """
        # TODO could use `copy.deepcopy(self.petab_problem)` from PetabMixin?
        petab_problem = petab.Problem.from_yaml(str(self.petab_yaml))

        if set_estimated_parameters is None and self.estimated_parameters:
            set_estimated_parameters = True

        for parameter_id, parameter_value in self.parameters.items():
            # If the parameter is to be estimated.
            if parameter_value == ESTIMATE:
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 1

                if set_estimated_parameters:
                    if parameter_id not in self.estimated_parameters:
                        raise ValueError(
                            "Not all estimated parameters are available "
                            "in `model.estimated_parameters`. Hence, the "
                            "estimated parameter vector cannot be set as "
                            "the nominal value in the PEtab problem. "
                            "Try calling this method with "
                            "`set_estimated_parameters=False`."
                        )
                    petab_problem.parameter_df.loc[
                        parameter_id, NOMINAL_VALUE
                    ] = self.estimated_parameters[parameter_id]
            # Else the parameter is to be fixed.
            else:
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 0
                petab_problem.parameter_df.loc[parameter_id, NOMINAL_VALUE] = (
                    parameter_string_to_value(parameter_value)
                )
                # parameter_value

        petab_yaml = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            petab_yaml = petab_problem.to_files_generic(
                prefix_path=output_path
            )

        return {
            PETAB_PROBLEM: petab_problem,
            PETAB_YAML: petab_yaml,
        }

    def get_hash(self) -> str:
        """Get the model hash.

        See the documentation for :class:`ModelHash` for more information.

        This is not implemented as ``__hash__`` because Python automatically
        truncates values in a system-dependent manner, which reduces
        interoperability
        ( https://docs.python.org/3/reference/datamodel.html#object.__hash__ ).

        Returns:
            The hash.
        """
        if self.model_hash is None:
            self.model_hash = ModelHash.from_model(model=self)
        return self.model_hash

    def __hash__(self) -> None:
        """Use `Model.get_hash` instead."""
        raise NotImplementedError("Use `Model.get_hash() instead.`")

    def __str__(self):
        """Get a print-ready string representation of the model.

        Returns:
            The print-ready string representation, in TSV format.
        """
        parameter_ids = "\t".join(self.parameters.keys())
        parameter_values = "\t".join(str(v) for v in self.parameters.values())
        header = "\t".join([MODEL_ID, PETAB_YAML, parameter_ids])
        data = "\t".join(
            [self.model_id, str(self.petab_yaml), parameter_values]
        )
        # header = f'{MODEL_ID}\t{PETAB_YAML}\t{parameter_ids}'
        # data = f'{self.model_id}\t{self.petab_yaml}\t{parameter_values}'
        return f"{header}\n{data}"

    def get_mle(self) -> dict[str, float]:
        """Get the maximum likelihood estimate of the model."""
        """
        FIXME(dilpath)
        # Check if original PEtab problem or PEtab Select model has estimated
        # parameters. e.g. can use some of `self.to_petab` to get the parameter
        # df and see if any are estimated.
        if not self.has_estimated_parameters:
            warn('The MLE for this model contains no estimated parameters.')
        if not all([
            parameter_id in getattr(self, ESTIMATED_PARAMETERS)
            for parameter_id in self.get_estimated_parameter_ids_all()
        ]):
            warn('Not all estimated parameters have estimates stored.')
        petab_problem = petab.Problem.from_yaml(str(self.petab_yaml))
        return {
            parameter_id: (
                getattr(self, ESTIMATED_PARAMETERS).get(
                    # Return estimated parameter from `petab_select.Model`
                    # if possible.
                    parameter_id,
                    # Else return nominal value from PEtab parameter table.
                    petab_problem.parameter_df.loc[
                        parameter_id, NOMINAL_VALUE
                    ],
                )
            )
            for parameter_id in petab_problem.parameter_df.index
        }
        # TODO rewrite to construct return dict in a for loop, for more
        # informative error message as soon as a "should-be-estimated"
        # parameter has not estimate available in `self.estimated_parameters`.
        """
        # TODO
        pass

    def get_estimated_parameter_ids_all(self) -> list[str]:
        estimated_parameter_ids = []

        # Add all estimated parameters in the PEtab problem.
        petab_problem = petab.Problem.from_yaml(str(self.petab_yaml))
        for parameter_id in petab_problem.parameter_df.index:
            if (
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE]
                == PETAB_ESTIMATE_TRUE
            ):
                estimated_parameter_ids.append(parameter_id)

        # Add additional estimated parameters, and collect fixed parameters,
        # in this model's parameterization.
        fixed_parameter_ids = []
        for parameter_id, value in self.parameters.items():
            if (
                value == ESTIMATE
                and parameter_id not in estimated_parameter_ids
            ):
                estimated_parameter_ids.append(parameter_id)
            elif value != ESTIMATE:
                fixed_parameter_ids.append(parameter_id)

        # Remove fixed parameters.
        estimated_parameter_ids = [
            parameter_id
            for parameter_id in estimated_parameter_ids
            if parameter_id not in fixed_parameter_ids
        ]

        return estimated_parameter_ids

    def get_parameter_values(
        self,
        parameter_ids: list[str] | None = None,
    ) -> list[TYPE_PARAMETER]:
        """Get parameter values.

        Includes ``ESTIMATE`` for parameters that should be estimated.

        The ordering is by ``parameter_ids`` if supplied, else
        ``self.petab_parameters``.

        Args:
            parameter_ids:
                The IDs of parameters that values will be returned for. Order
                is maintained.

        Returns:
            The values of parameters.
        """
        if parameter_ids is None:
            parameter_ids = list(self.petab_parameters)
        return [
            self.parameters.get(
                parameter_id,
                # Default to PEtab problem.
                self.petab_parameters[parameter_id],
            )
            for parameter_id in parameter_ids
        ]


def default_compare(
    model0: Model,
    model1: Model,
    criterion: Criterion,
    criterion_threshold: float = 0,
) -> bool:
    """Compare two calibrated models by their criterion values.

    It is assumed that the model ``model0`` provides a value for the criterion
    ``criterion``, or is the ``VIRTUAL_INITIAL_MODEL``.

    Args:
        model0:
            The original model.
        model1:
            The new model.
        criterion:
            The criterion by which models will be compared.
        criterion_threshold:
            The value by which the new model must improve on the original
            model. Should be non-negative.

    Returns:
        ``True` if ``model1`` has a better criterion value than ``model0``, else
        ``False``.
    """
    if not model1.has_criterion(criterion):
        warnings.warn(
            f'Model "{model1.model_id}" does not provide a value for the '
            f'criterion "{criterion}".',
            stacklevel=2,
        )
        return False
    if model0 == VIRTUAL_INITIAL_MODEL or model0 is None:
        return True
    if criterion_threshold < 0:
        warnings.warn(
            "The provided criterion threshold is negative. "
            "The absolute value will be used instead.",
            stacklevel=2,
        )
        criterion_threshold = abs(criterion_threshold)
    if criterion in [
        Criterion.AIC,
        Criterion.AICC,
        Criterion.BIC,
        Criterion.NLLH,
        Criterion.SSR,
    ]:
        return (
            model1.get_criterion(criterion)
            < model0.get_criterion(criterion) - criterion_threshold
        )
    elif criterion in [
        Criterion.LH,
        Criterion.LLH,
    ]:
        return (
            model1.get_criterion(criterion)
            > model0.get_criterion(criterion) + criterion_threshold
        )
    else:
        raise NotImplementedError(f"Unknown criterion: {criterion}.")


def models_from_yaml_list(
    model_list_yaml: TYPE_PATH,
    petab_problem: petab.Problem = None,
    allow_single_model: bool = True,
) -> list[Model]:
    """Generate a model from a PEtab Select list of model YAML file.

    Args:
        model_list_yaml:
            The path to the PEtab Select list of model YAML file.
        petab_problem:
            See :meth:`Model.from_dict`.
        allow_single_model:
            Given a YAML file that contains a single model directly (not in
            a 1-element list), if ``True`` then the single model will be read in,
            else a ``ValueError`` will be raised.

    Returns:
        A list of model instances, initialized with the provided
        attributes.
    """
    with open(str(model_list_yaml)) as f:
        model_dict_list = yaml.safe_load(f)
    if not model_dict_list:
        return []

    if not isinstance(model_dict_list, list):
        if allow_single_model:
            return [
                Model.from_dict(
                    model_dict_list,
                    base_path=Path(model_list_yaml).parent,
                    petab_problem=petab_problem,
                )
            ]
        raise ValueError("The YAML file does not contain a list of models.")

    return [
        Model.from_dict(
            model_dict,
            base_path=Path(model_list_yaml).parent,
            petab_problem=petab_problem,
        )
        for model_dict in model_dict_list
    ]


def models_to_yaml_list(
    models: list[Model | str] | dict[ModelHash, Model | str],
    output_yaml: TYPE_PATH,
    relative_paths: bool = True,
) -> None:
    """Generate a YAML listing of models.

    Args:
        models:
            The models.
        output_yaml:
            The location where the YAML will be saved.
        relative_paths:
            Whether to rewrite the paths in each model (e.g. the path to the
            model's PEtab problem) relative to the `output_yaml` location.
    """
    if isinstance(models, dict):
        models = list(models.values())

    skipped_indices = []
    for index, model in enumerate(models):
        if isinstance(model, Model):
            continue
        if model == VIRTUAL_INITIAL_MODEL:
            continue
        warnings.warn(f"Unexpected model, skipping: {model}.", stacklevel=2)
        skipped_indices.append(index)
    models = [
        model
        for index, model in enumerate(models)
        if index not in skipped_indices
    ]

    paths_relative_to = None
    if relative_paths:
        paths_relative_to = Path(output_yaml).parent
    model_dicts = [
        model.to_dict(paths_relative_to=paths_relative_to) for model in models
    ]
    model_dicts = None if not model_dicts else model_dicts
    with open(output_yaml, "w") as f:
        yaml.dump(model_dicts, f)


class ModelHash(str):
    """A class to handle model hash functionality.

    The model hash is designed to be human-readable and able to be converted
    back into the corresponding model. Currently, if two models from two
    different model subspaces are actually the same PEtab problem, they will
    still have different model hashes.

    Attributes:
        model_subspace_id:
            The ID of the model subspace of the model. Unique up to a single
            PEtab Select problem model space.
        model_subspace_indices_hash:
            A hash of the location of the model in its model
            subspace. Unique up to a single model subspace.
    """

    # FIXME petab problem--specific hashes that are cross-platform?
    """
    The model hash is designed to be: human-readable; able to be converted
    back into the corresponding model, and unique up to the same PEtab
    problem and parameters.

    Consider two different models in different model subspaces, with
    `ModelHash`s `model_hash0` and `model_hash1`, respectively. Assume that
    these two models end up encoding the same PEtab problem (e.g. they set the
    same parameters to be estimated).
    The string representation will be different,
    `str(model_hash0) != str(model_hash1)`, but their hashes will pass the
    equality check: `model_hash0 == model_hash1` and
    `hash(model_hash0) == hash(model_hash1)`.

    This means that different models in different model subspaces that end up
    being the same PEtab problem will have different human-readable hashes,
    but if these models arise during model selection, then only one of them
    will be calibrated.

    The PEtab hash size is computed automatically as the smallest size that
    ensures a collision probability of less than $2^{-64}$.
    N.B.: this assumes only one model subspace, and only 2 options for each
    parameter (e.g. `0` and `estimate`). You can manually set the size with
    :const:`petab_select.constants.PETAB_HASH_DIGEST_SIZE`.

    petab_hash:
        A hash that is unique up to the same PEtab problem, which is
        determined by: the PEtab problem YAML file location, nominal
        parameter values, and parameters set to be estimated. This means
        that different models may have the same `unique_petab_hash`,
        because they are the same estimation problem.
    """

    def __init__(
        self,
        model_subspace_id: str,
        model_subspace_indices_hash: str,
        # petab_hash: str,
    ):
        self.model_subspace_id = model_subspace_id
        self.model_subspace_indices_hash = model_subspace_indices_hash
        # self.petab_hash = petab_hash

    def __new__(
        cls,
        model_subspace_id: str,
        model_subspace_indices_hash: str,
        # petab_hash: str,
    ):
        hash_str = MODEL_HASH_DELIMITER.join(
            [
                model_subspace_id,
                model_subspace_indices_hash,
                # petab_hash,
            ]
        )
        instance = super().__new__(cls, hash_str)
        return instance

    def __getnewargs_ex__(self):
        return (
            (),
            {
                "model_subspace_id": self.model_subspace_id,
                "model_subspace_indices_hash": self.model_subspace_indices_hash,
                # 'petab_hash': self.petab_hash,
            },
        )

    def __copy__(self):
        return ModelHash(
            model_subspace_id=self.model_subspace_id,
            model_subspace_indices_hash=self.model_subspace_indices_hash,
            # petab_hash=self.petab_hash,
        )

    def __deepcopy__(self, memo):
        return self.__copy__()

    # @staticmethod
    # def get_petab_hash(model: Model) -> str:
    #     """Get a hash that is unique up to the same estimation problem.

    #     See :attr:`petab_hash` for more information.

    #     Args:
    #         model:
    #             The model.

    #     Returns:
    #         The unique PEtab hash.
    #     """
    #     digest_size = PETAB_HASH_DIGEST_SIZE
    #     if digest_size is None:
    #         petab_info_bits = len(model.model_subspace_indices)
    #         # Ensure <2^{-64} probability of collision
    #         petab_info_bits += 64
    #         # Convert to bytes, round up.
    #         digest_size = int(petab_info_bits / 8) + 1

    #     petab_yaml = str(model.petab_yaml.resolve())
    #     model_parameter_df = model.to_petab(set_estimated_parameters=False)[
    #         PETAB_PROBLEM
    #     ].parameter_df
    #     nominal_parameter_hash = hash_parameter_dict(
    #         model_parameter_df[NOMINAL_VALUE].to_dict()
    #     )
    #     estimate_parameter_hash = hash_parameter_dict(
    #         model_parameter_df[ESTIMATE].to_dict()
    #     )
    #     return hash_str(
    #         petab_yaml + estimate_parameter_hash + nominal_parameter_hash,
    #         digest_size=digest_size,
    #     )

    @staticmethod
    def from_hash(model_hash: str | ModelHash) -> ModelHash:
        """Reconstruct a :class:`ModelHash` object.

        Args:
            model_hash:
                The model hash.

        Returns:
            The :class:`ModelHash` object.
        """
        if isinstance(model_hash, ModelHash):
            return model_hash

        if model_hash == VIRTUAL_INITIAL_MODEL:
            return ModelHash(
                model_subspace_id=VIRTUAL_INITIAL_MODEL,
                model_subspace_indices_hash="",
                # petab_hash=VIRTUAL_INITIAL_MODEL,
            )

        (
            model_subspace_id,
            model_subspace_indices_hash,
            # petab_hash,
        ) = model_hash.split(MODEL_HASH_DELIMITER)
        return ModelHash(
            model_subspace_id=model_subspace_id,
            model_subspace_indices_hash=model_subspace_indices_hash,
            # petab_hash=petab_hash,
        )

    @staticmethod
    def from_model(model: Model) -> ModelHash:
        """Create a hash for a model.

        Args:
            model:
                The model.

        Returns:
            The model hash.
        """
        model_subspace_id = ""
        model_subspace_indices_hash = ""
        if model.model_subspace_id is not None:
            model_subspace_id = model.model_subspace_id
            model_subspace_indices_hash = (
                ModelHash.hash_model_subspace_indices(
                    model.model_subspace_indices
                )
            )

        return ModelHash(
            model_subspace_id=model_subspace_id,
            model_subspace_indices_hash=model_subspace_indices_hash,
            # petab_hash=ModelHash.get_petab_hash(model=model),
        )

    @staticmethod
    def hash_model_subspace_indices(model_subspace_indices: list[int]) -> str:
        """Hash the location of a model in its subspace.

        Args:
            model_subspace_indices:
                The location (indices) of the model in its subspace.

        Returns:
            The hash.
        """
        try:
            return "".join(
                MODEL_SUBSPACE_INDICES_HASH_MAP[index]
                for index in model_subspace_indices
            )
        except KeyError:
            return MODEL_SUBSPACE_INDICES_HASH_DELIMITER.join(
                str(i) for i in model_subspace_indices
            )

    def unhash_model_subspace_indices(self) -> list[int]:
        """Get the location of a model in its subspace.

        Returns:
            The location, as indices of the subspace.
        """
        if (
            MODEL_SUBSPACE_INDICES_HASH_DELIMITER
            in self.model_subspace_indices_hash
        ):
            return [
                int(s)
                for s in self.model_subspace_indices_hash.split(
                    MODEL_SUBSPACE_INDICES_HASH_DELIMITER
                )
            ]
        else:
            return [
                MODEL_SUBSPACE_INDICES_HASH_MAP.index(s)
                for s in self.model_subspace_indices_hash
            ]

    def get_model(self, petab_select_problem: Problem) -> Model:
        """Get the model that a hash corresponds to.

        Args:
            petab_select_problem:
                The PEtab Select problem. The model will be found in its model
                space.

        Returns:
            The model.
        """
        # if self.petab_hash == VIRTUAL_INITIAL_MODEL:
        #     return self.petab_hash

        return petab_select_problem.model_space.model_subspaces[
            self.model_subspace_id
        ].indices_to_model(
            self.unhash_model_subspace_indices(
                self.model_subspace_indices_hash
            )
        )

    def __hash__(self) -> str:
        """The PEtab hash.

        N.B.: this is not the model hash! As the equality between two models
        is determined by their PEtab hash only, this method only returns the
        PEtab hash. However, the model hash is the full string with the
        human-readable elements as well. :func:`ModelHash.from_hash` does not
        accept the PEtab hash as input, rather the full string.
        """
        return hash(str(self))

    def __eq__(self, other_hash: str | ModelHash) -> bool:
        """Check whether two model hashes are equivalent.

        Returns:
            Whether the two hashes correspond to equivalent PEtab problems.
        """
        # petab_hash = other_hash
        # # Check whether the PEtab hash needs to be extracted
        # if MODEL_HASH_DELIMITER in other_hash:
        #     petab_hash = ModelHash.from_hash(other_hash).petab_hash
        # return self.petab_hash == petab_hash
        return str(self) == str(other_hash)
