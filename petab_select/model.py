"""The `Model` class."""
import abc
from pathlib import Path
from os.path import relpath
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import yaml

from more_itertools import one
import numpy as np
import petab
from petab.C import (
    ESTIMATE,
    NOMINAL_VALUE,
)

from .constants import (
    Criterion,
    CRITERIA,
    ESTIMATED_PARAMETERS,
    MODEL_HASH,
    MODEL_ID,
    MODEL_SUBSPACE_ID,
    MODEL_SUBSPACE_INDICES,
    PARAMETERS,
    PETAB_ESTIMATE_TRUE,
    PETAB_PROBLEM,
    PETAB_YAML,
    PREDECESSOR_MODEL_ID,
    TYPE_CRITERION,
    TYPE_PATH,
    TYPE_PARAMETER,
)
from .criteria import (
    CriterionComputer,
)
from .misc import (
    hash_list,
    hash_str,
    hash_parameter_dict,
    parameter_string_to_value,
)
from .petab import PetabMixin


class Model(PetabMixin):
    """A (possibly uncalibrated) model.

    NB: some of these attribute names correspond to constants defined in the
    `constants.py` file, to facilitate loading models from/saving models to
    disk (see the `saved_attributes` attribute).

    Attributes:
        converters_load:
            Functions to convert attributes from YAML to `Model`.
        converters_save:
            Functions to convert attributes from `Model` to YAML.
        criteria:
            The criteria values of the calibrated model (e.g. AIC).
        hash_attributes:
            This attribute is currently not used.
            Attributes that will be used to calculate the hash of the
            `Model` instance. NB: this hash is used during pairwise comparison
            to determine whether any two `Model` instances are unique. The
            model instances are compared by their parameter estimation
            problems, as opposed to parameter estimation results, which may
            differ due to e.g. floating-point arithmetic.
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
            Select model YAML.
        saved_attributes:
            Attributes that will be saved to disk by the `Model.to_yaml`
            method.
    """

    saved_attributes = (
        MODEL_ID,
        MODEL_SUBSPACE_ID,
        MODEL_SUBSPACE_INDICES,
        MODEL_HASH,
        PREDECESSOR_MODEL_ID,
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
        PREDECESSOR_MODEL_ID: lambda x: x,
        PETAB_YAML: lambda x: x,
        PARAMETERS: lambda x: x,
        ESTIMATED_PARAMETERS: lambda x: x,
        CRITERIA: lambda x: {
            # `criterion_id_value` is the ID of the criterion in the enum `Criterion`.
            Criterion(criterion_id_value): criterion_value
            for criterion_id_value, criterion_value in x.items()
        },
    }
    converters_save = {
        MODEL_ID: lambda x: x,
        MODEL_SUBSPACE_ID: lambda x: x,
        MODEL_SUBSPACE_INDICES: lambda x: x,
        MODEL_HASH: lambda x: x,
        PREDECESSOR_MODEL_ID: lambda x: x,
        PETAB_YAML: lambda x: str(x),
        PARAMETERS: lambda x: x,
        # FIXME handle with a `set_estimated_parameters` method instead?
        # to avoid `float` cast here. Reason for cast is because e.g. pyPESTO
        # can provide type `np.float64`, which causes issues when writing to
        # YAML.
        #ESTIMATED_PARAMETERS: lambda x: x,
        ESTIMATED_PARAMETERS: lambda x: {
            id: float(value)
            for id, value in x.items()
        },
        CRITERIA: lambda x: {
            criterion_id.value: criterion_value
            for criterion_id, criterion_value in x.items()
        },
    }
    hash_attributes = {
        # MODEL_ID: lambda x: hash(x),  # possible circular dependency on hash
        # MODEL_SUBSPACE_ID: lambda x: hash(x),
        # MODEL_SUBSPACE_INDICES: hash_list,
        # TODO replace `YAML` with `PETAB_PROBLEM_HASH`, as YAML could refer to
        #      different problems if used on different filesystems or sometimes
        #      absolute and other times relative. Better to check whether the
        #      PEtab problem itself is unique.
        # TODO replace `PARAMETERS` with `PARAMETERS_ALL`, which should be al
        #      parameters in the PEtab problem. This avoids treating the PEtab problem
        #      differently to the model (in a subspace with the PEtab problem) that has
        #      all nominal values defined in the subspace.
        # TODO add `estimated_parameters`? Needs to be clarified whether this hash
        #      should be unique amongst only not-yet-calibrated models, or may also
        #      return the same value between differently parameterized models that ended
        #      up being calibrated to be the same... probably should be the former.
        #      Currently, the hash is stored, hence will "persist" after calibration
        #      if the same `Model` instance is used.
        #PETAB_YAML: lambda x: hash(x),
        PETAB_YAML: hash_str,
        PARAMETERS: hash_parameter_dict,
    }

    def __init__(
        self,
        petab_yaml: TYPE_PATH,
        model_subspace_id: str = None,
        model_id: str = None,
        model_subspace_indices: List[int] = None,
        predecessor_model_id: str = None,
        parameters: Dict[str, Union[int, float]] = None,
        estimated_parameters: Dict[str, Union[int, float]] = None,
        criteria: Dict[str, float] = None,
        # Optionally provided to reduce repeated parsing of `petab_yaml`.
        petab_problem: Optional[petab.Problem] = None,
        model_hash: Optional[Any] = None,
    ):
        self.model_id = model_id
        self.model_subspace_id = model_subspace_id
        self.model_subspace_indices = model_subspace_indices
        # TODO clean parameters, ensure single float or str (`ESTIMATE`) type
        self.parameters = parameters
        self.estimated_parameters = estimated_parameters
        self.criteria = criteria
        self.model_hash = model_hash

        self.predecessor_model_id = predecessor_model_id

        if self.parameters is None:
            self.parameters = {}
        if self.estimated_parameters is None:
            self.estimated_parameters = {}
        if self.criteria is None:
            self.criteria = {}

        super().__init__(petab_yaml=petab_yaml, petab_problem=petab_problem)

        if self.model_id is None:
            self.model_id = self.get_hash()

        self.criterion_computer = CriterionComputer(self)

    def set_criterion(self, criterion: Criterion, value: float) -> None:
        """Set a criterion value for the model.

        Args:
            criterion:
                The criterion (e.g. `petab_select.constants.Criterion.AIC`).
            value:
                The criterion value for the (presumably calibrated) model.
        """
        if criterion in self.criteria:
            warnings.warn(
                'Overwriting saved criterion value. '
                f'Criterion: {criterion}. Value: {self.get_criterion(criterion)}.'
            )
            # FIXME debug why value is overwritten during test case 0002.
            if False:
                print(
                    'Overwriting saved criterion value. '
                    f'Criterion: {criterion}. Value: {self.get_criterion(criterion)}.'
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
    ) -> Union[TYPE_CRITERION, None]:
        """Get a criterion value for the model.

        Args:
            criterion:
                The ID of the criterion (e.g. `petab_select.constants.Criterion.AIC`).
            compute:
                Whether to try to compute the criterion value based on other model
                attributes. For example, if the `'AIC'` criterion is requested, this
                can be computed from a predetermined model likelihood and its
                number of estimated parameters.

        Returns:
            The criterion value, or `None` if it is not available.
            TODO check for previous use of this method before `.get` was used
        """
        if criterion not in self.criteria and compute:
            self.compute_criterion(criterion=criterion)
            #value = self.criterion_computer(criterion=id)
            #self.set_criterion(id=id, value=value)

        return self.criteria.get(criterion, None)

    def compute_criterion(self, criterion: Criterion) -> TYPE_CRITERION:
        """Compute a criterion value for the model.

        The value will also be stored, which will overwrite any previously stored value
        for the criterion.

        Args:
            id:
                The ID of the criterion (e.g. `petab_select.constants.Criterion.AIC`).

        Returns:
            The criterion value.
        """
        criterion_value = self.criterion_computer(criterion)
        self.set_criterion(criterion, criterion_value)
        return criterion_value

    @staticmethod
    def from_dict(
        model_dict: Dict[str, Any],
        base_path: TYPE_PATH = None,
    ) -> 'Model':
        """Generate a model from a dictionary of attributes.

        Args:
            model_dict:
                A dictionary of attributes. The keys are attribute
                names, the values are the corresponding attribute values for
                the model. Required attributes are the required arguments of
                the `Model.__init__` method.
            base_path:
                The path that any relative paths in the model are relative to
                (e.g. the path to the PEtab problem YAML file
                `Model.petab_yaml` may be relative).

        Returns:
            A model instance, initialized with the provided attributes.
        """
        unknown_attributes = set(model_dict).difference(Model.converters_load)
        if unknown_attributes:
            warnings.warn(
                'Ignoring unknown attributes: ' +
                ', '.join(unknown_attributes)
            )

        if base_path is not None:
            model_dict[PETAB_YAML] = base_path / model_dict[PETAB_YAML]

        model_dict = {
            attribute: Model.converters_load[attribute](value)
            for attribute, value in model_dict.items()
            if attribute in Model.converters_load
        }
        return Model(**model_dict)

    @staticmethod
    def from_yaml(model_yaml: TYPE_PATH) -> 'Model':
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
                    'The provided YAML file contains a list with greater than '
                    'one element. Use the `models_from_yaml_list` method or '
                    'provide a PEtab Select model YAML file with only one '
                    'model specified.'
                )

        return Model.from_dict(model_dict, base_path=Path(model_yaml).parent)

    def to_dict(
        self,
        resolve_paths: bool = True,
        paths_relative_to: Union[str, Path] = None,
    ) -> Dict[str, Any]:
        """Generate a dictionary from the attributes of a `Model` instance.

        Args:
            resolve_paths:
                Whether to resolve relative paths into absolute paths.
            paths_relative_to:
                If not `None`, paths will be converted to be relative to this path.
                Takes priority over `resolve_paths`.

        Returns:
            A dictionary of attributes. The keys are attribute
            names, the values are the corresponding attribute values for
            the model. Required attributes are the required arguments of
            the `Model.__init__` method.
        """
        model_dict = {}
        for attribute in self.saved_attributes:
            model_dict[attribute] = self.converters_save[attribute](
                getattr(self, attribute)
            )
        # TODO test
        if resolve_paths:
            if model_dict[PETAB_YAML]:
                model_dict[PETAB_YAML] = \
                    str(Path(model_dict[PETAB_YAML]).resolve())
        if paths_relative_to is not None:
            if model_dict[PETAB_YAML]:
                model_dict[PETAB_YAML] = relpath(
                    Path(model_dict[PETAB_YAML]).resolve(),
                    Path(paths_relative_to).resolve(),
                )
        return model_dict

    def to_yaml(self, petab_yaml: TYPE_PATH, *args, **kwargs) -> None:
        """Generate a PEtab Select model YAML file from a `Model` instance.

        Parameters:
            petab_yaml:
                The location where the PEtab Select model YAML file will be
                saved.
            args, kwargs:
                Additional arguments are passed to `self.to_dict`.
        """
        # FIXME change `getattr(self, PETAB_YAML)` to be relative to
        # destination?
        # kind of fixed, as the path will be resolved in `to_dict`.
        with open(petab_yaml, 'w') as f:
            yaml.dump(self.to_dict(*args, **kwargs), f)
        #yaml.dump(self.to_dict(), str(petab_yaml))

    def to_petab(
        self,
        output_path: TYPE_PATH = None,
    ) -> Tuple[petab.Problem, TYPE_PATH]:
        """Generate a PEtab problem.

        Args:
            output_path:
                The directory where PEtab files will be written to disk. If not
                specified, the PEtab files will not be written to disk.

        Returns:
            A 2-tuple. The first value is a PEtab problem that can be used
            with a PEtab-compatible tool for calibration of this model. If
            `output_path` is not `None`, the second value is the path to a
            PEtab YAML file that can be used to load the PEtab problem (the
            first value) into any PEtab-compatible tool. If 
        """
        # TODO could use `copy.deepcopy(self.petab_problem)` from PetabMixin?
        petab_problem = petab.Problem.from_yaml(str(self.petab_yaml))
        for parameter_id, parameter_value in self.parameters.items():
            # If the parameter is to be estimated.
            if parameter_value == ESTIMATE:
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 1
            # Else the parameter is to be fixed.
            else:
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 0
                petab_problem.parameter_df.loc[parameter_id, NOMINAL_VALUE] = \
                    parameter_string_to_value(parameter_value)
                #parameter_value

        petab_yaml = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            petab_yaml = petab_problem.to_files_generic(prefix_path=output_path)

        return {
            PETAB_PROBLEM: petab_problem,
            PETAB_YAML: petab_yaml,
        }

    def get_hash(self) -> int:
        """Get the model hash.

        Currently designed to only use pre-calibration information, such that if a model
        is calibrated twice and the two calibrated models differ in their parameter
        estimates, then they will still have the same hash.

        This is not implemented as `__hash__` because Python automatically truncates
        values in a system-dependent manner, which reduces interoperability
        ( https://docs.python.org/3/reference/datamodel.html#object.__hash__ ).

        Returns:
            The hash.
        """
        if self.model_hash is None:
            self.model_hash = hash_list([
                method(getattr(self, attribute))
                for attribute, method in Model.hash_attributes.items()
            ])
        return self.model_hash

    def __hash__(self) -> None:
        """Use `Model.get_hash` instead."""
        raise NotImplementedError('Use `Model.get_hash() instead.`')

    def __str__(self):
        """Get a print-ready string representation of the model.

        Returns:
            The print-ready string representation, in TSV format.
        """
        parameter_ids = '\t'.join(self.parameters.keys())
        parameter_values = '\t'.join(str(v) for v in self.parameters.values())
        header = '\t'.join([MODEL_ID, PETAB_YAML, parameter_ids])
        data = '\t'.join([self.model_id, str(self.petab_yaml), parameter_values])
        #header = f'{MODEL_ID}\t{PETAB_YAML}\t{parameter_ids}'
        #data = f'{self.model_id}\t{self.petab_yaml}\t{parameter_values}'
        return f'{header}\n{data}'

    def get_mle(self) -> Dict[str, float]:
        """Get the maximum likelihood estimate of the model.

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

    def get_estimated_parameter_ids_all(self) -> List[str]:
        estimated_parameter_ids = []

        # Add all estimated parameters in the PEtab problem.
        petab_problem = petab.Problem.from_yaml(str(self.petab_yaml))
        for parameter_id in petab_problem.parameter_df.index:
            if petab_problem.parameter_df.loc[parameter_id, ESTIMATE] == PETAB_ESTIMATE_TRUE:  # noqa: E501
                estimated_parameter_ids.append(parameter_id)

        # Add additional estimated parameters, and collect fixed parameters,
        # in this model's parameterization.
        fixed_parameter_ids = []
        for parameter_id, value in self.parameters.items():
            if (
                value == ESTIMATE
                and
                parameter_id not in estimated_parameter_ids
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
        parameter_ids: Optional[List[str]] = None,
    ) -> List[TYPE_PARAMETER]:
        """Get parameter values.

        Includes `ESTIMATE` for parameters that should be estimated.

        The ordering is by `parameter_ids` if supplied, else
        `self.petab_parameters`.

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
                self.petab_parameters[parameter_id]
            )
            for parameter_id in parameter_ids
        ]


def default_compare(
    model0: Model,
    model: Model,
    criterion: str,
) -> bool:
    """Compare two calibrated models by their criterion values.

    It is assumed that the model `model0` provides a value for the criterion
    `criterion`.

    Args:
        model0:
            The original model.
        model:
            The new model.

    Returns:
        `True` if `model` has a better criterion value than `model0`, else
        `False`.
    """
    if not model.has_criterion(criterion):
        warnings.warn(
            f'Model "{model.model_id}" does not provide a value for the '
            f'criterion "{criterion}".'
        )
        return False
    if criterion in [
        Criterion.AIC,
        Criterion.AICC,
        Criterion.BIC,
        Criterion.NLLH,
    ]:
        return model.get_criterion(criterion) < model0.get_criterion(criterion)
    elif criterion in [
        Criterion.LH,
        Criterion.LLH,
    ]:
        return model.get_criterion(criterion) > model0.get_criterion(criterion)
    else:
        raise NotImplementedError(
            f'Unknown criterion: {criterion}.'
        )


def models_from_yaml_list(model_list_yaml: TYPE_PATH) -> List[Model]:
    """Generate a model from a PEtab Select list of model YAML file.

    Args:
        model_list_yaml:
            The path to the PEtab Select list of model YAML file.

    Returns:
        A list of model instances, initialized with the provided
        attributes.
    """
    with open(str(model_list_yaml)) as f:
        model_dict_list = yaml.safe_load(f)
    if model_dict_list is None:
        return []
    return [
        Model.from_dict(model_dict, base_path=Path(model_list_yaml).parent)
        for model_dict in model_dict_list
    ]
