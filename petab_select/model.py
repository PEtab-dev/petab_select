"""The `Model` class."""
import abc
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
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
    AIC,
    AICC,
    BIC,

    CRITERIA,
    MODEL_ID,
    PREDECESSOR_MODEL_ID,
    PARAMETERS,
    ESTIMATED_PARAMETERS,
    PETAB_YAML,
    TYPING_PATH,

    ESTIMATE_SYMBOL_INTERNAL,
    ESTIMATE_SYMBOL_UI,
)
from .misc import (
    hash_dictionary,
)


class Model(abc.ABC):
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
        index:
            The index of the model in the model space that generated it.
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
        #HASH,
        MODEL_ID,
        PREDECESSOR_MODEL_ID,
        PETAB_YAML,
        PARAMETERS,
        ESTIMATED_PARAMETERS,
        CRITERIA,
    )
    converters_load = {
        MODEL_ID: lambda x: x,
        PREDECESSOR_MODEL_ID: lambda x: x,
        PETAB_YAML: lambda x: x,
        PARAMETERS: lambda x: {
            id: (
                ESTIMATE_SYMBOL_INTERNAL
                if value == ESTIMATE_SYMBOL_UI
                else value
            )
            for id, value in x.items()
        },
        ESTIMATED_PARAMETERS: lambda x: x,
        CRITERIA: lambda x: {} if not x else x,
    }
    converters_save = {
        MODEL_ID: lambda x: x,
        PREDECESSOR_MODEL_ID: lambda x: x,
        PETAB_YAML: lambda x: str(x),
        PARAMETERS: lambda x: {
            id: (
                ESTIMATE_SYMBOL_UI
                if np.isnan(value)
                else value
            )
            for id, value in x.items()
        },
        # FIXME handle with a `set_estimated_parameters` method instead?
        # to avoid `float` cast here. Reason for cast is because e.g. pyPESTO
        # can provide type `np.float64`, which causes issues when writing to
        # YAML.
        #ESTIMATED_PARAMETERS: lambda x: x,
        ESTIMATED_PARAMETERS: lambda x: {
            id: float(value)
            for id, value in x.items()
        },
        CRITERIA: lambda x: None if not x else x,
    }
    hash_attributes = {
        # TODO replace `YAML` with `PETAB_PROBLEM_HASH`, as YAML could refer to
        #      different problems if used on different filesystems or sometimes
        #      absolute and other times relative. Better to check whether the
        #      PEtab problem itself is unique.
        PETAB_YAML: lambda x: hash(x),
        PARAMETERS: hash_dictionary,
    }

    def __init__(
        self,
        model_id: str,
        petab_yaml: TYPING_PATH,
        predecessor_model_id: str = None,
        parameters: Dict[str, Union[int, float]] = None,
        estimated_parameters: Dict[str, Union[int, float]] = None,
        criteria: Dict[str, float] = None,
        index: int = None,
    ):
        self.model_id = model_id
        self.petab_yaml = Path(petab_yaml)
        self.parameters = parameters
        self.estimated_parameters = estimated_parameters
        self.criteria = criteria
        self.index = index

        self.predecessor_model_id = predecessor_model_id

        if self.parameters is None:
            self.parameters = {}
        if self.estimated_parameters is None:
            self.estimated_parameters = {}
        if self.criteria is None:
            self.criteria = {}

    def set_criterion(self, id: str, value: float) -> None:
        """Set a criterion value for the model.

        Args:
            id:
                The ID of the criterion (e.g. `'AIC'`).
            value:
                The criterion value for the (presumably calibrated) model.
        """
        if id in self.criteria:
            warnings.warn(
                'Overwriting saved criterion value. '
                f'Criterion: {id}. Value: {self.criteria[id]}.'
            )
        self.criteria[id] = value

    def has_criterion(self, id: str) -> bool:
        """Check whether the model provides a value for a criterion.

        Args:
            id:
                The ID of the criterion (e.g. `'AIC'`).
        """
        # TODO also `and self.criteria[id] is not None`?
        return id in self.criteria

    def get_criterion(self, id: str):
        """Get a criterion value for the model.

        Args:
            id:
                The ID of the criterion (e.g. `'AIC'`).
        """
        return self.criteria[id]

    @staticmethod
    def from_dict(
        model_dict: Dict[str, Any],
        base_path: TYPING_PATH = None,
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
    def from_yaml(model_yaml: TYPING_PATH) -> 'Model':
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

    def to_dict(self, resolve_paths=True) -> Dict[str, Any]:
        """Generate a dictionary from the attributes of a `Model` instance.

        Args:
            resolve_paths:
                Whether to resolve relative paths into absolute paths.

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
        return model_dict

    def to_yaml(self, petab_yaml: TYPING_PATH) -> None:
        """Generate a PEtab Select model YAML file from a `Model` instance.

        Parameters:
            petab_yaml:
                The location where the PEtab Select model YAML file will be
                saved.
        """
        # FIXME change `getattr(self, PETAB_YAML)` to be relative to
        # destination?
        # kind of fixed, as the path will be resolved in `to_dict`.
        with open(petab_yaml, 'w') as f:
            yaml.dump(self.to_dict(), f)
        #yaml.dump(self.to_dict(), str(petab_yaml))

    def to_petab(
        self,
        output_path: TYPING_PATH = None,
    ) -> Tuple[petab.Problem, TYPING_PATH]:
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
        petab_problem = petab.Problem.from_yaml(str(self.petab_yaml))
        for parameter_id, parameter_value in self.parameters.items():
            # If the parameter is to be estimated.
            if np.isnan(parameter_value):
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 1
            # Else the parameter is to be fixed.
            else:
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 0
                petab_problem.parameter_df.loc[parameter_id, NOMINAL_VALUE] = \
                    parameter_value

        petab_yaml_path = None
        if output_path is not None:
            petab_yaml_path = \
                petab_problem.to_files_generic(prefix_path=output_path)

        return (petab_problem, petab_yaml_path)

    def __hash__(self):
        """Get the problem-specific model hash.

        TODO untested, unused
        TODO store previously computed hash to avoid repeated computation?
        """
        return hash(tuple([
            method(attribute)
            for attribute, method in Model.hash_attributes.items()
        ]))

    def __str__(self):
        parameter_ids = '\t'.join(self.parameters.keys())
        parameter_values = '\t'.join(str(v) for v in self.parameters.values())
        header = f'{MODEL_ID}\t{PETAB_YAML}\t{parameter_ids}'
        data = f'{self.model_id}\t{self.petab_yaml}\t{parameter_values}'
        return f'{header}\n{data}'

    def get_mle(self) -> Dict[str, float]:
        """

        # Check if original PEtab problem or PEtab Select model has estimated
        # parameters. e.g. can use some of `self.to_petab` to get the parameter
        # df and see if any are estimated.
        if not self.has_estimated_parameters:
            warn('The MLE for this model contains no estimated parameters.')
        if not all([
            parameter_id in getattr(self, ESTIMATED_PARAMETERS)
            for parameter_id in self.get_estimated_parameter_ids()
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


def default_compare(
    model0: 'petab_select.Model',
    model: 'petab_select.Model',
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
    if criterion not in [AIC, AICC, BIC]:
        raise NotImplementedError(
            f'Unknown criterion: {criterion}.'
        )
    # For AIC, AICc, and BIC, lower criterion values are better.
    if not model.has_criterion(criterion):
        warnings.warn(
            f'Model "{model.model_id}" does not provide a value for the '
            f'criterion "{criterion}".'
        )
        return False
    return model.get_criterion(criterion) < model0.get_criterion(criterion)


def models_from_yaml_list(model_list_yaml: TYPING_PATH) -> List[Model]:
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
    return [
        Model.from_dict(model_dict, base_path=Path(model_list_yaml).parent)
        for model_dict in model_dict_list
    ]
