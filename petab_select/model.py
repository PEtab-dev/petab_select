"""The `Model` class."""
import abc
from pathlib import Path
from typing import Any, Dict, Union
import warnings
import yaml

import numpy as np

from .constants import (
    CRITERIA,
    MODEL_ID,
    PARAMETERS,
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
        saved_attributes:
            Attributes that will be saved to disk by the `Model.to_yaml`
            method.
    """
    saved_attributes = (
        #HASH,
        MODEL_ID,
        PETAB_YAML,
        PARAMETERS,
        CRITERIA,
    )
    converters_load = {
        MODEL_ID: lambda x: x,
        PETAB_YAML: lambda x: x,
        PARAMETERS: lambda x: (
            {
                id: (
                    float(ESTIMATE_SYMBOL_INTERNAL)
                    if value == ESTIMATE_SYMBOL_UI
                    else value
                )
                for id, value in x.items()
            }
        ),
        CRITERIA: lambda x: {} if not x else x,
    }
    converters_save = {
        MODEL_ID: lambda x: x,
        PETAB_YAML: lambda x: str(x),
        PARAMETERS: lambda x: (
            {
                id: (
                    ESTIMATE_SYMBOL_UI
                    if np.isnan(value)
                    else value
                )
                for id, value in x.items()
            }
        ),
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
        parameters: Dict[str, Union[int, float]] = None,
        criteria: Dict[str, float] = None,
        index: int = None,
    ):
        self.model_id = model_id
        self.petab_yaml = Path(petab_yaml)
        self.parameters = parameters
        self.criteria = criteria
        self.index = index

        if self.parameters is None:
            self.parameters = {}
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

    def get_criterion(self, id: str):
        """Get a criterion value for the model.

        Args:
            id:
                The ID of the criterion (e.g. `'AIC'`).
        """
        return self.criteria[id]

    @staticmethod
    def from_dict(model_dict: Dict[str, Any]) -> 'Model':
        """Generate a model from a dictionary of attributes.

        Args:
            model_dict:
                A dictionary of attributes. The keys are attribute
                names, the values are the corresponding attribute values for
                the model. Required attributes are the required arguments of
                the `Model.__init__` method.

        Returns:
            A model instance, initialized with the provided attributes.
        """
        unknown_attributes = set(model_dict).difference(Model.converters_load)
        warnings.warn(
            'Ignoring unknown attributes: ' +
            ', '.join(unknown_attributes)
        )
        model_dict = {
            attribute: Model.converters_load[attribute](value)
            for attribute, value in model_dict.items()
            if attribute in Model.converters_load
        }
        return Model(**model_dict)

    @staticmethod
    def from_yaml(petab_yaml: TYPING_PATH) -> 'Model':
        """Generate a model from a PEtab Select model YAML file.

        Args:
            petab_yaml:
                The path to the PEtab Select model YAML file.

        Returns:
            A model instance, initialized with the provided attributes.
        """
        with open(str(petab_yaml)) as f:
            model_dict = yaml.safe_load(f)
        # TODO check that the hash is reproducible
        return Model.from_dict(model_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Generate a dictionary from the attributes of a `Model` instance.

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
        return model_dict

    def to_yaml(self, petab_yaml: TYPING_PATH) -> None:
        """Generate a PEtab Select model YAML file from a `Model` instance.

        Parameters:
            petab_yaml:
                The location where the PEtab Select model YAML file will be
                saved.
        """
        yaml.dump(self.to_dict(), str(petab_yaml))

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
