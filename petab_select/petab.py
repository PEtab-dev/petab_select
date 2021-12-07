from pathlib import Path
from typing import List, Optional

import petab
from petab.C import (
    ESTIMATE,
    NOMINAL_VALUE,
)

from .constants import (
    PETAB_ESTIMATE_FALSE,
    TYPE_PATH,
    TYPE_PARAMETER_DICT,
)


class PetabMixin():
    """Useful things for classes that contain a PEtab problem.

    All attributes/methods are prefixed with `petab_`.
    """
    def __init__(
        self,
        petab_yaml: Optional[TYPE_PATH] = None,
        petab_problem: Optional[petab.Problem] = None,
        parameters_as_lists: bool = False,
    ):
        if petab_yaml is None and petab_problem is None:
            raise ValueError(
                'Please supply at least one of either the location of the '
                'PEtab problem YAML file, or an instance of the PEtab problem.'
            )
        self.petab_yaml = petab_yaml
        if self.petab_yaml is not None:
            self.petab_yaml = Path(self.petab_yaml)

        self.petab_problem = petab_problem
        if self.petab_problem is None:
            self.petab_problem = petab.Problem.from_yaml(str(petab_yaml))

        self.petab_parameters = {
            parameter_id: (
                row[NOMINAL_VALUE]
                if row[ESTIMATE] == PETAB_ESTIMATE_FALSE
                else ESTIMATE
            )
            for parameter_id, row in self.petab_problem.parameter_df.iterrows()
        }
        if parameters_as_lists:
            self.petab_parameters = {k: [v] for k, v in self.petab_parameters.items()}

    @property
    def petab_parameter_ids_estimated(self) -> List[str]:
        return [
            parameter_id
            for parameter_id, parameter_value in self.petab_parameters.items()
            if parameter_value == ESTIMATE
        ]

    @property
    def petab_parameter_ids_fixed(self) -> List[str]:
        estimated = self.petab_parameter_ids_estimated
        return [
            parameter_id
            for parameter_id in self.petab_parameters
            if parameter_id not in estimated
        ]

    @property
    def petab_parameters_singular(self) -> TYPE_PARAMETER_DICT:
        return {
            parameter_id: one(parameter_value)
            for parameter_id, parameter_value in self.petab_parameters
        }
