"""Helper methods for working with PEtab problems."""

from typing import Literal

import numpy as np
import petab.v1 as petab
from petab.v1.C import ESTIMATE

__all__ = ["get_petab_parameters"]


def get_petab_parameters(
    petab_problem: petab.Problem, as_lists: bool = False
) -> dict[str, float | Literal[ESTIMATE] | list[float | Literal[ESTIMATE]]]:
    """Convert PEtab problem parameters to the format in model space files.

    Args:
        petab_problem:
            The PEtab problem.
        as_lists:
            Each value will be provided inside a list object, similar to the
            format for multiple values for a parameter in a model subspace.

    Returns:
        Keys are parameter IDs, values are the nominal values for fixed
        parameters, or :const:`ESTIMATE` for estimated parameters.
    """
    values = np.array(petab_problem.x_nominal, dtype=object)
    values[petab_problem.x_free_indices] = ESTIMATE
    if as_lists:
        values = [[v] for v in values]
    return dict(zip(petab_problem.x_ids, values, strict=True))
