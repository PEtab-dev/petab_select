from pathlib import Path

import fides
from more_itertools import one
import numpy as np
import pandas as pd
import petab_select
from petab_select import Model
from petab_select.constants import (
    CRITERIA,
    ESTIMATED_PARAMETERS,
    MODEL,
)
import pypesto.engine
import pypesto.optimize
import pypesto.select
import pytest


# Set to `[]` to test all
test_cases = [
    #'0003',
    #'0008',
]


@pytest.fixture
def test_cases_path():
    return Path(__file__).parent.parent.parent / 'test_cases'


@pytest.fixture
def minimize_options():
    # Reduce runtime but with high reproducibility
    return {
        'n_starts': 10,
        'optimizer': pypesto.optimize.FidesOptimizer(),
        'engine': pypesto.engine.MultiProcessEngine(),
    }


def test_pypesto(test_cases_path, minimize_options):
    for test_case_path in test_cases_path.glob('*'):
        if test_cases and test_case_path.stem not in test_cases:
            continue
        # Setup the pyPESTO model selector instance.
        petab_select_problem = petab_select.Problem.from_yaml(
            test_case_path / 'petab_select_problem.yaml',
        )
        pypesto_select_problem = \
            pypesto.select.Problem(petab_select_problem=petab_select_problem)

        # Run the selection process until "exhausted".
        pypesto_select_problem.select_to_completion(
            minimize_options=minimize_options,
        )

        # Get the best model, load the expected model.
        models = pypesto_select_problem.history.values()
        best_model = petab_select_problem.get_best(models)
        expected_model = Model.from_yaml(test_case_path / 'expected.yaml')

        def get_series(model) -> pd.Series:
            return pd.Series(
                getattr(model, dict_attribute),
                dtype=np.float64,
            ).sort_index()
        # The estimated parameters and criteria values are as expected.
        for dict_attribute in [CRITERIA, ESTIMATED_PARAMETERS]:
            pd.testing.assert_series_equal(
                get_series(expected_model),
                get_series(best_model),
                atol=1e-2,
            )
