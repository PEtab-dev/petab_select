from pathlib import Path

import fides
import numpy as np
import pandas as pd
import pypesto.engine
import pypesto.optimize
import pypesto.select
import pytest
from more_itertools import one

import petab_select
from petab_select import Model
from petab_select.constants import CRITERIA, ESTIMATED_PARAMETERS, MODEL

# Set to `[]` to test all
test_cases = [
    #'0004',
    #'0008',
]

test_cases_path = Path(__file__).resolve().parent.parent.parent / 'test_cases'

# Reduce runtime but with high reproducibility
minimize_options = {
    'n_starts': 10,
    'optimizer': pypesto.optimize.FidesOptimizer(verbose=0),
    'engine': pypesto.engine.MultiProcessEngine(),
    'filename': None,
}


def test_pypesto():
    for test_case_path in test_cases_path.glob('*'):
        if test_cases and test_case_path.stem not in test_cases:
            continue

        expected_model_yaml = test_case_path / 'expected.yaml'
        # Setup the pyPESTO model selector instance.
        petab_select_problem = petab_select.Problem.from_yaml(
            test_case_path / 'petab_select_problem.yaml',
        )
        pypesto_select_problem = pypesto.select.Problem(
            petab_select_problem=petab_select_problem
        )

        # Run the selection process until "exhausted".
        pypesto_select_problem.select_to_completion(
            minimize_options=minimize_options,
        )

        # Get the best model, load the expected model.
        best_model = petab_select_problem.get_best(
            models=pypesto_select_problem.calibrated_models.values(),
        )
        expected_model = Model.from_yaml(expected_model_yaml)

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
