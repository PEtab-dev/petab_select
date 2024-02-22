import os
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

os.environ["AMICI_EXPERIMENTAL_SBML_NONCONST_CLS"] = "1"

# Set to `[]` to test all
test_cases = [
    #'0004',
    #'0008',
]

# Do not use computationally-expensive test cases in CI
skip_test_cases = [
    '0009',
]

test_cases_path = Path(__file__).resolve().parent.parent.parent / 'test_cases'

# Reduce runtime but with high reproducibility
minimize_options = {
    'n_starts': 10,
    'optimizer': pypesto.optimize.FidesOptimizer(
        verbose=0, hessian_update=fides.BFGS()
    ),
    'engine': pypesto.engine.MultiProcessEngine(),
    'filename': None,
    'progress_bar': False,
}


def objective_customizer(obj):
    # obj.amici_solver.setAbsoluteTolerance(1e-17)
    obj.amici_solver.setRelativeTolerance(1e-12)


def test_pypesto():
    for test_case_path in test_cases_path.glob('*'):
        if test_cases and test_case_path.stem not in test_cases:
            continue

        if test_case_path.stem in skip_test_cases:
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
            objective_customizer=objective_customizer,
        )

        # Get the best model
        best_model = petab_select_problem.get_best(
            models=pypesto_select_problem.calibrated_models.values(),
        )

        # Load the expected model.
        expected_model = Model.from_yaml(expected_model_yaml)

        def get_series(model, dict_attribute) -> pd.Series:
            return pd.Series(
                getattr(model, dict_attribute),
                dtype=np.float64,
            ).sort_index()

        # The estimated parameters and criteria values are as expected.
        for dict_attribute in [CRITERIA, ESTIMATED_PARAMETERS]:
            pd.testing.assert_series_equal(
                get_series(expected_model, dict_attribute),
                get_series(best_model, dict_attribute),
                atol=1e-2,
            )
