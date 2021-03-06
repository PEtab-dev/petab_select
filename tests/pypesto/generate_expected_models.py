from pathlib import Path

import fides
from more_itertools import one
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


SKIP_TEST_CASES_WITH_PREEXISTING_EXPECTED_MODEL = False

# Set to `[]` to test all
test_cases = [
    #'0004',
    #'0008',
]

test_cases_path = Path(__file__).resolve().parent.parent.parent / 'test_cases'

# Reduce runtime but with high reproducibility
minimize_options = {
    'n_starts': 100,
    'optimizer': pypesto.optimize.FidesOptimizer(),
    'engine': pypesto.engine.MultiProcessEngine(),
}

# Indentation to match `test_pypesto.py`, to make it easier to keep files similar.
if True:
    for test_case_path in test_cases_path.glob('*'):
        if test_cases and test_case_path.stem not in test_cases:
            continue

        expected_model_yaml = test_case_path / 'expected.yaml'

        if (
            SKIP_TEST_CASES_WITH_PREEXISTING_EXPECTED_MODEL
            and expected_model_yaml.is_file()
        ):
            # Skip test cases that already have an expected model.
            continue
        print(f'Running test case {test_case_path.stem}')

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

        # Generate the expected model.
        best_model.to_yaml(expected_model_yaml, paths_relative_to=test_case_path)
