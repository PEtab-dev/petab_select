# TODO set `petab_yaml` path in 'expected.yaml'to be relative to
#      `test_case_path`
# TODO much of this is duplicated in `test_pypesto.py`.
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


SKIP_TEST_CASES_WITH_PREEXISTING_EXPECTED_MODEL = True

test_cases_path = Path() / '..' / '..' / 'test_cases'

# Reduce runtime but with high reproducibility
minimize_options = {
    'n_starts': 100,
    'optimizer': pypesto.optimize.FidesOptimizer(),
    'engine': pypesto.engine.MultiProcessEngine(),
}

for test_case_path in test_cases_path.glob('*'):
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
        test_case_path / 'selection_problem.yaml',
    )
    selector = pypesto.select.ModelSelector(problem=petab_select_problem)
    
    # Run the selection process until "exhausted".
    while True:
        try:
            _, _, selection_history = \
                selector.select(minimize_options=minimize_options)
        except StopIteration:
            break

    # Get the best model.
    models = [
        result[MODEL]
        for result in selection_history.values()
    ]
    best_model = petab_select_problem.get_best(models)

    # Generate the expected model.
    best_model.to_yaml(expected_model_yaml)
