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
import pytest


# Set to `[]` to test all
test_cases = [
    #'0007',
    #'0008',
]


@pytest.fixture
def test_cases_path():
    return Path() / '..' / '..' / 'test_cases'

@pytest.fixture
def minimize_options():
    # Reduce runtime but with high reproducibility
    return {
        'n_starts': 100,
        'optimizer': pypesto.optimize.FidesOptimizer(),
        'engine': pypesto.engine.MultiProcessEngine(),
    }

def test_pypesto(test_cases_path, minimize_options):
    for test_case_path in test_cases_path.glob('*'):
        if test_cases and test_case_path.stem not in test_cases:
            continue
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
    
        # Get the best model, load the expected model.
        models = [
            result[MODEL]
            for result in selection_history.values()
        ]
        best_model = petab_select_problem.get_best(models)
        expected_model = Model.from_yaml(test_case_path / 'expected.yaml')
    
        # The estimated parameters and criteria values are as expected.
        for dict_attribute in [CRITERIA, ESTIMATED_PARAMETERS]:
            pd.testing.assert_series_equal(
                pd.Series(getattr(expected_model, dict_attribute)),
                pd.Series(getattr(best_model, dict_attribute)),
                atol=1e-3,
            )
