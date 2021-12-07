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


SKIP_TEST_CASES_WITH_PREEXISTING_EXPECTED_MODEL = False

test_cases_path = Path(__file__).resolve().parent.parent.parent / 'test_cases'

# Reduce runtime but with high reproducibility
minimize_options = {
    'n_starts': 100,
    'optimizer': pypesto.optimize.FidesOptimizer(),
    'engine': pypesto.engine.MultiProcessEngine(),
}

for test_case_path in test_cases_path.glob('*'):
    #if test_case_path.stem not in ['0007']:
    #    continue
    # FIXME remove above
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
    selected_model = None
    while True:
        try:
            _selected, _local, selection_history = \
                selector.select(
                    initial_model=selected_model,
                    minimize_options=minimize_options,
                )
            selected_model = one(_selected)
            if False:
                from pprint import pprint
                md = {
                    model['model'].model_subspace_id: {
                        'MLE': model['MLE'],
                        'model': model['model'],
                        'aic': model['model'].get_criterion('AIC'),
                        'predecessor_model_id': model['model'].predecessor_model_id,
                    }
                    for _, model in selection_history.items()
                }
                pprint(md)
                breakpoint()
        except StopIteration:
            break

    # Get the best model.
    models = [
        result[MODEL]
        for result in selection_history.values()
    ]
    best_model = petab_select_problem.get_best(models)

    # for debugging
    if False:
        from pprint import pprint
        md = {
            model['model'].model_subspace_id: {
                'MLE': model['MLE'],
                'model': model['model'],
                'aic': model['model'].get_criterion('AIC'),
                'predecessor_model_id': model['model'].predecessor_model_id,
            }
            for _, model in selection_history.items()
        }
        pprint(md)
        breakpoint()


    # Generate the expected model.
    best_model.to_yaml(expected_model_yaml, paths_relative_to=test_case_path)
