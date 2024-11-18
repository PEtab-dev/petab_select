import os
from pathlib import Path

import fides
import pypesto.engine
import pypesto.optimize
import pypesto.select

import petab_select

SKIP_TEST_CASES_WITH_PREEXISTING_EXPECTED_MODEL = False
os.environ["AMICI_EXPERIMENTAL_SBML_NONCONST_CLS"] = "1"

# Set to `[]` to test all
test_cases = [
    #'0004',
    #'0008',
]

# Do not use computationally-expensive test cases in CI
skip_test_cases = [
    "0009",
]

test_cases_path = Path(__file__).resolve().parent.parent.parent / "test_cases"

# Reduce runtime but with high reproducibility
minimize_options = {
    "n_starts": 24,
    "optimizer": pypesto.optimize.FidesOptimizer(
        verbose=0, hessian_update=fides.BFGS()
    ),
    "engine": pypesto.engine.MultiProcessEngine(),
    "filename": None,
    "progress_bar": False,
}


def objective_customizer(obj):
    # obj.amici_solver.setAbsoluteTolerance(1e-17)
    obj.amici_solver.setRelativeTolerance(1e-12)


# Indentation to match `test_pypesto.py`, to make it easier to keep files similar.
if True:
    for test_case_path in test_cases_path.glob("*"):
        if test_cases and test_case_path.stem not in test_cases:
            continue

        if test_case_path.stem in skip_test_cases:
            continue

        expected_model_yaml = test_case_path / "expected.yaml"

        if (
            SKIP_TEST_CASES_WITH_PREEXISTING_EXPECTED_MODEL
            and expected_model_yaml.is_file()
        ):
            # Skip test cases that already have an expected model.
            continue
        print(f"Running test case {test_case_path.stem}")

        # Setup the pyPESTO model selector instance.
        petab_select_problem = petab_select.Problem.from_yaml(
            test_case_path / "petab_select_problem.yaml",
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

        # Generate the expected model.
        best_model.to_yaml(
            expected_model_yaml, paths_relative_to=test_case_path
        )

        petab_select.model.models_to_yaml_list(
            models=pypesto_select_problem.calibrated_models.values(),
            output_yaml="all_models.yaml",
        )
