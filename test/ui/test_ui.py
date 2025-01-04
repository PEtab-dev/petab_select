from pathlib import Path

import pytest
from more_itertools import one

import petab_select
from petab_select import Models
from petab_select.constants import (
    CANDIDATE_SPACE,
    MODELS,
    UNCALIBRATED_MODELS,
)


@pytest.fixture
def petab_select_problem():
    return petab_select.Problem.from_yaml(
        Path(__file__).parents[2]
        / "doc"
        / "examples"
        / "model_selection"
        / "petab_select_problem.yaml"
    )


def test_user_calibrated_models(petab_select_problem):
    """Test handling of user-calibrated models."""
    model_M1_2 = petab_select_problem.model_space.model_subspaces[
        "M1_2"
    ].indices_to_model((0, 0, 0))
    model_M1_2.set_criterion(
        criterion=petab_select_problem.criterion, value=12.3
    )
    user_calibrated_models = Models([model_M1_2])

    # Initial iteration: expect the "empty" model. Set dummy criterion and continue.
    iteration = petab_select.ui.start_iteration(
        problem=petab_select_problem,
        user_calibrated_models=user_calibrated_models,
    )
    model_M1_0 = one(iteration[UNCALIBRATED_MODELS])
    # The initial iteration proceeded as expected: the "empty" model was identified as a candidate.
    assert model_M1_0.model_subspace_id == "M1_0"
    model_M1_0.set_criterion(petab_select_problem.criterion, 100)
    iteration_results = petab_select.ui.end_iteration(
        problem=petab_select_problem,
        candidate_space=iteration[CANDIDATE_SPACE],
        calibrated_models=[model_M1_0],
    )

    # Second iteration. User calibrated models should now change behavior.
    iteration = petab_select.ui.start_iteration(
        problem=petab_select_problem,
        candidate_space=iteration_results[CANDIDATE_SPACE],
        user_calibrated_models=user_calibrated_models,
    )
    # The user calibrated model was not included in the iteration's uncalibrated models.
    uncalibrated_model_ids = [
        model.model_subspace_id for model in iteration[UNCALIBRATED_MODELS]
    ]
    assert set(uncalibrated_model_ids) == {"M1_1", "M1_3"}
    for uncalibrated_model in iteration[UNCALIBRATED_MODELS]:
        uncalibrated_model.set_criterion(petab_select_problem.criterion, 50)
    iteration_results = petab_select.ui.end_iteration(
        problem=petab_select_problem,
        candidate_space=iteration[CANDIDATE_SPACE],
        calibrated_models=iteration[UNCALIBRATED_MODELS],
    )
    iteration_model_ids = [
        model.model_subspace_id for model in iteration_results[MODELS]
    ]
    # The user-calibrated model is included in the final set of models for this iteration
    assert set(iteration_model_ids) == {"M1_1", "M1_2", "M1_3"}
