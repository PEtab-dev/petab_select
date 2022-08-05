from pathlib import Path

import numpy as np
import pandas as pd
import petab
import petab_select
import pytest
from petab_select import ESTIMATE, Method, Model, FamosCandidateSpace
from petab_select.model import default_compare
from petab_select.constants import Criterion


@pytest.fixture
def input_path():
    return Path(__file__).parent / "input"


@pytest.fixture
def petab_select_problem(input_path):
    return petab_select.Problem.from_yaml(
        input_path
        / "famos_synthetic_petab_problem"
        / "FAMoS_2019_petab_select_problem.yaml"
    )


@pytest.fixture
def predecessor_model(input_path):
    return Model(
        petab_yaml=(
            input_path
            / "famos_synthetic_petab_problem"
            / "FAMoS_2019_problem.yaml"
        ),
        model_subspace_id="model_subspace_1",
        model_id="M_1100110111000111",
        parameters={
            "mu_AB": "estimate",
            "mu_AC": 0,
            "mu_AD": "estimate",
            "mu_BA": "estimate",
            "mu_BC": 0,
            "mu_BD": 0,
            "mu_CA": "estimate",
            "mu_CB": 0,
            "mu_CD": "estimate",
            "mu_DA": "estimate",
            "mu_DB": "estimate",
            "mu_DC": "estimate",
            "ro_A": "estimate",
            "ro_B": "estimate",
            "ro_C": 0,
            "ro_D": 0,
        },
        estimated_parameters={
            "mu_AB": 0.09706971737957297,
            "mu_AD": -0.6055359156893474,
            "mu_BA": 0.6989700040781575,
            "mu_CA": -13.545121478780585,
            "mu_CD": -13.955162965672203,
            "mu_DA": -13.405909047226377,
            "mu_DB": -13.402598631022197,
            "mu_DC": -1.1619119214640863,
            "ro_A": -1.6431508614147425,
            "ro_B": 2.9912966824709097,
        },
        criteria={
            Criterion.AIC: 30330.782621349786,
            Criterion.AICC: 30332.80096997364,
            Criterion.BIC: 30358.657538777607,
            Criterion.NLLH: 15155.391310674893,
        },
    )


@pytest.fixture
def expected_criterion_values(input_path):
    calibration_results = pd.read_csv(
        input_path / "calibration_results.tsv",
        sep="\t",
    )
    expected_aicc = {}
    for index, row in calibration_results.iterrows():
        expected_aicc[row["model_id"]] = row["AICc"]
    return expected_aicc
    

@pytest.fixture
def expected_progress_list():
    return [
        (Method.LATERAL, set()),
        (Method.LATERAL, {4, 15}),
        (Method.LATERAL, {9, 13}),
        (Method.FORWARD, set()),
        (Method.FORWARD, {3}),
        (Method.FORWARD, {11}),
        (Method.BACKWARD, set()),
        (Method.BACKWARD, {6}),
        (Method.BACKWARD, {10}),
        (Method.BACKWARD, {8}),
        (Method.BACKWARD, {14}),
        (Method.BACKWARD, {1}),
        (Method.BACKWARD, {16}),
        (Method.BACKWARD, {4}),
        (Method.FORWARD, set()),
        (Method.LATERAL, set()),
        "M_0001011010010010",
        (Method.LATERAL, set()),
        (Method.LATERAL, {16, 7}),
        (Method.LATERAL, {5, 12}),
        (Method.LATERAL, {13, 15}),
        (Method.LATERAL, {1, 6}),
        (Method.FORWARD, set()),
        (Method.FORWARD, {3}),
        (Method.FORWARD, {7}),
        (Method.FORWARD, {2}),
        (Method.FORWARD, {11}),
        (Method.BACKWARD, set()),
        (Method.BACKWARD, {7}),
        (Method.BACKWARD, {16}),
        (Method.BACKWARD, {4}),
        (Method.FORWARD, set()),
        (Method.LATERAL, set()),
        (Method.LATERAL, {9, 15}),
        (Method.FORWARD, set()),
        (Method.BACKWARD, set()),
        (Method.LATERAL, set()),
    ]


def test_famos(
    petab_select_problem,
    predecessor_model,
    expected_criterion_values,
    expected_progress_list,
):
    def set_model_id(model):
        model.model_id = ("M_" + ''.join(
            '1' if v == ESTIMATE else '0'
            for v in model.parameters.values()
        ))

    def calibrate(
        model,
        expected_criterion_values=expected_criterion_values,
    ) -> None:
        model.set_criterion(
            criterion=petab_select_problem.criterion,
            value=expected_criterion_values[model.model_id]
        )

    history = {}
    progress_list = []

    lateral_method_switching = {
        (Method.BACKWARD, Method.FORWARD): Method.LATERAL,
        (Method.FORWARD, Method.BACKWARD): Method.LATERAL,
        (Method.BACKWARD, Method.LATERAL): None,
        (Method.FORWARD, Method.LATERAL): None,
        (Method.FORWARD,): Method.BACKWARD,
        (Method.BACKWARD,): Method.FORWARD,
        (Method.LATERAL,): Method.FORWARD,
        None: Method.LATERAL,
    }

    candidate_space = FamosCandidateSpace(
        method_switching=lateral_method_switching,
        predecessor_model=predecessor_model,
        critical_parameter_sets=[],
        swap_parameter_sets=[
            ["ro_A", "mu_BA", "mu_CA", "mu_DA"],
            ["ro_B", "mu_AB", "mu_CB", "mu_DB"],
            ["ro_C", "mu_AC", "mu_BC", "mu_DC"],
            ["ro_D", "mu_AD", "mu_BD", "mu_CD"],
        ],
        number_of_reattempts=1,
        swap_only_once=False,
    )

    try:
        while True:
            # Calibrated models in this iteration that improve on the predecessor
            # model.
            better_models = []
            # All calibrated models in this iteration (see second return value).
            local_history = {}

            previous_predecessor_model_parameters = (
                candidate_space.predecessor_model.get_parameter_values(
                    parameter_ids=predecessor_model.petab_parameters
                )
            )
            previous_predecessor_model_parameter_indices = [
                index + 1
                for index in range(len(previous_predecessor_model_parameters))
                if previous_predecessor_model_parameters[index] == "estimate"
            ]
            predecessor_model_parameters = predecessor_model.get_parameter_values(
                parameter_ids=predecessor_model.petab_parameters
            )
            predecessor_model_parameter_indices = [
                index + 1
                for index in range(len(predecessor_model_parameters))
                if predecessor_model_parameters[index] == "estimate"
            ]

            candidate_models = petab_select.ui.candidates(
                problem=petab_select_problem,
                candidate_space=candidate_space,
                excluded_model_hashes=list(history),
                predecessor_model=predecessor_model,
            ).models
            progress_list.append(
                (
                    candidate_space.inner_candidate_space.method,
                    set(
                        previous_predecessor_model_parameter_indices
                    ).symmetric_difference(set(predecessor_model_parameter_indices)),
                )
            )

            if not candidate_models:
                raise StopIteration("No valid models found.")

            for candidate_model in candidate_models:
                # set model_id to M_010101010101010 form
                set_model_id(candidate_model)
                # run calibration
                calibrate(candidate_model)

                local_history[candidate_model.model_id] = candidate_model
                # if candidate model has better criteria, add to better models
                if default_compare(
                    predecessor_model, candidate_model, petab_select_problem.criterion
                ):
                    better_models.append(candidate_model)

            history.update(local_history)
            best_model = None
            if better_models:
                best_model = petab_select.ui.best(
                    problem=petab_select_problem,
                    models=better_models,
                    criterion=petab_select_problem.criterion,
                )
            jumped_to_most_distant = candidate_space.update_after_calibration(
                history=history,
                local_history=local_history,
                criterion=petab_select_problem.criterion,
            )
            # If candidate space not Famos then ignored.
            # Else, in case we jumped to most distant in this iteration, update the
            # best_model to the predecessor_model (jumped to model) so it becomes
            # the predecessor model in next iteration.
            # Also update the local_history with it.
            if jumped_to_most_distant:
                best_model = candidate_space.predecessor_model
                set_model_id(best_model)
                calibrate(best_model)
                local_history[best_model.model_id] = best_model
                progress_list.append(best_model.model_id)
            if best_model:
                predecessor_model = best_model

    except StopIteration:
        assert progress_list == expected_progress_list
