from pathlib import Path

import numpy as np
import pandas as pd
import petab
import pytest

import petab_select
from petab_select import ESTIMATE, FamosCandidateSpace, Method, Model
from petab_select.constants import Criterion
from petab_select.model import default_compare


@pytest.fixture
def input_path():
    return Path(__file__).parent / "input" / "famos_synthetic"


@pytest.fixture
def petab_select_problem(input_path):
    return petab_select.Problem.from_yaml(
        input_path / "select" / "FAMoS_2019_petab_select_problem.yaml"
    )


@pytest.fixture
def expected_criterion_values(input_path):
    calibration_results = pd.read_csv(
        input_path / "test_files" / "calibration_results.tsv",
        sep="\t",
    ).set_index('model_id')
    return dict(calibration_results['AICc'])


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
    expected_criterion_values,
    expected_progress_list,
):
    def set_model_id(model):
        model.model_id = "M_" + ''.join(
            '1' if v == ESTIMATE else '0' for v in model.parameters.values()
        )

    def calibrate(
        model,
        expected_criterion_values=expected_criterion_values,
        history=None,
    ) -> None:
        model.set_criterion(
            criterion=petab_select_problem.criterion,
            value=expected_criterion_values[model.model_id],
        )

    history = {}
    progress_list = []


    candidate_space = petab_select_problem.new_candidate_space()

    try:
        predecessor_model = candidate_space.predecessor_model
        while True:
            # Save predecessor_models and find new candidates
            previous_predecessor_model = candidate_space.predecessor_model
            candidate_models, history, _ = petab_select.ui.candidates(
                problem=petab_select_problem,
                candidate_space=candidate_space,
                excluded_model_hashes=history,
                previous_predecessor_model=predecessor_model,
                history=history,
            )
            predecessor_model = candidate_space.predecessor_model

            # Prepare indicies to write to progress_list
            previous_predecessor_model_parameters = (
                previous_predecessor_model.get_parameter_values(
                    parameter_ids=predecessor_model.petab_parameters
                )
            )
            previous_predecessor_model_parameter_indices = [
                index + 1
                for index in range(len(previous_predecessor_model_parameters))
                if previous_predecessor_model_parameters[index] == "estimate"
            ]
            predecessor_model_parameters = (
                predecessor_model.get_parameter_values(
                    parameter_ids=predecessor_model.petab_parameters
                )
            )
            predecessor_model_parameter_indices = [
                index + 1
                for index in range(len(predecessor_model_parameters))
                if predecessor_model_parameters[index] == "estimate"
            ]

            # Calibrate candidate_models
            for candidate_model in candidate_models:
                # set model_id to M_010101010101010 form
                set_model_id(candidate_model)
                # run calibration
                calibrate(candidate_model, history=history)
            # Write the progress_list for this step
            if not candidate_space.jumped_to_most_distant:
                progress_list.append(
                    (
                        candidate_space.inner_candidate_space.method,
                        set(
                            previous_predecessor_model_parameter_indices
                        ).symmetric_difference(
                            set(predecessor_model_parameter_indices)
                        ),
                    )
                )
            else:
                progress_list.append(predecessor_model.model_id)
            # Stop iteration if there are no candidate_models
            if not candidate_models:
                raise StopIteration("No valid models found.")

    except StopIteration:
        assert progress_list == expected_progress_list
