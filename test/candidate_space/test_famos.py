from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import petab
import pytest
from more_itertools import one

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
        (Method.MOST_DISTANT, {2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 15}),
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
    ) -> None:
        model.set_criterion(
            criterion=petab_select_problem.criterion,
            value=expected_criterion_values[model.model_id],
        )

    def parse_summary_to_progress_list(summary_tsv: str) -> Tuple[Method, set]:
        """Get progress information from the summary file."""
        df_raw = pd.read_csv(summary_tsv, sep='\t')
        df = df_raw.loc[~pd.isnull(df_raw["predecessor change"])]

        parameter_list = list(
            petab_select_problem.model_space.model_subspaces[
                'model_subspace_1'
            ].parameters
        )

        progress_list = []

        for index, (_, row) in enumerate(df.iterrows()):
            method = Method(row["method"])

            model = {
                1 + parameter_list.index(parameter_id)
                for parameter_id in eval(row["current model"])
            }
            if index == 0:
                model0 = model

            difference = model.symmetric_difference(model0)
            progress_list.append((method, difference))
            model0 = model

        return progress_list

    progress_list = []
    calibrated_models = {}
    newly_calibrated_models = {}

    candidate_space = petab_select_problem.new_candidate_space()
    candidate_space.summary_tsv.unlink(missing_ok=True)
    candidate_space._setup_summary_tsv()

    with pytest.raises(StopIteration, match="No valid models found."):
        predecessor_model = candidate_space.predecessor_model
        while True:
            # Save predecessor_models and find new candidates
            if candidate_space.predecessor_model is not None:
                previous_predecessor_model = candidate_space.predecessor_model
            else:
                previous_predecessor_model = one(candidate_space.models)
            candidate_space = petab_select.ui.candidates(
                problem=petab_select_problem,
                candidate_space=candidate_space,
                calibrated_models=calibrated_models,
                newly_calibrated_models=newly_calibrated_models,
                previous_predecessor_model=previous_predecessor_model,
            )

            # Calibrate candidate models
            newly_calibrated_models = {}
            for candidate_model in candidate_space.models:
                set_model_id(candidate_model)
                calibrate(candidate_model)
                newly_calibrated_models[
                    candidate_model.get_hash()
                ] = candidate_model
                calibrated_models.update(newly_calibrated_models)

            # Stop iteration if there are no candidate models
            if not candidate_space.models:
                raise StopIteration("No valid models found.")

    progress_list = parse_summary_to_progress_list(candidate_space.summary_tsv)

    assert progress_list == expected_progress_list, progress_list
