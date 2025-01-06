from pathlib import Path

import pandas as pd
import pytest
from more_itertools import one

import petab_select
from petab_select import Method, ModelHash
from petab_select.constants import (
    CANDIDATE_SPACE,
    MODEL_HASH,
    TERMINATE,
    UNCALIBRATED_MODELS,
    Criterion,
)


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
    ).set_index(MODEL_HASH)
    return {
        ModelHash.model_validate(k): v
        for k, v in calibration_results[Criterion.AICC].items()
    }


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
    def calibrate(
        model,
        expected_criterion_values=expected_criterion_values,
    ) -> None:
        model.set_criterion(
            criterion=petab_select_problem.criterion,
            value=expected_criterion_values[model.hash],
        )

    def parse_summary_to_progress_list(summary_tsv: str) -> tuple[Method, set]:
        """Get progress information from the summary file."""
        df_raw = pd.read_csv(summary_tsv, sep="\t")
        df = df_raw.loc[~pd.isnull(df_raw["predecessor change"])]

        parameter_list = list(
            one(
                petab_select_problem.model_space.model_subspaces.values()
            ).parameters
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

    candidate_space = petab_select_problem.new_candidate_space()
    expected_repeated_model_hash0 = candidate_space.predecessor_model.hash
    candidate_space.summary_tsv.unlink(missing_ok=True)
    candidate_space._setup_summary_tsv()

    with (
        pytest.raises(StopIteration, match="No valid models found."),
        pytest.warns(RuntimeWarning) as warning_record,
    ):
        while True:
            # Initialize iteration
            iteration = petab_select.ui.start_iteration(
                problem=petab_select_problem,
                candidate_space=candidate_space,
            )

            # Calibrate candidate models
            for candidate_model in iteration[UNCALIBRATED_MODELS]:
                calibrate(candidate_model)

            # Finalize iteration
            iteration_results = petab_select.ui.end_iteration(
                problem=petab_select_problem,
                candidate_space=iteration[CANDIDATE_SPACE],
                calibrated_models=iteration[UNCALIBRATED_MODELS],
            )
            candidate_space = iteration_results[CANDIDATE_SPACE]

            # Stop iteration if there are no candidate models
            if iteration_results[TERMINATE]:
                raise StopIteration("No valid models found.")

    # A model is encountered twice and therefore skipped.
    expected_repeated_model_hash1 = petab_select_problem.get_model(
        model_subspace_id=one(
            petab_select_problem.model_space.model_subspaces
        ),
        model_subspace_indices=[int(s) for s in "0001011010010010"],
    ).hash
    # The predecessor model is also re-encountered.
    assert len(warning_record) == 2
    assert (
        str(expected_repeated_model_hash0) in warning_record[0].message.args[0]
    )
    assert (
        str(expected_repeated_model_hash1) in warning_record[1].message.args[0]
    )

    progress_list = parse_summary_to_progress_list(candidate_space.summary_tsv)

    assert progress_list == expected_progress_list, progress_list
