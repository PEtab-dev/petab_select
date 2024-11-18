from pathlib import Path

import pandas as pd
from more_itertools import one

import petab_select
from petab_select import MODEL_HASH, Criterion, Method, Model

input_path = (
    Path(__file__).resolve().parent.parent.parent
    / "test"
    / "candidate_space"
    / "input"
    / "famos_synthetic"
)
petab_select_problem_yaml = str(
    input_path / "select" / "FAMoS_2019_petab_select_problem.yaml"
)
expected_criterion_values = dict(
    pd.read_csv(
        input_path / "test_files" / "calibration_results.tsv", sep="\t"
    ).set_index(MODEL_HASH)[Criterion.AICC]
)

petab_select_problem = petab_select.Problem.from_yaml(
    petab_select_problem_yaml
)
criterion = petab_select_problem.criterion


def calibrate(
    model: Model,
    criterion=criterion,
    expected_criterion_values=expected_criterion_values,
) -> None:
    """Set the criterion value for a model."""
    model.set_criterion(
        criterion=criterion,
        value=float(expected_criterion_values[model.get_hash()]),
    )


def parse_summary_to_progress_list(
    summary_tsv: str,
) -> list[tuple[Method, set]]:
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
