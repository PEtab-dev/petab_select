from pathlib import Path
from typing import Tuple

import pandas as pd

import petab_select
from petab_select import ESTIMATE, Method, Model

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
    ).set_index('model_id')['AICc']
)

petab_select_problem = petab_select.Problem.from_yaml(
    petab_select_problem_yaml
)
criterion = petab_select_problem.criterion


def set_model_id(model: Model) -> None:
    """Set the model ID to a binary string for easier analysis."""
    model.model_id = "M_" + ''.join(
        str(v) for v in model.model_subspace_indices
    )


def calibrate(
    model: Model,
    criterion=criterion,
    expected_criterion_values=expected_criterion_values,
) -> None:
    """Set the criterion value for a model."""
    model.set_criterion(
        criterion=criterion,
        value=float(expected_criterion_values[model.model_id]),
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
