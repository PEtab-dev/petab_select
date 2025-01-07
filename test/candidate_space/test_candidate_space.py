from pathlib import Path

import pandas as pd
import pytest

# from petab_select.candidate_space import (
#    BackwardCandidateSpace,
#    BruteForceCandidateSpace,
#    ForwardCandidateSpace,
#    LateralCandidateSpace,
# )
from petab_select.constants import (
    ESTIMATE,
)
from petab_select.model_space import ModelSpace


@pytest.fixture
def ordered_model_parameterizations():
    good_models_ascending = [
        # forward
        "00000",
        "10000",
        "11000",
        "11100",
        "11110",
        # backward
        "01110",
        "01100",
        # forward
        "01101",
        "01111",
        # backward
        "00111",
        "00011",
    ]
    bad_models = [
        "01011",
        "11011",
    ]

    # All good models are unique
    assert len(set(good_models_ascending)) == len(good_models_ascending)
    # All bad models are unique
    assert len(set(bad_models)) == len(bad_models)
    # No models are defined as both bad and good.
    assert not set(good_models_ascending).intersection(bad_models)

    return good_models_ascending, bad_models


@pytest.fixture
def calibrated_model_space(ordered_model_parameterizations):
    good_models_ascending, bad_models = ordered_model_parameterizations

    # As good models are ordered ascending by "goodness", and criteria
    # decreases for better models, the criteria decreases as the index increases.
    good_model_criteria = {
        model: 100 - index for index, model in enumerate(good_models_ascending)
    }
    # All bad models are currently set to the same "bad" criterion value.
    bad_model_criteria = {model: 1000 for model in bad_models}

    model_criteria = {
        **good_model_criteria,
        **bad_model_criteria,
    }
    return model_criteria


@pytest.fixture
def model_space(calibrated_model_space) -> pd.DataFrame:
    data = {
        "model_subspace_id": [],
        "petab_yaml": [],
        "k1": [],
        "k2": [],
        "k3": [],
        "k4": [],
        "k5": [],
    }

    for model in calibrated_model_space:
        data["model_subspace_id"].append(f"model_subspace_{model}")
        data["petab_yaml"].append(
            Path(__file__).parent.parent.parent
            / "doc"
            / "examples"
            / "model_selection"
            / "petab_problem.yaml"
        )
        k1, k2, k3, k4, k5 = (
            "0" if parameter == "0" else ESTIMATE for parameter in model
        )
        data["k1"].append(k1)
        data["k2"].append(k2)
        data["k3"].append(k3)
        data["k4"].append(k4)
        data["k5"].append(k5)
    df = pd.DataFrame(data=data)
    model_space = ModelSpace.load(df)
    return model_space
