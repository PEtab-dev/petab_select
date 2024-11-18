from pathlib import Path

import yaml

import petab_select
from petab_select import (
    MODEL_HASH,
    MODEL_ID,
    MODEL_SUBSPACE_ID,
    MODEL_SUBSPACE_INDICES,
    PREDECESSOR_MODEL_HASH,
)

test_cases_path = Path(__file__).resolve().parent.parent.parent / "test_cases"


for test_case_path in test_cases_path.glob("*"):
    petab_select_problem = petab_select.Problem.from_yaml(
        test_case_path / "petab_select_problem.yaml",
    )
    expected_model_yaml = test_case_path / "expected.yaml"

    with open(expected_model_yaml) as f:
        model_dict = yaml.safe_load(f)

    model = petab_select_problem.model_space.model_subspaces[
        model_dict[MODEL_SUBSPACE_ID]
    ].indices_to_model(model_dict[MODEL_SUBSPACE_INDICES])
    model_dict[MODEL_ID] = str(model.model_id)
    model_dict[MODEL_HASH] = str(model.get_hash())
    model_dict[PREDECESSOR_MODEL_HASH] = None

    with open(expected_model_yaml, "w") as f:
        yaml.safe_dump(model_dict, f)
