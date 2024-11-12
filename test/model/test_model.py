import filecmp
from pathlib import Path

import pytest

from petab_select import Model

base_dir = Path(__file__).parent


@pytest.fixture
def output_path() -> Path:
    return base_dir / "output"


@pytest.fixture
def expected_output_path() -> Path:
    return base_dir / "expected_output"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(base_dir / "input" / "model.yaml")


def test_model_to_petab(model, output_path, expected_output_path) -> None:
    """Test conversion of a model to a PEtab problem and files."""
    output_path_petab = output_path / "petab"
    output_path_petab.mkdir(parents=True, exist_ok=True)
    # TODO test `petab_problem`? Shouldn't be necessary since the generated
    # files are tested below.
    petab_problem, petab_problem_yaml = model.to_petab(
        output_path=output_path_petab
    )

    comparison = filecmp.dircmp(
        expected_output_path / "petab",
        output_path_petab,
    )
    # The PEtab problem files are as expected.
    assert not comparison.diff_files
    assert sorted(comparison.same_files) == [
        "conditions.tsv",
        "measurements.tsv",
        "model.xml",
        "observables.tsv",
        "parameters.tsv",
        "problem.yaml",
    ]
