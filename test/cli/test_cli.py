import filecmp
from pathlib import Path

import pytest
from click.testing import CliRunner

# from petab_select import Model
import petab_select.cli

base_dir = Path(__file__).parent


@pytest.fixture
def output_path() -> Path:
    return base_dir / "output"


@pytest.fixture
def expected_output_path() -> Path:
    return base_dir / "expected_output"


@pytest.fixture
def model_yaml() -> Path:
    return base_dir / "input" / "model.yaml"


@pytest.fixture
def models_yaml() -> Path:
    return base_dir / "input" / "models.yaml"


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_model_to_petab(
    model_yaml,
    output_path,
    expected_output_path,
    cli_runner,
) -> None:
    """Test conversion of a model to PEtab problem files."""
    output_path_model = output_path / "model"
    output_path_model.mkdir(parents=True, exist_ok=True)

    result = cli_runner.invoke(
        petab_select.cli.model_to_petab,
        [
            "--model",
            model_yaml,
            "--output",
            output_path_model,
        ],
    )

    # The new PEtab problem YAML file is output to stdout correctly.
    assert (
        result.stdout == f'{base_dir / "output" / "model" / "problem.yaml"}\n'
    )

    comparison = filecmp.dircmp(
        expected_output_path / "model",
        output_path_model,
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


def test_models_to_petab(
    models_yaml,
    output_path,
    expected_output_path,
    cli_runner,
) -> None:
    """Test conversion of multiple models to PEtab problem files."""
    output_path_models = output_path / "models"
    output_path_models.mkdir(parents=True, exist_ok=True)

    result = cli_runner.invoke(
        petab_select.cli.models_to_petab,
        [
            "--models",
            models_yaml,
            "--output",
            output_path_models,
        ],
    )

    # The new PEtab problem YAML files are output with model IDs to `stdout`
    # correctly.
    assert result.stdout == (
        f'model_1\t{base_dir / "output" / "models" / "model_1" / "problem.yaml"}\n'
        f'model_2\t{base_dir / "output" / "models" / "model_2" / "problem.yaml"}\n'
    )

    comparison = filecmp.dircmp(
        expected_output_path / "models" / "model_1",
        output_path_models / "model_1",
    )
    # The first set of PEtab problem files are as expected.
    assert not comparison.diff_files
    assert sorted(comparison.same_files) == [
        "conditions.tsv",
        "measurements.tsv",
        "model.xml",
        "observables.tsv",
        "parameters.tsv",
        "problem.yaml",
    ]

    comparison = filecmp.dircmp(
        expected_output_path / "models" / "model_2",
        output_path_models / "model_2",
    )
    # The second set of PEtab problem files are as expected.
    assert not comparison.diff_files
    assert sorted(comparison.same_files) == [
        "conditions.tsv",
        "measurements.tsv",
        "model.xml",
        "observables.tsv",
        "parameters.tsv",
        "problem.yaml",
    ]

    comparison = filecmp.dircmp(
        output_path_models / "model_1",
        output_path_models / "model_2",
    )
    # The first and second set of PEtab problems only differ in their
    # parameters table and nowhere else.
    assert comparison.diff_files == ["parameters.tsv"]
    assert sorted(comparison.same_files) == [
        "conditions.tsv",
        "measurements.tsv",
        "model.xml",
        "observables.tsv",
        "problem.yaml",
    ]
