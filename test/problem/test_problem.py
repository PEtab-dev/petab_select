from pathlib import Path

import petab_select

test_path = Path(__file__).parent

problem_yaml = (
    test_path.parent.parent
    / "doc"
    / "examples"
    / "model_selection"
    / "petab_select_problem.yaml"
)


def test_round_trip():
    """Test storing/loading of a full problem."""
    problem0 = petab_select.Problem.from_yaml(problem_yaml)
    problem0.save(test_path / "output")

    with open(test_path / "expected_output/petab_select_problem.yaml") as f:
        problem_yaml0 = f.read()
    with open(test_path / "expected_output/model_space.tsv") as f:
        model_space_tsv0 = f.read()

    with open(test_path / "output/petab_select_problem.yaml") as f:
        problem_yaml1 = f.read()
    with open(test_path / "output/model_space.tsv") as f:
        model_space_tsv1 = f.read()

    # The exported problem YAML is as expected, with updated relative paths.
    assert problem_yaml1 == problem_yaml0
    # The exported model space TSV is as expected, with updated relative paths.
    assert model_space_tsv1 == model_space_tsv0
