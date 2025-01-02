from pathlib import Path

import petab_select

problem_yaml = (
    Path(__file__).parents[2]
    / "doc"
    / "examples"
    / "model_selection"
    / "petab_select_problem.yaml"
)


def test_round_trip():
    """Test storing/loading of a full problem."""
    problem0 = petab_select.Problem.from_yaml(problem_yaml)
    problem0.save("output")

    with open(
        Path(__file__) / "expected_output/petab_select_problem.yaml"
    ) as f:
        problem_yaml0 = f.read()
    with open(Path(__file__) / "expected_output/model_space.tsv") as f:
        model_space_tsv0 = f.read()

    with open(Path(__file__) / "output/petab_select_problem.yaml") as f:
        problem_yaml1 = f.read()
    with open(Path(__file__) / "output/model_space.tsv") as f:
        model_space_tsv1 = f.read()

    # The the exported problem YAML is as expected, with updated relative paths.
    assert problem_yaml1 == problem_yaml0
    # The the exported model space TSV is as expected, with updated relative paths.
    assert model_space_tsv1 == model_space_tsv0
