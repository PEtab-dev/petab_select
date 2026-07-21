"""Tests for metaparameters."""

from pathlib import Path

import pytest

from petab_select import Problem
from petab_select.constants import ESTIMATE, Criterion
from petab_select.model import Model

PETAB_YAML = (
    Path(__file__).parent.parent
    / "test_data"
    / "ode_timeseries"
    / "petab_problem.yaml"
)

METAPARAMETERS = {"mp1": ["k1", "k2"]}


def write_problem(directory: Path, model_space_row: str) -> Problem:
    """Write and load a PEtab Select problem that uses a metaparameter."""
    model_space_tsv = (
        "model_subspace_id\tmodel_subspace_petab_yaml\tmp1\tk3\n"
        f"{model_space_row}\n"
    )
    (directory / "model_space.tsv").write_text(model_space_tsv)
    problem_yaml = (
        "format_version: 1.0.0\n"
        "criterion: AIC\n"
        "method: forward\n"
        "model_space_files:\n"
        "- model_space.tsv\n"
        "metaparameters:\n"
        "  mp1:\n"
        "  - k1\n"
        "  - k2\n"
    )
    problem_path = directory / "petab_select_problem.yaml"
    problem_path.write_text(problem_yaml)
    return Problem.from_yaml(problem_path)


@pytest.fixture
def problem(tmp_path) -> Problem:
    return write_problem(tmp_path, f"M\t{PETAB_YAML}\t0;estimate\t0;estimate")


def test_problem_metaparameters(problem):
    """Metaparameters are parsed from the problem YAML."""
    assert problem.metaparameters == METAPARAMETERS


def test_subspace_metaparameters(problem):
    """The model subspace represents metaparameters.

    i.e. the parameters in the metaparameter group should not exist
    in the subspace explicitly, only via the metaparameter.
    """
    subspace = problem.model_space.model_subspaces["M"]
    assert subspace.metaparameters == METAPARAMETERS

    # The metaparameter, and not its parameters, exist
    assert "mp1" in subspace.parameters_all
    assert "k1" not in subspace.parameters_all
    assert "k2" not in subspace.parameters_all

    # The metaparameter can be "estimated"
    assert "mp1" in subspace.can_estimate
    # `sigma_x2` is estimated in the PEtab problem and is not grouped.
    assert subspace.must_estimate_all == ["sigma_x2"]


def test_model_metaparameter_expansion(problem):
    """Metaparameters expand to their member PEtab parameters."""
    subspace = problem.model_space.model_subspaces["M"]
    model = subspace.parameters_to_model({"mp1": ESTIMATE, "k3": 0})

    # Compact representation (as in the model space file).
    assert model.parameters == {"mp1": ESTIMATE, "k3": 0}
    # The generated PEtab correctly assigns the metaparameter value to its parameters.
    parameter_df = model.to_petab()["petab_problem"].parameter_df
    assert parameter_df.loc["k1", ESTIMATE] == 1
    assert parameter_df.loc["k2", ESTIMATE] == 1
    assert parameter_df.loc["k3", ESTIMATE] == 0


def test_estimated_parameter_ids_are_compact(problem):
    """A metaparameter is a single estimated parameter."""
    subspace = problem.model_space.model_subspaces["M"]
    model = subspace.parameters_to_model({"mp1": ESTIMATE, "k3": 0})
    # `mp1` (not `k1`/`k2`) is reported, plus the always-estimated `sigma_x2`.
    assert set(model.get_estimated_parameter_ids()) == {"mp1", "sigma_x2"}


def test_n_estimated_counts_members(problem):
    """The number of estimated parameters counts metaparameter members."""
    subspace = problem.model_space.model_subspaces["M"]

    model_on = subspace.parameters_to_model({"mp1": ESTIMATE, "k3": 0})
    petab_problem_on = model_on.to_petab()["petab_problem"]
    # Estimating `mp1` estimates both `k1` and `k2` (plus `sigma_x2`).
    assert set(petab_problem_on.x_free_ids) == {"k1", "k2", "sigma_x2"}

    problem.model_space.reset_exclusions()
    model_off = subspace.parameters_to_model({"mp1": 0, "k3": 0})
    petab_problem_off = model_off.to_petab()["petab_problem"]
    # Fixing `mp1` fixes both `k1` and `k2` (to ``0``).
    assert set(petab_problem_off.x_free_ids) == {"sigma_x2"}
    assert petab_problem_off.parameter_df.loc["k1", ESTIMATE] == 0
    assert petab_problem_off.parameter_df.loc["k2", ESTIMATE] == 0
    assert petab_problem_off.parameter_df.loc["k1", "nominalValue"] == 0
    assert petab_problem_off.parameter_df.loc["k2", "nominalValue"] == 0


def test_aic_uses_real_parameter_count(problem):
    """The AIC uses the real (expanded) number of estimated parameters."""
    subspace = problem.model_space.model_subspaces["M"]
    model = subspace.parameters_to_model({"mp1": ESTIMATE, "k3": 0})
    model.set_criterion(Criterion.NLLH, 10.0)
    # AIC = 2 * (n_estimated + NLLH) = 2 * (3 + 10) = 26.
    assert model.get_criterion(Criterion.AIC) == 26.0


def test_forward_move_is_single_step(problem):
    """Estimating a metaparameter is a single forward move."""
    subspace = problem.model_space.model_subspaces["M"]
    predecessor = subspace.parameters_to_model({"mp1": 0, "k3": 0})

    candidate_space = problem.new_candidate_space(
        predecessor_model=predecessor
    )
    problem.model_space.reset_exclusions()
    problem.model_space.search(candidate_space)

    candidate_parameters = [m.parameters for m in candidate_space.models]
    # Both single-parameter moves are equally minimal forward steps:
    # estimating the metaparameter, or estimating `k3`.
    assert {"mp1": ESTIMATE, "k3": 0} in candidate_parameters
    assert {"mp1": 0, "k3": ESTIMATE} in candidate_parameters
    assert len(candidate_parameters) == 2


def test_backward_move_is_single_step(problem):
    """Fixing a metaparameter is a single backward move."""
    subspace = problem.model_space.model_subspaces["M"]
    predecessor = subspace.parameters_to_model(
        {"mp1": ESTIMATE, "k3": ESTIMATE}
    )

    candidate_space = problem.new_candidate_space(
        method="backward",
        predecessor_model=predecessor,
    )
    problem.model_space.reset_exclusions()
    problem.model_space.search(candidate_space)

    candidate_parameters = [m.parameters for m in candidate_space.models]
    assert {"mp1": 0, "k3": ESTIMATE} in candidate_parameters
    assert {"mp1": ESTIMATE, "k3": 0} in candidate_parameters
    assert len(candidate_parameters) == 2


def test_yaml_round_trip(problem, tmp_path):
    """A model with a metaparameter survives a YAML round trip."""
    subspace = problem.model_space.model_subspaces["M"]
    model = subspace.parameters_to_model({"mp1": ESTIMATE, "k3": 0})

    model_yaml = tmp_path / "model.yaml"
    model.to_yaml(model_yaml)
    reloaded = Model.from_yaml(model_yaml)

    assert reloaded.metaparameters == METAPARAMETERS
    assert reloaded.parameters == {"mp1": ESTIMATE, "k3": 0}
    # The reloaded model still expands ``mp1`` into ``k1`` and ``k2``.
    petab_problem = reloaded.to_petab()["petab_problem"]
    assert set(petab_problem.x_free_ids) == {"k1", "k2", "sigma_x2"}
    assert len(petab_problem.x_free_indices) == 3


def test_invalid_metaparameter_member(tmp_path):
    """A metaparameter referencing an unknown PEtab parameter is rejected."""
    model_space_tsv = (
        "model_subspace_id\tmodel_subspace_petab_yaml\tmp1\n"
        f"M\t{PETAB_YAML}\t0;estimate\n"
    )
    (tmp_path / "model_space.tsv").write_text(model_space_tsv)
    problem_yaml = (
        "format_version: 1.0.0\n"
        "criterion: AIC\n"
        "method: forward\n"
        "model_space_files:\n"
        "- model_space.tsv\n"
        "metaparameters:\n"
        "  mp1:\n"
        "  - k1\n"
        "  - does_not_exist\n"
    )
    problem_path = tmp_path / "petab_select_problem.yaml"
    problem_path.write_text(problem_yaml)
    with pytest.raises(ValueError, match="not in the PEtab problem"):
        Problem.from_yaml(problem_path)


def test_metaparameter_member_as_column_is_rejected(tmp_path):
    """A metaparameter member must not also be its own model space column."""
    model_space_tsv = (
        "model_subspace_id\tmodel_subspace_petab_yaml\tmp1\tk1\n"
        f"M\t{PETAB_YAML}\t0;estimate\t0;estimate\n"
    )
    (tmp_path / "model_space.tsv").write_text(model_space_tsv)
    problem_yaml = (
        "format_version: 1.0.0\n"
        "criterion: AIC\n"
        "method: forward\n"
        "model_space_files:\n"
        "- model_space.tsv\n"
        "metaparameters:\n"
        "  mp1:\n"
        "  - k1\n"
        "  - k2\n"
    )
    problem_path = tmp_path / "petab_select_problem.yaml"
    problem_path.write_text(problem_yaml)
    with pytest.raises(ValueError, match="also appear as their own columns"):
        Problem.from_yaml(problem_path)
