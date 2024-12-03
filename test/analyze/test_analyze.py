from pathlib import Path

import pytest

from petab_select import (
    VIRTUAL_INITIAL_MODEL,
    Criterion,
    ModelHash,
    Models,
    analyze,
)

base_dir = Path(__file__).parent

DUMMY_HASH = "dummy_p0-0"
VIRTUAL_HASH = ModelHash.from_hash(VIRTUAL_INITIAL_MODEL)


@pytest.fixture
def models() -> Models:
    return Models.from_yaml(base_dir / "input" / "models.yaml")


def test_group_by_predecessor_model(models: Models) -> None:
    """Test ``analyze.group_by_predecessor_model``."""
    groups = analyze.group_by_predecessor_model(models)
    # Expected groups
    assert len(groups) == 2
    assert VIRTUAL_HASH in groups
    assert DUMMY_HASH in groups
    # Expected group members
    assert len(groups[DUMMY_HASH]) == 1
    assert "M-011" in groups[DUMMY_HASH]
    assert len(groups[VIRTUAL_HASH]) == 3
    assert "M-110" in groups[VIRTUAL_HASH]
    assert "M2-011" in groups[VIRTUAL_HASH]
    assert "M2-110" in groups[VIRTUAL_HASH]


def test_group_by_iteration(models: Models) -> None:
    """Test ``analyze.group_by_iteration``."""
    groups = analyze.group_by_iteration(models)
    # Expected groups
    assert len(groups) == 3
    assert 1 in groups
    assert 2 in groups
    assert 5 in groups
    # Expected group members
    assert len(groups[1]) == 2
    assert "M-011" in groups[1]
    assert "M2-011" in groups[1]
    assert len(groups[2]) == 1
    assert "M2-110" in groups[2]
    assert len(groups[5]) == 1
    assert "M-110" in groups[5]


def test_get_best_by_iteration(models: Models) -> None:
    """Test ``analyze.get_best_by_iteration``."""
    groups = analyze.get_best_by_iteration(models, criterion=Criterion.AIC)
    # Expected groups
    assert len(groups) == 3
    assert 1 in groups
    assert 2 in groups
    assert 5 in groups
    # Expected best models
    assert groups[1].get_hash() == "M2-011"
    assert groups[2].get_hash() == "M2-110"
    assert groups[5].get_hash() == "M-110"


def test_get_relative_criterion_values(models: Models) -> None:
    """Test ``analyze.get_relative_criterion_values``."""
    criterion_values = [model.get_criterion(Criterion.AIC) for model in models]
    test_value = analyze.get_relative_criterion_values(criterion_values)
    expected_value = [
        criterion_value - min(criterion_values)
        for criterion_value in criterion_values
    ]
    assert test_value == expected_value
