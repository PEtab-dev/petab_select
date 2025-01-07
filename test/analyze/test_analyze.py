from pathlib import Path

import numpy as np
import pytest

from petab_select import (
    VIRTUAL_INITIAL_MODEL,
    Criterion,
    Models,
    analyze,
)

base_dir = Path(__file__).parent

DUMMY_HASH = "dummy_p0-0"


@pytest.fixture
def models() -> Models:
    return Models.from_yaml(base_dir / "input" / "models.yaml")


def test_group_by_predecessor_model(models: Models) -> None:
    """Test ``analyze.group_by_predecessor_model``."""
    groups = analyze.group_by_predecessor_model(models)
    # Expected groups
    assert len(groups) == 2
    assert VIRTUAL_INITIAL_MODEL.hash in groups
    assert DUMMY_HASH in groups
    # Expected group members
    assert len(groups[DUMMY_HASH]) == 1
    assert "M-011" in groups[DUMMY_HASH]
    assert len(groups[VIRTUAL_INITIAL_MODEL.hash]) == 3
    assert "M-110" in groups[VIRTUAL_INITIAL_MODEL.hash]
    assert "M2-011" in groups[VIRTUAL_INITIAL_MODEL.hash]
    assert "M2-110" in groups[VIRTUAL_INITIAL_MODEL.hash]


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
    assert groups[1].hash == "M2-011"
    assert groups[2].hash == "M2-110"
    assert groups[5].hash == "M-110"


def test_relative_criterion_values(models: Models) -> None:
    """Test ``analyze.get_relative_criterion_values``."""
    # TODO move to test_models.py?
    criterion_values = models.get_criterion(criterion=Criterion.AIC)
    test_value = models.get_criterion(criterion=Criterion.AIC, relative=True)
    expected_value = [
        criterion_value - min(criterion_values)
        for criterion_value in criterion_values
    ]
    assert test_value == expected_value


def test_compute_weights(models: Models) -> None:
    """Test ``analyze.compute_weights``."""
    criterion_values = np.array(
        models.get_criterion(criterion=Criterion.AIC, relative=True)
    )
    expected_weights = (
        np.exp(-0.5 * criterion_values) / np.exp(-0.5 * criterion_values).sum()
    )
    test_weights = analyze.compute_weights(
        models=models, criterion=Criterion.AIC
    )
    np.testing.assert_allclose(test_weights, expected_weights)
