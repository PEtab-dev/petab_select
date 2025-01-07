from pathlib import Path

import pytest

from petab_select.candidate_space import (
    BackwardCandidateSpace,
    BruteForceCandidateSpace,
    ForwardCandidateSpace,
)
from petab_select.constants import (
    ESTIMATE,
    Criterion,
)
from petab_select.model_space import ModelSpace

base_dir = Path(__file__).parent


@pytest.fixture
def model_space_files() -> list[Path]:
    return [
        base_dir / "model_space_file_1.tsv",
        base_dir / "model_space_file_2.tsv",
    ]


@pytest.fixture
def model_space(model_space_files) -> ModelSpace:
    return ModelSpace.load(model_space_files)


def test_model_space_forward_virtual(model_space):
    candidate_space = ForwardCandidateSpace(criterion=Criterion.NLLH)
    model_space.search(candidate_space)

    # The forward candidate space is initialized without a model, so a virtual initial
    # model is used. This means the expected models are the "smallest" models (as many
    # fixed parameters as possible) in the model space.
    expected_models = [
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": 0.1, "k3": ESTIMATE, "k4": 0.0},
        ),
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": 0.1, "k3": ESTIMATE, "k4": 0.1},
        ),
        (
            "model_subspace_2",
            {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": ESTIMATE},
        ),
    ]

    models = [
        (model.model_subspace_id, model.parameters)
        for model in candidate_space.models
    ]

    # Search found only expected models.
    assert all(model in expected_models for model in models)
    # All expected models have now been added to the candidate space.
    assert all(model in models for model in expected_models)
    # Probably unnecessary: same number of models in expectation vs realization
    assert len(expected_models) == len(candidate_space.models)


@pytest.mark.filterwarnings("ignore:Model has been previously excluded")
def test_model_space_backward_virtual(model_space):
    candidate_space = BackwardCandidateSpace(criterion=Criterion.NLLH)
    model_space.search(candidate_space)

    # The backward candidate space is initialized without a model, so a virtual
    # initial model is used. This means the expected models are the "smallest"
    # models (as many fixed parameters as possible) in the model space.
    expected_models = [
        ("model_subspace_1", {f"k{i}": ESTIMATE for i in range(1, 5)}),
        # This model could be excluded, if the hashes/model comparisons enabled
        # identification of identical models between different subspaces.
        # TODO delete above, keep below comment, when implemented...
        # This model is not included because it is exactly the same as the
        # other model (same PEtab YAML and parameterization), hence has been
        # excluded from the candidate space.
        ("model_subspace_3", {f"k{i}": ESTIMATE for i in range(1, 5)}),
    ]

    models = [
        (model.model_subspace_id, model.parameters)
        for model in candidate_space.models
    ]

    # Search found only expected models.
    assert all(model in expected_models for model in models)
    # All expected models have now been added to the candidate space.
    assert all(model in models for model in expected_models)
    # Probably unnecessary: same number of models in expectation vs realization
    assert len(expected_models) == len(candidate_space.models)


def test_model_space_brute_force_limit(model_space):
    candidate_space = BruteForceCandidateSpace(criterion=Criterion.NLLH)
    model_space.search(candidate_space, limit=13)

    # There are fifteen total models in the model space. Limiting to 13 models should
    # result in all models except the last two models in the last model subspace.
    expected_models = [
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": 0.1, "k3": ESTIMATE, "k4": 0.0},
        ),
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": 0.1, "k3": ESTIMATE, "k4": 0.1},
        ),
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": 0.1, "k3": ESTIMATE, "k4": ESTIMATE},
        ),
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": ESTIMATE, "k3": ESTIMATE, "k4": 0.0},
        ),
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": ESTIMATE, "k3": ESTIMATE, "k4": 0.1},
        ),
        (
            "model_subspace_1",
            {"k1": 0.2, "k2": ESTIMATE, "k3": ESTIMATE, "k4": ESTIMATE},
        ),
        (
            "model_subspace_1",
            {"k1": ESTIMATE, "k2": 0.1, "k3": ESTIMATE, "k4": 0.0},
        ),
        (
            "model_subspace_1",
            {"k1": ESTIMATE, "k2": 0.1, "k3": ESTIMATE, "k4": 0.1},
        ),
        (
            "model_subspace_1",
            {"k1": ESTIMATE, "k2": 0.1, "k3": ESTIMATE, "k4": ESTIMATE},
        ),
        (
            "model_subspace_1",
            {"k1": ESTIMATE, "k2": ESTIMATE, "k3": ESTIMATE, "k4": 0.0},
        ),
        (
            "model_subspace_1",
            {"k1": ESTIMATE, "k2": ESTIMATE, "k3": ESTIMATE, "k4": 0.1},
        ),
        (
            "model_subspace_1",
            {"k1": ESTIMATE, "k2": ESTIMATE, "k3": ESTIMATE, "k4": ESTIMATE},
        ),
        (
            "model_subspace_2",
            {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": ESTIMATE},
        ),
    ]

    models = [
        (model.model_subspace_id, model.parameters)
        for model in candidate_space.models
    ]

    # Search found only expected models.
    assert all(model in expected_models for model in models)
    # All expected models have now been added to the candidate space.
    assert all(model in models for model in expected_models)
    # Probably unnecessary: same number of models in expectation vs realization
    assert len(expected_models) == len(candidate_space.models)


"""
@pytest.fixture
def e():
    return ESTIMATE

@pytest.fixture
def points(e):
    return [
        [0, 0, 0, 0],

        [e, 0, 0, 0],
        [0, e, 0, 0],
        [0, 0, e, 0],
        [0, 0, 0, e],

        [e, e, 0, 0],
        [e, 0, e, 0],
        [e, 0, 0, e],
        [0, e, e, 0],
        [0, e, 0, e],
        [0, 0, e, e],

        [e, e, e, 0],
        [e, e, 0, e],
        [e, 0, e, e],
        [0, e, e, e],

        [e, e, e, e],
    ]


def point_to_parameters(point):
    return {
        f'k{index}': value
        for index, value in enumerate(point)
    }


@pytest.fixture
def models(points):
    return [
        Model(
            model_id='',
            petab_yaml='',
            parameters=point_to_parameters(point),
            criteria=None,
        )
        for point in points
    ]


@pytest.fixture
def model_space(models):
    def model_iterator(models=models):
        for model in models:
            yield model
    return ModelSpace(model_iterator)


def test_distance(model_space, e):
    model0 = Model(
        model_id='model0',
        petab_yaml='.',
        parameters=point_to_parameters([e, 0, 0, 0]),
        criteria=None,
    )

    forward_candidate_space = ForwardCandidateSpace(model0)
    backward_candidate_space = BackwardCandidateSpace(model0)
    lateral_candidate_space = LateralCandidateSpace(model0)
    brute_force_candidate_space = BruteForceCandidateSpace(model0)

    neighbors = model_space.neighbors(forward_candidate_space)
    assert len(neighbors) == 3
    model_space.reset()
    neighbors = model_space.neighbors(backward_candidate_space)
    assert len(neighbors) == 1
    model_space.reset()
    # FIXME currently skips "same" model (same estimated parameters).
    #       However, might be that the fixed parameters are different values.
    neighbors = model_space.neighbors(lateral_candidate_space)
    assert len(neighbors) == 3
    model_space.reset()
    neighbors = model_space.neighbors(brute_force_candidate_space)
    assert len(neighbors) == 16
"""
