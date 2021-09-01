import pytest

from petab_select.constants import (
    ESTIMATE_SYMBOL_INTERNAL,
    FORWARD,
    BACKWARD,
    LATERAL,
)
from petab_select.model import (
    Model,
)
from petab_select.model_space import (
    #get_distances,
    #_is_neighbor,
    ModelSpace,
)

from petab_select.candidate_space import (
    BackwardCandidateSpace,
    BruteForceCandidateSpace,
    ForwardCandidateSpace,
    LateralCandidateSpace,
)

@pytest.fixture
def e():
    return float(ESTIMATE_SYMBOL_INTERNAL)

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
