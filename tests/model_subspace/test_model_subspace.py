from pathlib import Path

from more_itertools import one
import pandas as pd
import pytest

import petab_select
from petab_select.candidate_space import (
    BackwardCandidateSpace,
    BruteForceCandidateSpace,
    ForwardCandidateSpace,
)
from petab_select.constants import (
    ESTIMATE,
    MODEL_SUBSPACE_ID,
    PARAMETER_VALUE_DELIMITER,
    PETAB_YAML,
)
from petab_select.model import (
    Model,
)
from petab_select.model_subspace import (
    ModelSubspace,
)


@pytest.fixture
def definition() -> pd.Series:
    data = {
        MODEL_SUBSPACE_ID: 'model_subspace_1',
        PETAB_YAML: Path(__file__).parent.parent.parent / 'doc' / 'examples' / 'model_selection' / 'petab_problem.yaml',  # noqa: E501
        'k1': 0.2,
        'k2': PARAMETER_VALUE_DELIMITER.join(['0.1', ESTIMATE]),
        'k3': ESTIMATE,
        'k4': PARAMETER_VALUE_DELIMITER.join(['0', '0.1', ESTIMATE]),
    }
    return pd.Series(data=data, dtype=str)


@pytest.fixture
def model_subspace(definition) -> ModelSubspace:
    return petab_select.model_subspace.ModelSubspace.from_definition(definition)


@pytest.fixture
def initial_model(model_subspace) -> Model:
    estimated_parameters = ['k3', 'k4']
    model = one(model_subspace.get_models(estimated_parameters=estimated_parameters))
    # Initial model is parameterized as expected.
    assert model.parameters == {'k1': 0.2, 'k2': 0.1, 'k3': ESTIMATE, 'k4': ESTIMATE}
    return model


def test_from_definition(model_subspace):
    """A model subspace definition is parsed correctly."""
    # Model subspace ID is parsed
    assert model_subspace.model_subspace_id == 'model_subspace_1'
    # PEtab YAML is parsed
    assert model_subspace.petab_yaml.samefile(
        Path(__file__).parent.parent.parent / 'doc' / 'examples' / 'model_selection' / 'petab_problem.yaml',  # noqa: E501
    )
    # Fixed parameters are parsed
    assert model_subspace.parameters['k1'] == [0.2]
    # Parameters with multiple values are parsed
    assert (
        0.1 in model_subspace.parameters['k2']
        and ESTIMATE in model_subspace.parameters['k2']
    )
    # Estimated parameters are parsed
    assert model_subspace.parameters['k3'] == [ESTIMATE]


def test_get_models(model_subspace):
    """The getter for models with specific estimated parameters works."""
    estimated_parameters = ['k2', 'k3']
    models = list(model_subspace.get_models(estimated_parameters=estimated_parameters))
    expected_model_parameterizations = [
        {'k1': 0.2, 'k2': ESTIMATE, 'k3': ESTIMATE, 'k4': 0.0},
        {'k1': 0.2, 'k2': ESTIMATE, 'k3': ESTIMATE, 'k4': 0.1},
    ]
    model_parameterizations = [model.parameters for model in models]
    # Getter gets only expected models.
    assert all([
        parameterization in expected_model_parameterizations
        for parameterization in model_parameterizations
    ])
    # Getter gets all expected models.
    assert all([
        parameterization in model_parameterizations
        for parameterization in expected_model_parameterizations
    ])


def test_search_forward(model_subspace, initial_model):
    # TODO exclude history, use limit
    candidate_space = ForwardCandidateSpace(predecessor_model=initial_model)

    model_subspace.search(candidate_space=candidate_space)
    # Only one model is possible in the forward direction.
    assert len(candidate_space.models) == 1
    # The one model has the expected parameterization.
    expected_forward_parameterization = \
        {'k1': 0.2, 'k2': ESTIMATE, 'k3': ESTIMATE, 'k4': ESTIMATE}
    assert one(candidate_space.models).parameters == expected_forward_parameterization

    # Test limit via model subspace
    # FIXME currently only returns 1 model anyway
    limit_considered_candidates = 1
    model_subspace.reset_exclusions()
    candidate_space.reset(predecessor_model=initial_model)
    model_subspace.search(
        candidate_space=candidate_space,
        limit=limit_considered_candidates,
    )
    # Test limit: the number of candidate models is at the limit (all models in this
    # case).
    assert len(candidate_space.models) == limit_considered_candidates


def test_search_backward(model_subspace, initial_model):
    # TODO exclude history, use limit
    candidate_space = BackwardCandidateSpace(predecessor_model=initial_model)

    model_subspace.search(candidate_space=candidate_space)
    # Only two models are possible in the backward direction.
    assert len(candidate_space.models) == 2
    expected_backward_parameterizations = [
        {'k1': 0.2, 'k2': 0.1, 'k3': ESTIMATE, 'k4': 0},
        {'k1': 0.2, 'k2': 0.1, 'k3': ESTIMATE, 'k4': 0.1},
    ]
    model_parameterizations = [
        model.parameters
        for model in candidate_space.models
    ]
    # Search found only expected models.
    assert all([
        parameterization in expected_backward_parameterizations
        for parameterization in model_parameterizations
    ])
    # Search found all expected models.
    assert all([
        parameterization in model_parameterizations
        for parameterization in expected_backward_parameterizations
    ])

    # Test limit via model subspace
    limit_considered_candidates = 1
    model_subspace.reset_exclusions()
    candidate_space.reset(predecessor_model=initial_model)
    model_subspace.search(
        candidate_space=candidate_space,
        limit=limit_considered_candidates,
    )
    # Test limit: the number of candidate models is at the limit (all models in this
    # case).
    assert len(candidate_space.models) == limit_considered_candidates


def test_search_brute_force(model_subspace):
    # TODO exclude history, use limit
    candidate_space = BruteForceCandidateSpace()

    model_subspace.search(candidate_space=candidate_space)
    # All models (6) are accepted as candidates.
    assert len(candidate_space.models) == 6

    expected_parameterizations = [
        {'k1': 0.2, 'k2': 0.1, 'k3': ESTIMATE, 'k4': 0},
        {'k1': 0.2, 'k2': 0.1, 'k3': ESTIMATE, 'k4': 0.1},
        {'k1': 0.2, 'k2': 0.1, 'k3': ESTIMATE, 'k4': ESTIMATE},
        {'k1': 0.2, 'k2': ESTIMATE, 'k3': ESTIMATE, 'k4': 0},
        {'k1': 0.2, 'k2': ESTIMATE, 'k3': ESTIMATE, 'k4': 0.1},
        {'k1': 0.2, 'k2': ESTIMATE, 'k3': ESTIMATE, 'k4': ESTIMATE},
    ]
    parameterizations = [
        model.parameters
        for model in candidate_space.models
    ]
    # Search found only expected models.
    assert all([
        parameterization in expected_parameterizations
        for parameterization in parameterizations
    ])
    # Search found all expected models.
    assert all([
        parameterization in parameterizations
        for parameterization in parameterizations
    ])

    limit_accepted_candidates = 3
    candidate_space.reset(limit=limit_accepted_candidates)
    model_subspace.search(candidate_space=candidate_space)
    """ FIXME remove, since models now have to be explicitly excluded. TODO Test exclusions via problem `add_calibrated_model`
    # Test exclusions: no models are in the candidate space as the model subspace
    # already previously sent the models to the candidate space.
    assert len(candidate_space.models) == 0
    model_subspace.reset_exclusions()
    model_subspace.search(candidate_space=candidate_space)
    """
    # Test limit: the number of candidate models is at the limit.
    assert len(candidate_space.models) == limit_accepted_candidates
    # Test limit: a warning is emitted, as the candidate space is already at its limit
    # of models.
    with pytest.warns(RuntimeWarning, match="The candidate space has already reached its limit of accepted models.") as warning_record:
        model_subspace.search(candidate_space=candidate_space)
    # The search stopped after the limit was reached, hence only warned once.
    assert len(warning_record) == 1
    # Test limit: the search adds no additional models to the "full" candidate space.
    assert len(candidate_space.models) == limit_accepted_candidates

    limit_accepted_candidates = 6
    candidate_space.limit.set_limit(limit_accepted_candidates)
    with pytest.warns(RuntimeWarning, match="Model has been previously excluded from the candidate space so is skipped here.") as warning_record:
        model_subspace.search(candidate_space=candidate_space)
    # Three models were excluded from the candidate space in the previous code block.
    assert len(warning_record) == 3
    # Test limit: the number of candidate models is at the limit (all models in this
    # case).
    assert len(candidate_space.models) == limit_accepted_candidates
    # Search found only expected models.
    assert all([
        parameterization in expected_parameterizations
        for parameterization in parameterizations
    ])
    # Test exclusions: all models have now been added to the candidate space.
    # TODO ideally with only 3 additional calls to `candidate_space.consider`, assuming
    #      the model_subspace excluded the first three models it had already sent.
    assert all([
        parameterization in parameterizations
        for parameterization in expected_parameterizations
    ])

    # Test limit via model subspace
    limit_considered_candidates = 1
    model_subspace.reset_exclusions()
    candidate_space.reset()
    model_subspace.search(
        candidate_space=candidate_space,
        limit=limit_considered_candidates,
    )
    # Test limit: the number of candidate models is at the limit (all models in this
    # case).
    assert len(candidate_space.models) == limit_considered_candidates
