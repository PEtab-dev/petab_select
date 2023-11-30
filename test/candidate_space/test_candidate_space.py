from pathlib import Path

import pandas as pd
import pytest
from more_itertools import one

import petab_select
from petab_select.candidate_space import (  # BackwardCandidateSpace,; BruteForceCandidateSpace,; ForwardCandidateSpace,; ForwardAndBackwardCandidateSpace,; LateralCandidateSpace,
    BidirectionalCandidateSpace,
)
from petab_select.constants import (
    ESTIMATE,
    MODEL_SUBSPACE_ID,
    MODELS,
    PARAMETER_VALUE_DELIMITER,
    PETAB_YAML,
    Criterion,
)
from petab_select.model import Model, default_compare
from petab_select.model_space import ModelSpace, get_model_space_df
from petab_select.model_subspace import ModelSubspace


@pytest.fixture
def ordered_model_parameterizations():
    good_models_ascending = [
        # forward
        '00000',
        '10000',
        '11000',
        '11100',
        '11110',
        # backward
        '01110',
        '01100',
        # forward
        '01101',
        '01111',
        # backward
        '00111',
        '00011',
    ]
    bad_models = [
        '01011',
        '11011',
    ]

    # All good models are unique
    assert len(set(good_models_ascending)) == len(good_models_ascending)
    # All bad models are unique
    assert len(set(bad_models)) == len(bad_models)
    # No models are defined as both bad and good.
    assert not set(good_models_ascending).intersection(bad_models)

    return good_models_ascending, bad_models


@pytest.fixture
def calibrated_model_space(ordered_model_parameterizations):
    good_models_ascending, bad_models = ordered_model_parameterizations

    # As good models are ordered ascending by "goodness", and criteria
    # decreases for better models, the criteria decreases as the index increases.
    good_model_criteria = {
        model: 100 - index for index, model in enumerate(good_models_ascending)
    }
    # All bad models are currently set to the same "bad" criterion value.
    bad_model_criteria = {model: 1000 for model in bad_models}

    model_criteria = {
        **good_model_criteria,
        **bad_model_criteria,
    }
    return model_criteria


@pytest.fixture
def model_space(calibrated_model_space) -> pd.DataFrame:
    data = {
        "model_subspace_id": [],
        "petab_yaml": [],
        "k1": [],
        "k2": [],
        "k3": [],
        "k4": [],
        "k5": [],
    }

    for model in calibrated_model_space:
        data["model_subspace_id"].append(f"model_subspace_{model}")
        data["petab_yaml"].append(
            Path(__file__).parent.parent.parent
            / 'doc'
            / 'examples'
            / 'model_selection'
            / 'petab_problem.yaml'
        )
        k1, k2, k3, k4, k5 = [
            '0' if parameter == '0' else ESTIMATE for parameter in model
        ]
        data["k1"].append(k1)
        data["k2"].append(k2)
        data["k3"].append(k3)
        data["k4"].append(k4)
        data["k5"].append(k5)
    df = pd.DataFrame(data=data)
    df = get_model_space_df(df)
    model_space = ModelSpace.from_df(df)
    return model_space


#def test_bidirectional(
#    model_space, calibrated_model_space, ordered_model_parameterizations
#):
#    criterion = Criterion.AIC
#    model_id_length = one(
#        set([len(model_id) for model_id in calibrated_model_space])
#    )
#
#    candidate_space = BidirectionalCandidateSpace()
#    calibrated_models = []
#
#    # Perform bidirectional search until no more models are found.
#    search_iterations = 0
#    while True:
#        new_calibrated_models = []
#        search_iterations += 1
#
#        # Get models.
#        model_space.search(candidate_space)
#
#        # Calibrate models.
#        for model in candidate_space.models:
#            model_id = model.model_subspace_id[-model_id_length:]
#            model.set_criterion(
#                criterion=criterion, value=calibrated_model_space[model_id]
#            )
#            new_calibrated_models.append(model)
#
#        # End if no more models were found.
#        if not new_calibrated_models:
#            break
#
#        # Get next predecessor model as best of new models.
#        best_new_model = None
#        for model in new_calibrated_models:
#            if best_new_model is None:
#                best_new_model = model
#                continue
#            if default_compare(
#                model0=best_new_model, model1=model, criterion=criterion
#            ):
#                best_new_model = model
#
#        # Set next predecessor and exclusions.
#        calibrated_model_hashes = [
#            model.get_hash() for model in calibrated_models
#        ]
#        candidate_space.reset(
#            predecessor_model=best_new_model,
#            exclusions=calibrated_model_hashes,
#        )
#
#        # exclude calibrated model hashes from model space too?
#        model_space.exclude_model_hashes(model_hashes=calibrated_model_hashes)
#
#    # Check that expected models are found at each iteration.
#    (
#        good_model_parameterizations_ascending,
#        bad_model_parameterizations,
#    ) = ordered_model_parameterizations
#    search_iteration = 0
#    for history_item in candidate_space.method_history:
#        models = history_item[MODELS]
#        if not models:
#            continue
#        model_parameterizations = [
#            model.model_subspace_id[-5:] for model in models
#        ]
#
#        good_model_parameterization = good_model_parameterizations_ascending[
#            search_iteration
#        ]
#        # The expected good model was found.
#        assert good_model_parameterization in model_parameterizations
#        model_parameterizations.remove(good_model_parameterization)
#
#        if search_iteration == 0:
#            # All parameterizations have been correctly identified and removed.
#            assert not model_parameterizations
#            search_iteration += 1
#            continue
#
#        previous_model_parameterization = (
#            good_model_parameterizations_ascending[search_iteration - 1]
#        )
#
#        # The expected bad model is also found.
#        # If a bad model is the same dimension and also represents a similar stepwise move away from the previous
#        # model parameterization, it should also be in the parameterizations.
#        for bad_model_parameterization in bad_model_parameterizations:
#            # Skip if different dimensions
#            if sum(map(int, bad_model_parameterization)) != sum(
#                map(int, good_model_parameterization)
#            ):
#                continue
#            # Skip if different distances from previous model parameterization
#            if sum(
#                [
#                    a != b
#                    for a, b in zip(
#                        bad_model_parameterization,
#                        previous_model_parameterization,
#                    )
#                ]
#            ) != sum(
#                [
#                    a != b
#                    for a, b in zip(
#                        good_model_parameterization,
#                        previous_model_parameterization,
#                    )
#                ]
#            ):
#                continue
#            assert bad_model_parameterization in model_parameterizations
#            model_parameterizations.remove(bad_model_parameterization)
#
#        # All parameterizations have been correctly identified and removed.
#        assert not model_parameterizations
#        search_iteration += 1
#
#        # End test if all good models were found in the correct order.
#        if search_iteration >= len(good_model_parameterizations_ascending):
#            break
