"""Classes and methods related to candidate spaces."""
import abc
from typing import Dict, Optional, Union
import warnings

from more_itertools import one
import numpy as np

from .model import Model


INITIAL_MODEL_METHODS = [
    'backward',
    'lateral',
    'forward',
]


class CandidateSpace(abc.ABC):
    """A base class for collecting candidate models.

    The intended use of subclasses is to identify suitable models in a model
    space, that will be provided to a model selection method for selection.

    Attributes:
        distances:
            The distances of all candidate models from the initial model.
        model0:
            The initial model.
        models:
            The current set of candidate models.
    """
    def __init__(self, model0: Optional[Model] = None):
        self.reset(model0)

    @abc.abstractmethod
    def is_plausible(self, model: Model) -> bool:
        """Determine whether a candidate model is plausible.

        A plausible model is one that could possibly be chosen during the
        model selection algorithm, in the absence of information about other
        models in the model space.

        For example, given a forward selection method that starts with an
        initial model `self.model0` that has no estimated parameters, then only
        models with one or more estimated parameters are plausible.

        Args:
            model:
                The candidate model.

        Returns:
            `True` if `model` is plausible, else `False`.
        """
        pass

    @abc.abstractmethod
    def distance(self, model: Model) -> Union[float, int]:
        """Compute the distance between two models that are neighbors.

        Args:
            model:
                The candidate model.
            model0:
                The initial model.

        Returns:
            The distance from `model0` to `model`.
        """
        pass

    @abc.abstractmethod
    def consider(self, model: Model) -> None:
        """Add a candidate model, if it should be added.

        Args:
            model:
                The candidate model.
        """
        pass

    def reset(self, model0: Union[Model, None]):
        """Reset the candidate models, reinitialize with a model.

        Args:
            model0:
                The initial model.
        """
        self.model0 = model0
        self.models = []
        self.distances = []


class ForwardCandidateSpace(CandidateSpace):
    """The forward method class.

    Attributes:
        direction:
            `1` for the forward method, `-1` for the backward method.
    """
    direction = 1

    def is_plausible(self, model: Model) -> bool:
        distances = distances_in_estimated_parameters(model, self.model0)
        unsigned_size = self.direction * distances['size']
        # A model is plausible if the number of estimated parameters strictly
        # increases (or decreases, if `self.direction == -1`), and no
        # previously estimated parameters become fixed.
        if (
            unsigned_size > 0 and
            distances['l1'] == unsigned_size
        ):
            return True
        return False

    def distance(self, model: Model) -> int:
        # TODO calculated here and `is_plausible`. Rewrite to only calculate
        #      once?
        distances = distances_in_estimated_parameters(model, self.model0)
        return distances['l1']

    def consider(self, model) -> None:
        if not self.is_plausible(model):
            return

        distance = self.distance(model)

        # Get the distance of the current "best" plausible model(s)
        if self.distances:
            distance0 = one(self.distances)
        else:
            distance0 = np.inf

        # Only keep the best model(s).
        if distance == distance0:
            self.models.append(model)
        elif distance < distance0:
            self.models = [model]
            self.distances = [distance]


class BackwardCandidateSpace(ForwardCandidateSpace):
    """The backward method class."""
    direction = -1


class LateralCandidateSpace(ForwardCandidateSpace):
    """
    A method class to find models with the same number of estimated parameters.
    """
    def is_plausible(self, model: Model) -> bool:
        distances = distances_in_estimated_parameters(model, self.model0)
        # A model is plausible if the number of estimated parameters remains
        # the same, but some estimated parameters have become fixed and vice
        # versa.
        if (
            distances['size'] == 0 and
            # distances['size'] == 0 implies L1 % 2 == 0.
            # FIXME here and elsewhere, deal with models that are equal
            #       except for the values of their fixed parameters.
            distances['l1'] > 0
        ):
            return True
        return False


class BruteForceCandidateSpace(CandidateSpace):
    """The brute-force method class."""
    def is_plausible(self, model: Model) -> bool:
        # All models are plausible for the brute force method.
        # TODO check constraints.
        return True

    def distance(self, model: Model) -> int:
        # Distance is irrelevant for the brute force method.
        return 0

    def consider(self, model) -> None:
        self.models.append(model)


def distances_in_estimated_parameters(
    model: Model,
    model0: Model,
) -> Dict[str, Union[float, int]]:
    """Distance between two models in model space, using different metrics.

    All metrics are in terms of estimated parameters.

    Metrics:
        l1:
            The L_1 distance between two models.
        size:
            The difference in the number of estimated parameters between two
            models.

    Args:
        model:
            The candidate model.
        model0:
            The initial model.

    Returns:
        The distances between the models, as a dictionary, where a key is the
        name of the metric, and the value is the corresponding distance.
    """
    if set(model.parameters.keys()) != set(model0.parameters.keys()):
        warnings.warn(
            f'The models `{model.model_id}` and `{model0.model_id}` have '
            'different (estimated) parameter spaces. This is currently '
            'untested.'
        )

    parameters0 = [model0.parameters[k] for k in model.parameters]
    parameters = [model.parameters[k] for k in model.parameters]

    estimated0 = np.isnan(parameters0).astype(int)
    estimated = np.isnan(parameters).astype(int)

    difference = estimated - estimated0

    # Changes to(/from) estimated from(/to) not estimated
    l1 = np.abs(difference).sum()

    # Change in the number of estimated parameters.
    size = np.sum(difference)

    # TODO constants?
    distances = {
        'l1': l1,
        'size': size,
    }

    return distances
