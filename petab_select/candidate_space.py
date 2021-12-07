"""Classes and methods related to candidate spaces."""
import abc
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from more_itertools import one

from .constants import (ESTIMATE, VIRTUAL_INITIAL_MODEL,
                        VIRTUAL_INITIAL_MODEL_METHODS, Method)
from .handlers import TYPE_LIMIT, LimitHandler
from .misc import snake_case_to_camel_case
from .model import Model


class CandidateSpace(abc.ABC):
    """A base class for collecting candidate models.

    The intended use of subclasses is to identify suitable models in a model
    space, that will be provided to a model selection method for selection.

    Attributes:
        distances:
            The distances of all candidate models from the initial model.
            FIXME change list to int? Is storage of more than one value
            useful?
        predecessor_model:
            The model used for comparison, e.g. for stepwise methods.
        models:
            The current set of candidate models.
        exclusions:
            A list of model hashes. Models that match a hash in `exclusions` will not
            be accepted into the candidate space. The hashes of models that are accepted
            are added to `exclusions`.
        limit:
            A handler to limit the number of accepted models.
        #limited:
        #    A descriptor that handles the limit on the number of accepted models.
        #limit:
        #    Models will fail `self.consider` if `len(self.models) >= limit`.
    """

    def __init__(
        self,
        # TODO add MODEL_TYPE = Union[str, Model], str for VIRTUAL_INITIAL_MODEL
        predecessor_model: Optional[Model] = None,
        exclusions: Optional[List[Any]] = None,
        upper_limit: TYPE_LIMIT = np.inf,
    ):
        self.limit = LimitHandler(
            current=self.n_accepted,
            limit=upper_limit,
        )
        self.reset(predecessor_model=predecessor_model, exclusions=exclusions)

    def is_plausible(self, model: Model) -> bool:
        """Determine whether a candidate model is plausible.

        A plausible model is one that could possibly be chosen by
        model selection method, in the absence of information about other
        models in the model space.

        For example, given a forward selection method that starts with an
        initial model `self.predecessor_model` that has no estimated
        parameters, then only models with one or more estimated parameters are
        plausible.

        Args:
            model:
                The candidate model.

        Returns:
            `True` if `model` is plausible, else `False`.
        """
        return True

    def distance(self, model: Model) -> Union[None, float, int]:
        """Compute the distance between two models that are neighbors.

        Args:
            model:
                The candidate model.
            predecessor_model:
                The initial model.

        Returns:
            The distance from `predecessor_model` to `model`, or `None` if the
            distance should not be computed.
        """
        return None

    def accept(
        self,
        model: Model,
        distance: Union[None, float, int],
        #keep_others: bool = True,
    ) -> None:
        """Add a candidate model to the candidate space.

        Args:
            model:
                The model that will be added.
            distance:
                The distance of the model from the predecessor model.
            #keep_others:
            #    Whether to keep other models that were previously added to the
            #    candidate space.
        """
        model.predecessor_model_id = (
            self.predecessor_model.model_id
            if isinstance(self.predecessor_model, Model)
            else self.predecessor_model
        )
        self.models.append(model)
        self.distances.append(distance)
        self.exclude(model)

    def n_accepted(self) -> TYPE_LIMIT:
        """Get the current number of accepted models."""
        return len(self.models)

    def exclude(
        self,
        model: Union[Model, List[Model]],
    ) -> None:
        """Exclude model(s) from future consideration.

        Args:
            model:
                The model(s) that will be excluded.
        """
        if isinstance(model, list):
            for _model in model:
                self.exclusions.append(_model.get_hash())
        else:
            self.exclusions.append(model.get_hash())

    def excluded(
        self,
        model: Model,
    ) -> bool:
        """Whether a model is excluded."""
        return model.get_hash() in self.exclusions

    @abc.abstractmethod
    def _consider_method(self, model) -> bool:
        """Consider whether a model should be accepted, according to a method.

        Args:
            model:
                The candidate model.

        Returns:
            Whether a model should be accepted.
        """
        return True

    def consider(self, model: Union[Model, None]) -> bool:
        """Add a candidate model, if it should be added.

        Args:
            model:
                The candidate model. This value may be `None` if the `ModelSubspace`
                decided to exclude the model that would have been sent.

        Returns:
            Whether it is OK to send additional models to the candidate space. For
            example, if the limit of the number of accepted models has been reached,
            then no further models should be sent.
            TODO change to return whether the model was accepted, and instead add
                 `self.continue` to determine whether additional models should be sent.
        """
        # Model was excluded by the `ModelSubspace` that called this method, so can be
        # skipped.
        if model is None:
            # TODO use a different code than `True`?
            return True
        if self.limit.reached():
            return False
        if self.excluded(model):
            warnings.warn(f'Model has been previously excluded from the model space so is skipped here. Model subspace ID: {model.model_subspace_id}. Parameterization: {model.parameters}')
            return True
        if not self.is_plausible(model):
            return True
        if not self._consider_method(model):
            return True

        self.accept(model, distance=self.distance(model))
        return not self.limit.reached()

    def reset_accepted(self) -> None:
        """Reset the accepted models."""
        self.models = []
        self.distances = []

    def reset(
        self,
        predecessor_model: Optional[Union[Model, str, None]] = None,
        # FIXME change `Any` to some `TYPE_MODEL_HASH` (e.g. union of str/int/float)
        exclusions: Optional[Union[List[Any], None]] = None,
        limit: TYPE_LIMIT = None,
    ) -> None:
        """Reset the candidate models, optionally reinitialize with a model.

        Args:
            predecessor_model:
                The initial model.
            exclusions:
                Whether to reset model exclusions.
            limit:
                The new upper limit of the number of models in this candidate space.
        """
        # TODO if is not None?
        self.predecessor_model = predecessor_model
        if (
            self.predecessor_model == VIRTUAL_INITIAL_MODEL
            and self.method not in VIRTUAL_INITIAL_MODEL_METHODS
        ):
            raise ValueError(f'A virtual initial model was requested for a method ({self.method}) that does not support them.')  # noqa: E501
        self.reset_accepted()
        self.exclusions = exclusions
        if self.exclusions is None:
            self.exclusions = []
        if limit is not None:
            self.limit.set_limit(limit)

    def distances_in_estimated_parameters(
        self,
        model: Model,
        predecessor_model: Optional[Model] = None,
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

        Returns:
            The distances between the models, as a dictionary, where a key is the
            name of the metric, and the value is the corresponding distance.
        """
        model0 = predecessor_model
        if model0 is None:
            model0 = self.predecessor_model
        model1 = model

        if (
            model0 != VIRTUAL_INITIAL_MODEL
            and not model1.petab_yaml.samefile(model0.petab_yaml)
        ):
            raise NotImplementedError(
                'Computation of distances between different PEtab problems is '
                'currently not supported. This error is also raised if the same '
                'PEtab problem is read from YAML files in different locations.'
            )

        # All parameters from the PEtab problem are used in the computation.
        if model0 == VIRTUAL_INITIAL_MODEL:
            parameter_ids = list(model1.petab_parameters)
            # FIXME need to take superset of all parameters amongst all PEtab problems
            # in all model subspaces to get an accurate comparable distance. Currently
            # only reasonable when working with a single PEtab problem for all models
            # in all subspaces.
            if self.method == Method.FORWARD:
                parameters0 = np.array([0 for _ in parameter_ids])
            elif self.method == Method.BACKWARD:
                parameters0 = np.array([ESTIMATE for _ in parameter_ids])
            else:
                raise NotImplementedError(
                    'Distances for the virtual initial model have not yet been '
                    f'implemented for the method "{self.method}". Please notify the'
                    'developers.'
                )
        else:
            parameter_ids = list(model0.petab_parameters)
            parameters0 = np.array(model0.get_parameter_values(parameter_ids=parameter_ids))

        parameters1 = np.array(model1.get_parameter_values(parameter_ids=parameter_ids))

        # TODO change to some numpy elementwise operation
        estimated0 = np.array([p == ESTIMATE for p in parameters0]).astype(int)
        estimated1 = np.array([p == ESTIMATE for p in parameters1]).astype(int)

        difference = estimated1 - estimated0

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



class ForwardCandidateSpace(CandidateSpace):
    """The forward method class.

    Attributes:
        direction:
            `1` for the forward method, `-1` for the backward method.
    """
    method = Method.FORWARD
    direction = 1

    def __init__(
        self,
        *args,
        predecessor_model: Union[Model, str] = VIRTUAL_INITIAL_MODEL,
        **kwargs,
    ):
        super().__init__(*args, predecessor_model=predecessor_model, **kwargs)

    def is_plausible(self, model: Model) -> bool:
        distances = \
            self.distances_in_estimated_parameters(model)
        unsigned_size = self.direction * distances['size']
        # A model is plausible if the number of estimated parameters strictly
        # increases (or decreases, if `self.direction == -1`), and no
        # previously estimated parameters become fixed.
        if (
            self.predecessor_model == VIRTUAL_INITIAL_MODEL
            or (
                unsigned_size > 0
                and distances['l1'] == unsigned_size
            )
        ):
            return True
        return False

    def distance(self, model: Model) -> int:
        # TODO calculated here and `is_plausible`. Rewrite to only calculate
        #      once?
        distances = \
            self.distances_in_estimated_parameters(model)
        return distances['l1']

    def _consider_method(self, model) -> bool:
        """See `CandidateSpace._consider_method`."""
        distance = self.distance(model)

        # Get the distance of the current "best" plausible model(s)
        distance0 = np.inf
        if self.distances:
            distance0 = one(set(self.distances))
            # TODO store each or just one?
            #distance0 = one(self.distances)
        #breakpoint()

        # Only keep the best model(s).
        if distance > distance0:
            return False
        if distance < distance0:
            self.reset_accepted()
        return True


class BackwardCandidateSpace(ForwardCandidateSpace):
    """The backward method class."""
    method = Method.BACKWARD
    direction = -1


class LateralCandidateSpace(ForwardCandidateSpace):
    """
    A method class to find models with the same number of estimated parameters.
    """
    method = Method.LATERAL
    def is_plausible(self, model: Model) -> bool:
        distances = \
            self.distances_in_estimated_parameters(model)
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

    def _consider_method(self, model):
        return True


class BruteForceCandidateSpace(CandidateSpace):
    """The brute-force method class."""
    method = Method.BRUTE_FORCE

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            # FIXME remove?
            # FIXME at least support limit
            warnings.warn(
                'Arguments were provided but will be ignored, because of the '
                'brute force candidate space.'
            )
        super().__init__()

    def _consider_method(self, model):
        return True


candidate_space_classes = [
    ForwardCandidateSpace,
    BackwardCandidateSpace,
    LateralCandidateSpace,
    BruteForceCandidateSpace,
]


def method_to_candidate_space_class(method: str) -> str:
    """Instantiate a candidate space given its method name.

    Args:
        method:
            The name of the method corresponding to one of the implemented candidate
            spaces.

    Returns:
        The candidate space.
    """
    for candidate_space_class in candidate_space_classes:
        if candidate_space_class.method == method:
            return candidate_space_class
    raise NotImplementedError(
        f'The provided method name {method} does not correspond to an '
        'implemented candidate space.'
    )


'''
def distances_in_estimated_parameters(
    model: Model,
    predecessor_model: Model,
    #method: str = None,
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
        predecessor_model:
            The initial model.
        method:
            The search method in use. Necessary if the predecessor model is the
            virtual initial model.

    Returns:
        The distances between the models, as a dictionary, where a key is the
        name of the metric, and the value is the corresponding distance.
    """
    model0 = predecessor_model
    model1 = model

    if not model1.petab_yaml.samefile(model0.petab_yaml):
        raise NotImplementedError(
            'Computation of distances between different PEtab problems is '
            'currently not supported. This error is also raised if the same '
            'PEtab problem is read from YAML files in different locations.'
        )

    # All parameters from the PEtab problem are used in the computation.
    parameter_ids = list(model0.petab_parameters)

    parameters0 = np.array(model0.get_parameter_values(parameter_ids=parameter_ids))
    parameters1 = np.array(model1.get_parameter_values(parameter_ids=parameter_ids))

    estimated0 = (parameters0 == ESTIMATE).astype(int)
    estimated1 = (parameters1 == ESTIMATE).astype(int)

    difference = estimated1 - estimated0

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
'''
