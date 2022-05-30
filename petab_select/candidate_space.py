"""Classes and methods related to candidate spaces."""
import abc
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
#from argon2 import Parameters

import numpy as np
from more_itertools import one

from .constants import (
    ESTIMATE,
    METHOD,
    MODELS,
    VIRTUAL_INITIAL_MODEL,
    VIRTUAL_INITIAL_MODEL_METHODS,
    Method,
)
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
        method:
            The model selection method of the candidate space.
        governing_method:
            Used to store the search method that governs the choice of method during
            a search. In some cases, this is always the same as the method attribute.
            An example of a difference is in the bidirectional method, where `governing_method`
            stores the bidirectional method, whereas `method` may also store the forward or
            backward methods.
        retry_model_space_search_if_no_models:
            Whether a search with a candidate space should be repeated upon failure.
            Useful for the `BidirectionalCandidateSpace`, which switches directions
            upon failure.
        #limited:
        #    A descriptor that handles the limit on the number of accepted models.
        #limit:
        #    Models will fail `self.consider` if `len(self.models) >= limit`.
    """

    method: Method = None
    retry_model_space_search_if_no_models: bool = False

    def __init__(
        self,
        # TODO add MODEL_TYPE = Union[str, Model], str for VIRTUAL_INITIAL_MODEL
        predecessor_model: Optional[Model] = None,
        exclusions: Optional[List[Any]] = None,
        limit: TYPE_LIMIT = np.inf,
    ):
        self.limit = LimitHandler(
            current=self.n_accepted,
            limit=limit,
        )
        # Each candidate class specifies this as a class attribute.
        self.governing_method = self.method
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
        # keep_others: bool = True,
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
        model.predecessor_model_hash = (
            self.predecessor_model.get_hash()
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
            warnings.warn(
                f'Model has been previously excluded from the candidate space so is skipped here. Model subspace ID: {model.model_subspace_id}. Parameterization: {model.parameters}',
                RuntimeWarning,
            )
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

    def set_predecessor_model(
        self, predecessor_model: Union[Model, str, None]
    ):
        self.predecessor_model = predecessor_model
        if (
            self.predecessor_model == VIRTUAL_INITIAL_MODEL
            and self.method not in VIRTUAL_INITIAL_MODEL_METHODS
        ):
            raise ValueError(
                f'A virtual initial model was requested for a method ({self.method}) that does not support them.'
            )

    def get_predecessor_model(self):
        return self.predecessor_model

    def set_exclusions(self, exclusions: Union[List[str], None]):
        # TODO change to List[str] for hashes?
        self.exclusions = exclusions
        if self.exclusions is None:
            self.exclusions = []

    def get_exclusions(self):
        return self.exclusions

    def set_limit(self, limit: TYPE_LIMIT = None):
        if limit is not None:
            self.limit.set_limit(limit)

    def get_limit(self):
        return self.limit.get_limit()

    def wrap_search_subspaces(self, search_subspaces: Callable[[], None]):
        """Decorate the subspace searches of a model space.

        Used by candidate spaces to perform changes that alter the search.
        See `BidirectionalCandidateSpace` for an example, where it's used to switch directions.

        Args:
            search_subspaces:
                The method that searches the subspaces of a model space.
        """

        def wrapper():
            search_subspaces()

        return wrapper

    def reset(
        self,
        predecessor_model: Optional[Union[Model, str, None]] = None,
        # FIXME change `Any` to some `TYPE_MODEL_HASH` (e.g. union of str/int/float)
        exclusions: Optional[Union[List[str], None]] = None,
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
        self.set_predecessor_model(predecessor_model)
        self.reset_accepted()
        self.set_exclusions(exclusions)
        self.set_limit(limit)

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

        if model0 != VIRTUAL_INITIAL_MODEL and not model1.petab_yaml.samefile(
            model0.petab_yaml
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
            parameters0 = np.array(
                model0.get_parameter_values(parameter_ids=parameter_ids)
            )

        parameters1 = np.array(
            model1.get_parameter_values(parameter_ids=parameter_ids)
        )

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
        predecessor_model: Optional[Union[Model, str]] = None,
        **kwargs,
    ):
        # Although `VIRTUAL_INITIAL_MODEL` is `str` and can be used as a default
        # argument, `None` may be passed by other packages, so the default value
        # is handled here instead.
        if predecessor_model is None:
            predecessor_model = VIRTUAL_INITIAL_MODEL
        super().__init__(*args, predecessor_model=predecessor_model, **kwargs)

    def is_plausible(self, model: Model) -> bool:
        distances = self.distances_in_estimated_parameters(model)
        unsigned_size = self.direction * distances['size']
        # A model is plausible if the number of estimated parameters strictly
        # increases (or decreases, if `self.direction == -1`), and no
        # previously estimated parameters become fixed.
        if self.predecessor_model == VIRTUAL_INITIAL_MODEL or (
            unsigned_size > 0 and distances['l1'] == unsigned_size
        ):
            return True
        return False

    def distance(self, model: Model) -> int:
        # TODO calculated here and `is_plausible`. Rewrite to only calculate
        #      once?
        distances = self.distances_in_estimated_parameters(model)
        return distances['l1']

    def _consider_method(self, model) -> bool:
        """See `CandidateSpace._consider_method`."""
        distance = self.distance(model)

        # Get the distance of the current "best" plausible model(s)
        distance0 = np.inf
        if self.distances:
            distance0 = one(set(self.distances))
            # TODO store each or just one?
            # distance0 = one(self.distances)
        # breakpoint()

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


class BidirectionalCandidateSpace(ForwardCandidateSpace):
    """The bidirectional method class.

    Attributes:
        method_history:
            The history of models that were found at each search.
            A list of dictionaries, where each dictionary contains keys for the `METHOD`
            and the list of `MODELS`.
    """

    method = Method.BIDIRECTIONAL
    retry_model_space_search_if_no_models = True

    def __init__(
        self,
        *args,
        initial_method: Method = Method.FORWARD,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # FIXME cannot access from CLI
        self.initial_method = initial_method

        self.history: List[Dict[str, Union[Method, List[Model]]]] = []

    def update_method(self, method: Method):
        if method == Method.FORWARD:
            self.direction = 1
        elif method == Method.BACKWARD:
            self.direction = -1
        else:
            raise NotImplementedError(
                f'Bidirectional direction must be either `Method.FORWARD` or `Method.BACKWARD`, not {method}.'
            )

        self.method = method

    def switch_method(self):
        if self.method == Method.FORWARD:
            method = Method.BACKWARD
        elif self.method == Method.BACKWARD:
            method = Method.FORWARD

        self.update_method(method=method)

    def setup_before_model_subspaces_search(self):
        # If previous search found no models, then switch method.
        previous_search = None if not self.history else self.history[-1]
        if previous_search is None:
            self.update_method(self.initial_method)
            return

        self.update_method(previous_search[METHOD])
        if not previous_search[MODELS]:
            self.switch_method()
            self.retry_model_space_search_if_no_models = False

    def setup_after_model_subspaces_search(self):
        current_search = {
            METHOD: self.method,
            MODELS: self.models,
        }
        self.history.append(current_search)
        self.method = self.governing_method

    def wrap_search_subspaces(self, search_subspaces):
        def wrapper():
            def iterate():
                self.setup_before_model_subspaces_search()
                search_subspaces()
                self.setup_after_model_subspaces_search()

            # Repeat until models are found or switching doesn't help.
            iterate()
            while (
                not self.models and self.retry_model_space_search_if_no_models
            ):
                iterate()

            # Reset flag for next time.
            self.retry_model_space_search_if_no_models = True

        return wrapper



#TODO: 
# - if method did not return a better results switch method.
# - specifiy initial model (Implement)
# - make it work, without swap, compare results (similar)
# - write is_plausible
# - write get_most_distant
# - write coming from global
# later : General constraints (critical, swap, something else...)

class FAMoSCandidateSpace(CandidateSpace):
    """The FAMoS method class.

    Attributes:
        method_history:
            The history of models that were found at each search.
            A list of dictionaries, where each dictionary contains keys for the `METHOD`
            and the list of `MODELS`.
    """

    method = Method.FAMOS
    default_method_swapping = {(Method.FORWARD, Method.FORWARD): Method.BACKWARD,
                               (Method.SWAP, Method.FORWARD) : Method.BACKWARD,
                               (Method.BACKWARD, Method.FORWARD): Method.SWAP,
                               (Method.FORWARD, Method.BACKWARD): Method.SWAP,
                               (Method.BACKWARD, Method.BACKWARD): Method.FORWARD,
                               (Method.SWAP, ): None}

    def __init__(
        self,
        *args,
        initial_method: Method = Method.FORWARD,
        crit_parms: List = [],
        swap_parms: List = [],
        reattempt: int = 0,
        method_swapping: Dict[tuple,str] = default_method_swapping,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # FIXME cannot access from CLI
        self.initial_method = initial_method
        self.method_history = [] #take last 2

        self.history: List[Dict[str, Union[Method, List[Model]]]] = [] #need different history, from method_caller
        
        # have inside them lists with indexes of crit and swap parms
        # e.g. [[1,6,7], [0,2]]
        #specified by the user, add them up
        self.crit_parms = crit_parms
        self.swap_parms = swap_parms
        self.reattempt = reattempt
        self.method_swapping = method_swapping
        
        if self.reattempt:
            self.most_distant_max_number = self.reattempt
        else:
            self.most_distant_max_number = 1
        
        self.inner_candidate_space = method_to_candidate_space_class(initial_method)(*args, **kwargs)
        self.found_new_best = True
        self.failed_global = False
        self.do_not_jump = False
        self.jump_run = 0

        #Construct all the candidate spaces needed
        ForwardCandidateSpace_instance = ForwardCandidateSpace(*args, **kwargs)
        BackwardCandidateSpace_instance = BackwardCandidateSpace(*args, **kwargs)
        #TODO rename LATERAL to SWAP when swap is going to be made
        LateralCandidateSpace_instance = LateralCandidateSpace(*args, **kwargs) 

    def check_if_should_switch_method(self):
        #TODO this changes the method right away, doesn't only check...
        if not self.found_new_best:
            self.switch_method() 

        #return not self.found_new_best
    
    def update_method(self, method: Method):
        if method == Method.FORWARD:
            self.direction = 1
        elif method == Method.BACKWARD:
            self.direction = -1
        if method not in [Method.FORWARD, Method.SWAP, Method.BACKWARD]:
            raise NotImplementedError(
                f'FAMoS direction must be either `Method.FORWARD` or `Method.BACKWARD` or `Method.SWAP`, not {method}.'
            ) #TODO add here in the error that it comes from wrong method_swapping scheme?
        
        self.method = method
        self.method_history.append(method)

    def update_from_local_history(..., local_history):
        self.best_models # update best models
        # update found new best

    def switch_method(self):
        previous = self.method #TODO is this necessary here?
        
        #iterate through the method_swapping dictionary to see which method to switch to
        #check if the previous_methods match the 
        for previous_methods in self.method_swapping:
            if previous_methods == tuple(self.method_history[-len(previous_methods):]):
                method = self.method_swapping[previous_methods]
                #if found a switch just break (choosing first good switch)
                break
        
        #raise error if the method didn't change
        if method==previous:
            raise ValueError("Method didn't switch when it had to. \
                              The method_swapping provided is not sufficient. \
                              Please provide a correct method_swapping scheme")

        # if the next method is None (in default case
        # if SWAP method didn't find any better models) then terminate
        if not self.method:
            if self.reattempt and not self.do_not_jump:
                self.jump_to_most_distant()
                return
            else:
                raise StopIteration("No valid models found.") #TODO Should I do this?
        
        #If we try to switch to SWAP method but it's not available (no crit or swap parameter groups)
        if method == Method.SWAP and not self.crit_parms and not self.swap_parms:
            if self.reattempt and not self.do_not_jump:
                self.jump_to_most_distant()
                return
            else:
                raise StopIteration("No valid models found.") #TODO Should I do this?

        self.update_method(method=method)

    def setup_before_next_model_subspaces_search(self):
        # If current search found no models, then switch method for the next one.
        previous_search = None if not self.history else self.history[-1]
        if previous_search is None:
            self.update_method(self.initial_method)
            return

        self.update_method(previous_search[METHOD]) #FIXME
        # teminate if method found no valid neighbours
        if not previous_search[MODELS]:
            self.terminate_if_found_no_neighbors = True
            return

        # if we came to the null model with the backward method
        # change method to forward
        #TODO famos implements terminate if visited global, 
        # if previous_search[MODELS].sum()==0:
        #     self.previous = Method.BACKWARD
        #     self.method = Method.FORWARD
        #     # if we have come down from the superset model then terminate
        #     if self.failed_global:
        #         self.terminate_if_found_no_neighbors = True
        #     return


    def setup_after_model_subspaces_search(self):
        current_search = {
            METHOD: self.method,
            MODELS: self.models
            #CRITERION: self.criterion add for find furthest?
        }
        self.history.append(current_search) #What is this history?
        self.previous = self.governing_previous #QUES what is this?
        self.method = self.governing_method


    def jump_to_most_distant(self):
        # if we jumped already, we check if we arrived to the same "best" model again
        if(self.jump_run > 0):
            if(self.history[-1][MODELS] == "equal to last optimal"):
                self.do_not_jump = True
                return
            #maybe print if we're better or worse than the last "best" QUESTION
        #do we add do_not_fit? QUESTION, does that exist even, or do we deal with only do_fit parameters?
        new_init_model = self.get_most_distant()
        
        #check if the new_init_model is [0, 0, ..., 0], no untested model found
        if(new_init_model.sum()==0):
            self.do_not_jump = True
            return

        # if model not appropriate make it so by adding first crit parms
        if not self.is_appropriate(new_init_model):
            for i in len(self.crit_parms):
                new_init_model[self.crit_parms[i][0]] = 1
        
        #THIS is in their code SWAP, but for us doesn't have to be with arbitrary method swapping
        self.previous = self.method  
        self.method = Method.FORWARD

        self.jump_run += 1
        self.terminate = False

        self.models = [new_init_model] #TODO check if this is good
        return

    def get_most_distant(self) -> Model:
        #need history of last self.most_distant_max_number runs with their criteria result
        #sort history by criteria
        #get complement of all or last self.most_distant_max_number models
        #calculate L_\inf distance (abs of difference basically to all tested models and get min)
        #get the model with biggest L_\inf
        return "most distant"

    def is_plausible(self, model: Model) -> bool:
        if self.checkcritical(): 
            return self.currentcandidatespace.is_plausible()
        return False

# TODO rewrite so BidirectionalCandidateSpace inherits from ForwardAndBackwardCandidateSpace
#      instead
class ForwardAndBackwardCandidateSpace(BidirectionalCandidateSpace):
    method = Method.FORWARD_AND_BACKWARD
    governing_method = Method.FORWARD_AND_BACKWARD
    retry_model_space_search_if_no_models = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, initial_method=None)

    def wrap_search_subspaces(self, search_subspaces):
        def wrapper():
            for method in [Method.FORWARD, Method.BACKWARD]:
                self.update_method(method=method)
                search_subspaces()
                self.setup_after_model_subspaces_search()

        return wrapper

    # Disable unused interface
    setup_before_model_subspaces_search = None
    switch_method = None

    def setup_after_model_space_search(self):
        pass


class LateralCandidateSpace(ForwardCandidateSpace):
    """Find models with the same number of estimated parameters."""

    method = Method.LATERAL

    def __init__(
        self,
        *args,
        predecessor_model: Union[Model, str],
        **kwargs,
    ):
        super().__init__(*args, predecessor_model=predecessor_model, **kwargs)

    def is_plausible(self, model: Model) -> bool:
        distances = self.distances_in_estimated_parameters(model)
        # A model is plausible if the number of estimated parameters remains
        # the same, but some estimated parameters have become fixed and vice
        # versa.
        if (
            distances['size'] == 0
            and
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
        # if args or kwargs:
        #    # FIXME remove?
        #    # FIXME at least support limit
        #    warnings.warn(
        #        'Arguments were provided but will be ignored, because of the '
        #        'brute force candidate space.'
        #    )
        super().__init__(*args, **kwargs)

    def _consider_method(self, model):
        return True


candidate_space_classes = [
    ForwardCandidateSpace,
    BackwardCandidateSpace,
    BidirectionalCandidateSpace,
    LateralCandidateSpace,
    BruteForceCandidateSpace,
    ForwardAndBackwardCandidateSpace,
    FAMoSCandidateSpace
]


def method_to_candidate_space_class(method: Method) -> str:
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
        f'The provided method name {method} does not correspond to an implemented candidate space.'
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
