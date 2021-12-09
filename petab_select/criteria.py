"""Implementations of model selection criteria."""

import math

import petab
from petab.C import OBJECTIVE

from .constants import (
    #LH,
    #LLH,
    #NLLH,
    Criterion,
    PETAB_PROBLEM,
    TYPE_CRITERION,
)
#from .model import Model



# use as attribute e.g. `Model.criterion_computer`?
class CriterionComputer():
    """Compute various criterion."""

    def __init__(
        self,
        model: 'petab_select.Model',
    ):
        self.model = model
        # TODO refactor, if `petab_problem` is going to be produced here anyway, store
        #      in model instance instead, for use elsewhere (e.g. pyPESTO)
        self.petab_problem = model.to_petab()[PETAB_PROBLEM]

    def __call__(self, criterion: Criterion) -> float:
        """Get a criterion value.

        Args:
            criterion:
                The ID of the criterion.

        Returns:
            The criterion value.
        """
        return getattr(self, 'get_'+criterion.value.lower())()

    def get_aic(self) -> float:
        """Get the Akaike information criterion."""
        return calculate_aic(
            nllh=self.get_nllh(),
            n_estimated=self.get_n_estimated(),
        )

    def get_aicc(self) -> float:
        """Get the corrected Akaike information criterion."""
        return calculate_aicc(
            nllh=self.get_nllh(),
            n_estimated=self.get_n_estimated(),
            n_measurements=self.get_n_measurements(),
            n_priors=self.get_n_priors(),
        )

    def get_bic(self) -> float:
        """Get the Bayesian information criterion."""
        return calculate_bic(
            nllh=self.get_nllh(),
            n_estimated=self.get_n_estimated(),
            n_measurements=self.get_n_measurements(),
            n_priors=self.get_n_priors(),
        )

    def get_nllh(self) -> float:
        """Get the negative log-likelihood."""
        nllh = self.model.get_criterion(Criterion.NLLH, compute=False)
        if nllh is None:
            nllh = -1 * self.get_llh()
        return nllh

    def get_llh(self) -> float:
        """Get the log-likelihood."""
        llh = self.model.get_criterion(Criterion.LLH, compute=False)
        if llh is None:
            llh = math.log(self.get_lh())
        return llh

    def get_lh(self) -> float:
        """Get the likelihood."""
        lh = self.model.get_criterion(Criterion.LH, compute=False)
        llh = self.model.get_criterion(Criterion.LLH, compute=False)
        nllh = self.model.get_criterion(Criterion.NLLH, compute=False)

        if lh is not None:
            return lh
        elif llh is not None:
            return math.exp(llh)
        elif nllh is not None:
            return math.exp(-1 * nllh)

        raise ValueError('Please supply the likelihood (LH) or a compatible transformation. Compatible transformations: log(LH), -log(LH).')  # noqa: E501

    def get_n_estimated(self) -> int:
        """Get the number of estimated parameters."""
        return len(self.petab_problem.x_free_indices)

    def get_n_measurements(self) -> int:
        """Get the number of measurements."""
        # TODO remove `count_nan` from method.
        return petab.measurements.get_n_measurements(
            measurement_df=self.petab_problem.measurement_df,
            count_nan=False,
        )

    def get_n_priors(self) -> int:
        """Get the number of priors."""
        return len(petab.parameters.get_priors_from_df(
            parameter_df=self.petab_problem.parameter_df,
            mode=OBJECTIVE,
        ))

    #def get_criterion(self, id: str) -> TYPE_CRITERION:
    #    """Get a criterion value, by criterion ID.
    #    FIXME: superseded by `__call__`

    #    id:
    #        The ID of the criterion (e.g. `petab_select.constants.Criterion.AIC`).

    #    Returns:
    #        The criterion value.
    #    """
    #    return getattr(self, f'get_{id}')()


# TODO should fixed parameters count as measurements/priors when comparing to models
#      that estimate the same parameters?
def calculate_aic(
    nllh: float,
    n_estimated: int,
) -> float:
    """
    Calculate the Akaike information criterion (AIC) for a model.

    Args:
        nllh:
            The negative log likelihood.
        n_estimated:
            The number of estimated parameters in the model.

    Returns:
        The AIC value.
    """
    return 2*(n_estimated + nllh)


def calculate_aicc(
    nllh: float,
    n_estimated: int,
    n_measurements: int,
    n_priors: int,
) -> float:
    """
    Calculate the corrected Akaike information criterion (AICc) for a model.

    Args:
        nllh:
            The negative log likelihood.
        n_estimated:
            The number of estimated parameters in the model.
        n_measurements:
            The number of measurements used in the likelihood.
        n_priors:
            The number of priors used in the objective function.

    Returns:
        The AICc value.
    """
    return (
        calculate_aic(n_estimated, nllh)
        + 2*n_estimated*(n_estimated + 1)
        / (n_measurements + n_priors - n_estimated - 1)
    )


def calculate_bic(
    nllh: float,
    n_estimated: int,
    n_measurements: int,
    n_priors: int,
):
    """
    Calculate the Bayesian information criterion (BIC) for a model.

    Args
        nllh:
            The negative log likelihood.
        n_estimated:
            The number of estimated parameters in the model.
        n_measurements:
            The number of measurements used in the likelihood.
        n_priors:
            The number of priors used in the objective function.

    Returns:
        The BIC value.
    """
    return n_estimated*math.log(n_measurements + n_priors) + 2*nllh
