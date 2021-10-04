"""Implementations of model selection criteria."""

import math

import petab
from petab.C import OBJECTIVE

from .constants import (
    LH,
    LLH,
    NLLH,
)
from .model import Model


# use as attribute e.g. `Model.criterion_calculator`?
class CriteriaCalculator():
    """Compute various criterion."""

    def __init__(
        self,
        model: Model,
    ):
        self.model = model
        self.petab_problem, _ = model.to_petab()

    def __call__(self, criterion: str) -> float:
        """Get a criterion value.

        Args:
            criterion:
                The ID of the criterion.

        Returns:
            The criterion value.
        """
        return getattr(self, criterion.lower())()

    def get_aic(self) -> float:
        """Get the Akaike information criterion."""
        return calculate_aic(
            nllh=self.get_nllh(),
            n_estimated=self.get_n_estimated(),
        )

    def get_aicc(self) -> float:
        """Get the corrected Akaike information criterion."""
        return calculate_aic(
            nllh=self.get_nllh(),
            n_estimated=self.get_n_estimated(),
            n_measurements=self.get_n_measurements(),
            n_priors=self.get_n_priors(),
        )

    def get_bic(self) -> float:
        """Get the Bayesian information criterion."""
        return calculate_aic(
            nllh=self.get_nllh(),
            n_estimated=self.get_n_estimated(),
            n_measurements=self.get_n_measurements(),
            n_priors=self.get_n_priors(),
        )

    def get_nllh(self) -> float:
        """Get the negative log-likelihood."""
        nllh = self.model.get_criterion(NLLH)
        if nllh is None:
            nllh = -1 * self.get_llh()
        return nllh

    def get_llh(self) -> float:
        """Get the log-likelihood."""
        llh = self.model.get_criterion(LLH)
        if llh is None:
            llh = math.log(self.get_lh())
        return llh

    def get_lh(self) -> float:
        """Get the likelihood."""
        lh = self.model.get_criterion(LH)
        llh = self.model.get_criterion(LLH)
        nllh = self.model.get_criterion(NLLH)

        if lh is not None:
            return lh
        elif llh is not None:
            return math.exp(llh)
        elif nllh is not None:
            return math.exp(-1 * nllh)

        raise ValueError(
            'Please supply the likelihood or a compatible transformation.'
        )

    def get_n_estimated(self) -> int:
        """Get the number of estimated parameters."""
        return len(self.petab_problem.x_free_indices)

    def get_n_measurements(self) -> int:
        """Get the number of measurements."""
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


# TODO should fixed parameters count as measurements when comparing to models
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
        The AIC.
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
        negative_log_likelihood:
            The negative log likelihood.
        n_estimated:
            The number of estimated parameters in the model.
        n_measurements:
            The number of measurements used in the likelihood.
            FIXME: e.g.: `len(petab_problem.measurement_df)`.
        n_priors:
            The number of priors used in the objective function.
            FIXME: e.g.: `len(pypesto_problem.x_priors._objectives)`.
            FIXME: remove?

    Returns:
        The corrected AIC.
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
        n_estimated:
            The number of estimated parameters in the model.
        nllh:
            The negative log likelihood,
            e.g.: the `optimize_result.list[0]['fval']` attribute of the object
            returned by `pypesto.minimize`.
        n_measurements:
            The number of measurements used in the objective function of the
            model.
            e.g.: `len(petab_problem.measurement_df)`.
        n_priors:
            The number of priors used in the objective function of the model,
            e.g.: `len(pypesto_problem.x_priors._objectives)`.
            TODO make public property for number of priors in objective? or in
                 problem, since `x_priors` == None is possible.

    Returns:
        The BIC.
    """
    return n_estimated*math.log(n_measurements + n_priors) + 2*nllh


# def convert_likelihood(
#     lh: float = None,
#     llh: float = None,
#     nllh: float = None,
# ) -> float:
#     """Convert transformations of the likelihood into other transformations.
#
#     Exactly one transformation of the likelihood should be supplied to this
#     method.
#
#     Args:
#         lh:
#             The likelihood.
#         llh:
#             The log likelihood.
#         nllh:
#             The negative log likelihood.
#
#     Returns:
#         All transformations of the likelihood, described by the arguments.
#     """
#     if sum(1 if value is not None else 0 for value in [lh, llh, nllh]) != 1:
#         raise ValueError(
#             'Please supply (only) one of the arguments to this method.'
#         )
#
#     _lh = None
#     _llh = None
#     _nllh = None
#     if llh is not None:
#         _llh = llh
#         _lh = math.exp(llh)
#     if nllh is not None:
#         _nllh = nllh
#         _lh = math.exp(-1 * nllh)
