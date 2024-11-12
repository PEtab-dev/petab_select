"""Implementations of model selection criteria."""

import numpy as np
import petab.v1 as petab
from petab.v1.C import OBJECTIVE_PRIOR_PARAMETERS, OBJECTIVE_PRIOR_TYPE

import petab_select

from .constants import PETAB_PROBLEM, Criterion  # LH,; LLH,; NLLH,

__all__ = [
    "calculate_aic",
    "calculate_aicc",
    "calculate_bic",
    "CriterionComputer",
]


# use as attribute e.g. `Model.criterion_computer`?
class CriterionComputer:
    """Compute various criteria."""

    def __init__(
        self,
        model: "petab_select.model.Model",
    ):
        self.model = model
        self._petab_problem = None

    @property
    def petab_problem(self) -> petab.Problem:
        """The PEtab problem that corresponds to the model.

        Implemented as a property such that the :class:`petab.Problem` object
        is only constructed if explicitly requested.

        Improves speed of operations on models by a lot. For example, analysis of models
        that already have criteria computed can skip loading their PEtab problem again.
        """
        # TODO refactor, if `petab_problem` is going to be produced here anyway, store
        #      in model instance instead, for use elsewhere (e.g. pyPESTO)
        #      i.e.: this is a property of a `Model` instance, not `CriterionComputer`
        if self._petab_problem is None:
            self._petab_problem = self.model.to_petab()[PETAB_PROBLEM]
        return self._petab_problem

    def __call__(self, criterion: Criterion) -> float:
        """Get a criterion value.

        Args:
            criterion:
                The ID of the criterion.

        Returns:
            The criterion value.
        """
        return getattr(self, "get_" + criterion.value.lower())()

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
            llh = np.log(self.get_lh())
        return llh

    def get_lh(self) -> float:
        """Get the likelihood."""
        lh = self.model.get_criterion(Criterion.LH, compute=False)
        llh = self.model.get_criterion(Criterion.LLH, compute=False)
        nllh = self.model.get_criterion(Criterion.NLLH, compute=False)

        if lh is not None:
            return lh
        elif llh is not None:
            return np.exp(llh)
        elif nllh is not None:
            return np.exp(-1 * nllh)

        raise ValueError(
            "Please supply the likelihood (LH) or a compatible transformation. Compatible transformations: log(LH), -log(LH)."
        )

    def get_n_estimated(self) -> int:
        """Get the number of estimated parameters."""
        return len(self.petab_problem.x_free_indices)

    def get_n_measurements(self) -> int:
        """Get the number of measurements."""
        return len(self.petab_problem.measurement_df)

    def get_n_priors(self) -> int:
        """Get the number of priors."""
        df = self.petab_problem.parameter_df

        # At least one of the objective prior columns should be present.
        if not (
            OBJECTIVE_PRIOR_TYPE in df or OBJECTIVE_PRIOR_PARAMETERS in df
        ):
            return 0

        # If both objective prior columns are not present, raise an error.
        if not (
            OBJECTIVE_PRIOR_TYPE in df and OBJECTIVE_PRIOR_PARAMETERS in df
        ):
            raise NotImplementedError(
                "Currently expect that prior types are specified with prior parameters (no default values). Please provide an example for implementation."
            )

        # Expect that the number of non-empty values in both objective prior columns
        # are the same.
        if not (
            df[OBJECTIVE_PRIOR_TYPE].notna().sum()
            == df[OBJECTIVE_PRIOR_PARAMETERS].notna().sum()
        ):
            raise NotImplementedError(
                "Some objective prior values are missing."
            )

        number_of_priors = df[OBJECTIVE_PRIOR_TYPE].notna().sum()
        return number_of_priors

    # def get_criterion(self, id: str) -> TYPE_CRITERION:
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
    return 2 * (n_estimated + nllh)


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
    return calculate_aic(
        nllh=nllh, n_estimated=n_estimated
    ) + 2 * n_estimated * (n_estimated + 1) / (
        n_measurements + n_priors - n_estimated - 1
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
    return n_estimated * np.log(n_measurements + n_priors) + 2 * nllh
