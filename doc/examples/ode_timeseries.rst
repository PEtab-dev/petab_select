.. _ode_timeseries_example:

Example: ODE with timeseries data
=================================

In this example, we provide the PEtab and PEtab Select files for a model selection problem involving an ordinary differential equation (ODE) model and timeseries data. `All files are available for download <https://github.com/PEtab-dev/petab_select/tree/main/doc/examples/ode_timeseries>`_.


The ODE model
-------------

The model is describes the dynamics of two biochemical species: ``x1`` and ``x2``. There are three hypothetical reactions:

1. import of ``x1`` at a constant rate :math:`k_1` (:math:`\varnothing \rightarrow x_1`),
2. conversion of ``x1`` into ``x2`` at rate :math:`k_2 x_1` (:math:`x_1 \rightarrow x_2`), and
3. export of both species through their interaction at rate :math:`k_3 x_1 x_2` (:math:`x_1 + x_2 \rightarrow \varnothing`).

Assuming mass-action kinetics, the the ordinary differential equation (ODE) model is

.. math::

   \frac{\mathrm{d}x_1}{\mathrm{d}t} &= k_1 - k_2 x_1 - k_3 x_1 x_2, \\
   \frac{\mathrm{d}x_2}{\mathrm{d}t} &= k_2 x_1 - k_3 x_1 x_2,

where :math:`t` is time.  The initial condition is :math:`x_1(0) = x_2(0) = 0`.

Hence, the three hypothetical reactions are each controlled by a single parameter (:math:`k_1`, :math:`k_2`, or :math:`k_3`) and the model selection problem is to identify the raections (parameters) that are sufficient to explain the data.

In the files, the model is encoded in SBML, named ``model.xml``.

The data
--------

The (synthetic) experiment was conducted by collecting measurements of :math:`x_2` at six time points.

The PEtab measurement table is:

.. csv-table::
   :file: ode_timeseries/measurements.tsv
   :delim: tab
   :header-rows: 1

The observable model
--------------------

As :math:`x_2` is observed with measurement noise, we include a noise model.

The observable is

.. math::

   y_{\text{obs\_x2}} = x_2.

The measurements are assumed to be the observable plus some Gaussian-distributed noise:

.. math::

   \bar{y} = y_{\text{obs\_x2}} + \varepsilon, \qquad
   \varepsilon \sim \mathcal{N}(0, \sigma_{x2}^2),

where the Gaussian standard deviation :math:`\sigma_{x2}` is an estimated parameted.
This Gaussian noise model is what defines the likelihood that PEtab Select uses
to compute model selection criteria.

The PEtab observable table is:

.. csv-table::
   :file: ode_timeseries/observables.tsv
   :delim: tab
   :header-rows: 1

The estimated parameters
------------------------

Each of the estimated parameters are specified in the PEtab parameters table.

.. csv-table::
   :file: ode_timeseries/parameters.tsv
   :delim: tab
   :header-rows: 1

This table is adapted by PEtab Select to represent the different models in the model space. For example, the model ``M1_0`` in the model space table does not estimate :math:`k_2` and instead sets it to ``0``. PEtab Select implements this by changing the ``nominalValue`` of ``k2`` to ``0``, and turning off parameter estimation for ``k2`` by setting ``estimate`` to ``0``.

The model selection problem
---------------------------

The PEtab parameter table (``parameters.tsv``) contains the rate constants
:math:`k_1, k_2, k_3` (and the noise parameter :math:`\sigma_{x2}`, which is
always estimated).

The model selection problem is to determine which of the parameters :math:`k_1`, :math:`k_2`, and :math:`k_3` should be estimated in order to explain the data well.

This is encoded in the model space file. Each row is a
candidate model (a model subspace), and the columns ``k1``, ``k2``, ``k3`` give
the value each parameter takes: either a fixed value, or ``estimate`` to indicate that the parameter should be estimated.

.. csv-table::
   :file: ode_timeseries/model_space.tsv
   :delim: tab
   :header-rows: 1

For example, ``M1_0`` fixes all three rate constants to 0 (in this model, this means the associated reactions are turn off),
``M1_7`` estimates all three, and the middle rows encode models that estimate different subsets. When a parameter is not estimated, it is fixed
to a value (e.g. ``k1 = 0.2``, ``k2 = 0.1``, ``k3 = 0``).

In this example, we implement a single model per model subspace. A more concise definition of the model space is presented in :doc:`other_model_types`.

The PEtab Select problem YAML combines all files and remaining information required to define the model selection problem.

.. literalinclude:: ode_timeseries/petab_select_problem.yaml
   :language: yaml

Here, models are compared by the Akaike information criterion (``AIC``), and the
model space is explored with the ``forward`` method — starting from the
smallest model and iteratively adding estimated parameters as long as the AIC
improves.
