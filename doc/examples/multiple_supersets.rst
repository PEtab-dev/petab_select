.. _multiple_supersets:

Multiple superset models
========================

The PEtab Select model space table is designed to express large model spaces based on a single superset model, concisely. However, model selection problem can also involve multiple superset models that are structurally different. For example, one superset may be a linear model, and the other superset may be an ordinary differential equation (ODE) model. To express this concisely, we suggest using two model space tables; one per superset.

We illustrate this with two of the other examples in this documentation, specifically:

- :ref:`ode_timeseries_example`: an ODE model with parameters ``k1`` through ``k3``, and
- :doc:`examples/other_model_types`: a polynomial model with parameters ``k1`` through ``k5``.

Example file layout
-------------------

The PEtab Select problem YAML file and the two model space files are kept in a top directory. The PEtab problems for each superset model are kept in separate directories.

.. code-block:: text

   model_selection_problem/
   ├── petab_select_problem.yaml
   ├── model_space_polynomial.tsv
   ├── model_space_ode_timeseries.tsv
   ├── polynomial_petab_files/
   │   └── ...
   └── ode_timeseries_petab_files/
       └── ...

The PEtab Select problem YAML
-----------------------------

The PEtab Select problem YAML lists both model space files.

.. code-block:: yaml

   format_version: 1.0.0
   criterion: AIC
   method: brute_force
   model_space_files:
   - model_space_polynomial.tsv
   - model_space_ode_timeseries.tsv

The model space tables
----------------------

The model space can be defined in a single model space table; however, this can lead to sparse tables. Hence, we suggest splitting the table into one per superset.

``model_space_polynomial.tsv``:

.. list-table::
   :header-rows: 1

   * - model_subspace_id
     - model_subspace_petab_yaml
     - k0
     - k1
     - k2
     - k3
     - k4
     - k5
   * - polynomial
     - polynomial_petab_files/problem.yaml
     - 0;estimate
     - 0;estimate
     - 0;estimate
     - 0;estimate
     - 0;estimate
     - 0;estimate


``model_space_ode_timeseries.tsv``:

.. list-table::
   :header-rows: 1

   * - model_subspace_id
     - model_subspace_petab_yaml
     - k1
     - k2
     - k3
   * - ode_timeseries
     - ode_timeseries_petab_files/petab_problem.yaml
     - 0;estimate
     - 0;estimate
     - 0;estimate

Together, these two model spaces define :math:`2^6 + 2^3 = 72` candidate models.

Notes
-----

- **Method** The stepwise methods (``forward``, ``backward``)
  move between models will not visit model subspaces with a different ``model_subspace_petab_yaml`` than the
  initial model. Hence, multiple stepwise searches initialized in the model subspaces associated with each superset model are required to explore the full space. ``brute_force`` will search all models regardless.
- **Criteria** The information criteria (AIC etc.) require that models are compared on the same dataset. Hence, the PEtab problems inside ``polynomial_petab_files/`` and ``ode_timeseries_petab_files/`` should use the same data.
