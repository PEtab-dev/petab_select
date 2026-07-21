Example: metaparameters
=======================

Metaparameters are a PEtab Select feature that enables model selection with
model hypotheses that involve multiple parameters.

An example use case would be a model selection problem where one hypothesis is that
a process with a Michaelis-Menten kinetic exists. Since the Michaelis-Menten kinetic :math:`\frac{V_{\mathrm{max}}\, s}{K_m + s}` (parameters :math:`V_{\mathrm{max}}` and :math:`K_m`, and state variable :math:`s`) involves two parameters, both need to be set to be estimated simultaneously to include the process in the model, or set to ``0`` simultaneously to turn the process off and ensure the correct number of parameters are counted when computing model selection criteria such as the AIC.

We show an example of this below. As metaparameters only affect the PEtab Select files, we only show those.

Without metaparameters
----------------------

As ``Vmax`` and ``Km`` need to be toggled on or off simultaneously, without metaparameters, we require two model space rows.

.. list-table::
   :header-rows: 1

   * - model_subspace_id
     - model_subspace_petab_yaml
     - k1
     - Vmax
     - Km
     - k3
   * - M_mm_off
     - petab_problem.yaml
     - 0;estimate
     - 0
     - 0
     - 0;estimate
   * - M_mm_on
     - petab_problem.yaml
     - 0;estimate
     - estimate
     - estimate
     - 0;estimate

With metaparameters
-------------------

Instead, we can define a metaparameter ``mm`` in the problem YAML, which represents both ``Vmax`` and ``Km``.

.. code-block:: yaml

   format_version: 1.0.0
   criterion: AIC
   method: forward
   model_space_files:
   - model_space.tsv
   metaparameters:
     mm:
     - Vmax
     - Km

This enables concise specification in the model space table. ``Vmax`` and ``Km`` are assigned the same value as ``mm``.

.. list-table::
   :header-rows: 1

   * - model_subspace_id
     - model_subspace_petab_yaml
     - k1
     - mm
     - k3
   * - M
     - petab_problem.yaml
     - 0;estimate
     - 0;estimate
     - 0;estimate

Notes
-----

During model selection steps (e.g. with the ``forward`` and ``backward`` methods), a metaparameter is treated as a single parameter. This enables model selection in terms of selection of model hypotheses, rather than the individual parameters associated with hypotheses.

However, during computation of criteria such as the AIC, then the correct number of estimated parameters is used, i.e., with ``Vmax`` and ``Km`` in the example above if ``mm_R2`` is set to be estimated.
