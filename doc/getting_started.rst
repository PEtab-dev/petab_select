.. _getting_started:

Getting started
===============

Designing the PEtab and PEtab Select problem files
--------------------------------------------------
Please see to the PEtab documentation for a `guide on how to create PEtab files to describe the parameter estimation problem for a single model <https://petab.readthedocs.io/en/latest/v1/tutorial/tutorial.html>`__.

Then see one of the examples for users in our documentation, for creating the PEtab Select files.

Choice of criteria
------------------
Choice of criteria can be a philosophical or problem-specific question. The AIC tends to select larger models as the data set size increases, unlike the BIC. The criteria implemented in PEtab Select can also often produce similar results. As there is no "correct" choice here, we make no recommendation here. More information about the criteria can be found in [BA2002]_.

Choice of method
----------------
In general, if ``brute-force`` is computationally feasible, then we recommend it. Otherwise, we recommend the ``famos`` method. This is because more of the model space is reachable, than with classical ``backward`` or ``forward`` searches. However, FAMoS itself is primarily a sequence of local searches, which can result in longer runtimes in practice, compared to forward or backward searches. As the ``lateral`` method cannot reach models that differ in size to the initial model, we recommend only using it via FAMoS.

Regardless of choice of method, analysis can then be performed with only the "best" model, or alternatively with an ensemble of all "good" models to perform multi-model inference, as described in [BA2002]_.

Complete tutorials
------------------
Complete examples are provided in the documentation of each of the supported :ref:`calibration_tools`. If you want to use an alternative calibration tool, then see :doc:`examples/other_model_types`.

References
----------

.. [BA2002] Burnham, K.P. and Anderson, D.R. *Model Selection and Multimodel Inference*.
   Springer, New York, 2002.
   https://doi.org/10.1007/b97636
