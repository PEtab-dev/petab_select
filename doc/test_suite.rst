Model selection test suite
==========================

Several test cases are provided, to test the compatibility of a
PEtab-compatible calibration tool with different PEtab Select features.

The test cases are available in the ``test_cases`` directory, and are provided in
the model format.

.. list-table::
   :header-rows: 1

   * - Test ID
     - Criterion
     - Method
     - Model space files
     - Compressed format
     - Predecessor (initial) models files
   * - 0001
     - (all)
     - (only one model)
     - 1
     -
     -
   * - 0002 [#f1]_
     - AIC
     - forward
     - 1
     -
     -
   * - 0003
     - BIC
     - brute force
     - 1
     - Yes
     -
   * - 0004
     - AICc
     - backward
     - 1
     -
     -
   * - 0005
     - AIC
     - forward
     - 1
     -
     - 1
   * - 0006
     - AIC
     - forward
     - 1
     -
     -
   * - 0007 [#f2]_
     - AIC
     - forward
     - 1
     -
     -
   * - 0008 [#f2]_
     - AICc
     - backward
     - 1
     -
     -
   * - 0009 [#f3]_
     - AICc
     - FAMoS
     - 1
     - Yes
     - Yes

.. [#f1] Model ``M1_0`` differs from ``M1_1`` in three parameters, but only 1 additional estimated parameter. The effect of this on model selection criteria needs to be clarified. Test case 0006 is a duplicate of 0002 that doesn't have this issue.

.. [#f2] Noise parameter is removed, noise is fixed to ``1``.

.. [#f3] This is a computationally expensive problem to solve. Developers can try a model selection initialized with the provided predecessor model, which is a model start that reproducibly finds the expected model. To solve the problem reproducibly *ab initio*, on the order of 100 random model starts are required. This test case reproduces the model selection problem presented in https://doi.org/10.1016/j.cels.2016.01.002.
