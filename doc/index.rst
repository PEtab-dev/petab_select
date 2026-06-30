Welcome to PEtab Select's documentation!
========================================

PEtab Select brings
`model selection <https://en.wikipedia.org/wiki/Model_selection>`_ to
`PEtab <https://petab.readthedocs.io/>`_. PEtab Select comprises file
formats, a Python package and a command line interface.

Model selection is the process of choosing the best model from a set of
candidate models. PEtab Select provides a standardized and compact way to
specify the candidate model space, implements a number of model selection
algorithms and criteria.

Supported model selection algorithms:

* brute force
* `forward selection <https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches>`_
* `backward selection <https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches>`_
* `FAMoS <https://doi.org/10.1371/journal.pcbi.1007230>`_

Supported model selection criteria:

* (`corrected <https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size>`_)
  `Akaike Information Criterion <https://en.wikipedia.org/wiki/Akaike_information_criterion#Definition>`_ (AIC / AICc)
* `Bayesian Information Criterion <https://en.wikipedia.org/wiki/Bayesian_information_criterion#Definition>`_ (BIC)

Model calibration is performed outside of PEtab Select. For example,
PEtab Select is well-integrated with:

* `BasiCO <https://basico.readthedocs.io/>`_
  (`example <https://basico.readthedocs.io/en/latest/notebooks/Working_with_PEtab.html#Model-Selection>`__)
* `Data2Dynamics <https://github.com/Data2Dynamics/d2d>`_
  (`example <https://github.com/Data2Dynamics/d2d/wiki/Model-selection-with-PEtab-Select>`__)
* `PEtab.jl <https://sebapersson.github.io/PEtab.jl>`_
  (`example <https://sebapersson.github.io/PEtab.jl/stable/pest_select/>`__)
* `pyPESTO <https://pypesto.readthedocs.io/>`_
  (`example <https://pypesto.readthedocs.io/en/latest/example/model_selection.html>`__)

Other model calibration tools can easily be integrated using the provided
Python package or command line interface. An example of this is provided for the Python package statsmodels at :doc:`examples/other_model_types`.

For users, in most
cases, model selection is performed entirely via one of the calibration tools,
rather than the PEtab Select package directly. Hence, we recommend reading the `documentation from your calibration tool on model selection <calibration_tools>`_.
After model selection, the
analysis and visualization methods in the PEtab Select package can be used
directly with the results from your calibration tool.

For developers, we provide some examples of how integrate model selection with PEtab Select into unsupported calibration tools via the CLI and Python interfaces of the PEtab Select package.

Installation
------------

The Python 3 package provides both the Python 3 and command-line (CLI)
interfaces, and can be installed from PyPI, with:

.. code-block:: bash

    pip install petab-select


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Getting started <getting_started>
   problem_definition
   examples
   Calibration tools <calibration_tools>
   analysis
   Test Suite <test_suite>
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
