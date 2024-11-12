.. petab-select documentation master file, created by
   sphinx-quickstart on Mon Oct 23 09:01:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to petab-select's documentation!
========================================

PEtab Select brings
`model selection <https://en.wikipedia.org/wiki/Model_selection>`_ to
`PEtab <https://petab.readthedocs.io/>`_. PEtab Select comprises file
formats, a Python library and a command line interface.

Model selection is the process of choosing the best model from a set of
candidate models. PEtab Select provides a standardized and compact way to
specify the candidate model space, implements a number model selection
algorithms and criteria.

Supported model selection algorithms:

* brute force
* forward selection
* backward selection
* FAMoS

Supported model selection criteria:

* (corrected) Akaike Information Criterion (AIC / cAIC)
* Bayesian Information Criterion (BIC)
* ...

Model calibration is performed outside of PEtab Select. For example,
PEtab Select is well-integrated with
`pypesto <https://pypesto.readthedocs.io/>`_
(`model selection example <https://pypesto.readthedocs.io/en/latest/example/model_selection.html>`_).
Other model calibration tools can easily be integrated using the provided
Python library or command line interface.

Installation
------------

The Python 3 package provides both the Python 3 and command-line (CLI)
interfaces, and can be installed from PyPI, with

.. code-block:: bash

    pip install petab-select


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   problem_definition
   examples
   Test Suite <test_suite>
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
