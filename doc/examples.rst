Examples
========

There are currently notebooks aimed at users and developers.

Users that are working with PEtab problems should consult the model selection documentation of the calibration tool they wish to use. In this case, there is no need to use the PEtab Select package directly to perform model selection. After model selection, results can be analyzed with PEtab Select. See the API documentation for :mod:`petab_select.analyze`, or the "Visualization gallery" notebook.

Users who wish to apply model selection methods using a calibration tool that doesn't support PEtab can refer to the "Model selection with non-SBML models" notebook for a demonstration using statsmodels.

Developers wishing to implement model selection in their tool via PEtab Select can consult the notebooks on workflows ("Example usage with the CLI" and "Example usage with Python 3") or "FAMoS in PEtab Select".

.. toctree::
   :maxdepth: 1

   examples/visualization.ipynb
   examples/other_model_types.ipynb
   examples/example_cli_famos.ipynb
   examples/workflow_cli.ipynb
   examples/workflow_python.ipynb
