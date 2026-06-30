Examples
========

For users
---------

The examples for users cover setting up the PEtab Select files for different model classes, and visualization of model selection results.

If you use one of the supported calibration tools, then we recommend using PEtab Select via the interface provided by the :ref:`calibration_tools`. Otherwise, see the example with an unsupported calibration tool.

After model selection, results can be analyzed with PEtab Select. See the API documentation for :mod:`petab_select.analyze`, or the "Visualization gallery" notebook.

.. toctree::
   :maxdepth: 1

   An ordinary differential equation model with time-series data. <examples/ode_timeseries.rst>
   A polynomial (non-SBML) model with an unsupported calibration tool. <examples/other_model_types.ipynb>
   A model space with multiple structurally different models. <examples/multiple_supersets.rst>
   A demonstration of all visualizations of model selection results. <examples/visualization.ipynb>

For developers
--------------

Developers that want to implement model selection in their tool via PEtab Select can consult the notebooks for the CLI or Python interface. The FAMoS notebook is a complicated example that ensures your tool communicates with PEtab Select correctly over multiple iterations of model selection.

.. toctree::
   :maxdepth: 1

   Example for the Python interface <examples/workflow_python.ipynb>
   Example for the CLI <examples/workflow_cli.ipynb>
   Example with the FAMoS method <examples/example_cli_famos.ipynb>
