Analysis
========

After using PEtab Select to perform model selection, you may want to operate on all "good" calibrated models.
The PEtab Select Python library provides some methods to help with this. Please request any missing methods.

See the Python API docs for the :class:`petab_select.Models` class, which provides some methods. In particular, :attr:`petab_select.Models.df` can be used
to get a quick overview over all models, as a pandas dataframe.

Additionally, see the Python API docs for the :mod:`petab_select.analyze` module, which contains some methods to subset and group models,
or compute "weights" (e.g. Akaike weights).

Model hashes
^^^^^^^^^^^^

Model hashes are special objects in the library, that are generated from model-specific information that is unique within a single PEtab Select problem.

This means you can reconstruct the model given some model hash. For example, with this model hash `M1-000`, you can reconstruct the :class:`petab_select.ModelHash` from a string, then reconstruct the :class:`petab_select.Model`.

.. code-block:: language

   ModelHash.from_hash("M1-000").get_model(petab_select_problem)

You can use this to get the uncalibrated version of a calibrated model.

.. code-block:: language

   model.hash.get_model(petab_select_problem)
