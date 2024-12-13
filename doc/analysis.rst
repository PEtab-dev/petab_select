Analysis
========

After using PEtab Select to perform model selection, you may want to operate on all "good" calibrated models.
The PEtab Select Python library provides some methods to help with this. Please request any missing methods.

See the Python API docs for the ``Models`` class, which provides some methods. In particular, ``Models.df`` can be used
to get a quick overview over all models, as a pandas dataframe.

Additionally, see the Python API docs for the ``petab_select.analysis`` module, which contains some methods to subset and group models,
or compute "weights" (e.g. Akaike weights).
