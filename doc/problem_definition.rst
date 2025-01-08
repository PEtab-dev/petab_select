Problem definition and file formats
===================================

Model selection problems for PEtab Select are defined by the following files:

#. a general description of the model selection problem,
#. a specification of the model space, and
#. (optionally) a specification of the initial candidate model.

The different file formats are described below. The YAML file formats
come with a YAML-formatted JSON schema, such that these files can be
easily worked with independently of the PEtab Select library.

1. Selection problem
--------------------

A YAML file with a description of the model selection problem.

.. code-block:: yaml

   format_version: # string.
   criterion: # string.
   method: # string.
   model_space_files: # list[string]. Filenames.
   candidate_space_arguments: # list[string] (optional). Filenames.

- ``format_version``: The version of the model selection extension format
  (e.g. ``1``)
- ``criterion``: The criterion by which models should be compared
  (e.g. ``AIC``)
- ``method``: The method by which model candidates should be generated
  (e.g. ``forward``)
- ``model_space_files``: The filenames of model space files.
- ``candidate_space_arguments``: Additional arguments used to generate
  candidate models during model selection. For example, an initial candidate
  model can be specified with the following code, where
  ``predecessor_model.yaml`` is a valid :ref:`model file <section-model-yaml>`. Additional arguments are
  provided in the documentation of the ``CandidateSpace`` class, and an example is provided in
  `test case 0009 <https://github.com/PEtab-dev/petab_select/blob/main/test_cases/0009/petab_select_problem.yaml>`_.

.. code-block:: yaml

   candidate_space_arguments:
     predecessor_model: predecessor_model.yaml

Schema
^^^^^^

The schema is provided as `YAML-formatted JSON schema <_static/problem.yaml>`_, which enables easy validation with various third-party tools.

.. literalinclude:: standard/problem.yaml
   :language: yaml

2. Model space
--------------

A TSV file with candidate models, in compressed or uncompressed format.
Each row defines a model subspace, by specifying value(s) that each parameter
can take. The models in a model subspace are all combinations of values across
all parameters.

.. list-table::
   :header-rows: 1

   * - ``model_subspace_id``
     - ``model_subspace_petab_yaml``
     - ``parameter_id_1``
     - ...
     - ``parameter_id_n``
   * - (unique) [string]
     - [string]
     - (``;``-delimited list) [string/float]
     - ...
     - ...

- ``model_subspace_id``: An ID for the model subspace.
- ``model_subspace_petab_yaml``: The YAML filename of the PEtab problem that serves as the basis of all
  models in this subspace.
- ``parameter_id_1`` ... ``parameter_id_n``: Specify the values that a
  parameter can take in the model subspace. For example, this could be:

  - a single value

    - ``0.0``
    - ``1.0``
    - ``estimate``

  - one of several possible values, as a ``;``-delimited list

    - ``0.0;1.1;estimate`` (the parameter can take the values ``0.0`` or
      ``1.1``, or be estimated)

Example of concise specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the ``;``-delimited list format, a model subspace that has two parameters
(``p1, p2``) and six models:

- ``p1:=0, p2:=10``
- ``p1:=0, p2:=20``
- ``p1:=0, p2:=estimate``
- ``p1:=estimate, p2:=10``
- ``p1:=estimate, p2:=20``
- ``p1:=estimate, p2:=estimate``

can be specified like

.. list-table::
   :header-rows: 1

   * - model_subspace_id
     - model_subspace_petab_yaml
     - p1
     - p2
   * - subspace1
     - petab_problem.yaml
     - 0;estimate
     - 10;20;estimate

.. _section-model-yaml:

3. Model(s) (Predecessor models / model interchange / report)
-------------------------------------------------------------

- *Predecessor models* are used to initialize a compatible model selection
  method.
- *Model interchange* refers to the format used to transfer model information
  between PEtab Select and a PEtab-compatible calibration tool, during the
  model selection process.
- *Report* refers to the final results of the model selection process, which
  may include calibration results from any calibrated models, or just the
  selected model.

Here, the format for a single model is shown. Multiple models can be specified
as a YAML list of the same format. Some optional keys are required in different
contexts (for example, model comparison will require ``criteria``).

Brief format description
^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: yaml

   model_subspace_id: # str (required).
   model_subspace_indices: # list[int] (required).
   criteria: # dict[str, float] (optional). Criterion ID => criterion value.
   model_hash: # str (optional).
   model_subspace_petab_yaml: # str (required).
   estimated_parameters: # dict[str, float] (optional). Parameter ID => parameter value.
   iteration: # int (optional).
   model_id: # str (optional).
   parameters: # dict[str, float | int | "estimate"] (required). Parameter ID => parameter value or "estimate".
   predecessor_model_hash: # str (optional).

- ``model_subspace_id``: Same as in the model space files.
- ``model_subspace_indices``: The indices that locate this model in its model subspace.
- ``criteria``: The value of the criterion by which model selection was performed, at least. Optionally, other criterion values too.
- ``model_hash``: The model hash, generated by the PEtab Select library. The format is ``[MODEL_SUBSPACE_ID]-[MODEL_SUBSPACE_INDICES_HASH]``. If all parameters are in the model are defined like ``0;estimate``, then the hash is a string of ``1`` and ``0``, for parameters that are estimated or not, respectively.
- ``model_subspace_petab_yaml``: Same as in model space files.
- ``estimated_parameters``: Parameter estimates, including all estimated parameters that are not in the model selection problem; i.e., parameters that are set to be estimated in the model subspace PEtab problem but don't appear in the column header of the model space file.
- ``iteration``: The iteration of model selection in which this model appeared.
- ``model_id``: The model ID.
- ``parameters``: The parameter combination from the model space file that defines this model (either values or ``"estimate"``). Not the calibrated values, which are in ``estimated_parameters``!
- ``predecessor_model_hash``: The hash of the model that preceded this model during the model selection process. Will be ``virtual_initial_model-`` if the model had no predecessor model.

Schema
^^^^^^

The schema are provided as YAML-formatted JSON schema, which enables easy validation with various third-party tools. Schema are provided for:

- `a single model <_static/model.yaml>`_, and
- `a list of models <_static/models.yaml>`_, which is simply a YAML list of the single model format.

Below is the schema for a single model.

.. literalinclude:: standard/model.yaml
   :language: yaml
