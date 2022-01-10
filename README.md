# PEtab Select
The repository for the development of the extension to PEtab for model selection, including the additional file formats and Python 3 package.

## Install
The Python 3 package provides both the Python 3 and command-line (CLI) interfaces, and can be installed from PyPI, with `pip3 install petab-select`.

## Examples
There are example Jupyter notebooks for usage of PEtab Select with
- the command-line interface, and
- the Python 3 interface,

in the `doc/examples` directory.

## Supported features
### Criterion
- `AIC`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Definition
- `AICc`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
- `BIC`: https://en.wikipedia.org/wiki/Bayesian_information_criterion#Definition

### Methods
- `forward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `backward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `brute_force`: Optimize all possible model candidates, then return the model with the best criterion value.

Note that the directional methods (forward, backward) find models with the smallest step size (in terms of number of estimated parameters). For example, given the forward method and an initial model with 2 estimated parameters, if there are no models with 3 estimated parameters, but some models with 4 estimated parameters, then the search may return candidate models with 4 estimated parameters.

## File formats
Column or key names that are surrounding by square brackets (e.g. `[constraint_files]`) are optional.
### Selection problem
A YAML file with a description of the model selection problem.
```yaml
format_version: [string]
criterion: [string]
method: [string]
model_space_files: [List of filenames]
[constraint_files]: [List of filenames]
[initial_model_files]: [List of filenames]
```

- `format_version`: The version of the model selection extension format (e.g. `'beta_1'`)
- `criterion`: The criterion by which models should be compared (e.g. `'AIC'`)
- `method`: The method by which model candidates should be generated (e.g. `'forward'`)
- `model_space_files`: The filenames of model space files.
- `constraint_files`: The filenames of constraint files.
- `initial_model_files`: The filenames of initial model files.

### Model space
A TSV with candidate models, in compressed or uncompressed format.

| `model_subspace_id`  | `petab_yaml`     | [`sbml`]   | `parameter_id_1`                                       | ... | `parameter_id_n`                                       |
|----------------------|------------------|------------|--------------------------------------------------------|-----|--------------------------------------------------------|
| (Unique) [string]    | [string]         | [string]   | [string/float] OR [; delimited list of string/float]   | ... | [string/float] OR [; delimited list of string/float]   |

- `model_subspace_id`: An ID for the model subspace.
- `petab_yaml`: The PEtab YAML filename that serves as the base for a model.
- `sbml`: An SBML filename. If the PEtab YAML file specifies multiple SBML models, this can select a specific model by model filename.
- `parameter_id_1`...`parameter_id_n` : Parameter IDs that are specified to take specific values or be estimated. Example valid values are:
  - uncompressed format:
    - `0.0`
    - `1.0`
    - `estimate`
  - compressed format
    - `0.0;1.1;estimate` (the parameter can take the values `0.0` or `1.1`, or be estimated according to the PEtab problem)

### Constraints
A TSV file with constraints.

| `petab_yaml`     | [`if`]                                    | `constraint`                   |
|------------------|-------------------------------------------|--------------------------------|
| [string]         | [SBML L3 Formula expression]              | [SBML L3 Formula expression]   |

- `petab_yaml`: The filename of the PEtab YAML file that this constraint applies to.
- `if`: As a single YAML can relate to multiple models in the model space file, this ensures the constraint is only applies to the models that match this `if` statement
- `constraint`: If a model violates this constraint, it is skipped during the model selection process and not optimized.

### Model(s) (Initial models / model interchange / report)
- Initial models are used to initialize an appropriate model selection method. Model IDs should be unique here and compared to model IDs in any model space files.
- Model interchange refers to the format used to transfer model information between PEtab Select and a PEtab-compatible calibration tool, during the model selection process.
- Report refers to the final results of the model selection process, which may include calibration results from any calibrated models, or just the select model.

Here, the format for a single model is shown. Multiple models can be specified as a YAML list of the same format.

The only required key is the PEtab YAML, as a model requires a PEtab problem. All other keys are may be required, for the different uses of the format (e.g. the report format should include `estimated_parameters`), or at different stages of the model selection process (the PEtab-compatible calibration tool should provide `criteria` for model comparison).

```yaml
[criteria]: [Dictionary of criterion names and values]
[estimated_parameters]: [Dictionary of parameter IDs and values]
[model_hash]: [string]
[model_id]: [string]
[parameters]: [Dictionary of parameter IDs and values]
petab_yaml: [string]
[predecessor_model_hash]: [string]
[sbml]: [string]
```

- `criteria`: The value of the criterion by which model selection was performed, at least. Optionally, other criterion values too.
- `estimated_parameters`: Parameter estimates, not only of parameters specified to be estimated in a model space file, but also parameters specified to be estimated in the original PEtab problem of the model.
- `model_hash`: The model hash, generated by the PEtab Select library.
- `model_id`: The model ID.
- `model_subspace_id`: Same as in the model space files.
- `model_subspace_indices`: The indices that locate this model in its model subspace.
- `parameters`: The parameters from the problem (either values or `'estimate'`) (a specific combination from a model space file, but uncalibrated).
- `petab_yaml`: Same as in model space files.
- `predecessor_model_hash`: The hash of the model that preceded this model during the model selection process.
- `sbml`: Same as in model space files.

## Test cases
Several test cases are provided, to test the compatibility of a PEtab-compatible calibration tool with different PEtab Select features.

The test cases are available in the `test_cases` directory, and are provided in the model format.

| Test ID | Criterion | Method             | Model space files | Compressed format | Constraints files | Initial models files |
|---------|-----------| -------------------|-------------------|-------------------|-------------------|----------------------|
| 0001    | (all)     | (only one model)   | 1                 |                   |                   |                      |
| 0002<sup>[1](#test_case_0002)</sup>    | AIC       | forward            | 1                 |                   |                   |                      |
| 0003    | BIC       | all                | 1                 | Yes               |                   |                      |
| 0004    | AICc      | backward           | 1                 |                   | 1                 |                      |
| 0005    | AIC       | forward            | 1                 |                   |                   | 1                    |
| 0006    | AIC       | forward            | 1                 |                   |                   |                      |
| 0007<sup>[2](#test_case_0007_and_0008)</sup>    | AIC       | forward            | 1                 |                   |                   |                      |
| 0008<sup>[2](#test_case_0007_and_0008)</sup>    | AICc       | backward            | 1                 |                   |                   |                      |

<a name="test_case_0002">1</a>. Model `M1_0` differs from `M1_1` in three parameters, but only 1 additional estimated parameter. The effect of this on model selection criteria needs to be clarified. Test case 0006 is a duplicate of 0002 that doesn't have this issue.

<a name="test_case_0007_and_0008">2</a>. Noise parameter is removed, noise is fixed to `1`.
