# Draft model selection extension
Repository for development of the model selection extension

# Install
This Python 3 package can be installed from PyPI, with `pip3 install petab-select`.

# Examples
There are example Jupyter notebooks for usage of PEtab Select with
- the [command-line interface], and
- the [Python 3 interface],
in the `doc/examples` directory of this repository.

# Supported features
## Criterion
- `AIC`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Definition
- `AICc`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
- `BIC`: https://en.wikipedia.org/wiki/Bayesian_information_criterion#Definition

## Methods
- `forward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `backward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `brute_force`: Optimize all possible model candidates, then return the model with the best criterion value.

Note that the directional methods (forward, backward) find models with the smallest step size (in terms of number of estimated parameters). For example, given the forward method and an initial model with 2 estimated parameters, if there are no models with 3 estimated parameters, but some models with 4 estimated parameters, then the search may return candidate models with 4 estimated parameters.

# Format
## Selection problem file
A YAML file with a description of the model selection problem.
```yaml
format_version: [string]
criterion: [string]
method: [string]
model_specification_files: [List of filenames]
constraint_files: [List of filenames]
initial_model_files: [List of filenames]
```

- `format_version`: The version of the model selection extension format (e.g. `'beta_1'`)
- `criterion`: The criterion by which models should be compared (e.g. `'AIC'`)
- `method`: The method by which model candidates should be generated (e.g. `'forward'`)
- `model_specification_files`: The filenames of model specification files.
- `constraints_files`: The filenames of constraint files.
- `initial_model_files`: The filenames of initial model files.

## Model specifications files
A TSV with candidate models, in compressed or uncompressed format.

| `model_id`           | `petab_yaml`     | [`sbml`]   | `parameter_id_1`                                       | ... | `parameter_id_n`                                       |
|---------------------|------------|------------|--------------------------------------------------------|-----|--------------------------------------------------------|
| (Unique) [string]   | [string]   | [string]   | [string/float] OR [; delimited list of string/float]   | ... | [string/float] OR [; delimited list of string/float]   |

- `model_id`: An ID for the model (or model set, if the row is in the compressed format and specifies multiple models).
- `petab_yaml`: The PEtab YAML filename that serves as the base for a model.
- `sbml`: An SBML filename. If the PEtab YAML file specifies multiple SBML models, this can select a specific model by model filename.
- `parameter_id_1`...`parameter_id_n` : Parameter IDs that are specified to take specific values or be estimated. Example valid values are:
  - uncompressed format:
    - `0.0`
    - `1.0`
    - `estimate`
  - compressed format
    - `0.0;1.1;estimate` (the parameter can take the values `0.0` or `1.1`, or be estimated according to the PEtab problem)

## Constraints files
A TSV file with constraints.

| `petab_yaml`     | [`if`]                                    | `constraint`                   |
|------------|-------------------------------------------|--------------------------------|
| [string]   | (Optional) [SBML L3 Formula expression]   | [SBML L3 Formula expression]   |

- `petab_yaml`: The filename of the PEtab YAML file that this constraint applies to.
- `if`: As a single YAML can relate to multiple models in the model specification file, this ensures the constraint is only applies to the models that match this `if` statement
- `constraint`: If a model violates this constraint, it is skipped during the model selection process and not optimized.

## Initial models
A TSV with models that can be used to initialize a model selection method. This format is the PEtab Select model YAML format, described below.

Model IDs of initial models should be unique here and compared to model IDs in model selection files.

## Model interchange format
This file format is used to transfer information about models, and their parameter estimation problems, between PEtab Select and a PEtab-compatible tool, during model selection.
The format can also be used to specify initial models for a selection method, or the final results of the model selection.
Multiple models can be specified in the same file with a YAML list.

```yaml
model_id: [string]
petab_yaml: [string]
sbml: (Optional) [string]
parameters: [Dictionary of parameter IDs and values]
estimated_parameters: [Dictionary of parameter IDs and values]
criteria: [Dictionary of criterion names and values]
```

- `modelId`: see the format of the model specifications files
- `YAML`: see the format of the model specifications files
- `SBML`: see the format of the model specifications files
- `parameters`: the parameters from the problem (either values or `'estimate'`)
- `estimated_parameters`: parameter estimates, not only of parameters specified to be estimated in a model specification file, but also parameters specified to be estimated in the original PEtab problem of the model
- `criteria`: the value of the criterion by which model selection was performed, at least. Optionally, other criterion values too.

# Test cases
| Test ID | Criterion | Method    | Model specification files | Compressed format | Constraints files | Initial models files |
|---------|-----------| ----------|---------------------------|-------------------|-------------------|----------------------|
| 0001    | (all)       | (only one model)   | 1                         |                   |                   |                      |
| 0002    | AIC       | forward   | 1                         |                   |                   |                      |
| 0003    | BIC       | all       | 1                         | Yes               |                   |                      |
| 0004    | AICc      | backward  | 1                         |                   | 1                 |                      |
| 0005    | AIC       | forward   | 1                         |                   |                   | 1                    |
