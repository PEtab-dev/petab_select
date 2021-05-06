# Draft model selection extension
Repository for development of the model selection extension

# Supported features
## Criterion
- `AIC`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Definition
- `AICc`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
- `BIC`: https://en.wikipedia.org/wiki/Bayesian_information_criterion#Definition

## Methods
- `forward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `backward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `bidirectional`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `all`: Optimize all possible model candidates, then return the model with the best criterion value.

# Format
## Selection problem file
A YAML file with a description of the model selection problem.
```yaml
format_version: [string]
criterion: [string]
method: [string]
model_specifications_files: [List of filenames]
constraints_files: [List of filenames]
initial_model_files: [List of filenames]
```

- `format_version`: The version of the model selection extension format (e.g. `'beta_1'`)
- `criterion`: The criterion by which models should be compared (e.g. `'AIC'`)
- `method`: The method by which model candidates should be generated (e.g. `'forward'`)
- `model_specifications_files`: The filenames of model specifications files.
- `constraints_files`: The filenames of constraints files.
- `initial_model_files`: The filenames of initial model files.

## Model specifications files
A TSV with candidate models, in compressed or uncompressed format.

| `modelId`           | `YAML`     | [`SBML`]   | `parameter_id_1`                                       | ... | `parameter_id_n`                                       |
|---------------------|------------|------------|--------------------------------------------------------|-----|--------------------------------------------------------|
| (Unique) [string]   | [string]   | [string]   | [string/float] OR [; delimited list of string/float]   | ... | [string/float] OR [; delimited list of string/float]   |

- `modelId`: An ID for the model (or model set, if the row is in the compressed format and specifies multiple models).
- `YAML`: The PEtab YAML filename that serves as a base for the model.
- `SBML`: An SBML filename. If the PEtab YAML file specifies multiple SBML models, this can select a specific model by model filename.
- `parameter_id_1`...`parameter_id_n` : Parameter IDs that are specified to take specific values or be estimated. Example valid values are:
  - uncompressed format:
    - `0.0`
    - `1.0`
    - `-` (estimated)
  - compressed format
    - `0.0;1.0;-` (the parameter can take the values `0.0` or `1.0`, or be estimated according to the PEtab problem)

## Constraints files
A TSV file with constraints.

| `YAML`     | [`if`]                                    | `constraint`                   |
|------------|-------------------------------------------|--------------------------------|
| [string]   | (Optional) [SBML L3 Formula expression]   | [SBML L3 Formula expression]   |

- `YAML`: The filename of the PEtab YAML file that this constraint applies to.
- `if`: As a single YAML can relate to multiple models in the model specification file, this ensures the constraint is only applies to the models that match this `if` statement
- `constraint`: If a model violates this constraint, it is skipped during the model selection process and not optimized.

## Initial models
A TSV with models that can be used to start the model selection method. This format is identical to the format for model selection files.

Model IDs should be unique here and compared to model IDs in model selection files.

## Return format
To be decided. Could match the format of the expected results for the test cases (the "`expected.yaml`" file in each test case directory).

```yaml
modelId: [string]
YAML: [string]
SBML: (Optional) [string]
parameters: [Dictionary of parameter IDs and values]
criteria: [Dictionary of criterion names and values]
```

- `modelId`: see the format of the model specifications files
- `YAML`: see the format of the model specifications files
- `SBML`: see the format of the model specifications files
- `parameters`: the optimized parameter values.
- `criteria`: the value of the criterion by which model selection was performed, at least. Optionally, other criterion values too.

# Test cases
| Test ID | Criterion | Method    | Model specification files | Compressed format | Constraints files | Initial models files |
|---------|-----------| ----------|---------------------------|-------------------|-------------------|----------------------|
| 0001    | AIC       | forward   | 1                         |                   |                   |                      |
| 0002    | BIC       | all       | 1                         | Yes               |                   |                      |
| 0003    | AICc      | backward  | 1                         |                   | 1                 |                      |
| 0004    | AIC       | forward   | 1                         |                   |                   | 1                    |
