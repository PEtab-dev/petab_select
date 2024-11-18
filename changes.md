# Changes

## 0.2.0
There are some **major breaking changes**, to support users providing previous calibration results, e.g. from previous model selection runs. The following changes are reflected in the notebook examples.
- **breaking change** previously, calibration tools would call `candidates` at each iteration of model selection. `candidates` has now been renamed to `start_iteration`, and tools are now expected to run `end_iteration` after calibrating the iteration's models. This structure also simplifies the codebase for other features of PEtab Select.
- **breaking change** previously, calibration tools would determine whether to continue model selection based on whether the candidate space contains any models. Now, calibration tools should rely on the `TERMINATE` signal provided by `end_iteration` to determine whether to continue model selection.
- **breaking change** PEtab Select hides user-calibrated models from the calibration tool, until `end_iteration` is called. Hence, if a calibration tool does some analysis on the calibrated models of the current iteration, the tool should use the `MODELS` provided by `end_iteration`, and not the `MODELS` provided by `start_iteration`.
In summary, here's some pseudocode showing the old way.
```python
from petab_select.ui import candidates
while True:
    # Get iteration models
    models = candidates(...).models
    # Terminate if no models
    if not models:
        break
    # Calibrate iteration models
    for model in models:
        calibrate(model)
    # Print a summary/analysis of current iteration models (dummy code)
    print_summary_of_iteration_models(models)
```
And here's the new way. Full working examples are given in the updated notebooks, including how to handle the candidate space.
```python
from petab_select.ui import start_iteration, end_iteration
from petab_select.constants import MODELS, TERMINATE
while True:
    # Initialize iteration, get uncalibrated iteration models
    iteration = start_iteration(...)
    # Calibrate iteration models
    for model in iteration[MODELS]:
        calibrate(model)
    # Finalize iteration, get all iteration models and results
    iteration_results = end_iteration(...)
    # Print a summary/analysis of all iteration models (dummy code)
    print_summary_of_iteration_models(iteration_results[MODELS])
    # Terminate if indicated
    if iteration_results[TERMINATE]:
        break
```
- Other **major changes**
    - Many thanks to @dweindl for:
        - documentation! https://petab-select.readthedocs.io/
        - GitHub CI fixes and GHA deployments to PyPI and Zenodo
    - fixed a bug introduced in 0.1.8, where FAMoS "jump to most distant" moves were not handled correctly
    - the renamed `candidates`->`start_iteration`:
        - no longer accepts `calibrated_models`, as they are automatically stored in the `CandidateSpace` now with each `end_iteration`
        - `calibrated_models` and `newly_calibrated_models` no longer need to be tracked between iterations. They are now tracked by the candidate space.
        - exclusions via `exclude_models` is no longer supported. exclusions can be supplied with `set_excluded_hashes`
    - model hashes are more readable and composed of two parts:
        1. the model subspace ID
        2. the location of the model in its subspace (the model subspace indices)
    - users can now provide model calibrations from previous model selection runs. This enables them to skip re-calibration of the same models.

## 0.1.13
- fixed bug when no predecessor model is provided, introduced in 0.1.11 (#83)

## 0.1.12
- fixed bug in last version: non-FAMoS methods do not terminate

## 0.1.11
- fixed bug in stepwise moves when working with multiple subspaces (#65)
- fixed bug in FAMoS switching (#68)
- removed `BidirectionalCandidateSpace` and `ForwardAndBackwardCandidateSpace` (#68)
- set estimated parameters as the nominal values in exported PEtab problems (#77)
- many CI, doc, and code quality improvements by @dweindl (#57, #58, #59, #60, #61, #63, #69, #70, #71, #72, #74)
- fixed bug in model hash reproducibility (#78)
- refactored `governing_method` out of `CandidateSpace` (#73)
- fixed bug related to attempted calibration of virtual models (#75)

## 0.1.10
- added `Model.set_estimated_parameters`
  - now expected that `Model.estimated_parameters` has untransformed values

## 0.1.9
- improved reproducibility of test case 0009
  - new predecessor model

## 0.1.8
- improved reproducibility of `summary.tsv` files
    - sorted parameter IDs
    - store the previous predecessor model in the state and candidate space

## 0.1.7
The FAMoS implementation raised an unhandled `StopIteration` when the method switching scheme terminated. When using FAMoS via the UI, this is now handled. Expect an extra line in the summary file produced by the UI, with `# candidates=0`.

## 0.1.6
- The predecessor model in test case 0009 is now specified in the PEtab problem YAML, so is used automatically. A dummy negative log-likelihood value of `inf` was specified for the predecessor model, to enable comparison to calibrated models, without having to calibrate the predecessor model.
- All criterion values are now cast to float, for the `inf` in the predecessor model in test case 0009

## 0.1.5
Bugfix: criterion values are now explicitly type-cast to float (#39)

## 0.1.4
Bugfix: summary TSV now supports string-valued predecessor models (#37)

## 0.1.3
New test case: 0009 (Blasi)

## 0.1.2
Bugfix: ensure correct type for some model savers

## 0.1.1
Bugfix: pypesto test version

## 0.1.0
There are several new candidate spaces, including the FAMoS method. There are also code quality checks (`black` / `isort` / notebook checks) and tests / CI with `tox`.

Below are some significant changes. One major change is that the predecessor model is no longer explicitly specified, instead it is taken as the best model from a set of provided models.
This means a calibration tool only needs to calibrate all models that PEtab Select provides, then send all calibrated models back to PEtab select, at each iteration.
This change is implemented to support the FAMoS method.

### `CandidateSpace`
- new candidate spaces
  - `BidirectionalCandidateSpace`
    - tests models with both the forward and backward method at each iteration
  - `ForwardAndBackwardCandidateSpace`
    - alternates between the forward and backward method
  - `LateralCandidateSpace`
    - moves "sideways" through the model space with parameter swaps
      - a lateral move is where one parameter is turned off, and another parameter is turned on, simultaneously
  - `FamosCandidateSpace`
    - implements the FAMoS method ( https://doi.org/10.1371/journal.pcbi.1007230 )
- candidate spaces have a `update_after_calibration` method
  - this provides functionality to support FAMoS operations that occur inbetween calibration iterations

### `Problem`
- renamed method `add_calibrated_models` to `exclude_models`
- new method `exclude_model_hashes`
- cleaned `get_best`

### `ui.candidates`
- logic change: predecessor model is taken from the following in descending priority, to support the FAMoS method
  - best of `newly_calibarated_models`
  - best of `calibrated_models`
  - `previous_predecessor_model`
- refactored to instead take newly calibrated models as an argument, instead of assuming they are at `candidate_space.models`
- renamed `history` to `calibrated_models`

### CLI
- `candidates`
  - see logic change in `ui.candidates`
    - breaking change: renamed `--predecessor` to `--previous-predecessor-model` to ensure this logic change is noticed by developers
    - removed deprecated `--initial` argument
    - removed `--best`
    - added argument `--calibrated-models` to specify file(s) that contain all calibrated models from all iterations so far
    - added argument `--newly-calibrated-models` to specify file(s) that contain all calibrates models from the latest iteration
  - renamed arguments to align `cli.candidates` with `ui.candidates`
    - renamed `--yaml` to `--problem`
    - renamed `--excluded-model-file` to `--excluded-model-file`
    - renamed `--excluded-model-hash-file` to `--excluded-model-hashes`
- `best`
  - renamed arguments to align `cli.best` with `ui.best`
    - renamed `--yaml` to `--problem`
    - renamed `--models_yaml` to `--models`
- `model_to_petab` and `models_to_petab`, respectively
  - renamed to align `ui` and `cli` methods
    - renamed `--yaml` to `model` and `models`, respectively
    - renamed shorthand `-m` for `--model_id` to `-i` to avoid conflict with new `--model/--models` arguments

## 0.0.9
Fixed automatic exclusions from the Python interface. Now, exclusions are only automatically managed in `Problem`s (thereby also the model space and model subspaces) and `CandidateSpace`s.

## 0.0.8
Previously, use of a `limit` via the UI (e.g. CLI) limited the number of models sent to the candidate space. The candidate space may then reject some models, which means this limit is not the number of models seen in the output.

This version has a breaking change. The `limit` is now the number of models in the output (if the model space has sufficient acceptable models to reach the limit). This `limit` now matches the description that was already provided in the Python and CLI docs.

The behavior of the previous `limit` is now implemented as `limit_sent` in the Python or CLI UI.

## 0.0.7 (bugfix)
- fix implementation of limit in model subspace (GitHub issue #3)

## 0.0.6 (bugfix)
- fixed calculation of number of priors (GitHub issue #2)

## 0.0.5
- renamed `initial_model` to `predecessor_model` in the UI method `candidates`, to avoid confusion.
  - "Initial model" is used to refer to the first model used in any algorithm. "Predecessor model" is the first model in a specific iteration of the algorithm. For the first iteration, initial model and predecessor model are the same.
- added `excluded_model_hashes` as an option to the `candidates` UI method (an alternative to `excluded_models`)
  - named `excluded_model_hash_files` in the CLI -- files with one model hash per line
  - **breaking** renamed `exclude_models` to `excluded_model_files` for consistency / avoid naming conflicts
- the `LateralCandidateSpace` now explicitly requires a predecessor model
- getters and setters for the predecessor model, model exclusions, and limit on number of models, added to `CandidateSpace`
- renamed files in the test cases for consistency

## 0.0.4 (bugfix)

## 0.0.3
- **Added model subspaces**
  - efficiently searches for candidate models in stepwise methods
  - handles exponentially large model spaces
- **Virtual initial models** are now automatically generated if no initial model is provided, for methods that require an initial model
- Refactored methods in `criteria.py` -- now has a `CriteriaCalculator`
  - **PEtab problem calibrators only need to provide "sufficient" information** (just one of the following) in order to have criteria computed by PEtab Select
    - likelihood
    - log-likelihood
    - negative log-likelihood
  - includes methods to transform likelihood
  - should all for calculation of all possible criteria, given some initial subset of criteria
    - e.g., providing a model and its likelihood allows computation of
      - log-likelihood
      - negative log-likelihood
      - AIC
      - AICc
      - BIC
- Added `petab_select best` to the CLI, to get the best model from a list of calibrated models.
- Replaced the symbol for estimated parameters everywhere
  - was `ESTIMATE_SYMBOL_UI==np.nan` in some cases -- now use `ESTIMATE=='estimate'`
  - the following are removed and replaced with `ESTIMATE` in both the file format and Python package
    - `ESTIMATE_SYMBOL_UI`
    - `ESTIMATE_SYMBOL_INTERNAL`
    - `ESTIMATE_SYMBOL_INTERNAL_STR`
- More tests
- CLI (methods renamed to match common Python/CLI naming)
  - `model2petab` -> `model_to_petab`
  - `models2petab` -> `models_to_petab`
- Predecessor model ID, to indicate the model that was previous to a model during model selection iterations

## 0.0.2
- Renamed the `search` method of the CLI to `candidates`
- Added selection ability
  - PEtab Select will initialize a model search method (e.g. the forward method) with the model with the best criterion value, from a list of calibrated models in the report format
  - CLI interface: `-b`/`--best` argument
  - Python interface: `petab_select.Problem.get_best`
- Renamed "model specification files" to "model space files" for clarity
- Renamed `model0_id` to `predecessor_model_id` for readability
- The file format for initial models, model interchange, and reporting is combined into a single "model" format, with some optional keys that are required for different tasks (see README)
- Added `model2petab` and `models2petab` methods to the CLI

# Upcoming changes
- Produce graphs given a set of models and their predecessor models
- Allow specification of `model_id` in a model space file, if the row describes only a single model?
- Use `predecessor_model` everywhere (e.g. `VIRTUAL_PREDECESSOR_MODEL`, `predecessor_model_files`)
