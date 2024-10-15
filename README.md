# PEtab Select

[![PyPI - Version](https://img.shields.io/pypi/v/petab-select)](https://pypi.org/project/petab-select/)

The repository for the development of the extension to PEtab for model
selection, including the additional file formats and Python 3 package.

## Install

The Python 3 package provides both the Python 3 and command-line (CLI)
interfaces, and can be installed from PyPI, with `pip3 install petab-select`.

## Documentation

Further documentation is available at
[http://petab-select.readthedocs.io/](http://petab-select.readthedocs.io/).

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
- `brute_force`: Optimize all possible model candidates, then return the model
  with the best criterion value.
- `famos`: https://doi.org/10.1371/journal.pcbi.1007230

Note that the directional methods (forward, backward) find models with the
smallest step size (in terms of number of estimated parameters). For example,
given the forward method and a predecessor model with 2 estimated parameters,
if there are no models with 3 estimated parameters, but some models with 4
estimated parameters, then the search may return candidate models with 4
estimated parameters.

## Test cases

Several test cases are provided, to test the compatibility of a
PEtab-compatible calibration tool with different PEtab Select features.

The test cases are available in the `test_cases` directory, and are provided in
the model format.

| Test ID                                      | Criterion | Method           | Model space files | Compressed format | Constraints files | Predecessor (initial) models files |
|----------------------------------------------|-----------|------------------|-------------------|-------------------|-------------------|------------------------------------|
| 0001                                         | (all)     | (only one model) | 1                 |                   |                   |                                    |
| 0002<sup>[1](#test_case_0002)</sup>          | AIC       | forward          | 1                 |                   |                   |                                    |
| 0003                                         | BIC       | all              | 1                 | Yes               |                   |                                    |
| 0004                                         | AICc      | backward         | 1                 |                   | 1                 |                                    |
| 0005                                         | AIC       | forward          | 1                 |                   |                   | 1                                  |
| 0006                                         | AIC       | forward          | 1                 |                   |                   |                                    |
| 0007<sup>[2](#test_case_0007_and_0008)</sup> | AIC       | forward          | 1                 |                   |                   |                                    |
| 0008<sup>[2](#test_case_0007_and_0008)</sup> | AICc      | backward         | 1                 |                   |                   |                                    |
| 0009<sup>[3](#test_case_0009)</sup>          | AICc      | FAMoS            | 1                 | Yes               |                   | Yes                                |

<a name="test_case_0002">1</a>. Model `M1_0` differs from `M1_1` in three
parameters, but only 1 additional estimated parameter. The effect of this on
model selection criteria needs to be clarified. Test case 0006 is a duplicate
of 0002 that doesn't have this issue.

<a name="test_case_0007_and_0008">2</a>. Noise parameter is removed, noise is
fixed to `1`.

<a name="test_case_0009">3</a>. This is a computationally expensive problem to
solve. Developers can try a model selection initialized with the provided
predecessor model, which is a model start that reproducibly finds the expected
model. To solve the problem reproducibly <i>ab initio</i>, on the order of 100
random model starts are required. This test case reproduces the model selection
problem presented in https://doi.org/10.1016/j.cels.2016.01.002 .
