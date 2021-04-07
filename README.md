# pop_finder 

![](https://github.com/katieb1/pop_finder/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/katieb1/pop_finder/branch/main/graph/badge.svg)](https://codecov.io/gh/katieb1/pop_finder) ![Release](https://img.shields.io/github/v/release/katieb1/pop_finder?include_prereleases) [![Documentation Status](https://readthedocs.org/projects/pop_finder/badge/?version=latest)](https://pop_finder.readthedocs.io/en/latest/?badge=latest)

Python package that uses neural networks for population assignment

## Installation

```bash
$ pip install pop-finder
```

## Features

This package includes two main functions that use genetic data to assign individuals of unknown origin to source populations. Each of these functions both have K-Fold Cross-Validation functions for estimating the uncertainty of model predictions. 

1a. `pop_finder.pop_finder.run_neural_net()`: runs a classification neural network for population assignment. 

    A) `metrics.csv`: statistics relating to the model accuracy / precision / recall / F1 score.

    B) `pop_assign_freqs.csv`: the number of times an individual was assigned to each population across the entire ensemble of models.

    C) `pop_assign_ensemble.csv`: the top population of assignment for each individual of unknown origin, along with the frequency of assignment to that population across the entire ensemble of models.

    D) `preds.csv`: All data across all models.

1b. `pop_finder.pop_finder.kfcv()`: runs K-Fold Cross-Validation on models

2. `pop_finder.contour_classifier.contour_classifier()`: runs a regression neural network many times, then uses the combined output to create contour plots for population assignment.


**Package Data**: A small set of data including example VCF, HDF5, and tab-delimited input files are included for testing the functions. Some usage examples with this data are included below.

## Dependencies

The following `python` packages are required to run `pop_finder`:

* python = ">=3.7.1, <3.10"
* numpy = "1.19.4"
* pandas = "^1.2.3"
* h5py = "2.10.0"
* sklearn = "^0.0"
* keras-tuner = "1.0.2"
* matplotlib = "3.3.2"
* zarr = "^2.6.1"
* seaborn = "^0.11.1"
* wheel = "^0.36.2"
* scikit-allel = "1.3.3"
* scipy = ">=1.6.0, <2.0.0"
* tqdm = "^4.59.0"
* tensorflow-cpu = "2.4.1"

## Usage

Load the `pop_finder` library:

```
from pop_finder import pop_finder
```

Run the ensemble of neural networks on the sample data found in [this folder](https://github.com/katieb1/pop_finder/tree/main/tests/test_inputs):

```
# Path to helper data
infile_all = "tests/test_inputs/onlyAtl_500.recode.vcf.locator.hdf5"
infile_kfcv = "tests/test_inputs/onlyAtl_500_kfcv.recode.vcf"
sample_data1 = "tests/test_inputs/onlyAtl_truelocs.txt"
sample_data2 = "tests/test_inputs/onlyAtl_truelocs_NAs.txt" 

# Path to output
save_path = "outputs"

# Run the function
pop_finder.run_neural_net(infile_kfcv, infile_all, sample_data2,
                          save_allele_counts=False, save_weights=False,
                          patience=10, batch_size=32, max_epochs=2,
                          seed=1, train_prop=0.8, gpu_number='0',
                          tune_model=False, n_splits=5, n_reps=5,
                          save_best_mod=False, save_dir=save_path,
                          plot_history=False)
```

## Documentation

The official documentation is hosted on Read the Docs: https://pop_finder.readthedocs.io/en/latest/

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/katieb1/pop_finder/graphs/contributors).

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
