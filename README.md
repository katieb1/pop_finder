# pop_finder 

![](https://github.com/katieb1/pop_finder/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/katieb1/pop_finder/branch/main/graph/badge.svg)](https://codecov.io/gh/katieb1/pop_finder) ![Release](https://img.shields.io/github/v/release/katieb1/pop_finder?include_prereleases) [![Documentation Status](https://readthedocs.org/projects/pop_finder/badge/?version=latest)](https://pop_finder.readthedocs.io/en/latest/?badge=latest)

Python package that uses neural networks for population assignment

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/ pop_finder
```

## Features

This package includes one main function that runs an ensemble of neural networks that are trained using genetic data and individuals of known origin to assign individuals of unknown origin to populations.

1. `pop_finder.run_neural_net()`: produces multiple results files, including:

    A) `metrics.csv`: statistics relating to the model accuracy and uncertainty.

    B) `pop_assign_freqs.csv`: the number of times an individual was assigned to each population across the entire ensemble of models.

    C) `pop_assign_ensemble.csv`: the top population of assignment for each individual of unknown origin, along with the frequency of assignment to that population across the entire ensemble of models.

    D) `preds.csv`: All data across all models.

**Package Data**: A small set of data including example VCF, HDF5, and tab-delimited input files are included for testing the functions. Some usage examples with this data are included below.

## Dependencies

The following `python` packages are required to run `pop_finder`:

* python = ">=3.7.1, <4.0"
* numpy = "1.18.4"
* pandas = "^1.2.3"
* h5py = "2.10.0"
* argparse = "^1.4.0"
* sklearn = "^0.0"
* tensorflow = "2.3.0"
* keras-tuner = "1.0.2"
* matplotlib = "3.3.2"
* scikit-allel = "1.3.2"
* zarr = "^2.6.1"

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
