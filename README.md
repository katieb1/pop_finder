# pop_finder 

![](https://github.com/katieb1/pop_finder/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/katieb1/pop_finder/branch/main/graph/badge.svg)](https://codecov.io/gh/katieb1/pop_finder) ![Release](https://img.shields.io/github/v/release/katieb1/pop_finder?include_prereleases) [![Documentation Status](https://readthedocs.org/projects/pop_finder/badge/?version=latest)](https://pop_finder.readthedocs.io/en/latest/?badge=latest)

Python package that uses neural networks for population assignment

## Installation

```bash
$ pip install pop-finder
```

## Features

This package includes two main modules, `pop_finder` and `contour_classifier`, that use genetic data to assign individuals of unknown origin to source populations. Each of these modules have a K-Fold Cross-Validation function for estimating the uncertainty of model predictions. The `pop_finder` module also has a `hyper_tune` option, which allows you to tune the hyperparameters of the neural network before getting an accuracy estimate and predicting on unknowns.

### Module 1: `pop_finder`

1. `pop_finder.pop_finder.run_neural_net()`: runs a classification neural network for population assignment. 

    Outputs:

    * `metrics.csv`: statistics relating to the model accuracy / precision / recall / F1 score.
    
    Outputs if ensemble=True:

    * `pop_assign_freqs.csv`: the number of times an individual was assigned to each population across the entire ensemble of models.

    * `pop_assign_ensemble.csv`: the top population of assignment for each individual of unknown origin, along with the frequency of assignment to that population across the entire ensemble of models.

    * `ensemble_test_results.csv`: proportion of times an individual in the test set was assigned to each population across the entire ensemble of models. Used for assessing accuracy.

    Outputs if ensemble=False:

    * `pop_assign.csv`: populations assignments for individuals of unknown origin.  

    * `test_results.csv`: prediction values for each individual from the test set.

2. `pop_finder.pop_finder.hyper_tune()`: tunes the model hyperparameters for a given dataset to maximize accuracy and minimize loss.

    Outputs:

    * `best_mod`: the `save_dir` from running this function can later be used as the `mod_path` when running `run_neural_net` or `kfcv`, allowing you to use a model with tuned hyperparameters rather than the default model.

3. `pop_finder.pop_finder.kfcv()`: runs K-Fold Cross-Validation on model(s) and outputs metrics of model performance (accuracy, precision, recall, F1 score) and confusion matrix plots.

    Outputs:

    * `classification_report.csv`: scikit-learn's classification report containing information on accuracy, precision, recall, and F1 score for each population and the overall model.

    * `cm.png`: confusion matrix for single model predictions. If an ensemble is used, this is each individual model's predictions summed rather than the performance of the ensemble.

    ![](figures/cm.png)

    * `metrics.csv`: includes accuracy scores for the test set using single models, using the ensemble model, and using a weighted ensemble model. Also includes the 95% confidence interval and loss of the test values using the single model(s).

    Outputs if ensemble=True:

    * `ensemble_classification_report.csv`: classification report for the ensemble of models. Can be used to compare model performance between using an ensemble vs using a single model.

    ![](figures/ensemble_cm.png)

    * `ensemble_cm.png`: confusion matrix for ensemble model predictions.

    * `ensemble_test_results.csv`: proportion of times an individual in the test set was assigned to each population across the entire ensemble of models. Used for assessing accuracy.

    * `ensemble_preds.csv`: predictions across all models used in the ensemble.

4. `pop_finder.pop_finder.snp_rank()`: finds relative importance of SNPs on accuracy of model. Can be used to create SNP chips for future population assignment tasks.

    Output:

    * `perturbation_rank_results.csv`: table of SNPs and corresponding relative importance. SNP ID relates to the order in which the SNP was found in the VCF file.

5. `pop_finder.pop_finder.assign_plot()`: can be run with the output from `run_neural_net` to create a structure plot of model confidence in predictions for each population. This function uses the model prediction values for each sample, so takes into account how confident the model was in each prediction rather than if the model predicted correctly vs incorrectly.

    Output:

    * `assign_plot.png`



6. `pop_finder.pop_finder.structure_plot()`: can be run with the output from `kfcv` to create a structure plot of correct assignment of test sets to see general accuracy of model predictions. This function only uses whether the model predicted correctly vs. incorrectly, and thus does not indicate true model confidence.

    Output:

    * `structure_plot.png`

    ![](figures/structure_plot.png)

### Module 2: `contour_classifier`

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

### Python IDE

Load the `pop_finder` library:

```
from pop_finder import pop_finder
```

Run the ensemble of neural networks on the sample data found in [this folder](https://github.com/katieb1/pop_finder/tree/main/tests/test_inputs).

The genetic data corresponds to Atlantic Leach's storm-petrels (*Hydrobates* spp.) from the following colonies:

![](figures/lesp_colonies.png)

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

### Command Line

## Documentation

The official documentation is hosted on Read the Docs: https://pop_finder.readthedocs.io/en/latest/

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/katieb1/pop_finder/graphs/contributors).

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
