from pop_finder import __version__
from pop_finder import pop_finder
from pop_finder import read
from pop_finder import hp_tuning
import pandas as pd
import numpy as np
import os
import pytest

# Path to helper data
infile_all = "tests/test_inputs/onlyAtl_500.recode.vcf.locator.hdf5"
infile_kfcv = "tests/test_inputs/onlyAtl_500_kfcv.recode.vcf"
sample_data1 = "tests/test_inputs/onlyAtl_truelocs.txt"
sample_data2 = "tests/test_inputs/onlyAtl_truelocs_NAs.txt"


def test_version():
    assert __version__ == "0.1.14"


def test_read():

    # Read data w/o kfcv
    x = read.read_data(infile_all,
                       sample_data2,
                       save_allele_counts=False)
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], np.ndarray)
    assert isinstance(x[2], pd.core.frame.DataFrame)
    assert len(x) == 3

    # Read data w/ kfcv
    y = read.read_data(infile_all,
                       sample_data1,
                       save_allele_counts=False,
                       kfcv=True)
    assert isinstance(y, tuple)
    assert isinstance(y[0], pd.core.frame.DataFrame)
    assert isinstance(y[1], np.ndarray)
    assert len(y) == 2


def test_hp_tuning():

    hm_test = hp_tuning.classifierHyperModel(input_shape=2, num_classes=2)
    assert isinstance(hm_test, hp_tuning.classifierHyperModel)
    assert hm_test.input_shape == 2
    assert hm_test.num_classes == 2


def test_pop_finder():

    test_mod_name = pop_finder.get_model_name(2)
    assert test_mod_name == "model_2"

    save_path = "tests/test_output"
    pop_finder.run_neural_net(
        infile_kfcv,
        infile_all,
        sample_data2,
        save_allele_counts=False,
        save_weights=False,
        patience=10,
        batch_size=32,
        max_epochs=2,
        seed=1,
        train_prop=0.8,
        gpu_number="0",
        tune_model=False,
        n_splits=5,
        n_reps=1,
        save_best_mod=False,
        save_dir=save_path,
        plot_history=False,
    )
    assert os.path.isfile(save_path + "/metrics.csv")
    if os.path.isfile(save_path + "/metrics.csv"):
        os.remove(save_path + "/metrics.csv")
    assert os.path.isfile(save_path + "/pop_assign_freqs.csv")
    if os.path.isfile(save_path + "/pop_assign_freqs.csv"):
        os.remove(save_path + "/pop_assign_freqs.csv")
    assert os.path.isfile(save_path + "/pop_assign_ensemble.csv")
    if os.path.isfile(save_path + "/pop_assign_ensemble.csv"):
        os.remove(save_path + "/pop_assign_ensemble.csv")
    assert os.path.isfile(save_path + "/preds.csv")
    if os.path.isfile(save_path + "/preds.csv"):
        os.remove(save_path + "/preds.csv")
    os.rmdir(save_path)

    with pytest.raises(ValueError):
        pop_finder.run_neural_net(
            infile_kfcv,
            infile_all,
            sample_data2,
            save_allele_counts=False,
            save_weights=False,
            patience=10,
            batch_size=32,
            max_epochs=2,
            seed=1,
            train_prop=0.95,
            gpu_number="0",
            tune_model=False,
            n_splits=5,
            n_reps=1,
            save_best_mod=False,
            save_dir=save_path,
            plot_history=False,
        )
