from pop_finder import __version__
from pop_finder import pop_finder
from pop_finder import contour_classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import shutil
import pytest

# helper data
infile_all = "tests/test_inputs/onlyAtl_500.recode.vcf.locator.hdf5"
infile_all_vcf = "tests/test_inputs/onlyAtl_500.recode.vcf"
infile_kfcv = "tests/test_inputs/onlyAtl_500_kfcv.recode.vcf"
sample_data1 = "tests/test_inputs/onlyAtl_truelocs.txt"
sample_data2 = "tests/test_inputs/onlyAtl_truelocs_NAs.txt"
sample_data3 = "tests/test_inputs/onlyAtl_truelocs_badsamps.txt"
sample_data4 = "tests/test_inputs/onlyAtl_truelocs_3col.txt"
pred_path = "tests/test_inputs/test_out/loc_boot0_predlocs.txt"
X_train = np.load("tests/test_inputs/X_train.npy")
X_train_empty = np.zeros(shape=0)
y_train = pd.read_csv("tests/test_inputs/y_train.csv")
y_train_empty = pd.DataFrame()
X_test = np.load("tests/test_inputs/X_test.npy")
X_test_empty = np.zeros(shape=0)
y_test = pd.read_csv("tests/test_inputs/y_test.csv")
y_test_empty = pd.DataFrame()
unknowns = pd.read_csv("tests/test_inputs/test_unknowns.csv")
unknowns_empty = pd.DataFrame()
ukgen = np.load("tests/test_inputs/ukgen.npy")
ukgen_empty = np.zeros(shape=0)


def test_version():
    assert __version__ == "1.0.3"


def test_read_data():

    # Read data w/o kfcv
    x = pop_finder.read_data(infile_all, sample_data2)
    assert isinstance(x, tuple)
    assert isinstance(x[0], pd.core.frame.DataFrame)
    assert isinstance(x[1], np.ndarray)
    assert isinstance(x[2], pd.core.frame.DataFrame)
    assert len(x) == 3

    # Read data w/ kfcv
    y = pop_finder.read_data(infile_all, sample_data1, kfcv=True)
    assert isinstance(y, tuple)
    assert isinstance(y[0], pd.core.frame.DataFrame)
    assert isinstance(y[1], np.ndarray)
    assert len(y) == 2

    # Test inputs
    with pytest.raises(ValueError, match="Path to infile does not exist"):
        pop_finder.read_data(infile="hello", sample_data=sample_data2)
    with pytest.raises(
        ValueError, match="Infile must have extension 'zarr', 'vcf', or 'hdf5'"
    ):
        pop_finder.read_data(infile=sample_data1, sample_data=sample_data2)
    with pytest.raises(ValueError,
                       match="Path to sample_data does not exist"):
        pop_finder.read_data(infile_all, sample_data="hello")
    with pytest.raises(ValueError,
                       match="sample_data does not have correct columns"):
        pop_finder.read_data(infile_all, sample_data=sample_data4)
    with pytest.raises(
        ValueError,
        match="sample ordering failed! Check that sample IDs match VCF."
    ):
        pop_finder.read_data(infile_kfcv, sample_data3)


def test_hp_tuning():

    hm_test = pop_finder.classifierHyperModel(
        input_shape=2, num_classes=2)
    assert isinstance(hm_test,
                      pop_finder.classifierHyperModel)
    assert hm_test.input_shape == 2
    assert hm_test.num_classes == 2


def test_hyper_tune():

    # General run
    tuner_test = pop_finder.hyper_tune(
        infile=infile_all,
        sample_data=sample_data2,
        max_epochs=10,
        save_dir="tests/hyper_tune_test_out",
        mod_name="hyper_tune",
    )

    assert type(
        tuner_test[0] == "tensorflow.python.keras.engine.sequential.Sequential"
    )

    # Make sure correct files are output
    assert os.path.exists("tests/hyper_tune_test_out")
    assert os.path.exists("tests/hyper_tune_test_out/best_mod")
    assert os.path.exists("tests/hyper_tune_test_out/X_train.npy")
    assert os.path.exists("tests/hyper_tune_test_out/X_test.npy")
    assert os.path.exists("tests/hyper_tune_test_out/y_train.csv")
    assert os.path.exists("tests/hyper_tune_test_out/y_test.csv")

    # Remove files for next run
    if os.path.exists("tests/hyper_tune_test_out/best_mod"):
        shutil.rmtree("tests/hyper_tune_test_out/best_mod")

    # Test if value error thrown if y_val != y_train
    with pytest.raises(ValueError, match="train_prop is too high"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs=10,
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
            train_prop=0.99,
        )

    # Check all inputs
    # infile does not exist
    with pytest.raises(ValueError, match="infile does not exist"):
        pop_finder.hyper_tune(
            infile="tests/test_inputs/onlyAtl_500.vcf",
            sample_data=sample_data2,
            max_epochs=10,
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
        )
    # sample_data does not exist
    with pytest.raises(ValueError, match="sample_data does not exist"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data="hello.txt",
            max_epochs=10,
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
        )
    # max_trials not right format
    with pytest.raises(ValueError, match="max_trials should be integer"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs=10,
            max_trials=1.5,
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
        )
    # runs_per_trial not right format
    with pytest.raises(ValueError, match="runs_per_trial should be integer"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs=10,
            runs_per_trial=1.2,
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
        )
    # max_epochs not right format
    with pytest.raises(ValueError, match="max_epochs should be integer"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs="10",
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
        )
    # train_prop not right format
    with pytest.raises(ValueError, match="train_prop should be float"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs=10,
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
            train_prop=1,
        )
    # seed wrong format
    with pytest.raises(ValueError, match="seed should be integer or None"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs=10,
            save_dir="tests/hyper_tune_test_out",
            mod_name="hyper_tune",
            train_prop=0.8,
            seed="2",
        )
    # save_dir wrong format
    with pytest.raises(ValueError, match="save_dir should be string"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs=10,
            save_dir=2,
            mod_name="hyper_tune",
            train_prop=0.8,
        )
    # mod_name wrong format
    with pytest.raises(ValueError, match="mod_name should be string"):
        pop_finder.hyper_tune(
            infile=infile_all,
            sample_data=sample_data2,
            max_epochs=10,
            save_dir="tests/hyper_tune_test_out",
            mod_name=2,
            train_prop=0.8,
        )


def test_kfcv():

    report = pop_finder.kfcv(
        infile=infile_all,
        sample_data=sample_data2,
        n_splits=3,
        n_reps=1,
        patience=10,
        max_epochs=10,
        save_dir="tests/kfcv_test_output",
        mod_path="hyper_tune_test_out",
    )

    # Check output in correct format
    assert isinstance(report, pd.DataFrame)

    # Check that two outputs are created with ensemble
    report, ensemble_report = pop_finder.kfcv(
        infile=infile_all,
        sample_data=sample_data2,
        n_splits=3,
        n_reps=1,
        ensemble=True,
        nbags=2,
        patience=10,
        max_epochs=10,
        save_dir="tests/kfcv_test_output",
        mod_path="hyper_tune_test_out",
    )

    assert isinstance(report, pd.DataFrame)
    assert isinstance(ensemble_report, pd.DataFrame)

    # Check input errors
    # infile does not exist
    with pytest.raises(ValueError, match="path to infile does not exist"):
        pop_finder.kfcv(
            infile="hello.txt",
            sample_data=sample_data2,
            n_splits=3,
            n_reps=1,
            patience=10,
            max_epochs=10,
            save_dir="tests/kfcv_test_output",
            mod_path="hyper_tune_test_out",
        )
    # sample_data does not exist
    with pytest.raises(ValueError, match="path to sample_data incorrect"):
        pop_finder.kfcv(
            infile=infile_all,
            sample_data="hello.txt",
            n_splits=3,
            n_reps=1,
            patience=10,
            max_epochs=10,
            save_dir="tests/kfcv_test_output",
            mod_path="hyper_tune_test_out",
        )
    # n_splits wrong format
    with pytest.raises(ValueError, match="n_splits should be an integer"):
        pop_finder.kfcv(
            infile=infile_all,
            sample_data=sample_data2,
            n_splits=1.5,
            n_reps=1,
            patience=10,
            max_epochs=10,
            save_dir="tests/kfcv_test_output",
            mod_path="hyper_tune_test_out",
        )
    # n_reps wrong format
    with pytest.raises(ValueError, match="n_reps should be an integer"):
        pop_finder.kfcv(
            infile=infile_all,
            sample_data=sample_data2,
            n_splits=3,
            n_reps=1.5,
            patience=10,
            max_epochs=10,
            save_dir="tests/kfcv_test_output",
            mod_path="hyper_tune_test_out",
        )
    # ensemble wrong format
    with pytest.raises(ValueError, match="ensemble should be a boolean"):
        pop_finder.kfcv(
            infile=infile_all,
            sample_data=sample_data2,
            n_splits=3,
            n_reps=1,
            ensemble="True",
            patience=10,
            max_epochs=10,
            save_dir="tests/kfcv_test_output",
            mod_path="hyper_tune_test_out",
        )
    # save_dir wrong format
    with pytest.raises(ValueError, match="save_dir should be a string"):
        pop_finder.kfcv(
            infile=infile_all,
            sample_data=sample_data2,
            n_splits=3,
            n_reps=1,
            patience=10,
            max_epochs=10,
            save_dir=2,
            mod_path="hyper_tune_test_out",
        )
    # n_splits > 1
    with pytest.raises(ValueError, match="n_splits must be greater than 1"):
        pop_finder.kfcv(
            infile=infile_all,
            sample_data=sample_data2,
            n_splits=1,
            n_reps=1,
            patience=10,
            max_epochs=10,
            save_dir="tests/kfcv_test_output",
            mod_path="hyper_tune_test_out",
        )
    # n_splits cannot be greater than smallest pop
    with pytest.raises(
        ValueError,
        match="n_splits cannot be greater than number of samples",
    ):
        pop_finder.kfcv(
            infile=infile_all,
            sample_data=sample_data2,
            n_splits=10,
            n_reps=1,
            patience=10,
            max_epochs=10,
            save_dir="tests/kfcv_test_output",
            mod_path="hyper_tune_test_out",
        )


def test_pop_finder():

    test_dict = pop_finder.pop_finder(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        unknowns=unknowns,
        ukgen=ukgen,
        save_dir="tests/test_output",
        max_epochs=10,
    )

    assert isinstance(test_dict, dict)

    test_dict, tot_bag_df = pop_finder.pop_finder(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        unknowns=unknowns,
        ukgen=ukgen,
        ensemble=True,
        nbags=2,
        save_dir="tests/test_output",
        max_epochs=10,
    )

    assert isinstance(test_dict, dict)
    assert isinstance(tot_bag_df, pd.DataFrame)

    # Check inputs
    with pytest.raises(ValueError, match="y_train is not a pandas dataframe"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=2,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="y_train exists, but is empty"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train_empty,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="y_test is not a pandas dataframe"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=2,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="y_test exists, but is empty"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test_empty,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="X_train is not a numpy array"):
        pop_finder.pop_finder(
            X_train=2,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="X_train exists, but is empty"):
        pop_finder.pop_finder(
            X_train=X_train_empty,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="X_test is not a numpy array"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=2,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="X_test exists, but is empty"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test_empty,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="ukgen is not a numpy array"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=2,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="ukgen exists, but is empty"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen_empty,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="unknowns is not pandas dataframe"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns="unknowns",
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="unknowns exists, but is empty"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns_empty,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="ensemble should be a boolean"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            ensemble="True",
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="try_stacking should be a boolean"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            try_stacking="True",
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="nbags should be an integer"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            ensemble=True,
            nbags=1.5,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="train_prop should be a float"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            train_prop=1,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="predict should be a boolean"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            predict="True",
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="save_dir should be a string"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir=2,
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="save_weights should be a boolean"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_weights="True",
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="patience should be an integer"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            patience=5.6,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="batch_size should be an integer"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            batch_size=5.6,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="max_epochs should be an integer"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            max_epochs=5.6,
            save_dir="tests/test_output",
        )
    with pytest.raises(ValueError, match="plot_history should be a boolean"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            plot_history="True",
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError,
                       match="mod_path should be a string or None"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            mod_path=2,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="unknowns is not pandas dataframe"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns="hello",
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="unknowns exists, but is empty"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns_empty,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
        )
    with pytest.raises(ValueError, match="train_prop is too high"):
        pop_finder.pop_finder(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            unknowns=unknowns,
            ukgen=ukgen,
            save_dir="tests/test_output",
            max_epochs=10,
            train_prop=0.99,
            seed=1234,
        )


def test_run_neural_net():

    save_path = "tests/test_output"
    pop_finder.run_neural_net(
        infile_all,
        sample_data2,
        patience=10,
        max_epochs=2,
        save_dir=save_path,
    )
    # Check correct files are created
    assert os.path.isfile(save_path + "/metrics.csv")
    assert os.path.isfile(save_path + "/pop_assign.csv")
    shutil.rmtree(save_path)

    pop_finder.run_neural_net(
        infile_all,
        sample_data2,
        patience=10,
        max_epochs=2,
        ensemble=True,
        nbags=2,
        try_stacking=True,
        save_dir=save_path,
    )
    # Check correct files are created
    assert os.path.isfile(save_path + "/ensemble_test_results.csv")
    assert os.path.isfile(save_path + "/pop_assign_ensemble.csv")
    assert os.path.isfile(save_path + "/metrics.csv")
    assert os.path.isfile(save_path + "/pop_assign_freqs.csv")
    shutil.rmtree(save_path)

    # Check inputs
    with pytest.raises(ValueError, match="Path to infile does not exist"):
        pop_finder.run_neural_net(
            infile="hello",
            sample_data=sample_data2,
            patience=10,
            max_epochs=2,
            save_dir=save_path,
        )
    with pytest.raises(ValueError, match="Path to sample_data does not exist"):
        pop_finder.run_neural_net(
            infile_all,
            sample_data="hello",
            patience=10,
            max_epochs=2,
            save_dir=save_path,
        )
    with pytest.raises(ValueError,
                       match="save_allele_counts should be a boolean"):
        pop_finder.run_neural_net(
            infile_all,
            sample_data2,
            save_allele_counts="True",
            patience=10,
            max_epochs=2,
            save_dir=save_path,
        )
    with pytest.raises(ValueError,
                       match="mod_path should either be a string or None"):
        pop_finder.run_neural_net(
            infile_all,
            sample_data2,
            mod_path=2,
            patience=10,
            max_epochs=2,
            save_dir=save_path,
        )
    with pytest.raises(ValueError, match="Path to mod_path does not exist"):
        pop_finder.run_neural_net(
            infile_all,
            sample_data2,
            mod_path="hello",
            patience=10,
            max_epochs=2,
            save_dir=save_path,
        )
    with pytest.raises(ValueError, match="train_prop should be a float"):
        pop_finder.run_neural_net(
            infile_all,
            sample_data2,
            patience=10,
            max_epochs=2,
            save_dir=save_path,
            train_prop=1,
        )
    with pytest.raises(ValueError, match="train_prop is too high"):
        pop_finder.run_neural_net(
            infile_all,
            sample_data2,
            patience=10,
            max_epochs=2,
            save_dir=save_path,
            train_prop=0.99,
        )


def test_assign_plot():

    # Check inputs
    with pytest.raises(ValueError, match="save_dir should be string"):
        pop_finder.assign_plot(save_dir=2)
    with pytest.raises(ValueError, match="ensemble should be boolean"):
        pop_finder.assign_plot(save_dir="hello", ensemble="True")
    with pytest.raises(ValueError, match="col_scheme should be string"):
        pop_finder.assign_plot(save_dir="hello",
                               ensemble=False,
                               col_scheme=1)
    with pytest.raises(
        ValueError,
        match="pop_assign_freqs.csv does not exist in save_dir"
    ):
        pop_finder.assign_plot(save_dir="hello", ensemble=True)
    with pytest.raises(ValueError,
                       match="pop_assign.csv does not exist in save_dir"):
        pop_finder.assign_plot(save_dir="hello", ensemble=False)


def test_structure_plot():

    # Check outputs
    pop_finder.structure_plot(save_dir="tests/test_inputs/kfcv_test_output")
    assert os.path.exists(
        "tests/test_inputs/kfcv_test_output/structure_plot.png")
    if os.path.exists(
        "tests/test_inputs/kfcv_test_output/structure_plot.png"
    ):
        os.remove(
            "tests/test_inputs/kfcv_test_output/structure_plot.png"
        )

    pop_finder.structure_plot(
        save_dir="tests/test_inputs/kfcv_ensemble_test_output",
        ensemble=True
    )
    assert os.path.exists(
        "tests/test_inputs/kfcv_ensemble_test_output/structure_plot.png"
    )
    if os.path.exists(
        "tests/test_inputs/kfcv_ensemble_test_output/structure_plot.png"
    ):
        os.remove(
            "tests/test_inputs/kfcv_ensemble_test_output/structure_plot.png"
        )

    # Check inputs
    with pytest.raises(ValueError,
                       match="Path to ensemble_preds does not exist"):
        pop_finder.structure_plot(save_dir="incorrect", ensemble=True)

    with pytest.raises(ValueError,
                       match="Path to preds does not exist"):
        pop_finder.structure_plot(save_dir="incorrect",
                                  ensemble=False)

    with pytest.raises(ValueError,
                       match="col_scheme should be a string"):
        pop_finder.structure_plot(
            save_dir="tests/test_inputs/kfcv_test_output",
            ensemble=False, col_scheme=2
        )


def test_contour_classifier():

    with pytest.raises(ValueError, match="save_dir does not exist"):
        contour_classifier.contour_classifier(
            sample_data=sample_data1, save_dir="incorrect"
        )

    with pytest.raises(ValueError, match="path to sample_data incorrect"):
        contour_classifier.contour_classifier(
            sample_data="incorrect", save_dir="tests/test_inputs/test_out"
        )

    with pytest.raises(ValueError, match="path to genetic data incorrect"):
        contour_classifier.contour_classifier(
            sample_data=sample_data1,
            run_locator=True,
            gen_dat="incorrect",
            save_dir="tests/test_inputs/test_out",
        )

    with pytest.raises(ValueError, match="Cannot use hdf5 file"):
        contour_classifier.contour_classifier(
            sample_data=sample_data1,
            run_locator=True,
            gen_dat=infile_all,
            save_dir="tests/test_inputs/test_out",
        )

    with pytest.raises(ValueError, match="bootstraps"):
        contour_classifier.contour_classifier(
            sample_data=sample_data1,
            nboots=25,
            save_dir="tests/test_inputs/test_out",
            multi_iter=1,
        )

    with pytest.raises(ValueError, match="bootstraps"):
        contour_classifier.contour_classifier(
            sample_data=sample_data1,
            nboots=25,
            save_dir="tests/test_inputs/test_out",
            multi_iter=1,
        )

    with pytest.raises(
        ValueError,
        match="Something went wrong with the prediction data"
    ):
        contour_classifier.contour_classifier(
            sample_data=sample_data3,
            save_dir="tests/test_inputs/test_out"
        )

    with pytest.raises(
        ValueError,
        match="sample_data file should have columns x, y, pop, and sampleID"
    ):
        contour_classifier.contour_classifier(
            sample_data=sample_data4,
            save_dir="tests/test_inputs/test_out"
        )

    with pytest.raises(Exception,
                       match="Too few points to generate contours"):
        contour_classifier.contour_classifier(
            sample_data=sample_data2,
            run_locator=True,
            gen_dat=infile_all_vcf,
            nboots=1,
            max_epochs=1,
            save_dir="tests/test_inputs/test_out",
        )

    class_df = contour_classifier.contour_classifier(
        sample_data=sample_data2,
        save_dir="tests/test_inputs/test_out"
    )
    assert isinstance(class_df, pd.core.frame.DataFrame)
    assert (class_df.columns == ["sampleID",
                                 "classification",
                                 "kd_estimate"]).all()
    assert (class_df["kd_estimate"] <= 1).all()
    assert (class_df["kd_estimate"] >= 0).all()


def test_cont_finder():

    pred_dat = pd.read_csv(pred_path)
    pred_dat = pred_dat.rename({"x": "pred_x", "y": "pred_y"}, axis=1)
    true_lab = pd.read_csv(sample_data1, sep="\t")
    test_dat = pred_dat[pred_dat["sampleID"] == "LESP_65"]
    d_x = (max(test_dat["pred_x"]) - min(test_dat["pred_x"])) / 10
    d_y = (max(test_dat["pred_y"]) - min(test_dat["pred_y"])) / 10
    test_xlim = min(test_dat["pred_x"]) - d_x, max(test_dat["pred_x"]) + d_x
    test_ylim = min(test_dat["pred_y"]) - d_y, max(test_dat["pred_y"]) + d_y
    X, Y = np.mgrid[
        test_xlim[0]:test_xlim[1]:200j, test_ylim[0]:test_ylim[1]:200j
    ]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([test_dat["pred_x"], test_dat["pred_y"]])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    new_z = Z / np.max(Z)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    cset = ax.contour(X, Y, new_z, 10, colors="black")
    cset.levels = -np.sort(-cset.levels)
    res = contour_classifier.cont_finder(true_lab, cset)
    assert len(res) == 2
    assert res[0] == "Baccalieu"
    assert res[1] == 0.4
    plt.close()


def test_kfcv_contour():

    with pytest.raises(ValueError, match="path to sample_data incorrect"):
        contour_classifier.kfcv(
            sample_data="incorrect",
            gen_dat=infile_all_vcf,
            save_dir="tests/test_inputs/kfcv",
        )

    pred_labels, true_labels, report = contour_classifier.kfcv(
        sample_data=sample_data1,
        gen_dat=infile_all_vcf,
        n_splits=2,
        n_runs=2,
        max_epochs=1,
        nboots=10,
        save_dir="tests/test_inputs/kfcv",
    )

    true_dat = pd.read_csv(sample_data1, sep="\t")
    assert len(pred_labels) == len(true_labels)
    # Because function was run for 2 iters
    assert len(true_dat) * 2 == len(pred_labels)
    assert isinstance(report, pd.core.frame.DataFrame)
