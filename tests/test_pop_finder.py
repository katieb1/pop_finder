from pop_finder import __version__
from pop_finder import pop_finder
from pop_finder import read
from pop_finder import hp_tuning
from pop_finder import contour_classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pytest

# Path to helper data
infile_all = "tests/test_inputs/onlyAtl_500.recode.vcf.locator.hdf5"
infile_all_vcf = "tests/test_inputs/onlyAtl_500.recode.vcf"
infile_kfcv = "tests/test_inputs/onlyAtl_500_kfcv.recode.vcf"
sample_data1 = "tests/test_inputs/onlyAtl_truelocs.txt"
sample_data2 = "tests/test_inputs/onlyAtl_truelocs_NAs.txt"
sample_data3 = "tests/test_inputs/onlyAtl_truelocs_extracol.txt"
pred_path = "tests/test_inputs/test_out/loc_boot0_predlocs.txt"


def test_version():
    assert __version__ == "0.1.15"


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

    with pytest.raises(ValueError,
                       match="sample_data file is in wrong format"):
        contour_classifier.contour_classifier(
            sample_data=sample_data3,
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


def test_kfcv():

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
