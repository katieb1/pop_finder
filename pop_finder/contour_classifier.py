import locator_mod
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import itertools


def contour_classifier(
    sample_data,
    num_contours=10,
    run_locator=False,
    gen_dat=None,
    nboots=50,
    return_plots=True,
    return_df=True,
    save_dir="out",
    multi_iter=1,
    **kwargs,
):
    """
    Wrapper function that runs locator to generate a density of predictions,
    then uses contour lines to choose the most likely population.

    Parameters
    ----------
    sample_data : string
        Filepath to input file containing coordinates and populations of
        samples, including individuals from unknown populations.
    num_contours : int
        Number of contours to generate and search for closest population in
        (Default=10).
    run_locator : boolean
        Run locator and use outputs to generate classifications. If set to
        False, then will look in specified save_dir for the *_predlocs.txt
        files from a previous locator run. If set to True, ensure that
        gen_dat is not None (Default=False).
    gen_dat : string
        Filepath to input genetic data in VCF format (Default=None).
    nboots : int
        Number of bootstrap iterations (Default=50).
    return_plots : boolean
        Return contour plots of prediction densities overlayed with true
        population locations (Default=True).
    return_df : boolean
        If true, saves the results in a csv file in the save_dir folder
        (Default=True).
    save_dir : string
        Folder to save results. Folder should already be in directory. If
        using results from multiple iterations, only include prefix (e.g.
        'out' for 'out1', 'out2', 'out3'; Default='out').
    multi_iter : int
        How many times to run locator to get predictions. If the sample size
        is small, it is better to run multiple iterations (Default=1).

    Returns
    -------
    class_df : pd.DataFrame
        Dataframe containing classifications of samples of unknown origin.
    """

    # Check if save_dir exists
    if os.path.isdir(save_dir) and os.path.isdir(save_dir + "1") is not True:
        raise ValueError("save_dir does not exist")

    # Check is sample_data path exists
    if isinstance(sample_data, pd.DataFrame) is False:
        if os.path.exists(sample_data) is False:
            raise ValueError("path to sample_data incorrect")

    if run_locator is True:

        # Check components
        if os.path.exists(gen_dat) is not True:
            raise ValueError("path to genetic data incorrect")

        for i in range(1, multi_iter + 1):

            if os.path.exists(save_dir + str(i)):
                shutil.rmtree(save_dir + str(i))
            os.makedirs(save_dir + str(i))

            locator_mod.locator(
                sample_data,
                gen_dat,
                bootstrap=True,
                nboots=nboots,
                out=save_dir + str(i),
                **kwargs,
            )

        out_list = []

        for i in range(1, multi_iter + 1):
            for j in range(nboots):
                out_list.append(
                    save_dir + str(i) + "/loc_boot" + str(j) + "_predlocs.txt"
                )

        with open(out_list[0], "a") as outfile:
            for names in out_list[1:]:
                with open(names) as infile:
                    string = ""
                    outfile.write(string.join(infile.readlines()[1:]))

    else:

        if multi_iter == 1:

            if os.path.isdir(save_dir) is True:

                # Check to make sure right number of boots selected
                if os.path.exists(
                    save_dir + "/loc_boot" + str(nboots + 1) + "_predlocs.txt"
                ):
                    raise ValueError(
                        "Number of bootstraps in output directory does not\
                    match nboots specified"
                    )

                out_list = [
                    save_dir + "/loc_boot" + str(i) + "_predlocs.txt"
                    for i in range(nboots)
                ]

                if sum(1 for line in open(out_list[0])) != sum(
                    1 for line in open(out_list[1])
                ):

                    with open(out_list[0], "a") as outfile:
                        for names in out_list[1:]:
                            with open(names) as infile:
                                string = ""
                                outfile.write(
                                    string.join(
                                        infile.readlines()[1:]
                                    )
                                )

            else:

                # Check to make sure right number of boots selected
                if os.path.exists(
                    save_dir + "1/loc_boot" + str(nboots + 1) + "_predlocs.txt"
                ):
                    raise ValueError(
                        "Number of bootstraps in output directory does not\
                    match nboots specified"
                    )

                out_list = [
                    save_dir + "1/loc_boot" + str(i) + "_predlocs.txt"
                    for i in range(nboots)
                ]

                if sum(1 for line in open(out_list[0])) != sum(
                    1 for line in open(out_list[1])
                ):

                    with open(out_list[0], "a") as outfile:
                        for names in out_list[1:]:
                            with open(names) as infile:
                                string = ""
                                outfile.write(
                                    string.join(
                                        infile.readlines()[1:]
                                    )
                                )

        if multi_iter > 1:
            out_list = []

            for i in range(1, multi_iter + 1):
                for j in range(nboots):
                    out_list.append(
                        save_dir+str(i)+"/loc_boot"+str(j)+"_predlocs.txt"
                    )

            if sum(1 for line in open(out_list[0])) != sum(
                1 for line in open(out_list[1])
            ):

                with open(out_list[0], "a") as outfile:
                    for names in out_list[1:]:
                        with open(names) as infile:
                            string = ""
                            outfile.write(string.join(infile.readlines()[1:]))

    # Convert input data file (true locs) into correct file for contour wrapper
    if isinstance(sample_data, pd.DataFrame) is not True:
        true_dat = pd.read_csv(sample_data, sep="\t")
    else:
        true_dat = sample_data

    if len(true_dat.columns) != 4:
        raise ValueError(
            "sample_data file is in wrong format - make sure\
        it is tab-delimited with columns x, y, pop, and sampleID"
        )

    # Find number of NAs (samples for assignment)
    num_pred = sum(true_dat["x"].isna())
    true_dat = true_dat[["x", "y", "pop"]].dropna().drop_duplicates()

    # Wrangle prediction data
    pred_dat = pd.read_csv(out_list[0])

    if len(pred_dat) != num_pred * nboots * multi_iter:
        raise ValueError("Something went wrong with the prediction data")

    pred_dat = pred_dat.rename({"x": "pred_x", "y": "pred_y"}, axis=1)

    # Create dataframe to fill with classifications
    class_dat = {"sampleID": [], "classification": [], "kd_estimate": []}

    for sample in pred_dat["sampleID"].drop_duplicates():
        print(f"Calculating prediction for {sample} ...")
        tmp_dat = pred_dat[pred_dat["sampleID"] == sample]
        class_dat["sampleID"].append(sample)

        d_x = (max(tmp_dat["pred_x"]) - min(tmp_dat["pred_x"])) / 5
        d_y = (max(tmp_dat["pred_y"]) - min(tmp_dat["pred_y"])) / 5
        xlim = min(tmp_dat["pred_x"]) - d_x, max(tmp_dat["pred_x"]) + d_x
        ylim = min(tmp_dat["pred_y"]) - d_y, max(tmp_dat["pred_y"]) + d_y

        X, Y = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]

        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([tmp_dat["pred_x"], tmp_dat["pred_y"]])

        try:
            kernel = stats.gaussian_kde(values)
        except ValueError as e:
            raise Exception("Too few points to generate contours") from e

        Z = np.reshape(kernel(positions).T, X.shape)
        new_z = Z / np.max(Z)

        # Plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        cset = ax.contour(X, Y, new_z, levels=num_contours, colors="black")

        cset.levels = -np.sort(-cset.levels)
        for pop in true_dat["pop"].values:
            x = true_dat[true_dat["pop"] == pop]["x"].values[0]
            y = true_dat[true_dat["pop"] == pop]["y"].values[0]
            plt.scatter(x, y, cmap="inferno", label=pop)
        ax.clabel(cset, cset.levels, inline=1, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.title(sample)
        plt.legend()

        # Find predicted pop
        pred_pop, kd = cont_finder(true_dat, cset)
        class_dat["classification"].append(pred_pop)
        class_dat["kd_estimate"].append(kd)

        if return_plots is True:
            plt.savefig(save_dir + "/contour_" + sample + ".png", format="png")

    class_df = pd.DataFrame(class_dat)

    if return_df is True:
        class_df.to_csv(save_dir + "/results.csv")

    return class_df


def cont_finder(true_dat, cset):
    """
    Finds population in densest contour.

    Parameters
    ----------
    true_dat : pd.DataFrame
        Dataframe containing x and y coordinates of all populations in
        training set.
    cset : matplotlib.contour.QuadContourSet
        Contour values for each contour polygon.

    Returns
    pred_pop : string
        Name of population in densest contour.
    """

    cont_dict = {"pop": [], "cont": []}

    for pop in true_dat["pop"].values:
        cont_dict["pop"].append(pop)
        cont = 0
        point = np.array(
            [
                [
                    true_dat[true_dat["pop"] == pop]["x"].values[0],
                    true_dat[true_dat["pop"] == pop]["y"].values[0],
                ]
            ]
        )

        for i in range(1, len(cset.allsegs)):
            for j in range(len(cset.allsegs[i])):
                path = matplotlib.path.Path(cset.allsegs[i][j].tolist())
                inside = path.contains_points(point)
                if inside[0] is True:
                    cont = i
                    break
                else:
                    next
        cont_dict["cont"].append(np.round(cset.levels[cont], 2))

    pred_pop = cont_dict["pop"][np.argmin(cont_dict["cont"])]

    return pred_pop, min(cont_dict["cont"])


def kfcv(
    sample_data,
    gen_dat,
    n_splits=5,
    n_runs=5,
    return_plot=True,
    save_dir="kfcv",
    **kwargs,
):
    """
    Runs K-fold Cross-Validation and returns classification
    report on model performance with given data.

    Parameters
    ----------
    sample_data : string
        Path to tab-delimited file containing known sample
        information, including x, y, pop, and sampleID columns.
        NAs can be included in this file, but they will not be
        used in the analysis.
    gen_dat : string
        Path to VCF file containing genetic information for all
        samples of known origin. Do not include samples of unknown
        origin in this file.
    n_splits : int
        Number of folds for the K-fold cross-validation (Default=5).
    n_runs : int
        Number of times to run K-fold cross-validation (Default=5).
    return_plot : boolean
        Return confusion matrix plot in output folder (Default=True).
    save_dir : string
        Path to output folder where confusion matrix and classification
        report will be stored.
    **kwargs
        Arguments for locator. See locator documentation for details.

    Returns
    -------
    report : pd.DataFrame
        Classification report containing precision, recall, F1 score,
        and accuracy for the model.
    """
    # Create results directory
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Partition data into k-folds for each run
    true_dat = pd.read_csv(sample_data, sep="\t", engine="python")

    # Drop samples of unknown origin from kfcv calculations
    true_dat = true_dat.dropna()

    pred_labels = []
    true_labels = []
    for i in range(n_runs):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for train_index, test_index in skf.split(
            true_dat.index.values, true_dat["pop"]
        ):
            k = true_dat.copy()
            k.iloc[test_index, 0:3] = np.NaN
            k.to_csv("kfcv_tmp.csv", sep="\t", index=False)

            # k becomes the sample data that goes into locator
            class_df = contour_classifier(
                sample_data="kfcv_tmp.csv",
                run_locator=True,
                gen_dat=gen_dat,
                return_plots=False,
                return_df=False,
                save_dir="kfcv",
                **kwargs,
            )
            all_dat = class_df.merge(true_dat)

            # Create vector of predicted and true labels
            pred_labels.append(all_dat["classification"].values)
            true_labels.append(all_dat["pop"].values)

    # From vector of predicted and true labels create report
    pred_labels = np.concatenate(pred_labels).ravel()
    true_labels = np.concatenate(true_labels).ravel()
    report = classification_report(
        true_labels, pred_labels, zero_division=1, output_dict=True
    )
    report = pd.DataFrame(report).transpose()
    report.to_csv(save_dir + "/classification_report.csv")

    # Plot confusion matrix
    if return_plot is True:

        cm = confusion_matrix(true_labels, pred_labels, normalize="true")
        cm = np.round(cm, 2)
        plt.style.use("default")
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.ylabel("True Pop")
        plt.xlabel("Pred Pop")
        plt.title("Confusion Matrix")
        tick_marks = np.arange(len(np.unique(true_labels)))
        plt.xticks(tick_marks, np.unique(true_labels))
        plt.yticks(tick_marks, np.unique(true_labels))
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.tight_layout()
        plt.savefig(save_dir + "/cm.png")

    return report
