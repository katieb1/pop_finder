# Neural network for pop assignment

# Load packages
import tensorflow.keras as tf
import tensorflow as T
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
import numpy as np
import pandas as pd
import allel
import zarr
import h5py
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
import itertools
from itertools import product
import shutil
import sys
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sn


def hyper_tune(
    infile,
    sample_data,
    max_trials=10,
    runs_per_trial=10,
    max_epochs=100,
    train_prop=0.8,
    seed=None,
    save_dir="out",
    mod_name="hyper_tune",
):
    """
    Tunes hyperparameters of keras model for population assignment.

    Paramters
    ---------
    infile : string
        Path to VCF file containing genetic data.
    sample_data : string
        Path to tab-delimited file containing columns x, y,
        pop, and sampleID.
    max_trials : int
        Number of trials to run for RandomSearch (Default=10).
    runs_per_trial : int
        Number of runs per trial for RandomSearch (Default=10).
    max_epochs : int
        Number of epochs to train model (Default=100).
    train_prop : float
        Proportion of data to train on. Remaining data will be kept
        as a test set and not used until final model is trained
        (Default=0.8).
    seed : int
        Random seed (Default=None).
    save_dir : string
        Directory to save output to (Default='out').
    mod_name : string
        Name of model in save directory (Default='hyper_tune').

    Returns
    -------
    best_mod : keras sequential model
        Best model from hyperparameter tuning
    y_train : pd.DataFrame
        training labels
    y_val : pd.DataFrame
        Validation labels
    """
    # Check input types
    if os.path.exists(infile) is False:
        raise ValueError("infile does not exist")
    if os.path.exists(sample_data) is False:
        raise ValueError("sample_data does not exist")
    if isinstance(max_trials, np.int) is False:
        raise ValueError("max_trials should be integer")
    if isinstance(runs_per_trial, np.int) is False:
        raise ValueError("runs_per_trial should be integer")
    if isinstance(max_epochs, np.int) is False:
        raise ValueError("max_epochs should be integer")
    if isinstance(train_prop, np.float) is False:
        raise ValueError("train_prop should be float")
    if isinstance(seed, np.int) is False and seed is not None:
        raise ValueError("seed should be integer or None")
    if isinstance(save_dir, str) is False:
        raise ValueError("save_dir should be string")
    if isinstance(mod_name, str) is False:
        raise ValueError("mod_name should be string")

    # Create save_dir if doesn't already exist
    print(f"Output will be saved to: {save_dir}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Read data
    samp_list, dc = read_data(
        infile=infile,
        sample_data=sample_data,
        save_allele_counts=False,
        kfcv=True,
    )

    # Train prop can't be greater than num samples
    if len(dc) * (1 - train_prop) < len(np.unique(samp_list["pops"])):
        raise ValueError("train_prop is too high; not enough samples for test")

    # Create test set that will be used to assess model performance later
    X_train_0, X_test, y_train_0, y_test = train_test_split(
        dc, samp_list, stratify=samp_list["pops"], train_size=train_prop
    )

    # Save train and test set to save_dir
    np.save(save_dir + "/X_train.npy", X_train_0)
    y_train_0.to_csv(save_dir + "/y_train.csv", index=False)
    np.save(save_dir + "/X_test.npy", X_test)
    y_test.to_csv(save_dir + "/y_test.csv", index=False)

    # Split data into training and hold-out test set
    X_train, X_val, y_train, y_val = train_test_split(
        dc,
        samp_list,
        stratify=samp_list["pops"],
        train_size=train_prop,
        random_state=seed,
    )

    # Make sure all classes represented in y_val
    if len(np.unique(y_train["pops"])) != len(np.unique(y_val["pops"])):
        raise ValueError(
            "Not all pops represented in validation set \
                          choose smaller value for train_prop."
        )

    # One hot encoding
    enc = OneHotEncoder(handle_unknown="ignore")
    y_train_enc = enc.fit_transform(
        y_train["pops"].values.reshape(-1, 1)).toarray()
    y_val_enc = enc.fit_transform(
        y_val["pops"].values.reshape(-1, 1)).toarray()
    popnames = enc.categories_[0]

    hypermodel = classifierHyperModel(
        input_shape=X_train.shape[1], num_classes=len(popnames)
    )

    # Overwrite contents of save_dir
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    tuner = RandomSearch(
        hypermodel,
        objective="val_loss",
        seed=seed,
        max_trials=max_trials,
        executions_per_trial=runs_per_trial,
        directory=save_dir,
        project_name=mod_name,
    )

    tuner.search(
        X_train - 1,
        y_train_enc,
        epochs=max_epochs,
        validation_data=(X_val - 1, y_val_enc),
    )

    best_mod = tuner.get_best_models(num_models=1)[0]
    tuner.get_best_models(num_models=1)[0].save(save_dir + "/best_mod")

    return best_mod, y_train, y_val


def kfcv(
    infile,
    sample_data,
    mod_path=None,
    n_splits=5,
    n_reps=5,
    ensemble=False,
    save_dir="kfcv_output",
    return_plot=True,
    save_allele_counts=False,
    **kwargs,
):
    """
    Runs K-fold cross-validation to get an accuracy estimate of the model.

    Parameters
    ----------
    infile : string
        Path to VCF or hdf5 file with genetic information
        for all samples (including samples of unknown origin).
    sample_data : string
        Path to input file with all samples present (including
        samples of unknown origin), which is a tab-delimited
        text file with columns x, y, pop, and sampleID.
    n_splits : int
        Number of folds in k-fold cross-validation
        (Default=5).
    n_reps : int
        Number of times to repeat k-fold cross-validation,
        creating the number of models in the ensemble
        (Default=5).
    ensemble : bool
        Whether to use ensemble of models of single model (Default=False).
    save_dir : string
        Directory where results will be stored (Default='kfcv_output').
    return_plot : boolean
        Returns a confusion matrix of correct assignments (Default=True).
    save_allele counts : boolean
        Whether or not to store derived allele counts in hdf5
        file (Default=False).
    **kwargs
        Keyword arguments for pop_finder function.

    Returns
    -------
    report : pd.DataFrame
        Classification report for all models.
    ensemble_report : pd.DataFrame
        Classification report for ensemble of models.
    """

    # Check inputs
    # Check is sample_data path exists
    if os.path.exists(sample_data) is False:
        raise ValueError("path to sample_data incorrect")

    # Make sure hdf5 file is not used as gen_dat
    if os.path.exists(infile) is False:
        raise ValueError("path to infile does not exist")

    # Check data types
    if isinstance(n_splits, np.int) is False:
        raise ValueError("n_splits should be an integer")
    if isinstance(n_reps, np.int) is False:
        raise ValueError("n_reps should be an integer")
    if isinstance(ensemble, bool) is False:
        raise ValueError("ensemble should be a boolean")
    if isinstance(save_dir, str) is False:
        raise ValueError("save_dir should be a string")

    # Check nsplits is > 1
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1")

    samp_list, dc = read_data(
        infile=infile,
        sample_data=sample_data,
        save_allele_counts=save_allele_counts,
        kfcv=True,
    )

    popnames = np.unique(samp_list["pops"])

    # Check there are more samples in the smallest pop than n_splits
    if n_splits > samp_list.groupby(["pops"]).agg(["count"]).min().values[0]:
        raise ValueError(
            "n_splits cannot be greater than number of samples in smallest pop"
        )

    # Create stratified k-fold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_reps)

    pred_labels = []
    true_labels = []

    pred_labels_ensemble = []
    true_labels_ensemble = []

    ensemble_preds = pd.DataFrame()
    preds = pd.DataFrame()

    fold_var = 1

    for t, v in rskf.split(dc, samp_list["pops"]):

        # Subset train and validation data
        X_train = dc[t, :] - 1
        X_val = dc[v, :] - 1
        y_train = samp_list.iloc[t]
        y_val = samp_list.iloc[v]

        if ensemble:
            test_dict, tot_bag_df = pop_finder(
                X_train,
                y_train,
                X_val,
                y_val,
                save_dir=save_dir,
                ensemble=True,
                **kwargs,
            )

            # Unit tests for results from pop_finder
            if bool(test_dict) is False:
                raise ValueError("Empty dictionary from pop_finder")

            if tot_bag_df.empty:
                raise ValueError("Empty dataframe from pop_finder")

            if len(test_dict) == 1:
                raise ValueError(
                    "pop_finder results consists of single dataframe\
                                  however ensemble set to True"
                )

            ensemble_preds = ensemble_preds.append(tot_bag_df)

        else:
            test_dict = pop_finder(
                X_train,
                y_train,
                X_val,
                y_val,
                save_dir=save_dir,
                **kwargs,
            )

            # Unit tests for results from pop_finder
            if bool(test_dict) is False:
                raise ValueError("Empty dictionary from pop_finder")

            if len(test_dict["df"]) != 1:
                raise ValueError(
                    "pop_finder results contains ensemble of models\
                                  should be a single dataframe"
                )

            preds = preds.append(test_dict["df"][0])

        tmp_pred_label = []
        tmp_true_label = []
        for i in range(0, len(test_dict["df"])):
            tmp_pred_label.append(
                test_dict["df"][i].iloc[
                    :, 0:len(popnames)
                ].idxmax(axis=1).values
            )
            tmp_true_label.append(test_dict["df"][i]["true_pops"].values)

        if ensemble:
            pred_labels_ensemble.append(
                tot_bag_df.iloc[:, 0:len(popnames)].idxmax(axis=1).values
            )
            true_labels_ensemble.append(tmp_true_label[0])

        pred_labels.append(np.concatenate(tmp_pred_label, axis=0))
        true_labels.append(np.concatenate(tmp_true_label, axis=0))

        fold_var += 1

    # return pred_labels, true_labels
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    report = classification_report(
        true_labels, pred_labels, zero_division=1, output_dict=True
    )
    report = pd.DataFrame(report).transpose()
    report.to_csv(save_dir + "/classification_report.csv")

    if ensemble:
        ensemble_preds.to_csv(save_dir + "/ensemble_preds.csv")

        true_labels_ensemble = np.concatenate(true_labels_ensemble)
        pred_labels_ensemble = np.concatenate(pred_labels_ensemble)
        ensemble_report = classification_report(
            true_labels_ensemble,
            pred_labels_ensemble,
            zero_division=1,
            output_dict=True,
        )
        ensemble_report = pd.DataFrame(ensemble_report).transpose()
        ensemble_report.to_csv(
            save_dir + "/ensemble_classification_report.csv")
    else:
        preds.to_csv(save_dir + "/preds.csv")

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
        plt.close()

        if ensemble:
            # Plot second confusion matrix
            cm = confusion_matrix(
                true_labels_ensemble, pred_labels_ensemble, normalize="true"
            )
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
            for i, j in itertools.product(
                range(cm.shape[0]), range(cm.shape[1])
            ):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            plt.tight_layout()
            plt.savefig(save_dir + "/ensemble_cm.png")
            plt.close()

    if ensemble:
        return report, ensemble_report
    else:
        return report


def pop_finder(
    X_train,
    y_train,
    X_test,
    y_test,
    unknowns=None,
    ukgen=None,
    ensemble=False,
    try_stacking=False,
    nbags=10,
    train_prop=0.8,
    mod_path=None,
    predict=False,
    save_dir="out",
    save_weights=False,
    patience=20,
    batch_size=32,
    max_epochs=100,
    gpu_number="0",
    plot_history=False,
    seed=None,
):
    """
    Trains classifier neural network, calculates accuracy on
    test set, and makes predictions.

    Parameters
    ----------
    X_train: np.array
        Array of genetic data corresponding to train samples.
    y_train: pd.DataFrame
        Dataframe of train samples, including columns for samples and pops.
    X_test: np.array
        Array of genetic data corresponding to test samples.
    y_test: pd.DataFrame
        Dataframe of test samples, including columns for samples and pops.
    unknowns: pd.DataFrame
        Dataframe of unknowns calculated from read_data (Default=None).
    ukgen : np.array
        Array of genetic data corresponding to unknown samples
        (Default=None).
    ensemble : boolean
        If set to true, will train an ensemble of models using
        bootstrap aggregating (Default=False).
    try_stacking : boolean
        Use weights to influence ensemble model decisions. Must have
        ensemble set to True to use. Use caution: with low test set sizes,
        can be highly inaccurate and overfit (Default=False).
    nbags : int
        Number of "bags" (models) to create for the bootstrap
        aggregating algorithm. This option only needs to be set if
        ensemble is set to True (Default=20).
    train_prop : float
        Proportion of samples used in training (Default=0.8).
    mod_path : string
        Default=None
    predict : boolean
        Predict on unknown data. Must have unknowns in sample_data to use
        this feature (Default=False).
    save_dir : string
        Directory to save results to (Default="out").
    save_weights : boolean
        Save model weights for later use (Default=False).
    patience : int
        How many epochs to wait before early stopping if loss has not
        improved (Default=20).
    batch_size : int
        Default=32,
    max_epochs : int
        Default=100
    gpu_number : string
        Not in use yet, coming soon (Default="0").
    plot_history : boolean
        Plot training / validation history (Default=False).
    seed : int
        Random seed for splitting data (Default=None).

    Returns
    -------
    test_dict : dict
        Dictionary with test results.
    tot_bag_df : pd.DataFrame
        Dataframe with test results from ensemble.
    """
    print(f"Output will be saved to: {save_dir}")

    # Check if data is in right format
    if isinstance(y_train, pd.DataFrame) is False:
        raise ValueError("y_train is not a pandas dataframe")
    if y_train.empty:
        raise ValueError("y_train exists, but is empty")
    if isinstance(y_test, pd.DataFrame) is False:
        raise ValueError("y_test is not a pandas dataframe")
    if y_test.empty:
        raise ValueError("y_test exists, but is empty")
    if isinstance(X_train, np.ndarray) is False:
        raise ValueError("X_train is not a numpy array")
    if len(X_train) == 0:
        raise ValueError("X_train exists, but is empty")
    if isinstance(X_test, np.ndarray) is False:
        raise ValueError("X_test is not a numpy array")
    if len(X_test) == 0:
        raise ValueError("X_test exists, but is empty")
    if isinstance(ensemble, bool) is False:
        raise ValueError("ensemble should be a boolean")
    if isinstance(try_stacking, bool) is False:
        raise ValueError("try_stacking should be a boolean")
    if isinstance(nbags, int) is False:
        raise ValueError("nbags should be an integer")
    if isinstance(train_prop, np.float) is False:
        raise ValueError("train_prop should be a float")
    if isinstance(predict, bool) is False:
        raise ValueError("predict should be a boolean")
    if isinstance(save_dir, str) is False:
        raise ValueError("save_dir should be a string")
    if isinstance(save_weights, bool) is False:
        raise ValueError("save_weights should be a boolean")
    if isinstance(patience, np.int) is False:
        raise ValueError("patience should be an integer")
    if isinstance(batch_size, np.int) is False:
        raise ValueError("batch_size should be an integer")
    if isinstance(max_epochs, np.int) is False:
        raise ValueError("max_epochs should be an integer")
    if isinstance(plot_history, bool) is False:
        raise ValueError("plot_history should be a boolean")
    if isinstance(mod_path, str) is False and mod_path is not None:
        raise ValueError("mod_path should be a string or None")

    # Create save directory
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # If unknowns are not none
    if unknowns is not None:

        # Check if exists
        if isinstance(unknowns, pd.DataFrame) is False:
            raise ValueError("unknowns is not pandas dataframe")
        if unknowns.empty:
            raise ValueError("unknowns exists, but is empty")
        if isinstance(ukgen, np.ndarray) is False:
            raise ValueError("ukgen is not a numpy array")
        if len(ukgen) == 0:
            raise ValueError("ukgen exists, but is empty")

        uksamples = unknowns["sampleID"].to_numpy()

    # Add info about test samples
    y_test_samples = y_test["samples"].to_numpy()
    y_test_pops = y_test["pops"].to_numpy()

    # One hot encode test values
    enc = OneHotEncoder(handle_unknown="ignore")
    y_test_enc = enc.fit_transform(
        y_test["pops"].values.reshape(-1, 1)).toarray()
    popnames = enc.categories_[0]

    # results storage
    TEST_LOSS = []
    TEST_ACCURACY = []
    TEST_95CI = []
    yhats = []
    ypreds = []
    test_dict = {"count": [], "df": []}
    pred_dict = {"count": [], "df": []}
    top_pops = {"df": [], "pops": []}

    if ensemble:
        for i in range(nbags):
            n_prime = np.int(np.ceil(len(X_train) * 0.8))
            good_bag = False

            while good_bag is False:
                bag_X = np.zeros(shape=(n_prime, X_train.shape[1]))
                bag_y = pd.DataFrame({"samples": [], "pops": [], "order": []})
                for j in range(0, n_prime):
                    ind = np.random.choice(len(X_train))
                    bag_X[j] = X_train[ind]
                    bag_y = bag_y.append(y_train.iloc[ind])
                dup_pops_df = bag_y.groupby(["pops"]).agg(["count"])
                if (
                    pd.Series(popnames).isin(bag_y["pops"]).all()
                    and (dup_pops_df[("samples", "count")] > 1).all()
                ):
                    # Create validation set from training set
                    bag_X, X_val, bag_y, y_val = train_test_split(
                        bag_X, bag_y, stratify=bag_y["pops"],
                        train_size=train_prop
                    )
                    if (
                        pd.Series(popnames).isin(bag_y["pops"]).all()
                        and pd.Series(popnames).isin(y_val["pops"]).all()
                    ):
                        good_bag = True

            enc = OneHotEncoder(handle_unknown="ignore")
            bag_y_enc = enc.fit_transform(
                bag_y["pops"].values.reshape(-1, 1)).toarray()
            y_val_enc = enc.fit_transform(
                y_val["pops"].values.reshape(-1, 1)).toarray()

            if mod_path is None:
                model = tf.Sequential()
                model.add(tf.layers.BatchNormalization(
                    input_shape=(bag_X.shape[1],)))
                model.add(tf.layers.Dense(128, activation="elu"))
                model.add(tf.layers.Dense(128, activation="elu"))
                model.add(tf.layers.Dense(128, activation="elu"))
                model.add(tf.layers.Dropout(0.25))
                model.add(tf.layers.Dense(128, activation="elu"))
                model.add(tf.layers.Dense(128, activation="elu"))
                model.add(tf.layers.Dense(128, activation="elu"))
                model.add(tf.layers.Dense(len(popnames), activation="softmax"))
                aopt = tf.optimizers.Adam(lr=0.0005)
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=aopt,
                    metrics="accuracy"
                )

            else:
                model = tf.models.load_model(mod_path + "/best_mod")

            # Create callbacks
            checkpointer = tf.callbacks.ModelCheckpoint(
                filepath=save_dir + "/checkpoint.h5",
                verbose=1,
                # save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                # monitor="loss",
                save_freq="epoch",
            )
            earlystop = tf.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0, patience=patience
            )
            reducelr = tf.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=int(patience / 3),
                verbose=1,
                mode="auto",
                min_delta=0,
                cooldown=0,
                min_lr=0,
            )
            callback_list = [checkpointer, earlystop, reducelr]

            # Train model
            history = model.fit(
                bag_X - 1,
                bag_y_enc,
                batch_size=int(batch_size),
                epochs=int(max_epochs),
                callbacks=callback_list,
                validation_data=(X_val - 1, y_val_enc),
                verbose=0,
            )

            # Load best model
            model.load_weights(save_dir + "/checkpoint.h5")

            if not save_weights:
                os.remove(save_dir + "/checkpoint.h5")

            # plot training history
            if plot_history:
                plt.switch_backend("agg")
                fig = plt.figure(figsize=(3, 1.5), dpi=200)
                plt.rcParams.update({"font.size": 7})
                ax1 = fig.add_axes([0, 0, 1, 1])
                ax1.plot(
                    history.history["val_loss"][3:],
                    "--",
                    color="black",
                    lw=0.5,
                    label="Validation Loss",
                )
                ax1.plot(
                    history.history["loss"][3:],
                    "-",
                    color="black",
                    lw=0.5,
                    label="Training Loss",
                )
                ax1.set_xlabel("Epoch")
                ax1.legend()
                fig.savefig(
                    save_dir + "/model" + i + "_history.pdf",
                    bbox_inches="tight"
                )
                plt.close()

            test_loss, test_acc = model.evaluate(X_test - 1, y_test_enc)

            yhats.append(model.predict(X_test - 1))

            test_df = pd.DataFrame(model.predict(X_test - 1))
            test_df.columns = popnames
            test_df["sampleID"] = y_test_samples
            test_df["true_pops"] = y_test_pops
            test_df["bag"] = i
            test_dict["count"].append(i)
            test_dict["df"].append(test_df)

            # Fill test lists with information
            TEST_LOSS.append(test_loss)
            TEST_ACCURACY.append(test_acc)

            if predict:
                ypreds.append(model.predict(ukgen))

                tmp_df = pd.DataFrame(model.predict(ukgen))
                tmp_df.columns = popnames
                tmp_df["sampleID"] = uksamples
                tmp_df["bag"] = i
                pred_dict["count"].append(i)
                pred_dict["df"].append(tmp_df)

                # Find top populations for each sample
                top_pops["df"].append(i)
                top_pops["pops"].append(
                    pred_dict["df"][i].iloc[
                        :, 0:len(popnames)
                    ].idxmax(axis=1)
                )

        # Collect yhats and ypreds for weighted ensemble
        yhats = np.array(yhats)

        if predict:
            ypreds = np.array(ypreds)

        # Get ensemble accuracy
        tot_bag_df = test_dict["df"][0].iloc[
            :, 0:len(popnames)
        ].copy()
        for i in range(0, len(test_dict["df"])):
            tot_bag_df += test_dict["df"][i].iloc[:, 0:len(popnames)]
        # Normalize values to be between 0 and 1
        tot_bag_df = tot_bag_df / nbags
        tot_bag_df["top_samp"] = tot_bag_df.idxmax(axis=1)
        tot_bag_df["sampleID"] = test_dict["df"][0]["sampleID"]
        tot_bag_df["true_pops"] = test_dict["df"][0]["true_pops"]
        ENSEMBLE_TEST_ACCURACY = np.sum(
            tot_bag_df["top_samp"] == tot_bag_df["true_pops"]
        ) / len(tot_bag_df)
        tot_bag_df.to_csv(save_dir + "/ensemble_test_results.csv")

        if predict:
            top_pops_df = pd.DataFrame(top_pops["pops"])
            top_pops_df.columns = uksamples
            top_freqs = {"sample": [], "freq": []}

            for samp in uksamples:
                top_freqs["sample"].append(samp)
                top_freqs["freq"].append(
                    top_pops_df[samp].value_counts() / len(top_pops_df)
                )

            # Save frequencies to csv for plotting
            top_freqs_df = pd.DataFrame(top_freqs["freq"]).fillna(0)
            top_freqs_df.to_csv(save_dir + "/pop_assign_freqs.csv")

            # Create table to assignments by frequency
            freq_df = pd.concat(
                [
                    pd.DataFrame(top_freqs["freq"]).max(axis=1),
                    pd.DataFrame(top_freqs["freq"]).idxmax(axis=1),
                ],
                axis=1,
            ).reset_index()
            freq_df.columns = ["Assigned Pop",
                               "Frequency",
                               "Sample ID"]
            freq_df.to_csv(save_dir + "/pop_assign_ensemble.csv",
                           index=False)

        # Metrics
        AVG_TEST_LOSS = np.round(np.mean(TEST_LOSS), 2)
        AVG_TEST_ACCURACY = np.round(np.mean(TEST_ACCURACY), 2)
        test_err = 1 - AVG_TEST_ACCURACY
        TEST_95CI = 1.96 * np.sqrt(
            (test_err * (1 - test_err)) / len(y_test_enc))

        best_score = "N/A"
        if try_stacking:

            def stacked_preds(yhats, weights):
                summed = np.tensordot(yhats, weights, axes=((0), (0)))
                result = np.argmax(summed, axis=1)
                return result

            # Get accuracy of weighted stacked model
            y_test_max = np.argmax(y_test_enc, axis=1)
            w = np.linspace(1, 10, 3)
            best_score, best_weights = 0.0, None
            for weights in product(w, repeat=nbags):
                if len(set(weights)) == 1:
                    continue
                result = np.linalg.norm(weights, 1)
                if result == 0.0:
                    weights = weights
                weights = weights / result
                yhats_max = stacked_preds(yhats, weights)
                score = accuracy_score(y_test_max, yhats_max)
                if score > best_score:
                    best_score, best_weights = score, weights
                    print(">%s %.3f" % (best_weights, best_score))
                if best_score == 1.0:
                    break

            # Predict on unknowns for weighted stacked model
            if predict:
                ypreds_max = stacked_preds(ypreds, best_weights)
                ypreds_df = pd.DataFrame(ypreds_max)
                ypreds_df["sampleID"] = uksamples
                ypreds_df["pops"] = popnames[ypreds_max]
                ypreds_df.to_csv(save_dir + "/stacked_results.csv")

        # Print metrics to csv
        print("Creating outputs...")
        metrics = pd.DataFrame(
            {
                "metric": [
                    "Ensemble accuracy",
                    "Weighted ensemble accuracy",
                    "Test accuracy",
                    "Test 95% CI",
                    "Test loss",
                ],
                "value": [
                    ENSEMBLE_TEST_ACCURACY,
                    best_score,
                    AVG_TEST_ACCURACY,
                    TEST_95CI,
                    AVG_TEST_LOSS,
                ],
            }
        )
        metrics.to_csv(save_dir + "/metrics.csv", index=False)
        return test_dict, tot_bag_df

    else:

        # Test if train_prop is too high
        if len(X_train) * (1 - train_prop) < 1:
            raise ValueError(
                "train_prop is too high; not enough values for test")

        # Split training data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, stratify=y_train["pops"],
            random_state=seed
        )

        # Make sure all classes represented in y_val
        if len(
            np.unique(y_train["pops"])
        ) != len(np.unique(y_val["pops"])):
            raise ValueError(
                "Not all pops represented in validation set \
                 choose smaller value for train_prop."
            )

        # One hot encoding
        enc = OneHotEncoder(handle_unknown="ignore")
        y_train_enc = enc.fit_transform(
            y_train["pops"].values.reshape(-1, 1)).toarray()
        y_val_enc = enc.fit_transform(
            y_val["pops"].values.reshape(-1, 1)).toarray()
        popnames = enc.categories_[0]

        # Use default model
        if mod_path is None:
            model = tf.Sequential()
            model.add(
                tf.layers.BatchNormalization(
                    input_shape=(X_train.shape[1],)))
            model.add(tf.layers.Dense(128, activation="elu"))
            model.add(tf.layers.Dense(128, activation="elu"))
            model.add(tf.layers.Dense(128, activation="elu"))
            model.add(tf.layers.Dropout(0.25))
            model.add(tf.layers.Dense(128, activation="elu"))
            model.add(tf.layers.Dense(128, activation="elu"))
            model.add(tf.layers.Dense(128, activation="elu"))
            model.add(tf.layers.Dense(len(popnames), activation="softmax"))
            aopt = tf.optimizers.Adam(lr=0.0005)
            model.compile(
                loss="categorical_crossentropy",
                optimizer=aopt,
                metrics="accuracy"
            )

        else:
            # Use pre-tuned model
            model = tf.models.load_model(mod_path + "/best_mod")

        # Create callbacks
        checkpointer = tf.callbacks.ModelCheckpoint(
            filepath=save_dir + "/checkpoint.h5",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq="epoch",
        )
        earlystop = tf.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=patience
        )
        reducelr = tf.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=int(patience / 3),
            verbose=1,
            mode="auto",
            min_delta=0,
            cooldown=0,
            min_lr=0,
        )
        callback_list = [checkpointer, earlystop, reducelr]

        # Train model
        history = model.fit(
            X_train - 1,
            y_train_enc,
            batch_size=int(batch_size),
            epochs=int(max_epochs),
            callbacks=callback_list,
            validation_data=(X_val - 1, y_val_enc),
            verbose=0,
        )

        # Load best model
        model.load_weights(save_dir + "/checkpoint.h5")

        if not save_weights:
            os.remove(save_dir + "/checkpoint.h5")

        # plot training history
        if plot_history:
            plt.switch_backend("agg")
            fig = plt.figure(figsize=(3, 1.5), dpi=200)
            plt.rcParams.update({"font.size": 7})
            ax1 = fig.add_axes([0, 0, 1, 1])
            ax1.plot(
                history.history["val_loss"][3:],
                "--",
                color="black",
                lw=0.5,
                label="Validation Loss",
            )
            ax1.plot(
                history.history["loss"][3:],
                "-",
                color="black",
                lw=0.5,
                label="Training Loss",
            )
            ax1.set_xlabel("Epoch")
            ax1.legend()
            fig.savefig(save_dir + "/history.pdf",
                        bbox_inches="tight")
            plt.close()

        tf.backend.clear_session()

        test_loss, test_acc = model.evaluate(X_test - 1, y_test_enc)
        test_df = pd.DataFrame(model.predict(X_test - 1))
        test_df.columns = popnames
        test_df["sampleID"] = y_test_samples
        test_df["true_pops"] = y_test_pops
        test_dict["count"].append(1)
        test_dict["df"].append(test_df)

        # Find confidence interval of best model
        test_err = 1 - test_acc
        test_95CI = 1.96 * np.sqrt(
            (test_err * (1 - test_err)) / len(y_test_enc))

        # Fill test lists with information
        TEST_LOSS.append(test_loss)
        TEST_ACCURACY.append(test_acc)
        TEST_95CI.append(test_95CI)

        print(
            f"Accuracy of model is {np.round(test_acc, 2)}\
            +/- {np.round(test_95CI,2)}"
        )

        if predict:
            tmp_df = pd.DataFrame(model.predict(ukgen))
            tmp_df.columns = popnames
            tmp_df["sampleID"] = uksamples
            tmp_df.to_csv(save_dir + "/pop_assign.csv", index=False)

        # Print metrics to csv
        print("Creating outputs...")
        metrics = pd.DataFrame(
            {
                "metric": [
                    "Test accuracy",
                    "Test 95% CI",
                    "Test loss",
                ],
                "value": [
                    np.round(TEST_ACCURACY, 2),
                    np.round(TEST_95CI, 2),
                    np.round(TEST_LOSS, 2),
                ],
            }
        )

        metrics.to_csv(save_dir + "/metrics.csv", index=False)
        return test_dict

    print("Process complete")


def run_neural_net(
    infile,
    sample_data,
    save_allele_counts=False,
    mod_path=None,
    train_prop=0.8,
    seed=None,
    **kwargs,
):
    """
    Uses input arguments from the command line to tune, train,
    evaluate an ensemble of neural networks, then predicts the
    population of origin for samples of unknown origin.

    Parameters
    ----------
    infile : string
        Path to VCF or hdf5 file with genetic information
        for all samples (including samples of unknown origin).
    sample_data : string
        Path to input file with all samples present (including
        samples of unknown origin), which is a tab-delimited
        text file with columns x, y, pop, and sampleID.
    save_allele counts : boolean
        Whether or not to store derived allele counts in hdf5
        file (Default=False).
    mod_path : string
        Path to tuned model. If set to None, uses default model
        (Default=None).
    train_prop : float
        Proportion of data to be used in training the model
        (Default=0.8).
    seed : int
        Random seed for splitting data (Default=None).
    **kwargs
        Keyword arguments for pop_finder function.
    """

    # Check if paths exist
    if os.path.exists(infile) is False:
        raise ValueError("Path to infile does not exist")
    if os.path.exists(sample_data) is False:
        raise ValueError("Path to sample_data does not exist")
    if isinstance(save_allele_counts, bool) is False:
        raise ValueError("save_allele_counts should be a boolean")
    if isinstance(mod_path, str) is False and mod_path is not None:
        raise ValueError("mod_path should either be a string or None")
    if isinstance(train_prop, np.float) is False:
        raise ValueError("train_prop should be a float")

    # Read data w unknowns so errors caught before training/tuning
    samp_list, dc, unknowns = read_data(
        infile=infile,
        sample_data=sample_data,
        save_allele_counts=save_allele_counts,
        kfcv=False,
    )

    unknown_inds = pd.array(unknowns["order"])
    ukgen = dc[unknown_inds, :] - 1

    if mod_path is None:

        dc_new = np.delete(dc, unknowns["order"].values, axis=0)

        # Check if train_prop too high
        if len(dc_new) * (1 - train_prop) < len(
            np.unique(samp_list["pops"])
        ):
            raise ValueError(
                "train_prop is too high; not enough samples for test"
            )

        # Split data into training and hold-out test set
        X_train, X_test, y_train, y_test = train_test_split(
            dc_new,
            samp_list,
            stratify=samp_list["pops"],
            train_size=train_prop,
            random_state=seed,
        )

        # Make sure all classes represented in y_val
        if len(
            np.unique(y_train["pops"])
        ) != len(np.unique(y_test["pops"])):
            raise ValueError(
                "Not all pops represented in test set \
                 choose smaller value for train_prop."
            )

    else:

        if os.path.exists(mod_path) is False:
            raise ValueError("Path to mod_path does not exist")

        X_train = np.load(mod_path + "/X_train.npy")
        y_train = pd.read_csv(mod_path + "/y_train.csv")
        X_test = np.load(mod_path + "/X_test.npy")
        y_test = pd.read_csv(mod_path + "/y_test.csv")

    pop_finder(
        X_train,
        y_train,
        X_test,
        y_test,
        unknowns=unknowns,
        ukgen=ukgen,
        predict=True,
        **kwargs,
    )


def read_data(infile, sample_data, save_allele_counts=False, kfcv=False):
    """
    Reads a .zarr, .vcf, or h5py file containing genetic data and
    creates subsettable data for a classifier neural network.

    Parameters
    ----------
    infile : string
        Path to the .zarr, .vcf, or h5py file.
    sample_data : string
        Path to .txt file containing sample information
        (columns are x, y, sampleID, and pop).
    save_allele_counts : boolean
        Saves derived allele count information (Default=False).
    kfcv : boolean
        If being used to test accuracy with k-fold cross-
        validation (i.e. no NAs in the sample data), set to
        True (Default=False).

    Returns
    -------
    samp_list : dataframe
        Contains information on corresponding sampleID and
        population classifications.
    dc : np.array
        Array of derived allele counts.
    unknowns : dataframe
        If kfcv is set to False, returns a dataframe with
        information about sampleID and indices for samples
        of unknown origin.
    """

    # Check formats of datatypes
    if os.path.exists(infile) is False:
        raise ValueError("Path to infile does not exist")

    # Load genotypes
    print("loading genotypes")
    if infile.endswith(".zarr"):

        callset = zarr.open_group(infile, mode="r")
        gt = callset["calldata/GT"]
        gen = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]

    elif infile.endswith(".vcf") or infile.endswith(".vcf.gz"):

        vcf = allel.read_vcf(infile, log=sys.stderr)
        gen = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]

    elif infile.endswith(".locator.hdf5"):

        h5 = h5py.File(infile, "r")
        dc = np.array(h5["derived_counts"])
        samples = np.array(h5["samples"])
        h5.close()

    else:
        raise ValueError("Infile must have extension 'zarr', 'vcf', or 'hdf5'")

    # count derived alleles for biallelic sites
    if infile.endswith(".locator.hdf5") is False:

        print("counting alleles")
        ac = gen.to_allele_counts()
        biallel = gen.count_alleles().is_biallelic()
        dc = np.array(ac[biallel, :, 1], dtype="int_")
        dc = np.transpose(dc)

        if (save_allele_counts and
                not infile.endswith(".locator.hdf5")):

            print("saving derived counts for reanalysis")
            outfile = h5py.File(infile + ".locator.hdf5", "w")
            outfile.create_dataset("derived_counts", data=dc)
            outfile.create_dataset("samples", data=samples,
                                   dtype=h5py.string_dtype())
            outfile.close()

    # Load data and organize for output
    print("loading sample data")

    if os.path.exists(sample_data) is False:
        raise ValueError("Path to sample_data does not exist")

    locs = pd.read_csv(sample_data, sep="\t")

    if not pd.Series(["x",
                      "pop",
                      "y",
                      "sampleID"]).isin(locs.columns).all():
        raise ValueError("sample_data does not have correct columns")

    locs["id"] = locs["sampleID"]
    locs.set_index("id", inplace=True)

    # sort loc table so samples are in same order as genotype samples
    locs = locs.reindex(np.array(samples))

    # Create order column for indexing
    locs["order"] = np.arange(0, len(locs))

    # If kfcv, cannot have any NAs
    if kfcv is True:
        uk_remove = locs[locs["x"].isnull()]["order"].values
        dc = np.delete(dc, uk_remove, axis=0)
        samples = np.delete(samples, uk_remove)
        locs = locs.dropna()

    # check that all sample names are present
    if not all(
        [locs["sampleID"][x] == samples[x] for x in range(len(samples))]
    ):
        raise ValueError(
            "sample ordering failed! Check that sample IDs match VCF.")

    if kfcv:

        locs = np.array(locs["pop"])
        samp_list = pd.DataFrame({"samples": samples, "pops": locs})

        # Return the sample list to be funneled into kfcv
        return samp_list, dc

    else:

        # Find unknown locations as NAs in the dataset
        unknowns = locs.iloc[np.where(pd.isnull(locs["pop"]))]

        # Extract known location information for training
        samples = samples[np.where(pd.notnull(locs["pop"]))]
        locs = locs.iloc[np.where(pd.notnull(locs["pop"]))]
        order = np.array(locs["order"])
        locs = np.array(locs["pop"])
        samp_list = pd.DataFrame({"samples": samples,
                                  "pops": locs,
                                  "order": order})

        return samp_list, dc, unknowns


class classifierHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        """
        Initializes object of class classifierHyperModel.

        Parameters
        ----------
        input_shape : int
            Number of training examples.
        num_classes : int
            Number of populations or labels.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        """
        Builds a model with the specified hyperparameters.

        Parameters
        ----------
        hp : keras.tuners class
            Class that defines how to sample hyperparameters (e.g.
            RandomSearch()).

        Returns
        -------
        model : Keras sequential model
            Model with all the layers and specified hyperparameters.
        """
        model = tf.Sequential()
        model.add(tf.layers.BatchNormalization(
            input_shape=(self.input_shape,)))
        model.add(
            tf.layers.Dense(
                units=hp.Int(
                    "units_1",
                    # placeholder values for now
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128,
                ),
                activation=hp.Choice(
                    "dense_activation_1",
                    values=["elu", "relu", "tanh", "sigmoid"],
                    default="elu",
                ),
            )
        )
        model.add(
            tf.layers.Dense(
                units=hp.Int(
                    "units_2",
                    # placeholder values for now
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128,
                ),
                activation=hp.Choice(
                    "dense_activation_2",
                    values=["elu", "relu", "tanh", "sigmoid"],
                    default="elu",
                ),
            )
        )
        model.add(
            tf.layers.Dense(
                units=hp.Int(
                    "units_3",
                    # placeholder values for now
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128,
                ),
                activation=hp.Choice(
                    "dense_activation_3",
                    values=["elu", "relu", "tanh", "sigmoid"],
                    default="elu",
                ),
            )
        )
        model.add(
            tf.layers.Dropout(
                rate=hp.Float(
                    "dropout", min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        model.add(
            tf.layers.Dense(
                units=hp.Int(
                    "units_4",
                    # placeholder values for now
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128,
                ),
                activation=hp.Choice(
                    "dense_activation_4",
                    values=["elu", "relu", "tanh", "sigmoid"],
                    default="elu",
                ),
            )
        )
        model.add(
            tf.layers.Dense(
                units=hp.Int(
                    "units_5",
                    # placeholder values for now
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128,
                ),
                activation=hp.Choice(
                    "dense_activation_5",
                    values=["elu", "relu", "tanh", "sigmoid"],
                    default="elu",
                ),
            )
        )
        model.add(
            tf.layers.Dense(
                units=hp.Int(
                    "units_6",
                    # placeholder values for now
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128,
                ),
                activation=hp.Choice(
                    "dense_activation_6",
                    values=["elu", "relu", "tanh", "sigmoid"],
                    default="elu",
                ),
            )
        )
        model.add(tf.layers.Dense(self.num_classes,
                                  activation="softmax"))

        model.compile(
            optimizer=tf.optimizers.Adam(
                T.variable(
                    hp.Float(
                        "learning_rate",
                        min_value=1e-4,
                        max_value=1e-2,
                        sampling="LOG",
                        default=5e-4,
                    )
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


def assign_plot(save_dir, ensemble=False, col_scheme="Spectral"):
    """
    Plots the frequency of assignment of individuals
    from unknown populations to different populations
    included in the training data.

    Parameters
    ----------
    save_dir : string
        Path to output file where "preds.csv" lives and
        also where the resulting plot will be saved.
    ensemble : boolean
        Set to True if multiple models used to generate assignments
        (Default=False).
    col_scheme : string
        Colour scheme of confusion matrix. See
        matplotlib.org/stable/tutorials/colors/colormaps.html
        for available colour palettes (Default="Spectral").

    Returns
    -------
    assign_plot.png : PNG file
        PNG formatted assignment plot located in the
        save_dir folder.
    """
    # Check inputs
    if isinstance(save_dir, str) is False:
        raise ValueError("save_dir should be string")
    if isinstance(ensemble, bool) is False:
        raise ValueError("ensemble should be boolean")
    if isinstance(col_scheme, str) is False:
        raise ValueError("col_scheme should be string")

    # Load data
    if ensemble:

        # Check if right directory exists
        if os.path.exists(save_dir + "/pop_assign_freqs.csv") is False:
            raise ValueError(
                "pop_assign_freqs.csv does not exist in save_dir")

        e_preds = pd.read_csv(save_dir + "/pop_assign_freqs.csv")
        e_preds.rename(columns={e_preds.columns[0]: "sampleID"}, inplace=True)
    else:

        # Check if right directory exists
        if os.path.exists(save_dir + "/pop_assign.csv") is False:
            raise ValueError("pop_assign.csv does not exist in save_dir")

        e_preds = pd.read_csv(save_dir + "/pop_assign.csv")

    e_preds.set_index("sampleID", inplace=True)

    # Set number of classes
    num_classes = len(e_preds.columns)

    # Create plot
    sn.set()
    sn.set_style("ticks")
    e_preds.plot(
        kind="bar",
        stacked=True,
        colormap=ListedColormap(sn.color_palette(col_scheme, num_classes)),
        figsize=(12, 6),
        grid=None,
    )
    legend = plt.legend(
        loc="center right",
        bbox_to_anchor=(1.2, 0.5),
        prop={"size": 15},
        title="Predicted Pop",
    )
    plt.setp(legend.get_title(), fontsize="x-large")
    plt.xlabel("Sample ID", fontsize=20)
    plt.ylabel("Frequency of Assignment", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save plot to output directory
    plt.savefig(save_dir + "/assign_plot.png", bbox_inches="tight")


def structure_plot(save_dir, ensemble=False, col_scheme="Spectral"):
    """
    Takes results from running the neural network with
    K-fold cross-validation and creates a structure plot
    showing proportion of assignment of individuals from
    known populations to predicted populations.

    Parameters
    ----------
    save_dir : string
        Path to output file where "ensemble_preds.csv" or "preds.csv"
        lives and also where the resulting plot will be saved.
    ensemble : boolean
        Whether ensemble of models was used to generate results or not
        (Default=False).
    col_scheme : string
        Colour scheme of confusion matrix. See
        matplotlib.org/stable/tutorials/colors/colormaps.html
        for available colour palettes (Default="Spectral").

    Returns
    -------
    structure_plot.png : PNG file
        PNG formatted structure plot located in the
        save_dir folder.
    """
    # Check inputs and load data
    if ensemble is True:
        if os.path.exists(save_dir + "/ensemble_preds.csv") is False:
            raise ValueError("Path to ensemble_preds does not exist")
        preds = pd.read_csv(save_dir + "/ensemble_preds.csv")
    else:
        # Load data
        if os.path.exists(save_dir + "/preds.csv") is False:
            raise ValueError("Path to preds does not exist")
        preds = pd.read_csv(save_dir + "/preds.csv")
    if isinstance(col_scheme, str) is False:
        raise ValueError("col_scheme should be a string")

    preds = preds.drop(preds.columns[0], axis=1)
    npreds = preds.groupby(["true_pops"]).agg("mean")
    npreds = npreds.sort_values("true_pops", ascending=True)
    npreds = npreds / np.sum(npreds, axis=1)

    # Make sure values are correct
    if not np.round(np.sum(npreds, axis=1), 2).eq(1).all():
        raise ValueError("Incorrect input values")

    # Find number of unique classes
    num_classes = len(npreds.index)

    if not len(npreds.index) == len(npreds.columns):
        raise ValueError(
            "Number of pops does not \
             match number of predicted pops"
        )

    # Create plot
    sn.set()
    sn.set_style("ticks")
    npreds.plot(
        kind="bar",
        stacked=True,
        colormap=ListedColormap(sn.color_palette(col_scheme, num_classes)),
        figsize=(12, 6),
        grid=None,
    )
    legend = plt.legend(
        loc="center right",
        bbox_to_anchor=(1.2, 0.5),
        prop={"size": 15},
        title="Predicted Pop",
    )
    plt.setp(legend.get_title(), fontsize="x-large")
    plt.xlabel("Actual Pop", fontsize=20)
    plt.ylabel("Frequency of Assignment", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save plot to output directory
    plt.savefig(save_dir + "/structure_plot.png", bbox_inches="tight")


def snp_rank(infile, sample_data, mod_path=None,
             save_dir="snp_rank_results"):
    """
    Finds the most important SNPs in determining a model's performance.

    Parameters
    ----------
    infile : string
        Path to VCF file containing genetic data.
    sample_data : string
        Path to tab-delimited file containing columns x, y, pop, and
        sampleID.
    mod_path : string
        Path to tuned model. If set to None, just uses default model
        (Default=None).
    save_dir : string
        Path to output directory (Default="snp_rank_results").

    Returns
    -------
    ranking : pd.DataFrame
        DataFrame containing relative importance of each SNP, where SNPs
        are labelled from 1 to number of SNPs in order of appearance in
        the VCF file.
    """
    # Check inputs
    if os.path.exists(infile) is False:
        raise ValueError("Path to infile does not exist")
    if os.path.exists(sample_data) is False:
        raise ValueError("Path to sample_data does not exist")
    if isinstance(mod_path, str) is False and mod_path is not None:
        raise ValueError("mod_path should be string or None")

    samp_list, dc, = read_data(
        infile,
        sample_data,
        save_allele_counts=False,
        kfcv=True,
    )

    X = dc
    Y = samp_list["pops"]
    enc = OneHotEncoder(handle_unknown="ignore")
    Y_enc = enc.fit_transform(Y.values.reshape(-1, 1)).toarray()

    snp_names = np.arange(1, X.shape[1] + 1)

    if mod_path is None:
        model = tf.Sequential()
        model.add(tf.layers.BatchNormalization(
            input_shape=(X.shape[1],)))
        model.add(tf.layers.Dense(128, activation="elu"))
        model.add(tf.layers.Dense(128, activation="elu"))
        model.add(tf.layers.Dense(128, activation="elu"))
        model.add(tf.layers.Dropout(0.25))
        model.add(tf.layers.Dense(128, activation="elu"))
        model.add(tf.layers.Dense(128, activation="elu"))
        model.add(tf.layers.Dense(128, activation="elu"))
        model.add(tf.layers.Dense(len(np.unique(samp_list['pops'])),
                                  activation="softmax"))
        aopt = tf.optimizers.Adam(lr=0.0005)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=aopt,
            metrics="accuracy"
        )
    else:
        model = tf.models.load_model(mod_path + "/best_mod")

    errors = []

    for i in range(X.shape[1]):
        og_X = np.array(X[:, i])
        np.random.shuffle(X[:, i])

        pred = model.predict(X)
        error = log_loss(Y_enc, pred)

        errors.append(error)
        X[:, i] = og_X

    max_error = np.max(errors)
    importance = [e / max_error for e in errors]

    data = {"snp": snp_names, "error": errors, "importance": importance}
    ranking = pd.DataFrame(data, columns=["snp", "error", "importance"])
    ranking.sort_values(by=["importance"], ascending=[0], inplace=True)
    ranking.reset_index(inplace=True, drop=True)
    ranking.to_csv(save_dir + "/perturbation_rank_results.csv")

    return ranking
