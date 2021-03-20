# Neural network for pop assignment

# Load packages
import tensorflow.keras as tf
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
import numpy as np
import pandas as pd
import allel
import zarr
import h5py
import subprocess
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
import sys
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sn


# Create function for labelling models in kfcv
def get_model_name(k):
    """
    Returns a string model name.

    Parameters
    ----------
    k : int
        Model number.
    """
    return "model_" + str(k)


def run_neural_net(
    infile_kfcv,
    infile_all,
    sample_data,
    save_allele_counts=False,
    save_weights=False,
    patience=10,
    batch_size=32,
    max_epochs=10,
    seed=None,
    train_prop=0.5,
    gpu_number="0",
    tune_model=False,
    n_splits=5,
    n_reps=5,
    save_best_mod=False,
    save_dir="out",
    plot_history=False,
):
    """
    Uses input arguments from the command line to tune, train,
    evaluate an ensemble of neural networks, then predicts the
    population of origin for samples of unknown origin.

    Parameters
    ----------
    infile_kfcv : string
        Path to VCF or hdf5 file with genetic information for
        only samples of known origin. No samples of unknown
        origin should be in this file.
    infile_all : string
        Path to VCF or hdf5 file with genetic information
        for all samples (including samples of unknown origin).
    sample_data : string
        Path to input file with all samples present (including
        samples of unknown origin), which is a tab-delimited
        text file with columns x, y, pop, and sampleID.
    save_allele counts : boolean
        Whether or not to store derived allele counts in hdf5
        file (Default=False).
    save_weights : boolean
        Save model weights so you don't have to retrain again
        later (Default=False).
    patience : int
        Hyperparameter for leniency on early-stopping
        (Default=10).
    batch_size : int
        Number of samples to use in training for each batch
        (Default=32).
    max_epochs : int
        Number of epochs to train over (Default=10.
    seed : int
        Random seed (Default=None).
    train_prop : float
        Proportion of samples used in training (Default=0.5).
    gpu_number : string
        Whether to use GPUs (Default='0').
    tune_model : boolean
        Whether to tune model or just use default
        (Default=False).
    n_splits : int
        Number of folds in k-fold cross-validation
        (Default=5).
    n_reps : int
        Number of times to repeat k-fold cross-validation,
        creating the number of models in the ensemble
        (Default=5).
    save_best_mod : boolean
        Whether to save the model with the highest accuracy
        (Default=False).
    save_dir : string
        Directory where results will be stored (Default='out').
    plot_history : boolean
        Whether or not to plot the training vs validation loss
        over time (Default=False).


    Returns
    -------
    A dataframe in csv format called 'metrics.csv' that gives
        details on accuracy and performance of the best model
        and overall accuracy and performance of the ensemble.
    A dataframe in csv format called 'pop_assign_freqs.csv'
        outlining the frequency of assignment of an individual
        to each population.
    A dataframe in csv format called 'pop_assign_ensemble.csv'
        that specifies that top population for each individual
    """

    print(f"Output will be saved to: {save_dir}")

    # Read data
    print("Reading data...")
    samp_list, dc = read_data(
        infile=infile_kfcv,
        sample_data=sample_data,
        save_allele_counts=save_allele_counts,
        kfcv=True,
    )

    # Read data with unknowns so errors caught before training/tuning
    samp_list2, dc2, unknowns = read_data(
        infile=infile_all,
        sample_data=sample_data,
        save_allele_counts=save_allele_counts,
        kfcv=False,
    )

    # Split data into training and hold-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        dc, samp_list, stratify=samp_list["pops"], train_size=train_prop
    )

    # Make sure all classes are represented in test set
    if len(samp_list["pops"].unique()) != len(y_test["pops"].unique()):
        sys.exit(
            "Not all classes represented in test data;\
             choose smaller train_prop value"
        )

    # Want stratified because we want to preserve percentages of each pop
    print("Splitting data into K-folds...")
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_reps)

    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []

    # Create output directory to write results to
    subprocess.check_output(["mkdir", "-p", save_dir])
    fold_var = 1
    mod_list = list()

    for t, v in rskf.split(X_train, y_train["pops"]):

        # Set model name
        mod_name = get_model_name(fold_var)

        # Subset train and validation data
        traingen = X_train[t, :] - 1
        valgen = X_train[v, :] - 1

        # One hot encoding
        enc = OneHotEncoder(handle_unknown="ignore")
        y_train_enc = enc.fit_transform(
            y_train["pops"].values.reshape(-1, 1)
        ).toarray()
        popnames = enc.categories_[0]

        trainpops = y_train_enc[t]
        valpops = y_train_enc[v]

        valsamples = y_train["samples"].iloc[v].to_numpy()

        # Use default model
        if not tune_model:
            model = tf.Sequential()
            model.add(
                tf.layers.BatchNormalization(
                    input_shape=(traingen.shape[1],)
                )
            )
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
                optimizer=aopt, metrics="accuracy"
            )

        # Or tune the model for best results
        else:
            hypermodel = classifierHyperModel(
                input_shape=traingen.shape[1], num_classes=len(popnames)
            )

            # If tuned model already exists, rewrite
            if os.path.exists(save_dir + "/" + mod_name):
                subprocess.check_output(
                    ["rm", "-rf", save_dir + "/" + mod_name]
                )

            tuner = RandomSearch(
                hypermodel,
                objective="loss",
                seed=seed,
                max_trials=10,
                executions_per_trial=10,
                directory=save_dir,
                project_name=mod_name,
            )

            tuner.search(
                traingen, trainpops, epochs=10,
                validation_split=(train_prop - 1)
            )

            model = tuner.get_best_models(num_models=1)[0]

        # Create callbacks
        checkpointer = tf.callbacks.ModelCheckpoint(
            filepath=save_dir + "/" + mod_name + ".h5",
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
            traingen,
            trainpops,
            batch_size=int(batch_size),
            epochs=int(max_epochs),
            callbacks=callback_list,
            validation_data=(valgen, valpops),
        )

        # Load best model
        model.load_weights(save_dir + "/" + mod_name + ".h5")

        if not save_weights:
            os.remove(save_dir + "/" + mod_name + ".h5")

        if fold_var == 1:
            preds = pd.DataFrame(model.predict(valgen))
            preds.columns = popnames
            preds["sampleID"] = valsamples
            preds["model"] = mod_name
        else:
            preds_new = pd.DataFrame(model.predict(valgen))
            preds_new.columns = popnames
            preds_new["sampleID"] = valsamples
            preds_new["model"] = mod_name
            preds = preds.append(preds_new)

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
            fig.savefig(save_dir + "/" + mod_name + "_history.pdf",
                        bbox_inches="tight")

        # Save results
        results = model.evaluate(valgen, valpops)
        results = dict(zip(model.metrics_names, results))

        VALIDATION_ACCURACY.append(results["accuracy"])
        VALIDATION_LOSS.append(results["loss"])

        mod_list.append(model)

        tf.backend.clear_session()

        fold_var += 1

    # Add true populations to predictions dataframe and output csv
    preds = preds.merge(samp_list, left_on="sampleID", right_on="samples")
    preds = preds.drop("samples", axis=1)
    preds.to_csv(save_dir + "/" + "preds.csv", index=False)

    # Extract the best model and calculate accuracy on test set
    # print(len(mod_list))
    best_mod = mod_list[np.argmax(VALIDATION_ACCURACY)]

    # One hot encode test set
    y_train_enc = enc.fit_transform(
        y_train["pops"].values.reshape(-1, 1)
    ).toarray()
    y_test_enc = enc.fit_transform(
        y_test["pops"].values.reshape(-1, 1)
    ).toarray()

    # Create lists to fill with test information
    TEST_ACCURACY = []
    TEST_95CI = []
    TEST_LOSS = []

    checkpointer = tf.callbacks.ModelCheckpoint(
        filepath=save_dir + "/checkpoint.h5",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        monitor="loss",
        save_freq="epoch",
    )
    earlystop = tf.callbacks.EarlyStopping(
        monitor="loss", min_delta=0, patience=patience
    )
    reducelr = tf.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.2,
        patience=int(patience / 3),
        verbose=1,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )
    callback_list = [checkpointer, earlystop, reducelr]

    # Train model on all the data
    for i in range(n_reps * n_splits):
        mod = mod_list[i]
        history = mod.fit(
            X_train - 1, y_train_enc, epochs=int(max_epochs),
            callbacks=callback_list
        )

        test_loss, test_acc = mod.evaluate(X_test - 1, y_test_enc)

        # Find confidence interval of best model
        test_err = 1 - test_acc
        test_95CI = 1.96 * np.sqrt(
            (test_err * (1 - test_err)) / len(y_test_enc)
        )

        # Fill test lists with information
        TEST_LOSS.append(test_loss)
        TEST_ACCURACY.append(test_acc)
        TEST_95CI.append(test_95CI)

        print(
            f"Accuracy of model {i} is {np.round(test_acc, 2)}\
            +/- {np.round(test_95CI,2)}"
        )

    # Print metrics to csv
    print("Creating outputs...")
    metrics = pd.DataFrame(
        {
            "metric": [
                "Total validation accuracy",
                "Validation accuracy SD",
                "Best model validation accuracy",
                "Total validation loss",
                "Best model validation loss",
                "Best model",
                "Test accuracy",
                "Test 95% CI",
                "Test loss",
            ],
            "value": [
                np.mean(VALIDATION_ACCURACY),
                np.std(VALIDATION_ACCURACY),
                np.max(VALIDATION_ACCURACY),
                np.mean(VALIDATION_LOSS),
                np.min(VALIDATION_LOSS),
                get_model_name(np.argmax(VALIDATION_ACCURACY) + 1),
                np.mean(TEST_ACCURACY),
                np.mean(TEST_95CI),
                np.mean(TEST_LOSS),
            ],
        }
    )
    metrics.to_csv(save_dir + "/" + "metrics.csv", index=False)

    # Return the best model for future predictions
    if save_best_mod:
        print(save_dir + "/" + get_model_name(
            np.argmax(VALIDATION_ACCURACY) + 1)
             )
        os.mkdir(save_dir + "/best_model")
        best_mod.save(save_dir)

    # MAKE PREDICTIONS ON UNKNOWN DATA
    # One hot encode label data
    popnames = enc.categories_[0]

    # Organize unknown data
    unknown_inds = pd.array(unknowns["order"])
    ukgen = dc2[unknown_inds, :] - 1
    uksamples = unknowns["sampleID"].to_numpy()

    # Predict on unknown samples with ensemble of models
    pred_dict = {"count": [], "df": []}
    for i in range(n_splits * n_reps):
        mod = mod_list[i]
        tmp_df = pd.DataFrame(mod.predict(ukgen) * TEST_ACCURACY[i])
        tmp_df.columns = popnames
        tmp_df["sampleID"] = uksamples
        tmp_df["iter"] = i
        pred_dict["count"].append(i)
        pred_dict["df"].append(tmp_df)

    # Find the frequency of assignment for different populations
    top_pops = {"df": [], "pops": []}

    for i in range(n_splits * n_reps):
        top_pops["df"].append(i)
        top_pops["pops"].append(
            pred_dict["df"][i].iloc[:, 0:len(popnames)].idxmax(axis=1)
        )

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
    freq_df.columns = ["Assigned Pop", "Frequency", "Sample ID"]

    # Save predictions
    freq_df.to_csv(save_dir + "/pop_assign_ensemble.csv", index=False)

    if not save_weights:
        os.remove(save_dir + "/checkpoint.h5")

    print("Process complete")


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

    # count derived alleles for biallelic sites
    if not infile.endswith(".locator.hdf5"):

        print("counting alleles")
        ac = gen.to_allele_counts()
        biallel = gen.count_alleles().is_biallelic()
        dc = np.array(ac[biallel, :, 1], dtype="int_")
        dc = np.transpose(dc)

        if save_allele_counts and not infile.endswith(".locator.hdf5"):

            print("saving derived counts for reanalysis")
            outfile = h5py.File(infile + ".locator.hdf5", "w")
            outfile.create_dataset("derived_counts", data=dc)
            outfile.create_dataset(
                "samples", data=samples, dtype=h5py.string_dtype()
            )  # note this requires h5py v 2.10.0
            outfile.close()
            # sys.exit()

    # Load data and organize for output
    print("loading sample data")
    locs = pd.read_csv(sample_data, sep="\t")
    locs["id"] = locs["sampleID"]
    locs.set_index("id", inplace=True)

    # sort loc table so samples are in same order as genotype samples
    locs = locs.reindex(np.array(samples))

    # check that all sample names are present
    if not all(
        [
            locs["sampleID"][x] == samples[x] for x in range(len(samples))
        ]
    ):

        print("sample ordering failed! Check that sample IDs match the VCF.")
        sys.exit()

    if kfcv:

        locs = np.array(locs["pop"])
        samp_list = pd.DataFrame({"samples": samples, "pops": locs})

        # Return the sample list to be funneled into kfcv
        return samp_list, dc

    else:

        locs["order"] = np.arange(len(locs))

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
        model.add(
            tf.layers.BatchNormalization(
                input_shape=(self.input_shape,)
            )
        )
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
                    "dropout",
                    min_value=0.0,
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
        model.add(tf.layers.Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=5e-4,
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


def assign_plot(save_dir, col_scheme="Spectral"):
    """
    Plots the frequency of assignment of individuals
    from unknown populations to different populations
    included in the training data.

    Parameters
    ----------
    save_dir : string
        Path to output file where "preds.csv" lives and
        also where the resulting plot will be saved.
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

    # Load data
    e_preds = pd.read_csv(save_dir + "/pop_assign_freqs.csv")
    e_preds.rename(columns={e_preds.columns[0]: "sampleID"}, inplace=True)
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


def conf_matrix(save_dir, col_scheme="Purples"):
    """
    Takes results from running the neural network with
    K-fold cross-validation and creates a confusion
    matrix.

    Parameters
    ----------
    save_dir : string
        Path to output file where "preds.csv" lives and
        also where the resulting plot will be saved.
    col_scheme : string
        Colour scheme of confusion matrix. See
        matplotlib.org/stable/tutorials/colors/colormaps.html
        for available colour palettes (Default="Purples").

    Returns
    -------
    conf_mat.png : PNG file
        PNG formatted confusion matrix plot located in the
        save_dir folder.
    """

    # Load data
    preds = pd.read_csv(save_dir + "/preds.csv")
    npreds = preds.groupby(["pops"]).agg("mean")
    npreds = npreds.sort_values("pops", ascending=False)

    # Make sure values are correct
    if not np.round(np.sum(npreds, axis=1), 2).eq(1).all():
        raise ValueError("Incorrect input values")

    # Create heatmap
    sn.set(font_scale=1)
    cm_plot = sn.heatmap(
        npreds,
        annot=True,
        annot_kws={"size": 14},
        cbar_kws={"label": "Freq"},
        vmin=0,
        vmax=1,
        cmap="Purples",
    )
    cm_plot.set(xlabel="Predicted", ylabel="Actual")

    # Save to output folder
    plt.savefig(save_dir + "/conf_mat.png", bbox_inches="tight")


def structure_plot(save_dir, col_scheme="Spectral"):
    """
    Takes results from running the neural network with
    K-fold cross-validation and creates a structure plot
    showing proportion of assignment of individuals from
    known populations to predicted populations.

    Parameters
    ----------
    save_dir : string
        Path to output file where "preds.csv" lives and
        also where the resulting plot will be saved.
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

    # Load data
    preds = pd.read_csv(save_dir + "/preds.csv")
    npreds = preds.groupby(["pops"]).agg("mean")
    npreds = npreds.sort_values("pops", ascending=False)

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
