import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
