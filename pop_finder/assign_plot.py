import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
