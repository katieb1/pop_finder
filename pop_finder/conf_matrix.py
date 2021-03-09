import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

def conf_matrix(save_dir,
                col_scheme='Purples'):
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
    preds = pd.read_csv(save_dir+"/preds.csv")
    npreds = preds.groupby(['pops']).agg('mean')
    npreds = npreds.sort_values('pops', ascending=False)
    
    # Make sure values are correct
    if not np.round(np.sum(npreds, axis=1), 2).eq(1).all():
        raise ValueError("Incorrect input values")
        
    # Create heatmap
    sn.set(font_scale=1)
    cm_plot = sn.heatmap(npreds,
                         annot=True,
                         annot_kws={'size': 14},
                         cbar_kws={'label': 'Freq'},
                         vmin=0,
                         vmax=1,
                         cmap="Purples")
    cm_plot.set(xlabel="Predicted", ylabel="Actual")
    
    # Save to output folder
    plt.savefig(save_dir+"/conf_mat.png",
                bbox_inches = "tight")