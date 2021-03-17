import locator_mod
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

def contour_classifier(sample_data, run_locator=False, gen_dat=None, 
                       nboots=50, return_plots=True, save_dir='out'):
    """
    Wrapper function that runs locator to generate a density of predictions,
    then uses contour lines to choose the most likely population.
    
    Parameters
    ----------
    sample_data : string
        Filepath to input file containing coordinates and populations of samples,
        including individuals from unknown populations.
    run_locator : boolean
        Run locator and use outputs to generate classifications. If set to False, then
        will look in specified save_dir for the *_predlocs.txt files from a previous
        locator run. If set to True, ensure that gen_dat is not None (Default=False).
    gen_dat : string
        Filepath to input genetic data in VCF format (Default=None).
    nboots : int
        Number of bootstrap iterations (Default=50).
    return_plots : boolean
        Return contour plots of prediction densities overlayed with true population
        locations (Default=True).
    save_dir : string
        Folder to save results. Folder should already be in directory (Default='out').
    """
    
    
    # Run locator
    if run_locator==True:
        locator_mod.locator(sample_data,
                            gen_dat,
                            bootstrap=True,
                            nboots=nboots)
    
    # Create list of predloc files for concatenation
    out_list = [save_dir+"/loc_boot"+str(i)+"_predlocs.txt" for i in range(nboots)]
    
    # Append all predlocs files to first file
    with open(out_list[0], 'a') as outfile:
        for names in out_list[1:-1]:
            with open(names) as infile:
                string = ""
                outfile.write(string.join(infile.readlines()[1:]))
                
    # Convert input data file into correct file for contour wrapper
    true_dat = pd.read_csv(sample_data, sep="\t")
    true_dat = true_dat[['x', 'y', 'pop']].dropna().drop_duplicates()
    
    # Wrangle prediction data, read first predloc
    pred_dat = pd.read_csv(out_list[0])
    pred_dat = pred_dat.rename({'x': 'pred_x',
                                'y': 'pred_y'}, axis=1)
    
    # Create empty dataframe to fill with results
    class_dat = {'sampleID':[],
                 'classification':[],
                 'kd_estimate':[]}
    
    # For each unknown sample, get contours
    for sample in pred_dat['sampleID'].drop_duplicates():
        print(sample)
        tmp_dat = pred_dat[pred_dat['sampleID'] == sample]
        class_dat['sampleID'].append(sample)

        d_x = (max(test_dat['pred_x']) - min(test_dat['pred_x']))/5
        d_y = (max(test_dat['pred_y']) - min(test_dat['pred_y']))/5
        xlim = min(tmp_dat['pred_x']) - d_x, max(tmp_dat['pred_x']) + d_x
        ylim = min(tmp_dat['pred_y']) - d_y, max(tmp_dat['pred_y']) + d_y

        X, Y = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]

        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([tmp_dat['pred_x'], tmp_dat['pred_y']])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        new_z = Z / np.max(Z)

        # Plot
        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        cset = ax.contour(X, Y, new_z,
                          levels=10, colors='black')
        cset.levels = -np.sort(-cset.levels)
        
        for pop in true_dat['pop'].values:
            x = true_dat[true_dat['pop'] == pop]['x'].values[0]
            y = true_dat[true_dat['pop'] == pop]['y'].values[0]
            plt.scatter(x, y, cmap='inferno', label=pop)
            
        ax.clabel(cset, cset.levels, inline=1, fontsize=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.title(sample)
        plt.legend()

        # Find predicted pop
        pred_pop, kd = cont_finder(true_dat, cset)
        class_dat['classification'].append(pred_pop)
        class_dat['kd_estimate'].append(kd)
        
        if return_plots is True:
            plt.savefig(save_dir+'/contour_'+sample+'.png', format='png')
            
    class_df = pd.DataFrame(class_dat)
    class_df.to_csv(save_dir+'/results.csv')
    
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
    
    
    cont_dict = {'pop': [], 'cont': []}
    
    for pop in true_dat['pop'].values:   
        cont_dict['pop'].append(pop)
        cont = 0
        point = np.array([[true_dat[true_dat['pop'] == pop]['x'].values[0],
                           true_dat[true_dat['pop'] == pop]['y'].values[0]]])
        
        for i in range(1,len(cset.allsegs)): 
            for j in range(len(cset.allsegs[i])):    
                path = matplotlib.path.Path(cset.allsegs[i][j].tolist())
                inside = path.contains_points(point) 
                if inside[0] == True:
                    cont = i
                    break
                else:
                    next                 
        cont_dict['cont'].append(np.round(cset.levels[cont], 2))
        
    pred_pop = cont_dict['pop'][np.argmin(cont_dict['cont'])]
    
    return pred_pop, min(cont_dict['cont'])