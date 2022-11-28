# Load packages (original pop_finder)
import numpy as np
import pandas as pd
import allel
import zarr
import h5py
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import itertools
import shutil
import sys
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sn

# sklearn packages
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint


### read data function from pop_finder ###

def read_data(infile, sample_data, save_allele_counts=False):
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
		decoder = np.vectorize(lambda x: x.decode('UTF-8'))
		samples = decoder(samples)
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
	unknowns = locs.iloc[np.where(pd.isnull(locs["pop"]))]
	# handle presence of samples with unknown locations
	uk_remove = locs[locs["x"].isnull()]["order"].values
	dc_uk = dc[uk_remove,:] - 1
	dc = np.delete(dc, uk_remove, axis=0)
	samples_uk = samples[uk_remove]
	samples = np.delete(samples, uk_remove)
	locs_uk = locs[locs["pop"].isna()]
	locs = locs.dropna()
	# check that all sample names are present
	if not all(
		[locs["sampleID"][x] == samples[x] for x in range(len(samples))]
	):
		raise ValueError(
			"sample ordering failed! Check that sample IDs match VCF.")
	locs = np.array(locs["pop"])
	samp_list = pd.DataFrame({"samples": samples, "pops": locs})
	locs_uk = np.array(locs_uk["pop"])
	uk_list = pd.DataFrame({"samples":samples_uk, "pops": locs_uk})
	# Return the sample lists
	return samp_list, dc, uk_list, dc_uk, unknowns


###  Load in example data set ###

infile = "onlyAtl_500.recode.vcf.locator.hdf5"	
sample_data = "onlyAtl_truelocs_NAs.txt"
	
samp_list, dc, uk_list, dc_uk, unknowns = read_data(
			infile=infile,
			sample_data=sample_data,
			save_allele_counts=False,
				)			
				
### set reps and splits, and arrange cross-validation ###

n_splits = 3
n_reps = 5	

X_train, X_test, y_train, y_test = train_test_split(
	dc, samp_list["pops"], test_size=0.25, random_state=0)

# Create stratified k-fold with n_splits and n_reps, using the user specified random seed
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_reps, random_state=1111)

############################
## grid search cv approach##
############################
GRID = [
    {#'scaler': [StandardScaler()],
     'estimator': [MLPClassifier(random_state=1)],
     'estimator__solver': ['sgd','adam','lbfgs'],
	 'estimator__max_iter': [1000],
     'estimator__learning_rate_init': [0.001,0.0005],
     'estimator__hidden_layer_sizes': [(300, 200, 100), (400, 400, 400), (300, 300, 300), (200, 200, 200), (500,500,500)],
     'estimator__activation': ['logistic', 'tanh', 'relu'],
     'estimator__alpha': [0.0001, 0.001, 0.005],
     }
]

PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])

grid_search = GridSearchCV(estimator=PIPELINE, param_grid=GRID, 
                            scoring=make_scorer(accuracy_score),# average='macro'), 
                            n_jobs=-1, cv=rskf, refit=True, verbose=1, 
                            return_train_score=True)

grid_search.fit(X_train, y_train)
grid_search.score(X_test,y_test)
##############################

########################################
## cross validation without gridsearch##
########################################

clf = MLPClassifier(solver='sgd', alpha=1e-5,
	hidden_layer_sizes=(500,300,200, 100), random_state=1)

scores = cross_val_score(clf, X_train, y_train, cv=rskf)	

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

########################################

####################
## randomsearch cv##
####################

# gave best results so far?

model = MLPClassifier()

parameters = {"solver": ['sgd','adam','lbfgs'],
				"learning_rate_init": uniform(),
                  "hidden_layer_sizes" : randint(20, 500),
                  "activation": ['logistic', 'tanh', 'relu'],
				  "alpha": uniform(),
                 }
randm = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                               cv = rskf, n_iter = 100, n_jobs=-1)

randm.fit(X_train, y_train)

############################
#### randomsearch hyperopt##
#### NOT OPERATIONAL #######
############################

# hyperopt can be used to allow for control over the size of multiple hidden neuron layers
# (currently can't do this with randomsearchcv I don't think)
# However, I didn't get around to fully implementing it
# If interested, I was following this example: https://klane.github.io/databall/model/parameters/

from hyperopt import hp
from model_selection import calculate_metrics, optimize_param
space_mlp = {}
space_mlp['hidden_layer_sizes'] = (100 + hp.randint('hidden_layer_sizes', 400),100 + hp.randint('hidden_layer_sizes', 400),100 + hp.randint('hidden_layer_sizes', 400))
space_mlp['alpha'] = hp.loguniform('alpha', -8*np.log(10), 3*np.log(10))
space_mlp['activation'] = hp.choice('activation', ['relu', 'logistic', 'tanh'])
space_mlp['solver'] = hp.choice('solver', ['lbfgs', 'sgd', 'adam'])
model = MLPClassifier()

best_mlp, param_mlp = optimize_params(model, x_train, y_train, stats, space_mlp, max_evals=100)
#############################