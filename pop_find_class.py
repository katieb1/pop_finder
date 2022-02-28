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

# Define class that can be used to load data, train and hyper tune classifier

class pop_find_class:
	# Object initialization; reads in and splits data according to user's preferred accuracy method
	def __init__(self, infile, sample_data, ensemble = True, kfold=True, seed=None, train_prop=0.8, save_dir="out", n_splits=5, n_reps=5):
		"""
		Paramters
		---------
		infile : string
			Path to file containing genetic data.
		sample_data : string
			Path to tab-delimited file containing columns x, y,
			pop, and sampleID.
		kfold : boolean
			Accuracy method; either k-fold cross validation (True) or 
			hold out
		seed : int
			Random seed (Default=None).
		train_prop : float
			Proportion of data to train on. Remaining data will be kept
			as a test set and not used until final model is trained
			(Default=0.8).
		save_dir : string
			Directory to save output to (Default='out').
		n_splits : int
			Number of folds in k-fold cross-validation
			(Default=5).
		n_reps : int
			Number of times to repeat k-fold cross-validation,
			creating the number of models in the ensemble
		"""
		# Assign values passed in by user to object attributes
		# Need to add some more input checks here
		self.infile = infile
		self.sample_data = sample_data
		self.seed = seed
		self.train_prop = train_prop
		self.save_dir = save_dir
		self.kfold = kfold
		self.ensemble = ensemble

		# input checks
		if os.path.exists(self.infile) is False:
			raise ValueError("infile does not exist")
		if os.path.exists(self.sample_data) is False:
			raise ValueError("sample_data does not exist")
		if isinstance(self.ensemble, bool) is False:
			raise ValueError("ensemble should be a boolean")
		if isinstance(self.save_dir, str) is False:
			raise ValueError("save_dir should be a string")

		# Load data using read_data function, including unknowns
		# See read_data for further information
		self.samp_list, self.dc, self.uk_list, self.dc_uk, self.unknowns = read_data(
			infile=self.infile,
			sample_data=self.sample_data,
			save_allele_counts=False,
				)

		# Create save_dir if doesn't already exist
		print(f"Output will be saved to: {save_dir}")
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		os.makedirs(save_dir)

		# Split data for accuracy assessment
		if kfold:
			# kfold split here
			self.n_splits = n_splits
			self.n_reps = n_reps
			# Check nsplits is > 1
			if n_splits <= 1:
				raise ValueError("n_splits must be greater than 1")

			# Check there are more samples in the smallest pop than n_splits
			if n_splits > self.samp_list.groupby(["pops"]).agg(["count"]).min().values[0]:
				raise ValueError(
				"n_splits cannot be greater than number of samples in smallest pop"
				)

			# Create stratified k-fold
			self.rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_reps)

		else:
			# Train prop can't be greater than num samples
			if len(self.dc) * (1 - self.train_prop) < len(np.unique(self.samp_list["pops"])):
				raise ValueError("train_prop is too high; not enough samples for test")
			# Create test set that will be used to assess model performance later
			self.X_train_0, self.X_holdout, self.y_train_0, self.y_holdout = train_test_split(
				self.dc, self.samp_list, stratify=self.samp_list["pops"], train_size=self.train_prop
			)
			# Save train and test set to save_dir
			np.save(save_dir + "/X_train.npy", self.X_train_0)
			self.y_train_0.to_csv(save_dir + "/y_train.csv", index=False)
			np.save(save_dir + "/X_holdout.npy", self.X_holdout)
			self.y_holdout.to_csv(save_dir + "/y_holdout.csv", index=False)

	def hyper_tune(self,max_trials=10,runs_per_trial=10,max_epochs=100,train_prop=0.8,mod_name="hyper_tune"):
		
		# Do some checks on variables that are passed to hyper_tune
		if isinstance(max_trials, np.int) is False:
			raise ValueError("max_trials should be integer")
		if isinstance(runs_per_trial, np.int) is False:
			raise ValueError("runs_per_trial should be integer")
		if isinstance(max_epochs, np.int) is False:
			raise ValueError("max_epochs should be integer")
		if isinstance(train_prop, np.float) is False:
			raise ValueError("train_prop should be float")
		if isinstance(self.seed, np.int) is False and self.seed is not None:
			raise ValueError("seed should be integer or None")
		if isinstance(self.save_dir, str) is False:
			raise ValueError("save_dir should be string")
		if isinstance(mod_name, str) is False:
			raise ValueError("mod_name should be string")
		
		#K-fold vs. hold-out split
		if self.kfold:
			# use a single fold
			for t, v in self.rskf.split(self.dc, self.samp_list["pops"]):
				# Get train, test folds from rskf
				X_train = self.dc[t, :] - 1
				X_holdout = self.dc[v, :] - 1
				y_train = self.samp_list.iloc[t]
				y_holdout = self.samp_list.iloc[v]
				break
		else:
			# Split data into train/test
			X_train, X_val, y_train, y_val = train_test_split(
				self.X_train_0,
				self.y_train_0,
				stratify=self.y_train_0["pops"],
				train_size=self.train_prop,
				random_state=self.seed,
			)
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

		tuner = RandomSearch(
			hypermodel,
			objective="val_loss",
			seed=self.seed,
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
		self.hyp_mod = tuner.get_best_models(num_models=1)[0]
		tuner.get_best_models(num_models=1)[0].save(save_dir + "/hyper_tune_mod")

	def class_train(self, 
	plot_hist=True,
	nbags=10,
	save_weights=True, 
	patience=20,
	batch_size=32, 
	max_epochs=100, 
	):
		# Set up directory for training outputs
		save_dir = self.save_dir + "/training"
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		os.makedirs(save_dir)

		# Initialize model list
		self.model_list = []

		# Check to see if a model exists, create a basic one if not
		# Consider enforcing hypertuning?
		if hasattr(self, 'hyp_mod'):
			model = self.hyp_mod
		else:
			if self.kfold:
				for t, v in self.rskf.split(self.dc, self.samp_list["pops"]):
					# Subset train and validation data
					X_train = self.dc[t, :] - 1
					X_val = self.dc[v, :] - 1
					y_train = self.samp_list.iloc[t]
					y_val = self.samp_list.iloc[v]
					break
				X_train, X_val, y_train, y_val = train_test_split(
					X_train,
					y_train,
					stratify=y_train["pops"],
					train_size=self.train_prop,
					random_state=self.seed,
				)
				# One hot encoding
				enc = OneHotEncoder(handle_unknown="ignore")
				y_train_enc = enc.fit_transform(
					y_train["pops"].values.reshape(-1, 1)).toarray()
				y_val_enc = enc.fit_transform(
					y_val["pops"].values.reshape(-1, 1)).toarray()
				popnames = enc.categories_[0]
				model = basic_model(X_train,popnames)
				self.hyp_model = model
			else:
				X_train, X_val, y_train, y_val = train_test_split(
					self.X_train_0,
					self.y_train_0,
					stratify=self.y_train_0["pops"],
					train_size=self.train_prop,
					random_state=self.seed,
				)
				# One hot encoding
				enc = OneHotEncoder(handle_unknown="ignore")
				y_train_enc = enc.fit_transform(
					y_train["pops"].values.reshape(-1, 1)).toarray()
				y_val_enc = enc.fit_transform(
					y_val["pops"].values.reshape(-1, 1)).toarray()
				popnames = enc.categories_[0]
				model = basic_model(X_train,popnames)
				self.hyp_mod = model

		print(f"Output will be saved to: {save_dir}")

		# K-fold approach
		if self.kfold:
			pred_labels = []
			true_labels = []
			pred_labels_ensemble = []
			true_labels_ensemble = []
			if self.ensemble:
				ensemble_preds = pd.DataFrame()
				# Loop through splits
				fold = 0
				for t, v in self.rskf.split(self.dc, self.samp_list["pops"]):
					fold_dir = save_dir + "/fold_" + str(fold)
					os.makedirs(fold_dir)

					# Get train, test folds from rskf
					X_train = self.dc[t, :] - 1
					X_holdout = self.dc[v, :] - 1
					y_train = self.samp_list.iloc[t]
					y_holdout = self.samp_list.iloc[v]

					# Run ensemble
					fold_mods, test_dict, tot_bag_df = run_ensemble(X_train=X_train, 
						X_fold=X_holdout, 
						y_train=y_train, 
						y_fold=y_holdout, 
						nbags=nbags, 
						model=model,
						train_prop=self.train_prop,
						save_dir=fold_dir,
								patience=patience)

					self.model_list.append(fold_mods)
					ensemble_preds = ensemble_preds.append(tot_bag_df)
					fold = fold + 1

					tmp_pred_label = []
					tmp_true_label = []
					for i in range(0, len(test_dict["df"])):
						tmp_pred_label.append(
							test_dict["df"][i].iloc[
							:, 0:len(popnames)
							].idxmax(axis=1).values
						)
						tmp_true_label.append(test_dict["df"][i]["true_pops"].values)

					pred_labels_ensemble.append(
						tot_bag_df.iloc[:, 0:len(popnames)].idxmax(axis=1).values
					)
					
					true_labels_ensemble.append(tmp_true_label[0])

					pred_labels.append(np.concatenate(tmp_pred_label, axis=0))
					true_labels.append(np.concatenate(tmp_true_label, axis=0))
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
					save_dir + "/ensemble_classification_report.csv"
				)
				ensemble_preds.to_csv(save_dir + "/ensemble_preds.csv")
			else:
				# Run single neural network with k-fold approach
				fold = 0
				preds = pd.DataFrame()
				for t, v in self.rskf.split(self.dc, self.samp_list["pops"]):
					fold_dir = save_dir + "/fold_" + str(fold)
					os.makedirs(fold_dir)

					# get train, test folds, then get val set from train
					X_train = self.dc[t, :] - 1
					X_holdout = self.dc[v, :] - 1
					y_train = self.samp_list.iloc[t]
					y_holdout = self.samp_list.iloc[v]
					
					# One hot encode test values
					enc = OneHotEncoder(handle_unknown="ignore")
					y_train_enc = enc.fit_transform(
						y_train["pops"].values.reshape(-1, 1)).toarray()
					self.popnames = enc.categories_[0]
					
					X_train, X_val, y_train, y_val = train_test_split(
						X_train, y_train, stratify=y_train["pops"],
						random_state=self.seed
					)
					fold_mods, test_dict = reg_train(X_train = X_train, 
						X_val = X_val, 
						X_holdout = X_holdout, 
						y_train = y_train, 
						y_val = y_val, 
						y_holdout = y_holdout, 
						save_dir = fold_dir,
						model=model)

					self.model_list.append(fold_mods)
					fold = fold + 1
					preds = preds.append(test_dict["df"][0])

					# Assemble labels
					tmp_pred_label = []
					tmp_true_label = []
					for i in range(0, len(test_dict["df"])):
						tmp_pred_label.append(
							test_dict["df"][i].iloc[
							:, 0:len(popnames)
							].idxmax(axis=1).values
						)
						tmp_true_label.append(test_dict["df"][i]["true_pops"].values)

					pred_labels.append(np.concatenate(tmp_pred_label, axis=0))
					true_labels.append(np.concatenate(tmp_true_label, axis=0))
				preds.to_csv(save_dir + "/preds.csv")
				# return pred_labels, true_labels
				pred_labels = np.concatenate(pred_labels)
				true_labels = np.concatenate(true_labels)
				report = classification_report(
					true_labels, pred_labels, zero_division=1, output_dict=True
				)
				report = pd.DataFrame(report).transpose()
				report.to_csv(save_dir + "/classification_report.csv")
		else:
			# Run ensemble with hold-out
			if self.ensemble:
				os.makedirs(save_dir + "/ensemble_weights")
				temp_dir = save_dir + "/ensemble_weights"
				fold_mods, test_dict, tot_bag_df  = run_ensemble(X_train=self.X_train_0, 
						X_fold=self.X_holdout, 
						y_train=self.y_train_0, 
						y_fold=self.y_holdout, 
						nbags=nbags, 
						model=model,
						train_prop=self.train_prop,
						save_dir=temp_dir,
								patience=patience)
				self.model_list = fold_mods
			else:
				# Run regular neural training, with hold out
				X_train, X_val, y_train, y_val = train_test_split(
						self.X_train_0, self.y_train_0, stratify=self.y_train_0["pops"],
						random_state=self.seed)
				# Generate popnames
				enc = OneHotEncoder(handle_unknown="ignore")
				y_train_enc = enc.fit_transform(
					y_train["pops"].values.reshape(-1, 1)).toarray()
				y_val_enc = enc.fit_transform(
					y_val["pops"].values.reshape(-1, 1)).toarray()
				y_test_enc = enc.fit_transform(
					self.y_holdout["pops"].values.reshape(-1, 1)).toarray()
				self.popnames = enc.categories_[0]

				fold_mods, test_dict = reg_train(X_train = X_train, 
						X_val = X_val, 
						X_holdout = self.X_holdout, 
						y_train = y_train, 
						y_val = y_val, 
						y_holdout = self.y_holdout, 
						model=model,
						save_dir = save_dir)
				self.model_list = fold_mods
		self.hyp_mod = model

	def predict(self):
		save_dir = self.save_dir + "/predict_output"
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		os.makedirs(save_dir)
		uksamples = self.unknowns["sampleID"].to_numpy()
		ukgen = self.dc_uk
		popnames = self.popnames
		pred_dict = {"count": [], "df": []}
		top_pops = {"df": [], "pops": []}
		ypreds = []
		model = self.hyp_mod
		if len(self.model_list) > 1:
			i=0
			pred_dict = {"count": [], "df": []}
			top_pops = {"df": [], "pops": []}
			for checkpoint in self.model_list:
				model.load_weights(checkpoint[0])
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
				i= i+1
			ypreds = np.array(ypreds)
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
			freq_df.to_csv(self.save_dir + "/pop_assign_ensemble.csv",
							index=False)
		else:
			model.load_weights(self.model_list[0])
			tmp_df = pd.DataFrame(model.predict(ukgen))
			tmp_df.columns = popnames
			tmp_df["sampleID"] = uksamples
			tmp_df.to_csv(self.save_dir + "/pop_assign.csv", index=False)
	def kfold_valid(self, 
		n_splits=5,
		n_reps=5,
	):
		# Check data types
		if isinstance(n_splits, np.int) is False:
			raise ValueError("n_splits should be an integer")
		if isinstance(n_reps, np.int) is False:
			raise ValueError("n_reps should be an integer")
		if isinstance(self.ensemble, bool) is False:
			raise ValueError("ensemble should be a boolean")
		if isinstance(save_dir, str) is False:
			raise ValueError("save_dir should be a string")
		# Check nsplits is > 1
		if n_splits <= 1:
			raise ValueError("n_splits must be greater than 1")
		samp_list = self.y_train_0

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
def basic_model(bag_X, popnames):
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
		metrics="accuracy")
	return model

def plot_history(history, i=None, ensemble=False, save_dir="out"):
	# plot training history
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
		if ensemble:
			fig.savefig(
				save_dir + "/model" + str(i) + "_history.pdf",
				bbox_inches="tight"
			)
		else:
			fig.savefig(
				save_dir + "/history.pdf",
				bbox_inches="tight"
			)
		plt.close()

def run_ensemble(X_train, X_fold, y_train, y_fold, nbags, model, train_prop, save_dir, plot_hist=True, patience=20, batch_size=32, max_epochs=100):
	n_prime = np.int(np.ceil(len(X_train) * train_prop))
	
	# create list of models trained
	ensembl_fl = []
	
	# Add info about test samples
	y_test_samples = y_fold["samples"].to_numpy()
	y_test_pops = y_fold["pops"].to_numpy()

	# One hot encode test values
	enc = OneHotEncoder(handle_unknown="ignore")
	y_test_enc = enc.fit_transform(
		y_fold["pops"].values.reshape(-1, 1)).toarray()
	popnames = enc.categories_[0]

	TEST_LOSS = []
	TEST_ACCURACY = []
	TEST_95CI = []
	test_dict = {"count": [], "df": []}

	# Generate random splits in train set
	for i in range(nbags):
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

		# Hot encode train and valuation sets
		enc = OneHotEncoder(handle_unknown="ignore")
		bag_y_enc = enc.fit_transform(
			bag_y["pops"].values.reshape(-1, 1)).toarray()
		y_val_enc = enc.fit_transform(
			y_val["pops"].values.reshape(-1, 1)).toarray()

		# Create callbacks
		temp_str = "/checkpoint_" + str(i)+ ".h5"
		ensembl_fl.append(save_dir + temp_str)
		checkpointer = tf.callbacks.ModelCheckpoint(
			filepath=save_dir + temp_str,
			verbose=1,
			save_best_only=True,
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
		model.load_weights(save_dir + temp_str)
		if plot_hist:
			plot_history(history=history, 
				i=i, 
				save_dir=save_dir, 
				ensemble=True
			)

	# Generate scores
		test_loss, test_acc = model.evaluate(X_fold - 1, y_test_enc)
		test_df = pd.DataFrame(model.predict(X_fold - 1))
		test_df.columns = popnames
		test_df["sampleID"] = y_test_samples
		test_df["true_pops"] = y_test_pops
		test_dict["count"].append(1)
		test_dict["df"].append(test_df)
		test_df.to_csv(save_dir + "/test_results.csv")

		# Fill test lists with information
		TEST_LOSS.append(test_loss)
		TEST_ACCURACY.append(test_acc)

	# yhats = np.array(yhats)
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
		# Metrics
	AVG_TEST_LOSS = np.round(np.mean(TEST_LOSS), 2)
	AVG_TEST_ACCURACY = np.round(np.mean(TEST_ACCURACY), 2)
	test_err = 1 - AVG_TEST_ACCURACY
	TEST_95CI = 1.96 * np.sqrt(
		(test_err * (1 - test_err)) / len(y_test_enc))
	print("Creating outputs...")
	best_score = "N/A"
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
	print("Ensemble training complete")
	return ensembl_fl, test_dict, tot_bag_df

def reg_train(X_train, X_val, X_holdout, y_train, y_val, y_holdout, model, save_dir, plot_hist=True, batch_size=32, max_epochs=100, patience=20):
	ensembl_fl = []
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
	y_test_enc = enc.fit_transform(
		y_holdout["pops"].values.reshape(-1, 1)).toarray()
	popnames = enc.categories_[0]

	# Add info about test samples
	y_test_samples = y_holdout["samples"].to_numpy()
	y_test_pops = y_holdout["pops"].to_numpy()

	TEST_LOSS = []
	TEST_ACCURACY = []
	TEST_95CI = []
	test_dict = {"count": [], "df": []}

	# Create callbacks
	if os.path.exists(save_dir + "/default_mod_weights"):
		shutil.rmtree(save_dir + "/default_mod_weights")
	os.makedirs(save_dir + "/default_mod_weights")
	
	ensembl_fl.append(save_dir + "/default_mod_weights/checkpoint.h5")
	
	checkpointer = tf.callbacks.ModelCheckpoint(
	filepath=save_dir + "/default_mod_weights/checkpoint.h5",
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
	model.load_weights(save_dir + "/default_mod_weights/checkpoint.h5")

	if plot_hist:
		plot_history(history=history, save_dir=save_dir, ensemble=False)

	tf.backend.clear_session()

	test_loss, test_acc = model.evaluate(X_holdout - 1, y_test_enc)
	test_df = pd.DataFrame(model.predict(X_holdout - 1))
	test_df.columns = popnames
	test_df["sampleID"] = y_test_samples
	test_df["true_pops"] = y_test_pops
	test_dict["count"].append(1)
	test_dict["df"].append(test_df)
	test_df.to_csv(save_dir + "/test_results.csv")
	
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
	print("Non-ensemble training complete")
	return ensembl_fl, test_dict
