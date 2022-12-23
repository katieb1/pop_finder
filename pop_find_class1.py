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

class pop_find_class:
	# instance attribute
	def __init__(self, infile, sample_data, seed=None, train_prop=0.8, save_dir="out"):
		self.infile = infile
		self.sample_data = sample_data
		self.seed=seed
		self.train_prop=train_prop
		self.save_dir=save_dir
		if os.path.exists(self.infile) is False:
			raise ValueError("infile does not exist")
		if os.path.exists(self.sample_data) is False:
			raise ValueError("sample_data does not exist")
		self.samp_list, self.dc, self.uk_list, self.dc_uk, self.unknowns = read_data(
			infile=self.infile,
			sample_data=self.sample_data,
			save_allele_counts=False,
				)
		# Create test set that will be used to assess model performance later
		self.X_train_0, self.X_holdout, self.y_train_0, self.y_holdout = train_test_split(
			self.dc, self.samp_list, stratify=self.samp_list["pops"], train_size=self.train_prop
			)
		# Create save_dir if doesn't already exist
		print(f"Output will be saved to: {save_dir}")
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		os.makedirs(save_dir)
		# Save train and test set to save_dir
		np.save(save_dir + "/X_train.npy", self.X_train_0)
		self.y_train_0.to_csv(save_dir + "/y_train.csv", index=False)
		np.save(save_dir + "/X_holdout.npy", self.X_holdout)
		self.y_holdout.to_csv(save_dir + "/y_holdout.csv", index=False)
	def hyper_tune(self, y_train_0=None, dc=None,max_trials=10,runs_per_trial=10,max_epochs=100,train_prop=0.8,seed=None,save_dir="out",mod_name="hyper_tune"):
		y_train_0 = self.y_train_0
		dc = self.X_train_0
		seed=self.seed
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
		# Train prop can't be greater than num samples
		if len(dc) * (1 - train_prop) < len(np.unique(y_train_0["pops"])):
			raise ValueError("train_prop is too high; not enough samples for test")
		# Split data into training test
		X_train, X_val, y_train, y_val = train_test_split(
			dc,
			y_train_0,
			stratify=y_train_0["pops"],
			train_size=train_prop,
			random_state=seed,
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
		self.best_mod = tuner.get_best_models(num_models=1)[0]
		tuner.get_best_models(num_models=1)[0].save(save_dir + "/best_mod")
	def class_train(self, 
	ensemble=False,
	plot_hist=True,
	nbags=10,
	save_weights=True, 
	patience=20,
	batch_size=32, 
	max_epochs=100, 
	):
		print(f"Output will be saved to: {self.save_dir}")
		y_train = self.y_train_0
		dc = self.X_train_0
		train_prop = self.train_prop
		if len(dc) * (1 - train_prop) < 1:
			raise ValueError(
				"train_prop is too high; not enough values for test")
		
		seed=self.seed
		save_dir = self.save_dir + "/training_output"
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		os.makedirs(save_dir)
		y_test_samples = self.y_holdout["samples"].to_numpy()
		y_test_pops = self.y_holdout["pops"].to_numpy()
		# One hot encode test values
		enc = OneHotEncoder(handle_unknown="ignore")
		y_test_enc = enc.fit_transform(
			self.y_holdout["pops"].values.reshape(-1, 1)).toarray()
		popnames = enc.categories_[0]
		self.popnames=popnames
		# results storage
		TEST_LOSS = []
		TEST_ACCURACY = []
		TEST_95CI = []
		yhats = []
		ypreds = []
		test_dict = {"count": [], "df": []}
		if hasattr(self, 'best_mod'):
			model = self.best_mod
		else:
			# Test if train_prop is too high
			if len(dc) * (1 - train_prop) < 1:
				raise ValueError(
				"train_prop is too high; not enough values for test")
			X_train, X_val, y_train, y_val = train_test_split(
				dc,
				y_train,
				stratify=y_train["pops"],
				train_size=train_prop,
				random_state=seed,
			)
			# Make sure all classes represented in y_val
			if len(np.unique(y_train["pops"]) ) != len(np.unique(y_val["pops"])):
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
			popnames1 = enc.categories_[0]
			model = basic_model(X_train,popnames1)
			self.best_mod = model
		if ensemble:
			X_train = self.X_train_0
			y_train = self.y_train_0
			n_prime = np.int(np.ceil(len(X_train) * 0.8))
			self.ensembl_fl = []
			if os.path.exists(save_dir + "/ensemble_weights"):
					shutil.rmtree(save_dir + "/ensemble_weights")
			os.makedirs(save_dir + "/ensemble_weights")
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
				enc = OneHotEncoder(handle_unknown="ignore")
				bag_y_enc = enc.fit_transform(
					bag_y["pops"].values.reshape(-1, 1)).toarray()
				y_val_enc = enc.fit_transform(
					y_val["pops"].values.reshape(-1, 1)).toarray()
				# Create callbacks
				temp_str = "/ensemble_weights/checkpoint_" + str(i)+ ".h5"
				self.ensembl_fl.append(temp_str)
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
					plot_history(history=history, i=i, save_dir= save_dir, ensemble=True)
				test_loss, test_acc = model.evaluate(self.X_holdout - 1, y_test_enc)
				test_df = pd.DataFrame(model.predict(self.X_holdout - 1))
				test_df.columns = popnames
				test_df["sampleID"] = y_test_samples
				test_df["true_pops"] = y_test_pops
				test_dict["count"].append(1)
				test_dict["df"].append(test_df)
				test_df.to_csv(save_dir+"/test_results.csv")
				# Fill test lists with information
				TEST_LOSS.append(test_loss)
				TEST_ACCURACY.append(test_acc)
			yhats = np.array(yhats)
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
		else:
			# Split training data into training and validation
			X_train = self.X_train_0
			y_train = self.y_train_0
			X_train, X_val, y_train, y_val = train_test_split(
				dc, y_train, stratify=y_train["pops"],
				random_state=seed)
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
			# Create callbacks
			if os.path.exists(save_dir + "/default_mod_weights"):
				shutil.rmtree(save_dir + "/default_mod_weights")
			os.makedirs(save_dir + "/default_mod_weights")
			
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

			test_loss, test_acc = model.evaluate(self.X_holdout - 1, y_test_enc)
			test_df = pd.DataFrame(model.predict(self.X_holdout - 1))
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
		print("Process complete")
	def predict(self, ensemble=False):
		save_dir = self.save_dir + "/training_output"
		uksamples = self.unknowns["sampleID"].to_numpy()
		ukgen = self.dc_uk
		popnames = self.popnames
		pred_dict = {"count": [], "df": []}
		top_pops = {"df": [], "pops": []}
		ypreds = []
		#if hasattr(self, ensembl_fl):
		model = self.best_mod
		if ensemble:
			i=0
			pred_dict = {"count": [], "df": []}
			top_pops = {"df": [], "pops": []}
			for checkpoint in self.ensembl_fl:
				model.load_weights(save_dir + checkpoint)
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
			freq_df.to_csv(save_dir + "/pop_assign_ensemble.csv",
						   index=False)
		else:
			model.load_weights(save_dir + "/default_mod_weights/checkpoint.h5")
			tmp_df = pd.DataFrame(model.predict(ukgen))
			tmp_df.columns = popnames
			tmp_df["sampleID"] = uksamples
			tmp_df.to_csv(save_dir + "/pop_assign.csv", index=False)
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