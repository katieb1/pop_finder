import argparse
from pop_finder.pop_finder import hyper_tune
from pop_finder.pop_finder import kfcv
from pop_finder.pop_finder import run_neural_net
from pop_finder.pop_finder import snp_rank
from pop_finder.pop_finder import assign_plot
from pop_finder.pop_finder import structure_plot


def main():

    parser = argparse.ArgumentParser(
        prog='pop_finder',
        description='Neural network for pop assignment'
    )

    # Arguments for determining function to use
    parser.add_argument('--hyper_tune', action='store_true')
    parser.add_argument('--kfcv', action='store_true')
    parser.add_argument('--run_neural_net', action='store_true')
    parser.add_argument('--snp_rank', action='store_true')

    # Shared function arguments - not optional
    parser.add_argument('infile',
                        type=str, help="Path to genetic data file")
    parser.add_argument('sample_data',
                        type=str, help="Path to sample data file")

    # Shared function arguments - optional
    parser.add_argument('--mod_path', type=str, default=None,
                        help="Path to tuned model (save_dir from hyper_tuner)")
    parser.add_argument('--train_prop', type=float, default=0.8,
                        help="Proportion of data for training")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed value")
    parser.add_argument('--save_dir', type=str, default='out',
                        help="Directory to save output to")
    parser.add_argument('--ensemble', action='store_true',
                        help="Use ensemble of models")
    parser.add_argument('--save_allele_counts', action='store_true',
                        help="Save allele counts in hdf5 file")
    parser.add_argument('--try_stacking', action='store_true',
                        help="Use weighted averages of models")
    parser.add_argument('--max_epochs', type=int, default=100,
                        help="Number of epochs in neural network")
    parser.add_argument('--col_scheme', type=str, default="Spectral",
                        help="Colour scheme of structure / assign plot")

    # Function arguments for hyper_tune
    parser.add_argument('--max_trials', type=int, default=10,
                        help="Number of trials for hyperparameter tuner")
    parser.add_argument('--runs_per_trial', type=int, default=10,
                        help="Number of runs / trial for hyperparameter tuner")
    parser.add_argument('--mod_name', type=str, default='hyper_tune',
                        help="Name of project")

    # Function arguments for kfcv
    parser.add_argument('--n_splits', type=int, default=5,
                        help="Number of splits for K-fold CV")
    parser.add_argument('--n_reps', type=int, default=5,
                        help="Number of repetitions for K-fold CV")
    parser.add_argument('--return_plot', action='store_false',
                        help="Return confusion matrix plot")

    # Function arguments for pop_finder (**kwargs)
    parser.add_argument('--nbags', type=int, default=10,
                        help="Number of models if ensemble is True")
    parser.add_argument('--predict', action='store_true',
                        help="Predict on unknown samples")
    parser.add_argument('--save_weights', action='store_true',
                        help="Save model weights for later use")
    parser.add_argument('--patience', type=int, default=20,
                        help="Set model patience for early stopping")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Set batch size")
    parser.add_argument('--gpu_number', type=str, default="0",
                        help="Set GPU number (coming soon...)")
    parser.add_argument('--plot_history', action='store_true',
                        help="Plot training / validation history")

    args = parser.parse_args()

    if args.hyper_tune:
        print("Tuning hyperparameters of model")
        hyper_tune(
            infile=args.infile,
            sample_data=args.sample_data,
            max_trials=args.max_trials,
            runs_per_trial=args.runs_per_trial,
            max_epochs=args.max_epochs,
            train_prop=args.train_prop,
            seed=args.seed,
            save_dir=args.save_dir,
            mod_name=args.mod_name,
        )

    elif args.kfcv:
        print("Running K-Fold Cross-Validation")
        kfcv(
            infile=args.infile,
            sample_data=args.sample_data,
            mod_path=args.mod_path,
            n_splits=args.n_splits,
            n_reps=args.n_reps,
            ensemble=args.ensemble,
            save_dir=args.save_dir,
            return_plot=args.return_plot,
            save_allele_counts=args.save_allele_counts,
            try_stacking=args.try_stacking,
            nbags=args.nbags,
            train_prop=args.train_prop,
            predict=args.predict,
            save_weights=args.save_weights,
            patience=args.patience,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            gpu_number=args.gpu_number,
            plot_history=args.plot_history,
            seed=args.seed,
        )
        structure_plot(
            save_dir=args.save_dir,
            ensemble=args.ensemble,
            col_scheme=args.col_scheme
        )

    elif args.run_neural_net:
        print("Training neural network and generating predictions")
        run_neural_net(
            infile=args.infile,
            sample_data=args.sample_data,
            save_allele_counts=args.save_allele_counts,
            mod_path=args.mod_path,
            train_prop=args.train_prop,
            seed=args.seed,
            ensemble=args.ensemble,
            try_stacking=args.try_stacking,
            nbags=args.nbags,
            save_dir=args.save_dir,
            save_weights=args.save_weights,
            patience=args.patience,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            gpu_number=args.gpu_number,
            plot_history=args.plot_history,
        )
        assign_plot(
            save_dir=args.save_dir,
            ensemble=args.ensemble,
            col_scheme=args.col_scheme
        )

    elif args.snp_rank:
        print("Finding most important SNPs")
        snp_rank(
            infile=args.infile,
            sample_data=args.sample_data,
            mod_path=args.mod_path,
            save_dir=args.save_dir
        )
    else:
        raise ValueError("No function specified in arguments")


if __name__ == "__main__":
    main()
