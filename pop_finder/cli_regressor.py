import argparse
from pop_finder.contour_classifier import contour_classifier
from pop_finder.contour_classifier import kfcv


def main():

    parser = argparse.ArgumentParser(
        prog='contour_classifier',
        description='Contour wrapper for pop assignment'
    )

    # Arguments for deciding which function to use
    parser.add_argument('--contour_classifier', action="store_true")
    parser.add_argument('--kfcv', action="store_true")

    # Shared arguments - non optional
    parser.add_argument('sample_data', type=str,
                        help="Path to sample data")

    # Shared arguments - optional
    parser.add_argument('--gen_dat', type=str, default=None,
                        help="Path to genetic data")
    parser.add_argument('--save_dir', type=str, default="out",
                        help="Output directory to save results to")

    # Contour_classifier arguments
    parser.add_argument('--num_contours', type=int, default=10,
                        help="Number of contours")
    parser.add_argument('--run_locator', action="store_true",
                        help="Run instead of using results in save_dir")
    parser.add_argument('--nboots', type=int, default=50,
                        help="Number of bootstrap iterations")
    parser.add_argument('--return_plots', action="store_true",
                        help="Return plots of results")
    parser.add_argument('--return_df', action="store_true",
                        help="Return dataframe of results")
    parser.add_argument('--multi_iter', type=int, default=1,
                        help="Number of iterations to run")

    # kfcv arguments
    parser.add_argument('--n_splits', type=int, default=5,
                        help="Number of splits for K-Fold CV")
    parser.add_argument('--n_runs', type=int, default=5,
                        help="Number of repetitions for K-Fold CV")
    parser.add_argument('--return_plot', action="store_true")

    # locator_mod args - **kwargs
    parser.add_argument('--train_split', type=float, default=0.9,
                        help="Proportion of data for training")
    parser.add_argument('--jacknife', action="store_true",
                        help="Run jacknife on locator")
    parser.add_argument('--jacknife_prop', type=float, default=0.05,
                        help="Proportion for jacknife")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for model")
    parser.add_argument('--max_epochs', type=int, default=5000,
                        help="Number of epochs to run model")
    parser.add_argument('--patience', type=int, default=10,
                        help="Patience of model for early stopping")
    parser.add_argument('--min_mac', type=int, default=2,
                        help="Minimum minor allele count")
    parser.add_argument('--max_SNPs', type=int, default=None,
                        help="Maximum number of SNPs to use")
    parser.add_argument('--impute_missing', action="store_true",
                        help="Impute missing data")
    parser.add_argument('--dropout_prop', type=float, default=0.25,
                        help="Dropout proportion")
    parser.add_argument('--nlayers', type=int, default=10,
                        help="Number of layers in network")
    parser.add_argument('--width', type=int, default=256,
                        help="Width or number of nodes per layer")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for locator")
    parser.add_argument('--gpu_number', type=str, default=None,
                        help="GPU number (coming soon...)")
    parser.add_argument('--plot_history', action="store_false",
                        help="Plot training / validation history")
    parser.add_argument('--keep_weights', action="store_true",
                        help="Save weights for future")
    parser.add_argument('--load_params', type=str, default=None,
                        help="Path to json params file with model args")
    parser.add_argument('--keras_verbose', type=int, default=1,
                        help="How verbose keras output is, from 0-2")

    args = parser.parse_args()

    if args.contour_classifier:
        print("Running locator with contour classifier")
        contour_classifier(
            sample_data=args.sample_data,
            num_contours=args.num_contours,
            run_locator=args.run_locator,
            gen_dat=args.gen_dat,
            nboots=args.nboots,
            return_plots=args.return_plots,
            return_df=args.return_df,
            save_dir=args.save_dir,
            multi_iter=args.multi_iter,
            train_split=args.train_split,
            jacknife=args.jacknife,
            jacknife_prop=args.jacknife_prop,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_mac=args.min_mac,
            max_SNPs=args.max_SNPs,
            impute_missing=args.impute_missing,
            dropout_prop=args.dropout_prop,
            nlayers=args.nlayers,
            width=args.width,
            seed=args.seed,
            gpu_number=args.gpu_number,
            plot_history=args.plot_history,
            keep_weights=args.keep_weights,
            load_params=args.load_params,
            keras_verbose=args.keras_verbose,
        )

    elif args.kfcv:
        print("Running K-Fold Cross-Validation")
        kfcv(
            sample_data=args.sample_data,
            gen_dat=args.gen_dat,
            n_splits=args.n_splits,
            n_runs=args.n_runs,
            return_plot=args.return_plot,
            num_contours=args.num_contours,
            nboots=args.nboots,
            save_dir=args.save_dir,
            multi_iter=args.multi_iter,
            train_split=args.train_split,
            jacknife=args.jacknife,
            jacknife_prop=args.jacknife_prop,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_mac=args.min_mac,
            max_SNPs=args.max_SNPs,
            impute_missing=args.impute_missing,
            dropout_prop=args.dropout_prop,
            nlayers=args.nlayers,
            width=args.width,
            seed=args.seed,
            gpu_number=args.gpu_number,
            plot_history=args.plot_history,
            keep_weights=args.keep_weights,
            load_params=args.load_params,
            keras_verbose=args.keras_verbose,
        )

    else:
        raise ValueError("No function specified in arguments")


if __name__ == "__main__":
    main()
