from data_loaders import COIL20Loader, JAFFELoader, Pointing04Loader, UMISTLoader, \
    YaleBLoader, USPSLoader, MNISTTLoader, MNISTSLoader, MPEG7Loader, UMISTGaborLoader, MPEG7GrayLoader
import numpy as np
import os
import sys

"""
Load a specific dataset using command-line arguments.
Print loading information for the specified dataset.
By default, load all datasets.
"""

def load_and_save_dataset(loader, dataset_name, data_path, processed_path, verbose):
    """Helper function to load and save a dataset."""
    X, y = loader.load(data_path, verbose=verbose)  # Pass verbose to the loader
    np.savez(os.path.join(processed_path, f"{dataset_name}.npz"), X=X, y=y)
    print(f"*** {dataset_name.upper()} DATASET LOADED SUCCESSFULLY! *** "
          f"{X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")

# The rest of your main function remains unchanged.
def main():
    # Define the relative path for the raw data
    data_path = "./data/raw"
    processed_path = "./data/processed"

    # Ensure the processed directory exists
    os.makedirs(processed_path, exist_ok=True)

    # Create loader instances
    loaders = {
        "coil20": COIL20Loader(),
        "jaffe": JAFFELoader(),
        "pointing04": Pointing04Loader(),
        "umist": UMISTLoader(),
        "yaleb": YaleBLoader(),
        "usps": USPSLoader(),
        "mnist_t": MNISTTLoader(),
        "mnist_s": MNISTSLoader(),
        "mpeg7": MPEG7Loader(),
        "umist_gabor": UMISTGaborLoader(),
        "mpeg7_gray": MPEG7GrayLoader()
    }

    # Check if a specific dataset is requested
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()
        if dataset_name in loaders:
            load_and_save_dataset(loaders[dataset_name], dataset_name, data_path, processed_path, verbose=True)  # Set verbose=True
        else:
            print(f"Error: Dataset '{dataset_name}' not recognized. Available options are: {', '.join(loaders.keys())}")
    else:
        # Load all datasets
        for name, loader in loaders.items():
            load_and_save_dataset(loader, name, data_path, processed_path, verbose=False)
        print("\n*** ALL DATASETS HAVE BEEN SAVED TO NPZ FILES! ***")

if __name__ == '__main__':
    main()