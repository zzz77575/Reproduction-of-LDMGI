import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys


def visualize_samples(X, y, dataset_name, output_path):
    """
    Visualize 5 random samples from a dataset and save the figure.

    Args:
        X (np.ndarray): Image data (n_samples, n_features).
        y (np.ndarray): Class labels (n_samples,).
        dataset_name (str): Name of the dataset.
        output_path (str): Directory to save the generated image.
    """

    # Define image dimensions based on dataset name
    img_shapes = {
        "coil20": (32, 32),
        "usps": (16, 16),
        "mnist_t": (28, 28),
        "mnist_s": (28, 28),
        "umist": (23, 28),
        "yaleb": (32, 32),
        "jaffe": (26, 26),
        "pointing04": (28, 40)
    }

    if dataset_name not in img_shapes:
        print(f"Error: Unknown dataset {dataset_name}")
        return

    img_size = img_shapes[dataset_name]

    # Randomly select 5 samples
    sample_indices = random.sample(range(len(X)), 5)

    # Create figure for visualization
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"{dataset_name} Samples", fontsize=16)
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, 5, i + 1)  # 1 row, 5 columns
        img = X[idx].reshape(img_size)  # Reshape to the corresponding image size
        plt.imshow(img, cmap='gray')
        plt.title(f"Class {y[idx]}")
        plt.axis('off')

    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_path, f"{dataset_name}_samples.png")
    plt.savefig(output_file)
    plt.close()  # Close the figure to free memory
    print(f"Saved samples for {dataset_name} to {output_file}.")


def load_dataset(dataset_name, processed_path):
    """Load dataset from NPZ file."""
    file_path = os.path.join(processed_path, f"{dataset_name}.npz")
    if os.path.exists(file_path):
        data = np.load(file_path)
        return data['X'], data['y']
    else:
        print(f"Error: Dataset '{dataset_name}' not found at {file_path}.")
        return None, None


def main():
    # Define the relative path for the processed data
    processed_path = "./data/processed"
    output_path = "./image_samples"  # Fixed the output directory path

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List of dataset names
    datasets = [
        "coil20",
        "usps",
        "mnist_t",
        "mnist_s",
        "umist",
        "yaleb",
        "jaffe",
        "pointing04"
    ]

    # Check if a specific dataset is provided as a command line argument
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        if dataset_name in datasets:
            X, y = load_dataset(dataset_name, processed_path)
            if X is not None:
                visualize_samples(X, y, dataset_name, output_path)
        else:
            print(f"Error: Dataset '{dataset_name}' is not in the list.")
    else:
        # Load and visualize samples for all datasets
        for dataset_name in datasets:
            X, y = load_dataset(dataset_name, processed_path)
            if X is not None:
                visualize_samples(X, y, dataset_name, output_path)


if __name__ == '__main__':
    main()