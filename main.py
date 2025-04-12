import os
import numpy as np
import argparse
from ldmgi import LDMGI
from kmeans import KMeans
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = './data/processed'
OUTPUT_DIR = './results'


def load_dataset(name):
    """Load dataset from .npz file"""
    data = np.load(os.path.join(PROCESSED_DATA_DIR, f"{name}.npz"))
    return data['X'], data['y']


def evaluate_model_on_dataset(model, dataset, X, y):
    """Initialize and run model on given dataset"""
    if model == 'ldmgi':
        model = LDMGI(k=5, trials=20, lambdas=[1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8], results_dir=OUTPUT_DIR)
    else:
        model = KMeans(trials=20, results_dir=OUTPUT_DIR)

    return model.evaluate(X, y, dataset.upper())


def save_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description='Run clustering evaluation')

    datasets = {
        '1': 'coil20',
        '2': 'jaffe',
        '3': 'pointing04',
        '4': 'umist',
        '5': 'yaleb',
        '6': 'usps',
        '7': 'mnist_t',
        '8': 'mnist_s',
        '9': 'mpeg7',
        '10': 'umist_gabor',
        '11': 'mpeg7_gray'
    }

    dataset_name_to_key = {v: k for k, v in datasets.items()}

    models = {
        '1': 'ldmgi',
        '2': 'kmeans'
    }
    model_name_to_key = {v: k for k, v in models.items()}

    parser.add_argument('--dataset', nargs='+', default=['coil20'],
                        help='Datasets to evaluate: ' + ', '.join(
                            [f'{k}:{v}' for k, v in datasets.items()]) + ' or "all" or dataset names directly')
    parser.add_argument('--model', nargs='+', default=["ldmgi"],
                        help='Clustering model to use: ' + ', '.join(
                            [f'{k}:{v}' for k, v in models.items()]) + ' or "all" or model names directly')

    args = parser.parse_args()
    all_stats = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:

        if 'all' in args.dataset:
            selected_datasets = list(datasets.values())
        else:
            selected_datasets = []
            for item in args.dataset:
                if item in datasets:
                    selected_datasets.append(datasets[item])
                elif item in dataset_name_to_key:
                    selected_datasets.append(item)
                else:
                    print(f"Error: Unrecognized dataset.\n"
                          f"Available datasets are: {', '.join(f'{k}: {v}' for k, v in datasets.items())}.")

        if 'all' in args.model:
            selected_models = list(models.values())
        else:
            selected_models = []
            for item in args.model:
                if item in models:
                    selected_models.append(models[item])
                elif item in model_name_to_key:
                    selected_models.append(item)
                else:
                    print(f"Error: Unrecognized model.\n"
                          f"Available models are: {', '.join(f'{k}: {v}' for k, v in models.items())}.")

        for dataset in selected_datasets:
            X, y = load_dataset(dataset)
            print(f"Loaded {dataset.upper()}: {X.shape[0]} samples, {len(np.unique(y))} classes")

            for model in selected_models:
                print(f"Running {model.upper()} on {dataset.upper()}...")
                all_stats[f"{model}_{dataset}"] = evaluate_model_on_dataset(model, dataset, X, y)
                plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if all_stats:
            save_dict_to_txt(all_stats, './results/results.txt')
            print("Results have been saved to 'results.txt'.")
        else:
            print("No results to save.")

if __name__ == '__main__':
    main()
