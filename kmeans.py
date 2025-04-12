import numpy as np
import os
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans as SKLearnKMeans
from tqdm import tqdm
import time

class KMeans:
    """
    K-means clustering.
    """

    def __init__(self, trials=20, results_dir="results"):
        """
        Initialize KMeans evaluation with multiple trials.
        """
        self.trials = trials
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # Store results
        self.best_results = None
        self.all_results = []

    @staticmethod
    def eval_clustering(labels, labels_pred, c):
        """
        Evaluate clustering results with accuracy and NMI.
        """
        # Compute confusion matrix
        weight_matrix = np.zeros((c, c))
        for key1 in range(c):
            pred_indices = {index for index, value in enumerate(labels_pred) if value == key1}
            for key2 in range(c):
                true_indices = {index for index, value in enumerate(labels) if value == key2}
                common_elements = pred_indices.intersection(true_indices)
                weight_matrix[key1, key2] = len(common_elements)

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-weight_matrix)

        # Calculate accuracy
        n = len(labels)
        accuracy = weight_matrix[row_ind, col_ind].sum() / n

        # Calculate NMI
        nmi = normalized_mutual_info_score(labels, labels_pred, average_method='geometric')

        return accuracy, nmi

    def evaluate(self, images_array, labels, dataset_name):
        """
        Evaluate using multiple initializations.
        """
        print(f"\n==== Evaluating {dataset_name} with K-means ====")
        print(f"- Shape: {images_array.shape}, Samples: {len(labels)}")

        c = max(labels) + 1
        print(f"- Clusters: {c}")
        print(f"- Running {self.trials} initializations")

        # Prepare data
        flattened_images = images_array.reshape(images_array.shape[0], -1)

        all_results = []

        # Run multiple trials with different random initializations
        for trial in tqdm(range(self.trials), desc="Initializations"):
            seed = int(time.time() * 1000) % 10000 + trial

            # Run K-means clustering
            start_time = time.time()

            # Initialize with scikit-learn's KMeans
            km = SKLearnKMeans(
                n_clusters=c,
                init='k-means++',
                n_init=1,
                max_iter=300,
                tol=1e-4,
                random_state=seed
            )

            # Fit the model
            km.fit(flattened_images)
            pred = km.labels_

            runtime = time.time() - start_time

            # Evaluate
            acc, nmi = self.eval_clustering(labels, pred, c)

            # Store result
            result = {
                'trial': trial,
                'acc': acc,
                'nmi': nmi,
                'time': runtime,
                'pred': pred
            }
            all_results.append(result)

        # Store all results
        self.all_results = all_results

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Save results to CSV
        csv_path = os.path.join(self.results_dir, f"{dataset_name}_kmeans_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Calculate and print statistics
        stats = self._calc_stats(df, dataset_name)
        self.best_results = stats
        return stats

    def _calc_stats(self, df, dataset_name):
        """Calculate statistics from results."""

        # Best overall results
        best_acc_row = df.loc[df['acc'].idxmax()]
        best_nmi_row = df.loc[df['nmi'].idxmax()]

        # Calculate overall statistics
        mean_acc = df['acc'].mean()
        std_acc = df['acc'].std()
        max_acc = df['acc'].max()

        mean_nmi = df['nmi'].mean()
        std_nmi = df['nmi'].std()
        max_nmi = df['nmi'].max()

        mean_time = df['time'].mean()

        # Print summary of results
        print("\n===== K-MEANS CLUSTERING RESULTS =====")
        print(f"BEST OVERALL RESULTS:")
        print(f"- Best ACC: {max_acc * 100:.2f} (Trial #{int(best_acc_row['trial'])}, NMI={best_acc_row['nmi'] * 100:.2f})")
        print(f"- Best NMI: {max_nmi * 100:.2f} (Trial #{int(best_nmi_row['trial'])}, ACC={best_nmi_row['acc'] * 100:.2f})")

        print(f"\nMEAN PERFORMANCE:")
        print(f"- MEAN ACC: {mean_acc * 100:.2f} \u00B1 {std_acc * 100:.2f}")
        print(f"- MEAN NMI: {mean_nmi * 100:.2f} \u00B1 {std_nmi * 100:.2f}")
        # Store stats
        stats = {
            'dataset': dataset_name,
            'best_acc': {
                'value': max_acc,
                'nmi': best_acc_row['nmi']
            },
            'best_nmi': {
                'value': max_nmi,
                'acc': best_nmi_row['acc']
            },
            'overall': {
                'acc_mean': mean_acc,
                'acc_std': std_acc,
                'nmi_mean': mean_nmi,
                'nmi_std': std_nmi
            }
        }

        return stats