import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm


class LDMGI:

    def __init__(self, k=5, convergence_threshold=0.00001, trials=20,
                 lambdas=None, results_dir="results"):
        self.k = k
        self.p = convergence_threshold
        self.trials = trials

        if lambdas is None:
            self.lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]
        else:
            self.lambdas = lambdas

        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.best_results = None
        self.all_results = []

    def find_nearest_neighbors(self, image_set, k):
        image_set = image_set.reshape(image_set.shape[0], -1)
        distances = cdist(image_set, image_set)

        neighbors_indices = []
        for i in range(distances.shape[0]):
            nearest_indices = distances[i].argsort()[:k]
            neighbors_indices.append(nearest_indices.tolist())

        return neighbors_indices

    def Hd(self, d):
        I = np.eye(d)
        ones = np.ones((d, 1))
        H_d = I - (1 / d) * (ones @ ones.T)
        return H_d

    def L_to_diagonal(self, Li, n, k):
        L = np.zeros((n * k, n * k))
        for i in range(n):
            L[i * k:(i + 1) * k, i * k:(i + 1) * k] = Li[i, :, :]
        return L

    def f(self, X):
        if np.iscomplexobj(X):
            X = np.real(X)

        Z = X @ sqrtm(np.linalg.inv(X.T @ X))
        return Z

    def inv_f(self, Z):
        X = np.diag(np.diag(Z @ Z.T) ** (-1 / 2)) @ Z
        return X

    def cal_local_models(self, X, nearest_neighbors, n, m, k, lam):
        Xi = np.zeros((n, m, k))
        for i in range(n):
            X_indices = nearest_neighbors[i, :]
            Xi[i, :, :] = X[:, X_indices]

        X_tilde = np.zeros((n, m, k))
        for i in range(n):
            X_tilde[i, :, :] = Xi[i, :, :] @ self.Hd(k)

        Li = np.zeros((n, k, k))
        for i in range(n):
            Li[i, :, :] = self.Hd(k) @ np.linalg.inv(
                np.dot(X_tilde[i, :, :].T, X_tilde[i, :, :]) + lam * np.eye(k)
            ) @ self.Hd(k)

        Si = np.zeros((n, n, k))
        for i in range(n):
            F = nearest_neighbors[i, :]
            for q in range(k):
                Si[i, F[q], q] = 1

        return Li, Si, X_tilde

    def cal_global_L(self, Li, Si, n, k):
        L_ = self.L_to_diagonal(Li, n, k)
        S = Si.transpose(1, 0, 2).reshape(n, n * k)
        L = S @ L_ @ S.T
        return L

    def solve_eigen_problem(self, L, n, c):
        L = (L + L.T) / 2
        L_eigenvalues, L_eigenvectors = np.linalg.eigh(L)
        L_sorted_indices = np.argsort(L_eigenvalues)
        V = L_eigenvectors[:, L_sorted_indices][:, :c]
        return V

    def discrete_opt(self, Z, n, c, p, seed=None):
        if seed is not None:
            np.random.seed(seed)

        X_ = self.inv_f(Z)
        i = np.random.randint(0, n)
        R = np.zeros((c, c))
        R[:, 0] = X_[i, :]

        d = np.zeros((n, 1))
        for k in range(2, c + 1):
            d += np.abs(X_ @ R[:, k - 2].reshape(-1, 1))
            i = np.argmin(d)
            R[:, k - 1] = X_[i, :]

        phi_ = 1
        phi = 0
        t = 0

        while abs(phi - phi_) > p:
            t += 1
            phi_ = phi
            X_t = X_ @ R
            X_st = np.zeros((n, c))
            for i in range(n):
                for l in range(c):
                    if l == np.argmax(X_t[i, :]):
                        X_st[i, l] = 1

            U, o, U_ = np.linalg.svd(X_st.T @ X_)
            phi = np.sum(o)
            R = U_.T @ U.T

        labels_pred = np.array([np.where(row == 1)[0][0] for row in X_st])
        return labels_pred

    @staticmethod
    def eval_clustering(labels_true, labels_pred, n_clusters):
        weight_matrix = np.zeros((n_clusters, n_clusters))
        for cluster_id in range(n_clusters):
            pred_indices = {index for index, value in enumerate(labels_pred) if value == cluster_id}
            for class_id in range(n_clusters):
                true_indices = {index for index, value in enumerate(labels_true) if value == class_id}
                common_elements = pred_indices.intersection(true_indices)
                weight_matrix[cluster_id, class_id] = len(common_elements)

        row_ind, col_ind = linear_sum_assignment(-weight_matrix)
        n_samples = len(labels_true)
        accuracy = weight_matrix[row_ind, col_ind].sum() / n_samples
        nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
        return accuracy, nmi

    def evaluate(self, images_array, labels, dataset_name):
        print(f"\n==== Evaluating {dataset_name} ====")
        print(f"- Shape: {images_array.shape}, Samples: {len(labels)}")

        c = max(labels) + 1
        print(f"- Clusters: {c}")
        print(f"- Running {self.trials} trials \u00D7 {len(self.lambdas)} parameters")

        flattened_images = images_array.reshape(images_array.shape[0], -1)
        nearest_neighbors = self.find_nearest_neighbors(images_array, self.k)
        nearest_neighbors = np.array(nearest_neighbors)
        X = flattened_images.T
        m, n = X.shape

        all_results = []
        mean_results = []

        for lam in tqdm(self.lambdas, desc="Parameters"):
            print(f"\n \u03BB = {lam:.1e}")

            Li, Si, X_tilde = self.cal_local_models(X, nearest_neighbors, n, m, self.k, lam)
            L = self.cal_global_L(Li, Si, n, self.k)
            Z = self.solve_eigen_problem(L, n, c)

            trial_results = []
            for trial in range(self.trials):
                seed = int(time.time() * 1000) % 10000 + trial

                start_time = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = self.discrete_opt(Z, n, c, self.p, seed=seed)

                runtime = time.time() - start_time
                acc, nmi = self.eval_clustering(labels, pred, c)

                result = {
                    'trial': trial,
                    'lambda': lam,
                    'acc': acc,
                    'nmi': nmi,
                    'time': runtime,
                    'pred': pred
                }
                trial_results.append(result)
                all_results.append(result)

            acc_values = [r['acc'] for r in trial_results]
            nmi_values = [r['nmi'] for r in trial_results]

            mean_acc = np.mean(acc_values)
            std_acc = np.std(acc_values)
            max_acc = np.max(acc_values)

            mean_nmi = np.mean(nmi_values)
            std_nmi = np.std(nmi_values)
            max_nmi = np.max(nmi_values)

            mean_results.append({
                'lambda': lam,
                'mean_acc': mean_acc,
                'mean_nmi': mean_nmi,
                'std_acc': std_acc,
                'std_nmi': std_nmi
            })

            print(f"  ACC - Mean: {mean_acc * 100:.2f} \u00B1 {std_acc * 100:.2f}, Max: {max_acc * 100:.2f}")
            print(f"  NMI - Mean: {mean_nmi * 100:.2f} \u00B1 {std_nmi * 100:.2f}, Max: {max_nmi * 100:.2f}")

        self.all_results = all_results
        df = pd.DataFrame(all_results)

        csv_path = os.path.join(self.results_dir, f"{dataset_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        print("\n===== LDMGI CLUSTERING RESULTS =====")

        mean_df = pd.DataFrame(mean_results)
        best_mean_nmi_row = mean_df.loc[mean_df['mean_nmi'].idxmax()]
        best_mean_acc_row = mean_df.loc[mean_df['mean_acc'].idxmax()]

        best_nmi_row = df.loc[df['nmi'].idxmax()]
        best_acc_row = df.loc[df['acc'].idxmax()]

        print(f"BEST OVERALL RESULTS:")
        print(f"- Best ACC: {best_acc_row['acc'] * 100:.2f}")
        print(f"- Best NMI: {best_nmi_row['nmi'] * 100:.2f}")

        print(f"\nBEST MEAN PERFORMANCE:")
        print(
            f"- BEST MEAN ACC: {best_mean_acc_row['mean_acc'] * 100:.2f} \u00B1 {best_mean_acc_row['std_acc'] * 100: .2f} (\u03BB={best_mean_acc_row['lambda']:.1e})")
        print(
            f"- BEST MEAN NMI: {best_mean_nmi_row['mean_nmi'] * 100:.2f} \u00B1 {best_mean_nmi_row['std_nmi'] * 100: .2f} (\u03BB={best_mean_nmi_row['lambda']:.1e})")

        self._plot_results(df, dataset_name)

        # Store stats
        stats = {
            'dataset': dataset_name,
            'best_overall': {
                'best_acc': best_acc_row['acc'],
                'best_nmi': best_nmi_row['nmi']
            },
            # 'lambda_stats': mean_df,
            'best_mean': {
                'mean_acc': best_mean_acc_row['mean_acc'],
                "std_acc": best_mean_acc_row['std_acc'],
                'mean_nmi': best_mean_nmi_row['mean_nmi'],
                "std_nmi": best_mean_nmi_row['std_nmi'],
                'lambda': best_mean_acc_row['lambda']
            }
        }

        return stats

    def _plot_results(self, df, dataset_name):
        """Plot parameter sensitivity analysis"""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Group by lambda
        lambda_stats = df.groupby('lambda').agg({
            'acc': ['mean', 'std', 'max'],
            'nmi': ['mean', 'std', 'max']
        })

        # Log scale for x-axis
        x = np.log10(self.lambdas)

        # Plot ACC
        mean_acc = lambda_stats['acc']['mean'].values
        std_acc = lambda_stats['acc']['std'].values
        max_acc = lambda_stats['acc']['max'].values

        ax1.errorbar(x, mean_acc, yerr=std_acc, fmt='o-', capsize=5, linewidth=3,
                     label='Mean ACC \u00B1 std')
        ax1.plot(x, max_acc, 's--', color='red', linewidth=3, label='Best ACC')

        # Highlight best mean and best overall
        best_mean_idx = np.argmax(mean_acc)
        best_max_idx = np.argmax(max_acc)

        ax1.plot(x[best_mean_idx], mean_acc[best_mean_idx], 'o',
                 ms=10, mfc='none', mec='blue', label='Best Mean')
        ax1.plot(x[best_max_idx], max_acc[best_max_idx], 's',
                 ms=10, mfc='none', mec='red', label='Best Overall')

        ax1.set_xlabel('\u03BB', fontsize=14)
        ax1.set_ylabel('ACC', fontsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_title(f'{dataset_name} - ACC vs \u03BB', fontsize=18)
        ax1.grid(True)
        ax1.legend(fontsize=15)

        # Add x-ticks with lambda values
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'10^{int(xi)}' for xi in x], fontsize=14)

        # Plot NMI
        mean_nmi = lambda_stats['nmi']['mean'].values
        std_nmi = lambda_stats['nmi']['std'].values
        max_nmi = lambda_stats['nmi']['max'].values

        ax2.errorbar(x, mean_nmi, yerr=std_nmi, fmt='o-', capsize=5, linewidth=3,
                     label='Mean NMI \u00B1 std')
        ax2.plot(x, max_nmi, 's--', color='red', linewidth=3, label='Best NMI')

        # Highlight best mean and best overall
        best_mean_idx = np.argmax(mean_nmi)
        best_max_idx = np.argmax(max_nmi)

        ax2.plot(x[best_mean_idx], mean_nmi[best_mean_idx], 'o',
                 ms=10, mfc='none', mec='blue', label='Best Mean')
        ax2.plot(x[best_max_idx], max_nmi[best_max_idx], 's',
                 ms=10, mfc='none', mec='red', label='Best Overall')

        ax2.set_xlabel('\u03BB', fontsize=14)
        ax2.set_ylabel('NMI', fontsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.set_title(f'{dataset_name} - NMI vs \u03BB', fontsize=18)
        ax2.grid(True)
        ax2.legend(fontsize=15)

        # Add x-ticks with lambda values
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'10^{int(xi)}' for xi in x], fontsize=14)

        plt.tight_layout()

        # Save figure
        plt_file = os.path.join(self.results_dir, f"{dataset_name}_params.png")
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Plot saved to {plt_file}")

