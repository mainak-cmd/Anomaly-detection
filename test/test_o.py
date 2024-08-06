import numpy as np
from joblib import Parallel, delayed
import time

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-3, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar  # Regularization term to avoid singular matrix

    def fit(self, X):
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_components, 1 / self.n_components)
        random_row = np.random.randint(low=0, high=n_samples, size=self.n_components)
        self.means = X[random_row, :]
        self.covariances = np.array([np.cov(X.T) + np.eye(n_features) * self.reg_covar for _ in range(self.n_components)])
        log_likelihood = 0

        for i in range(self.max_iter):
            responsibilities = self._expectation(X)
            self._maximization(X, responsibilities)
            new_log_likelihood = self._log_likelihood(X)
            if abs(new_log_likelihood - log_likelihood) <= self.tol:
                break
            log_likelihood = new_log_likelihood

    def _expectation(self, X):
        likelihood = np.array([
            self._multivariate_gaussian(X, self.means[k], self.covariances[k])
            for k in range(self.n_components)
        ]).T
        weighted_likelihood = likelihood * self.weights
        responsibilities = weighted_likelihood / np.sum(weighted_likelihood, axis=1)[:, np.newaxis]
        return responsibilities

    def _maximization(self, X, responsibilities):
        N_k = np.sum(responsibilities, axis=0)
        self.weights = N_k / len(X)
        self.means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
        self.covariances = np.array([
            np.dot((responsibilities[:, k] * (X - self.means[k]).T), (X - self.means[k])) / N_k[k] + np.eye(X.shape[1]) * self.reg_covar
            for k in range(self.n_components)
        ])

    def _log_likelihood(self, X):
        likelihood = np.array([
            self._multivariate_gaussian(X, self.means[k], self.covariances[k])
            for k in range(self.n_components)
        ]).T
        log_likelihood = np.sum(np.log(np.dot(likelihood, self.weights)))
        return log_likelihood

    def _multivariate_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1))
        return exp_term / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(cov))

    def predict(self, X):
        responsibilities = self._expectation(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        responsibilities = self._expectation(X)
        return responsibilities

    def centroids(self):
        return self.means

class ClusterCentroid:
    def __init__(self, max_iter=20, cores=4, timeout=300):
        self.max_iter = max_iter
        self.cores = cores
        self.timeout = timeout

    def _silhouette_score(self, X, labels):
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters <= 1:
            return -1  # Silhouette score is invalid for 1 or fewer clusters

        distances = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        silhouette_scores = np.zeros(n_samples)

        for i in range(n_samples):
            current_label = labels[i]
            same_cluster_mask = labels == current_label
            a_i = np.mean(distances[i, same_cluster_mask] + np.finfo(float).eps)
            b_i = np.inf
            for label in unique_labels:
                if label == current_label:
                    continue
                other_cluster_mask = labels == label
                b_i = min(b_i, np.mean(distances[i, other_cluster_mask] + np.finfo(float).eps))
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        return np.mean(silhouette_scores)

    def _compute_silhouette(self, X, n_components):
        gmm = GaussianMixtureModel(n_components=n_components)
        gmm.fit(X)
        labels = gmm.predict(X)
        try:
            silhouette_avg = self._silhouette_score(X, labels)
        except ValueError:
            silhouette_avg = -1
        return silhouette_avg, n_components

    def cluster_make(self, X):
        silhouette_avg_list = []
        components_list = []

        start_time = time.time()
        results = Parallel(n_jobs=self.cores)(
            delayed(self._compute_silhouette)(X, i) for i in range(2, self.max_iter)
        )

        for silhouette_avg, n_components in results:
            if silhouette_avg != -1:  # Skip invalid silhouette scores
                components_list.append(n_components)
                silhouette_avg_list.append(silhouette_avg)

            if time.time() - start_time > self.timeout:
                if silhouette_avg_list:
                    max_silhouette_index = np.argmax(silhouette_avg_list)
                    best_component_outlier = components_list[max_silhouette_index]
                    best_silhouette_score_outlier = silhouette_avg_list[max_silhouette_index]
                    break
                else:
                    return None, None  # Return None if no valid result is found
        else:
            if silhouette_avg_list:  # Check if any valid scores were computed
                max_silhouette_index = np.argmax(silhouette_avg_list)
                best_component_outlier = components_list[max_silhouette_index]
                best_silhouette_score_outlier = silhouette_avg_list[max_silhouette_index]
            else:
                return None, None  # Return None if no valid result is found

        best_gmm = GaussianMixtureModel(n_components=best_component_outlier)
        best_gmm.fit(X)
        centroids_out = best_gmm.centroids()

        return centroids_out, best_component_outlier



from sklearn.datasets import make_blobs

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Initialize the ClusterCentroid class
clusterer = ClusterCentroid(max_iter=20, cores=4, timeout=300)

# Perform clustering
centroids, best_n_components = clusterer.cluster_make(X)

# Print the results
print(f"Best number of components: {best_n_components}")
print(f"Centroids of the clusters:\n{centroids}")
