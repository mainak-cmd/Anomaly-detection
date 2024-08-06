import numpy as np
from joblib import Parallel, delayed
import time

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components  # Number of clusters
        self.max_iter = max_iter          # Maximum number of iterations
        self.tol = tol                    # Tolerance for convergence
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize weights, means, and covariances
        self.weights = np.full(self.n_components, 1 / self.n_components)
        random_row = np.random.randint(low=0, high=n_samples, size=self.n_components)
        self.means = X[random_row, :]
        self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])
        
        log_likelihood = 0

        for i in range(self.max_iter):
            responsibilities = self._expectation(X)
            self._maximization(X, responsibilities)
            new_log_likelihood = self._log_likelihood(X)
            if abs(new_log_likelihood - log_likelihood) <= self.tol:
                break
            log_likelihood = new_log_likelihood
    
    def _expectation(self, X):
        likelihood = np.array([self._multivariate_gaussian(X, self.means[k], self.covariances[k])
                               for k in range(self.n_components)]).T
        weighted_likelihood = likelihood * self.weights
        responsibilities = weighted_likelihood / np.sum(weighted_likelihood, axis=1)[:, np.newaxis]
        return responsibilities
    
    def _maximization(self, X, responsibilities):
        N_k = np.sum(responsibilities, axis=0)
        self.weights = N_k / len(X)
        self.means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
        self.covariances = np.array([
            np.dot((responsibilities[:, k] * (X - self.means[k]).T), (X - self.means[k])) / N_k[k]
            for k in range(self.n_components)
        ])
    def _log_likelihood(self, X):
        likelihood = np.array([self._multivariate_gaussian(X, self.means[k], self.covariances[k])
                               for k in range(self.n_components)]).T
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

def silhouette_score(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        raise ValueError("Number of clusters must be greater than 1 for silhouette score calculation.")

    # Compute distances between points
    distances = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

    # Initialize silhouette score
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Get the cluster label of the current sample
        current_label = labels[i]

        # Compute intra-cluster distance a(i)
        same_cluster_mask = labels == current_label
        a_i = np.mean(distances[i, same_cluster_mask] + np.finfo(float).eps)  # Add eps to avoid division by zero

        # Compute inter-cluster distance b(i)
        b_i = np.inf
        for label in unique_labels:
            if label == current_label:
                continue
            other_cluster_mask = labels == label
            b_i = min(b_i, np.mean(distances[i, other_cluster_mask] + np.finfo(float).eps))  # Add eps to avoid division by zero

        # Compute silhouette score s(i)
        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

    # Return the average silhouette score
    return np.mean(silhouette_scores)


def cluster_make(X, max_iter=20, cores=4, timeout=300):
    silhouette_avg_list = []
    components_list = []
    
    start_time = time.time()

    # Using joblib to parallelize the silhouette score computation
    results = Parallel(n_jobs=cores)(
        delayed(silhouette_score)(X, i) for i in range(3, max_iter)
    )

    for silhouette_avg, n_components in results:
        components_list.append(n_components)
        silhouette_avg_list.append(silhouette_avg)

        if time.time() - start_time > timeout:
            # If timeout is reached, return the best result found so far
            max_silhouette_index = np.argmax(silhouette_avg_list)
            best_component_outlier = components_list[max_silhouette_index]
            best_silhouette_score_outlier = silhouette_avg_list[max_silhouette_index]
            break
    else:
        # If no timeout, find the best component based on silhouette scores
        max_silhouette_index = np.argmax(silhouette_avg_list)
        best_component_outlier = components_list[max_silhouette_index]
        best_silhouette_score_outlier = silhouette_avg_list[max_silhouette_index]

    best_gmm = GaussianMixtureModel(n_components=best_component_outlier)
    best_gmm.fit(X)
    centroids_out = best_gmm.centroids()

# Example usage:
if __name__ == "__main__":
    # Create some sample data
    np.random.seed(0)
    X = np.vstack([np.random.normal(0, 1, (100, 2)), np.random.normal(5, 1, (100, 2))])
    # Fit GMM
  
    print(cluster_make(X, max_iter=20, cores=4, timeout=300))