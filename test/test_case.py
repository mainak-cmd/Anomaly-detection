import anamolyDetection
from anamolyDetection import anamoly_detection 

import os
import shutil

def delete_pycache(root_dir='.'):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            shutil.rmtree(pycache_path)
            print(f"Deleted: {pycache_path}")

delete_pycache()





from sklearn.datasets import make_blobs

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
X1, _ = make_blobs(n_samples=3000, centers=4, cluster_std=0.60, random_state=0)
Y1, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)
# Initialize the ClusterCentroid class
clusterer = anamoly_detection.ClusterCentroid(max_iter=20, cores=4, timeout=300)
# Perform clustering
centroids, best_n_components = clusterer.cluster_make(X)
centroids_1, best_n_components_1 = clusterer.cluster_make(X1)
centroids_2, _ = clusterer.cluster_make(Y1)
# Print the results
print(f"Best number of components: {centroids_1}")
print(f"Centroids of the clusters:\n{centroids}")

a=anamoly_detection.compare_distance(centroids_2,centroids_1)
b=anamoly_detection.compare_distance(centroids_2,centroids)
print(anamoly_detection.angle_between_group(a,b))


def delete_pycache(root_dir='.'):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            shutil.rmtree(pycache_path)
            print(f"Deleted: {pycache_path}")

delete_pycache()