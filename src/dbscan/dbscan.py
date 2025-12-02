import numpy as np
from .utils import euclidean_distance


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.
    """

    def __init__(self, eps, min_samples):
        """
        Args:
            eps (float): The maximum distance between two samples for one to be considered
                         as in the neighborhood of the other.
            min_samples (int): The number of samples (or total weight) in a neighborhood
                               for a point to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = []  # Store cluster labels here (-1 means Noise)

    def fit(self, X):
        """
        Perform DBSCAN clustering from vector array.

        Args:
            X (np.array): Input data (points).
        Returns:
            self
        """
        n_samples = len(X)
        # Initialize all labels to -1 (Noise) by default
        self.labels_ = np.full(n_samples, -1)

        cluster_id = 0
        visited = np.full(n_samples, False)

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True

            # TODO: Get neighbors of the current point X[i]
            # neighbors = self._get_neighbors(X, i)

            # TODO: Check if the number of neighbors is less than self.min_samples
            # If yes: It is Noise (label remains -1), continue to next point.
            # If no: It is a Core Point.
            #        1. Increment cluster_id (or start from 0)
            #        2. Call self._expand_cluster(...)

        return self

    def _get_neighbors(self, X, point_idx):
        """
        Finds indices of neighbors within epsilon distance.

        Args:
            X (np.array): All data points.
            point_idx (int): Index of the point to find neighbors for.
        Returns:
            list: Indices of neighbors.
        """
        neighbors = []
        # TODO: Iterate over all points in X.
        # Calculate distance between X[point_idx] and every other point.
        # If distance <= self.eps, add index to neighbors.
        return neighbors

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        """
        Recursively expands the cluster from a starting core point.

        Args:
            X (np.array): All data points.
            point_idx (int): Index of the seed point.
            neighbors (list): Neighbors of the seed point.
            cluster_id (int): ID of the current cluster.
            visited (np.array): Boolean array tracking visited points.
        """
        # Assign the seed point to the current cluster
        self.labels_[point_idx] = cluster_id

        # TODO: Implement the expansion logic (Queue approach is recommended over recursion)
        # 1. Create a queue/list containing initial neighbors.
        # 2. While the queue is not empty:
        #    a. Pop a point index from the queue.
        #    b. If it was not visited, mark as visited.
        #    c. Get its neighbors.
        #    d. If it has enough neighbors (>= min_samples), add them to the queue.
        #    e. If the point does not belong to any cluster yet, assign it to cluster_id.
        pass