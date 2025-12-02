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
        self.labels_ = []

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

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # if not enough neighbors, label as Noise (-1).
                # Note: It might be revisited later and included in a cluster
                # if it's a border point of another cluster.
                self.labels_[i] = -1
            else:
                # Found a core point -> Start a new cluster
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

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

        for i, point in enumerate(X):
            # Calculate distance using our utility function
            dist = euclidean_distance(X[point_idx], point)

            if dist <= self.eps:
                neighbors.append(i)
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
        # assign the seed point to the current cluster
        self.labels_[point_idx] = cluster_id

        # we use a while loop to iterate through neighbors.
        # since we might append to 'neighbors' list we act like a queue.
        i = 0
        while i < len(neighbors):
            neighbor_point_idx = neighbors[i]

            if not visited[neighbor_point_idx]:
                visited[neighbor_point_idx] = True

                # find neighbors of this neighbor
                new_neighbors = self._get_neighbors(X, neighbor_point_idx)

                # if this neighbor is also a core point add its neighbors to the queue
                if len(new_neighbors) >= self.min_samples:
                    neighbors = neighbors + new_neighbors

            # if the point was previously labeled as Noise (-1) or not labeled yet
            if self.labels_[neighbor_point_idx] == -1:
                self.labels_[neighbor_point_idx] = cluster_id

            # move to the next point in the queue
            i += 1