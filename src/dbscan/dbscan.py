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

            # TODO: Check if the number of neighbors is less than self.min_samples
            # If yes: It is Noise (label remains -1), continue to next point.
            # If no: It is a Core Point.
            #        1. Increment cluster_id (or start from 0)
            #        2. Call self._expand_cluster(...)

            if len(neighbors) < self.min_samples:
                # If not enough neighbors, label as Noise (-1).
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
        # TODO: Iterate over all points in X.
        # Calculate distance between X[point_idx] and every other point.
        # If distance <= self.eps, add index to neighbors.
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

        # We use a while loop to iterate through neighbors.
        # Since we might append to 'neighbors' list, we act like a queue.
        i = 0
        while i < len(neighbors):
            neighbor_point_idx = neighbors[i]

            # If the neighbor has not been visited yet
            if not visited[neighbor_point_idx]:
                visited[neighbor_point_idx] = True

                # Find neighbors of this neighbor
                new_neighbors = self._get_neighbors(X, neighbor_point_idx)

                # If this neighbor is also a Core Point, add its neighbors to the queue
                if len(new_neighbors) >= self.min_samples:
                    neighbors = neighbors + new_neighbors

            # If the point was previously labeled as Noise (-1) or not labeled yet
            if self.labels_[neighbor_point_idx] == -1:
                self.labels_[neighbor_point_idx] = cluster_id

            # Move to the next point in the queue
            i += 1
        # pass