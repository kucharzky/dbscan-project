import pytest
import numpy as np
from dbscan.dbscan import DBSCAN


class TestDBSCAN:
    """Tests for DBSCAN algorithm"""

    def test_init(self):
        """Test DBSCAN class initialization"""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        assert dbscan.eps == 0.5
        assert dbscan.min_samples == 5
        assert dbscan.labels_ == []

    def test_simple_clustering(self):
        """Test basic clustering of two groups of points"""
        X = np.array([
            # points around origin
            [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3],
            [0.1, 0.1], [0.9, 0.9], [0.4, 0.6], [0.6, 0.4],
            [0.25, 0.25], [0.75, 0.75],
            #points around [10, 10]
            [10, 10], [10, 11], [11, 10], [11, 11], [10.5, 10.5],
            [10.2, 10.8], [10.8, 10.2], [10.3, 10.7], [10.7, 10.3],
            [10.1, 10.1], [10.9, 10.9], [10.4, 10.6], [10.6, 10.4],
            [10.25, 10.25], [10.75, 10.75]
        ])

        dbscan = DBSCAN(eps=2.0, min_samples=3)
        dbscan.fit(X)

        # Check if two clusters were found
        unique_labels = set(dbscan.labels_)
        assert len(unique_labels) == 2

        # Check that points from the same group have the same label
        labels_group1 = dbscan.labels_[:15]
        labels_group2 = dbscan.labels_[15:]

        # all points in group 1 should have the same label
        assert len(set(labels_group1)) == 1
        # all points in group 2 should have the same label
        assert len(set(labels_group2)) == 1
        # groups should have different labels
        assert labels_group1[0] != labels_group2[0]

    def test_noise_detection(self):
        """Test detection of noise (outliers)"""
        X = np.array([
            # 12 points
            [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.5, 1],
            [1, 0.5], [0.2, 0.2], [0.8, 0.8], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6],
            # isolated 5 points
            [10, 10], [15, 15], [20, 20], [-10, -10], [25, -5]
        ])

        dbscan = DBSCAN(eps=2.0, min_samples=3)
        dbscan.fit(X)

        # check if noise was marked as -1
        assert dbscan.labels_[12] == -1  # [10, 10]
        assert dbscan.labels_[13] == -1  # [15, 15]
        assert dbscan.labels_[14] == -1  # [20, 20]
        assert dbscan.labels_[15] == -1  # [-10, -10]
        assert dbscan.labels_[16] == -1  # [25, -5]

        # Check that cluster points have the same label
        cluster_labels = dbscan.labels_[:12]
        assert len(set(cluster_labels)) == 1
        assert cluster_labels[0] != -1

    def test_single_cluster(self):
        """Test for a single cluster"""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3],
            [0.1, 0.1], [0.9, 0.9], [0.4, 0.6], [0.6, 0.4],
            [0.25, 0.25], [0.75, 0.75], [1.2, 1.2], [1.3, 0.8]
        ])

        dbscan = DBSCAN(eps=2.0, min_samples=3)
        dbscan.fit(X)

        # All points should belong to the same cluster
        assert len(set(dbscan.labels_)) == 1
        assert -1 not in dbscan.labels_

    def test_all_noise(self):
        """Test when all points are noise"""
        X = np.array([
            [0, 0], [10, 10], [20, 20], [30, 30], [40, 40],
            [-10, -10], [-20, -20], [50, 50], [15, -15], [25, 25],
            [35, 35], [45, 45], [-15, -15], [-25, -25], [55, 55]
        ])

        dbscan = DBSCAN(eps=1.0, min_samples=4)
        dbscan.fit(X)

        # All points should be marked as noise
        assert all(label == -1 for label in dbscan.labels_)

    def test_three_clusters(self):
        """Test for three distinct clusters"""
        X = np.array([
            # Cluster 1 - 10 points around origin
            [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6],
            # Cluster 2 - 10 points around [10, 10]
            [10, 10], [10, 11], [11, 10], [11, 11], [10.5, 10.5],
            [10.2, 10.8], [10.8, 10.2], [10.3, 10.7], [10.7, 10.3], [10.4, 10.6],
            # Cluster 3 - 10 points around [20, 0]
            [20, 0], [20, 1], [21, 0], [21, 1], [20.5, 0.5],
            [20.2, 0.8], [20.8, 0.2], [20.3, 0.7], [20.7, 0.3], [20.4, 0.6]
        ])

        dbscan = DBSCAN(eps=2.0, min_samples=3)
        dbscan.fit(X)

        # Check if three clusters were found
        unique_labels = set(dbscan.labels_)
        assert len(unique_labels) == 3
        assert -1 not in unique_labels

        # Check that each group has unique label
        labels_cluster1 = dbscan.labels_[:10]
        labels_cluster2 = dbscan.labels_[10:20]
        labels_cluster3 = dbscan.labels_[20:]

        assert len(set(labels_cluster1)) == 1
        assert len(set(labels_cluster2)) == 1
        assert len(set(labels_cluster3)) == 1

        # All three should have different labels
        assert len({labels_cluster1[0], labels_cluster2[0], labels_cluster3[0]}) == 3

    def test_dense_cluster(self):
        """Test with densely packed points"""
        # Create 50 points in a tight cluster
        X = np.random.randn(50, 2) * 0.3  # Tight cluster around origin

        dbscan = DBSCAN(eps=1.0, min_samples=3)
        dbscan.fit(X)

        # All points should be in one cluster
        assert len(set(dbscan.labels_)) == 1
        assert -1 not in dbscan.labels_

    def test_fit_returns_self(self):
        """Test if fit method returns self"""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            [10, 10], [10, 11], [11, 10], [11, 11], [10.5, 10.5]
        ])

        dbscan = DBSCAN(eps=2.0, min_samples=2)
        result = dbscan.fit(X)

        assert result is dbscan