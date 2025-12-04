import pytest
import numpy as np
import time
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score

from dbscan.dbscan import DBSCAN as MyDBSCAN
from dbscan.utils import euclidean_distance

def test_euclidean_distance():
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    assert euclidean_distance(p1, p2) == 5.0

def test_compare_with_sklearn_blobs():
    x, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    eps, min_samples = 1.0, 5

    my_model = MyDBSCAN(eps=eps, min_samples=min_samples).fit(x)
    sklearn_model = SklearnDBSCAN(eps=eps, min_samples=min_samples).fit(x)

    score = adjusted_rand_score(my_model.labels_, sklearn_model.labels_)
    assert score == 1.0, f"Result mismatch! ARI: {score}"

def test_compare_with_sklearn_moons():
    x, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    eps, min_samples = 0.2, 5

    my_model = MyDBSCAN(eps=eps, min_samples=min_samples).fit(x)
    sklearn_model = SklearnDBSCAN(eps=eps, min_samples=min_samples).fit(x)

    score = adjusted_rand_score(my_model.labels_, sklearn_model.labels_)
    assert score > 0.99, f"Result mismatch! ARI: {score}"