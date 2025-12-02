import numpy as np
from sklearn.datasets import make_moons
from src.dbscan.dbscan import DBSCAN
from src.dbscan.visualization import plot_clusters

if __name__ == "__main__":
    # 1. Generate sample data (Moons shape)
    print("Generating data...")
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

    # 2. Run your implementation
    print("Running DBSCAN implementation...")
    # NOTE: You might need to tune eps and min_samples
    my_dbscan = DBSCAN(eps=0.25, min_samples=5)
    my_dbscan.fit(X)

    # 3. Print and Plot results
    print("Labels output:", my_dbscan.labels_)

    # Check if any cluster was found (if all are -1, logic might be incomplete)
    unique_labels = set(my_dbscan.labels_)
    print(f"Found clusters: {unique_labels}")

    plot_clusters(X, my_dbscan.labels_, title="My DBSCAN Implementation")