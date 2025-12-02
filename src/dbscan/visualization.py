import matplotlib.pyplot as plt
import numpy as np


def plot_clusters(X, labels, title="DBSCAN Clustering"):
    """
    Plots the data points colored by their cluster label.
    Noise points (label -1) are colored black.

    Args:
        X (np.array): Data points (2D).
        labels (list/array): Cluster labels for each point.
        title (str): Title of the plot.
    """
    # Get unique labels (e.g., -1, 0, 1, 2)
    unique_labels = set(labels)

    # Generate colors map
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(10, 6))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10, label=f'Cluster {k}' if k != -1 else 'Noise')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()