import numpy as np


def euclidean_distance(point_a, point_b):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point_a (np.array): Coordinates of the first point.
        point_b (np.array): Coordinates of the second point.

    Returns:
        float: The distance between points.
    """
    # implementation of Euclidean distance (sqrt of sum of squared differences)
    return np.sqrt(np.sum((point_a - point_b) ** 2))