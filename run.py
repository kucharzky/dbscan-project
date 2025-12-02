
from sklearn.datasets import make_moons, make_circles

from src.dbscan.dbscan import DBSCAN
from src.dbscan.visualization import plot_clusters

def run_experiment(x, eps, min_samples, title):
    """
    Helper function to run DBSCAN on a given dataset and plot results.
    """
    print(f"\n--- Running Experiment: {title} ---")
    print(f"Parameters: eps={eps}, min_samples={min_samples}")

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(x)

    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    n_noise = list(model.labels_).count(-1)

    print(f"Found clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    plot_clusters(x, model.labels_, title=f"{title} (eps={eps}, min={min_samples})")

if __name__ == "__main__":
    X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    run_experiment(X_moons, eps=0.2, min_samples=5, title="Moons Dataset")

    # factor of 0.5 means the inside circle will be half the size as the outside one
    X_circles, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

    run_experiment(X_circles, eps=0.18, min_samples=4, title="Circles Dataset")