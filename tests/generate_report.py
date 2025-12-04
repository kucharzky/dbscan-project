import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score


from dbscan.dbscan import DBSCAN as MyDBSCAN


def plot_on_axis(X, labels, ax, title):
    """
    Helper function to plot clusters on a specific Matplotlib axis.
    """
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]

        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=8)

    ax.set_title(title)
    ax.grid(True)


def generate_comparison_report(dataset_name='moons'):
    """
    Generates a side-by-side comparison report.
    """
    print(f"--- Generating Report for {dataset_name.upper()} dataset ---")

    if dataset_name == 'moons':
        X, _ = make_moons(n_samples=400, noise=0.1, random_state=42)
        eps, min_samples = 0.2, 5
    elif dataset_name == 'circles':
        X, _ = make_circles(n_samples=400, factor=0.5, noise=0.05, random_state=42)
        eps, min_samples = 0.18, 4
    elif dataset_name == 'blobs':
        X, _ = make_blobs(n_samples=400, random_state=42)
        eps, min_samples = 1.0, 5
    else:
        raise ValueError("Unknown dataset name")

    print(f"Parameters: eps={eps}, min_samples={min_samples}")

    print("Running Custom Implementation...")
    my_model = MyDBSCAN(eps=eps, min_samples=min_samples)
    my_model.fit(X)
    my_labels = my_model.labels_

    print("Running Scikit-Learn Reference...")
    sk_model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    sk_labels = sk_model.fit_predict(X)

    ari_score = adjusted_rand_score(my_labels, sk_labels)
    print(f"Adjusted Rand Index (Similarity): {ari_score:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_on_axis(X, my_labels, ax1, "My DBSCAN Implementation")

    plot_on_axis(X, sk_labels, ax2, "Scikit-Learn Implementation (Reference)")

    plt.suptitle(f"Comparison Report: {dataset_name.capitalize()}\nSimilarity Score (ARI): {ari_score:.4f}",
                 fontsize=16)

    filename = f"report_{dataset_name}.png"
    plt.savefig(filename)
    print(f"Report saved to: {filename}")
    # plt.show() # Optional: Uncomment if you want to see it pop up


if __name__ == "__main__":
    generate_comparison_report('moons')
    generate_comparison_report('circles')