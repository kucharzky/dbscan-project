import time
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN as SklearnDBSCAN

# Import your custom implementation
from dbscan.dbscan import DBSCAN as MyDBSCAN


def run_performance_benchmark(n_samples=300, eps=0.25, min_samples=5):
    """
    Runs a detailed performance benchmark comparing custom DBSCAN vs Scikit-Learn.
    Prints results to console for direct reporting.
    """

    # --- CONFIGURATION ---
    RANDOM_STATE = 42

    print("\n--- DBSCAN PERFORMANCE BENCHMARK ---")
    print(f"Dataset Size: N={n_samples}")
    print(f"Parameters: Epsilon={eps}, Min Samples={min_samples}")

    # 1. GENERATE DATA
    X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=RANDOM_STATE)

    # 2. BENCHMARK SCiKIT-LEARN (Reference Baseline)
    sk_model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    start_sk = time.perf_counter()
    sk_model.fit(X)
    duration_sklearn = time.perf_counter() - start_sk

    # 3. BENCHMARK CUSTOM IMPLEMENTATION
    my_model = MyDBSCAN(eps=eps, min_samples=min_samples)
    start_my = time.perf_counter()
    my_model.fit(X)
    duration_my = time.perf_counter() - start_my

    # --- CALCULATIONS ---
    if duration_sklearn == 0:
        slowness_factor = duration_my / 0.000001
    else:
        slowness_factor = duration_my / duration_sklearn

    # 4. REPORT
    print("\n-------------------------------------------")
    print("| Implementation | Time (s) | Slowness Factor |")
    print("|----------------|----------|-----------------|")
    print(f"| Scikit-Learn   | {duration_sklearn:<8.4f} | {'1.00x':<15} |")
    print(f"| My DBSCAN      | {duration_my:<8.4f} | {slowness_factor:<13.2f}x |")
    print("-------------------------------------------")

    if slowness_factor > 100:
        print("\nConclusion: The custom implementation is over 100x slower.")
        print("This confirms the theoretical issue: using nested Python loops (O(N^2))")
        print("instead of C-optimized, vectorized NumPy/spatial indexing is highly inefficient.")
    else:
        print("\nConclusion: Performance difference is acceptable, but custom code is still slower.")


if __name__ == "__main__":
    # You can change N_SAMPLES here to N=500 or N=1000 to see the huge difference
    run_performance_benchmark(n_samples=300)