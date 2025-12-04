.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/DBSCAN.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/DBSCAN
    .. image:: https://readthedocs.org/projects/DBSCAN/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://DBSCAN.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/DBSCAN/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/DBSCAN
    .. image:: https://img.shields.io/pypi/v/DBSCAN.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/DBSCAN/
    .. image:: https://img.shields.io/conda/vn/conda-forge/DBSCAN.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/DBSCAN
    .. image:: https://pepy.tech/badge/DBSCAN/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/DBSCAN
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/DBSCAN

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

================================
DBSCAN Clustering Implementation
================================


    This project presents a custom implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm in Python


The main goal was to create a clustering tool capable of identifying clusters of arbitrary shapes (e.g., moons, circles) and handling noise (outliers), which traditional algorithms like K-Means often fail to address. The implementation is built from scratch using numpy and is compared against the industry-standard scikit-learn library to ensure correctness.


.. _pyscaffold-notes:

Features
========
* Groups points that are closely packed together (points with many nearby neighbors).
* Automatically detects and labels outliers as noise (label -1).
* Customizable Parameters:

  * eps (Epsilon): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
  * min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

* Includes a module to plot clustering results with distinct colors for clusters and noise

* and here the parent list continues
Installation & Setup
====================
This project uses PyScaffold for structure and VirtualEnv for dependency management.

1. Clone the repo:
::

   git clone https://github.com/kucharzky/dbscan-project.git
   cd dbscan-project

2. Create and activate a virtual environment:
::

    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate

3. Install dependencies:
::

    pip install -e .

How to Run
==========
You can run the main script to see the algorithm in action on sample datasets (Moons and Circles).
::

    python run.py

This will execute the clustering on generated datasets and display a matplotlib window with the results.

**Example Usage in Python**
::

    from dbscan.dbscan import DBSCAN
    import numpy as np

    # 1. Prepare data
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

    # 2. Initialize and fit
    model = DBSCAN(eps=3, min_samples=2)
    model.fit(X)

    # 3. Get results
    print(model.labels_)
    # Output: [ 0  0  0  1  1 -1]

**Testing & Validation**
We use pytest to verify the logic and compare our implementation with scikit-learn.

To run the full test suite:
::

    pytest

Authors
=======
* Developer: Maciej Kucharski - Algorithm implementation and visualization.

* Tester: Adrian Homa - Unit tests, benchmarks, and validation reports.
