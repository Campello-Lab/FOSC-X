Basic Usage
===========

FOSC-X (Framework for Optimal Extraction of Clusters) is designed to extract
high-quality flat clusterings from a hierarchical clustering tree.

Rather than producing a single clustering, FOSC-X explores multiple possible
solutions under optional cluster-count constraints. It returns the ``top_M``
globally optimal solutions within these constraints, ranked from best to worst.

FOSC-X is designed to work with a range of common hierarchical clustering outputs,
including:

- HDBSCAN 
- scikit-learn (AgglomerativeClustering)
- SciPy linkage matrices
- Condensed Tree/Non-Binary tree matrices
- Pre-Computed Trees (JSON) (See :ref:`JSON Tree format <json_tree>`)

Using these formats is intentionally simple, just pass in the object or model you already have.
For more details, see :ref:`tree_formats`.

FOSC-X follows a familiar scikit-learn style API, so if you have used sklearn
estimators before, the workflow should feel natural.

Example
-------

Below is a minimal example using a SciPy-style hierarchical clustering.

We begin by generating some sample data and constructing a hierarchical tree:

.. code-block:: python

    import numpy as np
    from scipy.cluster.hierarchy import linkage
    from sklearn.datasets import make_blobs

    from foscx import FOSCX

    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Build hierarchical clustering (SciPy linkage)
    Z = linkage(X, method="ward")

Once we have a hierarchy, we can use FOSC-X to explore possible flat clusterings:

.. code-block:: python

    # Initialize FOSC-X
    model = FOSCX(top_M=5, kmin=2, kmax=None)

    # Fit model to hierarchy
    model.fit(Z)

Candidate Clusterings
---------------------

FOSC-X does not return a single clustering, but instead a set of candidate
solutions ranked by quality:

.. code-block:: python

    model.candidates_

A typical output might look like:

.. code-block:: text

    quality     n_clusters    selected_nodes
    28309.20    3             [301, 305, 306]
    25434.40    4             [301, 305, 313, 314]
    22298.88    6             [301, 305, 313, 329, 360, 361]
    22286.93    7             [301, 305, 354, 355, 356, 357, 314]
    22256.62    7             [301, 305, 313, 358, 359, 360, 361]

Each row corresponds to a different clustering solution:

- ``quality``: Objective value of the clustering  
- ``n_clusters``: Number of clusters  
- ``selected_nodes``: Internal representation of clusters  

This allows you to investigate multiple plausible clusterings rather than only 
obtaining a single result.


Extracting Labels
-----------------

To convert one of these candidate solutions into a flat partition:

.. code-block:: python

    labels = model.get_labels(candidate_index=0)

.. code-block:: text

    array([3, 3, 3, ..., 3, 3, 3])

Here, ``candidate_index`` selects which solution to use (from ``0`` to ``top_M - 1``).

The result is a NumPy array of cluster labels for each data point, which can be
used directly for analysis or visualization. Noise points are labeled as ``0``.

Cluster Count Constraints
-------------------------

FOSC-X can optionally restrict the range of clusterings considered using
``kmin`` and ``kmax``, which define the minimum and maximum number of clusters.

For example, setting ``kmin=2`` ensures that all returned solutions contain
at least two clusters. Leaving ``kmax=None`` allows any larger number of clusters.

These constraints are useful when you have prior knowledge about the expected
number of clusters, or want to limit the search space.


Refining Clusterings
--------------------

Once the model has been fitted, alternative clusterings can be explored
without reprocessing the hierarchy by calling the ``predict`` method.

This allows you to adjust parameters such as ``top_M``, ``kmin``, and ``kmax``
and quickly obtain new candidate solutions:

.. code-block:: python

    model.predict(top_M=5, kmin=5, kmax=10)



Quality Measures
----------------

FOSC-X supports multiple quality measures for evaluating clusterings, including
``Stability``, ``Modularity``, and ``PFCE``. Stability is the default quality measure.

The quality measure can be selected when initializing the model:

.. code-block:: python

    model = FOSCX(quality_measure="Stability")

Different measures may lead to different preferred clusterings depending on
the structure of the data. See the Quality Measures section for a more detailed
comparison.

Modularity Q
~~~~~~~~~~

The Modularity Q measure evaluates clusterings based on a similarity graph
constructed from the original data. As such, it requires access to the raw data,
unless this is already available from the input clustering (for example, with
HDBSCAN objects).

In addition, the number of nearest neighbors and a distance metric must be
specified. The metric should typically match the one used to generate the
hierarchical clustering.

.. code-block:: python

    model = FOSCX(
        quality_measure="modularity",
        nearest_neighbors=5,
        metric="euclidean"
    )
    model.fit(Z, y=X)

Alternatively, a precomputed similarity graph may be used by setting
``metric="precomputed"``, and supplied instead of the data X.

For HDBSCAN objects, both the nearest neighbors and metric are inferred
automatically from the clustering.

PFCE
~~~~

The PFCE measure is only available when using HDBSCAN objects generated with
the `hdbscan <https://hdbscan.readthedocs.io/>`_ package and requires no
additional inputs.