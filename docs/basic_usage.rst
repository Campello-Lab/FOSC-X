Basic Usage
===========

FOSC-X is designed to extract multiple high-quality flat clusterings from hierarchical
clustering trees.

Rather than producing a single clustering, FOSC-X explores a range of candidate
solutions and returns the ``top_M`` best ones, ranked from best to worst.
Optional constraints on the number of clusters (``kmin`` and ``kmax``) allow you
to restrict the search to solutions of interest, focusing on a meaningful range
of cluster counts.

FOSC-X performs **local, non-horizontal cuts** of the hierarchy. This
allows different branches of the tree to be cut at different levels, producing
clusterings that better reflect the underlying structure of the data.

The result is a set of high-quality, diverse clusterings that provide multiple
plausible interpretations of the data, rather than a single fixed solution.

FOSC-X follows a familiar scikit-learn style API, so if you have used sklearn
estimators before, the workflow should feel natural.

Supported Input Formats
-----------------------

FOSC-X is designed to work with a range of common hierarchical clustering outputs,
including:

- HDBSCAN*  
- scikit-learn (``AgglomerativeClustering`` and ``HDBSCAN*``)  
- SciPy linkage matrices  
- Condensed / non-binary tree matrices  
- Pre-computed trees (JSON format) (see :ref:`json_tree`)  

Using these formats is intentionally simple—just pass in the object or model you
already have. For more details, see :ref:`tree_formats`.


Example
-------

Below is a minimal example using a SciPy-style hierarchical clustering.

We begin by generating some sample data and constructing a hierarchy:

.. code-block:: python

    import numpy as np
    from scipy.cluster.hierarchy import linkage
    from sklearn.datasets import make_blobs

    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Build hierarchical clustering
    Z = linkage(X, method="ward")

Once we have a hierarchy, we can use FOSC-X to extract candidate clusterings:

.. code-block:: python

    from foscx import FOSCX

    model = FOSCX(top_M=5, kmin=2, kmax=None, min_cluster_size=None, quality_measure='stability',
    singletons_as_noise=False, keep_noise_quality=None, nearest_neighbors=None, metric=None, 
    density=False, tie_quality='stability', verbose=False)

    model.fit(Z)


Core Parameters
---------------

FOSC-X is primarily controlled by three parameters:

- ``top_M``  
  Number of candidate clusterings to return  

- ``kmin``  
  Minimum number of clusters allowed  

- ``kmax``  
  Maximum number of clusters allowed  

- ``quality_measure``  
  Objective function used to evaluate clusterings (see :ref:`quality_measures`)

Setting ``kmin=None`` and ``kmax=None`` performs an unconstrained search.

For more details, see :ref:`parameters`.


Candidate Clusterings
---------------------

FOSC-X returns a set of candidate solutions ranked by quality:

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

Each row corresponds to a different clustering:

- ``quality``: objective value of the clustering  
- ``n_clusters``: number of clusters  
- ``selected_nodes``: nodes selected in the hierarchy  


The FOSC-X API also provides several methods for visualising and interpreting these candidate solutions, as described in :ref:`visualization`.

Extracting Labels
-----------------

To convert a candidate solution into a flat clustering:

.. code-block:: python

    labels = model.get_labels(candidate_index=0)

The result is a NumPy array of cluster labels for each data point. Noise observations
are assigned the label ``-1``.


Refining Clusterings
--------------------

Once a hierarchy has been processed, alternative solutions can be explored
without recomputing everything by calling :meth:`FOSCX.predict`:

.. code-block:: python

    model.predict(top_M=5, kmin=5, kmax=10)

This allows rapid exploration of different constraints and solution sets.


Pre-processing and Noise
------------------------

FOSC-X includes several pre-processing steps that can affect the quality evaluation and solution space, including:
- **Condensation**: removes small clusters and treats their leaves as noise
- **Noise assignment**: allows manual specification of noise observations

These steps can improve the robustness of Stability-based extraction and provide more control over the solution space. See :ref:`preprocessing` for details.