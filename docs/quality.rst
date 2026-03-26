.. _quality_measures:

Quality Measures
================

FOSC-X selects clusterings by optimising a *quality measure* defined over the
cluster tree. Different measures capture different notions of what constitutes
a "good" clustering.

FOSC-X currently supports three quality measures:

- ``Stability / Excess of Mass`` (default)
- ``Modularity``
- ``PFCE``

Each measure evaluates clusters independently and assigns a score, which is then
summed across the selected clusters.


Overview
--------

The three quality measures differ primarily in how they evaluate clusters:

- ``Stability`` is derived from the structure of the cluster tree  
- ``Modularity`` is computed from a similarity graph constructed from the data  
- ``PFCE`` is computed from the minimum spanning tree (MST) associated with the cluster tree  

As a result, they may produce different clusterings when the tree structure and
data geometry do not fully align.


Stability
---------

Stability is derived directly from thestructure of the cluster tree.
In density-based hierarchies (e.g. HDBSCAN), Exess of Mass (EOM) corresponds to a density-based
formulation of stability, where cluster persistence is measured across density
levels.

Stability measures how long a cluster persists across the hierarchy. Clusters
that exist over a wide range of resolutions receive higher scores.

Key properties:

- Uses only the tree structure (no raw data required)  
- Computed from cluster lifetimes in the hierarchy  
- Computationally efficient and lightweight  
- Sensitive to structural changes in the tree  

Stability can be influenced by small or short-lived splits in the hierarchy.
Applying tree condensation (``min_cluster_size``) can help remove low-support
branches and produce more continuous cluster lifetimes.


Modularity Q
----------

Modularity Q is a graph-based quality measure that evaluates how well-connected
clusters are relative to a baseline.

It constructs a similarity graph from the data and scores clusters based on
their internal connectivity. Compared to tree-based measures, this requires constructing a k-nearest
neighbor graph, which can be computationally expensive for large datasets.

Key properties:

- Uses the original data (requires ``y`` in ``fit``)  
- Constructs a k-nearest neighbor (k-NN) similarity graph  
- Depends on the choice of ``nearest_neighbors`` and distance metric ``metric``  
- Computed independently of the hierarchical tree structure  

Example usage:

.. code-block:: python

    model = FOSCX(
        quality_measure="Modularity",
        nearest_neighbors=10,
        metric="euclidean"
    )
    model.fit(Z, y=X)

Notes:

- ``nearest_neighbors`` controls graph construction. Defaults to
  ``min_cluster_size`` when available (e.g. HDBSCAN)  
- ``metric`` should match the distance used to build the hierarchy.
  Defaults to the hierarchy metric when available (e.g. HDBSCAN,
  ``sklearn.AgglomerativeClustering``)  
- May assign small (or negative) scores to weakly connected clusters and noise 

PFCE
----

PFCE (Partition-Free Cluster Evaluation) is a graph-based quality measure defined
on the minimum spanning tree (MST) associated with the hierarchy.

It evaluates clusters using connectivity information derived from the MST,
capturing aspects of cluster cohesion and separation.

Key properties:

- Only available for HDBSCAN-based hierarchies  
- Uses the minimum spanning tree  
- Computed independently of the hierarchical tree structure  
- Incorporates notions of sparseness, separation, and dividedness  

Example usage:

.. code-block:: python

    model = FOSCX(quality_measure="PFCE")
    model.fit(hdbscan_clusterer)

Notes:

- Requires an HDBSCAN object with MST available
  (e.g. ``hdbscan.HDBSCAN(gen_min_span_tree=True)``)  
- Not supported for sklearn HDBSCAN  


Examples
--------

The following examples illustrate how different quality measures can produce
different clusterings depending on the structure of the data.

Example 1: Well-separated clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/example1.png

A dataset consisting of well-separated Gaussian clusters.

In this setting, the hierarchical structure closely matches the underlying data
distribution. As a result, all quality measures tend to produce similar
clusterings.



Example 2: Non-Convex Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/example2.png

A dataset consisting of two non-convex (curved) clusters, along with a smaller
dense cluster embedded within one region and a small amount of noise.

This structure is clearly defined, but is not well represented by simple convex
cluster boundaries. As a result, the hierarchy may introduce additional splits
within the curved regions.

In this example:

- Stability and Modularity produce similar clusterings, with some fragmentation
  of the non-convex structures  
- PFCE identifies the full curved structures without fragmentation  


Example 3: Density-based structure with noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/example3.png

A dataset with varying density, non-convex clusters, and noise points.

In this case, the hierarchical structure reflects density variations and
irregular cluster shapes. Noise and low-density regions introduce additional
flexibility, and different measures may emphasise different aspects of the
structure.

In this example:

- Stability identifies most of the underlying cluster structure, but fragments
  some clusters and selects a small number of noise points as clusters.
- Modularity identifies the cluster structure with minimal fragmentation, 
  but also selects some noise as clusters. 
- PFCE fails to identify any clustering structure.

