.. _parameters: 

Parameters
==========

FOSC-X exposes several parameters for controlling clustering behaviour,
including constraints on the solution space, quality evaluation, and
pre-processing of the hierarchy.


Core Parameters
---------------

top_M
~~~~~

Maximum number of candidate clusterings to return.

FOSC-X returns the ``top_M`` highest-quality solutions ranked from best to worst.
Increasing this allows exploration of alternative clusterings at modest additional cost.

Default is ``1``. In practice, values below ``10`` are typically sufficient.


kmin, kmax
~~~~~~~~~~

Constraints on the minimum (``kmin``) and maximum (``kmax``) number of clusters
in the returned solutions.

These constraints are enforced during optimisation (not post hoc), and can be used
to restrict the search space when prior knowledge about the expected number of
clusters is available.

- ``kmin``: minimum number of clusters (default: ``None``)  
- ``kmax``: maximum number of clusters (default: ``None``)  


Quality Parameters
------------------

quality_measure
~~~~~~~~~~~~~~~

Quality measure used to evaluate candidate clusterings.

Available options:

- ``"stability"`` *(default)*  
  Based on cluster lifetime and persistence  

- ``"modularity"``  
  Graph-based measure using a k-nearest neighbor similarity graph  

- ``"PFCE"``  
  Graph-based measure designed for HDBSCAN hierarchies  

- ``"B Cubed (B³)"``
  Label based semi-supervised measure

- ``"Constraint Satisfaction (Constraints)"``
  Constraint based semi-supervised measure

See :ref:`quality_measures` for further details.

.. note::

   ``"modularity"`` requires ``nearest_neighbors`` and ``metric``, unless these
   can be inferred from the input hierarchy. 

   ``"PFCE"`` requires an HDBSCAN object.

   This parameter is ignored for JSON trees with pre-computed quality values.



nearest_neighbors, metric
~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters used when ``quality_measure="modularity"`` to construct a
k-nearest neighbor (k-NN) similarity graph.

- ``nearest_neighbors``  
  Number of neighbors used to build the graph. Must be greater than ``1``.

- ``metric``  
  Distance metric used to compute nearest neighbors. Supported values are
  those accepted by ``sklearn.neighbors.NearestNeighbors``.

  - If ``metric="precomputed"``, the input ``y`` is interpreted as a
    **precomputed distance matrix**, and nearest neighbors are computed
    directly from this matrix.

  - If ``metric="precomputed_similarity"``, the input ``y`` is interpreted as a
    **precomputed similarity graph**, which is used directly without
    constructing a k-NN graph.

If not provided, these parameters are inferred automatically when possible
(e.g. from HDBSCAN or scikit-learn estimators).

It is generally recommended that ``metric`` matches the one used to construct
the hierarchy.

Hierarchy Interpretation
------------------------

density
~~~~~~~

Controls how distance values in the hierarchy are interpreted.

- ``False`` (default): values represent merge distances  
- ``True``: values represent density levels (e.g. HDBSCAN λ values)

This is automatically set when using density-based clustering objects (such as HDBSCAN).
Manual specification is primarily required for SciPy linkage matrices.


Pre-processing Parameters
-------------------------

min_cluster_size
~~~~~~~~~~~~~~~~

Minimum cluster size used for tree condensation.

Clusters smaller than this threshold are removed from the hierarchy, and their
descendant leaves are treated as noise. This simplifies the tree and can improve
stability-based extraction.

Default is ``None`` (no condensation).

.. note::

   This parameter is not available when using JSON trees with pre-computed
   quality values.


singletons_as_noise
~~~~~~~~~~~~~~~~~~~

Whether singleton clusters (clusters of size 1) are treated as noise.

Unlike ``min_cluster_size``, this does not modify the tree structure, but affects
how leaf nodes are interpreted during optimisation.

Default is ``False``.

.. note::

   For JSON trees, this requires either ``complete_tree=True`` or that
   ``cluster_size`` is provided.


keep_noise_quality
~~~~~~~~~~~~~~~~~~

Controls how noise nodes contribute to the objective function.

- If ``True`` (default): noise retains its assigned quality  
- If ``False``: noise nodes are assigned a quality of ``0``  

This primarily affects graph-based measures, where noise may otherwise receive
small or negative scores.