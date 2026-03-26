Parameters
==========

FOSC-X exposes several parameters for controlling clustering behaviour.


Core Parameters
---------------

top_M
~~~~~

Number of candidate clusterings to return (see :paramref:`FOSCX.top_M`).

FOSC-X is designed for exploring a small number of high-quality alternatives,
so values below ``10`` are typically sufficient.


kmin, kmax
~~~~~~~~~~

Define constraints on the minimum (``kmin``) and maximum (``kmax``) number of
clusters in the returned solutions.

These parameters can be used to restrict the search space when you have prior
knowledge about the expected number of clusters, or to avoid selecting large
numbers of uninformative or singleton clusters.


Quality Parameters
------------------

quality_measure
~~~~~~~~~~~~~~~

Specifies the quality measure used to evaluate candidate clusterings
(see :paramref:`FOSCX.quality_measure`).

Available options:

- ``"Stability"`` *(default)*  
  Based on cluster lifetime and persistence  

- ``"Modularity"``  
  Graph-based measure using a similarity graph; typically more suitable
  for distance-based trees  

- ``"PFCE"``  
  Graph-based measure designed for HDBSCAN hierarchies  

See the :ref:`quality_measures` section for more details.


nearest_neighbors, metric
~~~~~~~~~~~~~~~~~~~~~~~~~

Used when ``quality_measure="Modularity"`` to construct a k-nearest neighbors
similarity graph.

- ``nearest_neighbors`` must be greater than ``1``  
- ``metric`` can be any valid scikit-learn distance metric  

In most cases, the metric should match the one used to generate the clustering.

For HDBSCAN and scikit-learn hierarchies, these values are inferred automatically
when possible.


Pre-processing Parameters
-------------------------

min_cluster_size
~~~~~~~~~~~~~~~~

Controls tree condensation by collapsing clusters smaller than the specified
size into their parent cluster.

Setting ``min_cluster_size > 1`` simplifies the hierarchy and treats small
clusters as noise.


singletons_as_noise, keep_noise_quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``singletons_as_noise``  
  Whether singleton clusters are treated as noise  

- ``keep_noise_quality``  
  Whether noise nodes retain their original quality score or are assigned ``0``  