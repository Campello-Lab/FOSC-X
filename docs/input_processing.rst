Tree Processing and Noise
=========================

This section describes how preprocessing the cluster tree affects the behaviour
of FOSC-X.

Tree Condensation / Simplification
----------------------------------

FOSC-X supports optional preprocessing of the cluster tree via a minimum cluster size
parameter (``min_cluster_size > 1``). This process is commonly referred to as
*tree condensation* or *simplification*.

Condensation collapses clusters smaller than the specified size into their parent,
effectively removing low-support branches from the hierarchy. While this can be
applied to any tree, it has several important consequences.

Noise
~~~~~

When condensation is applied, clusters below the minimum size are treated as noise.

This means that:

- Not all observations are necessarily assigned to a cluster  
- The resulting solution may not form a complete partition  
- Some branches of the tree no longer require a cluster to be selected  

From the perspective of the optimisation, this relaxes the structural constraints.
In particular, subtrees consisting entirely of noise do not need to contribute a
cluster to the final solution, increasing the flexibility of the dynamic program.

This behaviour is consistent with the formulation described in the paper, where
noise reduces the lower bound on the number of clusters required in a solution.

Stability
~~~~~~~~~

Condensation has a direct impact on Stability-based quality measures.

Stability evaluates how long a cluster persists within the hierarchy. In unprocessed
trees, small or short-lived splits can artificially reduce the measured Stability
of an otherwise coherent cluster. For example, a cluster that briefly splits into a
large child and a very small child will have its lifetime interrupted, lowering its
Stability score.

Condensation removes these small, low-support splits, making clusters more
continuous in the tree. As a result:

- Cluster lifetimes increase  
- Stability scores become more robust  
- The optimisation is less sensitive to minor structural fluctuations  

As a result, condensation is commonly used to improve the robustness of
Stability-based extraction.

Other Quality Measures
~~~~~~~~~~~~~~~~~~~~~~

Condensation primarily affects measures derived from the tree structure, such as
Stability.

Graph-based measures such as ``Modularity`` and ``PFCE`` are computed from the
underlying data representation rather than the hierarchy itself. As a result, they
are largely unaffected by condensation, aside from changes to the set of candidate
clusters considered during optimisation.

Additionally, condensation may provide some computational benefit for
``Modularity`` and ``PFCE``, as it reduces the number of nodes for which these
measures are evaluated. However, the nodes removed are typically the smallest
clusters, which are also the least expensive to evaluate.

In practice:

- Use condensation when working with Stability-based extraction  
- It is less critical when using graph-based quality measures  


Noise Processing
----------------

Noise may be introduced into a hierarchy in several ways:

- **Condensation**: clusters below ``min_cluster_size`` are removed and their
  leaves are treated as noise  
- **``singletons_as_noise``**: singleton clusters are treated as noise without
  modifying the tree structure  
- **Pre-specified hierarchy**: noise labels provided directly in a JSON tree  
- **Post-fit assignment**: noise defined manually via
  ``model.cluster_tree_.is_noise``  

The presence of noise affects both the feasible solution space and how cluster
quality is evaluated.

Noise and Feasibility
~~~~~~~~~~~~~~~~~~~~~

When noise is present, subtrees consisting entirely of noise do not need to
contribute a cluster to the final solution. This relaxes the structural constraints
of the optimisation, as not every branch is required to produce a cluster.

In practice, this means:

- Fewer clusters may be required to form a valid solution  
- The algorithm has greater flexibility in selecting high-quality clusters  
- The lower bound on the number of clusters is effectively reduced  

Noise Quality
~~~~~~~~~~~~~

The treatment of noise in the objective function is controlled by the
``keep_noise_quality`` parameter.

- ``keep_noise_quality = True`` (default): noise retains its assigned quality  
- ``keep_noise_quality = False``: noise quality is set to zero  

The effect of this depends on the chosen quality measure:

- **Stability**: noise always has zero quality  
- **PFCE**: noise always has zero quality  
- **Modularity**: noise may receive small (often negative) values  
- **Precomputed (JSON)**: noise may be assigned arbitrary quality values  

