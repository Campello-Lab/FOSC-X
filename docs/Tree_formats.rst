.. _tree_formats:

Tree Formats
============

FOSC-X accepts a range of hierarchical clustering formats as input. The
input format is automatically detected and converted into a unified
internal representation.

Supported formats include:

- HDBSCAN clustering objects  
- scikit-learn clustering objects  
- SciPy linkage matrices  
- Condensed (non-binary) tree matrices  
- Pre-computed trees (JSON format)  


HDBSCAN
--------

FOSC-X directly accepts clustering objects produced by the
`hdbscan <https://hdbscan.readthedocs.io/>`_ package, as well as compatible
implementations such as
`fast_hdbscan <https://github.com/TutteInstitute/fast_hdbscan/>`_.

These objects provide a condensed tree representation and, optionally,
the minimum spanning tree (MST), enabling all quality measures
(including ``PFCE``) to be used.

.. code-block:: python

    import hdbscan

    # Build hierarchical clustering
    Z = hdbscan.HDBSCAN().fit(X)

    # Initialize FOSC-X
    model = FOSCX(top_M=5, kmin=2, kmax=None)

    # Extract clusterings
    model.fit(Z)


scikit-learn HDBSCAN
~~~~~~~~~~~~~~~~~~~~

FOSC-X also supports the HDBSCAN implementation available in
`scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html>`_
(``sklearn.cluster.HDBSCAN``).

The main limitation is that the minimum spanning tree is not available,
so the ``PFCE`` quality measure cannot be computed. All other functionality
remains unchanged.



scikit-learn AgglomerativeClustering
------------------------------------

FOSC-X accepts ``sklearn.cluster.AgglomerativeClustering`` objects from
`scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_.

The model must be configured with:

- ``distance_threshold=0``  
- ``n_clusters=None``  

This ensures that the full hierarchy is constructed and available for
processing.

.. code-block:: python

    from sklearn.cluster import AgglomerativeClustering

    # Build hierarchical clustering
    Z = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        linkage="ward"
    )
    Z.fit(X)

    # Initialize FOSC-X
    model = FOSCX(top_M=5, kmin=2, kmax=None)

    # Extract clusterings
    model.fit(Z)


SciPy Linkage Matrices
----------------------

SciPy linkage matrices are supported and follow the standard format:

``[left_child, right_child, distance, size]``

A valid linkage matrix can be produced using
`SciPy hierarchical clustering <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`_
or constructed manually. It must satisfy:

.. code-block:: python

    from scipy.cluster.hierarchy import is_valid_linkage

    is_valid_linkage(Z)

Interpretation of the ``distance`` column depends on the type of hierarchy:

- **Distance-based trees**: value represents the merge distance (A property of the parent)
- **Density-based trees**: value represents the level at which clusters split from their parent cluster (A property of the children)

This behaviour is controlled via the ``density`` parameter.

.. code-block:: python

    from scipy.cluster.hierarchy import linkage

    # Build hierarchical clustering
    Z = linkage(X, method="ward")

    # Initialize FOSC-X
    model = FOSCX(top_M=5, kmin=2, kmax=None)

    # Extract clusterings
    model.fit(Z)


Condensed / Non-Binary Tree Matrices
------------------------------------

FOSC-X supports condensed (non-binary) tree formats, such as those
produced internally by HDBSCAN.

These are expected in the form:

``[parent, child, distance, size]``

The interpretation of ``distance`` is the same as for linkage matrices and
depends on whether the tree is distance-based or density-based. This is
again controlled via the ``density`` parameter.

.. code-block:: python

    import hdbscan

    # Build hierarchical clustering
    clusterer = hdbscan.HDBSCAN().fit(X)
    Z = clusterer.condensed_tree_.to_pandas()

    # Initialize FOSC-X
    model = FOSCX(top_M=5, kmin=2, kmax=None, density=True)

    # Extract clusterings
    model.fit(Z)

.. _json_tree:

Pre-Computed JSON Trees
-----------------------

FOSC-X also supports user-defined trees with pre-computed quality values,
provided in JSON format.

This gives you full control over both the hierarchy and the objective function,
making it useful for custom pipelines or external clustering workflows.


JSON Structure
~~~~~~~~~~~~~~

The JSON file must define a flattened tree structure with the following fields:

.. code-block:: text

    root_id: str (required)
    complete_tree: bool (required)
    condensed_simplified_tree: bool (required)
    density_based: bool (required)
    data: array | object | null (optional)

    tree: {
        node_id: {
            "parent": str,              (required)
            "children": [str],          (required)
            "cluster_size": int,        (optional)
            "distance": float | list,   (optional)
            "noise": bool,              (optional)
            "quality": float            (required)
        },
        ...
    }

For full validation rules (including accepted data types and constraints),
see the JSON schema: :file:`hierarchy.schema.json` :contentReference[oaicite:0]{index=0}


Field descriptions
------------------

- ``root_id``  
  The identifier of the root node. Must be a non-empty string.

- ``complete_tree``  
  Indicates whether the hierarchy includes all observation-level leaf nodes.

- ``condensed_simplified_tree``  
  Indicates whether the tree has been pre-condensed or simplified.

- ``data`` *(optional)*  
  Optional data payload associated with the tree (e.g. original dataset or embeddings).

- ``tree``  
  A dictionary mapping node IDs (strings) to node objects. Each node must contain:

  - ``parent``  
    The parent node ID. Use an empty string (``""``) for the root node.

  - ``children``  
    A list of child node IDs. Use an empty list (``[]``) for leaf nodes.

  - ``cluster_size`` *(optional)*  
    Number of samples in the cluster. Must be a non-negative integer.

  - ``distance`` *(optional)*  
    Distance or density value associated with the node.  
    May be a float or a list of floats depending on the tree type.

  - ``noise`` *(optional)*  
    Boolean flag indicating whether the node represents noise.

  - ``quality``  
    Pre-computed quality score used by FOSC-X.


Notes
-----

- Node IDs must be unique, non-empty strings with no whitespace.
- The tree must contain at least one node.
- All referenced parent/child IDs must exist in the ``tree`` dictionary.

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    json_path = "path/to/your/tree.json"

    # Initialize FOSC-X
    model = FOSCX(top_M=5, kmin=2, kmax=None)

    # Fit model to hierarchy
    model.fit(json_path)

Key Requirements
~~~~~~~~~~~~~~~~

To ensure compatibility with FOSC-X, the JSON tree must satisfy the following:

- Each node must define both its ``parent`` and ``children``  
- A ``quality`` value must be provided for every node  
- The tree must be connected and acyclic  

The ``complete_tree`` flag specifies whether the hierarchy includes all
leaf nodes corresponding to individual observations:

- If ``True``: full functionality is available (e.g. label extraction, plotting)  
- If ``False``: the tree is treated as pre-pruned, and some features are disabled  


Behaviour and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~

When using pre-computed JSON trees, some built-in functionality is bypassed:

- Built-in quality measures (e.g. ``Stability``, ``Modularity``, ``PFCE``) are ignored  
- Tree condensation (``min_cluster_size``) is not available  
- The provided ``quality`` values are used directly  

The following options remain available:

- ``singletons_as_noise``  
  (requires ``complete_tree=True`` or that ``cluster_size`` is provided)

- ``keep_noise_quality``  

If the ``noise`` field is not specified, all nodes are assumed to be non-noise.

If both ``condensed_simplified_tree=True`` and ``complete_tree=True``, all leaf
nodes are treated as noise regardless of other settings.


Output
~~~~~~

Results are returned using the original node IDs defined in the JSON tree.