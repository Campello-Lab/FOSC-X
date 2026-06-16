.. _advanced_usage:

Advanced Usage: Working with the Hierarchy
===========================================

The hierarchy object
--------------------

After fitting, the underlying tree can be accessed via:

.. code-block:: python

    H = model.cluster_tree_

This object is an instance of :class:`foscx.hierarchy.hierarchy.Cluster_Tree`,
which provides access to node-level information and tree structure.

The following sections highlight the most commonly used functionality.


Inspecting Nodes
----------------

Individual nodes in the hierarchy can be inspected using:

.. code-block:: python

    H.node_summary(node_id)


This returns a dictionary containing key information about the node:

- ``parent``: parent node ID  
- ``children``: list of child node IDs  
- ``cluster_size``: number of points in the cluster  
- ``distance``: merge / hierarchy distance  
- ``noise``: whether the node is treated as noise  
- ``quality``: quality score for the node  

This is useful for understanding how specific clusters are represented in
the hierarchy.

.. code-block:: python

    H.node_summary(301)

.. code-block:: text

    {
        'parent': 300,
        'children': [302, 303],
        'cluster_size': 120,
        'distance': 0.42,
        'noise': False,
        'quality': 28309.19
    }

The nodes 0:N-1 correspond to the original observations, while nodes N:2N-2 represent internal clusters. The root node is always N.

Accessing and Modifying Quality
-------------------------------

The quality values used by FOSC-X are stored in:

.. code-block:: python

    H.clusteval

This is a NumPy array containing the score for each node.

Advanced users may modify this directly to experiment with custom quality
definitions:

.. code-block:: python

    H.clusteval[...] = custom_values

After modifying the quality vector, FOSC-X can be rerun without rebuilding the tree by calling model.predict(...).


.. _advanced_noise:

Accessing and Modifying Noise
-----------------------------

Noise assignments are stored in:

.. code-block:: python

    H.is_noise

This is a binary array indicating whether each observation/leaf is treated as noise.

Users may modify this to experiment with different noise definitions:

.. code-block:: python

    H.is_noise[...] = new_noise_mask

After modifying noise assignments, bounds and candidate solutions must be recomputed using H.compute_bounds() and model.predict(...).


Extracting Data from Nodes
--------------------------

To retrieve the data points associated with a node:

.. code-block:: python

    labels = H.get_node_indices(node_id)

This returns the indices of observations belonging to that cluster.


Recomputing After Changes
--------------------------

If internal values such as ``clusteval`` or ``is_noise`` are modified, the
following steps are typically required:

.. code-block:: python

    H.compute_bounds()
    model.predict(...)

