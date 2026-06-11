"""
cluster_tree_proxy.py
=====================
A numpy-recarray-compatible proxy around ``Cluster_Tree`` that lets every
function in ``plot_tree_functions.py`` work unmodified.

Usage
-----
    from cluster_tree_proxy import ClusterTreeProxy

    proxy = ClusterTreeProxy(my_cluster_tree, density=True)   # HDBSCAN style
    proxy = ClusterTreeProxy(my_cluster_tree, density=False)  # linkage style

    # pass proxy wherever a condensed recarray is expected:
    plot_data = get_plot_data_nb(proxy)
    plot_condensed_nb(proxy, plot_data, axis=ax)
    plot_condensed_bin(proxy, axis=ax)
    interactive_condensed(proxy, binary_tree=False)

Distance semantics
------------------
``Cluster_Tree`` stores one ``distance`` value per node.

* Density trees (``density=True``): ``ct.distance[child]`` is the lambda at
  which the child falls out of its parent.  Stored directly on the edge.

* Linkage trees (``density=False``): ``ct.distance[node]`` is the distance
  at which that node splits.  Stored on the parent->child edge as the
  parent's split distance, which is what both plot paths expect.

No distance inversion is applied in the proxy — plot_tree_functions.py
handles axis direction via the ``density`` parameter passed to the plot calls.
The non-binary path is fixed to seed cluster_y_coords[root] from the actual
minimum edge distance rather than hardcoding 0.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hierarchy.hierarchy import Cluster_Tree


# ---------------------------------------------------------------------------
# Internal: build edge arrays
# ---------------------------------------------------------------------------

def _build_edge_arrays(ct: "Cluster_Tree", density: bool) -> dict:
    """Return per-edge arrays (parent, child, distance, child_size).

    Emitted in BFS order from the root, children sorted smallest-first so
    the largest subtree sits on the right of the plot.
    """
    root = int(np.where(ct.parent == -1)[0][0])

    parents   = []
    children  = []
    distances = []
    sizes     = []

    queue = [root]
    while queue:
        node = queue.pop(0)
        s = int(ct.children_off[node])
        e = int(ct.children_off[node + 1])
        node_children = [int(c) for c in ct.children_flat[s:e]]
        node_children.sort(key=lambda c: int(ct.sizes[c]), reverse=False)

        for child in node_children:
            if density:
                # lambda at which child falls out of its parent cluster
                d = float(ct.distance[child])
            else:
                # distance at which the parent splits — this is the height at
                # which child's bar sits in the linkage dendrogram
                d = float(ct.distance[node])

            parents.append(node)
            children.append(child)
            distances.append(d)
            sizes.append(int(ct.sizes[child]))

            queue.append(child)

    return {
        'parent':     np.array(parents,   dtype=np.int64),
        'child':      np.array(children,  dtype=np.int64),
        'distance':   np.array(distances, dtype=np.float64),
        'child_size': np.array(sizes,     dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Dtype shim
# ---------------------------------------------------------------------------

class _DtypeShim:
    def __init__(self, names):
        self.names = tuple(names)


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

class ClusterTreeProxy:
    """Numpy-recarray-compatible view of a ``Cluster_Tree``."""

    def __init__(
        self,
        ct: "Cluster_Tree",
        density: bool = True,
        _arrays:  dict | None       = None,
        _mask:    np.ndarray | None = None,
        _fields:  list[str] | None  = None,
        _scalar:  bool              = False,
    ):
        self._ct      = ct
        self._density = density

        if _arrays is None:
            self._arrays = _build_edge_arrays(ct, density)
        else:
            self._arrays = _arrays

        self._mask   = _mask
        self._fields = _fields
        self._scalar = _scalar
        self.dtype   = _DtypeShim(list(self._arrays.keys()))

    def _masked_col(self, field: str):
        col = self._arrays[field]
        if self._mask is not None:
            col = col[self._mask]
        if self._scalar:
            return col.item()
        return col

    def _len(self) -> int:
        if self._scalar:
            return 1
        col = next(iter(self._arrays.values()))
        return int(self._mask.sum()) if self._mask is not None else len(col)

    def _active_fields(self):
        return self._fields if self._fields is not None else list(self._arrays.keys())

    def _make_sub(self, arrays=None, mask=None, fields=None, scalar=False):
        return ClusterTreeProxy(
            self._ct, self._density,
            _arrays=arrays if arrays is not None else self._arrays,
            _mask=mask,
            _fields=fields,
            _scalar=scalar,
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._masked_col(key)

        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            return self._make_sub(mask=self._mask, fields=key, scalar=self._scalar)

        if isinstance(key, np.ndarray) and key.dtype == bool:
            if self._mask is not None:
                abs_indices = np.where(self._mask)[0]
                new_mask = np.zeros(len(self._arrays['parent']), dtype=bool)
                new_mask[abs_indices[key]] = True
            else:
                new_mask = key
            return self._make_sub(mask=new_mask, fields=self._fields, scalar=(int(new_mask.sum()) == 1))

        if isinstance(key, (int, np.integer)):
            if self._mask is not None:
                abs_idx = int(np.where(self._mask)[0][int(key)])
            else:
                abs_idx = int(key)
            row_mask = np.zeros(len(self._arrays['parent']), dtype=bool)
            row_mask[abs_idx] = True
            return self._make_sub(mask=row_mask, fields=self._fields, scalar=True)

        raise TypeError(f"ClusterTreeProxy: unsupported index type {type(key)!r}")

    def __iter__(self):
        for i in range(self._len()):
            yield self[i]

    def __len__(self):
        return self._len()

    def __repr__(self):
        return (f"ClusterTreeProxy(rows={self._len()}, "
                f"fields={self._active_fields()}, "
                f"density={self._density}, scalar={self._scalar})")