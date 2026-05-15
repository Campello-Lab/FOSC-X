import numpy as np
from typing import Sequence, Optional, Iterable, Dict

from .tree_numba import (
    _compute_sizes,
    _compute_depths,
    _compute_stability,
    _build_ancestor_table,
    _kth_ancestor,
    _lca
    )

from .cluster_bounds import(
    _compute_leaf_noise_and_nonnoise_siblings,
    _compute_leaf_order_and_node_spans,
    _compute_bounds
    )

from .tree_build import (
    _build_hierarchy,
    _dict_to_hierarchy
)

from .modularity import _compute_modularity
from .pfce import _compute_PFCE
from .B3Measure import B3_F_Measure
from .constraint_score import compress_constraints_, constraint_scores_



class Cluster_Tree:
    """
    Cluster_Tree: efficient representation of a hierarchical clustering tree.
    Stores tree in compact NumPy arrays for fast processing with Numba.
    Supports construction from HDBSCAN condensed tree or dictionary tree.
    """

    # ------------------------------------------------------------
    # Construction methods
    # ------------------------------------------------------------

    def __init__(self,
                 parent: np.ndarray,
                 distance: np.ndarray,
                 children_flat: np.ndarray,
                 children_off: np.ndarray,
                 sizes: np.ndarray = None,
                 clusteval: np.ndarray = None):
        self.parent = parent.astype(np.int32)
        self.children_flat = children_flat.astype(np.int32)
        self.children_off = children_off.astype(np.int32)
        self.distance = distance.astype(np.float64)

        self.N = int(self.parent.shape[0])

        if sizes is None:
            self.sizes = np.zeros(self.N, dtype=np.int32)
        else:
            self.sizes = sizes.astype(np.int32)

        if clusteval is None:
            self.clusteval = np.zeros(self.N, dtype=np.float64)
        else:
            self.clusteval = clusteval.astype(np.float64)

        self._depths = None
        self._ancestor_table = None


    @classmethod
    def _from_hdbscan_condensed(cls,
                                    condensed: "np.ndarray",
                                    col_idx: Optional[Dict[str,int]] = None,
                                    field_names: Optional[Dict[str,str]] = None,
                                    density: bool = True
                                    ):
        """
        Build a FastHierarchy from an HDBSCAN condensed_tree_.to_numpy() result.
        Accepts either a structured 1D ndarray (with named fields) or a 2D numeric ndarray.
        Use field_names to control structured-field mapping (defaults to common HDBSCAN names).
        """
        parent, distance, children_flat, children_off, sizes = _build_hierarchy(
            condensed=condensed,
            col_idx=col_idx,
            field_names=field_names,
            density=density
        )
        return cls(parent=parent,
                distance=distance,
                children_flat=children_flat,
                children_off=children_off,
                sizes=sizes)

    @classmethod
    def _from_dict_tree(cls, dict_tree: Dict, complete_tree: bool = False):
        """
        Build a FastHierarchy from a dictionary tree representation.
        The dictionary tree should have the format:
        {

        tree: {
            node_id: {
                "parent": parent_id,
                "distance": distance_value,
                "children": [child_id1, child_id2, ...],
                ...
            },
            ...
            } 
        }
        """

        parent, distance, children_flat, children_off, sizes, clusteval, is_noise, id_map, rev_id_map = \
            _dict_to_hierarchy(tree=dict_tree, complete_tree=complete_tree)

        obj = cls(
            parent=parent,
            distance=distance,
            children_flat=children_flat,
            children_off=children_off,
            sizes=sizes,
            clusteval=clusteval
        )

        # Attach metadata
        obj.id_map = id_map
        obj.rev_id_map = rev_id_map
        obj.is_noise = is_noise
        return obj

    # ------------------------------------------------------------
    # Core Info
    # ------------------------------------------------------------

    def _compute_tree_depths(self):
        """
        Compute and store depths of all nodes in the tree.
        """
        self._depths = _compute_depths(self.parent, self.children_flat, self.children_off)
        return self._depths

    def _build_tree_ancestor_table(self):
        """
        Build ancestor table for fast LCA and kth-ancestor queries.
        """
        self._ancestor_table = _build_ancestor_table(self.parent)
        return self._ancestor_table        
    
    # ------------------------------------------------------------
    # Node ID mapping for JSON data
    # ------------------------------------------------------------

    def _to_user(self, node):
        """
        Convert compact ID to user-facing ID.
        """
        if hasattr(self, "rev_id_map"):
            return self.rev_id_map[node]
        else:
            return int(node)

    def _to_compact(self, node):
        """
        Convert user-facing node ID to compact ID.
        """
        if hasattr(self, "rev_id_map"):
            # mapping exists → user gave original ID
            if not hasattr(self, "_orig_to_compact"):
                self._orig_to_compact = {
                    oid: i for i, oid in enumerate(self.rev_id_map)
                }
            return self._orig_to_compact[node]
        else:
            # no mapping → already compact
            return int(node)

    # ------------------------------------------------------------
    # Core API: wrappers around numba kernels
    # ------------------------------------------------------------
    def node_summary(self, node):
        """
        Return a dictionary-style summary of a node, using user-facing IDs.

        Fields included (if available):
        - parent
        - children
        - cluster_size
        - distance
        - noise
        - quality
        """
        cnode = self._to_compact(node)

        out = {}

        # -------------------------
        # parent
        # -------------------------
        p = int(self.parent[cnode])
        if p == -1:
            out["parent"] = ""
        else:
            out["parent"] = self._to_user(p)

        # -------------------------
        # children
        # -------------------------
        s = self.children_off[cnode]
        e = self.children_off[cnode + 1]
        ch = self.children_flat[s:e]
        out["children"] = [self._to_user(c) for c in ch]

        # -------------------------
        # cluster size (if exists)
        # -------------------------
        if hasattr(self, "sizes") and self.sizes is not None:
            out["cluster_size"] = int(self.sizes[cnode])

        # -------------------------
        # distance (if exists)
        # -------------------------
        if hasattr(self, "distance") and self.distance is not None:
            out["distance"] = float(self.distance[cnode])

        # -------------------------
        # noise flag (if exists)
        # -------------------------
        if hasattr(self, "is_noise"):
            out["noise"] = bool(self.is_noise[cnode])

        # -------------------------
        # quality / clusteval (if exists)
        # -------------------------
        if hasattr(self, "clusteval") and self.clusteval is not None:
            out["quality"] = float(self.clusteval[cnode])

        return out

    def get_distance(self, nodes: Sequence) -> np.ndarray:
        """
        Return the distances of the given nodes.
        """
        cnodes = np.fromiter(
            (self._to_compact(n) for n in nodes),
            dtype=np.int32,
            count=len(nodes)
        )
        return self.distance[cnodes]

    def get_parent(self, node):
        """
        Return the parent of a node (user-facing ID).
        """
        cnode = self._to_compact(node)
        p = int(self.parent[cnode])
        if p == -1:
            return -1
        return self._to_user(p)

    def get_children(self, node):
        """
        Return a NumPy *view* of the children of node (no copy).
        """
        cnode = self._to_compact(node)
        s = self.children_off[cnode]
        e = self.children_off[cnode + 1]
        ch = self.children_flat[s:e]

        if hasattr(self, "rev_id_map"):
            return np.array([self.rev_id_map[c] for c in ch], dtype=object)
        else:
            return ch  # view, no copy
        
    def _compute_sizes(self):
        """
        Compute and store sizes of all nodes in the tree.
        """
        sizes = _compute_sizes(self.parent, self.children_flat, self.children_off, self.sizes.copy())
        self.sizes = sizes.astype(np.int32)


    def get_size(self, node) -> int:
        """
        Return number of original observations contained in node (leaf -> typically 1).
        """
        cnode = self._to_compact(node)
        return int(self.sizes[cnode])


    def is_leaf(self, node) -> bool:
        """
        Return True if node is a leaf (has no children).
        """
        cnode = self._to_compact(node)
        return (self.children_off[cnode + 1] - self.children_off[cnode]) == 0


        
    # ------------------------------------------------------------
    # Quality Measures
    # ------------------------------------------------------------

    def compute_stability(self, density=False):
        """
        Compute and store stability (exess of mass for density based trees) values for all nodes in the tree.
        """
        st = _compute_stability(parent=self.parent, 
            children_flat=self.children_flat, 
            children_off=self.children_off,
            distance=self.distance, 
            sizes=self.sizes, 
            density=bool(density)
        )

        self.clusteval = st.astype(np.float64)


    def compute_modularity(self,X, min_samples,*, metric='euclidean', HDBSCAN=True):
        """
        Compute and store modularity values for all nodes in the tree.
        Parameters
        ----------
        X : np.ndarray
            The original data points used to build the hierarchy.
        min_samples : int
            The number of nearest neighbors to consider for modularity calculation.
        metric : str, optional
            The distance metric to use (default is 'euclidean').
        HDBSCAN : bool, optional
            Whether the hierarchy was built using HDBSCAN (default is True).
        """

        Qmod = _compute_modularity(c_tree = self,X = X, 
            min_samples = min_samples,
            metric=metric,
            HDBSCAN=HDBSCAN
        )
        self.clusteval = Qmod

    def compute_PFCE(self,mst, min_cluster_size=5):
        """
        Compute and store Partition Free Cluster Evaluation (PFCE) scores for all nodes in the tree.
        Parameters
        ----------
        mst : np.ndarray
            The minimum spanning tree used to build the hierarchy.
        min_cluster_size : int, optional
            The minimum cluster size to consider for PFCE calculation (default is 5).
        """

        PFCE_scores = _compute_PFCE(
            mst = mst,
            leaf_order=self.leaf_order,
            node_start=self.node_start,
            node_end=self.node_end,
            sizes=self.sizes,
            parent_arr=self.parent,
            min_cluster_size = min_cluster_size
        )
        self.clusteval = PFCE_scores

    def compute_B3(self, GT_labels):
        """
        Compute the semi-supervised Bcubed measure of cluster evaluation.
        """
        B3scores = B3_F_Measure(
            cluster_tree = self ,
            labels = GT_labels,
            N_nodes = self.N)
        
        if self.clusteval is not None:
            self.clusteval = np.round(B3scores, 12) + 1e-12 * self.clusteval/max(self.clusteval)
        else:
            self.clusteval = B3scores

    def compute_constraint_score(self, constraints):
        """
        Compute the semi-supervised measure using must-link and must-not-link constraints.
        """
        compresed_constraints = compress_constraints_(constraints, max(self.sizes))

        scores = constraint_scores_(self, compresed_constraints)

        if self.clusteval is not None:
            self.clusteval = np.round(scores, 12) + 1e-12 * self.clusteval/max(self.clusteval)
        else:
            self.clusteval = scores


    def set_noise_quality(self,keep_noise_quality):
        """
        Set quality of noise nodes to zero if keep_noise_quality is False.
        """

        if keep_noise_quality is False:
            self.clusteval[self.is_noise == 1] = 0.0
    # ------------------------------------------------------------
    # Tree Functions
    # ------------------------------------------------------------
    def compute_leaf_noise_and_siblings(self, set_leaf_noise: bool = True, singleton_as_noise: bool = True):
        """
        Compute noise assignments and non-noise sibling counts.

        If ``self.is_noise`` already exists, it is used as a baseline and optionally
        updated. Otherwise, noise assignments are computed from scratch.

        Returns
        -------
        is_noise : numpy.ndarray
            Binary array indicating noise nodes (1 = noise, 0 = non-noise).

        nonnoise_sibling_count : numpy.ndarray
            Number of non-noise siblings for each node.

        Notes
        -----
        Updates ``self.is_noise`` and ``self.nonnoise_sibling_count`` in-place.
        """
        N = self.parent.shape[0]

        # -------------------------------------------------
        # Prepare optional noise input (Numba-safe)
        # -------------------------------------------------
        if hasattr(self, "is_noise") and self.is_noise is not None:
            is_noise_in = self.is_noise.astype(np.int8, copy=False)
        else:
            # empty sentinel → "no predefined noise"
            is_noise_in = np.empty(0, dtype=np.int8)

        # -------------------------------------------------
        # Apply singleton-as-noise (only to leaves)
        # -------------------------------------------------
        if singleton_as_noise:
            if is_noise_in.size == 0:
                # create baseline noise array
                is_noise_in = np.zeros(N, dtype=np.int8)

            is_noise_in[self.sizes == 1] = 1

        # -------------------------------------------------
        # Call numba kernel
        # -------------------------------------------------
        is_noise, nonnoise_sibling_count = _compute_leaf_noise_and_nonnoise_siblings(
            self.parent,
            self.children_flat,
            self.children_off,
            set_leaf_noise,
            is_noise_in,
        )

        # -------------------------------------------------
        # Save results
        # -------------------------------------------------
        self.is_noise = is_noise
        self.nonnoise_sibling_count = nonnoise_sibling_count

        return is_noise, nonnoise_sibling_count
    
    def compute_leaf_order_and_spans(self):
        """
        Compute leaf ordering and subtree spans.

        Assigns each leaf a position in a contiguous ordering and computes,
        for each node, the index range of its subtree.

        Returns
        -------
        leaf_order : numpy.ndarray
            Array of leaf node IDs in contiguous order.

        node_start : numpy.ndarray
            Start index (inclusive) of each node's subtree.

        node_end : numpy.ndarray
            End index (exclusive) of each node's subtree.

        Notes
        -----
        Updates ``self.leaf_order``, ``self.node_start`` and ``self.node_end`` in-place.
        """
        leaf_order, node_start, node_end = _compute_leaf_order_and_node_spans(
            self.children_flat, self.children_off, self.parent
        )
        self.leaf_order = leaf_order
        self.node_start = node_start
        self.node_end = node_end
        return leaf_order, node_start, node_end
    
    
    def get_node_indices(self, node_id):
        """
        Return a NumPy *view* of the leaf indices for node_id (no copy).
        """
        #if not hasattr(self, "leaf_order") or not hasattr(self, "node_start") or not hasattr(self, "node_end"):
        #    self.compute_leaf_order_and_spans()
        s = self.node_start[node_id]
        e = self.node_end[node_id]
        
        if e <= s:
            # return empty 1D array of int
            return np.empty(0, dtype=self.leaf_order.dtype)
        
        return self.leaf_order[s:e] 
    
    def compute_bounds(self):
        """
        Compute LB, UB and related arrays for the instance tree.
    
        Parameters
        ----------
        recompute_is_noise_if_missing : bool
            If True and `self.is_noise` is not present, call
            `self.compute_leaf_noise_and_siblings(set_leaf_noise)` to create it.
            If False and `self.is_noise` is missing, this raises an AttributeError.
        set_leaf_noise : bool
            Passed to `compute_leaf_noise_and_siblings` when recomputing is_noise.
            Default True (marks leaves as noise); change if you want different behaviour.
    
        Returns
        -------
        LB, UB, terminals_in_subtree, is_terminal, nonnoise_leaf_count
        """

        # make sure is_noise dtype is int8
        if not isinstance(self.is_noise, np.ndarray):
            self.is_noise = np.asarray(self.is_noise)
        if self.is_noise.dtype != np.int8:
            self.is_noise = self.is_noise.astype(np.int8)
    
        # call numba function
        LB, UB, terminals_in_subtree, is_terminal, nonnoise_leaf_count = _compute_bounds(
            self.parent, self.children_flat, self.children_off, self.is_noise
        )
    
        # store on self
        self.LB = LB
        self.UB = UB
        self.terminals_in_subtree = terminals_in_subtree
        self.is_terminal = is_terminal
        self.nonnoise_leaf_count = nonnoise_leaf_count
    
        return LB, UB, terminals_in_subtree, is_terminal, nonnoise_leaf_count
    


    # ------------------------------------------------------------
    def __repr__(self):
        return f"FastHierarchy(N={self.N}, children={len(self.children_flat)}, sizes_computed={self.sizes is not None})"
