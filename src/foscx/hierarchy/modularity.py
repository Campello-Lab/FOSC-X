# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 21:05:44 2025

@author: csimp
"""
import os
from typing import Tuple
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

from numba import njit

from .tree_numba import _lca



def _compute_modularity(c_tree, X, min_samples, *, metric='euclidean', HDBSCAN=True, max_workers = None):
    """
    Compute modularity-like quality for clusters in `c_tree`.

    Parameters
    ----------
    c_tree : dict
        Cluster tree dictionary to be annotated with 'clusteval' and 'cluster_size'.
    order : list[int]
        Leaf ordering (maps indices to rows/columns of graph).
    X : array-like
        If metric != 'precomputed', X is data matrix used to build knn graph.
    min_samples : int
        Number of neighbors for graph construction (k).
    metric : str
        Metric used by sklearn NearestNeighbors.
    HDBSCAN : bool
        If True use mutual reachability transformation for distances.

    Returns
    -------
    dict
        The same `c_tree` with added 'clusteval' and 'cluster_size' entries.
    """
    # Build k-NN graph from data X


    k_graph = _build_knn_graph(X, min_samples, metric=metric, HDBSCAN=HDBSCAN)


    # Compute structural similarity matrix (sparse)
    sigma = cosine_similarity(k_graph, dense_output=False)
    sigma.setdiag(0)            # zero the diagonal
    sigma.eliminate_zeros()     # remove explicit zeros from storage


    Q = _modularity(sigma,c_tree,directed=True,max_workers=max_workers)

    
    
    return Q


def _build_knn_graph(X, min_samples, *, metric='euclidean', HDBSCAN=False):
    """
    Build a symmetric sparse k-NN similarity graph.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric sparse matrix of pairwise similarities (shape n_samples x n_samples).
    """
    if min_samples is None or min_samples < 1:
        raise ValueError("min_samples must be an integer >= 1")

    nn = NearestNeighbors(n_neighbors=min_samples, metric=metric).fit(X)
    dists, indices = nn.kneighbors(X)

    if HDBSCAN:
        # Mutual reachability distances produced as sparse distance matrix
        k_graph = _knn_mutual_reachability(indices, dists, min_samples)
        # Convert distances to similarity in a normalized way
        if k_graph.data.size:
            k_graph.data = 1 - k_graph.data / np.max(k_graph.data)
        k_graph = k_graph.maximum(k_graph.T)  # make symmetric
    else:
        # Convert distances to similarity in [0,1]
        max_d = np.max(dists) if dists.size else 1.0
        similarity = 1 - dists / max_d
        n_samples, K = indices.shape

        row_idx = np.repeat(np.arange(n_samples), K)
        col_idx = indices.flatten()
        data = similarity.flatten()
        k_graph = csr_matrix((data, (row_idx, col_idx)), shape=(n_samples, n_samples))
        k_graph = k_graph.maximum(k_graph.T)  # symmetrize

    return k_graph


def _knn_mutual_reachability(indices, dists, min_samples):
    """
    Compute mutual reachability distances from a k-NN neighbor index/distance arrays.

    Parameters
    ----------
    indices : array_like, shape (n_samples, k)
    dists : array_like, shape (n_samples, k)
    min_samples : int

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of mutual reachability distances.
    """
    n_samples, k = indices.shape

    # Core distances = distance to min_samples-th neighbor (indexing safe)
    if min_samples > k:
        warnings.warn("min_samples > provided k in neighbor results; using last column for core distances.")
        core_dists = dists[:, -1]
    else:
        core_dists = dists[:, min_samples - 1]

    row_idx = np.repeat(np.arange(n_samples), k)
    col_idx = indices.flatten()
    edge_dists = dists.flatten()

    core_i = np.repeat(core_dists, k)
    core_j = core_dists[indices].flatten()
    mr_dists = np.maximum.reduce([edge_dists, core_i, core_j])

    mr_graph = csr_matrix((mr_dists, (row_idx, col_idx)), shape=(n_samples, n_samples))

    return mr_graph




# -------------------------
# Numba worker: processes a chunk of edges and returns S_partial
# -------------------------
@njit(cache=True)
def _process_edge(row, col, data, start_idx, end_idx, leaf_map, ancestor_table, depths, Nnodes):
    """
    Numba-compiled worker: iterate edges [start_idx, end_idx) and accumulate S_partial per node.
    All inputs must be NumPy arrays with appropriate dtypes and contiguous.
    Returns a 1D float64 array length Nnodes.
    """
    S = np.zeros(Nnodes, dtype=np.float64)
    for ei in range(start_idx, end_idx):
        u = int(row[ei])
        v = int(col[ei])
        w = float(data[ei])
        nu = int(leaf_map[u])
        nv = int(leaf_map[v])
        # call your numba LCA (also njit)
        L = int(_lca(ancestor_table, depths, nu, nv))
        S[L] += w
    return S

# -------------------------
# Postprocess helper (pure python but fast enough) - can be left as-is
# -------------------------
def _postprocess_subtrees(S_total: np.ndarray, indptr: np.ndarray, indices: np.ndarray,
                          deg: np.ndarray, leaf_map: np.ndarray,
                          parent: np.ndarray, children_off: np.ndarray, children_flat: np.ndarray,
                          n_obs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given S_total (per-node sums of edge weights whose LCA == node), compute:
      - W_internal (subtree accumulation of S_total)
      - K: subtree degree sums
      - Q: modularity contributions
    """
    Nnodes = parent.shape[0]

    # build postorder traversal (iterative)
    root = -1
    for i in range(Nnodes):
        if parent[i] == -1:
            root = i
            break
    if root == -1:
        raise ValueError("No root found in parent array (no -1).")

    post = []
    stack = [(root, 0)]
    while stack:
        node, state = stack.pop()
        if state == 0:
            stack.append((node, 1))
            s = children_off[node]; e = children_off[node+1]
            for j in range(e-1, s-1, -1):
                stack.append((children_flat[j], 0))
        else:
            post.append(node)

    # subtree accumulation of S -> W_internal
    W_diag = np.zeros(Nnodes, dtype=np.float64)
    for node in post:
        total = float(S_total[node])
        s = children_off[node]; e = children_off[node+1]
        for j in range(s, e):
            ch = int(children_flat[j])
            total += W_diag[ch]
        W_diag[node] = total
    W_internal = W_diag

    # compute K: degrees into leaves then subtree-sum
    K_leaf = np.zeros(Nnodes, dtype=np.float64)
    for obs in range(n_obs):
        ln = int(leaf_map[obs])
        K_leaf[ln] += float(deg[obs])
    K = np.zeros(Nnodes, dtype=np.float64)
    for node in post:
        total = float(K_leaf[node])
        s = children_off[node]; e = children_off[node+1]
        for j in range(s, e):
            ch = int(children_flat[j])
            total += K[ch]
        K[node] = total

    # compute Q
    total_deg = float(deg.sum())
    m = total_deg * 0.5
    Q = np.zeros(Nnodes, dtype=np.float64)
    if m > 0.0:
        denom1 =  2 * m
        denom2 = 4.0 * m * m
        for i in range(Nnodes):
            Q[i] = (W_internal[i] / denom1) - (K[i] * K[i]) / denom2
    return W_internal, K, Q

# -------------------------
# Top-level: chunk edges and call numba worker per chunk via ThreadPoolExecutor
# -------------------------
def _modularity(A_csr: csr_matrix, tree, leaf_node_ids=None, directed=False, max_workers=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chunked modularity using a Numba-compiled worker that calls lca_numba inside the inner loop.
    Behaves like previous function but inner loop is JITted.
    - does NOT remap labels when leaf_node_ids is None (identity mapping).
    """
    if not isinstance(A_csr, csr_matrix):
        A_csr = csr_matrix(A_csr)

    n_obs = A_csr.shape[0]
    coo = A_csr.tocoo()
    row = coo.row.copy()
    col = coo.col.copy()
    data = coo.data.copy()
    # ensure dtypes
    row = row.astype(np.int64, copy=False)
    col = col.astype(np.int64, copy=False)
    data = data.astype(np.float64, copy=False)

    if not directed:
        mask = row <= col
        row = row[mask]; col = col[mask]; data = data[mask]

    if leaf_node_ids is None:
        leaf_map = np.arange(n_obs, dtype=np.int64)
    else:
        leaf_map = np.asarray(leaf_node_ids, dtype=np.int64)
        if leaf_map.shape[0] != n_obs:
            raise ValueError("leaf_node_ids must have length equal to A.shape[0]")

    parent = np.asarray(tree.parent, dtype=np.int64)
    Nnodes = parent.size
    children_off = np.asarray(tree.children_off, dtype=np.int64)
    children_flat = np.asarray(tree.children_flat, dtype=np.int64)

    # ancestor_table and depths must be present and arrays
    if not hasattr(tree, "_ancestor_table") or tree._ancestor_table is None:
        raise ValueError("tree._ancestor_table is None — build it before calling (tree.build_ancestor_table())")
    if not hasattr(tree, "_depths") or tree._depths is None:
        raise ValueError("tree._depths is None — compute depths before calling (tree.compute_depths())")
    ancestor_table = np.ascontiguousarray(np.asarray(tree._ancestor_table))
    depths = np.ascontiguousarray(np.asarray(tree._depths))

    # precompute degrees
    deg = np.asarray(A_csr.sum(axis=1)).ravel().astype(np.float64)

    # decide workers
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if max_workers < 1:
        max_workers = 1

    E = row.shape[0]
    if E == 0:
        return (np.zeros(Nnodes, dtype=np.float64),
                np.zeros(Nnodes, dtype=np.float64),
                np.zeros(Nnodes, dtype=np.float64))

    P = min(max_workers, E)
    sizes = [(E // P) + (1 if i < (E % P) else 0) for i in range(P)]
    bounds = []
    start = 0
    for s in sizes:
        bounds.append((start, start + s))
        start += s

    # make contiguous arrays for safe passing into numba
    row = np.ascontiguousarray(row)
    col = np.ascontiguousarray(col)
    data = np.ascontiguousarray(data)
    leaf_map = np.ascontiguousarray(leaf_map)

    # submit compiled worker per chunk
    partials = []
    futures = []
    with ThreadPoolExecutor(max_workers=P) as ex:
        for (s, e) in bounds:
            # Note: process_edge_chunk_numba_jit is njit compiled and thread-safe for read-only inputs
            futures.append(ex.submit(_process_edge, row, col, data, s, e, leaf_map, ancestor_table, depths, Nnodes))
        for fut in as_completed(futures):
            partials.append(fut.result())

    # sum partials
    S_total = np.zeros(Nnodes, dtype=np.float64)
    for part in partials:
        S_total += part

    # postprocess to get W_internal, K, Q
    W_internal, K, Q = _postprocess_subtrees(S_total, A_csr.indptr.astype(np.int64),
                                             A_csr.indices.astype(np.int64),
                                             deg, leaf_map, parent, children_off, children_flat, n_obs)
    return Q
