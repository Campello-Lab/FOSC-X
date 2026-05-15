# -*- coding: utf-8 -*-
from .._numba import njit
import numpy as np
import math

# ------------------------------------------------------------
# Cluster tree leaf access
# ------------------------------------------------------------
@njit(cache=True)
def _get_node_labels_PFCE(leaf_order, node_start, node_end, node_id):
    """
    Return a NumPy *view* of the leaf labels for node_id (no copy).
    """

    s = node_start[node_id]
    e = node_end[node_id]
    if e <= s:
        return leaf_order[0:0]
    return leaf_order[s:e]


# ------------------------------------------------------------
# Build GLOBAL MST CSR 
# ------------------------------------------------------------
@njit(cache=True)
def _build_global_mst_csr(mst, n_obs):
    """
    Build CSR representation of the global MST from edge list.
    mst: (M, 3) array of edges (u, v, w)
    n_obs: number of observations (nodes)
    Returns: row_ptr, col_ind, weights
    """

    M = mst.shape[0]

    degree = np.zeros(n_obs, dtype=np.int64)
    for i in range(M):
        u = int(mst[i, 0])
        v = int(mst[i, 1])
        degree[u] += 1
        degree[v] += 1

    row_ptr = np.empty(n_obs + 1, dtype=np.int64)
    row_ptr[0] = 0
    for i in range(n_obs):
        row_ptr[i + 1] = row_ptr[i] + degree[i]

    col_ind = np.empty(row_ptr[-1], dtype=np.int64)
    weights = np.empty(row_ptr[-1], dtype=np.float64)

    offset = np.zeros(n_obs, dtype=np.int64)
    for i in range(M):
        u = int(mst[i, 0])
        v = int(mst[i, 1])
        w = mst[i, 2]

        pu = row_ptr[u] + offset[u]
        col_ind[pu] = v
        weights[pu] = w
        offset[u] += 1

        pv = row_ptr[v] + offset[v]
        col_ind[pv] = u
        weights[pv] = w
        offset[v] += 1

    return row_ptr, col_ind, weights


# ------------------------------------------------------------
# Extract cluster MST from global CSR 
# ------------------------------------------------------------
@njit(cache=True)
def _extract_cluster_mst_from_csr_labels(
    row_ptr,
    col_ind,
    weights,
    labels,          # <── cluster members
    in_cluster,      # still needed for neighbor check
    u_buf,
    v_buf,
    w_buf
):
    """
    Extract cluster MST edges from global CSR representation using cluster labels.
    Returns number of edges extracted.
    """

    cnt = 0
    max_edges = u_buf.shape[0]

    for i in range(labels.shape[0]):
        u = labels[i]

        for p in range(row_ptr[u], row_ptr[u + 1]):
            v = col_ind[p]
            if v > u and in_cluster[v]:
                if cnt < max_edges:
                    u_buf[cnt] = u
                    v_buf[cnt] = v
                    w_buf[cnt] = weights[p]
                    cnt += 1

    return cnt


# ------------------------------------------------------------
# Percentiles (NumPy-equivalent)
# ------------------------------------------------------------
@njit(cache=True)
def _percentile_linear(sorted_w, q):
    """
    Compute q-th percentile using linear interpolation.
    sorted_w: sorted 1D array
    q: quantile in [0, 1]
    """

    n = sorted_w.shape[0]
    if n == 1:
        return sorted_w[0]

    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return sorted_w[lo]

    return sorted_w[lo] * (hi - pos) + sorted_w[hi] * (pos - lo)


@njit(cache=True)
def _mst_edge_weight_stats(w):
    """
    Compute MST edge weight statistics: max, Q1, Q2, Q3
    w: 1D array of edge weights
    """

    if w.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0

    ws = np.sort(w)
    return (
        ws[-1],
        _percentile_linear(ws, 0.25),
        _percentile_linear(ws, 0.50),
        _percentile_linear(ws, 0.75),
    )


# ------------------------------------------------------------
# SDI / dividing weight
# ------------------------------------------------------------
@njit(cache=True)
def _dividing_weight_from_mst_numba(u, v, w, n):
    """
    Compute the dividing edge weight using the SDI criterion.

    Parameters
    ----------
    u, v : int64 arrays
        Endpoints of MST edges (local node indices 0..n-1)
    w : float64 array
        Corresponding edge weights
    n : int
        Number of nodes in the cluster

    Returns
    -------
    w_div : float
        Edge weight that maximizes SDI
    sdi_max : float
        Maximum SDI value achieved
    """

    # Trivial clusters have no meaningful division
    if n <= 1:
        return np.nan, 0.0

    # Sort edges by increasing weight
    order = np.argsort(w)

    # Union–Find initialization
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)

    # Track size of largest connected component
    max_size = 1

    def find(x):
        # Path-compressed find
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        nonlocal max_size
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return

        # Union by size
        if size[ra] < size[rb]:
            ra, rb = rb, ra

        parent[rb] = ra
        size[ra] += size[rb]

        # Update largest CC size
        if size[ra] > max_size:
            max_size = size[ra]

    # Largest CC before current weight threshold
    CCmax_before = 1

    # Best SDI found so far
    sdi_max = -1.0
    w_div = np.nan

    i = 0
    while i < order.shape[0]:
        # Process all edges with identical weight together
        j = i
        w_cur = w[order[i]]

        # Add all edges with this weight
        while j < order.shape[0] and w[order[j]] == w_cur:
            union(u[order[j]], v[order[j]])
            j += 1

        # Largest CC after adding these edges
        CCmax_after = max_size

        # SDI formula
        sdi = w_cur * (CCmax_after - CCmax_before) / CCmax_after

        # Track maximum SDI
        if sdi > sdi_max:
            sdi_max = sdi
            w_div = w_cur

        # Prepare for next weight level
        CCmax_before = CCmax_after
        i = j

    return w_div, sdi_max







@njit(cache=True)
def _remap_cluster_edges(u, v, labels, ru, rv):
    """
    Remap global node ids in u, v to local indices 0..k-1
    labels: cluster leaf labels
    """
    k = labels.shape[0]
    max_label = labels.max()

    mapping = np.full(max_label + 1, -1, dtype=np.int64)
    for i in range(k):
        mapping[labels[i]] = i

    cnt = 0
    for i in range(u.shape[0]):
        a = mapping[u[i]]
        b = mapping[v[i]]
        if a != -1 and b != -1:
            ru[cnt] = a
            rv[cnt] = b
            cnt += 1

    return cnt

@njit(cache=True)
def _build_cluster_boundary_edges(
    row_ptr,
    col_ind,
    weights,
    labels,        # cluster nodes
    in_cluster,
    C_o,
    bu, bv, bw
):
    """
    Build boundary edges between contracted cluster node C_o and outside nodes.
    Returns number of boundary edges created.
    """

    n_obs = in_cluster.shape[0]
    ext_min = np.full(n_obs, np.inf)
    cnt = 0

    # Only scan edges incident to cluster nodes
    for i in range(labels.shape[0]):
        u = labels[i]
        for p in range(row_ptr[u], row_ptr[u + 1]):
            v = col_ind[p]
            if in_cluster[v]:
                continue
            w = weights[p]
            if w < ext_min[v]:
                ext_min[v] = w

    # Create contracted edges C_o ↔ outside
    for v in range(n_obs):
        if ext_min[v] < np.inf:
            bu[cnt] = C_o
            bv[cnt] = v
            bw[cnt] = ext_min[v]
            cnt += 1

    return cnt

@njit(cache=True)
def _surrounding_edge_weights_and_sew(
    row_ptr,
    col_ind,
    weights,
    bu, bv, bw, bcount,
    C_o,
    ET,
    surrounding_buf,
    stack,
    visited,
    parent
):
    """
    DFS to find surrounding edge weights and SEW using implicit cluster node C_o.
    Returns: scount, SEWc
    """

    visited[:] = 0
    parent[:] = -1

    stack_size = 1
    stack[0] = C_o
    visited[C_o] = 1

    scount = 0
    sep = np.inf
    cap = surrounding_buf.shape[0]

    while stack_size > 0:
        node = stack[stack_size - 1]
        stack_size -= 1

        # -------------------------------
        # Case 1: contracted node C_o
        # -------------------------------
        if node == C_o:
            for i in range(bcount):
                nb_node = bv[i]   # C_o -> outside
                w = bw[i]

                if w <= ET:
                    if scount < cap:
                        surrounding_buf[scount] = w
                        scount += 1
                    if visited[nb_node] == 0:
                        visited[nb_node] = 1
                        parent[nb_node] = node
                        stack[stack_size] = nb_node
                        stack_size += 1
                else:
                    if w < sep:
                        sep = w
            continue

        # -------------------------------
        # Case 2: normal MST node
        # -------------------------------
        for p in range(row_ptr[node], row_ptr[node + 1]):
            nb_node = col_ind[p]
            w = weights[p]

            if parent[node] == nb_node:
                continue

            if w <= ET:
                if scount < cap:
                    surrounding_buf[scount] = w
                    scount += 1
                if visited[nb_node] == 0:
                    visited[nb_node] = 1
                    parent[nb_node] = node
                    stack[stack_size] = nb_node
                    stack_size += 1
            else:
                if w < sep:
                    sep = w

    return scount, sep
# ------------------------------------------------------------
# MAIN DRIVER
# ------------------------------------------------------------
@njit(cache=True)
def _compute_PFCE(
    mst,
    leaf_order,
    node_start,
    node_end,
    sizes,
    parent_arr,
    min_cluster_size
):
    """
    Compute PFCE scores for all clusters in the tree.
    mst: (M, 3) array of global MST edges (u, v, weight)
    leaf_order: (N,) array of leaf labels in DFS order
    node_start, node_end: (T,) arrays of node leaf spans
    sizes: (T,) array of cluster sizes
    parent_arr: (T,) array of parent indices
    min_cluster_size: minimum cluster size to consider
    Returns: (T,) array of PFCE scores
    """
    # --------------------------------------------------
    # Basic sizes
    # --------------------------------------------------
    n_tree_nodes = sizes.shape[0]
    n_obs = leaf_order.shape[0]

    # Find total number of observations (root size)
    total_obs = 0
    for i in range(n_tree_nodes):
        if parent_arr[i] == -1:
            total_obs = sizes[i]
            break

    # --------------------------------------------------
    # Build GLOBAL MST CSR (ONCE)
    # --------------------------------------------------
    row_ptr, col_ind, mst_w = _build_global_mst_csr(mst, n_obs)

    # --------------------------------------------------
    # Allocate reusable buffers
    # --------------------------------------------------
    in_cluster = np.zeros(n_obs, dtype=np.uint8)

    # Cluster MST buffers (≤ |C|-1 edges)
    u_buf = np.empty(n_obs - 1, dtype=np.int64)
    v_buf = np.empty(n_obs - 1, dtype=np.int64)
    w_buf = np.empty(n_obs - 1, dtype=np.float64)

    # Remapped MST buffers (local indices)
    ru_buf = np.empty(n_obs - 1, dtype=np.int64)
    rv_buf = np.empty(n_obs - 1, dtype=np.int64)

    # Boundary-edge buffers (C_o ↔ outside)
    bu = np.empty(n_obs, dtype=np.int64)
    bv = np.empty(n_obs, dtype=np.int64)
    bw = np.empty(n_obs, dtype=np.float64)



    # DFS / surrounding-edge buffers
    surrounding_buf = np.empty(2 * n_obs, dtype=np.float64)
    stack = np.empty(n_obs + 1, dtype=np.int64)
    visited = np.zeros(n_obs + 1, dtype=np.uint8)
    parent = np.empty(n_obs + 1, dtype=np.int64)

    # Results
    results = np.full(n_tree_nodes, 0.0, dtype=np.float64)

    # --------------------------------------------------
    # Main loop over clusters
    # --------------------------------------------------
    for node_id in range(n_tree_nodes):

        C_size = sizes[node_id]
        if C_size < min_cluster_size or C_size == total_obs:
            continue

        # ----------------------------------------------
        # Cluster membership
        # ----------------------------------------------
        in_cluster[:] = 0
        labels = _get_node_labels_PFCE(
            leaf_order, node_start, node_end, node_id
        )
        k = labels.shape[0]

        for i in range(k):
            in_cluster[labels[i]] = 1

        # ----------------------------------------------
        # Extract cluster MST (label-based, fast)
        # ----------------------------------------------
        edge_count = _extract_cluster_mst_from_csr_labels(
            row_ptr,
            col_ind,
            mst_w,
            labels,
            in_cluster,
            u_buf,
            v_buf,
            w_buf
        )

        if edge_count < k - 1:
            continue

        # ----------------------------------------------
        # MST edge statistics
        # ----------------------------------------------
        wmax, Q1c, Q2c, Q3c = _mst_edge_weight_stats(
            w_buf[:edge_count]
        )

        if Q2c <= 0.0:
            continue

        # ----------------------------------------------
        # Remap MST edges to local indices
        # ----------------------------------------------
        remapped_cnt = _remap_cluster_edges(
            u_buf[:edge_count],
            v_buf[:edge_count],
            labels,
            ru_buf,
            rv_buf
        )

        if remapped_cnt < k - 1:
            continue

        # ----------------------------------------------
        # Dividing weight (SDI)
        # ----------------------------------------------
        wdiv, _ = _dividing_weight_from_mst_numba(
            ru_buf[:remapped_cnt],
            rv_buf[:remapped_cnt],
            w_buf[:remapped_cnt],
            k
        )

        if np.isnan(wdiv):
            continue

        # ----------------------------------------------
        # Edge threshold
        # ----------------------------------------------
        ET = Q3c + 3.0 * (Q3c - Q1c)

        # ----------------------------------------------
        # Build boundary edges ONLY (no outside scan)
        # ----------------------------------------------
        C_o = n_obs
        bcount = _build_cluster_boundary_edges(
            row_ptr,
            col_ind,
            mst_w,
            labels,
            in_cluster,
            C_o,
            bu,
            bv,
            bw
        )



        # ----------------------------------------------
        # Surrounding edges & SEW
        # ----------------------------------------------
        scount, SEWc = _surrounding_edge_weights_and_sew(
            row_ptr,
            col_ind,
            mst_w,
            bu,
            bv,
            bw,
            bcount,
            C_o,
            ET,
            surrounding_buf,
            stack,
            visited,
            parent
        )

        # ----------------------------------------------
        # SDc
        # ----------------------------------------------
        if scount == 0:
            SDc = SEWc
        else:
            mean_SE = 0.0
            for i in range(scount):
                mean_SE += surrounding_buf[i]
            mean_SE /= scount

            if math.isinf(SEWc):
                SDc = mean_SE
            else:
                SDc = 0.5 * (SEWc + mean_SE)

        if SDc <= 0.0:
            continue

        # ----------------------------------------------
        # Final scores
        # ----------------------------------------------
        SPAR = (wmax - Q2c) / (wmax + Q2c)
        DIV = (wdiv - Q2c) / (wdiv + Q2c)
        SEP = (SDc - Q2c) / (SDc + Q2c)

        VC = 3.0 + SEP - DIV - SPAR
        results[node_id] = VC * (C_size / total_obs)

    return results

