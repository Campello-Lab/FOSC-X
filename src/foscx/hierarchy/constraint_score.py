import numpy as np
from itertools import combinations
from collections import defaultdict

from numba import njit, prange


def generate_pairwise_constraints_(
    labels
):
    """
    Fast pairwise constraint generation.
    """

    labels = np.asarray(labels)

    # ------------------------------------------------------------
    # Keep only labeled observations
    # ------------------------------------------------------------
    labeled_mask = labels != -1

    idx = np.nonzero(labeled_mask)[0]
    y = labels[labeled_mask]

    # ------------------------------------------------------------
    # Group indices by class
    # ------------------------------------------------------------
    unique_labels, inverse = np.unique(y, return_inverse=True)

    groups = [
        idx[inverse == k]
        for k in range(len(unique_labels))
    ]

    # ------------------------------------------------------------
    # Must-link constraints
    # ------------------------------------------------------------
    ml_parts = []

    for g in groups:

        m = len(g)

        if m < 2:
            continue

        ii, jj = np.triu_indices(m, k=1)

        ml_parts.append(
            np.column_stack((g[ii], g[jj]))
        )

    ml = (
        np.concatenate(ml_parts, axis=0)
        if ml_parts else
        np.empty((0, 2), dtype=int)
    )

    # ------------------------------------------------------------
    # Cannot-link constraints
    # ------------------------------------------------------------
    cl_parts = []

    for g1, g2 in combinations(groups, 2):

        a = np.repeat(g1, len(g2))
        b = np.tile(g2, len(g1))

        cl_parts.append(
            np.column_stack((a, b))
        )

    cl = (
        np.concatenate(cl_parts, axis=0)
        if cl_parts else
        np.empty((0, 2), dtype=int)
    )

    return ml, cl


# ==================================================================
# Numba kernels for compress_constraints_
# ==================================================================

@njit(cache=True)
def _find(parent, x):

    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]

    return x


@njit(cache=True)
def _union(parent, a, b):

    ra = _find(parent, a)
    rb = _find(parent, b)

    if ra != rb:
        parent[rb] = ra


@njit(cache=True)
def _union_find_labels(must_link, n_samples):
    """
    Build ML components via union-find and relabel to
    consecutive component ids, entirely in compiled code.
    """

    parent = np.arange(n_samples)

    for i in range(must_link.shape[0]):
        _union(parent, must_link[i, 0], must_link[i, 1])

    roots = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        roots[i] = _find(parent, i)

    label_map = -np.ones(n_samples, dtype=np.int64)
    component = np.empty(n_samples, dtype=np.int64)
    next_label = 0

    for i in range(n_samples):

        r = roots[i]

        if label_map[r] == -1:
            label_map[r] = next_label
            next_label += 1

        component[i] = label_map[r]

    return component


def compress_constraints_(
    constraints,
    n_samples
):
    """
    Compress constraints without explicit propagation.

    Returns
    -------
    component : ndarray
        ML component id for each observation.

    cl_components : dict[int, set[int]]
        Cannot-link relations between ML components.
    """

    must_link, cannot_link = constraints

    must_link = np.asarray(must_link, dtype=np.int64).reshape(-1, 2)
    cannot_link = np.asarray(cannot_link, dtype=np.int64).reshape(-1, 2)

    # ------------------------------------------------------------
    # Union-Find (compiled)
    # ------------------------------------------------------------
    component = _union_find_labels(must_link, n_samples)

    # ------------------------------------------------------------
    # Component-level CL graph (vectorized, then packed into a
    # dict[int, set[int]] for API compatibility)
    # ------------------------------------------------------------
    cl_components = defaultdict(set)

    if cannot_link.shape[0] > 0:

        ca = component[cannot_link[:, 0]]
        cb = component[cannot_link[:, 1]]

        bad = ca == cb

        if np.any(bad):
            i = int(np.nonzero(bad)[0][0])
            raise ValueError(
                f"Inconsistent constraints: "
                f"{cannot_link[i, 0]}, {cannot_link[i, 1]}"
            )

        lo = np.minimum(ca, cb)
        hi = np.maximum(ca, cb)

        pairs = np.unique(np.column_stack((lo, hi)), axis=0)

        for c1, c2 in pairs:
            c1 = int(c1)
            c2 = int(c2)
            cl_components[c1].add(c2)
            cl_components[c2].add(c1)

    return component, cl_components


# ==================================================================
# Numba kernel for constraint_scores_
# ==================================================================

@njit(cache=True, parallel=True)
def _score_nodes(node_start, node_end, leaf_order, component, comp_sizes,
                  cl_c1, cl_c2, total_constraints):

    n_nodes = node_start.shape[0]
    n_components = comp_sizes.shape[0]
    n_pairs = cl_c1.shape[0]

    scores = np.zeros(n_nodes, dtype=np.float64)

    for node in prange(n_nodes):

        start = node_start[node]
        end = node_end[node]

        if end <= start:
            continue

        # --------------------------------------------------------
        # Component counts inside this node (replaces Counter)
        # --------------------------------------------------------
        counts = np.zeros(n_components, dtype=np.int64)

        for i in range(start, end):
            counts[component[leaf_order[i]]] += 1

        # --------------------------------------------------------
        # ML satisfaction
        # --------------------------------------------------------
        ml_sat = 0

        for c in range(n_components):
            k = counts[c]
            ml_sat += k * (k - 1)

        # --------------------------------------------------------
        # CL satisfaction (exactly one endpoint inside)
        # --------------------------------------------------------
        cl_sat = 0

        for p in range(n_pairs):

            c1 = cl_c1[p]
            c2 = cl_c2[p]

            inside1 = counts[c1]
            inside2 = counts[c2]

            outside1 = comp_sizes[c1] - inside1
            outside2 = comp_sizes[c2] - inside2

            cl_sat += inside1 * outside2 + inside2 * outside1

        scores[node] = (ml_sat + cl_sat) / (2.0 * total_constraints)

    return scores


def constraint_scores_(
    tree,
    compresed_constraints
):

    component, cl_components = compresed_constraints
    component = np.asarray(component, dtype=np.int64)

    scores = np.zeros(tree.N, dtype=np.float64)

    # ------------------------------------------------------------
    # Component sizes
    # ------------------------------------------------------------
    comp_sizes = np.bincount(component).astype(np.int64)

    # ------------------------------------------------------------
    # Unique CL component pairs (flattened to arrays for numba)
    # ------------------------------------------------------------
    c1_list = []
    c2_list = []

    for c1, nbrs in cl_components.items():
        for c2 in nbrs:
            if c1 < c2:
                c1_list.append(c1)
                c2_list.append(c2)

    cl_c1 = np.asarray(c1_list, dtype=np.int64)
    cl_c2 = np.asarray(c2_list, dtype=np.int64)

    # ------------------------------------------------------------
    # Total ML / CL constraints
    # ------------------------------------------------------------
    total_ml = np.sum(
        comp_sizes * (comp_sizes - 1) // 2
    )

    total_cl = (
        int(np.sum(comp_sizes[cl_c1] * comp_sizes[cl_c2]))
        if cl_c1.size else 0
    )

    total_constraints = int(total_ml) + total_cl

    if total_constraints == 0:
        return scores

    # ------------------------------------------------------------
    # get_node_indices is backed by tree.node_start / node_end /
    # leaf_order, which is already a CSR layout - reuse it directly
    # instead of rebuilding one with a Python-level call per node.
    # ------------------------------------------------------------
    node_ids = np.arange(tree.N)

    try:
        # Fast path: _to_compact supports vectorized (array) input.
        compact_ids = np.asarray(tree._to_compact(node_ids), dtype=np.int64)
    except Exception:
        # Fallback: call per node (still cheap - just an index lookup,
        # not a data copy like get_node_indices was doing before).
        compact_ids = np.fromiter(
            (tree._to_compact(int(i)) for i in node_ids),
            dtype=np.int64,
            count=tree.N,
        )

    node_start = np.asarray(tree.node_start, dtype=np.int64)[compact_ids]
    node_end = np.asarray(tree.node_end, dtype=np.int64)[compact_ids]
    leaf_order = np.ascontiguousarray(tree.leaf_order, dtype=np.int64)

    scores = _score_nodes(
        node_start,
        node_end,
        leaf_order,
        component,
        comp_sizes,
        cl_c1,
        cl_c2,
        float(total_constraints),
    )

    return scores
