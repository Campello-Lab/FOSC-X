import numpy as np
from itertools import combinations
from collections import defaultdict, Counter


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
    
    parent = np.arange(n_samples)

    # ------------------------------------------------------------
    # Union-Find
    # ------------------------------------------------------------
    def find(x):

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):

        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    # ------------------------------------------------------------
    # Build ML components
    # ------------------------------------------------------------
    for a, b in must_link:
        union(a, b)

    roots = np.array([find(i) for i in range(n_samples)])

    _, component = np.unique(
        roots,
        return_inverse=True
    )

    # ------------------------------------------------------------
    # Component-level CL graph
    # ------------------------------------------------------------
    cl_components = defaultdict(set)

    for a, b in cannot_link:

        ca = component[a]
        cb = component[b]

        if ca == cb:
            raise ValueError(
                f"Inconsistent constraints: {a}, {b}"
            )

        cl_components[ca].add(cb)
        cl_components[cb].add(ca)

    return component, cl_components




def constraint_scores_(
    tree,
    compresed_constraints
):

    component, cl_components = compresed_constraints
    scores = np.zeros(tree.N, dtype=np.float64)

    # ------------------------------------------------------------
    # Component sizes
    # ------------------------------------------------------------
    comp_sizes = np.bincount(component)

    # ------------------------------------------------------------
    # Unique CL component pairs
    # ------------------------------------------------------------
    cl_pairs = []

    for c1, nbrs in cl_components.items():

        for c2 in nbrs:

            if c1 < c2:
                cl_pairs.append((c1, c2))

    # ------------------------------------------------------------
    # Total ML constraints
    # ------------------------------------------------------------
    total_ml = np.sum(
        comp_sizes * (comp_sizes - 1) // 2
    )

    # ------------------------------------------------------------
    # Total CL constraints
    # ------------------------------------------------------------
    total_cl = sum(
        comp_sizes[c1] * comp_sizes[c2]
        for c1, c2 in cl_pairs
    )

    total_constraints = total_ml + total_cl

    if total_constraints == 0:
        return scores

    # ------------------------------------------------------------
    # Score each node
    # ------------------------------------------------------------
    for node in range(tree.N):

        obs = tree.get_node_indices(node)

        if len(obs) == 0:
            continue

        counts = Counter(component[obs])

        # --------------------------------------------------------
        # ML satisfaction
        # --------------------------------------------------------
        ml_sat = sum(
            k * (k - 1) 
            for k in counts.values()
        )

        # --------------------------------------------------------
        # CL satisfaction
        # --------------------------------------------------------
        cl_sat = 0

        for c1, c2 in cl_pairs:

            inside1 = counts.get(c1, 0)
            inside2 = counts.get(c2, 0)

            outside1 = comp_sizes[c1] - inside1
            outside2 = comp_sizes[c2] - inside2

            # exactly one endpoint inside
            cl_sat += (
                inside1 * outside2 +
                inside2 * outside1
            )

        scores[node] = (
            ml_sat + cl_sat
        ) / (2*total_constraints)

    return scores