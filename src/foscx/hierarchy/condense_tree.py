# Functions adapted from fast_hdbscan package

import numpy as np
from .._numba import njit

@njit(cache=True)
def _bfs_from_hierarchy(hierarchy, bfs_root, num_points):
    """
    BFS over the binary hierarchy representation used by the project.
    Returns a Python list of visited node ids in BFS order (root first).
    This mirrors the original helper you used.
    """
    to_process = [bfs_root]
    result = []

    while to_process:
        # extend result with current frontier
        for v in to_process:
            result.append(v)
        next_to_process = []
        for n in to_process:
            if n >= num_points:
                i = n - num_points
                next_to_process.append(int(hierarchy[i, 0]))
                next_to_process.append(int(hierarchy[i, 1]))
        to_process = next_to_process

    return result


@njit(cache=True)
def _eliminate_branch(branch_node, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore, hierarchy,
                     num_points, density, min_cluster_size):
    """
    Attach leaves from branch_node subtree to parent_node.
    - density==True: attached leaves inherit lambda_value (1/d) as before.
    - density==False: attached leaves inherit parent's *distance* (lambda_value == d),
      so all children under the same parent share the same value.
    """
    if branch_node < num_points:
        parents[idx] = parent_node
        children[idx] = branch_node
        if density:
            lambdas[idx] = lambda_value
        else:
            lambdas[idx] = lambda_value
        idx += 1
    else:
        for sub_node in _bfs_from_hierarchy(hierarchy, branch_node, num_points):
            if sub_node < num_points:
                parents[idx] = parent_node
                children[idx] = sub_node
                if density:
                    lambdas[idx] = lambda_value
                else:
                    lambdas[idx] = lambda_value
                idx += 1
            else:
                ignore[sub_node] = True

    return idx



@njit(cache=True)
def _condense_tree(hierarchy, min_cluster_size=10, max_cluster_size=np.inf, sample_weights=None, density=True):
    """
    Condense a linkage/hdbscan-style hierarchy.
    hierarchy: (n_internal, 4) array: [left, right, dist, size]
    - If density==True: behaves like HDBSCAN condense (lambda = 1/d)
    - If density==False: behaves like linkage-style condense (lambda = parent's d for all children)
    Returns: parents_arr, children_arr, lambdas_arr, sizes_arr
    """
    root = 2 * hierarchy.shape[0]
    num_points = hierarchy.shape[0] + 1
    next_label = num_points + 1

    node_list = _bfs_from_hierarchy(hierarchy, root, num_points)

    relabel = np.zeros(root + 1, dtype=np.int64)
    relabel[root] = num_points

    parents = np.ones(root, dtype=np.int64)
    children = np.empty(root, dtype=np.int64)
    lambdas = np.empty(root, dtype=np.float64)
    sizes = np.ones(root, dtype=np.float32)

    ignore = np.zeros(root + 1, dtype=np.bool_)

    if sample_weights is None:
        sample_weights = np.ones(num_points, dtype=np.float32)

    idx = 0

    for node in node_list:
        if ignore[node] or node < num_points:
            continue

        parent_node = relabel[node]
        l, r, d, _ = hierarchy[node - num_points]
        left = np.int64(l)
        right = np.int64(r)

        # Compute the value we will record for this node depending on mode
        if density:
            if d > 0.0:
                lambda_value = 1.0 / d
            else:
                lambda_value = np.inf
        else:
            # linkage: default to parent distance
            lambda_value = d
            # precompute child distances (needed if one child survives)
            left_d  = hierarchy[left  - num_points, 2] if left  >= num_points else d
            right_d = hierarchy[right - num_points, 2] if right >= num_points else d


        left_count = np.float32(hierarchy[left - num_points, 3]) if left >= num_points else sample_weights[left]
        right_count = np.float32(hierarchy[right - num_points, 3]) if right >= num_points else sample_weights[right]

        # Fast-paths kept in original order for performance
        
        if left < num_points and right_count >= min_cluster_size:
            relabel[right] = parent_node
            parents[idx] = parent_node
            children[idx] = left
            if not density:
                lambda_value = right_d 
            lambdas[idx] = lambda_value   # note: same parent lambda_value used for this child.
            idx += 1

        elif left_count < min_cluster_size and right_count >= min_cluster_size:
            relabel[right] = parent_node
            if not density:
                lambda_value = right_d 
            idx = _eliminate_branch(left, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                   hierarchy, num_points, density, min_cluster_size)

        elif left_count >= min_cluster_size and right_count < min_cluster_size:
            relabel[left] = parent_node
            if not density:
                lambda_value = left_d
            idx = _eliminate_branch(right, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                   hierarchy, num_points, density, min_cluster_size)

        elif left_count < min_cluster_size and right_count < min_cluster_size:
            if density:
                idx = _eliminate_branch(left, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                    hierarchy, num_points, density, min_cluster_size)
                idx = _eliminate_branch(right, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                    hierarchy, num_points, density, min_cluster_size)
            else:
                idx = _eliminate_branch(
                left, parent_node, 0.0,
                parents, children, lambdas, sizes,
                idx, ignore, hierarchy, num_points,
                density, min_cluster_size
                )
                idx = _eliminate_branch(
                    right, parent_node, 0.0,
                    parents, children, lambdas, sizes,
                    idx, ignore, hierarchy, num_points,
                    density, min_cluster_size
                )

        elif density and left_count > max_cluster_size and right_count > max_cluster_size:
            relabel[left] = parent_node
            relabel[right] = parent_node

        else:
            # Default behaviour: create synthetic children. Use parent's lambda_value for linkage too.
            relabel[left] = next_label
            parents[idx] = parent_node
            children[idx] = next_label
            lambdas[idx] = lambda_value
            sizes[idx] = left_count
            next_label += 1
            idx += 1

            relabel[right] = next_label
            parents[idx] = parent_node
            children[idx] = next_label
            lambdas[idx] = lambda_value
            sizes[idx] = right_count
            next_label += 1
            idx += 1

    return parents[:idx], children[:idx], lambdas[:idx], sizes[:idx]


@njit(cache=True)
def _scipy_to_condensed(hierarchy):
    """
    Convert SciPy linkage `hierarchy` (n_internal x 4: left,right,dist,size)
    to condensed-style arrays, relabeling internal nodes to contiguous ids
    starting at num_points (N), N+1, ...
    Returns:
        parents_arr, children_arr, distances_arr, child_sizes_arr
    - Leaves (0..num_points-1) preserved.
    - Internal nodes remapped to labels starting at num_points in BFS order.
    - distances record the parent's raw linkage distance.
    - child_sizes: 1.0 for leaves, hierarchy[child - num_points, 3] for internals.
    """
    n_internal = hierarchy.shape[0]
    if n_internal == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float32))

    num_points = n_internal + 1
    root = 2 * n_internal

    # BFS ordering from existing helper (must exist in same file)
    node_list = _bfs_from_hierarchy(hierarchy, root, num_points)
    node_list_len = len(node_list)

    # relabel map (indexable by node id). -1 means not an internal (or not assigned yet)
    relabel = -1 * np.ones(root + 1, dtype=np.int64)

    # Assign compact labels to all internal nodes encountered in BFS order
    next_label = num_points
    for i in range(node_list_len):
        node = node_list[i]
        if node >= num_points:
            relabel[node] = next_label
            next_label += 1

    # Preallocate output arrays: two edges per internal node
    cap = max(4, n_internal * 2)
    parents = np.empty(cap, dtype=np.int64)
    children = np.empty(cap, dtype=np.int64)
    distances = np.empty(cap, dtype=np.float64)
    child_sizes = np.empty(cap, dtype=np.float32)
    out_idx = 0

    # Second pass: emit edges for each internal node in BFS order
    for i in range(node_list_len):
        node = node_list[i]
        # skip leaves
        if node < num_points:
            continue

        parent_label = relabel[node]
        hidx = node - num_points
        left = int(hierarchy[hidx, 0])
        right = int(hierarchy[hidx, 1])
        d = float(hierarchy[hidx, 2])

        # left child edge
        parents[out_idx] = parent_label
        if left < num_points:
            children[out_idx] = left
            child_sizes[out_idx] = 1.0
        else:
            children[out_idx] = relabel[left]
            child_sizes[out_idx] = float(hierarchy[left - num_points, 3])
        distances[out_idx] = d
        out_idx += 1

        # right child edge
        parents[out_idx] = parent_label
        if right < num_points:
            children[out_idx] = right
            child_sizes[out_idx] = 1.0
        else:
            children[out_idx] = relabel[right]
            child_sizes[out_idx] = float(hierarchy[right - num_points, 3])
        distances[out_idx] = d
        out_idx += 1

    # Return slices (copied) for safe use outside numba
    return parents[:out_idx].copy(), children[:out_idx].copy(), distances[:out_idx].copy(), child_sizes[:out_idx].copy()

@njit(cache=True)
def _scipy_to_condensed_2(hierarchy):

    n_internal = hierarchy.shape[0]

    if n_internal == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float32),
        )

    num_points = n_internal + 1

    parents = np.empty(2 * n_internal, dtype=np.int64)
    children = np.empty(2 * n_internal, dtype=np.int64)
    distances = np.empty(2 * n_internal, dtype=np.float64)
    child_sizes = np.empty(2 * n_internal, dtype=np.float32)

    out = 0

    for i in range(n_internal):

        parent = num_points + i

        left = int(hierarchy[i, 0])
        right = int(hierarchy[i, 1])

        dist = float(hierarchy[i, 2])

        for child in (left, right):

            parents[out] = parent
            children[out] = child
            distances[out] = dist

            if child < num_points:
                child_sizes[out] = 1.0
            else:
                child_sizes[out] = hierarchy[child - num_points, 3]

            out += 1

    return parents, children, distances, child_sizes