from .._numba import njit
import numpy as np
from typing import Optional, Dict, Tuple, Union

@njit(cache=True)
def _bsearch(arr, val):
    """
    Return index i such that arr[i] == val. Assumes arr sorted ascending and val exists.
    """
    lo = 0
    hi = arr.shape[0] - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        mval = arr[mid]
        if mval == val:
            return mid
        elif mval < val:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1  # should not happen for valid inputs


# ---------- main numba-only builder ----------
@njit(cache=True)
def _build_from_arrays(parent_old, child_old, lambda_arr, child_size_arr, child_size_present, density):
    """
    All arguments are 1-D numpy arrays (parent_old,int64), (child_old,int64),
    (lambda_arr,float64), (child_size_arr,int32 or empty), a bool flag child_size_present,
    and density: bool.  Returns: parent_out (int32), distance_out (float64),
             children_flat (int32), children_off (int32), sizes_out (int32)
    """
    # 1) basic shapes
    n_rows = parent_old.shape[0]

    # 2) compute max id and build presence bitmaps to infer leaves
    max_id = int(parent_old[0])
    # find max id
    for i in range(1, n_rows):
        if parent_old[i] > max_id:
            max_id = int(parent_old[i])
        if child_old[i] > max_id:
            max_id = int(child_old[i])
    if child_old[0] > max_id:
        max_id = int(child_old[0])

    # allocate presence arrays of length max_id+1
    size_map = max_id + 1
    parent_present = np.zeros(size_map, dtype=np.uint8)
    child_present = np.zeros(size_map, dtype=np.uint8)

    for i in range(n_rows):
        p = int(parent_old[i])
        c = int(child_old[i])
        parent_present[p] = 1
        child_present[c] = 1

    # leaf ids are child ids that never appear as parent
    # find max leaf id to infer n_leaves = max_leaf + 1
    max_leaf = -1
    for idv in range(size_map):
        if child_present[idv] == 1 and parent_present[idv] == 0:
            if idv > max_leaf:
                max_leaf = idv
    # assume there is at least one leaf (well-formed input)
    n_leaves = max_leaf + 1

    # 3) build ordered_old_ids: leaves 0..n_leaves-1, then cluster ids >= n_leaves in ascending order
    # first count clusters to allocate array
    # cluster ids are any id >= n_leaves that appear as parent or child
    cluster_count = 0
    for idv in range(n_leaves, size_map):
        if (parent_present[idv] == 1) or (child_present[idv] == 1):
            cluster_count += 1

    N = n_leaves + cluster_count
    ordered_old_ids = np.empty(N, dtype=np.int64)
    # fill leaves 0..n_leaves-1
    for i in range(n_leaves):
        ordered_old_ids[i] = i
    # fill clusters ascending
    pos = n_leaves
    for idv in range(n_leaves, size_map):
        if (parent_present[idv] == 1) or (child_present[idv] == 1):
            ordered_old_ids[pos] = idv
            pos += 1

    # 4) map parent_old/child_old -> parent_idx/child_idx using binary search per element
    parent_idx = np.empty(n_rows, dtype=np.int32)
    child_idx = np.empty(n_rows, dtype=np.int32)
    for r in range(n_rows):
        p_old = int(parent_old[r])
        c_old = int(child_old[r])
        pnew = _bsearch(ordered_old_ids, p_old)
        cnew = _bsearch(ordered_old_ids, c_old)
        parent_idx[r] = pnew
        child_idx[r] = cnew

    # 5) counts per parent -> children_off
    counts = np.zeros(N, dtype=np.int32)
    for r in range(n_rows):
        counts[parent_idx[r]] += 1
    children_off = np.empty(N + 1, dtype=np.int32)
    children_off[0] = 0
    s = 0
    for i in range(N):
        s += counts[i]
        children_off[i+1] = s
    total_children = int(children_off[N])

    # 6) allocate children_flat and other outputs
    children_flat = np.empty(total_children, dtype=np.int32)
    parent_out = -1 * np.ones(N, dtype=np.int32)
    distance_out = np.zeros(N, dtype=np.float64)
    sizes_out = np.zeros(N, dtype=np.int32)

    # if no child_size provided, set leaves sizes to 1
    if not child_size_present:
        for i in range(n_leaves):
            sizes_out[i] = 1

    # 7) write rows into children_flat using per-parent write_pos buffer
    write_pos = np.empty(N, dtype=np.int32)
    for i in range(N):
        write_pos[i] = children_off[i]

    for r in range(n_rows):
        p = int(parent_idx[r])
        c = int(child_idx[r])
        w = write_pos[p]
        children_flat[w] = c
        write_pos[p] = w + 1
        parent_out[c] = p

        # assign distance depending on mode:
        if density:
            # density mode: lambda values assigned to the child node (same as original behaviour)
            distance_out[c] = lambda_arr[r]
        else:
            # linkage mode: lambda_arr carries parent distance -> assign to parent index
            distance_out[p] = lambda_arr[r]

        if child_size_present:
            sizes_out[c] = int(child_size_arr[r])

    # 8) ensure leaf sizes at least 1 (in case some leaf rows didn't set them)
    for i in range(n_leaves):
        if sizes_out[i] == 0:
            sizes_out[i] = 1

    # 9) set root size to total leaves if root has zero size (root = node with parent == -1)
    # find first parent == -1 (assume exactly one)
    root_idx = -1
    for i in range(N):
        if parent_out[i] == -1:
            root_idx = i
            break
    if root_idx >= 0:
        total_leaves = 0
        for i in range(n_leaves):
            total_leaves += sizes_out[i]
        if sizes_out[root_idx] == 0:
            sizes_out[root_idx] = total_leaves
        # For density mode keep previous behaviour: set root distance explicitly to 0.0
        # For linkage mode the parent distance was assigned above from lambda_arr (so preserve it).
        if density:
            distance_out[root_idx] = 0.0

    # For linkage mode, ensure leaves have distance 0.0 (explicit)
    if not density:
        for i in range(n_leaves):
            distance_out[i] = 0.0

    # cast outputs in-place already correct dtypes used above
    return parent_out, distance_out, children_flat, children_off, sizes_out


# ---------- Python wrapper that accepts structured ndarray or 2D array ----------
def _build_hierarchy(condensed: Union[np.ndarray],
                               col_idx: Optional[Dict[str,int]] = None,
                               field_names: Optional[Dict[str,str]] = None,
                               density: bool = True
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper that accepts:
      - a structured 1-D numpy array (dtype with names like 'parent','child','lambda_val','child_size')
      - or a 2-D numeric ndarray with columns in order given by col_idx

    It extracts the plain arrays and calls the numba-compiled implementation.

    New: density bool flag. If False, treat the third column as linkage parent-distance (d),
         otherwise treat it as lambda (1/d).
    """
    if col_idx is None:
        col_idx = {'parent': 0, 'child': 1, 'lambda': 2, 'child_size': 3}
    if field_names is None:
        field_names = {'parent': 'parent', 'child': 'child', 'lambda': 'lambda_val', 'child_size': 'child_size'}

    # handle structured array
    if isinstance(condensed, np.ndarray) and condensed.dtype.names is not None:
        names = condensed.dtype.names
        parent_old = condensed[field_names['parent']].astype(np.int64)
        child_old = condensed[field_names['child']].astype(np.int64)
        if field_names.get('lambda') in names:
            lambda_arr = condensed[field_names['lambda']].astype(np.float64)
        else:
            # fallback if named differently
            lambda_arr = condensed['lambda_val'].astype(np.float64) if 'lambda_val' in names else np.zeros(parent_old.shape[0], dtype=np.float64)
        if field_names.get('child_size') in names:
            child_size_arr = np.asarray(condensed[field_names['child_size']]).astype(np.int32)
            child_size_present = True
        else:
            child_size_arr = np.empty(0, dtype=np.int32)
            child_size_present = False
    else:
        # 2D numeric
        if not isinstance(condensed, np.ndarray) or condensed.ndim != 2:
            raise ValueError("condensed must be structured 1D ndarray or 2D numeric ndarray")
        n_rows, n_cols = condensed.shape
        def col_ok(idx):
            return (idx is not None) and (0 <= idx < n_cols)
        parent_old = condensed[:, col_idx['parent']].astype(np.int64)
        child_old = condensed[:, col_idx['child']].astype(np.int64)
        lambda_arr = condensed[:, col_idx['lambda']].astype(np.float64) if col_ok(col_idx.get('lambda', None)) else np.zeros(parent_old.shape[0], dtype=np.float64)
        if col_ok(col_idx.get('child_size', None)):
            child_size_arr = condensed[:, col_idx['child_size']].astype(np.int32)
            child_size_present = True
        else:
            child_size_arr = np.empty(0, dtype=np.int32)
            child_size_present = False

    # call numba core with density flag
    return _build_from_arrays(parent_old, child_old, lambda_arr, child_size_arr, child_size_present, density)

# ---------- dictionary-based tree to FastHierarchy converter ----------
def _dict_to_hierarchy(tree: dict,complete_tree: bool = False):
    """
    Convert dictionary-based tree into FastHierarchy flat arrays.

    Returns
    -------
    parent : int32 (N,)
    distance : float64 (N,)
    children_flat : int32 (M,)
    children_off : int32 (N+1,)
    sizes : int32 (N,)
    clusteval : float64 (N,)
    is_noise : int8 (N,)
    id_map : dict {orig_id -> compact_id}
    rev_id_map : list {compact_id -> orig_id}
    """

    # -------------------------
    # 1. Stable ID mapping
    # -------------------------
    orig_ids = list(tree.keys())
    N = len(orig_ids)

    id_map = {oid: i for i, oid in enumerate(orig_ids)}
    rev_id_map = orig_ids[:]

    # -------------------------
    # 2. Allocate arrays
    # -------------------------
    parent = np.full(N, -1, dtype=np.int32)
    distance = np.full(N, np.nan, dtype=np.float64)
    sizes = np.zeros(N, dtype=np.int32)

    clusteval = np.zeros(N, dtype=np.float64)
    is_noise = np.zeros(N, dtype=np.int8)

    children_lists = [[] for _ in range(N)]
    root_count = 0

    # -------------------------
    # 3. Populate arrays
    # -------------------------
    for orig_id, node in tree.items():
        cid = id_map[orig_id]

        # ----- parent -----
        p = node.get("parent", "")
        if p in ("", None):
            parent[cid] = -1
            root_count += 1
        else:
            parent[cid] = id_map[p]

        # ----- children -----
        for ch in node.get("children", []):
            children_lists[cid].append(id_map[ch])

        # ----- distance -----
        if "distance" in node:
            d = node["distance"]
            if isinstance(d, (list, tuple)):
                if len(d) > 0:
                    distance[cid] = float(d[0])
            else:
                distance[cid] = float(d)

        # ----- size -----
        if "cluster_size" in node:
            sizes[cid] = int(node["cluster_size"])

        # ----- clusteval -----
        if "clusteval" in node:
            clusteval[cid] = float(node["clusteval"])

        # ----- noise -----
        if "noise" in node:
            is_noise[cid] = 1 if bool(node["noise"]) else 0

    # -------------------------
    # 4. Validate single root
    # -------------------------
    if root_count != 1:
        raise ValueError(f"Tree must have exactly one root, found {root_count}")

    # -------------------------
    # 5. Build children_flat / children_off
    # -------------------------
    counts = np.array([len(ch) for ch in children_lists], dtype=np.int32)
    children_off = np.empty(N + 1, dtype=np.int32)
    children_off[0] = 0
    np.cumsum(counts, out=children_off[1:])

    M = int(children_off[-1])
    children_flat = np.empty(M, dtype=np.int32)

    pos = 0
    for i in range(N):
        ch = children_lists[i]
        if ch:
            children_flat[pos:pos + len(ch)] = ch
            pos += len(ch)

    # -------------------------
    # 6. Infer missing sizes bottom-up
    # -------------------------
    if complete_tree:
        # For complete trees, infer sizes bottom-up
        stack = []
        root = int(np.where(parent == -1)[0][0])
        stack.append((root, False))

        while stack:
            node, done = stack.pop()
            if done:
                if counts[node] > 0 and sizes[node] == 0:
                    sizes[node] = sum(sizes[ch] for ch in children_lists[node])
            else:
                stack.append((node, True))
                for ch in children_lists[node]:
                    stack.append((ch, False))
    



    return (
        parent,
        distance,
        children_flat,
        children_off,
        sizes,
        clusteval,
        is_noise,
        id_map,
        rev_id_map,
    )
