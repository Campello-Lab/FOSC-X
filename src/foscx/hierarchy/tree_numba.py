
import numpy as np
from numba import njit


# ---------------------------
# Fast utilities (numba-ready)
# ---------------------------

@njit(cache=True)
def _compute_sizes(parent: np.ndarray,
                        children_flat: np.ndarray,
                        children_off: np.ndarray,
                        sizes: np.ndarray) -> np.ndarray:
    """
    Postorder bottom-up sizes (leaves assumed size >=1).
    Modifies sizes in-place and returns it.
    parent: int32 (N,)
    children_flat: int32 (M,)
    children_off: int32 (N+1,)
    sizes: int32 (N,) (may be prefilled with leaves =1 or zeros)
    """
    N = parent.shape[0]

    # find roots
    roots_cnt = 0
    for i in range(N):
        if parent[i] == -1:
            roots_cnt += 1
    roots = np.empty(roots_cnt, dtype=np.int32)
    ri = 0
    for i in range(N):
        if parent[i] == -1:
            roots[ri] = i
            ri += 1

    # iterative postorder stack: store node and visited flag
    stack_node = np.empty(2 * N, dtype=np.int32)
    stack_vis = np.empty(2 * N, dtype=np.int8)
    sp = 0
    for r in range(roots_cnt):
        stack_node[sp] = roots[r]
        stack_vis[sp] = 0
        sp += 1

    post = np.empty(N, dtype=np.int32)
    post_len = 0

    while sp > 0:
        sp -= 1
        node = stack_node[sp]
        vis = stack_vis[sp]
        if vis == 1:
            post[post_len] = node
            post_len += 1
            continue
        # push visited
        stack_node[sp] = node
        stack_vis[sp] = 1
        sp += 1
        # push children
        s = children_off[node]
        e = children_off[node + 1]
        for j in range(s, e):
            c = int(children_flat[j])
            stack_node[sp] = c
            stack_vis[sp] = 0
            sp += 1

    # compute sizes bottom-up
    for idx in range(post_len):
        node = post[idx]
        s = children_off[node]
        e = children_off[node + 1]
        if e - s == 0:
            if sizes[node] == 0:
                sizes[node] = 1
        else:
            tot = 0
            for j in range(s, e):
                c = int(children_flat[j])
                tot += int(sizes[c])
            sizes[node] = int(tot)
    return sizes


@njit(cache=True)
def _compute_depths(parent: np.ndarray,
                         children_flat: np.ndarray,
                         children_off: np.ndarray) -> np.ndarray:
    """
    Compute depths (root depth = 0). Returns int32 array (N,)
    """
    N = parent.shape[0]
    depths = -1 * np.ones(N, dtype=np.int32)
    # find roots
    rcount = 0
    for i in range(N):
        if parent[i] == -1:
            rcount += 1
    roots = np.empty(rcount, dtype=np.int32)
    ri = 0
    for i in range(N):
        if parent[i] == -1:
            roots[ri] = i
            ri += 1
            depths[i] = 0

    # stack
    stack = np.empty(N, dtype=np.int32)
    sp = 0
    for i in range(rcount):
        stack[sp] = roots[i]
        sp += 1

    while sp > 0:
        sp -= 1
        node = stack[sp]
        d = depths[node]
        s = children_off[node]
        e = children_off[node + 1]
        for j in range(s, e):
            c = int(children_flat[j])
            if depths[c] == -1:
                depths[c] = d + 1
                stack[sp] = c
                sp += 1
    return depths


@njit(cache=True)
def _compute_stability(parent: np.ndarray,
                            children_flat: np.ndarray,
                            children_off: np.ndarray,
                            distance: np.ndarray,
                            sizes: np.ndarray,
                            density: bool) -> np.ndarray:
    """
    Compute clusteval / stability. Matches compute_stability semantics in Tree.py.
    Returns float64 (N,)
    """
    N = parent.shape[0]
    stability = np.zeros(N, dtype=np.float64)
    dist = distance.astype(np.float64)

    if not density:
        for node in range(N):
            node_d = float(dist[node])
            p = int(parent[node])
            if p == -1:
                lifetime = 0.0
            else:
                parent_d = float(dist[p])
                lifetime = parent_d - node_d
                # small numeric tolerance
                if lifetime < 0.0 and lifetime > -1e-12:
                    lifetime = 0.0
            stability[node] = float(sizes[node]) * float(lifetime)
        return stability

    # density mode
    lambda_vals = dist.copy()
    for v in range(N):
        s_idx = children_off[v]
        e_idx = children_off[v + 1]
        if e_idx <= s_idx:
            stability[v] = 0.0
            continue
        lam_v = float(lambda_vals[v])
        ssum = 0.0
        for j in range(s_idx, e_idx):
            c = int(children_flat[j])
            lam_c = float(lambda_vals[c])
            delta = lam_c - lam_v
            if not np.isfinite(delta):
                # follow Tree.py logic: if lam_c is +inf and lam_v finite => +inf (keeps inf)
                if np.isposinf(lam_c) and (not np.isposinf(lam_v)):
                    delta = np.inf
                else:
                    delta = 0.0
            if delta > 0.0:
                ssum += float(sizes[c]) * float(delta)
        stability[v] = ssum
    return stability


# ---------------------------
# Ancestor table / LCA (numba)
# ---------------------------

@njit(cache=True)
def _build_ancestor_table(parent: np.ndarray) -> np.ndarray:
    """
    Return up: shape (K, N) int32 where up[k, v] = 2^k-th ancestor or -1.
    """
    N = parent.shape[0]
    # compute depths to get max depth
    # (we call compute_depths_numba building on parent/children requires children arrays;
    #  but here we can approximate K by ceil(log2(N+1)) which is safe upper bound)
    # Use K = ceil(log2(N+1))
    K = 1
    ntemp = N
    while (1 << K) <= ntemp:
        K += 1
    up = -1 * np.ones((K, N), dtype=np.int32)
    # level 0 is parent
    for i in range(N):
        up[0, i] = parent[i]
    for k in range(1, K):
        for v in range(N):
            prev = up[k - 1, v]
            if prev == -1:
                up[k, v] = -1
            else:
                up[k, v] = up[k - 1, prev]
    return up


@njit(cache=True)
def _kth_ancestor(up: np.ndarray, node: int, k: int) -> int:
    """
    up: ancestor table (K,N). Return k-th ancestor of node or -1.
    """
    if k <= 0:
        return node
    cur = node
    bit = 0
    while k and cur != -1:
        if (k & 1):
            # if bit >= up.shape[0] -> -1
            if bit >= up.shape[0]:
                return -1
            cur = int(up[bit, cur])
            if cur == -1:
                return -1
        k >>= 1
        bit += 1
    return cur


@njit(cache=True)
def _lca(up: np.ndarray, depths: np.ndarray, a: int, b: int) -> int:
    """
    LCA using ancestor table up and depths array.
    """
    if a == b:
        return a
    if depths[a] < depths[b]:
        tmp = a
        a = b
        b = tmp
    diff = int(depths[a] - depths[b])
    bit = 0
    while diff:
        if diff & 1:
            a = int(up[bit, a])
        diff >>= 1
        bit += 1
    if a == b:
        return a
    K = up.shape[0]
    for k in range(K - 1, -1, -1):
        pa = int(up[k, a])
        pb = int(up[k, b])
        if pa != pb:
            a = pa
            b = pb
    # now parent[a] == parent[b] is LCA
    return int(up[0, a])


@njit(cache=True)
def _postorder(parent, children_flat, children_off):
    """
    Return nodes in postorder for the whole forest (children before parent).
    """
    N = parent.shape[0]
    # collect roots
    rc = 0
    for i in range(N):
        if parent[i] == -1:
            rc += 1
    roots = np.empty(rc, dtype=np.int32)
    ri = 0
    for i in range(N):
        if parent[i] == -1:
            roots[ri] = i; ri += 1

    st = np.empty(2 * N, dtype=np.int32)
    vis = np.empty(2 * N, dtype=np.int8)
    sp = 0
    for i in range(rc):
        st[sp] = roots[i]; vis[sp] = 0; sp += 1

    post = np.empty(N, dtype=np.int32)
    plen = 0
    while sp > 0:
        sp -= 1
        nd = st[sp]; v = vis[sp]
        if v == 1:
            post[plen] = nd; plen += 1
            continue
        st[sp] = nd; vis[sp] = 1; sp += 1
        s = children_off[nd]; e = children_off[nd+1]
        for j in range(s, e):
            st[sp] = int(children_flat[j]); vis[sp] = 0; sp += 1

    return post[:plen]

