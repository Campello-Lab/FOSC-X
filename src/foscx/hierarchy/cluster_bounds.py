# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:44:17 2025

@author: csimp
"""

import numpy as np
from .._numba import njit

@njit(cache=True)
def _compute_leaf_noise_and_nonnoise_siblings(
    parent: np.ndarray,
    children_flat: np.ndarray,
    children_off: np.ndarray,
    set_leaf_noise: bool,
    is_noise_in: np.ndarray,
):
    """
    Extended behavior:
    - if is_noise_in.size == 0:
        behave exactly like original function
    - if is_noise_in.size > 0 and set_leaf_noise == False:
        use is_noise_in verbatim
    - if is_noise_in.size > 0 and set_leaf_noise == True:
        start from is_noise_in and ADD leaf noise
        (never remove noise)
    """

    N = parent.shape[0]

    is_noise = np.zeros(N, dtype=np.int8)
    nonnoise_sibling_count = np.zeros(N, dtype=np.int32)

    # -------------------------------------------------
    # 1) determine is_noise
    # -------------------------------------------------
    if is_noise_in.size == 0:
        # original behavior
        for i in range(N):
            if children_off[i + 1] == children_off[i]:
                is_noise[i] = 1 if set_leaf_noise else 0
            else:
                is_noise[i] = 0
    else:
        # start from provided noise
        for i in range(N):
            is_noise[i] = is_noise_in[i]

        if set_leaf_noise:
            # add leaf noise, never remove
            for i in range(N):
                if children_off[i + 1] == children_off[i]:
                    if is_noise[i] == 0:
                        is_noise[i] = 1

    # -------------------------------------------------
    # 2) compute non-noise child count PER PARENT
    # -------------------------------------------------
    nonnoise_child_count = np.zeros(N, dtype=np.int32)

    for p in range(N):
        s = children_off[p]
        e = children_off[p + 1]
        if s == e:
            continue
        cnt = 0
        for j in range(s, e):
            ch = children_flat[j]
            if is_noise[ch] == 0:
                cnt += 1
        nonnoise_child_count[p] = cnt

    # -------------------------------------------------
    # 3) assign sibling counts
    # -------------------------------------------------
    for i in range(N):
        p = parent[i]
        if p != -1:
            nonnoise_sibling_count[i] = nonnoise_child_count[p]
        else:
            nonnoise_sibling_count[i] = 0

    return is_noise, nonnoise_sibling_count





@njit(cache=True)
def _compute_leaf_order_and_node_spans(children_flat: np.ndarray,
                                            children_off: np.ndarray,
                                            parent: np.ndarray):
    """
    Compute contiguous leaf order and per-node [start, end) spans over that order.

    Paramaters
    ----
    children_flat : int32 (M,)    - flattened children indices
    children_off  : int32 (N+1,)  - offsets into children_flat
    parent        : int32 (N,)    - parent indices (root: -1)

    Returns
    -------
    leaf_order : int32 (n_leaves,)  - leaf node ids in left->right order
    node_start : int32 (N,)         - inclusive start index into leaf_order
    node_end   : int32 (N,)         - exclusive end index into leaf_order

    Notes
    -----
    - This assumes leaves are genuine leaves (children_off[i+1] - children_off[i] == 0).
    - If a leaf is not encountered in traversal or is missing from the inferred leaf order,
      the function will assert (indicating a data/logic bug upstream).
    """
    N = parent.shape[0]

    # 1) count leaves to size leaf_order
    n_leaves = 0
    for i in range(N):
        if children_off[i + 1] - children_off[i] == 0:
            n_leaves += 1

    # allocate
    leaf_order = np.empty(n_leaves, dtype=np.int32)

    # 2) find roots
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

    # 3) DFS stack to collect leaves left->right
    stack = np.empty(N, dtype=np.int32)
    sp = 0
    for i in range(rcount):
        stack[sp] = roots[i]
        sp += 1

    # temporary buffer for children to push in reverse order
    buf = np.empty(N, dtype=np.int32)
    lo_pos = 0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        s = children_off[node]
        e = children_off[node + 1]
        if e - s == 0:
            # leaf
            leaf_order[lo_pos] = node
            lo_pos += 1
            continue
        # push children in reverse order so leftmost processed first
        cc = 0
        for j in range(s, e):
            buf[cc] = int(children_flat[j])
            cc += 1
        for k in range(cc - 1, -1, -1):
            stack[sp] = buf[k]
            sp += 1

    # safety: lo_pos must equal n_leaves
    assert lo_pos == n_leaves

    # 4) build leaf -> position map
    leaf_pos = -1 * np.ones(N, dtype=np.int32)
    for i in range(n_leaves):
        leaf_pos[int(leaf_order[i])] = i

    # 5) postorder traversal to compute node spans (children before parent)
    st_nodes = np.empty(2 * N, dtype=np.int32)
    st_vis = np.empty(2 * N, dtype=np.int8)
    sp2 = 0
    for i in range(rcount):
        st_nodes[sp2] = roots[i]; st_vis[sp2] = 0; sp2 += 1

    post = np.empty(N, dtype=np.int32)
    plen = 0
    while sp2 > 0:
        sp2 -= 1
        nd = st_nodes[sp2]
        v = st_vis[sp2]
        if v == 1:
            post[plen] = nd
            plen += 1
            continue
        st_nodes[sp2] = nd
        st_vis[sp2] = 1
        sp2 += 1
        s = children_off[nd]
        e = children_off[nd + 1]
        for j in range(s, e):
            st_nodes[sp2] = int(children_flat[j])
            st_vis[sp2] = 0
            sp2 += 1

    node_start = np.empty(N, dtype=np.int32)
    node_end = np.empty(N, dtype=np.int32)

    # compute spans (postorder -> children done first)
    for idx in range(plen):
        node = post[idx]
        s = children_off[node]
        e = children_off[node + 1]
        if e - s == 0:
            # leaf: must have a position
            pos = leaf_pos[node]
            # Assert that leaf was found in leaf_order; if not, signal an upstream bug.
            assert pos != -1
            node_start[node] = pos
            node_end[node] = pos + 1
        else:
            # internal node: span is min(start(children)), max(end(children))
            start = 2**31 - 1
            end = -1
            for j in range(s, e):
                ch = int(children_flat[j])
                cstart = node_start[ch]
                cend = node_end[ch]
                if cstart < start:
                    start = cstart
                if cend > end:
                    end = cend
            # if something weird happens (no children with spans) set empty
            if end < start:
                node_start[node] = 0
                node_end[node] = 0
            else:
                node_start[node] = start
                node_end[node] = end

    return leaf_order, node_start, node_end

@njit(cache=True)
def _compute_bounds(parent: np.ndarray,
                         children_flat: np.ndarray,
                         children_off: np.ndarray,
                         is_noise: np.ndarray):
    """
    Compute LB and UB arrays for every node with the semantics:
      - LB counts only sibling subtrees that contain at least one non-noise leaf.
      - UB uses terminal counts (last non-noise nodes).

    Paramaters
    ----
    parent         : int32 (N,)          - parent indices (root: -1)
    children_flat  : int32 (M,)          - flattened children indices
    children_off   : int32 (N+1,)        - offsets into children_flat
    is_noise       : int8  (N,)          - 1 if node is noise (only leaves may be noise), 0 otherwise

    Returns
    -------
    LB : int32 (N,)   - lower bound for each node (top-down, using non-noise-leaf sibling counts)
    UB : int32 (N,)   - upper bound for each node (bottom-up, using terminal counts)
    terminals_in_subtree : int32 (N,) - number of terminal nodes inside each subtree
    is_terminal : int8 (N,)          - 1 if node is terminal, 0 otherwise
    nonnoise_leaf_count : int32 (N,) - number of non-noise leaves in each subtree
    """
    N = parent.shape[0]

    # -------------------------
    # 1) determine is_terminal
    #    terminal: node is non-noise AND has no child that is non-noise
    # -------------------------
    is_terminal = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if is_noise[i] == 1:
            is_terminal[i] = 0
        else:
            s = children_off[i]
            e = children_off[i + 1]
            has_nonnoise_child = False
            for j in range(s, e):
                ch = int(children_flat[j])
                if is_noise[ch] == 0:
                    has_nonnoise_child = True
                    break
            if has_nonnoise_child:
                is_terminal[i] = 0
            else:
                is_terminal[i] = 1

    # -------------------------
    # 2) postorder traversal to compute:
    #    - terminals_in_subtree (for UB)
    #    - nonnoise_leaf_count (for LB decisions)
    # -------------------------
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

    # build postorder (children before parent)
    st_nodes = np.empty(2 * N, dtype=np.int32)
    st_vis = np.empty(2 * N, dtype=np.int8)
    sp = 0
    for i in range(rcount):
        st_nodes[sp] = roots[i]; st_vis[sp] = 0; sp += 1

    post = np.empty(N, dtype=np.int32)
    plen = 0
    while sp > 0:
        sp -= 1
        nd = st_nodes[sp]
        v = st_vis[sp]
        if v == 1:
            post[plen] = nd
            plen += 1
            continue
        st_nodes[sp] = nd
        st_vis[sp] = 1
        sp += 1
        s = children_off[nd]
        e = children_off[nd + 1]
        for j in range(s, e):
            st_nodes[sp] = int(children_flat[j])
            st_vis[sp] = 0
            sp += 1

    terminals_in_subtree = np.zeros(N, dtype=np.int32)
    nonnoise_leaf_count = np.zeros(N, dtype=np.int32)

    for idx in range(plen):
        node = post[idx]
        s = children_off[node]
        e = children_off[node + 1]

        # sum children's values
        term_sum = 0
        leaf_nonnoise_sum = 0
        for j in range(s, e):
            ch = int(children_flat[j])
            term_sum += terminals_in_subtree[ch]
            leaf_nonnoise_sum += nonnoise_leaf_count[ch]

        # if leaf, check is_noise to count non-noise leaves
        if e - s == 0:
            # leaf node: nonnoise_leaf_count = 1 if this leaf is non-noise, else 0
            if is_noise[node] == 0:
                leaf_nonnoise_sum = 1
            else:
                leaf_nonnoise_sum = 0

        # add self if terminal
        if is_terminal[node] == 1:
            term_sum += 1

        terminals_in_subtree[node] = term_sum
        nonnoise_leaf_count[node] = leaf_nonnoise_sum

    # compute total terminals in whole forest (sum over roots to avoid double counting)
    total_terminals = 0
    for i in range(rcount):
        total_terminals += terminals_in_subtree[roots[i]]
    # fallback (shouldn't be needed) - if zero, fall back to sum of is_terminal
    if total_terminals == 0:
        tmp = 0
        for i in range(N):
            tmp += is_terminal[i]
        total_terminals = tmp

    # -------------------------
    # 3) UB (bottom-up)
    #    UB[node] = total_terminals - terminals_in_subtree[node]
    # -------------------------
    UB = np.empty(N, dtype=np.int32)
    for i in range(N):
        UB[i] = total_terminals - terminals_in_subtree[i]

    # -------------------------
    # 4) LB (top-down) using nonnoise_leaf_count
    #    For child c of parent p:
    #      LB[c] = LB[p] + ( number_of_siblings_with_nonnoise_leaves
    #                        - (1 if c_subtree_has_nonnoise_leaf else 0) )
    #    Root(s) LB = 0
    # -------------------------
    LB = np.empty(N, dtype=np.int32)
    for i in range(N):
        if parent[i] == -1:
            LB[i] = 0

    # preorder traversal: stack from roots, push children in reverse so left-to-right order
    stack = np.empty(N, dtype=np.int32)
    sp2 = 0
    for i in range(rcount):
        stack[sp2] = roots[i]; sp2 += 1

    # temp buffer
    buf = np.empty(N, dtype=np.int32)

    while sp2 > 0:
        sp2 -= 1
        node = stack[sp2]
        s = children_off[node]
        e = children_off[node + 1]
        if e - s == 0:
            continue

        # count how many children have at least one non-noise leaf
        children_with_nonnoise_leaves = 0
        cc = 0
        for j in range(s, e):
            ch = int(children_flat[j])
            buf[cc] = ch
            if nonnoise_leaf_count[ch] > 0:
                children_with_nonnoise_leaves += 1
            cc += 1

        # assign LB for each child and push onto stack (reverse order)
        for j in range(cc):
            ch = buf[j]
            exclude = 1 if nonnoise_leaf_count[ch] > 0 else 0
            LB[ch] = LB[node] + (children_with_nonnoise_leaves - exclude)

        for k in range(cc - 1, -1, -1):
            stack[sp2] = buf[k]
            sp2 += 1

    return LB, UB, terminals_in_subtree, is_terminal, nonnoise_leaf_count