from ._numba import njit
import numpy as np

_NEG_INF = -1e300


@njit(cache=True)
def _merge_children_pruned(
    children,
    m,
    node_cand_off,
    node_cand_count,
    cand_scores,
    cand_ncs,
    LB_node,
    UB_node,
    kmin,
    kmax,
    top_c
):
    """
    Safe pruned Cartesian merge across all children.

    Returns:
        cur_len,
        cur_scores,
        cur_ncs,
        cur_choices   (shape = [cur_len, m])
    """

    # ---- Feasible final nc band ----
    nc_min = kmin - UB_node
    nc_max = kmax - LB_node

    # ---- Precompute per-child min/max nc ----
    child_min_nc = np.empty(m, dtype=np.int64)
    child_max_nc = np.empty(m, dtype=np.int64)

    for ii in range(m):
        ch = children[ii]
        off = node_cand_off[ch]
        cnt = node_cand_count[ch]

        mn = cand_ncs[off]
        mx = cand_ncs[off]

        for j in range(1, cnt):
            v = cand_ncs[off + j]
            if v < mn:
                mn = v
            if v > mx:
                mx = v

        child_min_nc[ii] = mn
        child_max_nc[ii] = mx

    # ---- Initialise with first child ----
    first = children[0]
    offA = node_cand_off[first]
    lenA = node_cand_count[first]

    cur_len = lenA
    cur_scores = np.empty(cur_len, dtype=np.float64)
    cur_ncs = np.empty(cur_len, dtype=np.int64)
    cur_choices = np.empty((cur_len, 1), dtype=np.int64)

    for i in range(lenA):
        cur_scores[i] = cand_scores[offA + i]
        cur_ncs[i] = cand_ncs[offA + i]
        cur_choices[i, 0] = offA + i

    # ---- Merge remaining children ----
    for child_idx in range(1, m):

        ch = children[child_idx]
        offB = node_cand_off[ch]
        lenB = node_cand_count[ch]

        # Precompute remaining bounds AFTER this child
        min_rem = 0
        max_rem = 0
        for r in range(child_idx + 1, m):
            min_rem += child_min_nc[r]
            max_rem += child_max_nc[r]

        # Collect B arrays
        B_ncs = np.empty(lenB, dtype=np.int64)
        B_scores = np.empty(lenB, dtype=np.float64)
        B_ids = np.empty(lenB, dtype=np.int64)

        for j in range(lenB):
            B_ncs[j] = cand_ncs[offB + j]
            B_scores[j] = cand_scores[offB + j]
            B_ids[j] = offB + j

        # Sort B by nc ascending (for searchsorted)
        orderB = np.argsort(B_ncs)
        B_ncs_sorted = B_ncs[orderB]

        # First pass: count valid pairs safely
        valid = 0

        for ia in range(cur_len):
            na = cur_ncs[ia]

            # safe partial prune (before B)
            if na + child_max_nc[child_idx] + max_rem < nc_min:
                continue
            if na + child_min_nc[child_idx] + min_rem > nc_max:
                continue

            lo = np.searchsorted(B_ncs_sorted, nc_min - na - max_rem, side='left')
            hi = np.searchsorted(B_ncs_sorted, nc_max - na - min_rem, side='right')

            if lo < hi:
                valid += (hi - lo)

        if valid == 0:
            return (np.int64(0),np.empty(0, dtype=np.float64),np.empty(0, dtype=np.int64),np.empty((0, m), dtype=np.int64))


        tmp_scores = np.empty(valid, dtype=np.float64)
        tmp_ncs = np.empty(valid, dtype=np.int64)
        tmp_choices = np.empty((valid, child_idx + 1), dtype=np.int64)
        idx_tmp = 0

        for ia in range(cur_len):
            sa = cur_scores[ia]
            na = cur_ncs[ia]

            if na + child_max_nc[child_idx] + max_rem < nc_min:
                continue
            if na + child_min_nc[child_idx] + min_rem > nc_max:
                continue

            lo = np.searchsorted(B_ncs_sorted, nc_min - na - max_rem, side='left')
            hi = np.searchsorted(B_ncs_sorted, nc_max - na - min_rem, side='right')

            for k in range(lo, hi):
                jb = orderB[k]
                nb = B_ncs[jb]

                nc_total = na + nb

                # final safety check
                if nc_total + min_rem > nc_max:
                    continue
                if nc_total + max_rem < nc_min:
                    continue

                tmp_scores[idx_tmp] = sa + B_scores[jb]
                tmp_ncs[idx_tmp] = nc_total

                for c in range(child_idx):
                    tmp_choices[idx_tmp, c] = cur_choices[ia, c]

                tmp_choices[idx_tmp, child_idx] = B_ids[jb]

                idx_tmp += 1

        # ---- Per-nc cap ----
        # ---- FAST Per-nc cap (correct semantics) ----

        if idx_tmp == 0:
            return (
                np.int64(0),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty((0, m), dtype=np.int64)
            )

        # ---- find true nc domain of tmp results ----
        mink = tmp_ncs[0]
        maxk = tmp_ncs[0]
        for i in range(1, idx_tmp):
            v = tmp_ncs[i]
            if v < mink:
                mink = v
            if v > maxk:
                maxk = v

        nc_span = maxk - mink + 1

        bucket_scores = np.full((nc_span, top_c), -1e308, dtype=np.float64)
        bucket_counts = np.zeros(nc_span, dtype=np.int64)
        bucket_choices = np.full((nc_span, top_c, child_idx + 1), -1, dtype=np.int64)

        # ---- single pass top_c per nc ----
        for i in range(idx_tmp):
            nc_val = tmp_ncs[i]
            bid = nc_val - mink
            sc = tmp_scores[i]
            bcnt = bucket_counts[bid]

            if bcnt < top_c:
                bucket_scores[bid, bcnt] = sc
                for c in range(child_idx + 1):
                    bucket_choices[bid, bcnt, c] = tmp_choices[i, c]
                bucket_counts[bid] = bcnt + 1
            else:
                # replace worst
                worst = 0
                worst_score = bucket_scores[bid, 0]
                for u in range(1, top_c):
                    s2 = bucket_scores[bid, u]
                    if s2 < worst_score:
                        worst_score = s2
                        worst = u
                if sc > worst_score:
                    bucket_scores[bid, worst] = sc
                    for c in range(child_idx + 1):
                        bucket_choices[bid, worst, c] = tmp_choices[i, c]

        # ---- materialize ----
        write = 0
        for bid in range(nc_span):
            write += bucket_counts[bid]

        new_scores = np.empty(write, dtype=np.float64)
        new_ncs = np.empty(write, dtype=np.int64)
        new_choices = np.empty((write, child_idx + 1), dtype=np.int64)

        w = 0
        for bid in range(nc_span):
            cnt = bucket_counts[bid]
            nc_val = bid + mink
            for u in range(cnt):
                new_scores[w] = bucket_scores[bid, u]
                new_ncs[w] = nc_val
                for c in range(child_idx + 1):
                    new_choices[w, c] = bucket_choices[bid, u, c]
                w += 1

        cur_len = write
        cur_scores = new_scores
        cur_ncs = new_ncs
        cur_choices = new_choices

    return cur_len, cur_scores, cur_ncs, cur_choices

@njit(cache=True)
def _merge_children_pruned_Backup(
    children,
    m,
    node_cand_off,
    node_cand_count,
    cand_scores,
    cand_ncs,
    LB_node,
    UB_node,
    kmin,
    kmax,
    top_c
):
    """
    Safe pruned Cartesian merge across all children.

    Returns:
        cur_len,
        cur_scores,
        cur_ncs,
        cur_choices   (shape = [cur_len, m])
    """

    # ---- Feasible final nc band ----
    nc_min = kmin - UB_node
    nc_max = kmax - LB_node

    # ---- Precompute per-child min/max nc ----
    child_min_nc = np.empty(m, dtype=np.int64)
    child_max_nc = np.empty(m, dtype=np.int64)

    for ii in range(m):
        ch = children[ii]
        off = node_cand_off[ch]
        cnt = node_cand_count[ch]

        mn = cand_ncs[off]
        mx = cand_ncs[off]

        for j in range(1, cnt):
            v = cand_ncs[off + j]
            if v < mn:
                mn = v
            if v > mx:
                mx = v

        child_min_nc[ii] = mn
        child_max_nc[ii] = mx

    # ---- Initialise with first child ----
    first = children[0]
    offA = node_cand_off[first]
    lenA = node_cand_count[first]

    cur_len = lenA
    cur_scores = np.empty(cur_len, dtype=np.float64)
    cur_ncs = np.empty(cur_len, dtype=np.int64)
    cur_choices = np.empty((cur_len, 1), dtype=np.int64)

    for i in range(lenA):
        cur_scores[i] = cand_scores[offA + i]
        cur_ncs[i] = cand_ncs[offA + i]
        cur_choices[i, 0] = offA + i

    # ---- Merge remaining children ----
    for child_idx in range(1, m):

        ch = children[child_idx]
        offB = node_cand_off[ch]
        lenB = node_cand_count[ch]

        # Precompute remaining bounds AFTER this child
        min_rem = 0
        max_rem = 0
        for r in range(child_idx + 1, m):
            min_rem += child_min_nc[r]
            max_rem += child_max_nc[r]

        # Collect B arrays
        B_ncs = np.empty(lenB, dtype=np.int64)
        B_scores = np.empty(lenB, dtype=np.float64)
        B_ids = np.empty(lenB, dtype=np.int64)

        for j in range(lenB):
            B_ncs[j] = cand_ncs[offB + j]
            B_scores[j] = cand_scores[offB + j]
            B_ids[j] = offB + j

        # Sort B by nc ascending (for searchsorted)
        orderB = np.argsort(B_ncs)
        B_ncs_sorted = B_ncs[orderB]

        # First pass: count valid pairs safely
        # ---------------------------------------------------------
        # STREAMING MERGE WITH STRICT PER-NC TOP_C (NO TMP ARRAYS)
        # ---------------------------------------------------------

        # ---- find possible nc bounds for this merge stage ----
        # (same logic as before to determine feasible nc range)
        mink = nc_min
        maxk = nc_max
        nc_span = maxk - mink + 1

        if nc_span <= 0:
            return (
                np.int64(0),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty((0, m), dtype=np.int64)
            )

        # bucket storage
        bucket_scores = np.full((nc_span, top_c), -1e308, dtype=np.float64)
        bucket_counts = np.zeros(nc_span, dtype=np.int64)
        bucket_min_score = np.full(nc_span, -1e308, dtype=np.float64)
        bucket_min_index = np.zeros(nc_span, dtype=np.int64)
        bucket_choices = np.full((nc_span, top_c, child_idx + 1), -1, dtype=np.int64)

        # ---------------------------------------------------------
        # STREAM CARTESIAN PRODUCT DIRECTLY INTO BUCKETS
        # ---------------------------------------------------------

        for ia in range(cur_len):

            sa = cur_scores[ia]
            na = cur_ncs[ia]

            # safe partial prune (before B)
            if na + child_max_nc[child_idx] + max_rem < nc_min:
                continue
            if na + child_min_nc[child_idx] + min_rem > nc_max:
                continue

            lo = np.searchsorted(B_ncs_sorted, nc_min - na - max_rem, side='left')
            hi = np.searchsorted(B_ncs_sorted, nc_max - na - min_rem, side='right')

            for k in range(lo, hi):

                jb = orderB[k]
                nb = B_ncs[jb]
                nc_total = na + nb

                # final safety
                if nc_total + min_rem > nc_max:
                    continue
                if nc_total + max_rem < nc_min:
                    continue

                score = sa + B_scores[jb]
                bid = nc_total - mink

                count = bucket_counts[bid]

                # -------------------------------------------------
                # EARLY BREAK (SAFE)
                # -------------------------------------------------
                if count >= top_c and score <= bucket_min_score[bid]:
                    # because B_scores sorted descending,
                    # further k will only decrease score
                    break

                # -------------------------------------------------
                # INSERTION LOGIC (STRICTLY IDENTICAL SEMANTICS)
                # -------------------------------------------------

                if count < top_c:
                    # append
                    bucket_scores[bid, count] = score

                    # copy choices
                    for c in range(child_idx):
                        bucket_choices[bid, count, c] = cur_choices[ia, c]
                    bucket_choices[bid, count, child_idx] = B_ids[jb]

                    bucket_counts[bid] = count + 1

                    # update min tracking
                    if count == 0 or score < bucket_min_score[bid]:
                        bucket_min_score[bid] = score
                        bucket_min_index[bid] = count

                else:
                    # replace worst if strictly better
                    if score > bucket_min_score[bid]:

                        worst = bucket_min_index[bid]

                        bucket_scores[bid, worst] = score

                        for c in range(child_idx):
                            bucket_choices[bid, worst, c] = cur_choices[ia, c]
                        bucket_choices[bid, worst, child_idx] = B_ids[jb]

                        # recompute new minimum in this bucket
                        new_min = bucket_scores[bid, 0]
                        new_idx = 0
                        for u in range(1, top_c):
                            s2 = bucket_scores[bid, u]
                            if s2 < new_min:
                                new_min = s2
                                new_idx = u

                        bucket_min_score[bid] = new_min
                        bucket_min_index[bid] = new_idx


        # ---------------------------------------------------------
        # MATERIALIZE BUCKETS INTO cur_* ARRAYS
        # ---------------------------------------------------------

        write = 0
        for bid in range(nc_span):
            write += bucket_counts[bid]

        if write == 0:
            return (
                np.int64(0),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty((0, m), dtype=np.int64)
            )

        new_scores = np.empty(write, dtype=np.float64)
        new_ncs = np.empty(write, dtype=np.int64)
        new_choices = np.empty((write, child_idx + 1), dtype=np.int64)

        w = 0
        for bid in range(nc_span):
            cnt = bucket_counts[bid]
            nc_val = bid + mink
            for u in range(cnt):
                new_scores[w] = bucket_scores[bid, u]
                new_ncs[w] = nc_val
                for c in range(child_idx + 1):
                    new_choices[w, c] = bucket_choices[bid, u, c]
                w += 1

        cur_len = write
        cur_scores = new_scores
        cur_ncs = new_ncs
        cur_choices = new_choices


    return cur_len, cur_scores, cur_ncs, cur_choices


@njit(cache=True)
def _filter_candidates(
    scores_in, ncs_in, n_in,
    LB_node, UB_node, kmin, kmax, top_c
):
    if n_in == 0:
        return 0, -np.ones(0, dtype=np.int64)

    # ---- compute mink, maxk ----
    mink = ncs_in[0]
    maxk = ncs_in[0]
    for i in range(1, n_in):
        v = ncs_in[i]
        if v < mink:
            mink = v
        if v > maxk:
            maxk = v

    # ---- TRUE feasible coverage domain ----
    p_min = kmin - maxk
    if p_min < LB_node:
        p_min = LB_node

    p_max = kmax - mink
    if p_max > UB_node:
        p_max = UB_node

    if p_max < p_min:
        return 0, -np.ones(0, dtype=np.int64)

    cov_len = p_max - p_min + 1
    nk = cov_len * top_c

    coverage = np.zeros(cov_len, dtype=np.int32)
    coverage_sum = 0

    keep_idx = -np.ones(n_in, dtype=np.int64)
    kept = 0

    # ---- full-width fast path ----
    uniform_mode = True
    uniform_level = 0

    # ---- unsaturated window ----
    min_unsat = 0
    max_unsat = cov_len - 1

    for idx in range(n_in):

        k = ncs_in[idx]

        # interval in absolute p-space
        l_add = kmin - k
        if l_add < LB_node:
            l_add = LB_node

        u_add = kmax - k
        if u_add > UB_node:
            u_add = UB_node

        if u_add < l_add:
            continue

        # convert to feasible coverage index space
        start = l_add - p_min
        end   = u_add - p_min

        if start < 0:
            start = 0
        if end >= cov_len:
            end = cov_len - 1

        if start > end:
            continue

        # ---- FULL WIDTH FAST PATH ----
        if uniform_mode and start == 0 and end == cov_len - 1:

            if uniform_level < top_c:
                uniform_level += 1
                coverage_sum += cov_len

                keep_idx[kept] = idx
                kept += 1

                if coverage_sum >= nk:
                    break

                continue
            else:
                break

        # ---- materialise if needed ----
        if uniform_mode:
            for p in range(cov_len):
                coverage[p] = uniform_level
            uniform_mode = False

        # ---- clamp to unsaturated window ----
        if start < min_unsat:
            start = min_unsat
        if end > max_unsat:
            end = max_unsat

        if start > end:
            continue

        moved = 0

        # ---- merged scan + update ----
        for p in range(start, end + 1):
            if coverage[p] < top_c:
                coverage[p] += 1
                moved += 1

                # update unsat window edges
                if coverage[p] == top_c:
                    if p == min_unsat:
                        while min_unsat <= max_unsat and coverage[min_unsat] == top_c:
                            min_unsat += 1
                    if p == max_unsat:
                        while max_unsat >= min_unsat and coverage[max_unsat] == top_c:
                            max_unsat -= 1

        if moved > 0:
            keep_idx[kept] = idx
            kept += 1
            coverage_sum += moved

            if coverage_sum >= nk:
                break

    if kept == 0:
        return 0, -np.ones(0, dtype=np.int64)

    out = np.empty(kept, dtype=np.int64)
    for i in range(kept):
        out[i] = keep_idx[i]

    return kept, out


@njit(cache=True)
def _efosc(children_flat, children_off, post, clusteval, is_noise,
          cluster_key, top_c, kmin, kmax, LB_arr, UB_arr, n_leaves):
    """
    Choice-buffered EFOSC (numba) — corrected sorting mapping.

    Inputs:
      children_flat, children_off : CSR representation of children (int64 arrays)
      post : postorder node array (int64)
      clusteval : float64 per-node score
      is_noise : int8/int64 per-node (0/1)
      cluster_key : int root node index
      top_c, kmin, kmax : ints
      LB_arr, UB_arr : int64 arrays per-node bounds
      n_leaves : int scalar

    Returns: (root_scores, root_ncs, sel_nodes, sel_counts, root_k)
    """
    N = clusteval.shape[0]
    # FAST PATH: top_c == 1, kmin <= 2, kmax >= N
    if top_c == 1 and kmin <= 2 and kmax >= n_leaves:
        best_score = np.full(N, _NEG_INF, dtype=np.float64)
        best_nc = np.zeros(N, dtype=np.int64)
        best_choice = np.zeros(N, dtype=np.int8)
        for idx in range(post.shape[0]):
            node = int(post[idx])
            s = int(children_off[node]); e = int(children_off[node+1])
            if e - s == 0:
                qv = clusteval[node]
                if is_noise[node] == 1:
                    best_score[node] = qv
                    best_nc[node] = 0
                    best_choice[node] = 1
                else:
                    best_score[node] = qv
                    best_nc[node] = 1
                    best_choice[node] = 1
                continue
            sum_q = 0.0
            sum_nc = 0
            for j in range(s, e):
                ch = int(children_flat[j])
                sum_q += best_score[ch]
                sum_nc += best_nc[ch]
            parent_q = clusteval[node]
            parent_nc = 0 if is_noise[node] == 1 else 1

            # Disallow selecting the root itself when kmin == 2
            allow_parent = not (kmin == 2 and node == cluster_key)

            if allow_parent and (
                (parent_q > sum_q) or
                (parent_q == sum_q and parent_nc > sum_nc)
            ):
                best_score[node] = parent_q
                best_nc[node] = parent_nc
                best_choice[node] = 1
            else:
                best_score[node] = sum_q
                best_nc[node] = sum_nc
                best_choice[node] = 0

        # reconstruct selected nodes
        root = cluster_key
        root_scores = np.full(top_c, _NEG_INF, dtype=np.float64)
        root_ncs = np.zeros(top_c, dtype=np.int64)
        sel_nodes = -np.ones((top_c, N), dtype=np.int64)
        sel_counts = np.zeros(top_c, dtype=np.int64)

        stack = np.empty(N, dtype=np.int64)
        top = 0
        stack[top] = root
        top += 1
        out_ptr = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            if best_choice[node] == 1:
                if is_noise[node] == 0:
                    sel_nodes[0, out_ptr] = node
                    out_ptr += 1
            else:
                s2 = int(children_off[node]); e2 = int(children_off[node+1])
                for j in range(s2, e2):
                    stack[top] = int(children_flat[j])
                    top += 1

        sel_counts[0] = out_ptr
        root_scores[0] = best_score[root]
        root_ncs[0] = best_nc[root]
        return root_scores, root_ncs, sel_nodes, sel_counts, 1

    # ----------------------------
    # GENERAL PATH (top_c > 1)
    # ----------------------------
    # ----------------------------
    # GENERAL PATH (top_c > 1)
    # ----------------------------

    init_cap = max(64, N * 4)

    cand_scores = np.empty(init_cap, dtype=np.float64)
    cand_ncs = np.empty(init_cap, dtype=np.int64)
    cand_owner = np.empty(init_cap, dtype=np.int64)
    cand_pos = 0

    node_cand_off = -np.ones(N, dtype=np.int64)
    node_cand_count = np.zeros(N, dtype=np.int64)

    choice_buf = np.empty(init_cap * 4, dtype=np.int64)
    for i in range(choice_buf.shape[0]):
        choice_buf[i] = -1
    choice_pos = 0

    cand_choice_off = -np.ones(init_cap, dtype=np.int64)
    cand_choice_len = np.zeros(init_cap, dtype=np.int64)

    # ----------------------------
    # Buffer growth helpers
    # ----------------------------

    def _ensure_cand_capacity(req):
        nonlocal cand_scores, cand_ncs, cand_owner
        nonlocal cand_choice_off, cand_choice_len
        if req <= cand_scores.shape[0]:
            return
        newsize = max(req, cand_scores.shape[0] * 2)

        ns = np.empty(newsize, dtype=np.float64)
        ni = np.empty(newsize, dtype=np.int64)
        no = np.empty(newsize, dtype=np.int64)

        for i in range(cand_scores.shape[0]):
            ns[i] = cand_scores[i]
            ni[i] = cand_ncs[i]
            no[i] = cand_owner[i]

        cand_scores = ns
        cand_ncs = ni
        cand_owner = no

        new_choice_meta = -np.ones(newsize, dtype=np.int64)
        new_choice_len = np.zeros(newsize, dtype=np.int64)

        for i in range(cand_choice_off.shape[0]):
            new_choice_meta[i] = cand_choice_off[i]
            new_choice_len[i] = cand_choice_len[i]

        cand_choice_off = new_choice_meta
        cand_choice_len = new_choice_len

    def _ensure_choice_capacity(req):
        nonlocal choice_buf
        if req <= choice_buf.shape[0]:
            return
        newsize = max(req, choice_buf.shape[0] * 2)
        nb = np.empty(newsize, dtype=np.int64)
        for i in range(choice_buf.shape[0]):
            nb[i] = choice_buf[i]
        for i in range(choice_buf.shape[0], newsize):
            nb[i] = -1
        choice_buf = nb

    # ----------------------------
    # Bottom-up DP
    # ----------------------------

    for pidx in range(post.shape[0]):

        node = int(post[pidx])
        s = int(children_off[node])
        e = int(children_off[node + 1])
        m = e - s

        # ---- Leaf ----
        if m == 0:

            qv = clusteval[node]
            nc = 0 if is_noise[node] == 1 else 1

            _ensure_cand_capacity(cand_pos + 1)

            cand_scores[cand_pos] = qv
            cand_ncs[cand_pos] = nc
            cand_owner[cand_pos] = node
            cand_choice_off[cand_pos] = -1
            cand_choice_len[cand_pos] = 0

            node_cand_off[node] = cand_pos
            node_cand_count[node] = 1

            cand_pos += 1
            continue

        # ---- Gather children ----
        children = np.empty(m, dtype=np.int64)
        for ii in range(m):
            children[ii] = int(children_flat[s + ii])

        # ---- Ensure each child has at least one candidate ----
        for ii in range(m):
            ch = children[ii]
            if node_cand_count[ch] == 0:
                qv = clusteval[ch]
                nc = 0 if is_noise[ch] == 1 else 1

                _ensure_cand_capacity(cand_pos + 1)

                cand_scores[cand_pos] = qv
                cand_ncs[cand_pos] = nc
                cand_owner[cand_pos] = ch
                cand_choice_off[cand_pos] = -1
                cand_choice_len[cand_pos] = 0

                node_cand_off[ch] = cand_pos
                node_cand_count[ch] = 1

                cand_pos += 1

        LB_node = int(LB_arr[node])
        UB_node = int(UB_arr[node])

        # ---- SAFE PRUNED MERGE ----
        cur_len, cur_scores, cur_ncs, cur_choices = _merge_children_pruned(
            children,
            m,
            node_cand_off,
            node_cand_count,
            cand_scores,
            cand_ncs,
            LB_node,
            UB_node,
            kmin,
            kmax,
            top_c
        )

        # ---- Build local candidate list (parent + children) ----
        tot = cur_len + 1

        cand_scores_local = np.empty(tot, dtype=np.float64)
        cand_ncs_local = np.empty(tot, dtype=np.int64)

        cand_scores_local[0] = clusteval[node]
        cand_ncs_local[0] = 0 if is_noise[node] == 1 else 1

        for ii in range(cur_len):
            cand_scores_local[ii + 1] = cur_scores[ii]
            cand_ncs_local[ii + 1] = cur_ncs[ii]

        # ---- Sort (score desc, nc desc) ----
        orig_idx = np.empty(tot, dtype=np.int64)
        for ii in range(tot):
            orig_idx[ii] = ii

        order = np.argsort(cand_ncs_local)
        order = order[::-1]
        order = order[np.argsort(cand_scores_local[order])[::-1]]

        sorted_scores = np.empty(tot, dtype=np.float64)
        sorted_ncs = np.empty(tot, dtype=np.int64)

        for ii in range(tot):
            sorted_scores[ii] = cand_scores_local[order[ii]]
            sorted_ncs[ii] = cand_ncs_local[order[ii]]

        kept_cnt, kept_idx_sorted = _filter_candidates(
            sorted_scores,
            sorted_ncs,
            tot,
            LB_node,
            UB_node,
            kmin,
            kmax,
            top_c
        )

        if kept_cnt == 0:
            node_cand_off[node] = -1
            node_cand_count[node] = 0
            continue

        _ensure_cand_capacity(cand_pos + kept_cnt)

        node_cand_off[node] = cand_pos
        node_cand_count[node] = kept_cnt

        for kk in range(kept_cnt):

            k_sorted = int(kept_idx_sorted[kk])
            orig_local = int(order[k_sorted])

            cand_scores[cand_pos] = cand_scores_local[orig_local]
            cand_ncs[cand_pos] = cand_ncs_local[orig_local]
            cand_owner[cand_pos] = node

            if orig_local == 0:
                cand_choice_off[cand_pos] = -1
                cand_choice_len[cand_pos] = 0
            else:
                comb_idx = orig_local - 1
                _ensure_choice_capacity(choice_pos + m)
                cand_choice_off[cand_pos] = choice_pos
                cand_choice_len[cand_pos] = m

                for d in range(m):
                    choice_buf[choice_pos] = cur_choices[comb_idx, d]
                    choice_pos += 1

            cand_pos += 1


    # ================================
    # Root extraction (FAST DAG walk)
    # ================================
    root = cluster_key
    root_off = int(node_cand_off[root])
    root_cnt = int(node_cand_count[root])

    root_scores = np.full(top_c, _NEG_INF, dtype=np.float64)
    root_ncs = np.zeros(top_c, dtype=np.int64)
    sel_nodes = -np.ones((top_c, N), dtype=np.int64)
    sel_counts = np.zeros(top_c, dtype=np.int64)

    if root_cnt <= 0:
        return root_scores, root_ncs, sel_nodes, sel_counts, 0

    out_k = root_cnt if root_cnt <= top_c else top_c
    picked = np.zeros(root_cnt, dtype=np.int8)

    # Preallocate once
    stack = np.empty(cand_pos, dtype=np.int64)

    for out_i in range(out_k):

        best_idx = -1
        best_sc = _NEG_INF
        best_nc = -9223372036854775808

        for j in range(root_cnt):
            if picked[j] == 1:
                continue
            sc = cand_scores[root_off + j]
            nc = cand_ncs[root_off + j]
            if sc > best_sc or (sc == best_sc and nc > best_nc):
                best_sc = sc
                best_nc = nc
                best_idx = j

        if best_idx == -1:
            break

        picked[best_idx] = 1
        root_scores[out_i] = best_sc
        root_ncs[out_i] = best_nc

        # --- fast traversal ---
        sp = 0
        stack[sp] = root_off + best_idx
        sp += 1
        out_ptr = 0

        while sp > 0:
            sp -= 1
            gid = stack[sp]

            coff = cand_choice_off[gid]
            clen = cand_choice_len[gid]

            if clen > 0:
                # push children in reverse
                for ii in range(clen - 1, -1, -1):
                    stack[sp] = choice_buf[coff + ii]
                    sp += 1
            else:
                owner = cand_owner[gid]
                if is_noise[owner] == 0:
                    sel_nodes[out_i, out_ptr] = owner
                    out_ptr += 1

        sel_counts[out_i] = out_ptr

    return root_scores, root_ncs, sel_nodes, sel_counts, out_k

