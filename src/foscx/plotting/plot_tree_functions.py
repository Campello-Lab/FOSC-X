"""
plot_functions.py
=================
Condensed-tree visualisation utilities supporting both:

  * **Non-binary trees** (HDBSCAN icicle / bar-width plot) — ``binary_tree=False``
  * **Binary trees** (dendrogram / U-shape line plot)      — ``binary_tree=True``

Both paths accept trees whose third value column is named either
``'distance'`` or ``'lambda_val'``.  Pass ``value_field='distance'`` or
``value_field='lambda_val'`` to override auto-detection.

Public API
----------
Non-binary
    get_plot_data_nb        – pre-compute bar/line data
    plot_condensed_nb       – draw icicle plot
    plot_selected_nb        – overlay ellipses (non-binary coords)

Binary
    get_plot_data_bin       – pre-compute dendrogram coords
    plot_condensed_bin      – draw dendrogram
    plot_selected_bin       – overlay ellipses (stem coords)

Shared
    clear_selected_clusters – remove ellipse artists

Unified entry point
    interactive_condensed   – ipywidgets slider browser
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CB_LEFT   = 0
CB_RIGHT  = 1
CB_BOTTOM = 2
CB_TOP    = 3


# ---------------------------------------------------------------------------
# Shared internal helper: resolve which field holds split values
# ---------------------------------------------------------------------------

def _resolve_value_field(condensed_tree, value_field=None):
    """Return the name of the column that holds split values (distance / lambda).

    Auto-detects from the recarray dtype when *value_field* is None.
    Raises ``ValueError`` if neither standard name is present and no override
    was supplied.
    """
    if value_field is not None:
        return value_field
    names = condensed_tree.dtype.names
    if 'distance' in names:
        return 'distance'
    if 'lambda_val' in names:
        return 'lambda_val'
    raise ValueError(
        "Cannot auto-detect value column. "
        "Pass value_field='distance' or value_field='lambda_val'."
    )


# ===========================================================================
# NON-BINARY TREE
# ===========================================================================

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _recurse_leaf_dfs(cluster_tree, current_node):
    children = cluster_tree[cluster_tree['parent'] == current_node]['child']
    if len(children) == 0:
        return [current_node]
    return sum([_recurse_leaf_dfs(cluster_tree, child) for child in children], [])


def _get_leaves(condensed_tree):
    cluster_tree = condensed_tree[condensed_tree['child_size'] > 1]
    root = cluster_tree['parent'].min()
    return _recurse_leaf_dfs(cluster_tree, root)


# ---------------------------------------------------------------------------
# Public: pre-compute plot data
# ---------------------------------------------------------------------------

def get_plot_data_nb(
    condensed_tree,
    leaf_separation=1,
    log_size=False,
    max_rectangle_per_icicle=20,
    value_field=None,
    density=True,
    **kwargs,
):
    """Pre-compute bar and line data for the non-binary icicle plot.

    Parameters
    ----------
    condensed_tree : numpy recarray
        Must have fields ``parent``, ``child``, ``child_size``, and either
        ``distance`` or ``lambda_val`` (auto-detected; override with
        *value_field*).
    leaf_separation : float, optional
        Horizontal spacing between leaf clusters. (default 1)
    log_size : bool, optional
        Use log scale for cluster size. (default False)
    max_rectangle_per_icicle : int, optional
        Maximum bars emitted per cluster branch. (default 20)
    value_field : str or None, optional
        Name of the split-value column.  Auto-detected when ``None``.
    **kwargs
        Accepted and ignored (allows a shared kwargs dict to be passed in).

    Returns
    -------
    dict with keys:
        ``bar_centers``, ``bar_tops``, ``bar_bottoms``, ``bar_widths``,
        ``line_xs``, ``line_ys``, ``cluster_bounds``, ``value_field``
    """
    vf = _resolve_value_field(condensed_tree, value_field)

    leaves    = _get_leaves(condensed_tree)

    # Identify root: the parent node that never appears as a child.
    all_parents  = set(condensed_tree['parent'])
    all_children = set(condensed_tree['child'])
    root = (all_parents - all_children).pop()

    cluster_x_coords = dict(zip(
        leaves, [leaf_separation * x for x in range(len(leaves))]
    ))

    # Build a top-down BFS ordering of internal nodes, then reverse it to get
    # a bottom-up order.  This guarantees children are processed before their
    # parents so cluster_x_coords lookups always find their children already
    # assigned, regardless of whether node IDs are a contiguous numeric range.
    bfs_order = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node in all_parents:
            bfs_order.append(node)
            for child in condensed_tree[condensed_tree['parent'] == node]['child']:
                queue.append(child)
    bottom_up = list(reversed(bfs_order))

    # For density trees the root bar starts at 0 (it exists from the very
    # beginning and children fall out of it at increasing lambda values).
    # For linkage trees the root starts at its own split distance, which is
    # the minimum edge value among its children (all sharing the same parent
    # split distance).
    root_children = condensed_tree[condensed_tree['parent'] == root]
    # For density trees root starts at 0.
    # For linkage trees root bottom == root top == its split distance,
    # so the bar has zero height and nothing is drawn.
    root_y = 0 if density else float(root_children[vf].max())
    cluster_y_coords = {root: root_y}

    for cluster in bottom_up:
        split = condensed_tree[['child', vf]]
        split = split[
            (condensed_tree['parent'] == cluster) &
            (condensed_tree['child_size'] > 1)
        ]
        if len(split['child']) > 1:
            left_child, right_child = split['child']
            cluster_x_coords[cluster] = np.mean([
                cluster_x_coords[left_child],
                cluster_x_coords[right_child],
            ])
            cluster_y_coords[left_child]  = split[vf][0]
            cluster_y_coords[right_child] = split[vf][1]

        # Ensure every internal node has an x-coord even if it had fewer than
        # 2 internal children (e.g. root with only leaf children).
        if cluster not in cluster_x_coords:
            all_child_xs = [
                cluster_x_coords[ch]
                for ch in condensed_tree[condensed_tree['parent'] == cluster]['child']
                if ch in cluster_x_coords
            ]
            if all_child_xs:
                cluster_x_coords[cluster] = np.mean(all_child_xs)

    bar_centers  = []
    bar_tops     = []
    bar_bottoms  = []
    bar_widths   = []
    cluster_bounds = {}

    scaling = np.sum(condensed_tree[condensed_tree['parent'] == root]['child_size'])
    if log_size:
        scaling = np.log(scaling)

    for c in bottom_up:
        cluster_bounds[c] = [0, 0, 0, 0]

        c_children         = condensed_tree[condensed_tree['parent'] == c]
        current_size       = np.sum(c_children['child_size'])
        current_lambda     = cluster_y_coords[c]
        cluster_max_size   = current_size
        cluster_max_lambda = c_children[vf].max()
        cluster_min_size   = np.sum(
            c_children[c_children[vf] == cluster_max_lambda]['child_size']
        )

        if log_size:
            current_size     = np.log(current_size)
            cluster_max_size = np.log(cluster_max_size)
            cluster_min_size = np.log(cluster_min_size)

        total_size_change = float(cluster_max_size - cluster_min_size)
        step_size_change  = total_size_change / max_rectangle_per_icicle

        cluster_bounds[c][CB_LEFT]   = cluster_x_coords[c] * scaling - (current_size / 2.0)
        cluster_bounds[c][CB_RIGHT]  = cluster_x_coords[c] * scaling + (current_size / 2.0)
        if not density and c == root:
            # Linkage root has no parent distance — set bottom == top so no bar is drawn
            cluster_bounds[c][CB_BOTTOM] = float(np.max(c_children[vf]))
            cluster_bounds[c][CB_TOP]    = float(np.max(c_children[vf]))
            current_lambda               = float(np.max(c_children[vf]))
            last_step_lambda             = current_lambda
        else:
            cluster_bounds[c][CB_BOTTOM] = cluster_y_coords[c]
            cluster_bounds[c][CB_TOP]    = np.max(c_children[vf])

        last_step_size   = current_size
        last_step_lambda = current_lambda

        for i in np.argsort(c_children[vf]):
            row = c_children[i]
            if row[vf] != current_lambda and (
                last_step_size - current_size > step_size_change
                or row[vf] == cluster_max_lambda
            ):
                bar_centers.append(cluster_x_coords[c] * scaling)
                bar_tops.append(row[vf] - last_step_lambda)
                bar_bottoms.append(last_step_lambda)
                bar_widths.append(last_step_size)
                last_step_size   = current_size
                last_step_lambda = current_lambda

            if log_size:
                exp_size = np.exp(current_size) - row['child_size']
                current_size = np.log(exp_size) if exp_size > 0.01 else 0.0
            else:
                current_size -= row['child_size']
            current_lambda = row[vf]

    line_xs = []
    line_ys = []
    for row in condensed_tree[condensed_tree['child_size'] > 1]:
        parent     = row['parent']
        child      = row['child']
        child_size = np.log(row['child_size']) if log_size else row['child_size']
        sign = np.sign(cluster_x_coords[child] - cluster_x_coords[parent])
        line_xs.append([
            cluster_x_coords[parent] * scaling,
            cluster_x_coords[child]  * scaling + sign * (child_size / 2.0),
        ])
        line_ys.append([cluster_y_coords[child], cluster_y_coords[child]])

    return {
        'bar_centers':    bar_centers,
        'bar_tops':       bar_tops,
        'bar_bottoms':    bar_bottoms,
        'bar_widths':     bar_widths,
        'line_xs':        line_xs,
        'line_ys':        line_ys,
        'cluster_bounds': cluster_bounds,
        'value_field':    vf,
    }


# ---------------------------------------------------------------------------
# Public: draw non-binary icicle plot
# ---------------------------------------------------------------------------

def plot_condensed_nb(
    condensed_tree,
    plot_data,
    axis=None,
    cmap='viridis',
    colorbar=True,
    log_size=False,
    density=True,
    **kwargs,
):
    """Draw the non-binary icicle / bar-width condensed tree.

    Parameters
    ----------
    condensed_tree : numpy recarray
    plot_data : dict
        As returned by :func:`get_plot_data_nb`.
    axis : matplotlib axis, optional
    cmap : str or colormap, optional
        Use ``'none'`` for solid black bars. (default 'viridis')
    colorbar : bool, optional (default True)
    log_size : bool, optional (default False)
    density : bool, optional
        ``True`` → y-axis shows the raw value (lambda / distance);
        ``False`` → y-axis is re-labelled as distance by inverting lambda.
        (default True)
    **kwargs
        Accepted and ignored.

    Returns
    -------
    axis : matplotlib axis
    """
    vf = plot_data.get('value_field') or _resolve_value_field(condensed_tree)

    if axis is None:
        axis = plt.gca()

    if cmap != 'none':
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(0, max(plot_data['bar_widths'])),
        )
        sm.set_array(plot_data['bar_widths'])
        bar_colors = [sm.to_rgba(x) for x in plot_data['bar_widths']]
    else:
        bar_colors = 'black'

    axis.bar(
        plot_data['bar_centers'],
        plot_data['bar_tops'],
        bottom=plot_data['bar_bottoms'],
        width=plot_data['bar_widths'],
        color=bar_colors,
        align='center',
        linewidth=0,
    )

    for xs, ys in zip(plot_data['line_xs'], plot_data['line_ys']):
        axis.plot(xs, ys, color='black', linewidth=1)

    axis.set_xticks([])
    for side in ('right', 'top', 'bottom'):
        axis.spines[side].set_visible(False)

    if density:
        # Density trees: values increase away from root; invert so root is
        # visually at the top.
        axis.invert_yaxis()
        axis.set_ylabel('lambda value' if vf == 'lambda_val' else 'distance')
    else:
        # Linkage trees: raw split distances are already large at the root and
        # small at leaves, so bars grow upward naturally — no inversion needed.
        # Re-label ticks to show the original distance values.
        axis.set_ylabel('distance')

    if colorbar and cmap != 'none':
        cb = plt.colorbar(sm, ax=axis)
        cb.ax.set_ylabel('log(Number of points)' if log_size else 'Number of points')

    return axis


# ---------------------------------------------------------------------------
# Public: overlay ellipses — non-binary
# ---------------------------------------------------------------------------

def plot_selected_nb(
    selected_clusters,
    plot_data,
    axis,
    label_clusters=False,
    selection_palette=None,
    **kwargs,
):
    """Overlay ellipses on a non-binary condensed-tree plot.

    Ellipses wrap the full bounding box of each selected cluster's bar stack.

    Parameters
    ----------
    selected_clusters : list[int]
    plot_data : dict
        As returned by :func:`get_plot_data_nb`.
    axis : matplotlib axis
    label_clusters : bool, optional (default False)
    selection_palette : list, optional
    **kwargs
        Accepted and ignored.

    Returns
    -------
    artists : list
    """
    from matplotlib.patches import Ellipse

    cluster_bounds = plot_data['cluster_bounds']
    artists = []

    for i, c in enumerate(selected_clusters):
        c_bounds = cluster_bounds[c]
        width  = c_bounds[CB_RIGHT]  - c_bounds[CB_LEFT]
        height = c_bounds[CB_TOP]    - c_bounds[CB_BOTTOM]
        center = (
            np.mean([c_bounds[CB_LEFT],   c_bounds[CB_RIGHT]]),
            np.mean([c_bounds[CB_BOTTOM], c_bounds[CB_TOP]]),
        )
        oval_color = (
            selection_palette[i]
            if selection_palette is not None and len(selection_palette) >= len(selected_clusters)
            else 'r'
        )
        ellipse = Ellipse(
            center,
            2.0  * width,
            1.01 * height,
            facecolor='none',
            edgecolor=oval_color,
            linewidth=1.5,
        )
        axis.add_artist(ellipse)
        artists.append(ellipse)

        if label_clusters:
            text = axis.annotate(
                str(i),
                xy=center,
                xytext=(center[0] - 4.0 * width, center[1] + 0.65 * height),
                horizontalalignment='left',
                verticalalignment='bottom',
            )
            artists.append(text)

    return artists


# ===========================================================================
# BINARY TREE
# ===========================================================================

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _line_width(child, condensed_tree):
    """Return ``child_size`` for *child* (1.0 for leaves not in the tree)."""
    rows = condensed_tree[condensed_tree['child'] == child]
    return 1.0 if len(rows) == 0 else float(rows[0]['child_size'])


def _child_distance(node, condensed_tree, value_field):
    """Return the split value at which *node* was formed (0.0 for leaves)."""
    rows = condensed_tree[condensed_tree['parent'] == node]
    return 0.0 if len(rows) == 0 else float(rows[0][value_field])


def _build_dendrogram_coords(condensed_tree, value_field):
    """Traverse a binary condensed tree and produce icoord / dcoord arrays.

    Returns
    -------
    icoord, dcoord, node_children, node_x, children_of, root
    """
    children_of = defaultdict(list)
    for row in condensed_tree:
        children_of[row['parent']].append((row['child'], row[value_field]))

    all_parents  = set(condensed_tree['parent'])
    all_children = set(condensed_tree['child'])
    roots = all_parents - all_children
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}: {roots}")
    root = roots.pop()

    leaf_counter = [0]
    node_x = {}

    def assign_positions(node):
        if node not in children_of:
            leaf_counter[0] += 1
            node_x[node] = leaf_counter[0] * 10
            return node_x[node]
        xs = [assign_positions(child) for child, _ in children_of[node]]
        node_x[node] = np.mean(xs)
        return node_x[node]

    assign_positions(root)

    icoord        = []
    dcoord        = []
    node_children = []

    def build_coords(node):
        if node not in children_of:
            return
        (left_child, left_val), (right_child, right_val) = children_of[node]
        build_coords(left_child)
        build_coords(right_child)

        lx = node_x[left_child]
        rx = node_x[right_child]
        left_bottom  = 0.0 if left_child  not in children_of else _child_distance(left_child,  condensed_tree, value_field)
        right_bottom = 0.0 if right_child not in children_of else _child_distance(right_child, condensed_tree, value_field)

        icoord.append([lx, lx, rx, rx])
        dcoord.append([left_bottom, left_val, right_val, right_bottom])
        node_children.append((left_child, right_child))

    build_coords(root)
    return icoord, dcoord, node_children, node_x, children_of, root


# ---------------------------------------------------------------------------
# Public: pre-compute plot data
# ---------------------------------------------------------------------------

def get_plot_data_bin(condensed_tree, value_field=None, **kwargs):
    """Pre-compute dendrogram coordinates and stem bounds for a binary tree.

    Parameters
    ----------
    condensed_tree : numpy recarray
        Must have fields ``parent``, ``child``, ``child_size``, and either
        ``distance`` or ``lambda_val`` (auto-detected; override with
        *value_field*).
    value_field : str or None, optional
        Name of the split-value column.  Auto-detected when ``None``.
    **kwargs
        Accepted and ignored.

    Returns
    -------
    dict with keys:
        ``icoord``, ``dcoord``, ``node_children``, ``node_x``,
        ``children_of``, ``root``, ``stem_coords``, ``cluster_bounds``,
        ``value_field``
    """
    vf = _resolve_value_field(condensed_tree, value_field)

    icoord, dcoord, node_children, node_x, children_of, root = \
        _build_dendrogram_coords(condensed_tree, vf)

    # stem_coords: node_id -> (stem_x, y_bottom, y_top)
    stem_coords = {}
    for (left_child, right_child), x_seg, y_seg in zip(node_children, icoord, dcoord):
        stem_coords[int(left_child)]  = (x_seg[0], y_seg[0], y_seg[1])
        stem_coords[int(right_child)] = (x_seg[3], y_seg[3], y_seg[2])

    cluster_bounds = {
        node_id: (sx, sx, sy_bot, sy_top)
        for node_id, (sx, sy_bot, sy_top) in stem_coords.items()
    }

    return dict(
        icoord=icoord,
        dcoord=dcoord,
        node_children=node_children,
        node_x=node_x,
        children_of=children_of,
        root=root,
        stem_coords=stem_coords,
        cluster_bounds=cluster_bounds,
        value_field=vf,
    )


# ---------------------------------------------------------------------------
# Public: draw binary dendrogram
# ---------------------------------------------------------------------------

def plot_condensed_bin(
    condensed_tree,
    plot_data=None,
    axis=None,
    vary_line_width=True,
    cmap='viridis',
    colorbar=True,
    value_field=None,
    **kwargs,
):
    """Draw the binary condensed tree as a dendrogram.

    Parameters
    ----------
    condensed_tree : numpy recarray
        Fields: ``parent``, ``child``, ``child_size``, and either
        ``distance`` or ``lambda_val``.
    plot_data : dict, optional
        As returned by :func:`get_plot_data_bin`. Computed if not supplied.
    axis : matplotlib axis, optional
    vary_line_width : bool, optional
        Scale branch thickness by cluster size. (default True)
    cmap : str or colormap, optional
        Use ``'none'`` for uniform black lines. (default 'viridis')
    colorbar : bool, optional (default True)
    value_field : str or None, optional
        Passed to :func:`get_plot_data_bin` when *plot_data* is not supplied.
    **kwargs
        Accepted and ignored.

    Returns
    -------
    axis : matplotlib axis
    """
    if plot_data is None:
        plot_data = get_plot_data_bin(condensed_tree, value_field=value_field)

    vf = plot_data['value_field']
    X             = plot_data['icoord']
    Y             = plot_data['dcoord']
    node_children = plot_data['node_children']

    if axis is None:
        axis = plt.gca()

    if vary_line_width:
        linewidths = [
            (_line_width(lc, condensed_tree), _line_width(rc, condensed_tree))
            for lc, rc in node_children
        ]
    else:
        linewidths = [(1.0, 1.0)] * len(Y)

    if cmap != 'none':
        all_lw = np.log2(np.array(linewidths).flatten())
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(0, all_lw.max()),
        )
        sm.set_array(all_lw)

    for x, y, lw in zip(X, Y, linewidths):
        lw_l = np.log2(1 + lw[0])
        lw_r = np.log2(1 + lw[1])

        if cmap != 'none':
            axis.plot(x[:2], y[:2], color=sm.to_rgba(np.log2(lw[0])),
                      linewidth=lw_l, solid_joinstyle='miter', solid_capstyle='butt')
            axis.plot(x[2:], y[2:], color=sm.to_rgba(np.log2(lw[1])),
                      linewidth=lw_r, solid_joinstyle='miter', solid_capstyle='butt')
        else:
            axis.plot(x[:2], y[:2], color='k',
                      linewidth=lw_l, solid_joinstyle='miter', solid_capstyle='butt')
            axis.plot(x[2:], y[2:], color='k',
                      linewidth=lw_r, solid_joinstyle='miter', solid_capstyle='butt')

        axis.plot(x[1:3], y[1:3], color='k', linewidth=1.0,
                  solid_joinstyle='miter', solid_capstyle='butt')

    if colorbar and cmap != 'none':
        cb = plt.colorbar(sm, ax=axis)
        cb.ax.set_ylabel('log(Number of points)')

    axis.set_xticks([])
    for side in ('right', 'top', 'bottom'):
        axis.spines[side].set_visible(False)

    axis.set_ylabel('lambda value' if vf == 'lambda_val' else 'distance')

    return axis


# ---------------------------------------------------------------------------
# Public: overlay ellipses — binary
# ---------------------------------------------------------------------------

def plot_selected_bin(
    selected_clusters,
    plot_data,
    axis,
    label_clusters=False,
    selection_palette=None,
    **kwargs,
):
    """Overlay ellipses on a binary condensed-tree plot around each cluster's stem.

    The ellipse wraps the vertical stem of each selected node: a narrow oval
    centred on the stem's x coordinate, spanning its y range.

    Parameters
    ----------
    selected_clusters : list[int]
    plot_data : dict
        As returned by :func:`get_plot_data_bin`.
    axis : matplotlib axis
    label_clusters : bool, optional (default False)
    selection_palette : list, optional
    **kwargs
        Accepted and ignored.

    Returns
    -------
    artists : list
    """
    from matplotlib.patches import Ellipse

    stem_coords = plot_data['stem_coords']

    all_x = [sx for sx, _, _ in stem_coords.values()]
    x_range = (max(all_x) - min(all_x)) if len(all_x) > 1 else 10.0
    ellipse_half_width = 0.02 * x_range

    all_y = [v for _, sy_bot, sy_top in stem_coords.values() for v in (sy_bot, sy_top)]
    y_range = (max(all_y) - min(all_y)) if all_y else 1.0
    y_pad = 0.01 * y_range

    artists = []

    for i, c in enumerate(selected_clusters):
        if c not in stem_coords:
            continue

        sx, sy_bottom, sy_top = stem_coords[c]
        ell_width  = 2 * ellipse_half_width
        ell_height = (sy_top - sy_bottom) + 2 * y_pad
        center     = (sx, (sy_bottom + sy_top) / 2.0)

        oval_color = (
            selection_palette[i]
            if selection_palette is not None and len(selection_palette) > i
            else 'r'
        )
        ellipse = Ellipse(
            center,
            ell_width,
            ell_height,
            facecolor='none',
            edgecolor=oval_color,
            linewidth=1.5,
            zorder=250,
        )
        axis.add_artist(ellipse)
        artists.append(ellipse)

        if label_clusters:
            text = axis.annotate(
                str(i),
                xy=center,
                xytext=(sx + ellipse_half_width * 1.2, sy_top + y_pad),
                horizontalalignment='left',
                verticalalignment='bottom',
            )
            artists.append(text)

    return artists


# ===========================================================================
# SHARED
# ===========================================================================

def clear_selected_clusters(artists):
    """Remove ellipse / annotation artists added by any ``plot_selected_*`` call."""
    for artist in artists:
        artist.remove()


# ===========================================================================
# INTERACTIVE ENTRY POINT
# ===========================================================================

def interactive_condensed(
    condensed_tree,
    solutions=None,
    binary_tree=False,
    figsize=(10, 8),
    label_clusters=False,
    selection_palette=None,
    **plot_kwargs,
):
    """Interactive ipywidgets slider for browsing a collection of cluster solutions.

    Draws the condensed tree once and re-draws ellipses around the selected
    clusters each time the slider moves.

    Parameters
    ----------
    condensed_tree : numpy recarray
        Fields: ``parent``, ``child``, ``child_size``, and either
        ``distance`` or ``lambda_val`` (auto-detected).
    solutions : list[list[int]] or list[int] or None, optional
        Each element is a list of cluster node-ids for one solution.
        A single flat list (one solution) is wrapped automatically.
        ``None`` or ``[]`` renders the tree with no highlights and no slider.
    binary_tree : bool, optional
        ``False`` (default) → non-binary icicle plot.
        ``True``            → binary dendrogram plot.
    figsize : tuple, optional (default (10, 8))
    label_clusters : bool, optional (default False)
    selection_palette : list, optional
        Colours for highlighted ellipses.
    **plot_kwargs
        Forwarded to the underlying ``get_plot_data_*`` and
        ``plot_condensed_*`` functions.  Unknown keys are silently ignored.

    Returns
    -------
    slider : ipywidgets.IntSlider or None
        ``None`` when *solutions* is empty/None (no slider is shown).
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError as e:
        raise ImportError(
            "ipywidgets is required for interactive plotting. "
            "Install it with `pip install ipywidgets`."
        ) from e

    # ------------------------------------------------------------------
    # Normalise solutions
    # ------------------------------------------------------------------
    if solutions is None or (hasattr(solutions, '__len__') and len(solutions) == 0):
        solutions = []
    elif solutions and not isinstance(solutions[0], (list, tuple, np.ndarray)):
        # Single flat list of cluster ids → wrap as one solution
        solutions = [list(solutions)]

    # ------------------------------------------------------------------
    # Pre-compute plot data once, routing nb-specific kwargs appropriately
    # ------------------------------------------------------------------
    _NB_DATA_KEYS = {'leaf_separation', 'log_size', 'max_rectangle_per_icicle', 'value_field'}

    if binary_tree:
        plot_data  = get_plot_data_bin(condensed_tree, **plot_kwargs)
        _plot_fn   = lambda ax: plot_condensed_bin(condensed_tree, plot_data, axis=ax, **plot_kwargs)
        _select_fn = plot_selected_bin
    else:
        nb_data_kwargs  = {k: plot_kwargs[k] for k in _NB_DATA_KEYS if k in plot_kwargs}
        nb_plot_kwargs  = {k: v for k, v in plot_kwargs.items() if k not in _NB_DATA_KEYS}
        # density must reach get_plot_data_nb (controls root y-coord) as well
        # as plot_condensed_nb (controls axis direction) — pass it explicitly.
        if 'density' in plot_kwargs:
            nb_data_kwargs['density'] = plot_kwargs['density']
        plot_data  = get_plot_data_nb(condensed_tree, **nb_data_kwargs)
        _plot_fn   = lambda ax: plot_condensed_nb(condensed_tree, plot_data, axis=ax, **nb_plot_kwargs)
        _select_fn = plot_selected_nb

    # ------------------------------------------------------------------
    # No-solutions case: render tree only, no slider
    # ------------------------------------------------------------------
    if not solutions:
        output = widgets.Output()
        with output:
            fig, ax = plt.subplots(figsize=figsize)
            _plot_fn(ax)
            plt.show()
            plt.close(fig)
        display(output)
        return None

    # ------------------------------------------------------------------
    # Normal case: slider + per-solution ellipses
    # ------------------------------------------------------------------
    output = widgets.Output()
    info   = widgets.HTML()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(solutions) - 1,
        step=1,
        description='Solution',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px'),
    )

    def redraw(solution_idx):
        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)
            _plot_fn(ax)
            selected = solutions[solution_idx]
            _select_fn(
                selected, plot_data, ax,
                label_clusters=label_clusters,
                selection_palette=selection_palette,
            )
            info.value = (
                f"<b>Solution:</b> {solution_idx} &nbsp;&nbsp; "
                f"<b>Clusters:</b> {len(selected)}"
            )
            plt.show()
            plt.close(fig)

    slider.observe(lambda change: redraw(change['new']), names='value')

    display(widgets.VBox([slider, info, output]))
    redraw(0)

    return slider