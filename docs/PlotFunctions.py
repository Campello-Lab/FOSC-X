from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse
import warnings

try:
    from umap import UMAP as _UMAP
    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False
from sklearn.decomposition import PCA

def plot_tree_base_fast(
    tree,
    figsize=(12, 6),
    max_bar_width=1.0,
    min_bar_width=0.05,
    log_width_scale=False,
    cmap="viridis",
    hdbscan_style=True,
    root_offset=1.0,
    default_distance=1.0,
):
    fig, ax = plt.subplots(figsize=figsize)

    parent = tree.parent
    children_flat = tree.children_flat
    children_off = tree.children_off
    distance = tree.distance
    sizes = tree.sizes
    is_noise = getattr(tree, "is_noise", np.zeros(tree.N, dtype=np.int8))

    N = tree.N
    root = int(np.where(parent == -1)[0][0])

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    max_size = max(1, sizes.max())
    norm = mcolors.Normalize(vmin=1, vmax=max_size)
    cmap_obj = cm.get_cmap(cmap)

    def scale_width(sz):
        sz = max(sz, 1e-9)
        if log_width_scale:
            return max_bar_width * np.log1p(sz) / np.log1p(max_size)
        return max_bar_width * sz / max_size

    # --------------------------------------------------
    # fabricate distances if needed
    # --------------------------------------------------
    if tree._depths is None:
        tree.compute_depths()
    depths = tree._depths

    plot_dist = distance.copy()
    finite_mask = np.isfinite(plot_dist)

    max_provided = np.nanmax(plot_dist) if np.any(finite_mask) else 0.0

    missing = ~finite_mask
    if np.any(missing):
        plot_dist[missing] = (
            max_provided
            + (depths.max() - depths[missing] + 1) * default_distance
        )

    # --------------------------------------------------
    node_positions = {}
    node_bounds = {}
    leaf_x = 0

    def children_of(n):
        s = children_off[n]
        e = children_off[n + 1]
        return children_flat[s:e]

    # --------------------------------------------------
    def traverse(n):
        nonlocal leaf_x

        kids = [c for c in children_of(n) if not is_noise[c]]

        # x-position
        if not kids:
            x = leaf_x
            leaf_x += 1
        else:
            xs = [traverse(c) for c in kids]
            x = np.mean(xs)

        node_positions[n] = x

        node_d = plot_dist[n]

        if parent[n] == -1:
            parent_d = node_d + root_offset
        else:
            parent_d = plot_dist[parent[n]]

        # -----------------------------
        # vertical span logic
        # -----------------------------
        if hdbscan_style:
            y0 = node_d
            y1 = max([plot_dist[c] for c in children_of(n)], default=node_d)
        else:
            y0 = parent_d
            y1 = node_d

        # widths (always computed)
        cs = max(1, sizes[n])
        top_w = max(scale_width(cs), min_bar_width)

        child_sum = sum(sizes[c] for c in kids)
        bot_w = max(scale_width(child_sum or cs / 2), min_bar_width / 2)

        # vertical bar (ALWAYS draw)
        ax.fill_betweenx(
            [y0, y1],
            [x - top_w / 2, x - bot_w / 2],
            [x + top_w / 2, x + bot_w / 2],
            color=cmap_obj(norm(cs)),
            linewidth=0,
            alpha=0.9,
        )

        # horizontal connector
        if len(kids) > 1:
            xs = [node_positions[c] for c in kids]
            if hdbscan_style:
                ax.hlines(y1, min(xs), max(xs), colors="k", linewidth=0.8)
            else:
                ax.hlines(y1, min(xs), max(xs), colors="k", linewidth=0.8)

        node_bounds[n] = (x, y0, y1, top_w, bot_w)
        return x

    traverse(root)

    ax.set_ylabel("Lambda" if hdbscan_style else "Distance")
    if hdbscan_style:
        ax.invert_yaxis()

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    plt.colorbar(sm, ax=ax, label="Cluster Size")

    plt.tight_layout()
    return fig, ax, node_positions, node_bounds

def plot_tree_with_highlight(tree, highlight_nodes, figsize=(12,6)):

    # draw the original tree exactly as before
    fig, ax, node_positions, node_bounds = plot_tree_base_fast(
        tree,
        figsize=figsize
    )

    # remove colourbar (paper version)
    if len(fig.axes) > 1:
        fig.axes[-1].remove()

    # overlay highlights
    for nid in highlight_nodes:

        if nid not in node_bounds:
            continue

        x, yb, yt, topw, botw = node_bounds[nid]

        mid_y = 0.5 * (yb + yt)
        height = abs(yt - yb) * 1
        width = max(topw, botw) * 1

        ellipse = Ellipse(
            (x, mid_y),
            width=width,
            height=height,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
            zorder=1000
        )

        ax.add_patch(ellipse)

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

def plot_Clustering_2d(
    ax,
    X,
    labels,
    point_size=3,
    alpha=0.9,
    cmap="tab10"
):
    """
    Plot a clustering partition in 2D.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    X : array (n_samples, 2)
        2D coordinates.
    labels : array (n_samples,)
        Cluster labels (0 = noise).
    point_size : float
        Marker size.
    alpha : float
        Point transparency.
    cmap : str
        Colormap name for clusters.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    cmap_obj = mpl_cm.get_cmap(cmap)

    markers = ['o','s','^','D','v','P','X','<','>','*']

    unique = np.unique(labels)

    # -------------------------
    # plot noise first
    # -------------------------
    if 0 in unique:
        mask = labels == 0
        ax.scatter(
            X[mask,0],
            X[mask,1],
            s=point_size,
            color=(0.6,0.6,0.6,0.6),
            alpha=alpha,
        )

    # -------------------------
    # plot clusters
    # -------------------------
    clusters = unique[unique != 0]
    
    # --- NEW: sort clusters by size (largest first)
    sizes = [(lab, np.sum(labels == lab)) for lab in clusters]
    sizes.sort(key=lambda x: x[1], reverse=True)
    clusters = [lab for lab, _ in sizes]

    for i, lab in enumerate(clusters):

        mask = labels == lab

        ax.scatter(
            X[mask,0],
            X[mask,1],
            s=point_size,
            color=cmap_obj(i % cmap_obj.N),
            marker=markers[i % len(markers)],
            alpha=alpha,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")