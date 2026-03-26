# fosc/plotting/plot_functions.py
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

def _plot_tree(
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

def plot_tree_base_fast_old(
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
    """
    Tree plot for FastHierarchy.

    Parameters
    ----------
    tree : FastHierarchy
        The hierarchy to plot.
    figsize : tuple
        Figure size.
    max_bar_width : float   
        Maximum width of bars.
    min_bar_width : float
        Minimum width of bars.
    log_width_scale : bool
        Whether to use logarithmic scaling for bar widths.
    cmap : str
        Colormap name.
    hdbscan_style : bool
        If True, inverts y-axis to match HDBSCAN style.
    root_offset : float
        Offset for root node distance if needed. (Not used if distances are provided.)
    default_distance : float
        Default distance increment for fabricated distances.
    """

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

        if not kids:
            x = leaf_x
            leaf_x += 1
        else:
            xs = [traverse(c) for c in kids]
            x = np.mean(xs)

        node_positions[n] = x

        y0 = plot_dist[n]
        y1 = max([plot_dist[c] for c in children_of(n)], default=y0)

        cs = max(1, sizes[n])
        top_w = max(scale_width(cs), min_bar_width)

        child_sum = sum(sizes[c] for c in kids)
        bot_w = max(scale_width(child_sum or cs / 2), min_bar_width / 2)

        ax.fill_betweenx(
            [y0, y1],
            [x - top_w / 2, x - bot_w / 2],
            [x + top_w / 2, x + bot_w / 2],
            color=cmap_obj(norm(cs)),
            linewidth=0,
            alpha=0.9,
        )

        if len(kids) > 1:
            xs = [node_positions[c] for c in kids]
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


def _interactive_highlight(
    *,
    fig,
    ax,
    node_bounds,
    selections,
    qlist=None,
    tree=None,
):
    """
    Interactive slider + click highlighting for tree plots.

    Parameters
    ----------
    fig, ax : matplotlib Figure and Axes
    node_bounds : dict
        {node_id: (x, y_bottom, y_top, top_width, bottom_width)}
    selections : list[list[int]]
        Candidate solutions (lists of node ids)
    qlist : list[float] or None
        Candidate quality scores
    tree : FastHierarchy or None
        Optional, used only for richer click annotations
    """

    if not selections:
        plt.show()
        return

    # ----------------------------
    # Layout
    # ----------------------------
    fig.subplots_adjust(bottom=0.22)

    ax_slider = plt.axes([0.2, 0.06, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Candidate",
        1,
        len(selections),
        valinit=1,
        valstep=1,
    )

    ax_info = plt.axes([0.2, 0.02, 0.6, 0.03])
    ax_info.axis("off")
    info_text = ax_info.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=10
    )

    # ----------------------------
    # Highlight storage
    # ----------------------------
    patches = []

    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        ha="left",
        va="bottom",
        fontsize=9,
        visible=False,
        zorder=999999,
    )

    # ----------------------------
    # Slider update
    # ----------------------------
    def update(val):
        idx = int(slider.val) - 1

        # remove old patches
        for p in patches:
            p.remove()
        patches.clear()

        # draw new highlights
        for nid in selections[idx]:
            if nid not in node_bounds:
                continue

            x, yb, yt, topw, botw = node_bounds[nid]
            mid_y = 0.5 * (yb + yt)
            height = abs(yt - yb) * 1.1

            ellipse = Ellipse(
                (x, mid_y),
                width=(topw + botw) / 2,
                height=height,
                edgecolor="red",
                facecolor="none",
                linewidth=1.8,
            )
            ax.add_patch(ellipse)
            patches.append(ellipse)

        # update title / info
        qval = qlist[idx] if (qlist and idx < len(qlist)) else None

        title = f"Candidate Solution {idx + 1}"
        if qval is not None:
            title += f" | Quality: {qval:.3f}"
        ax.set_title(title, fontsize=12)

        info = f"Candidate {idx + 1}/{len(selections)} | {len(selections[idx])} clusters"
        if qval is not None:
            info += f" | Quality: {qval:.3f}"
        info_text.set_text(info)

        annotation.set_visible(False)
        fig.canvas.draw_idle()

    # ----------------------------
    # Click inspection
    # ----------------------------
    def on_click(event):
        if event.inaxes is not ax:
            return

        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return

        min_dist = float("inf")
        nearest = None

        for nid, (x, yb, yt, *_rest) in node_bounds.items():
            mid_y = 0.5 * (yb + yt)
            d = ((x - cx) ** 2 + (mid_y - cy) ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
                nearest = nid

        if min_dist > 0.5 or nearest is None:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return

        # ------------------------
        # Build annotation text
        # ------------------------
        lines = [f"Node ID: {nearest}"]

        if tree is not None:
            try:
                lines.append(f"Size: {tree.sizes[nearest]}")
                lines.append(f"Dist/Lambda: {tree.distance[nearest]:.3f}")
                if hasattr(tree, "clusteval"):
                    lines.append(f"Quality: {tree.clusteval[nearest]:.3f}")
            except Exception:
                pass

        idx = int(slider.val) - 1
        if qlist and idx < len(qlist):
            try:
                lines.append(f"Candidate Q: {qlist[idx]:.3f}")
            except Exception:
                pass

        annotation.xy = (cx, cy)
        annotation.set_text("\n".join(lines))
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    # ----------------------------
    # Wire up
    # ----------------------------
    slider.on_changed(update)
    fig.canvas.mpl_connect("button_press_event", on_click)

    update(1)
    plt.show()




def _plot_fosc(
    fosc,
    X=None,
    projection="pca",      
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    random_state=None,
    figsize=(8, 6),
    point_size=30,
    alpha=0.9,
    cmap="tab10",
    show=True,
):
    """
    Interactive plot for FOSC candidate solutions.
    Parameters
    ---------- 
    fosc : FOSC
        Fitted FOSC instance.
    X : array-like, shape (n_samples, n_features), optional
        Original data used for clustering. Required for plotting.
        If None, uses the data provided during fit(). Default is None.
    projection : str, optional
        Dimensionality reduction method to use ('umap', 'pca', or 'none').
    umap_n_neighbors : int
        Number of neighbors for UMAP (if used).
    umap_min_dist : float
        Minimum distance for UMAP (if used).
    random_state : int or None
        Random state for reproducibility.
    figsize : tuple
        Figure size.
    point_size : int
        Size of scatter plot points.
    alpha : float
        Alpha transparency for points.
    cmap : str
        Colormap name for cluster coloring.
    show : bool
        Whether to show the plot immediately.
    """

    # ---------------------------
    # Validate fitted estimator
    # ---------------------------
    if not hasattr(fosc, "candidate_Clist_") or not hasattr(fosc, "cluster_tree_"):
        raise ValueError(
            "Provided FOSC instance does not appear to be fitted "
            "(missing candidate_Clist_ or cluster_tree_)."
        )

    # ---------------------------
    # Get data
    # ---------------------------
    if X is None:
        X = getattr(fosc, "data", None)
        if X is None:
            raise ValueError("No data supplied and fosc.data is None.")

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")

    # ---------------------------
    # Project to 2D
    # ---------------------------
    if X.shape[1] > 2 and projection != "none":
        if projection == "umap":
            if not _UMAP_AVAILABLE:
                warnings.warn("UMAP not available — falling back to PCA.")
                proj_model = PCA(n_components=2, random_state=random_state)
            else:
                proj_model = _UMAP(
                    n_components=2,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    random_state=random_state,
                )
        else:
            proj_model = PCA(n_components=2, random_state=random_state)

        XY = proj_model.fit_transform(X)[:, :2]
    else:
        if X.shape[1] < 2:
            raise ValueError("X must have at least 2 columns when projection='none'.")
        XY = X[:, :2]

    # ---------------------------
    # Prepare candidate label sets
    # ---------------------------
    candidates = fosc.candidate_Clist_
    if not candidates:
        raise ValueError("No candidate solutions found in fosc.candidate_Clist_.")

    # Normalize: ensure list[list[node_id]]
    normalized = []
    for cand in candidates:
        if isinstance(cand, (list, tuple)):
            normalized.append(list(cand))
        else:
            normalized.append([cand])

    label_sets = [
        fosc.get_labels(Nodes=cand)
        for cand in normalized
    ]

    # ---------------------------
    # Color mapping
    # ---------------------------
    import matplotlib.cm as mpl_cm
    cmap_obj = mpl_cm.get_cmap(cmap)

    def labels_to_colors(labels):
        unique = np.unique(labels)
        color_map = {}
        nonzero = unique[unique != 0]

        for i, lab in enumerate(nonzero):
            color_map[lab] = cmap_obj(i % cmap_obj.N)

        color_map[0] = (0.6, 0.6, 0.6, 0.6)  # noise
        return [color_map[int(l)] for l in labels]

    # ---------------------------
    # Initial plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=figsize)
    colors0 = labels_to_colors(label_sets[0])

    sc = ax.scatter(
        XY[:, 0],
        XY[:, 1],
        s=point_size,
        c=colors0,
        alpha=alpha,
    )

    ax.set_title(f"Candidate 1 / {len(label_sets)}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.subplots_adjust(bottom=0.18)

    # ---------------------------
    # Slider
    # ---------------------------
    ax_slider = plt.axes([0.12, 0.05, 0.76, 0.04], facecolor="lightgoldenrodyellow")
    slider = Slider(
        ax_slider,
        "Solution",
        1,
        len(label_sets),
        valinit=1,
        valstep=1,
        valfmt="%0.0f",
    )

    def _update(val):
        idx = int(slider.val) - 1
        labels = label_sets[idx]
        colors = labels_to_colors(labels)
        sc.set_facecolors(colors)
        sc.set_edgecolors(colors)
        ax.set_title(f"Candidate {idx + 1} / {len(label_sets)}")
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    if show:
        plt.show()

    return fig, ax, slider
