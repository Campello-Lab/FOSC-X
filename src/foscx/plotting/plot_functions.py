# fosc/plotting/plot_functions.py
"""
Scatter-plot visualisation utilities for FOSC candidate solutions.

Public API
----------
get_scatter_data     – pre-compute 2-D projection + per-solution label arrays
plot_fosc_solution   – draw a single static scatter for one solution
interactive_fosc     – ipywidgets slider browser across all candidate solutions
_plot_fosc           – legacy entry point (kept for backward compatibility)
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.colors import hsv_to_rgb

    return plt, mcolors, cm, hsv_to_rgb


def _get_pca():
    from sklearn.decomposition import PCA
    return PCA


def _get_umap():
    try:
        from umap import UMAP as _UMAP
        return _UMAP, True
    except Exception:
        return None, False


def _get_tsne():
    # Prefer openTSNE (much faster); fall back to sklearn.
    try:
        from openTSNE import TSNE as _TSNE
        return _TSNE, "opentsne"
    except Exception:
        pass
    try:
        from sklearn.manifold import TSNE as _TSNE
        return _TSNE, "sklearn"
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _build_color_map(labels):
    """Return a dict mapping label → RGBA tuple.

    Clusters are coloured with well-separated HSV hues (golden-ratio spacing),
    sorted largest-first so the biggest cluster gets the most distinguishable
    hue.  Noise points (label == -1) receive a neutral grey.
    """
    _, _, _, hsv_to_rgb = _get_matplotlib()

    labels = np.asarray(labels)
    noise_label = -1
    cluster_labels = np.unique(labels)
    cluster_labels = cluster_labels[cluster_labels != noise_label]

    cluster_sizes = [(lab, int(np.sum(labels == lab))) for lab in cluster_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    golden_ratio = 0.618033988749895
    color_map: dict = {}
    for i, (lab, _) in enumerate(cluster_sizes):
        hue = (i * golden_ratio) % 1.0
        rgb = hsv_to_rgb([hue, 0.78, 0.88])
        color_map[lab] = (*rgb, 1.0)

    color_map[noise_label] = (0.55, 0.55, 0.55, 0.55)
    return color_map


def _labels_to_rgba(labels, color_map):
    return np.array([color_map[int(l)] for l in labels])


# ---------------------------------------------------------------------------
# Public: pre-compute scatter data
# ---------------------------------------------------------------------------

def get_scatter_data(
    fosc,
    X=None,
    projection: str = "pca",
    umap_n_neighbors: int = 12,
    umap_min_dist: float = 0.1,
    umap_n_epochs: int = 150,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 500,
    random_state=None,
):
    """Pre-compute the 2-D projection and per-solution label arrays.

    This is the expensive step (PCA / UMAP / t-SNE).  Call it once and pass
    the result to :func:`plot_fosc_solution` or :func:`interactive_fosc`.

    Parameters
    ----------
    fosc : FOSC
        A fitted FOSC instance.
    X : array-like, shape (n_samples, n_features), optional
        Original data.  Falls back to ``fosc.data`` when *None*.
    projection : {'pca', 'umap', 'tsne', 'none'}
        Dimensionality reduction applied to ``X`` when it has > 2 columns.
    umap_n_neighbors : int
        UMAP neighbours.  Smaller = faster; 10–15 is fine for visualisation.
        Default 12 is faster than UMAP's own default of 15.
    umap_min_dist : float
        UMAP minimum distance between embedded points.
    umap_n_epochs : int
        UMAP optimisation iterations.  150 cuts runtime substantially vs the
        default (200–500) with negligible visual difference.
    tsne_perplexity : float
        t-SNE perplexity.  Typical range 5–50; default 30.
    tsne_n_iter : int
        t-SNE iterations for sklearn fallback.  500 is enough for
        visualisation (sklearn default 1000).  Ignored when openTSNE is used.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``XY``           – ndarray (n_samples, 2) — projected coordinates
        ``label_sets``   – list[ndarray] — one label array per candidate solution
        ``candidates``   – list of candidate node lists (normalised)
        ``projection``   – str, the projection method actually used
        ``color_maps``   – list[dict], pre-built colour map for each solution
    """
    # ---- validate ----
    if not hasattr(fosc, "candidate_nodes_") or not hasattr(fosc, "cluster_tree_"):
        raise ValueError(
            "Provided FOSC instance does not appear to be fitted "
            "(missing candidate_nodes_ or cluster_tree_)."
        )

    # ---- data ----
    if X is None:
        X = getattr(fosc, "data", None)
        if X is None:
            raise ValueError("No data supplied and fosc.data is None.")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")

    # ---- project to 2-D ----
    actual_projection = projection
    tsne_backend = None   # set inside the tsne branch; used at fit time
    if X.shape[1] > 2 and projection != "none":
        if projection == "umap":
            _UMAP, umap_ok = _get_umap()
            if not umap_ok:
                warnings.warn("UMAP not available — falling back to PCA.")
                actual_projection = "pca"
                model = _get_pca()(n_components=2, random_state=random_state)
            else:
                model = _UMAP(
                    n_components=2,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    n_epochs=umap_n_epochs,
                    low_memory=False,   # faster at the cost of some RAM
                    random_state=random_state,
                )

        elif projection == "tsne":
            _TSNE, tsne_backend = _get_tsne()
            if _TSNE is None:
                warnings.warn("Neither openTSNE nor sklearn found — falling back to PCA.")
                actual_projection = "pca"
                model = _get_pca()(n_components=2, random_state=random_state)
            elif tsne_backend == "opentsne":
                # openTSNE uses FFT-accelerated Barnes-Hut; n_jobs=-1 for all cores
                model = _TSNE(
                    n_components=2,
                    perplexity=tsne_perplexity,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                # sklearn fallback
                model = _TSNE(
                    n_components=2,
                    perplexity=tsne_perplexity,
                    n_iter=tsne_n_iter,
                    method="barnes_hut",
                    random_state=random_state,
                )

        else:
            actual_projection = "pca"
            model = _get_pca()(n_components=2, random_state=random_state)

        if projection == "tsne" and tsne_backend == "opentsne":
            # openTSNE's fit() returns the embedding directly
            XY = np.array(model.fit(X))[:, :2]
        else:
            XY = np.array(model.fit_transform(X))[:, :2]
    elif X.shape[1] >= 2:
        XY = X[:, :2]
        actual_projection = "none"
    else:
        raise ValueError("X must have at least 2 columns when projection='none'.")

    # ---- candidate solutions ----
    candidates = fosc.candidate_nodes_
    if not candidates:
        raise ValueError("No candidate solutions found in fosc.candidate_nodes_.")

    normalised = [
        list(cand) if isinstance(cand, (list, tuple)) else [cand]
        for cand in candidates
    ]

    label_sets = [fosc.get_labels(nodes=cand) for cand in normalised]
    color_maps = [_build_color_map(labels) for labels in label_sets]

    return {
        "XY": XY,
        "label_sets": label_sets,
        "candidates": normalised,
        "projection": actual_projection,
        "color_maps": color_maps,
    }


# ---------------------------------------------------------------------------
# Public: draw a single solution
# ---------------------------------------------------------------------------

def plot_fosc_solution(
    scatter_data: dict,
    solution_idx: int = 0,
    axis=None,
    figsize: tuple = (7, 6),
    point_size: float = 8.0,
    alpha: float = 0.85,
    show: bool = False,
):
    """Draw a scatter plot for one candidate solution.

    Parameters
    ----------
    scatter_data : dict
        As returned by :func:`get_scatter_data`.
    solution_idx : int
        Index into ``scatter_data['label_sets']``.
    axis : matplotlib Axes or None
        Draw onto an existing axes; create a new figure when *None*.
    figsize : tuple
        Figure size (only used when *axis* is None).
    point_size : float
        Marker area in points² passed to ``scatter``.
    alpha : float
        Marker transparency.
    show : bool
        Call ``plt.show()`` at the end (handy for non-interactive scripts).

    Returns
    -------
    fig, ax : tuple
    """
    plt, mcolors, cm, _ = _get_matplotlib()

    XY         = scatter_data["XY"]
    labels     = scatter_data["label_sets"][solution_idx]
    color_map  = scatter_data["color_maps"][solution_idx]
    n_sol      = len(scatter_data["label_sets"])
    proj_name  = scatter_data["projection"].upper()

    rgba = _labels_to_rgba(labels, color_map)

    unique_labels = np.unique(labels)
    n_clusters    = int(np.sum(unique_labels != -1))
    n_noise       = int(np.sum(np.asarray(labels) == -1))

    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax  = axis
        fig = ax.get_figure()

    # ---- styled background grid ----
    ax.set_facecolor("#f5f5f5")
    ax.grid(
        True,
        color="white",
        linewidth=0.9,
        linestyle="-",
        zorder=0,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)

    # ---- scatter ----
    ax.scatter(
        XY[:, 0],
        XY[:, 1],
        s=point_size,
        c=rgba,
        alpha=alpha,
        linewidths=0.0,
        zorder=3,
    )

    # ---- labels ----
    x_label, y_label = (
        ("UMAP 1", "UMAP 2")      if scatter_data["projection"] == "umap"  else
        ("PC 1",   "PC 2")        if scatter_data["projection"] == "pca"   else
        ("Feature 1", "Feature 2")
    )
    ax.set_xlabel(x_label, fontsize=10, color="#444444")
    ax.set_ylabel(y_label, fontsize=10, color="#444444")
    ax.set_title(
        f"Solution {solution_idx + 1} / {n_sol}   —   "
        f"{n_clusters} cluster{'s' if n_clusters != 1 else ''}  "
        f"({n_noise} noise)",
        fontsize=11,
        color="#222222",
        pad=8,
    )
    ax.tick_params(colors="#666666", labelsize=8)

    if show:
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# Public: interactive ipywidgets browser
# ---------------------------------------------------------------------------

def interactive_fosc(
    fosc,
    X=None,
    projection: str = "pca",
    umap_n_neighbors: int = 12,
    umap_min_dist: float = 0.1,
    umap_n_epochs: int = 150,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 500,
    random_state=None,
    figsize: tuple = (7, 6),
    point_size: float = 8.0,
    alpha: float = 0.85,
    scatter_data: dict | None = None,
):
    """Interactive ipywidgets slider for browsing FOSC candidate solutions.

    The 2-D projection is computed **once** (or supplied via *scatter_data*);
    only colours are updated on each slider move.

    Parameters
    ----------
    fosc : FOSC
        A fitted FOSC instance.
    X : array-like or None
        Original data (falls back to ``fosc.data``).
    projection : {'pca', 'umap', 'tsne', 'none'}
        Projection method forwarded to :func:`get_scatter_data`.
    umap_n_neighbors, umap_min_dist, umap_n_epochs
        UMAP settings forwarded to :func:`get_scatter_data`.
    tsne_perplexity, tsne_n_iter
        t-SNE settings forwarded to :func:`get_scatter_data`.
    random_state : int or None
        Seed for reproducibility.
    figsize : tuple
        Figure size.
    point_size : float
        Marker size (points²).
    alpha : float
        Marker transparency.
    scatter_data : dict or None
        Pre-computed result of :func:`get_scatter_data`.  Pass this to skip
        the projection step entirely.

    Returns
    -------
    slider : ipywidgets.IntSlider
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError as exc:
        raise ImportError(
            "ipywidgets is required for interactive_fosc. "
            "Install it with `pip install ipywidgets`."
        ) from exc

    # ---- pre-compute once ----
    if scatter_data is None:
        scatter_data = get_scatter_data(
            fosc,
            X=X,
            projection=projection,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_n_epochs=umap_n_epochs,
            tsne_perplexity=tsne_perplexity,
            tsne_n_iter=tsne_n_iter,
            random_state=random_state,
        )

    n_solutions = len(scatter_data["label_sets"])

    # ---- widgets ----
    output = widgets.Output()
    info   = widgets.HTML()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_solutions - 1,
        step=1,
        description="Solution",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    def redraw(idx: int):
        labels    = scatter_data["label_sets"][idx]
        n_clusters = int(np.sum(np.unique(labels) != -1))
        n_noise    = int(np.sum(np.asarray(labels) == -1))

        with output:
            clear_output(wait=True)
            fig, ax = plot_fosc_solution(
                scatter_data,
                solution_idx=idx,
                figsize=figsize,
                point_size=point_size,
                alpha=alpha,
                show=False,
            )
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        info.value = (
            f"<b>Solution:</b>&nbsp;{idx + 1} / {n_solutions}"
            f"&nbsp;&nbsp;&nbsp;"
            f"<b>Clusters:</b>&nbsp;{n_clusters}"
            f"&nbsp;&nbsp;"
            f"<b>Noise:</b>&nbsp;{n_noise}"
        )

    slider.observe(lambda change: redraw(change["new"]), names="value")

    display(widgets.VBox([slider, info, output]))
    redraw(0)

    return slider


# ---------------------------------------------------------------------------
# Legacy entry point (backward compatibility)
# ---------------------------------------------------------------------------

def _plot_fosc(
    fosc,
    X=None,
    projection: str = "pca",
    umap_n_neighbors: int = 12,
    umap_min_dist: float = 0.1,
    umap_n_epochs: int = 150,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 500,
    random_state=None,
    figsize: tuple = (7, 6),
    point_size: float = 8.0,
    alpha: float = 0.85,
    cmap: str = "tab10",    # kept for signature compatibility; ignored
    show: bool = True,
):
    """Legacy interactive plot for FOSC candidate solutions.

    .. deprecated::
        Prefer :func:`interactive_fosc` (uses ipywidgets) or
        :func:`plot_fosc_solution` (static, single solution).

    Returns ``(None, None, slider)`` so callers that unpack three values
    (``fig, ax, slider = _plot_fosc(...)``) continue to work.  The *cmap*
    and *show* arguments are accepted but ignored.
    """
    warnings.warn(
        "_plot_fosc is deprecated. Use interactive_fosc() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    scatter_data = get_scatter_data(
        fosc,
        X=X,
        projection=projection,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_n_epochs=umap_n_epochs,
        tsne_perplexity=tsne_perplexity,
        tsne_n_iter=tsne_n_iter,
        random_state=random_state,
    )

    slider = interactive_fosc(
        fosc,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        scatter_data=scatter_data,
    )

    # Return (fig, ax, slider) so legacy callers can still unpack three values.
    return slider