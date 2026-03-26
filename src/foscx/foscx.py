import json
import warnings
from importlib import resources

import numpy as np
import pandas as pd
from jsonschema import Draft7Validator
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.cluster.hierarchy import is_valid_linkage
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from . import _ensure_numba_cache
from .efosc import _efosc
from .hierarchy.condense_tree import _condense_tree, _scipy_to_condensed
from .hierarchy.hierarchy import Cluster_Tree
from .hierarchy.tree_numba import _postorder
from .plotting.plot_functions import (
    _interactive_highlight,
    _plot_fosc,
    _plot_tree,
)

with resources.open_text("fosc", "hierarchy.schema.json") as f:
    FOSC_JSON_SCHEMA = json.load(f)

FOSC_SCHEMA_VALIDATOR = Draft7Validator(FOSC_JSON_SCHEMA)


class FOSCX(BaseEstimator):
    def __init__(
        self,
        top_M: int = 1,
        kmin: int = None,
        kmax: int = None,
        *,
        min_cluster_size: int = None,
        quality_measure: str = "stability",
        singletons_as_noise: bool = False,
        keep_noise_quality: bool = None,
        nearest_neighbors: int = None,
        metric: str = "euclidean",
        density: bool = False,
    ):

        """
        FOSC: Framework for Optimal Extraction of Clusters from Hierarchies.

        Selects one or more high-quality clusterings from a hierarchical clustering
        tree using dynamic programming and user-defined quality measures.

        Parameters
        ----------
        top_M : int, optional
            Maximum number of candidate clusterings to return. FOSC returns the
            top ``top_M`` solutions ranked by quality. Increasing this allows
            exploration of alternative clusterings at modest additional cost.

            Default is 1.

        kmin : int, optional
            Minimum number of clusters allowed in the final solutions. When set,
            only clusterings with at least ``kmin`` clusters are considered.
            This constraint is enforced during optimisation, not post hoc.

            Default is None (no lower bound).

        kmax : int, optional
            Maximum number of clusters allowed in the final solutions. When set,
            only clusterings with at most ``kmax`` clusters are considered.
            Together with ``kmin``, this restricts the feasible solution space.

            Default is None (no upper bound).

        min_cluster_size : int, optional
            Minimum cluster size used for tree condensation. Clusters smaller than
            this threshold are removed from the hierarchy, and their descendant
            leaves are treated as noise. This effectively collapses low-support
            branches into their parent, simplifying the tree and often improving
            stability-based extraction.

            Default is None (no condensation). Not available for JSON trees with
            pre-computed quality.

        singletons_as_noise : bool, optional
            If True, singleton clusters (clusters of size 1) are treated as noise.
            Unlike ``min_cluster_size``, this does not modify the tree structure,
            but instead affects how leaf nodes are interpreted during optimisation.

            Default is False.

            For JSON trees, a complete tree (``complete_tree=True``) or explicit
            cluster sizes must be provided for this to be applied correctly.

        keep_noise_quality : bool, optional
            Controls how noise nodes contribute to the objective function.

            If True, noise retains any assigned quality (e.g. from modularity or
            precomputed values). If False, all noise nodes are assigned zero quality.

            This mainly affects graph-based measures, where noise may otherwise
            receive small (sometimes negative) scores.

            Default is True.

        quality_measure : str, optional
            Quality measure used to evaluate clusters.

            Available options are ``"stability"``, ``"modularity"``, and ``"PFCE"``.

            Default is ``"stability"``.

            Notes
            -----
            ``modularity`` requires ``nearest_neighbors`` and ``metric``, or a
            hierarchy object that provides them (e.g. HDBSCAN).

            ``PFCE`` requires an HDBSCAN object from the ``hdbscan`` package.

            Ignored for JSON trees with precomputed quality.

        nearest_neighbors : int, optional
            Number of neighbors used to construct the k-nearest neighbor graph
            for modularity.

            If not provided, this is inferred from the clustering object when
            available (e.g. ``min_samples`` or ``min_cluster_size`` in HDBSCAN).

            Larger values produce denser graphs and may smooth local structure,
            but increase computational cost.

        metric : str, optional
            Distance metric used when constructing the similarity graph for
            modularity.

            If not provided, this is inferred from the input hierarchy when
            available (e.g. HDBSCAN or sklearn AgglomerativeClustering).

            It is generally recommended to match the metric used to construct
            the hierarchy.

        density : bool, optional
            Whether the hierarchy should be treated as density-based.

            This determines how the ``distance`` values in the tree are interpreted:

            - If False: values represent merge distances (standard hierarchical clustering)
            - If True: values represent density levels (e.g. HDBSCAN λ values)

            This is automatically set when using density-based clustering objects.
            Manually setting this is primarily intended for SciPy linkage inputs.

        Attributes
        ----------
        candidates_ : pandas.DataFrame
            DataFrame containing candidate clusterings with the following columns:

            - ``quality``: Quality score of the clustering
            - ``n_clusters``: Number of clusters
            - ``selected_nodes``: List of selected node IDs representing clusters

        cluster_tree_ : :class:`~foscx.hierarchy.hierarchy.Cluster_Tree`
            The underlying hierarchical representation used by FOSC. Provides
            access to node-level information, tree structure, and quality values.

        candidate_Qlist_ : list of float
            Quality scores of candidate clusterings.

        candidate_Clist_ : list of list of int
            Selected node IDs for each candidate.

        candidate_NClist_ : list of int
            Number of clusters in each candidate solution.

        Examples
        --------
        >>> from foscx import FOSCX
        >>> model = FOSCX(top_M=3, kmin=2)
        >>> model.fit(X)
        >>> candidates = model.predict()
        >>> labels = model.get_labels(0)
        """
        _ensure_numba_cache()
        
        self.top_M = top_M
        self.kmin = kmin
        self.kmax = kmax

        self.min_cluster_size = min_cluster_size  # If min_samples != None, condense the tree based on value. This includes min_samples = 1?
        self.nearest_neighbors = nearest_neighbors
        self.quality_measure = quality_measure
        self.singletons_as_noise = singletons_as_noise
        self.keep_noise_quality = keep_noise_quality
        self.metric = metric
        self.density = density

        self.source = None
        self.hdbscan_ = False
        self.min_samples = None

    @staticmethod
    def _validate_positive_int_or_none(value, name, *, minimum=1):
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"'{name}' must be an integer or None, got {type(value)}.")
        if value < minimum:
            raise ValueError(f"'{name}' must be >= {minimum}, got {value}.")

    def _validate_runtime_params(self, *, top_M, kmin, kmax):
        self._validate_positive_int_or_none(top_M, "top_M")
        self._validate_positive_int_or_none(kmin, "kmin")
        self._validate_positive_int_or_none(kmax, "kmax")
        if kmin is not None and kmax is not None and kmin > kmax:
            raise ValueError(f"'kmin' ({kmin}) cannot be greater than 'kmax' ({kmax}).")

    def _validate_constructor_params(self):
        self._validate_runtime_params(top_M=self.top_M, kmin=self.kmin, kmax=self.kmax)
        self._validate_positive_int_or_none(
            self.min_cluster_size, "min_cluster_size"
        )
        self._validate_positive_int_or_none(
            self.nearest_neighbors, "nearest_neighbors"
        )
        if self.min_cluster_size == 1:
            warnings.warn(
                "min_cluster_size=1 will not condense the tree; the original hierarchy will be used.",
                stacklevel=2,
            )
        if self.quality_measure == "EOM":
            warnings.warn(
                "quality_measure='EOM' is treated as an alias for 'stability'.",
                stacklevel=2,
            )

    def __repr__(self):
        return (
            f"FOSC(top_M={self.top_M}, "
            f"quality_measure='{self.quality_measure}', "
            f"min_cluster_size={self.min_cluster_size}, "
            f"density={self.density})"
        )

    def fit(self, X, y=None, **params):
        """
        Fit FOSC to a hierarchical clustering tree.

        Parameters
        ----------
        X : object, ndarray, or str
            Representation of a hierarchical clustering. Supported inputs include:

            - HDBSCAN clustering object  
            - sklearn ``AgglomerativeClustering`` instance  
            - SciPy linkage matrix  
            - JSON tree (file path or dictionary)  

        y : ndarray, optional
            Original data or similarity structure. Required for graph-based quality
            measures (e.g. ``"modularity"``) and visualisation.

        **params : dict, optional
            Additional parameters passed to :meth:`set_params`.

        Returns
        -------
        self : FOSC
            Fitted estimator.
        """

        
        if X is None:
            raise ValueError("Cluster Tree must be supplied")

        self.data_ = y

        if params:
            try:
                self.set_params(**params)
            except ValueError as exc:
                raise ValueError(f"Invalid fit() parameters: {exc}") from exc

        self._validate_constructor_params()

        self._raw_tree_ = X

        # HDBSCAN (hdbscan/fast_hdbscan)
        if hasattr(X, "condensed_tree_") and hasattr(X, "min_cluster_size"):
            tree = X.single_linkage_tree_.to_numpy()

            if not self.min_cluster_size:
                self.min_cluster_size = X.min_cluster_size

            self.min_samples = getattr(X, "min_samples", X.min_cluster_size)

            self.density = True
            self.hdbscan_ = True
            raw_data = getattr(X, "_raw_data", None)
            if y is not None and raw_data is not None:
                warnings.warn(
                    "Both `y` and clustering-object raw data were provided; using explicit `y` from fit().",
                    stacklevel=2,
                )
            else:
                self.data_ = raw_data
            if hasattr(X, "minimum_spanning_tree_"):
                self.mst = X.minimum_spanning_tree_.to_numpy()
            self.metric = getattr(X, "metric", "euclidean")
            self.source = "HDBSCAN"

        # SKLEARN HDBSCAN
        elif getattr(type(X), "__module__", "").startswith("sklearn.cluster._hdbscan"):
            tree = X._single_linkage_tree_
            tree = structured_to_unstructured(tree, dtype=np.float64)

            if not self.min_cluster_size:
                self.min_cluster_size = X.min_cluster_size

            self.min_samples = getattr(X, "min_samples", X.min_cluster_size)

            self.density = True
            self.hdbscan_ = True
            self.metric = getattr(X, "metric", "euclidean")
            self.source = "SKLEARN_HDBSCAN"

        # SKLEARN agglomerative
        elif getattr(type(X), "__module__", "").startswith(
            "sklearn.cluster._agglomerative"
        ):
            if not hasattr(X, "distances_"):
                raise ValueError(
                    "For sklearn AgglomerativeClustering, distances_ must be available (set compute_distances=True or n_clusters=None)"
                )
            counts = np.zeros(X.children_.shape[0])
            n_samples = len(X.labels_)
            for i, merge in enumerate(X.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count
            tree = np.column_stack([X.children_, X.distances_, counts]).astype(float)
            self.metric = getattr(X, "metric", None)
            self.source = "SKLEARN"

            self.density = getattr(self, "density", False)

        # Precomputed JSON tree
        elif isinstance(X, (str, dict)):
            # Handle various input formats: file path, JSON string, or dict
            if isinstance(X, str):
                try:
                    # Try to load as file path first
                    with open(X, encoding="utf-8") as f:
                        tree_data = json.load(f)
                except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                    # If file doesn't exist or invalid JSON, try parsing as JSON string
                    try:
                        tree_data = json.loads(X)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "Input string is neither a readable JSON file path nor a valid JSON string."
                        ) from exc
            elif isinstance(X, dict):
                tree_data = X
            else:
                raise TypeError(
                    "Input must be a file path (str), JSON string (str), or dictionary"
                )

            errors = sorted(
                FOSC_SCHEMA_VALIDATOR.iter_errors(tree_data), key=lambda e: e.path
            )
            if errors:
                msgs = []
                for err in errors:
                    path = ".".join(str(p) for p in err.path)
                    msgs.append(f"At '{path}': {err.message}")
                raise ValueError("JSON schema validation failed:\n" + "\n".join(msgs))

            tree = tree_data.get("tree") if isinstance(tree_data, dict) else tree_data
            is_complete_tree_ = tree_data.get("complete_tree", None)
            self.density = tree_data.get("density", False)
            self.condensed_simplified_tree = tree_data.get(
                "condensed_simplified_tree", False
            )

            self.cluster_tree_ = Cluster_Tree._from_dict_tree(
                tree, complete_tree=is_complete_tree_
            )

            if self.condensed_simplified_tree and not is_complete_tree_:
                warnings.warn(
                    "The provided tree is marked as 'condensed_simplified_tree' but not 'complete_tree'. Assuming noise is pruned or pre-defined in JSON."
                )
                self.set_leaf_noise = False
            elif getattr(self, "set_leaf_noise", False) and is_complete_tree_:
                self.set_leaf_noise = True
            elif self.condensed_simplified_tree and is_complete_tree_:
                self.set_leaf_noise = True
            else:
                self.set_leaf_noise = False

            if self.keep_noise_quality is None:
                self.keep_noise_quality = True

            self.source = "JSON"
            self.quality_measure = "precomputed"

        # SCIPY
        else:
            try:
                is_linkage = is_valid_linkage(X, throw=True)
            except Exception as exc:
                raise TypeError(
                    "Unsupported input type for X. Expected HDBSCAN object, sklearn "
                    "AgglomerativeClustering/HDBSCAN object, SciPy linkage matrix, JSON dict, "
                    "JSON string, or JSON file path."
                ) from exc
            if is_linkage:
                tree = X
                self.source = "SCIPY"
                self.density = getattr(self, "density", False)

        if self.source is None:
            raise RuntimeError("Failed to infer hierarchy source from input X.")

        if self.source != "JSON":
            # If min_samples set or HDBSCAN, condense then load
            if (
                self.min_cluster_size is not None and self.min_cluster_size > 1
            ):  # or hdbscan
                tree = _condense_tree(
                    tree, min_cluster_size=self.min_cluster_size, density=self.density
                )
                self.set_leaf_noise = True
                self.condensed_simplified_tree = True
            else:
                tree = _scipy_to_condensed(tree)
                self.condensed_simplified_tree = False
                if self.singletons_as_noise:
                    self.set_leaf_noise = True
                else:
                    self.set_leaf_noise = False

            tree = np.asarray(tree).T
            self.cluster_tree_ = Cluster_Tree._from_hdbscan_condensed(
                tree, density=self.density
            )

        self.cluster_tree_._compute_tree_depths()
        self.cluster_tree_._build_tree_ancestor_table()
        self.cluster_tree_.compute_leaf_order_and_spans()

        if self.source != "JSON":
            if self.quality_measure.casefold() in {"stability", "eom"}:
                self.keep_noise_quality = False
                self.cluster_tree_.compute_stability(density=self.density)

            elif self.quality_measure.casefold() in {"modularity","modularity q"}:
                self.nearest_neighbors = (
                    self.nearest_neighbors
                    or getattr(self, "min_samples", None)
                    or getattr(self, "min_cluster_size", None)
                )
                if self.data_ is None:
                    raise ValueError(
                        "Data must be provided for Modularity Q measure, either during FOSC fit() or by providing a clustering object with _raw_data attribute."
                    )
                elif not self.nearest_neighbors:
                    raise ValueError(
                        "For Modularity Q measure, nearest_neighbors (number of nearest neighbors) must be specified, either during FOSC class initilization or by providing a clustering object with min_samples or min_cluster_size attributes."
                    )

                if self.keep_noise_quality is None:
                    self.keep_noise_quality = True
                self.cluster_tree_.compute_modularity(
                    self.data_,
                    min_samples=self.nearest_neighbors,
                    metric=self.metric,
                    HDBSCAN=self.hdbscan_,
                )

            elif self.quality_measure.casefold() in {"pfce"}:
                if self.source == "SKLEARN_HDBSCAN":
                    raise ValueError(
                        "PFCE measure is not compatible with the sklearn HDBSCAN implementation."
                    )
                elif self.source != "HDBSCAN":
                    raise ValueError(
                        "PFCE measure is only compatible with HDBSCAN clustering objects."
                    )
                elif not hasattr(self, "mst"):
                    raise ValueError(
                        "PFCE measure requires the minimum spanning tree from the HDBSCAN clustering object (gen_min_span_tree=True)."
                    )
                else:
                    if self.keep_noise_quality is None:
                        self.keep_noise_quality = True
                    min_cluster_size = (
                        getattr(self, "min_cluster_size", None)
                        or getattr(self, "min_samples", None)
                        or 5
                    )
                    self.cluster_tree_.compute_PFCE(
                        self.mst, min_cluster_size=min_cluster_size
                    )
            else:
                warnings.warn(
                    f"Measure {self.quality_measure} not recognized, defaulting to 'Stability'"
                )
                self.keep_noise_quality = False
                self.cluster_tree_.compute_stability(density=self.density)

        self.cluster_tree_.compute_leaf_noise_and_siblings(
            set_leaf_noise=self.set_leaf_noise,
            singleton_as_noise=self.singletons_as_noise,
        )
        self.cluster_tree_.compute_bounds()
        self.cluster_tree_.set_noise_quality(keep_noise_quality=self.keep_noise_quality)

        self._postorder = _postorder(
            self.cluster_tree_.parent,
            self.cluster_tree_.children_flat,
            self.cluster_tree_.children_off,
        )

        self.candidate_quality_, self.candidate_nodes_, self.candidate_n_clusters_ = (
            self._efosc(top_M=self.top_M, kmin=self.kmin, kmax=self.kmax)
        )

        self.candidates_ = pd.DataFrame(
            {
                "quality": self.candidate_quality_,
                "n_clusters": self.candidate_n_clusters_,
                "selected_nodes": self.candidate_nodes_,
            }
        )
        # Backward-compatible aliases
        self.candidate_Qlist_ = self.candidate_quality_
        self.candidate_Clist_ = self.candidate_nodes_
        self.candidate_NClist_ = self.candidate_n_clusters_
        return self

    def _efosc(self, top_M=1, kmin=1, kmax=None):
        """
        Compute candidate clusterings using the Extended-FOSC algorithm.
        
        Parameters
        ----------
        top_M : int
            Number of candidate clusterings to return.
        kmin : int
            Minimum number of clusters to consider. 
        kmax : int
            Maximum number of clusters to consider.
        Returns
        -------
        Qlist : list of float
            List of quality scores for the candidate clusterings.
        Clist : list of list of int
            List of selected node IDs for each candidate clustering.
        NClist : list of int
            List of number of clusters for each candidate clustering.
        """
        self._validate_runtime_params(top_M=top_M, kmin=kmin, kmax=kmax)
        if not kmax:
            kmax = len(self.cluster_tree_.leaf_order)
        if not kmin:
            kmin = 1
        root = int(np.where(self.cluster_tree_.parent == -1)[0][0])
        root_scores, root_ncs, sel_nodes, sel_counts, root_k = _efosc(
            self.cluster_tree_.children_flat,
            self.cluster_tree_.children_off,
            self._postorder,
            self.cluster_tree_.clusteval,
            self.cluster_tree_.is_noise,
            root,
            top_M,
            kmin,
            kmax,
            self.cluster_tree_.LB,
            self.cluster_tree_.UB,
            len(self.cluster_tree_.leaf_order),
        )
        quality_list = [float(root_scores[i]) for i in range(root_k)]
        cluster_list = [
            [self.cluster_tree_._to_user(n) for n in sel_nodes[i, : sel_counts[i]]]
            for i in range(root_k)
        ]
        n_clusters_list = [int(root_ncs[i]) for i in range(root_k)]
        return quality_list, cluster_list, n_clusters_list

    def predict(self, top_M=1, kmin=1, kmax=None):
        """
        Predict candidate clusterings using the fitted FOSC model.

        This method allows extraction of clusterings with different ``top_M``,
        ``kmin``, and ``kmax`` settings without recomputing the hierarchy or
        re-running preprocessing.

        Parameters
        ----------
        top_M : int, optional
            Number of candidate clusterings to return. Solutions are ranked by
            quality, and the top ``top_M`` are returned.

            Default is 1.

        kmin : int, optional
            Minimum number of clusters allowed in the returned solutions.

            Default is 1.

        kmax : int, optional
            Maximum number of clusters allowed in the returned solutions.

            Default is None (no upper bound).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing candidate clusterings with the following columns:

            - ``quality``: Quality score of the clustering
            - ``n_clusters``: Number of clusters
            - ``selected_nodes``: List of selected node IDs representing clusters
        """

        check_is_fitted(self, "cluster_tree_")
        self._validate_runtime_params(top_M=top_M, kmin=kmin, kmax=kmax)
        self.candidate_quality_, self.candidate_nodes_, self.candidate_n_clusters_ = (
            self._efosc(top_M=top_M, kmin=kmin, kmax=kmax)
        )
        self.candidates_ = pd.DataFrame(
            {
                "quality": self.candidate_quality_,
                "n_clusters": self.candidate_n_clusters_,
                "selected_nodes": self.candidate_nodes_,
            }
        )
        self.candidate_Qlist_ = self.candidate_quality_
        self.candidate_Clist_ = self.candidate_nodes_
        self.candidate_NClist_ = self.candidate_n_clusters_
        return self.candidates_

    def get_labels(self, candidate_index: int = None, nodes: list = None):
        """
        Convert a candidate clustering or list of nodes to partition labels.

        Parameters
        ----------
        candidate_index : int, optional
            Index of the candidate clustering to convert. If None, the first
            candidate (index 0) is used.

        Nodes : list of int, optional
            List of node IDs representing clusters. If provided, this overrides
            ``candidate_index`` and labels are generated directly from the given nodes.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples,)`` containing cluster labels for each data point.

            Cluster labels are positive integers, and noise points are labeled as ``0``.
        """
        check_is_fitted(self, "cluster_tree_")

        if candidate_index is None and nodes is None:
            candidate_index = 0

        if nodes is not None:
            if candidate_index is not None:
                warnings.warn(
                    "Both `candidate_index` and `nodes` were provided; `nodes` takes precedence.",
                    stacklevel=2,
                )
            if not isinstance(nodes, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"'nodes' must be a sequence of node identifiers, got {type(nodes)}."
                )
            clusters = nodes
        else:
            if not isinstance(candidate_index, (int, np.integer)):
                raise TypeError(
                    f"'candidate_index' must be an integer, got {type(candidate_index)}."
                )
            if candidate_index < 0 or candidate_index >= len(self.candidate_nodes_):
                raise IndexError(
                    f"'candidate_index' out of range: {candidate_index}. "
                    f"Valid range is 0 to {len(self.candidate_nodes_) - 1}."
                )
            clusters = self.candidate_nodes_[candidate_index]

        labels_ = np.zeros(len(self.cluster_tree_.leaf_order), dtype=int)

        for i in range(0, len(clusters)):

            labels_[self.cluster_tree_.get_node_labels(clusters[i])] = i + 1
        return labels_

    def labels_to_partition(self, candidate_index: int = None, nodes: list = None):
        """Backward-compatible alias for :meth:`get_labels`."""
        return self.get_labels(candidate_index=candidate_index, nodes=nodes)

    def plot_tree(self, **kwargs):
        """
        Plot the hierarchy tree and optionally interactively scroll
        through candidate solutions.
        """

        # ensure fitted
        if not hasattr(self, "cluster_tree_"):
            raise ValueError("FOSC instance must be fitted before calling plot_tree().")

        fig, ax, node_positions, node_bounds = _plot_tree(
            tree=self.cluster_tree_,
            hdbscan_style=self.density,
            **kwargs,
        )

        # attach interactivity if candidates exist
        if getattr(self, "candidate_nodes_", None):
            if self.source == "JSON":
                selections = [
                    [self.cluster_tree_._to_compact(n) for n in sel]
                    for sel in self.candidate_nodes_
                ]
            else:
                selections = self.candidate_nodes_

            _interactive_highlight(
                fig=fig,
                ax=ax,
                node_bounds=node_bounds,
                selections=selections,
                qlist=getattr(self, "candidate_quality_", None),
            )
        else:
            import matplotlib.pyplot as plt

            plt.show()

        return fig, ax

    def plot(
        self,
        X=None,
        projection="pca",
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        random_state=None,
        figsize=(8, 6),
        point_size=10,
        alpha=0.9,
        cmap="tab10",
        show=True,
        return_handles=False,
    ):
        """
        Plot candidate FOSC solutions in 2D with an interactive slider.

        This method visualises candidate clusterings and allows interactive
        selection between them. If the data is not already 2-dimensional, a
        projection is applied before plotting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Original data used for clustering. If not provided, the data passed
            during :meth:`fit` is used.

        projection : {"umap", "pca", "none"}, optional
            Dimensionality reduction method used when the data has more than two
            dimensions.

            - ``"pca"``: linear projection (default)  
            - ``"umap"``: non-linear projection  
            - ``"none"``: no projection (requires 2D data)  

            Default is ``"pca"``.

        umap_n_neighbors : int, optional
            Number of neighbors used for UMAP projection. Ignored unless
            ``projection="umap"``.

            Default is 15.

        umap_min_dist : float, optional
            Minimum distance parameter for UMAP projection. Controls how tightly
            points are packed in the embedding.

            Default is 0.1.

        random_state : int or None, optional
            Random state used for reproducibility of projections.

            Default is None.

        figsize : tuple of int, optional
            Size of the matplotlib figure.

            Default is ``(8, 6)``.

        point_size : int, optional
            Size of the plotted data points.

            Default is 10.

        alpha : float, optional
            Transparency of the data points.

            Default is 0.9.

        cmap : str or matplotlib.colors.Colormap, optional
            Colormap used to assign colors to clusters.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or no candidate clusterings are available.
        """

        if not hasattr(self, "candidate_nodes_"):
            raise ValueError("FOSC instance must be fitted before calling plot().")

        if X is None:
            X = getattr(self, "data_", None)
            if X is None:
                raise ValueError("No data available. Provide X or fit() with data.")

        fig, ax, slider = _plot_fosc(
            fosc=self,
            X=X,
            projection=projection,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            random_state=random_state,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
            show=show,
        )

        if return_handles:
            return fig, ax, slider

        return None
