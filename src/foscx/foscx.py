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


from .efosc import _efosc
from .hierarchy.condense_tree import _condense_tree, _scipy_to_condensed
from .hierarchy.hierarchy import Cluster_Tree
from .hierarchy.tree_numba import _postorder
from .hierarchy.constraint_score import generate_pairwise_constraints_
from .plotting.plot_functions import _plot_fosc
from .plotting.plot_tree_functions import interactive_condensed


with resources.open_text("foscx", "hierarchy.schema.json") as f:
    FOSC_JSON_SCHEMA = json.load(f)

FOSC_SCHEMA_VALIDATOR = Draft7Validator(FOSC_JSON_SCHEMA)


class FOSCX(BaseEstimator):
    def __init__(
        self,
        top_M: int = 5,
        kmin: int = 2,
        kmax: int = None,
        *,
        min_cluster_size: int = None,
        quality_measure: str = "stability",
        singletons_as_noise: bool = False,
        keep_noise_quality: bool = None,
        nearest_neighbors: int = None,
        metric: str = None,
        density: bool = False,
        tie_quality: str = "stability",
        verbose: bool = False,
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

            Available options are ``"stability"``, ``"modularity"``,
            ``"PFCE"``, ``"B3"``, and ``"constraints"``.

            Default is ``"stability"``.

            Additional notes:

            ``modularity`` requires ``nearest_neighbors`` and ``metric``,
            or a hierarchy object that provides them (e.g. HDBSCAN).

            ``PFCE`` requires an HDBSCAN object from the ``hdbscan`` package.

            ``B3`` requires semi-supervised labels where unlabeled
            observations have value ``-1``.

            ``constraints`` requires either labels or explicit
            must-link/cannot-link constraints.

            Ignored for JSON trees with precomputed quality.

        nearest_neighbors : int, optional
            Number of neighbors used to construct the k-nearest neighbor graph
            for modularity.

            If not provided, this is inferred from the clustering object when
            available (e.g. ``min_samples`` or ``min_cluster_size`` in HDBSCAN).
            For precomputed_similarity metric, this is not required.

            Larger values produce denser graphs and may smooth local structure,
            but increase computational cost.

        metric : str, optional
            Distance metric used when constructing the similarity graph for
            modularity. Available metrics are those compatible with
            ``sklearn.neighbors.NearestNeighbors``.

            Special options include:

            - ``"precomputed"``: for a precomputed distance matrix
            - ``"precomputed_similarity"``: for a precomputed similarity graph

            If not provided, this is inferred from the input hierarchy when
            available (e.g. HDBSCAN or sklearn AgglomerativeClustering). 
            Otherwise, it defaults to ``"euclidean"``.

            If provided, the selected choice overrides the hierarchy information.

            It is generally recommended to match the metric used to construct
            the hierarchy.

        density : bool, optional
            Whether the hierarchy should be treated as density-based.

            This determines how the ``distance`` values in the tree are interpreted:

            - If False: values represent merge distances (standard hierarchical clustering)
            - If True: values represent density levels (e.g. HDBSCAN λ values)

            This is automatically set when using density-based clustering objects.
            Manually setting this is primarily intended for SciPy linkage inputs.

        tie_quality : str, optional
            Quality measure used to break ties when semi-supervised measures are used.

            Available options are ``"stability"``, ``"modularity"``, and ``"PFCE"``.

            Default is ``"stability"``.

            Additional notes:
            
            ``modularity`` requires ``nearest_neighbors`` and ``metric``, or a
            hierarchy object that provides them (e.g. HDBSCAN).

            ``PFCE`` requires an HDBSCAN object from the ``hdbscan`` package.

            Ignored for JSON trees with precomputed quality.

        verbose : bool, optional
            If True, print progress messages during fitting (e.g. building the tree,
            computing quality, running FOSC optimisation).

            Default is False.


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


        Examples
        --------
        >>> from foscx import FOSCX
        >>> model = FOSCX(top_M=3, kmin=2)
        >>> model.fit(X)
        >>> candidates = model.predict()
        >>> labels = model.get_labels(0)
        """

        self.top_M = top_M
        self.kmin = kmin
        self.kmax = kmax

        self.min_cluster_size = min_cluster_size
        self.nearest_neighbors = nearest_neighbors
        self.quality_measure = quality_measure
        self.singletons_as_noise = singletons_as_noise
        self.keep_noise_quality = keep_noise_quality
        self.metric = metric
        self.density = density
        self.tie_quality = tie_quality
        self.verbose = verbose

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

    def _log(self, msg: str) -> None:
        """Print *msg* when ``verbose=True``."""
        if self.verbose:
            print(f"[FOSCX] {msg}")

    def fit(self, X, y=None, z=None, **params):
        """
        Fit FOSC-X to a hierarchical clustering tree.

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

        z : ndarray or tuple(ndarray, ndarray), optional
            Semi-supervision information for quality measures.

            Supported formats:

            - Partial labels:
                1D array of shape ``(n_samples,)`` where ``-1`` denotes
                unlabeled observations.

            - Pairwise constraints:
                Tuple ``(must_link, cannot_link)`` where each element is
                an integer array of shape ``(n_constraints, 2)``.

        **params : dict, optional
            Additional parameters passed to :meth:`set_params`.

        Returns
        -------
        self : FOSC-X
            Fitted estimator.
        """

        
        if X is None:
            raise ValueError("Cluster Tree must be supplied")

        self.data_ = y

        if z is None:
            self.constraints_ = None
            self.GT_labels_ = None
        elif isinstance(z, tuple):
            self.constraints_ = z
            self.GT_labels_ = None
        else:
            self.GT_labels_ = np.asarray(z)
            self.constraints_ = None


        if params:
            try:
                self.set_params(**params)
            except ValueError as exc:
                raise ValueError(f"Invalid fit() parameters: {exc}") from exc

        self._validate_constructor_params()

        self._raw_tree_ = X

        # HDBSCAN (hdbscan/fast_hdbscan)
        if hasattr(X, "condensed_tree_") and hasattr(X, "min_cluster_size") and hasattr(X,"single_linkage_tree_"):
            tree = X.single_linkage_tree_.to_numpy()

            if not self.min_cluster_size:
                self.min_cluster_size = X.min_cluster_size

            self.min_samples = getattr(X, "min_samples", X.min_cluster_size)

            if not self.density:
                warnings.warn(
                    "HDBSCAN input detected: overriding density=False to density=True.",
                    stacklevel=2,
                )
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
            if self.metric is None:
                self.metric = getattr(X, "metric", None)
            self.source = "HDBSCAN"

        # SKLEARN HDBSCAN
        elif getattr(type(X), "__module__", "").startswith("sklearn.cluster._hdbscan"):
            tree = X._single_linkage_tree_
            tree = structured_to_unstructured(tree, dtype=np.float64)

            if not self.min_cluster_size:
                self.min_cluster_size = X.min_cluster_size

            self.min_samples = getattr(X, "min_samples", X.min_cluster_size)

            if not self.density:
                warnings.warn(
                    "sklearn HDBSCAN input detected: overriding density=False to density=True.",
                    stacklevel=2,
                )
            self.density = True
            self.hdbscan_ = True
            if self.metric is None:
                self.metric = getattr(X, "metric", None)
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
            if self.metric is None:
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
            if self.quality_measure != "precomputed":
                warnings.warn(
                    f"JSON trees use precomputed quality; ignoring quality_measure={self.quality_measure!r}.",
                    stacklevel=2,
                )
            self.quality_measure = "precomputed"

        # SCIPY or Condensed
        else:
            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = X

            try:
                X_arr = structured_to_unstructured(X_arr, dtype=np.float64)
            except Exception:
                X_arr = np.asarray(X_arr)

            # Try SciPy linkage first
            try:
                is_linkage = is_valid_linkage(X_arr, throw=False)
            except Exception:
                is_linkage = False

            if is_linkage:
                tree = X_arr
                self.source = "SCIPY"
                self.density = getattr(self, "density", False)
                if self.density and (self.min_cluster_size is None or self.min_cluster_size < 2):
                    warnings.warn(
                        "SciPy linkage trees treated as density-based require min_cluster_size >= 2. "
                        f"Overriding min_cluster_size from {self.min_cluster_size!r} to 2.",
                        stacklevel=2,
                    )
                    self.min_cluster_size = 2

            elif X_arr.ndim == 2 and X_arr.shape[1] == 4:
                # Treat as condensed tree
                tree = X_arr
                self.source = "CONDENSED"

                if self.min_cluster_size is not None and self.min_cluster_size > 1:
                    warnings.warn(
                        "Input appears to already be a condensed tree; "
                        "min_cluster_size > 1 will be ignored.",
                        stacklevel=2,
                    )
                    self.min_cluster_size = None

                self.density = getattr(self, "density", False)
                self.condensed_simplified_tree = True
                self.set_leaf_noise = True

            else:
                raise TypeError(
                    "Unsupported input type for X. Expected HDBSCAN object, sklearn "
                    "AgglomerativeClustering/HDBSCAN object, SciPy linkage matrix, "
                    "condensed tree (n,4), JSON dict/string/path."
                )

        if self.source is None:
            raise RuntimeError("Failed to infer hierarchy source from input X.")

        self._log(f"Input source detected: {self.source}")

        if self.source not in ("JSON", "CONDENSED"):
            # If min_samples set or HDBSCAN, condense then load
            if (
                self.min_cluster_size is not None and self.min_cluster_size > 1
            ):  # or hdbscan
                self._log(
                    f"Condensing tree (min_cluster_size={self.min_cluster_size}) ..."
                )
                tree = _condense_tree(
                    tree, min_cluster_size=self.min_cluster_size, density=self.density
                )
                self.set_leaf_noise = True
                self.condensed_simplified_tree = True
            else:
                self._log("Converting linkage to condensed format ...")
                tree = _scipy_to_condensed(tree)
                self.condensed_simplified_tree = False
                if self.singletons_as_noise:
                    self.set_leaf_noise = True
                else:
                    self.set_leaf_noise = False

            tree = np.asarray(tree).T
            self._log("Building cluster tree ...")
            self.cluster_tree_ = Cluster_Tree._from_hdbscan_condensed(
                tree, density=self.density
            )

        if self.source == "CONDENSED":
            #tree = np.asarray(tree).T
            self._log("Building cluster tree from condensed input ...")
            self.cluster_tree_ = Cluster_Tree._from_hdbscan_condensed(
                tree, density=self.density
            )


        self.cluster_tree_._compute_tree_depths()
        self.cluster_tree_._build_tree_ancestor_table()
        self.cluster_tree_.compute_leaf_order_and_spans()

        if self.source != "JSON":

            if self.quality_measure.casefold() in {"b3","constraints"}:
                self._log(f"Computing tie-breaking quality ({self.tie_quality}) ...")
                self.compute_quality(quality_measure = self.tie_quality)
                self._log(f"Computing primary quality ({self.quality_measure}) ...")
                self.compute_quality(quality_measure = self.quality_measure)
            else:
                self._log(f"Computing quality ({self.quality_measure}) ...")
                self.compute_quality(quality_measure = self.quality_measure)

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

        self._log(
            f"Running FOSC (top_M={self.top_M}, kmin={self.kmin}, kmax={self.kmax}) ..."
        )
        self.candidate_quality_, self.candidate_nodes_, self.candidate_n_clusters_ = (
            self._efosc(top_M=self.top_M, kmin=self.kmin, kmax=self.kmax)
        )
        self._log(
            f"Done. Found {len(self.candidate_nodes_)} candidate clustering(s)."
        )

        self.candidates_ = pd.DataFrame(
            {
                "quality": self.candidate_quality_,
                "n_clusters": self.candidate_n_clusters_,
                "selected_nodes": self.candidate_nodes_,
            }
        )
        # Backward-compatible aliases
        #self.candidate_Qlist_ = self.candidate_quality_
        #self.candidate_Clist_ = self.candidate_nodes_
        #self.candidate_NClist_ = self.candidate_n_clusters_
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
        #self.candidate_Qlist_ = self.candidate_quality_
        #self.candidate_Clist_ = self.candidate_nodes_
        #self.candidate_NClist_ = self.candidate_n_clusters_
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

            Cluster labels are positive integers, and noise points are labeled as ``-1``.
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

        labels_ = np.full(len(self.cluster_tree_.leaf_order), -1, dtype=int)

        for i in range(len(clusters)):
            labels_[self.cluster_tree_.get_node_indices(clusters[i])] = i

        return labels_

    def labels_to_partition(self, candidate_index: int = None, nodes: list = None):
        """Backward-compatible alias for :meth:`get_labels`."""
        return self.get_labels(candidate_index=candidate_index, nodes=nodes)

    def plot_tree(
        self,
        figsize=(5, 4),
        label_clusters=False,
        selection_palette=None,
        cmap="viridis",
        colorbar=True,
        vary_line_width=True,
        leaf_separation=1,
        log_size=False,
        max_rectangle_per_icicle=20,
        value_field=None,
        density=None,
        **kwargs,
    ):
        """
        Plot the condensed hierarchy tree with an interactive slider for
        browsing candidate solutions.

        The tree type (icicle or dendrogram) is selected automatically based
        on the clustering source and ``min_cluster_size``.  Candidate
        solutions and quality scores are extracted from the fitted instance
        automatically.

        Parameters
        ----------
        figsize : tuple of int, optional
            Size of the matplotlib figure.

            Default is ``(10, 8)``.

        label_clusters : bool, optional
            Whether to annotate highlighted clusters with their index number.

            Default is ``False``.

        selection_palette : list of colours, optional
            Colours used for the ellipses drawn around selected clusters.
            Cycles if fewer colours than clusters are provided.  Uses red by
            default when ``None``.

        cmap : str or matplotlib.colors.Colormap, optional
            Colormap used to colour the tree branches or bars by cluster
            size.  Pass ``'none'`` for solid black.

            Default is ``'viridis'``.

        colorbar : bool, optional
            Whether to display a colorbar showing the cluster-size scale.

            Default is ``True``.

        vary_line_width : bool, optional
            Scale branch thickness by cluster size in the dendrogram.
            Only applies when a binary tree is used (i.e. when the source
            is ``"SKLEARN"`` or ``"SCIPY"`` with ``min_cluster_size < 2``).
            Has no effect on the icicle plot.

            Default is ``True``.

        leaf_separation : float, optional
            Horizontal spacing between leaf clusters in the icicle plot.
            Only applies to non-binary trees.

            Default is 1.

        log_size : bool, optional
            Whether to use a log scale for cluster size in the icicle plot.
            Only applies to non-binary trees.

            Default is ``False``.

        max_rectangle_per_icicle : int, optional
            Maximum number of bars emitted per cluster branch in the icicle
            plot.  Only applies to non-binary trees.

            Default is 20.

        value_field : str or None, optional
            Name of the column holding split values in the condensed tree
            (``'distance'`` or ``'lambda_val'``).  Auto-detected when
            ``None``.

            Default is ``None``.

        density : bool or None, optional
            Whether the tree was built from a density-based clusterer.
            Controls the root y-coordinate and axis direction of the icicle
            plot.  Inferred from the fitted instance when ``None``.

            Default is ``None``.

        **kwargs
            Additional keyword arguments forwarded to the underlying plot
            functions.

        Returns
        -------
        composite : ipywidgets.VBox or None
            The interactive widget (figure + slider).  ``None`` when no
            candidate solutions are available.

        Raises
        ------
        ValueError
            If the instance has not been fitted before calling this method.
        """
        if not hasattr(self, "cluster_tree_"):
            raise ValueError("FOSC instance must be fitted before calling plot_tree().")

        if self.source in {"SKLEARN", "SCIPY"} and (
            self.min_cluster_size is None or self.min_cluster_size < 2
        ):
            binary_tree = True
        else:
            binary_tree = False

        interactive_tree =  interactive_condensed(
            self,
            binary_tree=binary_tree,
            figsize=figsize,
            label_clusters=label_clusters,
            selection_palette=selection_palette,
            density=density if density is not None else self.density,
            cmap=cmap,
            colorbar=colorbar,
            vary_line_width=vary_line_width,
            leaf_separation=leaf_separation,
            log_size=log_size,
            max_rectangle_per_icicle=max_rectangle_per_icicle,
            value_field=value_field,
            **kwargs,
        )
        return None


    def plot(
        self,
        X=None,
        projection="pca",
        umap_n_neighbors=12,
        umap_min_dist=0.1,
        umap_n_epochs=150,
        tsne_perplexity=30.0,
        tsne_n_iter=500,
        random_state=None,
        figsize=(5, 4),
        point_size=2,
        alpha=0.9,
        cmap="tab10",
        show=True,
        return_handles=False,
    ):
        """
        Plot candidate FOSC solutions in 2D with an interactive slider.

        The projection is computed once and reused across all candidate
        solutions; only point colours are updated on each slider move.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Original data used for clustering.  If not provided, the data
            passed during :meth:`fit` is used.

        projection : {"pca", "umap", "tsne", "none"}, optional
            Dimensionality reduction method applied when the data has more
            than two dimensions.

            - ``"pca"``  : linear projection (default, fastest)
            - ``"umap"`` : non-linear, preserves local structure
            - ``"tsne"`` : non-linear, good cluster separation
            - ``"none"`` : no projection (data must already be 2D)

            Default is ``"pca"``.

        umap_n_neighbors : int, optional
            Number of neighbours used by UMAP.  Smaller values are faster
            and emphasise local structure; larger values give a more global
            view.  Ignored unless ``projection="umap"``.

            Default is 12.

        umap_min_dist : float, optional
            Minimum distance between points in the UMAP embedding.  Lower
            values produce tighter clusters.  Ignored unless
            ``projection="umap"``.

            Default is 0.1.

        umap_n_epochs : int, optional
            Number of optimisation iterations for UMAP.  Lower values are
            faster with minimal visual difference for visualisation purposes.
            Ignored unless ``projection="umap"``.

            Default is 150.

        tsne_perplexity : float, optional
            Perplexity parameter for t-SNE.  Roughly controls the balance
            between local and global structure.  Typical range 5–50.
            Ignored unless ``projection="tsne"``.

            Default is 30.0.

        tsne_n_iter : int, optional
            Number of optimisation iterations for t-SNE (sklearn fallback
            only).  Ignored unless ``projection="tsne"`` and openTSNE is
            not installed.

            Default is 500.

        random_state : int or None, optional
            Random seed for reproducibility of the projection.

            Default is ``None``.

        figsize : tuple of int, optional
            Size of the matplotlib figure.

            Default is ``(7, 6.3)``.

        point_size : float, optional
            Marker size of the scatter plot points (points²).

            Default is 10.

        alpha : float, optional
            Transparency of the scatter plot points.  Must be in [0, 1].

            Default is 0.9.

        cmap : str, optional
            Accepted for API compatibility; cluster colours are assigned
            automatically via golden-ratio HSV spacing and this argument is
            not used.

        show : bool, optional
            Accepted for API compatibility; the widget is always displayed
            automatically and this argument is not used.

        return_handles : bool, optional
            Accepted for API compatibility; the method always returns
            ``None`` and this argument is not used.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the instance has not been fitted, or no data is available.
        """
        if not hasattr(self, "candidate_nodes_"):
            raise ValueError("FOSC instance must be fitted before calling plot().")

        if X is None:
            X = getattr(self, "data_", None)
            if X is None:
                raise ValueError("No data available. Provide X or fit() with data.")

        interactive_plot = _plot_fosc(
            fosc=self,
            X=X,
            projection=projection,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_n_epochs=umap_n_epochs,
            tsne_perplexity=tsne_perplexity,
            tsne_n_iter=tsne_n_iter,
            random_state=random_state,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
            show=show,
        )

        return None

    def compute_quality(self, quality_measure):
        """
        Internal method to determine and compute the requested quality measure.
        """

        if quality_measure.casefold() in {"stability", "EOM"}:
            self.keep_noise_quality = False
            self.cluster_tree_.compute_stability(density=self.density)

        elif quality_measure.casefold() in {"modularity", "modularity q"}:
            self.nearest_neighbors = (
                self.nearest_neighbors
                or getattr(self, "min_samples", None)
                or getattr(self, "min_cluster_size", None)
            )
            if self.data_ is None:
                raise ValueError(
                    "Data or a precomputed similarity graph must be provided for the "
                    "Modularity Q measure, either via fit(y=...) or by passing a "
                    "clustering object that exposes a _raw_data attribute."
                )
            elif (
                not self.nearest_neighbors
                and getattr(self, "metric", None) != "precomputed_similarity"
            ):
                raise ValueError(
                    "For the Modularity Q measure, nearest_neighbors must be specified "
                    "either during FOSCX initialisation or via a clustering object with "
                    "min_samples or min_cluster_size attributes."
                )
            if self.metric is None:
                self.metric = "euclidean"
            if self.keep_noise_quality is None:
                self.keep_noise_quality = True
            self.cluster_tree_.compute_modularity(
                self.data_,
                min_samples=self.nearest_neighbors,
                metric=self.metric,
                HDBSCAN=self.hdbscan_,
            )

        elif quality_measure.casefold() in {"pfce"}:
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
                    "PFCE measure requires the minimum spanning tree from the HDBSCAN "
                    "clustering object (gen_min_span_tree=True)."
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

        elif quality_measure.casefold() in {"b3"}:
            if self.GT_labels_ is None:
                raise ValueError(
                    "Semi-supervised measures such as B3 require partial labels; "
                    "unlabeled observations should have a value of -1."
                )
            else:
                self.cluster_tree_.compute_B3(GT_labels=self.GT_labels_)
                if self.keep_noise_quality is None:
                    # Noise has no ground-truth membership, so assigning it a B3
                    # score is meaningless.
                    self.keep_noise_quality = False

        elif quality_measure.casefold() in {"constraints"}:
            if self.GT_labels_ is None and self.constraints_ is None:
                raise ValueError(
                    "Semi-supervised measures such as constraints require either partial "
                    "labels (unlabeled observations = -1) or explicit must-link / "
                    "cannot-link constraint arrays."
                )
            else:
                if self.constraints_ is None:
                    self.constraints_ = generate_pairwise_constraints_(self.GT_labels_)
                self.cluster_tree_.compute_constraint_score(constraints=self.constraints_)
                if self.keep_noise_quality is None:
                    # Noise trivially satisfies cannot-link constraints, so retaining
                    # its quality is generally reasonable.
                    self.keep_noise_quality = True

        else:
            warnings.warn(
                f"Quality measure {quality_measure!r} not recognised; "
                "defaulting to 'stability'.",
                stacklevel=2,
            )
            self.keep_noise_quality = False
            self.cluster_tree_.compute_stability(density=self.density)

        return None