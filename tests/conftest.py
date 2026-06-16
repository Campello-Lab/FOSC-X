import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import foscx
import json
import string
import pytest

@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    return rng.normal(size=(200, 2))


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 2))

@pytest.fixture
def json_distance():
    X, labels = make_blobs()
    Z = AgglomerativeClustering(n_clusters=None,distance_threshold=0)
    Z.fit(X)
    f1 = foscx.FOSCX()
    f1.fit(Z)
    j1 = save_fosc_tree(f1,"testJSON_distance.json")
    return j1

@pytest.fixture
def json_density():
    X, labels = make_blobs()
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    f2 = foscx.FOSCX()
    f2.fit(clusterer)
    j2 = save_fosc_tree(f2,"testJSON_distance.json")
    return j2



def int_to_alpha(n):
    """
    Convert:
        0  -> a
        1  -> b
        ...
        25 -> z
        26 -> aa
        27 -> ab
        ...
    """
    letters = string.ascii_lowercase

    result = ""
    n += 1  # convert to 1-based indexing

    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = letters[rem] + result

    return result


def save_fosc_tree(fosc, filename):
    """
    Save a FOSC cluster tree with alphabetic node IDs.
    """

    tree = {}
    root_id = None

    node_id = 0

    while True:
        try:
            summary = fosc.cluster_tree_.node_summary(node_id)
        except Exception:
            break

        parent = summary["parent"]

        if parent is None or parent == "":
            root_id = str(node_id)
            parent_str = ""
        else:
            parent_str = str(parent)

        tree[str(node_id)] = {
            "clusteval": summary["quality"],
            "children": [str(c) for c in summary["children"]],
            "parent": parent_str,
            "cluster_size": summary["cluster_size"],
            "noise": summary["noise"],
            "distance": summary["distance"],
        }

        node_id += 1

    # --------------------------------------------------
    # Build old-id -> alpha-id mapping
    # --------------------------------------------------

    id_map = {
        old_id: int_to_alpha(int(old_id))
        for old_id in tree.keys()
    }

    # --------------------------------------------------
    # Rename nodes and references
    # --------------------------------------------------

    renamed_tree = {}

    for old_id, node in tree.items():

        new_id = id_map[old_id]

        parent = node["parent"]
        if parent != "":
            parent = id_map[parent]

        children = [
            id_map[child]
            for child in node["children"]
        ]

        renamed_tree[new_id] = {
            **node,
            "parent": parent,
            "children": children,
        }

    result = {
        "Root_ID": id_map[root_id],
        "condensed_simplified_tree": bool(
            fosc.condensed_simplified_tree
        ),
        "complete_tree": True,
    }

    if hasattr(fosc, "density"):
        result["density_based"] = bool(fosc.density)

    if hasattr(fosc, "data"):
        result["data"] = fosc.data

    result["tree"] = renamed_tree

    #with open(filename, "w", encoding="utf-8") as f:
    #    json.dump(result, f, indent=2)

    return result