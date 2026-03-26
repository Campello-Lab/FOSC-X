import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage

from foscx import FOSCX


def test_fit_rejects_none_input():
    with pytest.raises(ValueError, match="Cluster Tree must be supplied"):
        FOSCX().fit(None)


def test_constructor_validates_k_bounds():
    Z = linkage(np.array([[0.0], [1.0], [2.0]]), method="single")
    with pytest.raises(ValueError, match="kmin.*cannot be greater"):
        FOSCX(kmin=3, kmax=2).fit(Z)


def test_predict_validates_top_M_type(tiny_dataset):
    Z = linkage(tiny_dataset, method="single")
    model = FOSCX().fit(Z)
    with pytest.raises(TypeError, match="'top_M' must be an integer"):
        model.predict(top_M="2")


def test_get_labels_checks_candidate_bounds(tiny_dataset):
    Z = linkage(tiny_dataset, method="single")
    model = FOSCX(top_M=1).fit(Z)
    with pytest.raises(IndexError, match="out of range"):
        model.get_labels(candidate_index=2)


def test_get_labels_warns_when_nodes_and_candidate_index_provided(tiny_dataset):
    Z = linkage(tiny_dataset, method="single")
    model = FOSCX(top_M=1).fit(Z)
    with pytest.warns(UserWarning, match="nodes.*takes precedence"):
        labels = model.get_labels(candidate_index=0, nodes=model.candidate_nodes_[0])
    assert labels.shape[0] == tiny_dataset.shape[0]


def test_min_cluster_size_one_warns(tiny_dataset):
    Z = linkage(tiny_dataset, method="single")
    with pytest.warns(UserWarning, match="min_cluster_size=1"):
        FOSCX(min_cluster_size=1).fit(Z)
