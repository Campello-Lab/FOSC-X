import numpy as np
from scipy.cluster.hierarchy import linkage
from foscx import FOSCX
import hdbscan


def test_scipy_linkage_basic(small_dataset):
    Z = linkage(small_dataset, method="single")

    fosc = FOSCX(top_M=2)
    fosc.fit(Z)

    assert fosc.candidates_ is not None
    assert len(fosc.candidates_) > 0

    labels = fosc.labels_to_partition(0)
    assert labels.shape[0] == small_dataset.shape[0]

def test_condesned_tree(small_dataset):
    model = hdbscan.HDBSCAN()
    model.fit(small_dataset)

    Z = model.condensed_tree_.to_numpy()

    fosc = FOSCX(top_M=2,density=True)
    fosc.fit(Z)

    assert fosc.candidates_ is not None
    assert len(fosc.candidates_) > 0