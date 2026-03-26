import numpy as np
from scipy.cluster.hierarchy import linkage
from foscx import FOSCX


def test_scipy_linkage_basic(small_dataset):
    Z = linkage(small_dataset, method="single")

    fosc = FOSCX(top_M=2)
    fosc.fit(Z)

    assert fosc.candidates_ is not None
    assert len(fosc.candidates_) > 0

    labels = fosc.labels_to_partition(0)
    assert labels.shape[0] == small_dataset.shape[0]
