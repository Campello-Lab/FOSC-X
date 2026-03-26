import hdbscan
from foscx import FOSCX


def test_hdbscan_basic(small_dataset):
    model = hdbscan.HDBSCAN()
    model.fit(small_dataset)

    fosc = FOSCX(top_M=1)
    fosc.fit(model)

    assert fosc.candidates_ is not None

def test_hdbscan_modularity(small_dataset):
    model = hdbscan.HDBSCAN(gen_min_span_tree=True)
    model.fit(small_dataset)

    fosc = FOSCX(top_M=1,quality_measure="modularity")
    fosc.fit(model)

    assert fosc.candidates_ is not None

def test_hdbscan_PFCE(small_dataset):
    model = hdbscan.HDBSCAN(gen_min_span_tree=True)
    model.fit(small_dataset)

    fosc = FOSCX(top_M=1,quality_measure="PFCE")
    fosc.fit(model)

    assert fosc.candidates_ is not None

