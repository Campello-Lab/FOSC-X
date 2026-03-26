from scipy.cluster.hierarchy import linkage
from foscx import FOSCX


def test_efosc_candidate_ordering(tiny_dataset):
    Z = linkage(tiny_dataset, method="single")

    fosc = FOSCX(top_M=3)
    fosc.fit(Z)

    qualities = fosc.candidates_["quality"].values

    # qualities should be sorted descending
    assert all(qualities[i] >= qualities[i + 1] for i in range(len(qualities) - 1))
