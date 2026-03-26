from sklearn.cluster import AgglomerativeClustering
from foscx import FOSCX


def test_agglomerative_basic(small_dataset):
    model = AgglomerativeClustering(
        distance_threshold=0,
        n_clusters=None
    )
    model.fit(small_dataset)

    fosc = FOSCX(top_M=1)
    fosc.fit(model)

    assert fosc.candidates_ is not None

def test_agglomerative_modularity(small_dataset):
    model = AgglomerativeClustering(
        distance_threshold=0,
        n_clusters=None
    )
    model.fit(small_dataset)

    fosc = FOSCX(top_M=1,quality_measure="modularity",nearest_neighbors=5)
    fosc.fit(model,small_dataset)

    assert fosc.candidates_ is not None