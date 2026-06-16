import numpy as np
from scipy.cluster.hierarchy import linkage
from foscx import FOSCX
import hdbscan


def test_json_linkage(json_distance):
    
    fosc = FOSCX(top_M=2)
    fosc.fit(json_distance)

    assert fosc.candidates_ is not None
    assert len(fosc.candidates_) > 0

    labels = fosc.labels_to_partition(0)

def test_json_densit(json_density):

    fosc = FOSCX(top_M=2)
    fosc.fit(json_density)

    assert fosc.candidates_ is not None
    assert len(fosc.candidates_) > 0

    labels = fosc.labels_to_partition(0)