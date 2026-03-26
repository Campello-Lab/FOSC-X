# FOSCX

FOSC-X: An Extended Framework for Optimal Local Cuts and Non-Horizontal Flattening of Clustering Hierarchies

## Install from GitHub

```bash
pip install git+https://github.com/Campello-Lab/FOSC-X.git
```

## Instal Locally

```bash
pip install .
```

## Quick start

```python
from foscx import FOSCX
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

Z = AgglomerativeClustering(n_clusters=None,distance_threshold=0, linkage="ward")
Z.fit(X)

model = FOSCX(top_M=5, kmin=2, kmax=10, quality_measure="Stability")
model.fit(Z)
print(model.candidates_)
```

## Dependencies

FOSCX depends on:

- numpy
- scipy
- scikit-learn
- numba
- pandas
- matplotlib
- umap-learn
- jsonschema

## Development

Install development extras and run tests:

```bash
pip install -e .[dev,test]
python -m pytest
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
