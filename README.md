# FOSC-X

FOSC-X: An Extended Framework for Optimal Local Cuts and Non-Horizontal Flattening of Clustering Hierarchies.

FOSC-X is a Python library for extracting high-quality flat clusterings from
hierarchical clustering trees.

Rather than returning a single clustering, FOSC-X explores multiple candidate
solutions and returns the best ones according to a chosen quality measure.
It supports a range of common hierarchical clustering formats and provides
tools for pre-processing, evaluation, and visualisation.

---

## Features

- Extract multiple high-quality clusterings from a single hierarchy  
- Supports HDBSCAN, scikit-learn, SciPy, and custom JSON trees  
- Multiple quality measures (stability, modularity, PFCE)  
- Tree pre-processing (e.g. condensation, noise handling)  
- Visualisation tools for both clusterings and tree structure  

---

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/Campello-Lab/FOSC-X.git
```

### Install locally

```bash
pip install .
```

---

## Quick Start

```python
from foscx import FOSCX
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Build hierarchical clustering
Z = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0,
    linkage="ward"
)
Z.fit(X)

# Extract clusterings
model = FOSCX(
    top_M=5,
    kmin=2,
    kmax=10,
    quality_measure="stability"
)

model.fit(Z)

# Access candidate solutions
print(model.candidates_)
```

---

## Visualisation

```python
model.plot()        # Plot cluster assignments
model.plot_tree()   # Visualise hierarchy
```

---

## Documentation

Full documentation is available at:

https://your-docs-link-here

---

## Performance Notes

- First run includes a Numba compilation step (~10–20 seconds)  
- Subsequent runs are fast (typically sub-second)  
- Best performance is achieved with the "stability" quality measure  

---

## Dependencies

FOSC-X depends on:

- numpy  
- scipy  
- scikit-learn  
- numba  
- pandas  
- matplotlib  
- umap-learn  
- jsonschema  

---

## Development

Install development dependencies and run tests:

```bash
pip install -e .[dev,test]
pytest
```

---

## License

BSD 3-Clause License. See LICENSE.
