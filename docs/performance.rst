Performance
===========

FOSC-X is designed to efficiently extract high-quality clusterings from
hierarchical structures. In practice, it scales well with tree size and
supports fast exploration of multiple candidate solutions.


Scaling Behaviour
-----------------

The runtime of FOSC-X is primarily determined by the size of the hierarchy
(i.e. the number of nodes) and the choice of quality measure.

In practice:

- Runtime scales approximately linearly with the number of nodes  
- Dependence on ``kmax`` is typically weak (near constant or sublinear)  
- Computing multiple candidate solutions (``top_M``) scales quadratically  
  (however, ``top_M`` is typically small in practice)

The figure below illustrates empirical scaling behaviour across different
settings.

.. image:: _static/performance_scaling.png


Quality Measures
----------------

The choice of quality measure has a significant impact on performance:

- ``"Stability"``  
  Fast and lightweight (tree-based)

- ``"Modularity"``  
  More computationally expensive due to k-nearest neighbor graph construction  

- ``"PFCE"``  
  More computationally intensive than ``"Stability"`` and ``"Modularity"``, based on the minimum
  spanning tree  

In general, ``"Stability"`` is the most efficient option and may be recommended
for larger datasets.



Numba Compilation
-----------------

FOSC-X uses `Numba <https://numba.pydata.org/>`_ to accelerate core computations.

On first execution, functions are compiled, which introduces a noticeable
one-time overhead (typically around 10–20 seconds).

Once compiled, the generated code is reused, and subsequent runs are
significantly faster. In most cases, the actual runtime on datasets is
on the order of fractions of a second.

.. note::

   The compilation cost is typically incurred only once per installation
   (or when the environment changes). If the package remains installed,
   subsequent usage will not require recompilation.


Effect of Condensation
----------------------

Applying tree condensation (``min_cluster_size > 1``) reduces the size of the
hierarchy by removing small clusters.

This can improve performance by reducing the number of nodes that must be
evaluated, although the removed nodes are typically small and inexpensive to
process.


Theoretical Complexity
----------------------

In the worst case, the dynamic programming procedure has complexity:

.. math::

    O(N \cdot M^2 \cdot k_{\max} \cdot \log(M \cdot k_{\max}))

where:

- ``N`` is the number of nodes in the hierarchy  
- ``M`` is the number of candidate solutions (``top_M``)  

In practice, the algorithm behaves much closer to linear in ``N``, with only
weak dependence on ``kmax``.