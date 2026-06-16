The FOSC-X Cluster Extraction from Hierarchies Library
======================================================

FOSC-X is a suite of tools for extracting high-quality clusterings from
hierarchical clustering trees.

At its core, the FOSC-X algorithm provides a principled approach for selecting
optimal flat partitions from hierarchies produced by a range of popular
clustering algorithms across several Python libraries.

In addition to cluster extraction, FOSC-X includes tools for:

- Pre-processing hierarchical structures (e.g. tree condensation)  
- Handling noise and small clusters  
- Evaluating solutions using multiple quality measures  
- Visualising both cluster assignments and hierarchical structure  

Together, these components enable efficient exploration and analysis of
multiple candidate clusterings derived from a single hierarchy.

.. toctree::
   :maxdepth: 2
   :caption: Usage

   basic_usage
   visualisation
   Tree_formats
   parameters
   quality
   input_processing
   advanced
   performance

.. toctree::
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`