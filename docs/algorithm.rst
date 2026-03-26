How FOSC Works
==============

FOSC extracts flat clusterings from a hierarchical clustering tree by exploring
many possible ways of “cutting” the tree and selecting the best ones according
to a chosen quality measure.

A useful way to think about this is: every clustering corresponds to selecting a
set of nodes in the tree that together cover the data. The goal of FOSC is to
find the best such selections.


Dynamic Programming over the Tree
----------------------------------

The key observation behind FOSC is that the globally optimal clustering can be
constructed from optimal solutions to smaller subtrees.

Because the objective function is additive (the total score is the sum of cluster
scores), the best solution for a subtree depends only on that subtree. This means
that if we know the best way to cluster each child subtree, we can combine them to
form the best solution for the parent. This property is what makes dynamic
programming possible.

Rather than trying every possible combination of nodes (which would be
computationally infeasible), FOSC works from the bottom of the tree upwards.

The process begins at the leaves. At this level, the choices are trivial: each
leaf either forms a cluster on its own or contributes to a larger cluster higher
up in the tree.

As we move up the tree, each node represents a potential cluster that contains
all of its descendants. At this point, FOSC faces a decision:

- Treat this node as a single cluster, or  
- Defer the decision and instead use the best clusterings found in its children  

These two options correspond to either “cutting” the tree at that node, or
continuing to split into smaller clusters below it.

For every node, FOSC evaluates both possibilities and keeps track of the best
partial solutions for that subtree. Because optimal solutions can be built from
optimal sub-solutions, considering only the best options at each step is sufficient
to recover the global optimum.

By the time the root is reached, the globally optimal clustering has been
constructed from these local decisions.


Tracking Multiple Solutions (``top_M``) (Unconstrained)
--------------------------------------------------------

The same idea extends naturally to finding not just the single best clustering,
but the top ``top_M`` solutions.

Instead of storing only the best solution at each node, FOSC stores the best
``top_M`` solutions for each subtree.

The key insight is that the top ``top_M`` global solutions can be constructed
entirely from the top ``top_M`` solutions of each child subtree. Any solution
built using a worse (lower-ranked) child solution can always be improved by
replacing it with a better one.

As a result, we only need to keep a small number of candidates at each node,
while still guaranteeing that the globally best ``top_M`` solutions are found.

All solutions are tracked simultaneously during a single pass over the tree,
making this efficient in practice.


Cluster Count Constraints (``kmin``, ``kmax``)
-----------------------------------------------

FOSC-X extends this framework by allowing constraints on the number of clusters
in the final solution.

At a high level, we now require that the total number of selected clusters lies
between ``kmin`` and ``kmax``.

However, introducing these constraints breaks the key property used above:
locally optimal solutions are no longer guaranteed to lead to globally optimal
solutions.

A solution that is optimal within a subtree may:

- Become infeasible when combined with the rest of the tree, or  
- Need to be replaced by a lower-scoring alternative to satisfy the constraints  

This means we can no longer keep only the best solutions based purely on quality.
Instead, we must also consider whether a solution can still lead to a valid final
clustering.


Bounding the Number of Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To handle this, FOSC-X computes bounds for every node describing how many
clusters must or can be selected outside its subtree.

- **Lower bound (LB)**: the minimum number of clusters that must be selected
  outside the subtree  

- **Upper bound (UB)**: the maximum number of clusters that can be selected
  outside the subtree  

These bounds come directly from the structure of the tree.

For the lower bound:

- Any subtree containing non-noise data must contribute at least one cluster  
- Therefore, outside the current subtree, we must select at least one cluster
  from each sibling subtree that contains data  

For the upper bound:

- Along any root-to-leaf path, only one cluster can be selected  
- The maximum number of clusters is therefore given by the number of terminal
  (deepest valid) clusters in the tree  

The bounds for a node describe how much “cluster budget” remains outside that
subtree.

Noise plays an important role here: subtrees consisting entirely of noise may
contribute zero clusters, reducing the lower bound.


Using Bounds During Dynamic Programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each partial solution at a node has a known number of clusters.

When extending this solution to the full tree, additional clusters must be
selected outside the subtree. The bounds tell us how many clusters we are allowed
(or required) to add.

A partial solution is only kept if it can still be extended into a valid final
solution.

Specifically:

- If even the **minimum required extension** exceeds ``kmax``, the solution is
  discarded  

- If even the **maximum possible extension** cannot reach ``kmin``, the solution
  is discarded  

This ensures that only solutions that can still lead to a valid clustering are
propagated upward.


Effect on the Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

The structure of the dynamic program remains the same, but the pruning strategy
changes.

We no longer keep only the top ``top_M`` solutions by quality. Instead, we keep all 
solutions that are still feasible under the cluster-count constraints.

This means that lower-scoring solutions may need to be kept if they are required
to construct a valid global solution.


Pruning and Efficiency
----------------------

To keep the algorithm efficient, FOSC-X applies **dominance pruning**.

The key idea is that solutions should be compared not just by their score, but
also by how many additional clusters they allow outside their subtree.

Each partial solution is associated with a range:

- The minimum number of additional clusters it requires  
- The maximum number it allows  

If two solutions are feasible for the same number of additional clusters, then
only the higher-scoring one needs to be kept.

More generally:

- For each possible number of additional clusters, we keep only the best
  ``top_M`` solutions  
- Any solution that is never optimal for any feasible completion is discarded  

**Example**

Consider two partial solutions at a node:

- Solution A: score = 10, feasible for 1-2 additional clusters  
- Solution B: score = 9, feasible for 2-3 additional clusters  

Now consider how they behave:

- For **k = 2**, both A and B are feasible → A is better, so B is not needed  
- For **k = 1**, only A is feasible → A must be kept  
- For **k = 3**, only B is feasible → B must be kept  

Even though B has a lower score, it is the *only* valid solution for some
completions, so it cannot be discarded.

In contrast, if we had:

- Solution A: score = 10, feasible for 1-3 additional clusters  
- Solution B: score = 9, feasible for 1-3 additional clusters  

Then A dominates B for all possible cases, and B can be safely removed.

This dramatically reduces the number of solutions that need to be stored, while
still guaranteeing that the globally optimal solutions are preserved.

In practice, this means that instead of tracking all feasible solutions, we only
keep a small set of representative ones that cover all possible ways of completing
the clustering.


Overall Summary
---------------

FOSC can be understood as a structured way of exploring how to cut a
hierarchical clustering tree.

Starting from the leaves, it builds clusterings bottom-up by deciding at each
node whether to stop and form a cluster, or continue splitting into smaller
clusters below. These local decisions are combined efficiently using dynamic
programming.

In the unconstrained setting, optimal solutions can be constructed entirely from
locally optimal choices. When multiple solutions are required, the same idea
extends by tracking only the best ``top_M`` candidates at each step.

When cluster count constraints (``kmin``, ``kmax``) are introduced, local
optimality alone is no longer sufficient. FOSC therefore tracks not only the
quality of solutions, but also how they can be completed into valid global
clusterings.

This is achieved using:

- Bounds on how many clusters must or can be selected outside each subtree  
- Pruning rules that retain only solutions that are optimal for at least one
  feasible completion  

Together, these ideas allow FOSC to efficiently explore a large space of possible
clusterings, while guaranteeing that the best valid solutions are found.