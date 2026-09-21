---
tags: [layer/hypergraphs, concept]
code: graphs/processing/HypergraphAnalysis.py
status: stable
---

# Hypergraph Analysis

## What it is

`HypergraphAnalysis` answers connectivity questions about a graph: what feeds what, what
lies to the left of a subgraph, and which nodes enter or leave it.

```python
a = hga.HypergraphAnalysis(graph)
a.left_subgraphs(target)    # what feeds `target`
a.right_subgraphs(target)   # what `target` feeds
a.nodes_left(target)
a.nodes_right(target)
```

`HypergraphSpecialTag.LEFT` and `HypergraphSpecialTag.RIGHT` denote the world outside the
graph, meaning the graph's own domain and codomain, so a subgraph fed by `LEFT` is fed from
outside.

## The link structure

The indices are built from three named tuples:

```
Graph2Node  (graph, cod_segment, node)     a subgraph's output reaches a node
Node2Graph  (node, dom_segment, graph)     a node reaches a subgraph's input
Graph2Graph (left, cod_seg, dom_seg, right)  the join of the two
```

`link(g2n, n2g)` is the join, and everything else feeds it: `cod_g2n`, `dom_n2g`,
`graph_left_nodes` and `graph_right_nodes`.

## The rules

> [!warning] It answers over one level of direct children
> A graph is typically several levels of `Composed` and `ProductOfMorphisms` deep, so an
> analysis of the top graph states nothing about the leaves. The idiom for a deep walk is
> `flat_subgraphs` with a producer and consumer lookup by node identity, which is what
> `einops_rearrange.rearrange_einops` does, for exactly that reason.

> [!warning] Indices are `cached_property`, but instances are constructed fresh per query
> so nothing is shared between two analyses. Constructing one is not free, so a caller
> asking many questions about one graph should build one analysis and keep it.

## Where it is used

- [[Hypergraph to Morphism]] — the branch construction consults it constantly.
- [[Pathway Collapse]] — finding what reads a wire, and on which side of a rewrite.

`ReverseCrawler` lives in this file as well as in
`graphs/processing/hypergraph_crawler.py`, per [[Crawlers]].

## See also

- [[Hypergraphs]]
- [[Crawlers]] — the guided-traversal abstraction built on top
