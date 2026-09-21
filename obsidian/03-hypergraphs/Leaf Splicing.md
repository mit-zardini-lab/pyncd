---
tags: [layer/hypergraphs, algorithm]
code: graphs/processing/leaf_splicing.py
agent: Claude Opus 5 (1M context), reasoning effort high, 2026-09-20
status: stable
---

# Leaf Splicing

## What it is

Replacing an operation of a hypergraph by one or more new operations, wherever in the
nesting that operation sits, and then rebuilding the domain and codomain of every
container around it.

The three functions read a `Multigraph`, a `HypergraphBlock` and a `HypergraphRoot` and
nothing else. `quantization/processing/conversion_insertion.py` and
`quantization/processing/quantise_model.py` are the callers: inserting a cast between two
quantisations replaces one leaf by three, wherever in the nesting that leaf sits.

## Where it lives

| function | what it does |
|---|---|
| `all_leaves(graph)` | every operation of the graph, descending through every block, loops included |
| `splice(graph, replacements)` | each leaf named by uid in `replacements` replaced by the tuple of subgraphs given for it, spliced flat into the container the leaf occupied, followed by `rescope` |
| `rescope(graph)` | every container's `dom` and `cod` rebuilt from what its members now consume and produce, with each wire's object taken from its producer |

The three are written from the recursions `_splice_seq`, `_splice_body`, `_splice_one`,
`_node_objects` and `_rescope`. `_splice_body` handles a block whose body is one
subgraph rather than a `Multigraph`.

## The rules

> [!warning] `flat_subgraphs` is not the leaf walk
> `hg.flat_subgraphs(g, remove_blocks=True)` descends through a block only when its
> `repetition` is 1, so a loop block comes back whole and a walk that reads `.wraps` on
> it raises. `all_leaves` descends into `.body`, per [[Hypergraphs]].

> [!warning] Call `rescope` after any splice that changes a container's `dom` or `cod`
> A block that gains an operation reading a wire from outside has gained that wire on its
> domain, and nothing else works that out. The symptom is
> `HypergraphObject(...) is not in list` from [[Hypergraph to Morphism]].

**A container's existing wire order is kept, and only genuinely new wires are
appended.** `_rescope` recomputes which wires appear and leaves the order alone, placing
the new ones in order of first consumption. The order a container was built with was a
deliberate choice made where it was built, and re-deriving it from what the body happens
to consume first crosses the wires. A loop body that splits one operand before it reads
another, where the surrounding graph produces the second first, is the case that shows it.

**A loop body's last `carried` wires stay last.** They are the seeds of the values the
loop carries, one per wire of its codomain, and a new wire is inserted before them.

**Splice the operation alone and leave its consumers alone.** Rewriting the whole region
connects the consumers to the value from before the splice and bypasses the operation
just inserted, per [[Rewriting]].

## See also

- [[Hypergraphs]] — the form these functions walk
- [[Hypergraph to Morphism]] — the conversion that reports a wiring left unrebuilt
- [[Quantization]] — the pass that splices a cast between two quantisations
