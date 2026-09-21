---
tags: [layer/categories, concept]
code: data_structure/BroadcastedCategory.py
status: stable
---

# Broadcasted Category

## What it is

**Br** is the category deep learning models are morphisms in. Its objects are **arrays**
, meaning a datatype with a shape of axes. Its seed morphism is `Broadcasted`: one operation
applied pointwise over a tiling of indices, with each input read through an affine
reindexing.

A model is a `ProdCategory` ([[Product Categories]]) whose leaves are `Broadcasted`.

## The shape of a `Broadcasted`

```python
Broadcasted(
    operator:       Operator,
    input_weaves:   Prod[Weave],      # one per input array
    output_weaves:  Prod[Weave],      # one per output array
    reindexings:    Prod[StrideCategory],   # one per INPUT
    backup_degree:  ProdObject | None,      # the degree, None unless dom is empty
)
```

- A **weave** states, position by position, whether that axis is part of the operation's
  **target**, which is the array the underlying operation receives, or is a `WeaveMode.TILED`
  position broadcast over. See [[Weaves and Degree]].
- The **degree** is the tiling common to all the weaves: `degree()` is
  `iallequals(m.dom() for m in reindexings)`. Every reindexing shares it by construction,
  and `degree()` raises loudly if they do not. A morphism with an empty domain has no
  reindexing, so `degree()` returns its `backup_degree`.
- `dom()` is each input weave with the *codomain* of its reindexing imprinted into the
  TILED positions; `cod()` is each output weave with the degree imprinted.

One `Broadcasted` therefore states: for every point of the degree, apply `operator` to the target
slices found at the reindexed positions of each input, and write the result at this
point.*

## The mathematics

The split is what makes broadcasting a first-class, analysable thing rather than a runtime
convention. Because the reindexings are affine ([[Stride Category]]), one operator
expresses:

- **transposes** — permute the degree into the target;
- **diagonals** — one degree index feeding two input positions;
- **repetitions / broadcasts** — a degree axis absent from an input's mapping;
- **convolution windows** — an output index `(x, k)` reading input `x + k`.

`README.md`'s `broadcast_weave.png` is the picture.

## Where it lives

`data_structure/BroadcastedCategory.py`.

| name | what it is |
|---|---|
| `Datatype`, `Reals`, `Natural` | what an array holds. `Natural(max_value)` is an index type |
| `Array[B, A]` | `datatype` plus `_shape: Prod[A]` |
| `WeaveMode.TILED` | the marker for a broadcast-over position |
| `Weave[B, A]` | datatype plus a shape of (axis \| TILED) |
| `Operator` | the base for every operation, covered by [[Operators]] |
| `Broadcasted[B, A, O]` | the seed morphism |

`data_structure/Category.py` re-exports everything from **St**, **Br** and
[[Product Categories]]. Almost all code imports `data_structure.Category as cat`.
`data_structure/BrTyping.py` gives the type aliases for `Composed`/`Product`/`Block`
specialised to Br.

## The rules

- **All reindexings of one `Broadcasted` share a domain.** If they do not, `degree()`
  raises `"Inconsistent reindexing morphisms"`, which almost always means a construction
  bug upstream.
- **An `Einops` has exactly one output segment.** `Einops.template` asserts it, and
  [[Einops Rearrangement]] relies on it.
- `target()` strips the reindexings to identities, giving the un-broadcast operation, and
  empties `backup_degree` where the domain is empty, for the same reason.
- **`backup_degree` is a `ProdObject` on a `Broadcasted` whose domain is empty and `None`
  on every other one.** `has_empty_domain()` and `backup_degree is not None` therefore
  report the same condition. `__post_init__` enforces the correspondence: it raises when a
  morphism with inputs is given a `backup_degree`, and gives a morphism built with an
  empty domain and no degree the empty one. A morphism with an empty domain that is not
  broadcast carries `ProdObject(())` rather than `None`.
  `backup_degree` exists because the degree is *derived*, and a morphism with no inputs
  has nothing to derive it from. Without it an operator with no operands could only be
  broadcast as a value followed by a repeating `View`, which is one value copied, and
  wrong for anything stochastic. Every rewrite that rebuilds a `Broadcasted` by hand propagates it
  (`node_expansion`, `operator_expansion`, `lift`), and the
  signature parsers set it when a signature has no inputs.

## See also

- [[Weaves and Degree]] — the representation of broadcasting, in detail
- [[Operators]] — the seed operations
- [[Stride Category]] — the reindexings
- [[Construction Helpers]] — how these are actually built
