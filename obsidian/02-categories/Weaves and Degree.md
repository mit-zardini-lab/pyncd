---
tags: [layer/categories, concept]
code: data_structure/BroadcastedCategory.py
status: stable
---

# Weaves and Degree

## What it is

The split between a weave and its target is `pyncd`'s representation of broadcasting, and
it is the single idea most worth having straight before reading anything else here.

A `Weave._shape` is a shape whose entries are either an axis or `WeaveMode.TILED`:

```
Weave(Reals, (TILED, d, TILED, x))
       ^ broadcast over   ^ the operation consumes these
```

- The TILED positions are the ones the operation is broadcast over. Collectively, across all
  of a morphism's weaves, they are the degree.
- The remaining positions are the target, which is the array the underlying operation
  receives. `weave.target()` extracts it.

The listing in [[Agent Display]] prints the split directly. `%4[qTλ, {dRλ}]` states that the
operation is broadcast over `qTλ` and consumes `dRλ`, with the braces marking the target.

## The operations on a weave

| method | what it does |
|---|---|
| `target()` | the `Array` of the non-TILED positions |
| `select_degree(xs)` and `select_target(xs)` | project a parallel sequence onto the TILED positions, or onto the non-TILED ones |
| `target_idx()` | the positions of the target |
| `imprint(tilings)` | fill the TILED slots from an iterator, keeping the axes |
| `imprint_target(array)` | replace the target, keeping the TILED slots |
| `imprint_to_degree(degree)` | fill the TILED slots with the degree, giving the full `Array` |
| `imprint_axes(tilings, axes)` | fill both |
| `rearrangement(degree)` and `inverse_rearrangement(degree)` | the permutation between `(degree…, target…)` and the woven order |
| `Weave.from_arrays(arrays)` | every position a target position |

`imprint_to_degree` is the central one. It is what `Broadcasted.dom()` and `.cod()` use
to turn a weave back into an array.

## Degree

The degree of a `Broadcasted` is the tuple of axes every TILED position broadcasts against.
It is `broadcasted.degree()`, the common domain of all its reindexings.

A degree axis need not appear in any input's reindexing. Where none does, every operand is
broadcast along it and the output is the operator's value repeated along that axis. A repeat
is written that way, as a `View` whose reindexing drops the axis, which is why there is no
repeat operator, per [[Expression Simplification]].
Each input's reindexing states which degree positions its own TILED slots pull from, and with
what strides.

One operator can therefore express a transpose, a diagonal, a repetition or a convolution
window: the operation is fixed and the reindexing of the degree varies.

> [!tip] Reading a shape
> A `Broadcasted`'s input array is `input_weave.imprint_to_degree(reindexing.cod())`, and its
> output array is `output_weave.imprint_to_degree(degree())`. An input goes through the
> reindexing and an output does not.

> [!important] Every weave has exactly `len(degree())` TILED positions
> The degree is one tuple shared by the whole morphism, and each weave's TILED slots are that
> weave's view of the tuple, so the counts have to agree. `imprint_to_degree` does not report
> a disagreement. It consumes the degree lazily and stops when the slots run out, so a weave
> one slot short still produces a plausible array, built out of a prefix of the degree. What
> breaks is anything that pairs a degree position with a weave position: a reindexing's
> `mapping` indexes the degree, and against a short weave it indexes past the end. In
> [[Diagram Display|tsncd]] the failure surfaces as
> `Cannot read properties of undefined (reading 'anchors')`, thrown from `link_weaves`. An
> axis that is in the degree is TILED, even where the same axis also appears in the operator's
> target elsewhere in the expression.

## Common traps

- **`weave.target()` drops every TILED axis.** Rebuilding a value that has to keep a
  broadcast axis, meaning `s[q]` rather than a bare scalar, calls `imprint_to_degree` instead.
  Section 4 of [[Einops Rearrangement]] writes it out, because it was a bug.
- **`dom()` and `cod()` report the lifted shape**, degree axes included. The pre-broadcast
  shape of an operation has to come from each weave's own `.target()`, which is exactly why
  [[Linear Expansion]] reads shapes off the weaves rather than off `dom()`.
- **A `Broadcasted` with an empty domain keeps its degree in `backup_degree`.** `degree()`
  is derived from the reindexings, so an operator with no inputs has nothing to derive it
  from, and `backup_degree` holds it for that case alone. A morphism with an empty domain
  always carries a `ProdObject`, empty in turn when that morphism is not broadcast, and a
  morphism with inputs always carries `None`, so `has_empty_domain()` and
  `backup_degree is not None` report the same condition. `__post_init__` enforces the
  correspondence. Before the field existed, such a morphism was written as a value followed
  by a repeating `View`, which is one value copied. A copy is right for a constant and wrong
  for a sampler, where n independent draws were not expressible at all. In
  [[Diagram Display|tsncd]] such a degree is drawn with a dot on every degree wire, because
  `link_weaves` dots any degree anchor that no input reindexing names, and a morphism with
  an empty domain names none.

## See also

- [[Broadcasted Category]] — the morphism weaves live in
- [[Stride Category]] — the reindexings the degree is mapped through
- [[Expression Simplification]] — permuting a weave by a reindexing's mapping, and when that is legal
- [[Agent Display]] — how weaves are printed
- [[Operators]] — `Einops.signature` lines up positionally with the non-TILED positions
