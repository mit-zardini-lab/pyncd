---
tags: [layer/categories, concept]
code: data_structure/StrideCategory.py
status: stable
---

# Stride Category

## What it is

**St** is the category of axes and **affine index maps**. Its objects are axes (each with
a symbolic size. Its morphisms send a tuple of domain indices to a tuple of codomain
indices, each codomain index an affine function of the domain:

$$\text{cod}_j = \sum_i \text{stride}_{ji} \cdot \text{dom}_i + \text{shift}_j$$

`StrideMorphism` stores exactly that: `_dom: Prod[A]` and
`_cod_stride_shift: Prod[(axis, strides, shift)]`. A codomain index outside
`[0, size)` of its axis names no position, and a read there yields the universal unit,
per *A read outside an axis* below.

## Why affine

**Affineness is what the rest rests on.** It makes the dependency between two axes
integer linear algebra rather than symbolic execution. The question of which positions of
one array a position of another reads is answered by composing matrices, and every answer
is exact.

It is also expressive enough for the patterns that matter. With output index $(i, j)$
mapping to input index:

| map | what it expresses |
|---|---|
| $(i, j)$ | the identity |
| $(j, i)$ | a transpose |
| $(i, i)$ from $i$ | a diagonalisation |
| $x + k$ from $(x, k)$ | a **convolution window** |
| $s\cdot i$ | a strided read |

That last one is why convolution needs no special case anywhere in the package: it is a
reindexing followed by a contraction.

## Where it lives

`data_structure/StrideCategory.py`, which is a short file again since 2026-09-15: the
axis, the affine map and nothing else.

| name | what it is |
|---|---|
| `Axis` | a `UTerm` with `_size: Numeric`; `named(name)` sets both the name and a matching `\|name\|` size symbol |
| `RawAxis` | an ordinary axis, with no affine form on it |
| `StrideMorphism[A]` | the affine map; `strides()`, `from_matrix(*rows)` |
| `StrideCategory[A]` | `ProdCategory[A, StrideMorphism[A]]` |
| `StrideRow[A]` | one codomain row, as the axis, its strides and its shift |

`Rearrangement`, from [[Product Categories]], is the case that only permutes, copies and deletes. It is a
`StrideMorphism` whose matrix is a 0/1 selection with no shift, and most reindexings in
practice *are* rearrangements. `term_utilities.get_mapping` collapses one to a
`Prod[int]`.

## A read outside an axis

A codomain index may fall outside `[0, size)` of its axis. The position read there holds
the universal unit of [[The Universal Unit]], which a fold over an axis ignores and a
pointwise operation preserves. The requester ruled on 2026-09-14 that this is the whole
mechanism. A negative index is the unit, always, an index at or past the size is the unit
as well, and no separate guard exists.

That is the category's whole account of an empty position. Which positions of a read
array are empty is an affine form of the position, and deriving that form, carrying it
through further reads and folds, and restoring it where two forms meet is
[[Advanced Axis Dynamics]], in `advanced_axis_dynamics/`. Nothing in this module mentions
it, and the reads that produce one are in
[[Padding and Masks as Sparse Axes]].

The sign tests the derivation rests on are here, in the algebra section of
`data_structure/Numeric.py`: `nm.is_nonnegative_for_positive_symbols` and its companions
treat every size symbol as a positive integer, which is what lets a row be tested at the
corners of its domain box.

## The rules

- **Axes are identified by UID, never by name.** `RawAxis.named('q')` twice gives two
  different axes with two different size symbols that both print `|q|`. See
  [[UIDs and Names]].
- Every `Broadcasted` in [[Broadcasted Category]] holds one reindexing **per input
  segment**, and they all share a domain, which is the **degree**.
- **A `StrideMorphism` is read contravariantly by a `View` and covariantly by
  `aops.CovariantView`.** The covariant reading is a function only for an injection from
  the domain box, which `mark_sparse_codomains.merge_groups` tests, and it is the merge
  whose inverse no reindexing states, per [[Advanced Axis Dynamics]].
- **A negative index is the unit, always, and so is an index at or past the size.**
  Write a condition no read states as a relative read followed by a merge, and never as a
  guard field, a mask operator or a row onto an axis a `Rearrangement` deletes, which the
  Cartesian structure lets the algebra erase.

## See also

- [[Broadcasted Category]] — where reindexings are used
- [[Weaves and Degree]] — how a reindexing lines up with a shape
- [[Advanced Axis Dynamics]] — the empty positions a read leaves, and what is derived
  from them
- [[Product Categories]]
