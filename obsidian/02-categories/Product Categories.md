---
tags: [layer/categories, concept]
code: data_structure/ProductCategory.py
status: stable
---

# Product Categories

## What it is

The compositional skeleton every expression is built from. A **product category** has
objects that are *products* (tuples) of some base type, and morphisms that can be
composed sequentially, placed side by side, and permuted.

`pyncd` uses it twice, with different bases:

- **St**, the [[Stride Category]] — objects are axes, morphisms are affine index maps.
- **Br**, the [[Broadcasted Category]] — objects are arrays, morphisms are broadcast
  operations.

## The five constructors

```
ProdCategory[L, M] =
      M                                  a seed morphism
    | Rearrangement[L]                   permute / copy / delete wires
    | Composed[L, ProdCategory[L, M]]    sequential composition
    | ProductOfMorphisms[L, …]           parallel product
    | Block[L, …]                        a named, possibly repeated grouping
```

| term | `dom()` / `cod()` | notes |
|---|---|---|
| `ProdObject[L]` | — | a tuple of `L`, with `identity()` |
| `Composed` | first's dom, last's cod | `content: Prod[M]` |
| `ProductOfMorphisms` | concatenated | has `partition` / `partition_codomain` to split a flat sequence back into per-factor pieces |
| `Rearrangement` | `_dom`, and `apply(_dom)` | a `mapping: Prod[int]`; `apply` goes dom to cod, `invert` goes back |
| `Block` | the body's | carries a `BlockTag` |

`Rearrangement` is what makes the category **symmetric monoidal with copy and delete**: a
mapping may repeat an index (copy), omit one (delete), or reorder (symmetry). `invert`
uses `iallequals`, so inverting a copy checks the copies agree.

## A definition pairs two morphisms

```python
DefinedExpression(left_hand_side: ProdCategory[L, M], right_hand_side: ProdCategory[L, M])
```

A `DefinedExpression` states that its left-hand side is defined to be its right-hand
side. The two sides have one domain and one codomain, and
`DefinedExpression.template` raises `SidesOfADefinitionDisagree` where they differ. It
is a `Term` that is drawn and is never composed, so it has no `dom()` and no `cod()`.
tsncd draws the two sides in one row with `:=` between them, the listing of
[[Agent Display]] prints the two listings with `:=` on a line between them, and
`notebook_diagrams.show_diagram` applies each presentation to each side. The user asked
for it on 2026-09-17, for two uses.
`algebra.define_by_expansion.define_by_standard_expansion` pairs an operator with the
expansion `algebra.registries.standard_expansions` registers for it. An operator whose
value depends on an index is defined at that index, with the operator followed by the
read of one position on the left and the expression for the value there on the right,
per [[Representing Models]]. No rewrite reads a definition yet, so a defined operator
stays a `GenericOperator` in the expression that holds it.

## A block carries meaning as well as appearance

```python
BlockTag(repetition: Numeric = Integer(1), aesthetics: BlockAesthetics | None = None)
```

- **`repetition != 1` means a loop.** A block repeated `repetition` times denotes a loop
  that carries a running value from one iteration to the next, which is how a stream is
  written out.
- **`aesthetics` carries how the block is drawn.** [[Diagram Display]] reads it, and it
  says nothing about the semantics of the body.

## Operator overloads

`ProductCategory` declares `__matmul__`, `__mul__`, `__rrshift__` and raises
`NotImplementedError`. The implementations are installed by [[Construction Helpers]],
which is why `import construction_helpers as ch` is needed for its side effects even where
the name is unused.

| operator | meaning |
|---|---|
| `@` | sequential composition, with automatic axis alignment |
| `*` | parallel product |
| `>>` | batch lifting over an axis |

## See also

- [[Stride Category]], [[Broadcasted Category]] — the two instantiations
- [[Construction Helpers]] — the overloads
- [[Hypergraphs]] — the rewriting form this converts to and from
- [[Operators]] — the seed morphisms
