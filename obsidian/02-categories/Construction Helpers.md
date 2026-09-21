---
tags: [layer/categories, concept]
code: construction_helpers/
status: stable
---

# Construction Helpers

## What it is

The layer that makes expressions *writable*. It installs the operator overloads on
[[Product Categories]] so that a model can be written as algebra:

```python
import construction_helpers as ch          # for its SIDE EFFECTS
import data_structure.Operators as ops

qk   = ops.Einops.template('q h d, x h d -> h q d')
sm   = ops.SoftMax.template()
mask = ops.WeightedTriangularLower()
sv   = ops.Einops.template('h q x, x h d -> q h d')

attention_core = qk @ sm @ mask @ sv
```

> [!important] Import it even if unused
> `import construction_helpers as ch` is required for the overloads to exist. The
> methods on `ProductCategory.Morphism` raise `NotImplementedError` until this package
> is imported.

## The three operators

| operator | file | meaning |
|---|---|---|
| `@` | `composition.py` | sequential composition, **with automatic axis alignment** |
| `*` | `product.py` | parallel product; `Object * Object -> Object`, `Morphism * Morphism -> Morphism`, and tuples flatten |
| `>>` | `lift.py` | batch lifting: `Axes >> Morphism` broadcasts the morphism over the axes |

`einops.py` parses the `'q d, x d -> q x'` signature strings into weaves and reindexings;
`signature.py` handles the shorter generic-operator signatures; `simple_helper.py` has
`make_composed` / `make_product`, which flatten nested `Composed` / `ProductOfMorphisms`
rather than nesting them.

## Axis alignment

Alignment is the part with real content. When `f @ g` is formed, `f`'s codomain and `g`'s
domain have to be the same objects. They rarely are, because they are separately constructed axes
with separate UIDs that merely print alike ([[UIDs and Names]]).

`composition.align_axes` / `align_axis` line them up **positionally** and then identify
them with an `EqualityClass`, per [[Rewriting]]. It never aligns by name, because a name exists for display.
The sizes merge through `size_class`: two size symbols become one class, a symbol beside
a size written in other symbols is replaced by that size, and two such sizes are left as
they are. Since 2026-09-15 an axis may therefore be declared with a product size, as the
query axis of `DeepSeekV41Flash.ipynb` is with `|a| |b|`, and every template axis it meets
takes the product; `fd.canonical_rank` orders a class whose canonical has no uid.
`add_excess_lift` handles the case where one side has more axes than the other, lifting
the shorter side.

> [!warning] `add_excess_lift` reads the first objects alone, and lifts the whole side
> `f @ g` compares `f.cod()[0].shape()` against `g.dom()[0].shape()` and lifts *all* of
> the shorter morphism by the excess. A product mixing a rank-deficient operation with
> `hold`s, such as `(SoftMax.template() * hold(SEL))` after a rank-3 wire, therefore lifts the
> holds too and dies with "Cannot align axes of different lengths". Pre-lift the one
> operation explicitly: `chl.morphism_object_lift(op, ProdObject(axes))` (the `over`
> helper in the `sota/` notebooks). Single-object chains never hit this.

> [!note] Lifting a 0-input `Broadcasted` puts the axes on `backup_degree`
> A morphism with an empty domain has no reindexing to compose the lift into, so
> `broadcasted_stride_lift` prepends the lifted axes to its `backup_degree`, which is the
> same place the weaves put their new `TILED` slots. Until 2026-08-21 this was a
> hard rule against lifting a parameter array at all: `degree()` came back empty,
> the output weave gained `TILED` slots with nothing to imprint, and the first `cod()`
> downstream died with `StopIteration` inside `imprint_to_degree`. The modelling advice
> stands on its own: a weight does not vary with the batch, so the batch axis is
> honest **explicit in the einsums beside it** (`'q m, e m f -> q e f'`) and a lifted
> weight would *say* it was drawn once per batch element. What changed is that the lift
> is now well-formed when it is what the caller means ([[Broadcasted Category]],
>).

> [!warning] Bare-tuple composition infers its `Rearrangement`'s domain from *positions*
> `(0, 1, 0, 2, 3) @ f` builds a `Rearrangement`, and its `_dom` has to be read from the first
> cod position naming each index, rather than by indexing `f.dom()` with the mapping value.
> The wrong version worked by coincidence for every order-preserving mapping in the
> existing notebooks (`(0, 0)`, `(0, 1, 1)`) and broke on any interleaving one.

## Lifting

`lift.py` defines four liftings, and the docstring is the spec:

```
Axes            >> (Datatype | Array)   -> Array
StrideCategory  >> (Datatype | Array)   -> Broadcasted
StrideCategory  >> Broadcasted          -> Broadcasted
Axes            >> BroadcastedCategory  -> BroadcastedCategory
```

`dynamic_object_lift` is the dispatcher. [[Linear Expansion]] uses `chl` directly to
re-lift a rewritten `Linear` over the same degree it was broadcast across.

## See also

- `notebooks/base_features/BuildingAModel.ipynb` — each rule of construction beside the
  code that follows it, asserted
- [[Product Categories]] — what the overloads build
- [[Operators]] — what goes at the leaves
- [[Broadcasted Category]] — what alignment is aligning
- [[Rewriting]] — how alignment actually identifies two axes
