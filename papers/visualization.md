# Visualizing Diagrams in Br: How tsncd Works

This document gives a non-technical overview of how `tsncd` (the TypeScript
companion to `pyncd`) converts a categorical diagram into an interactive SVG
display. It is aimed at readers who know the mathematics but are not
TypeScript programmers.

---

## 1. The Big Picture

A diagram in the broadcasted category **Br** is authored in Python using
`pyncd`. When you ask `pyncd` to render a diagram it serializes the entire
expression tree — operators, weaves, axes, datatypes — into a JSON blob and
hands it to `tsncd`. `tsncd` then reconstructs the data structure, lays out
the diagram, and draws it in the browser as scalable vector graphics (SVG).

The three-step pipeline is:

```
Python (pyncd)                 JSON over file / WebSocket          TypeScript (tsncd)
─────────────────────                                              ──────────────────────────────
Broadcasted expression tree ──────────────────────────────────►  reconstruct → lay out → draw
```

Everything visible in the browser is produced entirely on the TypeScript
side. Python only produces data; TypeScript decides where every pixel goes.

---

## 2. Reconstructing the Data Structure

Before any drawing can happen, `tsncd` must reconstruct the Python objects
from JSON.

### Term serialization

Every Python class that appears in the expression tree — axes, datatypes,
operators, weaves, the `Broadcasted` wrapper itself — is a subclass of
`Term`. When serialized, each object becomes a JSON array
`["ClassName", arg0, arg1, ...]` where the arguments are the constructor
parameters in order.

On the TypeScript side there is a parallel class hierarchy. Each class
registers itself in a global `TermDirectory` at module load time (via a
`@register_term` decorator). When `tsncd` deserializes JSON, it looks up
`ClassName` in `TermDirectory` and calls the constructor with the positional
arguments. The result is a live TypeScript object of the right type.

This means **field order in TypeScript constructors must exactly match the
Python dataclass field order**. Class inheritance is included: a Python
subclass serializes inherited fields first, so the TypeScript subclass
constructor must also accept them first.

### What gets reconstructed

The top-level object is a `Broadcasted<Datatype, Axis, Operator>`. It has:

- **`operator`**: the categorical operator (e.g. `TensorEquation`, `Einops`,
  `Linear`)
- **`input_weaves`** / **`output_weaves`**: lists of `Weave` objects, each
  describing one input or output wire bundle
- Each `Weave` carries a list of `Axis` objects and a `Datatype` (e.g.
  `Reals()`, `Bool()`, `Natural()`)

---

## 3. The Rendering Architecture

### The element tree

Once the data structure is reconstructed, `tsncd` builds a tree of
**`DiagramElement`** objects that mirrors the compositional structure of the
diagram. Each element knows its children, and its own dimensions are
recursively derived from its children's dimensions. This tree has two
conceptual levels:

- **Layout elements** that occupy space (boxes, wire bundles, gaps)
- **Wire elements** that draw the connections between boxes

### Two-pass rendering

Drawing happens in two passes:

1. **`post_placement()`** — called top-down through the tree after all
   dimensions are known. Each element computes its absolute position
   (translation) relative to its parent. After this pass every element knows
   exactly where it sits on the canvas.

2. **`update()`** — called top-down after placement. Each element issues its
   actual drawing commands: rectangles, curves, annotations, etc. Because
   positions are already fixed, `update()` is purely imperative.

### The draw handler

All drawing commands go through an abstract `DrawHandler` interface that
provides primitives like `curve`, `arcCurve`, `deltaPolygon`, `circle`, and
`polyline`. The concrete implementation is `HTMLDrawHandler`, which turns
these calls into SVG elements appended to a `<div>` in the browser. The
abstraction means rendering could be retargeted to a different backend
without touching the layout code.

---

## 4. Representing Objects: Meridians and Anchors

In string diagram notation, an object is drawn as one or more **wires**
— vertical lines running between morphism boxes. In `tsncd`:

- A **`Meridian`** represents one wire bundle (one `Weave`). It contains a
  list of **`Anchor`** objects, one per axis.
- An **`Anchor`** represents a single wire. It stores the axis it corresponds
  to and draws a short line segment at its position.

Anchors are linked into a graph: each anchor has a `further` (right) and
`prior` (left) list pointing to the corresponding anchor on the next or
previous morphism box. This graph is built by `link()` pairing up the
right-side anchors of one element with the left-side anchors of the next.
When `update()` runs, each anchor draws a curve to its linked neighbours
using `next_terminal()` / `prior_terminal()` to find the far endpoint.

A **`ProdObjectMeridian`** stacks multiple `Meridian` objects vertically
(one per factor in a product object) and can insert separators between them.

### Annotations on wires

Wires can carry text labels. `tsncd` uses an `AnnotationElement` for
positioned math text rendered via MathJax or similar. Wire labels — axis
names, sizes, or Iverson expression strings — are placed by calling
`annotation_handler.addAnnotation(rect, annotation)` during `update()`.

---

## 5. Representing Morphisms: Operation Boxes

Every operator in the diagram becomes an **`OperationBox`** — a rectangular
box with input wire bundles on the left and output wire bundles on the right.

### The opsRegistry

`tsncd` maintains a registry (`opsRegistry`) that maps operator classes to
`OperationBox` subclasses. When a `BroadcastedBox` is constructed for a
`Broadcasted` object, it looks up the operator's class in the registry and
instantiates the corresponding box. This means adding a new visual style for
a new operator requires only: (a) writing an `OperationBox` subclass and (b)
registering it with `@opsRegistry.registerClass(OperatorClass)`.

Built-in operator boxes include:

| Operator | Box | Visual |
|---|---|---|
| `TensorEquation` | `TensorEquationBox` | `∃` (Bool output) or `Σ` (Reals output) |
| `Einops` | `EinopsBox` | Straight box + arc cups over shared axes |
| `Linear` | `LinearBox` | Notched parallelogram with operator name |
| `Embedding` | `EmbeddingBox` | Same shape as Linear |
| `SoftMax` | `SoftMaxBox` | Triangle/ramp shape |
| `Elementwise` | `ElementwiseBox` | Small box with arrow-tip decorations |
| `ReLU` | `ElementwiseBox` | Same as Elementwise |
| `Identity` | `IdentityBox` | Minimal pass-through marker |
| `Normalize` | `NormalizeBox` | Circular box with oscillating line |
| `WeightedTriangularLower` | `WeightedTriangularLowerBox` | Rectangle with lower-triangle fill |
| `AdditionOp` | `AdditionOpBox` | Box with `+` annotation |

If no specific box is registered for an operator, `OperationBox` itself is
used as a fallback (plain magenta rectangle).

### The BroadcastedBox

`BroadcastedBox` is the top-level container for a single `Broadcasted` node.
It builds:

1. Left wire bundles (`left_anchors`) for input weaves
2. An inner `base_box` from the operator registry
3. Right wire bundles (`right_anchors`) for output weaves
4. An optional `node_box` for index reindexing (stride morphisms)

It has three display modes: **WEAVE** (all wires shown), **NODE** (compact
hexagon node), and **JOIN** (weave-to-node connection).

---

## 6. Composition: ComposedBox and BlockBox

A categorical diagram is usually a **composition** of several morphisms, not
a single one.

- **`ComposedBox`** renders a `Composed` term as a left-to-right sequence of
  `BroadcastedBox` elements with `ComposedGap` connectors between them.
  The connector draws the wires between adjacent boxes using the anchor link
  graph. `ThreadedComposed` is a subclass of `Composed` and uses the same
  `ComposedBox` — the live-pool routing table is a data-level concern that
  does not affect rendering.

- **`BlockBox`** renders a `Block` term — a repeated morphism — with
  repetition brackets drawn at the top and bottom, and an optional title.

- **`ScanBox`** renders a `Scan` term. It extends `BlockBox` (reusing the
  teal backdrop, title, and brackets) and adds a red feedback arc (the
  carry) to mark the state wire looping back from step output to step input.
  The bracket annotation includes the step count N and the recurrence axis
  label. See [iteration.md §7](iteration.md#7-diagrammatic-notation-a-string-diagram-realization-of-the-scanbox)
  for the full rendering specification.

- **`ProductBox`** renders a `ProductOfMorphisms` (stacked objects) vertically.

- **`RearrangementBox`** renders a `Rearrangement` morphism (axis permutations).

The top-level entry point is `CategoryRenderer.display_category()`, which
inspects the term type and dispatches to the appropriate box constructor.

---

## 7. Datatype-Annotated Wire Bundles

Not all wire bundles carry `Reals`-typed data. When a weave's datatype is
`Natural` or `Bool`, the wire bundle gets an extra visual marker: a curved
arrow with a triangle arrowhead and a LaTeX label, drawn by
`DatatypeAnchor`.

In `ArrayMeridian`:

- **`Reals`** weaves: plain grey wires
- **`Natural`** weaves: lime-coloured wires + `ℕ` anchor marker
- **`Bool`** weaves: amber-coloured dotted wires + `𝔹` anchor marker

The dotted style (`stroke-dasharray: '2,6'`) visually signals that Bool
arrays are binary-valued predicates. The `DatatypeAnchor` draws the labelled
arrow beside the first wire of the bundle.

---

## 8. Bool Semiring Support

The Bool semiring is fully supported in `tsncd`. The relevant code is
localised to four files and does not modify the base layout machinery.

### 8.1 `Bool` datatype (`BroadcastedCategory.ts`)

`Bool extends Datatype` is a registered `Term` class whose `to_latex()`
returns `\mathbb{B}`. It is exported from `Category.ts` alongside `Reals`
and `Natural`.

### 8.2 `iverson_expr` on Weave and Array (`BroadcastedCategory.ts`)

`Weave` and `Array` each carry an optional `iverson_expr: string | null`
field (last constructor parameter, defaulting to `null`). When a weave
carries a predicate computed from an Iverson bracket expression, Python
serializes the expression string here. The `target()` and
`imprint_to_degree()` methods forward this field so it is preserved through
categorical operations.

### 8.3 Amber/dotted wires and `𝔹` anchor (`BroadcastedCategoryRenderer.ts`)

`ArrayMeridian` detects `Bool`-typed weaves and applies:

- **Amber colour** (`#f59e0b`) on all wire anchors of the bundle
- **Dotted stroke** (`stroke-dasharray: '2,6'`) to mark binary-valued data
- A `DatatypeAnchor` with amber stroke drawing the `𝔹` LaTeX label
- If `iverson_expr` is set on the weave, that string (e.g. `q ≤ x`) replaces
  the axis name as the wire label on the first anchor — making the predicate
  immediately readable in the diagram

`DatatypeAnchor` accepts an `extra_curve_attributes` parameter so the Bool
branch passes `{stroke: '#f59e0b'}` independently of the Natural branch's
lime styling.

### 8.4 Iverson expression tree terms (`TensorLogic.ts`)

Python serializes `TensorEquation` operators with Iverson expression subtrees
in their `rhs` field. The following `Term` subclasses are registered so
deserialization succeeds:

- **`TensorRef`**: a reference to a named tensor by axis list
- **`IversonConst`**: a numeric constant inside an Iverson bracket
- **`IversonBinOp`**: a binary operation (comparison or arithmetic) inside an
  Iverson bracket
- **`IversonUnaryOp`**: a unary operation (negation, absolute value) inside
  an Iverson bracket

These are deserialization stubs: the display code does not interpret the
expression tree content. `TensorEquation` is also registered here; its
constructor field order mirrors the Python dataclass inheritance order —
`name` (inherited from `Operator`), then `lhs_name`, `lhs_indices`, `rhs`,
`operator` — so positional deserialization produces the correct object.

### 8.5 `TensorEquationBox` (`additionalOperationBoxes.ts`)

`TensorEquationBox` is registered for `TensorEquation` operators:

- **Bool output** (`output_weaves[0].datatype instanceof Bool`): displays **∃**
  (there exists) — the Bool-output TensorEquation tests whether any contracted
  assignment satisfies the predicate, matching the Heaviside demotion in
  Python's `ConstructedTensorEquation`.
- **Reals output**: displays **Σ**, matching the standard einsum/contraction
  interpretation.

The symbol is rendered as a centred LaTeX annotation inside a `40×30`
rectangle box.

---

## 9. Entry Point and Module Loading

The TypeScript entry point (`index.ts`) does:

1. Read JSON from a file or receive it over a WebSocket
2. Create an `HTMLRenderHandler` (manages the `<div>` in the browser)
3. Create a `BroadcastedRenderer` (holds settings and the draw handler)
4. Call `termPass(json_term)`:
   a. Wipe the renderer
   b. Deserialize the JSON into a `Broadcasted` expression tree
   c. Create a `MultilineComposedBox` (a sequence of composed diagrams, one
      per line)
   d. Call `post_placement()` to fix all positions
   e. Call `update()` to issue all draw commands

For all operator registrations and term registrations to be in place before
step 4b runs, `index.ts` imports `additionalOperationBoxes.ts`, which in
turn imports `TensorLogic.ts`. TypeScript module imports execute at load
time, so the `@register_term` and `@opsRegistry.registerClass` decorators
run before any JSON is processed.

---

## Summary

| Concern | Where it lives |
|---|---|
| Base term types: Bool, Reals, Natural, Weave, Broadcasted | `data_structure/BroadcastedCategory.ts` |
| Composition types: Composed, Block, Scan, ThreadedComposed | `data_structure/ProductCategory.ts` |
| Operator definitions: Einops, Linear, SoftMax, etc. | `data_structure/Operators.ts` |
| TensorEquation, Iverson expression terms | `data_structure/TensorLogic.ts` |
| Term serialization and registry | `Term.ts`, `@register_term` |
| Abstract drawing commands | `DrawHandler.ts` |
| Element tree and two-pass layout | `RenderHandler.ts` (DiagramElement) |
| Wire bundles and anchor linking | `CategoryRenderer.ts` (Meridian, Anchor) |
| Composition/block/scan box dispatch | `CategoryRenderer.ts` (display_category) |
| Operator-to-box dispatch | `BroadcastedCategoryRenderer.ts` (opsRegistry) |
| Datatype wire rendering (Bool/Natural) | `BroadcastedCategoryRenderer.ts` (ArrayMeridian) |
| Built-in operator boxes + TensorEquationBox | `additionalOperationBoxes.ts` |
| Multi-line diagram layout | `Multiline.ts` (MultilineComposedBox) |
| Browser render target | `HTMLRenderHandler.ts`, `HTMLDrawHandler.ts` |
