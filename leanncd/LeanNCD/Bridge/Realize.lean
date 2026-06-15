-- LeanNCD/Bridge/Realize.lean
import LeanNCD.Base.Br          -- Axis, StObj, StMat, ArrayType, DType, Br
import LeanNCD.DSL.Target       -- AxisP, StMatP, StObjP
import LeanNCD.DSL.SizeExpr     -- SizeExpr.toNumeric

namespace LeanNCD

/-- Realize a presentation axis. Size via the §12.2 bridge `SizeExpr.toNumeric`. SORRY-FREE. -/
noncomputable def realizeAxis (a : AxisP) : Axis :=
  { name := a.name, size := a.size.toNumeric }

/-- SORRY-FREE. -/
noncomputable def realizeStObj (s : StObjP) : StObj := s.map realizeAxis

/-- Coerce an integer reindexing coefficient to `Numeric`. OBLIGATION (negative case):
    `Numeric = MvPolynomial String ℕ` has no additive inverses, so a negative look-back
    stride cannot be represented; the negative branch is `sorry` (feedback: `St` may need
    ℤ-coefficients — §2.2/§7.2). The non-negative branch is real. -/
noncomputable def intToNumeric (n : Int) : Numeric :=
  if 0 ≤ n then MvPolynomial.C (n.toNat : ℕ) else sorry

/-- A presentation stride map realized into `StMat dom cod`. The list dims must match
    `dom.length`/`cod.length`; mismatches are an OBLIGATION (the presentation erased the
    dependent length indices). Index the lists by `i.val`/`j.val` and default out-of-range
    to `0` — the realized matrix is correct WHEN the dims match (a `sorry`-ed side condition,
    not needed if you default-on-mismatch). -/
noncomputable def realizeStMat (m : StMatP) (dom cod : StObj) : StMat dom cod :=
  { coeffs := fun (i : Fin cod.length) (j : Fin dom.length) =>
      intToNumeric (((m.coeffs.getD i.val []).getD j.val 0))
    bias   := fun (i : Fin cod.length) => intToNumeric (m.bias.getD i.val 0) }

/-- SORRY-FREE. -/
noncomputable def realizeWeaveSlot : WeaveSlotP → WeaveSlot
  | .fixed a => .fixed (realizeAxis a)
  | .tiled   => .tiled

/-- SORRY-FREE. -/
noncomputable def realizeWeaveShape (w : WeaveShapeP) : WeaveShape := w.map realizeWeaveSlot

/-- One ArrayType per weave. OBLIGATION (dtype): `BrBaseP`/`AxisP` carry no `DType`, so
    `dtype := .reals` (predicate/Bool outputs not distinguished — feedback: `BrBaseP` should
    carry an op-semiring/dtype tag). The SHAPE is real (the weave's target axes). -/
noncomputable def weaveToArrayType (w : WeaveShapeP) : ArrayType :=
  { dtype := .reals, shape := (realizeWeaveShape w).targetAxes }

/-- Realize a presentation base op to a `BrBase` over derived objects. The objects are
    computed from the weaves (not given). The `inputWeaves`/`outputWeaves` functions are
    realized from the lists by `Fin`-indexing (defaulting out-of-range to `[]`). The
    dependent `reindexings : ∀ i, StMat degree (inputWeaves i).targetAxes` is realized
    by indexing `b.reindexings` and passing the field's index `(inputWeaves i).targetAxes`
    as the `cod` argument directly, so the dependent typing holds definitionally. -/
noncomputable def realizeBrBaseP (b : BrBaseP) : Σ (dom cod : BrObj), BrBase dom cod :=
  let dom : BrObj := b.inputWeaves.map weaveToArrayType
  let cod : BrObj := b.outputWeaves.map weaveToArrayType
  ⟨dom, cod, {
    op := b.op
    degree := realizeStObj b.degree
    inputWeaves  := fun i => realizeWeaveShape (b.inputWeaves.getD i.val [])
    outputWeaves := fun i => realizeWeaveShape (b.outputWeaves.getD i.val [])
    reindexings  := fun i =>
      realizeStMat (b.reindexings.getD i.val default) (realizeStObj b.degree)
        ((realizeWeaveShape (b.inputWeaves.getD i.val [])).targetAxes)
  }⟩

end LeanNCD
