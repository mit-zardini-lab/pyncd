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

/-- Coerce an integer reindexing coefficient to `Coeff = MvPolynomial String ℤ`. SORRY-FREE:
    `Coeff` is signed (a `CommRing`), so negative look-back offsets realize faithfully via
    `MvPolynomial.C` (this is the resolution of the former negative-coefficient obstruction —
    `StMat` now carries `Coeff`, not the ℕ-semiring `Numeric`; see §2.2/§7.2). -/
noncomputable def intToCoeff (n : Int) : Coeff := MvPolynomial.C n

/-- A presentation stride map realized into `StMat dom cod`. The list dims must match
    `dom.length`/`cod.length`; out-of-range indices default to `0`, so the realized matrix
    is total and correct WHEN the dims match (no `sorry`). -/
noncomputable def realizeStMat (m : StMatP) (dom cod : StObj) : StMat dom cod :=
  { coeffs := fun (i : Fin cod.length) (j : Fin dom.length) =>
      intToCoeff (((m.coeffs.getD i.val []).getD j.val 0))
    bias   := fun (i : Fin cod.length) => intToCoeff (m.bias.getD i.val 0) }

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

/-- Realize a routed-DAG presentation (`ThreadedComposed`, §12.4) into ONE `Br` morphism
    `BrMorph dom cod`.

    `dom`/`cod` are computed REALLY from the presentation:
    * `cod` (the composite's outputs) = the LAST step's `outputWeaves`, realized to
      `ArrayType`s. The final step is the DAG sink, so its outputs are the composite's
      outputs. (If `steps = []`, `getLast?` gives `none` and `cod = []`.)
    * `dom` (the composite's external inputs) = the FIRST step's `inputWeaves`, realized to
      `ArrayType`s. This is a DEFENSIBLE but PARTIAL derivation: the presentation routes
      external inputs via wires with `step = nExternal` (see `Wire`/`ThreadedComposed`),
      and in general the external `dom` is the concatenation of all input weaves across
      steps whose wire targets the sentinel — capped to `nExternal`. The presentation as
      given under-determines the exact external-input assembly (which slots are external
      vs. produced internally is encoded only in `routing`, not directly in the weaves),
      so we take the first step's inputs as the faithful representative external `dom`.
      DOCUMENTED OBLIGATION: a fully faithful `dom` must walk `routing`, collect the
      `step = nExternal` wires, and gather the corresponding input weaves in wire order.

    OBLIGATION (the `BrMorph dom cod` body, `sorry`): threading the per-step
    `realizeBrBaseP` morphisms along `routing` into a single `BrMorph dom cod` requires
    * `BrMorph.comp` to sequence steps (available, sorry-free), and
    * `Br.tensorHom` to place independent steps side-by-side and `Br.swap` to permute
      inputs/outputs per the wiring — BOTH are Milestone-B+ `sorry`s in `LeanNCD/Base/Br.lean`.
    Per the E2b scope decision we do NOT close `Br.tensorHom`/`Br.swap`, so the threaded
    morphism is necessarily `sorry`-dependent here; this `sorry` CITES that B+ dependence. -/
noncomputable def realize (tc : ThreadedComposed) : Σ (dom cod : BrObj), BrMorph dom cod :=
  let cod : BrObj := ((tc.steps.getLast?.map (fun b => b.outputWeaves)).getD []).map weaveToArrayType
  let dom : BrObj := ((tc.steps.head?.map (fun b => b.inputWeaves)).getD []).map weaveToArrayType
  ⟨dom, cod, sorry⟩  -- OBLIGATION: thread `realizeBrBaseP` steps along `routing` via
                     -- `BrMorph.comp` + `Br.tensorHom`/`Br.swap` (the latter two are B+ sorries).

end LeanNCD
