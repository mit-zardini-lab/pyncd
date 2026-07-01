-- LeanNCD/DSL/Target.lean
import LeanNCD.DSL.SizeExpr   -- SizeExpr
import Lean                    -- Lean.ToExpr

namespace LeanNCD

/-- Computable presentation of `StMat` (§2.2): an integer-affine coordinate map
    `dom → cod`. `coeffs` is `codLen × domLen` row-major; `bias` has length `codLen`.
    Integer (not `Numeric`) because DSL reindexings — Slice/Reindex/Scatter — are
    integer-affine (`IdxExpr` carries `Int`). Lengths are fields, not type indices. -/
structure StMatP where
  domLen : Nat
  codLen : Nat
  coeffs : List (List Int)
  bias   : List Int
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

/-- Computable presentation of `Axis` (§2.2): name + a `SizeExpr` (not `Numeric`). -/
structure AxisP where
  name : Option String
  size : SizeExpr
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

abbrev StObjP := List AxisP

/-- Computable presentation of `WeaveSlot` (§2.3). -/
inductive WeaveSlotP
  | fixed : AxisP → WeaveSlotP
  | tiled : WeaveSlotP
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

abbrev WeaveShapeP := List WeaveSlotP

/-- The concrete operation tag for a `BrBaseP` step.
    Replaces the previous `op : String` to give `route` a type-checked interface.
    `BrOp.toString` converts back to the string the Python execution layer dispatches on. -/
inductive BrOp
  | contract   -- standard tensor contraction (×, Σ, 0)
  | maxreduce  -- tropical max contraction (×, max, −∞)
  | scatter    -- affine-LHS scatter write
  | relu       -- relu nonlinearity
  | softmax    -- softmax (with optional mask)
  | normalize  -- normalize (with optional mask)
  | scan       -- recurrent scan (general)
  | scanAffine -- nonlinearity-free scan (Prop 8.7, O(log N) parallel prefix)
  | scanPre    -- scan step from recurMorphism escape hatch
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

def BrOp.toString : BrOp → String
  | .contract   => "contract"
  | .maxreduce  => "maxreduce"
  | .scatter    => "scatter"
  | .relu       => "relu"
  | .softmax    => "softmax"
  | .normalize  => "normalize"
  | .scan       => "scan"
  | .scanAffine => "scan_affine"
  | .scanPre    => "scan_pre"

/-- Computable presentation of `BrBase` (§2.3). The math-tower `Fin _ → WeaveShape`
    and `∀ i, StMat …` function fields become `List`s; the dependent
    `StMat degree (inputWeaves i).targetAxes` typing is dropped (an invariant the
    E2b bridge re-establishes). -/
structure BrBaseP where
  op           : BrOp
  degree       : StObjP
  inputWeaves  : List WeaveShapeP
  outputWeaves : List WeaveShapeP
  reindexings  : List StMatP
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

/-- A wire in the routed DAG (§12.4): either an external program input (`slot` = the external
    index) or output `slot` of producing step `step`. An inductive sum — not a struct with a
    sentinel `step = nExternal` — so the two cases stay unambiguous even when the external count
    is below the step count (a deep model has more steps than inputs, and a producer at index
    `nExternal` would otherwise be indistinguishable from external slot 0). -/
inductive Wire
  | external (slot : Nat)               -- an external program input
  | internal (step : Nat) (slot : Nat)  -- output `slot` of producing step `step`
  deriving DecidableEq, BEq, Repr, Lean.ToExpr, Inhabited

/-- Computable presentation of one `Br` morphism as a routed DAG (§12.4). The
    math-tower `routing : Fin steps.length → ℕ → Wire` function becomes
    `routing : List (List Wire)` — `routing[i]` are the input wires of `steps[i]`. -/
structure ThreadedComposed where
  steps     : List BrBaseP
  routing   : List (List Wire)
  nExternal : Nat
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

/-- Rank of a presentation weave = number of `fixed` (retained/target) axes; `tiled` slots are
    broadcast/contracted and carry no wire dimension. -/
def weaveRank (w : WeaveShapeP) : Nat :=
  w.countP fun s => match s with | .fixed _ => true | .tiled => false

/-- First port `(stepIdx, inputIdx)` consuming external slot `k`: the first `routing[i][j]` equal
    to `Wire.external k`. `none` if `k` is unreferenced — impossible for a pipeline-built `tc`
    (`extNames ⊆ reads`), and made an explicit requirement by `wellFormedDom`. -/
def ThreadedComposed.externalPort (tc : ThreadedComposed) (k : Nat) : Option (Nat × Nat) :=
  (List.range tc.routing.length).findSome? fun i =>
    let wires := tc.routing.getD i []
    (List.range wires.length).findSome? fun j =>
      match wires.getD j (.external 0) with
      | .external k' => if k' == k then some (i, j) else none
      | .internal .. => none

/-- §12.4 well-formedness guard for the `dom` reconstruction (`realize.md` §6d): every external
    slot `< nExternal` is referenced, and all ports consuming a slot agree on RANK (count of
    `fixed` axes). Bound-name differences across reads are allowed; unreferenced slots and genuine
    rank conflicts are rejected. Holds for any compiled `tc` (`checkReadRanks` pins external arity,
    `extNames ⊆ reads` pins referencedness); an explicit guard for hand-built `recurMorphism`
    graphs that bypass the pipeline. Validated by `route` so a compiled `tc` satisfies it. -/
def ThreadedComposed.wellFormedDom (tc : ThreadedComposed) : Bool :=
  (List.range tc.nExternal).all fun k =>
    match tc.externalPort k with
    | none          => false
    | some (i₀, j₀) =>
      let r₀ := weaveRank ((tc.steps.getD i₀ default).inputWeaves.getD j₀ [])
      (List.range tc.routing.length).all fun i =>
        let wires := tc.routing.getD i []
        (List.range wires.length).all fun j =>
          match wires.getD j (.external 0) with
          | .external k' => !(k' == k) || weaveRank ((tc.steps.getD i default).inputWeaves.getD j []) == r₀
          | .internal .. => true

/-- Propositional form of `wellFormedDom` (decidable via the `Bool`). -/
abbrev ThreadedComposed.WellFormedDom (tc : ThreadedComposed) : Prop := tc.wellFormedDom = true

end LeanNCD
