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

/-- A wire in the routed DAG (§12.4): output slot `slot` of step `step`
    (or, when `step = nExternal`, an external input). -/
structure Wire where
  step : Nat
  slot : Nat
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

/-- Computable presentation of one `Br` morphism as a routed DAG (§12.4). The
    math-tower `routing : Fin steps.length → ℕ → Wire` function becomes
    `routing : List (List Wire)` — `routing[i]` are the input wires of `steps[i]`. -/
structure ThreadedComposed where
  steps     : List BrBaseP
  routing   : List (List Wire)
  nExternal : Nat
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

end LeanNCD
