import LeanNCD.Eval.Shape
import LeanNCD.Eval.Gather
namespace LeanNCD.Eval
open Std

/-- All axis-UIDs referenced by an index expression. -/
def idxAxisUIDs : IdxExpr → List UID
  | .axis a      => [a.uid]
  | .const _     => []
  | .scale _ a   => [a.uid]
  | .shift a _   => [a.uid]
  | .affine _ xs => xs.map (·.2.uid)

/-- All axis-UIDs referenced by predicate arithmetic. -/
def predAxisUIDs : PredArith → List UID
  | .embed e => idxAxisUIDs e
  | .mul a b => predAxisUIDs a ++ predAxisUIDs b
  | .iabs a  => predAxisUIDs a

/-- All axis-UIDs referenced by a Boolean/mask predicate. -/
def boolAxisUIDs : BoolExpr → List UID
  | .rel _ a b => predAxisUIDs a ++ predAxisUIDs b
  | .and a b   => boolAxisUIDs a ++ boolAxisUIDs b
  | .or  a b   => boolAxisUIDs a ++ boolAxisUIDs b
  | .not a     => boolAxisUIDs a
  | .ieq a b   => predAxisUIDs a ++ predAxisUIDs b

/-- The free axes (LHS) of an assign, as UID list (affine slots contribute none). -/
def freeAxisUIDs (slots : List LHSSlot) : List UID := slots.filterMap lhsAxisUID?

/-- Every axis-UID appearing in the RHS reads/masks. -/
def readAxisUIDs (rhs : RHSExpr) : List UID :=
  rhs.body.terms.flatMap (fun t => t.factors.flatMap (fun
    | .read _ es => es.flatMap idxAxisUIDs
    | .iverson b => boolAxisUIDs b))

/-- Every `.read` tensor name appearing in the RHS. -/
def readNames (rhs : RHSExpr) : List String :=
  rhs.body.terms.flatMap (fun t => t.factors.filterMap (fun
    | .read nm _ => some nm
    | .iverson _ => none))

/-- The cartesian product of `[0..d-1]` ranges (one per dimension). Reuses `allCoords`. -/
def cartesian (dims : List Nat) : List (List Nat) := DenseTensor.allCoords dims

/-- Evaluate a `.plain` assign (identity nonlin) to its output tensor.
    `mul` combines factors within a product term; `combine` folds the per-(term, contracted-coord)
    contributions onto the accumulator starting from `unit0`. Default is the ℝ contraction `(*, +, 0)`.

    All `.read` tensor names are validated against `env` up front, so the inner accumulation
    (inside `ofFn`/`Id.run`) is pure: a `gather` error there can only be a legitimate out-of-range
    pad, which contributes `0.0`. A genuinely-missing input tensor is surfaced as `.error`. -/
def evalAssignWith (mul : Float → Float → Float) (combine : Float → Float → Float) (unit0 : Float)
    (env : HashMap String DenseTensor) (sizes : HashMap UID Nat)
    (nm : String) (slots : List LHSSlot) (rhs : RHSExpr) : Except EvalError (String × DenseTensor) := do
  -- up-front validation: every read name must be a known tensor
  for rn in (readNames rhs).eraseDups do
    if !(env.contains rn) then
      throw s!"evalAssign: unknown tensor {rn}"
  let frees := freeAxisUIDs slots
  -- fail loud: a free output axis with no inferred size would silently yield a 0-extent tensor.
  for u in frees do
    if (sizes[u]?).isNone then
      throw s!"evalAssign {nm}: output axis (uid {u}) has no inferable size (it appears in no read position)"
  let contr := (readAxisUIDs rhs).eraseDups.filter (fun u => ! frees.contains u)
  let outShape := outputShape sizes slots
  let contrSizes := contr.map (fun u => (sizes[u]?).getD 1)
  let out := DenseTensor.ofFn outShape (fun fcoord =>
    Id.run do
      let baseCoord : HashMap UID Int := (frees.zip fcoord).foldl (fun m (u, v) => m.insert u (Int.ofNat v)) {}
      let mut acc := unit0
      for t in rhs.body.terms do
        for cc in cartesian contrSizes do
          let coord := (contr.zip cc).foldl (fun m (u, v) => m.insert u (Int.ofNat v)) baseCoord
          let mut prod := 1.0
          for f in t.factors do
            match gather env coord f with
            | .ok v    => prod := mul prod v
            | .error _ => prod := mul prod 0.0   -- only reachable as a legit out-of-range pad
          acc := combine acc prod
      pure acc)
  return (nm, out)

/-- The default tensor (ℝ) contraction: multiply factors, then sum contributions. -/
def evalAssign := evalAssignWith (· * ·) (· + ·) 0.0

/-- The contraction "semiring" to use for an output, selected by its declared dtype. -/
structure Combine where
  mul     : Float → Float → Float
  combine : Float → Float → Float
  unit0   : Float

/-- The ℝ contraction `(×, Σ)`: multiply factors, then sum. -/
def Combine.real : Combine := ⟨(· * ·), (· + ·), 0.0⟩

/-- The Boolean contraction `(∧, ∃)` on 0/1 Floats: `min` factors (∧), `max` terms (∃). -/
def Combine.bool : Combine := ⟨min, max, 0.0⟩

/-- The tropical max contraction `(×, max, −∞)`: multiply factors within a term, then take max
    across terms and contracted axes. Identity is `−∞` so all-negative inputs reduce correctly. -/
def Combine.max : Combine := ⟨(· * ·), fun (a b : Float) => Max.max a b, -1.0 / 0.0⟩

/-- The declared name of a `Decl`. -/
def declName : Decl → String
  | .tensor n _      => n
  | .predicate n _   => n
  | .linear n _ _    => n
  | .axis ax _       => ax.name

/-- Pick the `Combine` for an output given its decl and the RHS aggregation op.
    Priority: `agg = .max` ⇒ tropical max; `predicate` ⇒ bool; else real. -/
def combineFor (decls : List Decl) (nm : String) (agg : AggOp) : Combine :=
  match agg with
  | .max => Combine.max
  | .sum => match decls.find? (fun d => declName d == nm) with
      | some (.predicate _ _) => Combine.bool
      | _                     => Combine.real

/-- dtype-aware assign: choose the `Combine` from the decls and `rhs.agg`, then evaluate.
    `agg = .max` ⇒ tropical `(×, max, −∞)`;
    `predicate` ⇒ Boolean `(∧, ∃)`;
    else ℝ `(×, Σ, 0)`. -/
def evalAssignDtyped (decls : List Decl)
    (env : HashMap String DenseTensor) (sizes : HashMap UID Nat)
    (nm : String) (slots : List LHSSlot) (rhs : RHSExpr) :
    Except EvalError (String × DenseTensor) :=
  let c := combineFor decls nm rhs.agg
  evalAssignWith c.mul c.combine c.unit0 env sizes nm slots rhs

/-- Like `evalAssign`, but with a `seed : HashMap UID Int` of axis-UIDs pinned to fixed values
    (e.g. the iteration axis of a scan, pinned to the current slice `l`). The seeded UIDs are
    excluded from BOTH the free (output) axes and the contracted axes; every per-output coord is
    seeded with these fixed values. Output shape/order follows the NON-seeded free slots.
    Reuses the ℝ contraction `(*, +, 0)`. -/
def evalAssignSeeded (env : HashMap String DenseTensor) (sizes : HashMap UID Nat)
    (seed : HashMap UID Int) (nm : String) (slots : List LHSSlot) (rhs : RHSExpr) :
    Except EvalError (String × DenseTensor) := do
  for rn in (readNames rhs).eraseDups do
    if !(env.contains rn) then
      throw s!"evalAssign: unknown tensor {rn}"
  -- free axes = LHS slot axes minus the seeded UIDs (and minus affine slots, which contribute none)
  let freesAll := freeAxisUIDs slots
  let frees := freesAll.filter (fun u => ! seed.contains u)
  let contr := (readAxisUIDs rhs).eraseDups.filter (fun u => ! frees.contains u && ! seed.contains u)
  -- output shape: each non-seeded free slot's size, in slot order
  let outShape := slots.filterMap (fun sl => match lhsAxisUID? sl with
    | some u => if seed.contains u then none else some ((sizes[u]?).getD 0)
    | none   => none)
  let contrSizes := contr.map (fun u => (sizes[u]?).getD 1)
  let out := DenseTensor.ofFn outShape (fun fcoord =>
    Id.run do
      let baseCoord : HashMap UID Int :=
        (frees.zip fcoord).foldl (fun m (u, v) => m.insert u (Int.ofNat v)) seed
      let mut acc := 0.0
      for t in rhs.body.terms do
        for cc in cartesian contrSizes do
          let coord := (contr.zip cc).foldl (fun m (u, v) => m.insert u (Int.ofNat v)) baseCoord
          let mut prod := 1.0
          for f in t.factors do
            match gather env coord f with
            | .ok v    => prod := prod * v
            | .error _ => prod := prod * 0.0
          acc := acc + prod
      pure acc)
  return (nm, out)

end LeanNCD.Eval
