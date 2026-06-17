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

end LeanNCD.Eval
