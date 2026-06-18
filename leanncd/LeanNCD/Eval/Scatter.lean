import LeanNCD.Eval.Contract
namespace LeanNCD.Eval
open Std

/-- The index expression an LHS slot maps to (for `evalIdx`): the affine output coordinate. -/
def lhsSlotIdx : LHSSlot → IdxExpr
  | .affine e   => e
  | .free a     => .axis a
  | .iterAt _ n => .const n
  | .iterNext a => .shift a 1

/-- Source axes of a scatter: the axis-UIDs appearing in the affine LHS slots, unioned with the
    RHS read/mask axes; de-duplicated, in first-seen order. -/
def scatterSourceAxes (slots : List LHSSlot) (rhs : RHSExpr) : List UID :=
  (slots.flatMap (fun sl => idxAxisUIDs (lhsSlotIdx sl)) ++ readAxisUIDs rhs).eraseDups

/-- Evaluate a scatter stmt. `outShape` is the output dims (caller supplies, from the decl/inference).
    Each source coord → rhs value (∏ over factors, Σ over terms; no contraction axes beyond source)
    → written to the affine output coord. Output is initialised to `opts.fill`; with
    `opts.reduce = some "sum"` collisions accumulate, otherwise the last write wins (overwrite).
    Out-of-range output coordinates are skipped. -/
def evalScatter (env : HashMap String DenseTensor) (sizes : HashMap UID Nat)
    (nm : String) (slots : List LHSSlot) (rhs : RHSExpr) (opts : ScatterOpts) (outShape : List Nat) :
    Except EvalError (String × DenseTensor) := do
  -- up-front validation: every read name must be a known tensor
  for rn in (readNames rhs).eraseDups do
    if !(env.contains rn) then
      throw s!"evalScatter: unknown tensor {rn}"
  let srcAxes := scatterSourceAxes slots rhs
  let srcSizes := srcAxes.map (fun u => (sizes[u]?).getD 1)
  let mut out := DenseTensor.ofFn outShape (fun _ => Float.ofInt opts.fill)
  for sc in cartesian srcSizes do
    let coord : HashMap UID Int := (srcAxes.zip sc).foldl (fun m (u, v) => m.insert u (Int.ofNat v)) {}
    -- rhs value at this source coord: ∏ over a term's factors, Σ over terms (no extra contraction).
    let mut val := 0.0
    for t in rhs.body.terms do
      let mut prod := 1.0
      for f in t.factors do
        match gather env coord f with
        | .ok v    => prod := prod * v
        | .error _ => prod := prod * 0.0   -- legit out-of-range pad
      val := val + prod
    -- output coordinate = each slot's affine image at this source coord
    let outCoordZ : List Int := slots.map (fun sl => evalIdx coord (lhsSlotIdx sl))
    if (outCoordZ.zip outShape).all (fun (z, d) => 0 ≤ z && z < (d : Int)) then
      let oc := outCoordZ.map Int.toNat
      let prev := out.get! oc
      let new := if opts.reduce == some "sum" then prev + val else val
      out := out.set! oc new
  return (nm, out)

end LeanNCD.Eval
