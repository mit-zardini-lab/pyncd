import LeanNCD.Eval.Tensor
import LeanNCD.DSL.Ast
namespace LeanNCD.Eval
open Std

/-- Evaluate an index expression at a coordinate assignment → an integer coordinate. -/
def evalIdx (coord : HashMap UID Int) : IdxExpr → Int
  | .axis a      => (coord[a.uid]?).getD 0
  | .const n     => n
  | .scale c a   => c * (coord[a.uid]?).getD 0
  | .shift a n   => (coord[a.uid]?).getD 0 + n
  | .affine n xs => xs.foldl (fun acc (c, a) => acc + c * (coord[a.uid]?).getD 0) n

/-- Evaluate predicate arithmetic (PredArith) at a coordinate → Int. -/
def evalPred (coord : HashMap UID Int) : PredArith → Int
  | .embed e => evalIdx coord e
  | .mul a b => evalPred coord a * evalPred coord b
  | .iabs a  => ((evalPred coord a).natAbs : Int)        -- |·| ; natAbs : Nat, cast back to Int

/-- Evaluate a Boolean/mask predicate at a coordinate → Bool.
    NB: `.ieq` is the modular/wrapping equality of the DSL; here we approximate it with
    structural Int equality (`==`), which coincides whenever no wraparound occurs. -/
def evalBool (coord : HashMap UID Int) : BoolExpr → Bool
  | .rel op a b =>
      let x := evalPred coord a
      let y := evalPred coord b
      match op with
      | .lt => x < y
      | .le => x ≤ y
      | .eq => x = y
      | .ne => x ≠ y
      | .ge => x ≥ y
      | .gt => x > y
  | .and a b => evalBool coord a && evalBool coord b
  | .or  a b => evalBool coord a || evalBool coord b
  | .not a   => ! evalBool coord a
  | .ieq a b => evalPred coord a == evalPred coord b

/-- Read a tensor at a coordinate: bounds-checked, zero-padded on out-of-range (the standard
    zero-pad). Shared by `gather`'s `.read` and `.unaryFn` cases — a `.unaryFn` factor reads
    identically to a plain `.read`, then applies its function to the resulting value. -/
def gatherRead (env : HashMap String DenseTensor) (coord : HashMap UID Int)
    (nm : String) (es : List IdxExpr) : Except EvalError Float :=
  match env[nm]? with
  | none => .error s!"gather: unknown tensor {nm}"
  | some t =>
      let srcZ : List Int := es.map (evalIdx coord)
      -- bounds check against t.shape; out-of-range ⇒ 0.0
      if (srcZ.zip t.shape).any (fun (z, d) => z < 0 || z ≥ (d : Int)) then
        .ok 0.0
      else
        .ok (t.get! (srcZ.map Int.toNat))

/-- Apply a unary transcendental function to a gathered value. `log`/`sqrt`/`recip` fail loud on
    a domain violation (consistent with this evaluator's fail-loud conventions elsewhere —
    unsized axes, unknown tensors, causality violations); `exp`/`sin`/`cos` are total, no domain
    check needed. `recip` (the friendly `/` operator's desugared form) fails on exactly zero
    rather than propagating `inf`. -/
def applyUnaryFn : UnaryOp → Float → Except EvalError Float
  | .log,  v => if v ≤ 0.0 then .error s!"log domain error: log({v}) undefined for non-positive input"
                else .ok (Float.log v)
  | .sqrt, v => if v < 0.0 then .error s!"sqrt domain error: sqrt({v}) undefined for negative input"
                else .ok (Float.sqrt v)
  | .exp,  v => .ok (Float.exp v)
  | .sin,  v => .ok (Float.sin v)
  | .cos,  v => .ok (Float.cos v)
  | .recip, v => if v == 0.0 then .error s!"div domain error: 1/{v} undefined for zero input"
                 else .ok (1.0 / v)

/-- Gather one factor's value at a coordinate, from the input tensors.
    `.read nm es`: map each `eᵢ` to its source coord via `evalIdx`; out-of-range (any coord < 0
    or ≥ dim) pads with 0.0 (the standard zero-pad). `.iverson b`: 1.0 if the predicate holds
    else 0.0. `.unaryFn op nm es`: reads exactly like `.read nm es` (via `gatherRead`), then
    applies `op` to the resulting value. -/
def gather (env : HashMap String DenseTensor) (coord : HashMap UID Int) : Factor → Except EvalError Float
  | .iverson b => .ok (if evalBool coord b then 1.0 else 0.0)
  | .read nm es => gatherRead env coord nm es
  | .unaryFn op nm es => (gatherRead env coord nm es).bind (applyUnaryFn op)

end LeanNCD.Eval
