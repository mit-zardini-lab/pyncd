import LeanNCD.Eval.Tensor
import LeanNCD.DSL.Ast
import Std.Data.HashMap

namespace LeanNCD.Eval
open Std

/-- The LHS slots of a statement (assign/scatter; `recurMorphism` has none here). -/
def Stmt.lhsSlots : Stmt → List LHSSlot
  | .assign _ ls _    => ls
  | .scatter _ ls _ _ => ls
  | .recurMorphism _ _ _ => []

/-- All `(name, [idxExprs])` reads in a stmt's RHS (assign/scatter; recurMorphism has none here). -/
def Stmt.readsOf : Stmt → List (String × List IdxExpr)
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.filterMap (fun
        | .read nm es => some (nm, es)
        | .iverson _  => none))
  | .recurMorphism _ _ _ => []

/-- Infer axis-UID → concrete size from the input tensors + bare-axis read positions.
    For a read `name[e₁,…,eₘ]` whose input `env[name]` has shape `[d₁,…,dₘ]`, every `eᵢ`
    that is a bare `.axis a` binds `a.uid ↦ dᵢ`. Conflicting sizes for one UID ⇒ error. -/
def inferAxisSizes (env : HashMap String DenseTensor)
    (stmts : List Stmt) : Except EvalError (HashMap UID Nat) := do
  let mut sizes : HashMap UID Nat := {}
  for s in stmts do
    for (nm, es) in Stmt.readsOf s do
      match env[nm]? with
      | none => pure ()    -- an internal/intermediate name not in inputs; skip (sized when produced)
      | some t =>
        for (e, d) in es.zip t.shape do
          match e with
          | .axis a =>
            match sizes[a.uid]? with
            | some d' => if d' != d then throw s!"axis size conflict for uid {a.uid}: {d'} vs {d}"
            | none    => sizes := sizes.insert a.uid d
          | _ => pure ()   -- affine read position doesn't pin a single axis here
  return sizes

/-- The axis UID of an LHS slot, if it has one (`affine` slots do not). -/
def lhsAxisUID? : LHSSlot → Option UID
  | .free a     => some a.uid
  | .iterAt a _ => some a.uid
  | .iterNext a => some a.uid
  | .affine _   => none

/-- The output shape: the size of each LHS slot's axis (free/iterAt/iterNext), in slot order.
    `affine` slots (scatter) are handled in Task 6 — `0` placeholder here. -/
def outputShape (sizes : HashMap UID Nat) (slots : List LHSSlot) : List Nat :=
  slots.map (fun sl => match lhsAxisUID? sl with
    | some u => (sizes[u]?).getD 0
    | none   => 0)

end LeanNCD.Eval
