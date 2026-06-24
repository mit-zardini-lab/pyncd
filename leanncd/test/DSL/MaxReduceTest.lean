import LeanNCD.DSL.Elab
import LeanNCD.DSL.Syntax
import LeanNCD.DSL.Pipeline.Structural
import LeanNCD.Eval.Contract

namespace LeanNCD
open Lean Elab Std

-- ---------------------------------------------------------------------------
-- Parse tests
-- ---------------------------------------------------------------------------

-- maxreduce(A[i,k]) parses to agg = .max, nonlin = .identity
run_cmd do
  let r ← Command.liftTermElabM <| LeanNCD.elabTLRHS (← `(tl_rhs| maxreduce(A[i, k])))
  unless r.agg == AggOp.max do throwError s!"expected agg = .max, got {repr r.agg}"
  unless r.nonlin == Nonlin.identity do throwError s!"expected nonlin = .identity, got {repr r.nonlin}"
  unless r.body.terms.length == 1 do throwError "expected one product term"

-- bare sum_expr still has agg = .sum
run_cmd do
  let r ← Command.liftTermElabM <| LeanNCD.elabTLRHS (← `(tl_rhs| A[i, k]))
  unless r.agg == AggOp.sum do throwError s!"bare rhs should have agg = .sum, got {repr r.agg}"

-- ---------------------------------------------------------------------------
-- checkDtypes: predicate tensor with non-sum agg ⇒ predicateAgg
-- ---------------------------------------------------------------------------

run_cmd do
  let ax : AxisSpec := { name := "i", uid := 1, kind := .real none }
  let s : Stmt := .assign "P" [.free ax]
    { body := { terms := [] }, nonlin := .identity, agg := .max }
  let env : DeclEnv := ({} : HashMap String Decl).insert "P" (.predicate "P" [ax])
  let rp : ResolvedProgram := { decls := [], env, extNames := ∅, stmts := [s], extraStmts := #[] }
  match checkDtypes rp |>.run 0 with
  | .error (.predicateAgg "P") _ => pure ()
  | .error e _                   => throwError s!"wrong error: {repr e}"
  | .ok _ _                      => throwError "expected predicateAgg for max-agg on predicate output"

end LeanNCD

-- ---------------------------------------------------------------------------
-- Eval tests (in LeanNCD.Eval namespace for DenseTensor / inferAxisSizes)
-- ---------------------------------------------------------------------------

namespace LeanNCD.Eval
open Std

private def ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
private def tensorOf (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩
private def floatMax (a b : Float) : Float := Max.max a b
private def negInf : Float := -1.0 / 0.0

-- Basic max-pool: C[i] := maxreduce(A[i,k])
-- A = [[3,1,4],[1,5,9]] (2×3). max over k: [4, 9]
run_cmd do
  let i := ax "i" 1; let k := ax "k" 2
  let A := tensorOf [2, 3] [3, 1, 4, 1, 5, 9]
  let env : HashMap String DenseTensor := ({} : HashMap String DenseTensor).insert "A" A
  let rhs : RHSExpr := { body := { terms := [{ factors := [.read "A" [.axis i, .axis k]] }] },
                          nonlin := .identity, agg := .max }
  let stmt := Stmt.assign "C" [.free i] rhs
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError e
  | .ok (sizes, _) =>
    match evalAssignWith (· * ·) floatMax negInf env sizes "C" [.free i] rhs with
    | .error e => throwError e
    | .ok (_, C) =>
        unless DenseTensor.approxEq C (tensorOf [2] [4, 9]) do
          throwError s!"max-pool wrong: {repr C.data}"

-- All-negative inputs: unit0 = −∞ does not pollute — result is the true max (not 0).
-- A = [[-2, -5, -1]] (1×3). max = -1.
run_cmd do
  let i := ax "i" 1; let k := ax "k" 2
  let A := tensorOf [1, 3] [-2, -5, -1]
  let env : HashMap String DenseTensor := ({} : HashMap String DenseTensor).insert "A" A
  let rhs : RHSExpr := { body := { terms := [{ factors := [.read "A" [.axis i, .axis k]] }] },
                          nonlin := .identity, agg := .max }
  match inferAxisSizes {} env [Stmt.assign "C" [.free i] rhs] with
  | .error e => throwError e
  | .ok (sizes, _) =>
    match evalAssignWith (· * ·) floatMax negInf env sizes "C" [.free i] rhs with
    | .error e => throwError e
    | .ok (_, C) =>
        unless DenseTensor.approxEq C (tensorOf [1] [-1]) do
          throwError s!"all-negative max-pool wrong: {repr C.data}"

-- Multi-factor max: C[i] := maxreduce(A[i,k] · B[k])
-- A=[[1,2],[3,4]] (2×2), B=[2,1] (2).
-- C[0]=max(1*2, 2*1)=2, C[1]=max(3*2, 4*1)=6
run_cmd do
  let i := ax "i" 1; let k := ax "k" 2
  let A := tensorOf [2, 2] [1, 2, 3, 4]
  let B := tensorOf [2] [2, 1]
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "A" A).insert "B" B
  let rhs : RHSExpr :=
    { body := { terms := [{ factors := [.read "A" [.axis i, .axis k], .read "B" [.axis k]] }] },
      nonlin := .identity, agg := .max }
  match inferAxisSizes {} env [Stmt.assign "C" [.free i] rhs] with
  | .error e => throwError e
  | .ok (sizes, _) =>
    match evalAssignWith (· * ·) floatMax negInf env sizes "C" [.free i] rhs with
    | .error e => throwError e
    | .ok (_, C) =>
        unless DenseTensor.approxEq C (tensorOf [2] [2, 6]) do
          throwError s!"multi-factor max wrong: {repr C.data}"

end LeanNCD.Eval
