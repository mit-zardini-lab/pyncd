-- LeanNCD/DSL/Pipeline/Structural.lean
import LeanNCD.DSL.Pipeline.Types
import LeanNCD.DSL.Traverse
import LeanNCD.Exec.Uid
import Std.Data.HashMap

namespace LeanNCD
open Std

/-! ## Axis-name collectors

Axis identity in tensor logic is name-based within program scope (§12.1): a name appearing
in multiple places denotes the same axis. These collectors gather every source axis name in
program order; `TLProgram.axisNames` de-duplicates. Exhaustive structural recursion (Lean's
totality check forces every constructor) guarantees no axis name is silently dropped. -/

private def axNamesIdx : IdxExpr → List String
  | .axis a => [a.name] | .const _ => [] | .scale _ a => [a.name]
  | .shift a _ => [a.name] | .affine _ xs => xs.map (fun p => p.2.name)

private def axNamesPred : PredArith → List String
  | .embed e => axNamesIdx e | .mul a b => axNamesPred a ++ axNamesPred b | .iabs a => axNamesPred a

private def axNamesBool : BoolExpr → List String
  | .rel _ a b => axNamesPred a ++ axNamesPred b
  | .and a b => axNamesBool a ++ axNamesBool b | .or a b => axNamesBool a ++ axNamesBool b
  | .not a => axNamesBool a | .ieq a b => axNamesPred a ++ axNamesPred b

private def axNamesNonlin : Nonlin → List String
  | .softmax (some m) => axNamesBool m | .normalize (some m) => axNamesBool m | _ => []

private def axNamesFactor : Factor → List String
  | .read _ es => es.flatMap axNamesIdx | .iverson b => axNamesBool b

private def axNamesRHS (r : RHSExpr) : List String :=
  (r.body.terms.flatMap (fun t => t.factors.flatMap axNamesFactor)) ++ axNamesNonlin r.nonlin

private def axNamesLHS : LHSSlot → List String
  | .free a => [a.name] | .iterAt a _ => [a.name] | .iterNext a => [a.name] | .affine e => axNamesIdx e

private def axNamesDecl : Decl → List String
  | .tensor _ ax => ax.map (·.name) | .predicate _ ax => ax.map (·.name)
  | .linear _ i o _ => i.map (·.name) ++ o.map (·.name)

private def axNamesStmt : Stmt → List String
  | .assign _ ls r => ls.flatMap axNamesLHS ++ axNamesRHS r
  | .scatter _ ls r _ => ls.flatMap axNamesLHS ++ axNamesRHS r

/-- The ordered, de-duplicated list of axis names occurring anywhere in the program. -/
def TLProgram.axisNames (p : TLProgram) : List String :=
  (p.decls.flatMap axNamesDecl ++ p.stmts.flatMap axNamesStmt).eraseDups

/-! ## UID collectors (public — for tests and later phases) -/

private def uidsIdx : IdxExpr → List UID
  | .axis a => [a.uid] | .const _ => [] | .scale _ a => [a.uid]
  | .shift a _ => [a.uid] | .affine _ xs => xs.map (fun p => p.2.uid)

private def uidsPred : PredArith → List UID
  | .embed e => uidsIdx e | .mul a b => uidsPred a ++ uidsPred b | .iabs a => uidsPred a

private def uidsBool : BoolExpr → List UID
  | .rel _ a b => uidsPred a ++ uidsPred b
  | .and a b => uidsBool a ++ uidsBool b | .or a b => uidsBool a ++ uidsBool b
  | .not a => uidsBool a | .ieq a b => uidsPred a ++ uidsPred b

private def uidsNonlin : Nonlin → List UID
  | .softmax (some m) => uidsBool m | .normalize (some m) => uidsBool m | _ => []

private def uidsFactor : Factor → List UID
  | .read _ es => es.flatMap uidsIdx | .iverson b => uidsBool b

private def uidsRHS (r : RHSExpr) : List UID :=
  (r.body.terms.flatMap (fun t => t.factors.flatMap uidsFactor)) ++ uidsNonlin r.nonlin

private def uidsLHS : LHSSlot → List UID
  | .free a => [a.uid] | .iterAt a _ => [a.uid] | .iterNext a => [a.uid] | .affine e => uidsIdx e

/-- Every `AxisSpec.uid` reachable in a statement, in program order. -/
def Stmt.uids : Stmt → List UID
  | .assign _ ls r => ls.flatMap uidsLHS ++ uidsRHS r
  | .scatter _ ls r _ => ls.flatMap uidsLHS ++ uidsRHS r

/-! ## The `assignUIDs` phase -/

/-- Mint a fresh UID that is guaranteed non-zero. `0` is E1's "unassigned" sentinel
    (every emitted `AxisSpec.uid` starts at `0`); the §12 post-condition requires assigned
    UIDs to be distinguishable from it. `freshUData` is strictly increasing, so at most the
    first mint can be `0` — mint once more in that case. -/
private def freshNonZero : FreshM UID := do
  let d ← freshUData
  if d.uid == 0 then return (← freshUData).uid else return d.uid

/-- Mint one fresh non-zero UID per distinct axis name, then relabel every `AxisSpec.uid`
    by keying on the axis's source name. Equal names ⇒ equal UID; distinct names ⇒ distinct. -/
def assignUIDs (p : TLProgram) : FreshM LabeledProgram := do
  let mut memo : HashMap String UID := {}
  for nm in p.axisNames do
    let u ← freshNonZero
    memo := memo.insert nm u
  let relabel : UData → UData := fun u =>
    match u.name with
    | some nm => match memo[nm]? with | some v => { u with uid := v } | none => u
    | none    => u
  let p' := TermTraversable.traverseUID relabel p
  return { decls := p'.decls, stmts := p'.stmts }

end LeanNCD
