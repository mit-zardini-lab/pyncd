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

/-! ## The `resolveDecls` phase

Builds the `DeclEnv` and classifies tensor names as external inputs vs internally produced.
Per the §12.1 contract this phase is purely constructive: §12.1 example programs READ names like
`W`, `X`, `Q`, `K` with no `tensor` declaration, so an undeclared read is an external input — not
an error. `resolveDecls` therefore NEVER throws. -/

/-- The declaration's tensor name. -/
def Decl.name : Decl → String
  | .tensor n _ => n | .predicate n _ => n | .linear n _ _ _ => n

/-- The tensor name a stmt writes to (its LHS). -/
def Stmt.lhsName : Stmt → String
  | .assign n _ _ => n | .scatter n _ _ _ => n

/-- The tensor names a stmt reads (from `.read` factors; iverson factors read nothing). -/
def Stmt.readNames : Stmt → List String
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.filterMap (fun
        | .read nm _ => some nm
        | .iverson _ => none))

/-- Build the declaration environment and classify external-input names.
    `extNames` = names READ in some stmt but never PRODUCED (never a stmt LHS).
    `extraStmts := #[]`: bias materialization for `linear … bias := true` is deferred — none of
    the five §12.1 examples declare a `linear` weight, so it is unexercised here. Never throws. -/
def resolveDecls (lp : LabeledProgram) : FreshM ResolvedProgram := do
  let env : DeclEnv := lp.decls.foldl (fun m d => m.insert d.name d) {}
  let produced : List String := lp.stmts.map Stmt.lhsName
  let reads    : List String := lp.stmts.flatMap Stmt.readNames
  let extNames : Finset String :=
    reads.foldl (fun s n => if produced.contains n then s else insert n s) ∅
  return { decls := lp.decls, stmts := lp.stmts, env,
           extNames, extraStmts := #[] }

end LeanNCD
