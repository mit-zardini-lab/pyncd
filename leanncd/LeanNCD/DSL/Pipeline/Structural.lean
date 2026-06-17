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
  | .recurMorphism _ ax _ => [ax.name]

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
  | .recurMorphism _ ax _ => [ax.uid]

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
  | .assign n _ _ => n | .scatter n _ _ _ => n | .recurMorphism n _ _ => n

/-- The tensor names a stmt reads (from `.read` factors; iverson factors read nothing). -/
def Stmt.readNames : Stmt → List String
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.filterMap (fun
        | .read nm _ => some nm
        | .iverson _ => none))
  | .recurMorphism _ _ _ => []   -- recurMorphism reads not introspected (E2c)

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

/-! ## The `unifyAxes` phase (§7.4 UID coequalizer)

Groups every axis occurrence sharing a NAME within program scope, picks the largest UID as the
canonical representative (the §7.3 cocone vertex), and substitutes it throughout the program. In
the full pipeline `assignUIDs` already binds each name to one UID, so this is effectively identity
there; the standalone test feeds DISTINCT UIDs for one name to genuinely exercise the merge. -/

/-- Pair each axis name with its `AxisSpec.uid`. Mirrors `TLProgram.axisNames` exactly, but keeps
    the uid alongside the name. Exhaustive structural recursion (every constructor) ⇒ nothing
    dropped. Public: the test reads it back to assert post-unification UIDs. -/
private def axNameUIDIdx : IdxExpr → List (String × UID)
  | .axis a => [(a.name, a.uid)] | .const _ => [] | .scale _ a => [(a.name, a.uid)]
  | .shift a _ => [(a.name, a.uid)] | .affine _ xs => xs.map (fun p => (p.2.name, p.2.uid))

private def axNameUIDPred : PredArith → List (String × UID)
  | .embed e => axNameUIDIdx e | .mul a b => axNameUIDPred a ++ axNameUIDPred b
  | .iabs a => axNameUIDPred a

private def axNameUIDBool : BoolExpr → List (String × UID)
  | .rel _ a b => axNameUIDPred a ++ axNameUIDPred b
  | .and a b => axNameUIDBool a ++ axNameUIDBool b | .or a b => axNameUIDBool a ++ axNameUIDBool b
  | .not a => axNameUIDBool a | .ieq a b => axNameUIDPred a ++ axNameUIDPred b

private def axNameUIDNonlin : Nonlin → List (String × UID)
  | .softmax (some m) => axNameUIDBool m | .normalize (some m) => axNameUIDBool m | _ => []

private def axNameUIDFactor : Factor → List (String × UID)
  | .read _ es => es.flatMap axNameUIDIdx | .iverson b => axNameUIDBool b

private def axNameUIDRHS (r : RHSExpr) : List (String × UID) :=
  (r.body.terms.flatMap (fun t => t.factors.flatMap axNameUIDFactor)) ++ axNameUIDNonlin r.nonlin

private def axNameUIDLHS : LHSSlot → List (String × UID)
  | .free a => [(a.name, a.uid)] | .iterAt a _ => [(a.name, a.uid)]
  | .iterNext a => [(a.name, a.uid)] | .affine e => axNameUIDIdx e

private def axNameUIDDecl : Decl → List (String × UID)
  | .tensor _ ax => ax.map (fun a => (a.name, a.uid))
  | .predicate _ ax => ax.map (fun a => (a.name, a.uid))
  | .linear _ i o _ => i.map (fun a => (a.name, a.uid)) ++ o.map (fun a => (a.name, a.uid))

private def axNameUIDStmt : Stmt → List (String × UID)
  | .assign _ ls r => ls.flatMap axNameUIDLHS ++ axNameUIDRHS r
  | .scatter _ ls r _ => ls.flatMap axNameUIDLHS ++ axNameUIDRHS r
  | .recurMorphism _ ax _ => [(ax.name, ax.uid)]

/-- Every (axis-name, axis-uid) pair occurring anywhere in the program, in program order. -/
def collectAxisNameUID (p : TLProgram) : List (String × UID) :=
  p.decls.flatMap axNameUIDDecl ++ p.stmts.flatMap axNameUIDStmt

/-- Realize the §7.4 UID coequalizer: group UIDs by axis name, canonical = largest UID per name,
    substitute throughout. Pure inside; lifted via `return`. Identity when each name has one UID. -/
def unifyAxes (rp : ResolvedProgram) : FreshM CanonicalProgram := do
  let prog : TLProgram := { decls := rp.decls, stmts := rp.stmts }
  -- group UIDs by axis name
  let byName : HashMap String (Finset UID) :=
    (collectAxisNameUID prog).foldl
      (fun m (nm, u) => m.insert nm (insert u (m.getD nm ∅))) {}
  -- one EqClass per name; canonical = largest UID
  let ctx : Context AxisSpec :=
    byName.fold (fun c _ uids =>
      match uids.max with
      | some top => Context.merge c { bucket := uids, canonical := { data := default, uid := top } }
      | none     => c) { classes := [] }
  let prog' := Context.apply ctx prog        -- TermTraversable TLProgram
  return { decls := prog'.decls, stmts := prog'.stmts,
           env := rp.env, extNames := rp.extNames, ctx }

/-! ## The `lowerArith` phase (Phase 4 — affine index arithmetic)

E2a SCOPING DECISION (a deliberate divergence from §12.4): affine *reads* (e.g.
`X[i+p, 2*j+r]`) are LEFT IN PLACE here — the later `route` phase absorbs each read's affine
`IdxExpr` into the consuming step's `reindexings` field (exactly where stride-maps live in
`BrBase`, §2.3). So `lowerArith` emits NO separate Slice/Reindex intermediate steps and leaves
`auxStmts := #[]`. Its real job is the affine-LHS → `Stmt.scatter` reclassification plus a
*conservative* `overlappingScatter` injectivity guard (a const LHS coord collapses a dimension
and so needs `reduce sum`; strided coords like upsample `2*i` are injective). -/

/-- A LHS slot that denotes an affine output coordinate (a Scatter write). Plain `free`
    axes and the scan slots (`iterAt`/`iterNext`) are NOT affine-scatter slots. -/
def LHSSlot.isAffine : LHSSlot → Bool
  | .affine _   => true
  | .free _     => false
  | .iterAt _ _ => false
  | .iterNext _ => false

/-- Conservative non-injectivity test (E2a): a constant LHS coordinate collapses a
    dimension and so needs `reduce sum`; a strided coordinate (`scale`/general affine
    over an axis, e.g. upsample `2*i`) is injective. -/
def LHSSlot.collapses : LHSSlot → Bool
  | .affine (.const _) => true
  | _                  => false

def lowerArith (cp : CanonicalProgram) : FreshM LoweredProgram := do
  let stmts' ← cp.stmts.mapM (fun s => do
    match s with
    | .assign nm slots rhs =>
        if slots.any LHSSlot.isAffine then
          if slots.any LHSSlot.collapses then throw (CompileError.overlappingScatter nm)
          else return Stmt.scatter nm slots rhs { fill := 0, reduce := none }
        else return s
    | .scatter nm slots rhs opts =>
        if (slots.any LHSSlot.collapses) && opts.reduce ≠ some "sum" then
          throw (CompileError.overlappingScatter nm)
        else return s
    | .recurMorphism _ _ _ => return s)   -- no affine LHS; passes through unchanged
  return { decls := cp.decls, stmts := stmts', env := cp.env,
           extNames := cp.extNames, ctx := cp.ctx, auxStmts := #[] }

/-! ## The `finalizeScans` phase (Phase 5 — recurrence → Scan nodes)

A scan is a base case (`LHSSlot.iterAt`, the `l = 0` slot) plus a recurrence step
(`LHSSlot.iterNext`, the `l+1` slot). Statements for DIFFERENT tensor names that share the SAME
iteration-axis UID form a COUPLED scan (the §12.1 example: `G` and `H` both recur over `l`), so
they are grouped into ONE `ScanStmt.scan`. Validation: every recur step needs a matching base
step of the same name (else `missingBaseCase`); and a recur step may not read its iteration axis
"ahead" (`l+1` look-ahead on the RHS — else `causalityViolation`). -/

/-- The LHS slots of a statement. -/
def Stmt.slots : Stmt → List LHSSlot
  | .assign _ ls _ => ls | .scatter _ ls _ _ => ls | .recurMorphism _ _ _ => []

/-- The nonlinearity wrapping a stmt's step. A `recurMorphism` is pre-built (already-lowered),
    so it is affine-neutral (`identity`). Used by `finalizeScans` to detect ScanAffine (Prop 8.7):
    a scan whose every recurrence stmt is `identity`-nonlin carries no nonlinearity and is thus
    associative/parallel-prefix-able. This MUST be checked here (pre-`splitNonlins`), since
    `splitNonlins` later lifts nonlinearities out of `RHSExpr.nonlin` into separate steps. -/
def Stmt.nonlinOf : Stmt → Nonlin
  | .assign _ _ r => r.nonlin
  | .scatter _ _ r _ => r.nonlin
  | .recurMorphism _ _ _ => .identity

/-- `(iteration-axis uid, axis, isRecur)` if this stmt is a scan base/recur stmt. A stmt has
    at most one iteration slot, so the first match is the only one. -/
def Stmt.iterInfo (s : Stmt) : Option (UID × AxisSpec × Bool) :=
  s.slots.findSome? (fun
    | .iterAt a _ => some (a.uid, a, false)
    | .iterNext a => some (a.uid, a, true)
    | _           => none)

/-- All `IdxExpr`s read on the RHS of a stmt. -/
def Stmt.rhsReads : Stmt → List IdxExpr
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.flatMap (fun
        | .read _ es => es
        | .iverson _ => []))
  | .recurMorphism _ _ _ => []

/-- Conservative causality check: does any RHS read reference iteration axis `u` with a
    strictly-positive look-ahead offset (`shift a n`, `n > 0`)? -/
def readsIterAhead (s : Stmt) (u : UID) : Bool :=
  s.rhsReads.any (fun
    | .shift a n => a.uid == u && n > 0
    | _          => false)

/-- Rewrite a stmt's `iterAt` base-case slot to carry the iteration axis `ax`. The E1 parser
    cannot name the base-case iteration axis (it emits a placeholder `idxAxis ""`, so the slot's
    uid is the placeholder's, NOT the recurrence's), so the matching recurrence's axis is adopted
    here. A no-op on stmts with no `iterAt` slot, and (since `ax` is the recurrence's axis) on
    base cases already carrying the correct axis. -/
def Stmt.adoptBaseIterAxis (ax : AxisSpec) : Stmt → Stmt
  | .assign nm ls r    => .assign  nm (ls.map (fun | .iterAt _ n => .iterAt ax n | sl => sl)) r
  | .scatter nm ls r o => .scatter nm (ls.map (fun | .iterAt _ n => .iterAt ax n | sl => sl)) r o
  | s@(.recurMorphism _ _ _) => s

/-- Group `iterAt`/`iterNext` stmts by iteration-axis UID into (coupled) `ScanStmt.scan` nodes;
    pass everything else through as `ScanStmt.plain`. Validates base-case coverage and causality.

    PRE-PASS: each recurrence (`iterNext`) carries the real iteration axis (name + uid); each base
    case (`iterAt`) carries only the E1 placeholder axis (`idxAxis ""`). Before grouping, every base
    case adopts the iteration axis of a recurrence with the same tensor name, so base and recurrence
    land in the same UID group. -/
def finalizeScans (lp : LoweredProgram) : FreshM ScanProgram := do
  -- Recover each base case's iteration axis from the matching (same-name) recurrence.
  let recurAxisFor (nm : String) : Option AxisSpec :=
    lp.stmts.findSome? (fun s => match s.iterInfo with
      | some (_, a, true) => if s.lhsName == nm then some a else none
      | _                 => none)
  let stmts0 := lp.stmts.map (fun s =>
    match s.iterInfo with
    | some (_, _, false) => match recurAxisFor s.lhsName with
                            | some ax => s.adoptBaseIterAxis ax
                            | none    => s
    | _                  => s)
  let lp := { lp with stmts := stmts0 }
  -- recurMorphism stmts convert directly to `.scanPre` (NOT grouped with iterAt/iterNext).
  let preNodes : List ScanStmt := lp.stmts.filterMap (fun s => match s with
    | .recurMorphism nm ax tc => some (ScanStmt.scanPre nm ax tc)
    | _                       => none)
  let nonPre     := lp.stmts.filter (fun s => match s with | .recurMorphism _ _ _ => false | _ => true)
  let scanStmts  := nonPre.filter (fun s => s.iterInfo.isSome)
  let plainStmts := nonPre.filter (fun s => s.iterInfo.isNone)
  let uids := (scanStmts.filterMap (fun s => s.iterInfo.map (·.1))).eraseDups
  let mut nodes : List ScanStmt := []
  for u in uids do
    let group := scanStmts.filter (fun s => (s.iterInfo.map (fun t => t.1 == u)).getD false)
    let axis : AxisSpec := (group.findSome? (fun s => s.iterInfo.map (·.2.1))).getD default
    let baseStmts  := group.filter (fun s => (s.iterInfo.map (fun t => t.2.2 == false)).getD false)
    let recurStmts := group.filter (fun s => (s.iterInfo.map (fun t => t.2.2 == true)).getD false)
    for r in recurStmts do
      unless baseStmts.any (fun b => b.lhsName == r.lhsName) do
        throw (CompileError.missingBaseCase r.lhsName)
      if readsIterAhead r u then throw (CompileError.causalityViolation r.lhsName)
    let repName := (group.head?.map Stmt.lhsName).getD ""
    -- ScanAffine (Prop 8.7): the recurrence carries NO nonlinearity (every recur stmt is
    -- identity-nonlin) ⇒ associative/parallel-prefix-able. Empty `recur` ⇒ vacuously affine.
    let isAffine : Bool := recurStmts.all (fun s => Stmt.nonlinOf s == Nonlin.identity)
    nodes := nodes ++ [ ScanStmt.scan repName axis baseStmts recurStmts isAffine ]
  return { decls := lp.decls, stmts := plainStmts.map ScanStmt.plain ++ preNodes ++ nodes,
           env := lp.env, extNames := lp.extNames, ctx := lp.ctx }

end LeanNCD
