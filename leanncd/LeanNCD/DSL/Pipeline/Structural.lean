-- LeanNCD/DSL/Pipeline/Structural.lean
import LeanNCD.DSL.Pipeline.Types
import LeanNCD.DSL.Traverse
import LeanNCD.Exec.Uid
import Std.Data.HashMap

namespace LeanNCD
open Std

/-! ## Axis collectors

Axis identity in tensor logic is name-based within program scope (§12.1): a name appearing
in multiple places denotes the same axis. The `specs*` family gathers every source `AxisSpec`
in program order via structural recursion. Most of these (`specsIdx`, `specsPred`, `specsBool`,
`specsFactor`, `specsLHS`) are exhaustive matches, so Lean's totality check forces every new
constructor to be handled — nothing is silently dropped there. `specsNonlin` is the one
exception: it uses a wildcard fallback (`_ => []`), which is safe ONLY because every current
non-masked `Nonlin` variant genuinely contributes no axis specs. **Assumption that must hold for
any future `Nonlin` variant:** if it carries an optional mask (`Option BoolExpr`, like
`softmax`/`normalize`/`l2normalize`), it needs an explicit `some m => specsBool m` arm here —
otherwise the wildcard silently swallows that mask's axis specs (this bit `l2normalize` once
already; see the git history). The three public collectors below are thin projections of one
traversal: by name (`TLProgram.axisNames`, de-duplicated), by uid (`Stmt.uids`), or both
(`collectAxisNameUID`). -/

private def specsIdx : IdxExpr → List AxisSpec
  | .axis a => [a] | .const _ => [] | .scale _ a => [a]
  | .shift a _ => [a] | .affine _ xs => xs.map (·.2)

private def specsPred : PredArith → List AxisSpec
  | .embed e => specsIdx e | .mul a b => specsPred a ++ specsPred b | .iabs a => specsPred a

private def specsBool : BoolExpr → List AxisSpec
  | .rel _ a b => specsPred a ++ specsPred b
  | .and a b => specsBool a ++ specsBool b | .or a b => specsBool a ++ specsBool b
  | .not a => specsBool a | .ieq a b => specsPred a ++ specsPred b

private def specsNonlin : Nonlin → List AxisSpec
  | .softmax (some m) => specsBool m | .normalize (some m) => specsBool m
  | .l2normalize (some m) => specsBool m | _ => []

private def specsFactor : Factor → List AxisSpec
  | .read _ es => es.flatMap specsIdx | .iverson b => specsBool b
  | .unaryFn _ _ es => es.flatMap specsIdx

private def specsRHS (r : RHSExpr) : List AxisSpec :=
  (r.body.terms.flatMap (fun t => t.factors.flatMap specsFactor)) ++ specsNonlin r.nonlin

private def specsLHS : LHSSlot → List AxisSpec
  | .free a => [a] | .freeNorm a => [a]
  | .iterAt a _ => [a] | .iterNext a => [a] | .affine e => specsIdx e

private def specsDecl : Decl → List AxisSpec
  | .tensor _ ax => ax | .predicate _ ax => ax | .linear _ ax _ => ax
  | .axis ax _ => [ax]

private def specsStmt : Stmt → List AxisSpec
  | .assign _ ls r => ls.flatMap specsLHS ++ specsRHS r
  | .scatter _ ls r _ => ls.flatMap specsLHS ++ specsRHS r
  | .recurMorphism _ ax _ => [ax]

/-- Every `AxisSpec` occurring anywhere in the program, in program order (decls then stmts). -/
private def TLProgram.axisSpecs (p : TLProgram) : List AxisSpec :=
  p.decls.flatMap specsDecl ++ p.stmts.flatMap specsStmt

/-- The ordered, de-duplicated list of axis names occurring anywhere in the program. -/
def TLProgram.axisNames (p : TLProgram) : List String :=
  (p.axisSpecs.map (·.name)).eraseDups

/-! ## UID collectors (public — for tests and later phases) -/

/-- Every `AxisSpec.uid` reachable in a statement, in program order. -/
def Stmt.uids (s : Stmt) : List UID := (specsStmt s).map (·.uid)

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

/-- The declaration's tensor name. (`axis` decls name an AXIS, not a tensor; `resolveDecls`
    skips them when building the tensor-keyed `DeclEnv`.) -/
def Decl.name : Decl → String
  | .tensor n _ => n | .predicate n _ => n | .linear n _ _ => n | .axis ax _ => ax.name

/-- The tensor name a stmt writes to (its LHS). -/
def Stmt.lhsName : Stmt → String
  | .assign n _ _ => n | .scatter n _ _ _ => n | .recurMorphism n _ _ => n

/-- The tensor names a stmt reads (from `.read` factors; iverson factors read nothing). -/
def Stmt.readNames : Stmt → List String
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.filterMap (fun
        | .read nm _ => some nm
        | .iverson _ => none
        | .unaryFn _ nm _ => some nm))
  | .recurMorphism _ _ _ => []   -- recurMorphism reads not introspected (E2c)

/-- Build the declaration environment and classify external-input names.
    `extNames` = names READ in some stmt but never PRODUCED (never a stmt LHS). Never throws. -/
def resolveDecls (lp : LabeledProgram) : FreshM ResolvedProgram := do
  let env : DeclEnv := lp.decls.foldl (fun m d => match d with
    | .axis _ _ => m                  -- axis decls name an axis, not a tensor: keep them out of the env
    | _         => m.insert d.name d) {}
  let produced : List String := lp.stmts.map Stmt.lhsName
  let reads    : List String := lp.stmts.flatMap Stmt.readNames
  let extNames : Finset String :=
    reads.foldl (fun s n => if produced.contains n then s else insert n s) ∅
  return { decls := lp.decls, stmts := lp.stmts, env,
           extNames }

/-! ## The `checkReadRanks` phase

Validates that every `read nm idxExprs` in the program uses a number of index positions consistent
with `nm`'s declaration. Two cases:
- **Declared tensors** (`nm ∈ env`): `idxExprs.length` must equal the decl's axis count.
- **External tensors** (`nm ∈ extNames`): no declaration exists, so we check internal consistency —
  all reads of the same external name must agree on arity (first read wins as the expected rank).

`recurMorphism` stmts are invisible here (their `readsOf` returns `[]`), consistent with how the
rest of the pipeline treats that escape hatch. -/

private def Decl.axisCount : Decl → Nat
  | .tensor _ ax | .predicate _ ax | .linear _ ax _ => ax.length
  | .axis _ _ => 0   -- axis decls are excluded from DeclEnv; never reached via env lookup

private def stmtReads (s : Stmt) : List (String × Nat) :=
  match s with
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.filterMap (fun
        | .read nm es => some (nm, es.length)
        | .iverson _  => none
        | .unaryFn _ nm es => some (nm, es.length)))
  | .recurMorphism _ _ _ => []

/-- Will this LHS be lowered to a `scatter` (publishing its full slot-count rank)? True for an affine
    LHS (`Out[2*i,2*j]`) or a diagonal LHS with a repeated free axis (`Y[i,i]`). Shared by the
    read-rank guard (`stmtLhsRank`) and `lowerArith` so they agree on the published rank. -/
def slotsBecomeScatter (slots : List LHSSlot) : Bool :=
  slots.any (fun sl => match sl with | .affine _ => true | _ => false)
  || (let us := slots.filterMap (fun sl => match sl with | .free a => some a.uid | _ => none)
      us.length ≠ us.eraseDups.length)

/-- The produced (published) rank of a stmt's LHS — what a reader's arity must match. A LHS that
    becomes a `scatter` (`slotsBecomeScatter`: affine `Out[2*i,2*j]` ⇒ 2, or diagonal `Y[i,i]` ⇒ 2)
    publishes its full placement rank (`ls.length`); otherwise it publishes the dedup'd free-axis
    count (what `tensorAxes` emits). Used to arity-check reads of produced-but-undeclared
    intermediates. -/
private def stmtLhsRank (s : Stmt) : Nat :=
  match s with
  | .assign _ ls _ | .scatter _ ls _ _ =>
      if slotsBecomeScatter ls then ls.length
      else (ls.filterMap (fun
        | .free a | .freeNorm a | .iterNext a => some a.uid
        | .iterAt a _ => some a.uid
        | .affine _   => none)).eraseDups.length
  | .recurMorphism _ _ _ => 0

def checkReadRanks (rp : ResolvedProgram) : FreshM ResolvedProgram := do
  let reads : List (String × Nat) := rp.stmts.flatMap stmtReads
  -- declared tensors: check against DeclEnv
  for (nm, arity) in reads do
    if let some decl := rp.env[nm]? then
      let expected := decl.axisCount
      if arity != expected then throw (.rankMismatch nm expected arity)
  -- external tensors: check internal consistency (first read site establishes expected rank)
  let mut extRanks : HashMap String Nat := {}
  for (nm, arity) in reads do
    if nm ∈ rp.extNames then
      match extRanks[nm]? with
      | none   => extRanks := extRanks.insert nm arity
      | some r => if arity != r then throw (.rankMismatch nm r arity)
  -- produced-but-undeclared intermediates: no declaration justifies over-indexing, so a read whose
  -- arity ≠ the produced (dedup'd) rank is malformed (Track A #1). FAIL LOUD rather than route an
  -- ill-typed reindexing / silently broadcast at eval.
  let producedRank : HashMap String Nat :=
    rp.stmts.foldl (fun m s => m.insert s.lhsName (stmtLhsRank s)) {}
  for (nm, arity) in reads do
    unless rp.env.contains nm || decide (nm ∈ rp.extNames) do
      match producedRank[nm]? with
      | some r => if arity != r then throw (.rankMismatch nm r arity)
      | none   => pure ()
  return rp

/-! ## The `checkDtypes` phase

Two dtype invariants enforced after `checkReadRanks`:

**A — Axis-kind on LHS slots.**
- `iterAt`/`iterNext` slots must use a `nat`-kinded axis (scans iterate over discrete indices).
- `freeNorm` slots (the `m.`-marked softmax/normalize reduction axis) must use a `real`-kinded axis.

**B — Predicate outputs must have `identity` nonlinearity.**
A stmt writing to a `predicate`-declared tensor carries {0,1} values; applying relu/softmax/normalize
to such an output is a semantic error. Reading a predicate tensor on the RHS is intentionally valid
(the indicator-function pattern); only the output-nonlin combination is rejected.

`recurMorphism` stmts and `.affine`/`.free` slots are unconstrained and pass through. -/

private def isNat : AxisKind → Bool | .nat _ => true | _ => false
private def isReal : AxisKind → Bool | .real _ => true | _ => false

def checkDtypes (rp : ResolvedProgram) : FreshM ResolvedProgram := do
  for s in rp.stmts do
    -- Check A: axis kinds on LHS slots
    let slots : List LHSSlot := match s with
      | .assign _ ls _ | .scatter _ ls _ _ => ls
      | .recurMorphism _ _ _ => []
    for slot in slots do
      match slot with
      | .iterAt a _ | .iterNext a =>
          unless isNat a.kind do throw (.iterAxisNotNat a.name)
      | .freeNorm a =>
          unless isReal a.kind do throw (.normAxisNotReal a.name)
      | _ => pure ()
    -- Check B: predicate outputs must have identity nonlinearity and sum aggregation
    match s with
    | .assign nm _ rhs | .scatter nm _ rhs _ =>
        if let some (.predicate _ _) := rp.env[nm]? then
          unless rhs.nonlin == .identity do throw (.predicateNonlin nm)
          unless rhs.agg   == .sum       do throw (.predicateAgg nm)
    | .recurMorphism _ _ _ => pure ()
  return rp

/-! ## The `unifyAxes` phase (§7.4 UID coequalizer)

Groups every axis occurrence sharing a NAME within program scope, picks the largest UID as the
canonical representative (the §7.3 cocone vertex), and substitutes it throughout the program. In
the full pipeline `assignUIDs` already binds each name to one UID, so this is effectively identity
there; the standalone test feeds DISTINCT UIDs for one name to genuinely exercise the merge. -/

/-- Every (axis-name, axis-uid) pair occurring anywhere in the program, in program order.
    Mirrors `TLProgram.axisNames` exactly but keeps the uid alongside the name; the test reads
    it back to assert post-unification UIDs. -/
def collectAxisNameUID (p : TLProgram) : List (String × UID) :=
  p.axisSpecs.map (fun a => (a.name, a.uid))

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
`BrBase`, §2.3). So `lowerArith` emits NO separate Slice/Reindex intermediate steps. Its real job is the affine-LHS → `Stmt.scatter` reclassification plus a
*conservative* `overlappingScatter` injectivity guard (a const LHS coord collapses a dimension
and so needs `reduce sum`; strided coords like upsample `2*i` are injective). -/

/-- A LHS slot that denotes an affine output coordinate (a Scatter write). Plain `free`
    axes and the scan slots (`iterAt`/`iterNext`) are NOT affine-scatter slots. -/
def LHSSlot.isAffine : LHSSlot → Bool
  | .affine _   => true
  | .free _     => false
  | .freeNorm _ => false
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
        -- Reclassify affine (Out[2*i,2*j]) OR diagonal (Y[i,i], a repeated free axis) LHS to a
        -- scatter that publishes its full placement rank — so declared/full-rank reads of it are
        -- sound (B1). `collapses` only guards affine `.const` dimension-collapse; a diagonal is
        -- injective (i ↦ (i,i)) so it passes.
        if slotsBecomeScatter slots then
          if slots.any LHSSlot.collapses then throw (CompileError.overlappingScatter nm)
          else return Stmt.scatter nm slots rhs { fill := 0, reduce := none }
        else return s
    | .scatter nm slots _ opts =>
        if (slots.any LHSSlot.collapses) && opts.reduce ≠ some "sum" then
          throw (CompileError.overlappingScatter nm)
        else return s
    | .recurMorphism _ _ _ => return s)   -- no affine LHS; passes through unchanged
  return { decls := cp.decls, stmts := stmts', env := cp.env,
           extNames := cp.extNames, ctx := cp.ctx }

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

/-- All iteration slots of a stmt: `(uid, axis, isRecur, slot-position)` for each `iterAt`/`iterNext`.
    A 1-D scan yields a single-element list; multi-axis scans yield one entry per advancing slot. -/
def Stmt.iterInfo (s : Stmt) : List (UID × AxisSpec × Bool × Nat) :=
  s.slots.zipIdx.filterMap (fun (sl, i) => match sl with
    | .iterAt a _ => some (a.uid, a, false, i)
    | .iterNext a => some (a.uid, a, true, i)
    | _           => none)

/-- All `IdxExpr`s read on the RHS of a stmt. -/
def Stmt.rhsReads : Stmt → List IdxExpr
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.flatMap (fun
        | .read _ es => es
        | .iverson _ => []
        | .unaryFn _ _ es => es))
  | .recurMorphism _ _ _ => []

/-- Conservative causality check: does any RHS read reference iteration axis `u` with a
    strictly-positive look-ahead offset (`shift a n`, `n > 0`)? -/
def readsIterAhead (s : Stmt) (u : UID) : Bool :=
  s.rhsReads.any (fun
    | .shift a n => a.uid == u && n > 0
    | _          => false)

/-- Map slot-position → iteration axis, read from a step (`iterNext`) stmt. -/
def Stmt.stepAxisAt (step : Stmt) : Nat → Option AxisSpec := fun p =>
  step.iterInfo.findSome? (fun (_, a, isRec, i) => if isRec && i == p then some a else none)

/-- Rewrite a base stmt's `iterAt` slots so each adopts the step's iteration axis at the SAME
    slot position (the E1 parser leaves base iter-axes as placeholders — `idxAxis ""` — so the
    slot's uid is the placeholder's, NOT the recurrence's). A base `iterAt` whose position has no
    matching step `iterNext` axis is left as-is. Positional recovery generalises the old single-axis
    `adoptBaseIterAxis` to multi-axis scans (each advancing slot of the step names one base slot). -/
def Stmt.adoptBaseIterAxes (base step : Stmt) : Stmt :=
  let remap : List LHSSlot → List LHSSlot := fun ls =>
    ls.zipIdx.map (fun (sl, p) =>
      match sl with
      | .iterAt _ n =>
          match step.stepAxisAt p with
          | some a => .iterAt a n
          | none   => sl
      | _ => sl)
  match base with
  | .assign nm ls r    => .assign  nm (remap ls) r
  | .scatter nm ls r o => .scatter nm (remap ls) r o
  | .recurMorphism _ _ _ => base

/-- Group `iterAt`/`iterNext` stmts by iteration-axis UID into (coupled) `ScanStmt.scan` nodes;
    pass everything else through as `ScanStmt.plain`. Validates base-case coverage and causality.

    PRE-PASS: each recurrence (`iterNext`) carries the real iteration axis (name + uid); each base
    case (`iterAt`) carries only the E1 placeholder axis (`idxAxis ""`). Before grouping, every base
    case adopts the iteration axis of a recurrence with the same tensor name, so base and recurrence
    land in the same UID group. -/
def finalizeScans (lp : LoweredProgram) : FreshM ScanProgram := do
  -- Recover each base case's iteration axes from the matching (same-name) step, BY SLOT POSITION.
  -- A base stmt has `iterAt` slots (no `iterNext`); its matching step has `iterNext` slots. Each
  -- base `iterAt` adopts the step's `iterNext` axis at the same slot position (multi-axis general).
  let stepFor (nm : String) : Option Stmt :=
    lp.stmts.find? (fun s => s.lhsName == nm && s.iterInfo.any (fun t => t.2.2.1 == true))
  let stmts0 := lp.stmts.map (fun s =>
    if s.iterInfo.any (fun t => t.2.2.1 == false) && !s.iterInfo.any (fun t => t.2.2.1 == true) then
      match stepFor s.lhsName with
      | some step => s.adoptBaseIterAxes step
      | none      => s
    else s)
  let lp := { lp with stmts := stmts0 }
  -- recurMorphism stmts convert directly to `.scanPre` (NOT grouped with iterAt/iterNext).
  let preNodes : List ScanStmt := lp.stmts.filterMap (fun s => match s with
    | .recurMorphism nm ax tc => some (ScanStmt.scanPre nm ax tc)
    | _                       => none)
  let nonPre     := lp.stmts.filter (fun s => match s with | .recurMorphism _ _ _ => false | _ => true)
  let iterStmts  := nonPre.filter (fun s => !s.iterInfo.isEmpty)
  -- DEPENDENCY ANALYSIS (the per-step-intermediate fix). Map each produced tensor name to the set
  -- of scan iteration-axis UIDs it (transitively) depends on: seed every scan-state name with ALL
  -- of its own iteration axes, then propagate through reads to a fixpoint. A NON-iter stmt whose LHS
  -- name acquires a nonempty set is a per-step *intermediate* of that scan — e.g. the transformer's
  -- Q/K/V/S/…, recomputed from the layer state every step — and belongs in the recurrence body. A
  -- non-iter stmt with the empty set is genuinely loop-invariant and stays `.plain` (evaluated once,
  -- the original behaviour). Bounded fixpoint: `dep` grows monotonically, capped by #stmts passes.
  let mut dep : HashMap String (List UID) := {}
  for s in nonPre do
    unless s.iterInfo.isEmpty do
      dep := dep.insert s.lhsName ((dep.getD s.lhsName [] ++ s.iterInfo.map (·.1)).eraseDups)
  for _ in List.range (nonPre.length + 1) do
    let mut changed := false
    for s in nonPre do
      let cur := dep.getD s.lhsName []
      let merged := (cur ++ s.readNames.flatMap (fun r => dep.getD r [])).eraseDups
      if merged.length != cur.length then
        dep := dep.insert s.lhsName merged
        changed := true
    unless changed do break
  -- CONNECTED COMPONENTS over iteration-axis UIDs: two iter-stmts are coupled iff their axis-sets
  -- share a UID (the §12.1 `G`/`H` coupled scan, and each axis of a genuine multi-axis scan). One
  -- `ScanStmt.scan` per component. Bounded union-find: repeated merge, capped by #stmts passes.
  let axSet : Stmt → List UID := fun s => (s.iterInfo.map (·.1)).eraseDups
  -- The axis-UID of an LHS slot, if it has one (`.affine` slots contribute none) — used below to
  -- detect an unsupported in-scan per-step projection (§KG-scanprojection).
  let slotUID : LHSSlot → Option UID := fun sl => match sl with
    | .free a | .freeNorm a | .iterAt a _ | .iterNext a => some a.uid
    | .affine _ => none
  let mut comps : List (List UID) := iterStmts.map axSet
  for _ in List.range (iterStmts.length + 1) do
    comps := comps.foldl (fun acc c =>
      match acc.find? (fun d => d.any (fun u => c.contains u)) with
      | some d => (acc.erase d) ++ [(d ++ c).eraseDups]
      | none   => acc ++ [c]) []
  -- A non-iter intermediate whose scan-axis deps span more than ONE component is an unsupported
  -- cross-scan coupling (no §12.1 example needs it); fail loud. Within a single (possibly
  -- multi-axis) component it is a normal per-step intermediate.
  for s in nonPre do
    if s.iterInfo.isEmpty then
      let d := dep.getD s.lhsName []
      if !d.isEmpty && !comps.any (fun c => d.all (fun u => c.contains u)) then
        throw (CompileError.shapeMismatch
          s!"{s.lhsName}: per-step intermediate spans multiple scan components" "a single scan component")
  let mut nodes : List ScanStmt := []
  for comp in comps do
    -- membership tested against the ORIGINAL `nonPre` order so the recurrence body keeps source
    -- order (producers before consumers — exactly what `evalScan`'s step loop relies on).
    let inComp  : Stmt → Bool := fun s => (axSet s).any (fun u => comp.contains u)
    let isBase  : Stmt → Bool := fun s => inComp s && s.iterInfo.all (fun t => t.2.2.1 == false)
    let isState : Stmt → Bool := fun s => inComp s && s.iterInfo.any (fun t => t.2.2.1 == true)
    let isInter : Stmt → Bool := fun s => s.iterInfo.isEmpty &&
      (let d := dep.getD s.lhsName []; !d.isEmpty && d.all (fun u => comp.contains u))
    let baseStmts  := nonPre.filter (fun s => !s.iterInfo.isEmpty && isBase s)
    let stateRecur := nonPre.filter isState
    -- axis list in step slot order, from a representative step (each advancing slot ⇒ one axis).
    let axes : List AxisSpec := (stateRecur.head?.map (fun st =>
      ((st.iterInfo.filter (·.2.2.1)).mergeSort (fun a b => a.2.2.2 ≤ b.2.2.2)).map (·.2.1))).getD []
    -- FAIL LOUD (design §5): every state recurrence in a component MUST advance over the
    -- component's FULL axis set. A heterogeneous coupling — e.g. `H` advancing over `{c}` coupled
    -- (via shared `c`) with `G` advancing over `{r,c}` — would drop the non-head axes when `axes`
    -- is taken from the head alone, and `evalScan` would silently mis-address the shorter tensor.
    -- Compare axis-UID SETS (order-independent) against the component's unioned axis set `comp`.
    for r in stateRecur do
      let radv := ((r.iterInfo.filter (·.2.2.1)).map (·.1)).eraseDups
      unless radv.length == comp.length && comp.all (fun u => radv.contains u) do
        throw (CompileError.inconsistentScanAxes
          s!"{r.lhsName}: coupled scan statements advance over different axis sets (each must advance over the component's full axis set)")
    -- validation concerns only the genuine state recurrences: per-step intermediates have no base
    -- case and read the state at the current step, so neither check applies to them.
    for r in stateRecur do
      unless baseStmts.any (fun b => b.lhsName == r.lhsName) do
        throw (CompileError.missingBaseCase r.lhsName)
      for u in comp do
        if readsIterAhead r u then throw (CompileError.causalityViolation r.lhsName)
    -- FAIL LOUD (KG-scanprojection): an `isInter` statement (a per-step intermediate with no
    -- base case) whose OWN LHS references the component's iteration axis is ambiguous — it
    -- looks like the user wants a per-step read-out tracked across every `l`, but that's not
    -- materialized (only same-step scratch intermediates, which never reference `l` on their own
    -- LHS, are supported here). Reject rather than silently discard; the fully-general workaround
    -- is to write it as a separate top-level statement after the scan, reading the fully
    -- materialized state (see SS2 in the portfolio doc).
    for s in nonPre do
      if isInter s then
        if (s.slots.filterMap slotUID).any (fun u => comp.contains u) then
          throw (CompileError.scanProjectionUnsupported s.lhsName)
    -- recurrence body = per-step intermediates ++ state recurrences, in source order.
    let recurStmts := nonPre.filter (fun s => isInter s || isState s)
    let repName := ((recurStmts.head?.orElse (fun _ => baseStmts.head?)).map Stmt.lhsName).getD ""
    -- ScanAffine (Prop 8.7): affine only for a genuine 1-axis component (multi-axis ⇒ sequential)
    -- whose every recurrence stmt (intermediates included) is identity-nonlin.
    let isAffine : Bool := axes.length ≤ 1 && recurStmts.all (fun s => Stmt.nonlinOf s == Nonlin.identity)
    nodes := nodes ++ [ ScanStmt.scan repName axes baseStmts recurStmts isAffine ]
  -- plain = non-iter stmts with NO scan dependency (loop-invariant; evaluated once).
  let plainStmts := nonPre.filter (fun s => s.iterInfo.isEmpty && (dep.getD s.lhsName []).isEmpty)
  return { decls := lp.decls, stmts := plainStmts.map ScanStmt.plain ++ preNodes ++ nodes,
           env := lp.env, extNames := lp.extNames, ctx := lp.ctx }

end LeanNCD
