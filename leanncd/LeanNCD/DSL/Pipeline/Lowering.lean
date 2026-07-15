-- LeanNCD/DSL/Pipeline/Lowering.lean
-- Phases 6–8 of the tensor-logic DSL back-end. Phase 6 (`splitNonlins`) isolates each
-- nonlinearity (relu/softmax/normalize) into its own step so contraction steps and
-- nonlinearity steps don't mix. Phases 1–5 live in `Structural.lean`.
import LeanNCD.DSL.Pipeline.Structural
import LeanNCD.DSL.Target

namespace LeanNCD

/-! ## Phase 6 — `splitNonlins`

For each `Stmt` whose `RHSExpr.nonlin ≠ .identity`, split it into TWO stmts:
1. a LINEAR step computing the pre-activation into a fresh intermediate tensor
   (`nonlin := .identity`, original body, same LHS slots); and
2. a NONLIN step that READS that intermediate at the output's own coordinates and carries
   the original nonlinearity (and its mask).
A stmt already at `.identity` is emitted unchanged. -/

/-- Axis indices that index the output of a stmt (its free/scan slots). An `.affine` slot is a
    scatter output; the §12.1 examples never apply a nonlinearity over a scatter, so it is
    dropped here. -/
def LHSSlot.toReadIdx : LHSSlot → Option IdxExpr
  | .free a     => some (.axis a)
  | .freeNorm a => some (.axis a)
  | .iterAt a _ => some (.axis a)
  | .iterNext a => some (.axis a)
  | .affine _   => none      -- scatter outputs: skipped (no example needs nonlin-over-scatter)

/-- Split one stmt's nonlinearity into (≤2) stmts. Identity stmts and scatters pass through
    unchanged (scatters carry no nonlinearity in the §12.1 examples). -/
def splitStmt (s : Stmt) : FreshM (List Stmt) := do
  match s with
  | .assign nm slots rhs =>
      if rhs.nonlin == Nonlin.identity then return [s]
      else
        let d ← freshUData
        let interName := s!"%nl{d.uid}"
        let linStep : Stmt := .assign interName slots { body := rhs.body, nonlin := .identity }
        let readIdxs := slots.filterMap LHSSlot.toReadIdx
        let nlStep : Stmt := .assign nm slots
          { body := { terms := [ { factors := [ .read interName readIdxs ] } ] },
            nonlin := rhs.nonlin }
        return [linStep, nlStep]
  | .scatter .. => return [s]   -- scatters carry no nonlinearity in the §12.1 examples
  | .recurMorphism .. => return [s]   -- pre-built morphism: nothing to split

/-- Split nonlinearities within a `ScanStmt`. For `.scan`, split each stmt in `base`/`recur`
    and flatten back into the base/recur lists, keeping the node coupled. -/
def splitScan (sc : ScanStmt) : FreshM (List ScanStmt) := do
  match sc with
  | .plain s => (← splitStmt s).mapM (fun s' => pure (ScanStmt.plain s'))
  | .scan nm ax base recur isAff =>
      let base'  ← base.flatMapM splitStmt
      let recur' ← recur.flatMapM splitStmt
      return [ ScanStmt.scan nm ax base' recur' isAff ]
  | .scanPre nm ax tc => return [ ScanStmt.scanPre nm ax tc ]

/-- Isolate every relu/softmax/normalize into its own step across the whole program. -/
def splitNonlins (sp : ScanProgram) : FreshM LinearProgram := do
  let stmts' ← sp.stmts.flatMapM splitScan
  return { decls := sp.decls, stmts := stmts', env := sp.env,
           extNames := sp.extNames }

/-! ## Phase 7 — `schedule`

Ordering only; no statement pruning (KG-multiout: every produced name may be an intended
output, so the scheduler keeps all top-level statements and lets the caller ignore unwanted
keys). Statements are topologically sorted so producers precede their consumers.
Never-read *external* input names are still dropped (`liveExtNames`). -/

/-- Tensor names a ScanStmt writes (its LHS name(s)). -/
def ScanStmt.writes : ScanStmt → List String
  | .plain s        => [s.lhsName]
  | .scan _ _ b r _ => (b.map Stmt.lhsName ++ r.map Stmt.lhsName).eraseDups
  | .scanPre nm _ _ => [nm]

/-- The true output names of a ScanStmt: for `.scan`, names written by BOTH base AND recur
    (drops `%nl` intermediates that appear in only one side). -/
def ScanStmt.outputs : ScanStmt → List String
  | .plain s        => [s.lhsName]
  | .scan _ _ b r _ => (b.map Stmt.lhsName).filter (fun n => (r.map Stmt.lhsName).contains n)
  | .scanPre nm _ _ => [nm]

/-- Tensor names a ScanStmt reads. -/
def ScanStmt.reads : ScanStmt → List String
  | .plain s        => s.readNames
  | .scan _ _ b r _ => (b.flatMap Stmt.readNames ++ r.flatMap Stmt.readNames).eraseDups
  | .scanPre _ _ _  => []

/-! ### Topological sort (stable Kahn's algorithm)

`topoSortFuel` repeatedly emits the first currently-eligible statement — one whose
internal-read dependencies are all already emitted — scanning `remaining` in source order
on each pass to get a stable, no-op-on-already-sorted output.

An **internal dependency** is a read name that some stmt in the full list writes.
**Self-edges** (a stmt's reads ∩ its own writes) are excluded: scan nodes read their own
state names across iterations, which is not a forward dependency.

Termination: on each recursive call with `fuel > 0`, at least one stmt is emitted (or
`remaining` is already empty), so the fuel strictly decreases. Setting `fuel := stmts.length`
is sufficient. Using a `Nat` fuel parameter (rather than `partial`) means Lean accepts the
definition without a `decreasing_by` proof, and equation lemmas exist for Phase 4. -/

/-- Is `sc` eligible to emit given `emitted` names? Eligible iff every name it reads that
    is an internal dependency (written by some OTHER stmt in `all`, excluding self-edges)
    is already in `emitted`. -/
private def eligible (sc : ScanStmt) (all : List ScanStmt) (emitted : List String) : Bool :=
  let selfWrites := sc.writes
  sc.reads.all (fun r =>
    -- not an internal dep: either a self-edge or not written by any other stmt
    selfWrites.contains r ||
    all.all (fun s => s.writes == selfWrites || !s.writes.contains r) ||
    emitted.contains r)

/-- Fuel-bounded stable Kahn's sort. `all` is the full stmt list (fixed); `remaining` is
    what's left to emit; `emitted` accumulates written names of already-emitted stmts;
    `acc` accumulates emitted stmts in order. -/
def topoSortFuel : Nat → List ScanStmt → List ScanStmt → List String →
    List ScanStmt → List ScanStmt
  | 0,    _,   remaining, _,       acc => acc ++ remaining   -- fuel gone: append as-is
  | _,    _,   [],        _,       acc => acc
  | n+1,  all, remaining, emitted, acc =>
      match remaining.findIdx? (fun sc => eligible sc all emitted) with
      | none   => acc ++ remaining   -- cycle: shouldn't occur in valid DAGs
      | some i =>
          let sc   := remaining[i]!
          let rest := remaining.eraseIdx i
          topoSortFuel n all rest (emitted ++ sc.writes) (acc ++ [sc])

/-- Topological sort of a list of `ScanStmt`s: producers precede consumers.
    Stable — already-sorted input is returned unchanged (first-eligible-in-source-order). -/
def topoSort (stmts : List ScanStmt) : List ScanStmt :=
  topoSortFuel stmts.length stmts stmts [] []

/-- Is `ordered` a valid topological order of `all`? Walks `ordered` left to right, replaying
    the same `eligible`/`emitted` bookkeeping `topoSortFuel` uses: each stmt must be `eligible`
    given only the names emitted by stmts strictly before it. On a genuine DAG this always
    holds for `topoSort`'s output; on a cycle, `topoSortFuel`'s fallback (`acc ++ remaining`)
    appends the un-orderable remainder in source order, so the first such stmt fails
    `eligible` here (its internal-dependency read was never emitted before it). External reads
    (not written by any stmt in `all`) are not internal dependencies, so `eligible` already
    treats them as vacuously satisfied — they never cause a false cycle report. -/
def isTopoOrdered (all : List ScanStmt) (ordered : List ScanStmt) : Bool :=
  (ordered.foldl (fun (acc : Bool × List String) sc =>
      let (ok, emitted) := acc
      (ok && eligible sc all emitted, emitted ++ sc.writes))
    (true, ([] : List String))).1

def schedule (lp : LinearProgram) : FreshM ScheduledProgram := do
  let ordered   := topoSort lp.stmts
  -- Fail loud on cyclic dataflow (Spike 1h): topoSort cannot fully order a cycle, and its
  -- fallback silently returns source order. routeCore already rejects cycles (cyclicDataflow);
  -- schedule now does the same on the eval-only path instead of dying later with a generic
  -- "unknown tensor".
  if !isTopoOrdered lp.stmts ordered then
    throw (CompileError.cyclicDataflow "schedule: cyclic dataflow")
  -- collect the sizes pinned by `axis … = n` decls (UIDs are canonical by this phase).
  let explicitSizes : Std.HashMap UID Nat := lp.decls.foldl (fun m d => match d with
    | .axis ax (some n) => m.insert ax.uid n
    | _                 => m) {}
  let orderedReads := ordered.flatMap ScanStmt.reads
  let liveExtNames := lp.extNames.filter (fun nm => orderedReads.contains nm)
  return { decls := lp.decls, stmts := ordered, env := lp.env, extNames := liveExtNames,
           explicitSizes }

/-! ## Phase 8 — `route`

The executable back-end's final phase: turn the scheduled statements into a `ThreadedComposed`
— a routed DAG of `BrBaseP` steps. Build is two-pass:

1. PASS 1 (indexing). Assign each `ScanStmt` a step index `0,1,…`. Build `nameToStep`
   mapping every produced tensor name (for a `.scan` node, ALL its `writes` names) to its
   step index. Number the external read names `0,1,…` in first-seen order (over reads, NOT
   `Finset.toList`, which is noncomputable) into `extIndex`.

2. PASS 2 (build). For each step's representative stmt — for `.scan`, the first recurrence
   stmt, else the first base stmt (it carries the reads/axes):
   * LHS (retained) axes = the `AxisSpec`s named by `free`/`iterAt`/`iterNext` slots.
   * Read axes = every `AxisSpec` in the read-factor index expressions.
   * Contracted axes = read axes whose `uid` is NOT among the LHS axis uids.
   * `degree` = (LHS ++ contracted) de-duplicated by uid; each → `AxisP (some name) (var name)`
     (symbolic size — sizes aren't load-bearing in E2a). LHS axes first, then contracted.
   * `op` = `BrOp` constructor from `rhs.nonlin`/`rhs.agg`/stmt kind; scan nodes use scan variants.
   * `inputWeaves` = one shape per read factor; `outputWeaves` = one shape; each over `degree`,
     mapping contracted axes (by uid) to `.tiled`, retained axes to `.fixed a`.
   * `reindexings` = one `StMatP` per read factor; `idxToRow` expresses each read coordinate as
     an integer-affine combination of the degree axes (column order = degree order).
   * `routing[i]` = per read factor: `Wire.internal j 0` (output of producer step `j`) if
     internal, else `Wire.external (extIndex nm)`.

`nExternal := sp.extNames.card`. -/

/-- The read factors (tensor name + read index expressions) of a stmt, in order. -/
def Stmt.readFactors : Stmt → List (String × List IdxExpr)
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.filterMap (fun
        | .read nm es => some (nm, es)
        | .iverson _  => none
        | .unaryFn _ nm es => some (nm, es)))
  | .recurMorphism _ _ _ => []

/-- The retained-output `AxisSpec`s of a stmt: those named by `free`/`iterAt`/`iterNext` slots.
    `.affine` (scatter) slots carry no single retained axis and are skipped. -/
def Stmt.lhsAxes : Stmt → List AxisSpec
  | .assign _ ls _ | .scatter _ ls _ _ =>
      ls.filterMap (fun
        | .free a     => some a
        | .freeNorm a => some a
        | .iterAt a _ => some a
        | .iterNext a => some a
        | .affine _   => none)
  | .recurMorphism _ _ _ => []

/-- The `RHSExpr.nonlin` of a stmt. -/
def Stmt.nonlin : Stmt → Nonlin
  | .assign _ _ r | .scatter _ _ r _ => r.nonlin
  | .recurMorphism _ _ _ => .identity

/-- The `RHSExpr.agg` of a stmt. -/
def Stmt.agg : Stmt → AggOp
  | .assign _ _ r | .scatter _ _ r _ => r.agg
  | .recurMorphism _ _ _ => .sum

/-- Every `AxisSpec` appearing in a single read index expression. -/
def idxAxes : IdxExpr → List AxisSpec
  | .axis a     => [a]
  | .const _    => []
  | .scale _ a  => [a]
  | .shift a _  => [a]
  | .affine _ xs => xs.map (·.2)

/-- The empty-statement sentinel used as a `getD` default for a `ScanStmt` with no representative
    stmt (a `scanPre` node). Named so `buildStep`/`tensorAxes` callers share one definition. -/
def emptyStmt : Stmt := .assign "" [] { body := { terms := [] }, nonlin := .identity }

/-- Deduplicate axis specs by uid, keeping the first occurrence in order. Shared by `buildStep`'s
    `degAxes` and `tensorAxes` so producer (output weave) and consumer (input weave) derive a wire's
    axes the SAME way — making conjunct 2 hold even for repeated-LHS programs (e.g. `Y[i,i]`). -/
def dedupByUid (as : List AxisSpec) : List AxisSpec :=
  as.foldl (fun acc a => if acc.any (fun b => b.uid == a.uid) then acc else acc ++ [a]) []

/-- `dedupByUid` produces a uid-distinct list: the uids of its output are `Nodup`. The "canonical
    degree" invariant — a step's index space has exactly one column per distinct axis. -/
theorem dedupByUid_uid_nodup (as : List AxisSpec) : ((dedupByUid as).map (·.uid)).Nodup := by
  suffices h : ∀ (as acc : List AxisSpec), ((acc.map (·.uid)).Nodup) →
      (((as.foldl (fun acc a => if acc.any (fun b => b.uid == a.uid) then acc else acc ++ [a]) acc)).map
        (·.uid)).Nodup by
    exact h as [] (by simp)
  intro as
  induction as with
  | nil => intro acc hacc; simpa using hacc
  | cons a t ih =>
      intro acc hacc
      simp only [List.foldl_cons]
      apply ih
      cases hb : acc.any (fun b => b.uid == a.uid) with
      | true => simpa using hacc
      | false =>
          show ((acc ++ [a]).map (·.uid)).Nodup
          rw [List.map_append, List.nodup_append]
          refine ⟨hacc, List.nodup_singleton _, ?_⟩
          intro u hu u' hu'
          simp only [List.map_cons, List.map_nil, List.mem_singleton] at hu'
          subst hu'
          intro heq
          subst heq
          obtain ⟨b, hbmem, hbeq⟩ := List.mem_map.mp hu
          have hcontra : acc.any (fun b => b.uid == a.uid) = true :=
            List.any_eq_true.mpr ⟨b, hbmem, by simp [hbeq]⟩
          rw [hb] at hcontra
          exact absurd hcontra (by simp)

/-- A stmt's published (retained) output axes, in LHS order (deduplicated by uid) — what a producer
    emits and a consumer receives. -/
def tensorAxes (s : Stmt) : List AxisP :=
  (dedupByUid s.lhsAxes).map (fun a => AxisP.mk (some a.name) (SizeExpr.var a.name))

/-- One representative axis per READ POSITION, for external reads (no producer to publish a type). -/
def readPosAxis : IdxExpr → AxisP
  | .axis a      => AxisP.mk (some a.name) (SizeExpr.var a.name)
  | .shift a _   => AxisP.mk (some a.name) (SizeExpr.var a.name)
  | .scale _ a   => AxisP.mk (some a.name) (SizeExpr.var a.name)
  | .affine _ xs => match xs.head? with
                    | some (_, a) => AxisP.mk (some a.name) (SizeExpr.var a.name)
                    | none        => AxisP.mk none (SizeExpr.var "_")
  | .const _     => AxisP.mk none (SizeExpr.var "_")

/-- Express a read coordinate `IdxExpr` as an integer-affine combination of the degree axes
    identified by uids `us` (column order = `us`): returns `(coeff-row, bias)`. The dense view of the
    shared `idxAffineForm` primitive (M2 dedup, §6.2) — `coeffs = densify (affine coeffs) over us`,
    `bias = affine const`. -/
def idxToRow (us : List UID) (e : IdxExpr) : (List Int × Int) :=
  (idxDensify (idxAffineForm e).2 us, (idxAffineForm e).1)

/-- Every `idxToRow` coefficient row has exactly one entry per degree axis (`idxDensify` is a
    `us.map`), so its length is `us.length`. The load-bearing fact for `StMatP` well-formedness of
    reindexings (Track A): a built reindexing's `coeffs` rows are `domLen`-wide. -/
theorem idxToRow_fst_length (us : List UID) (e : IdxExpr) : (idxToRow us e).1.length = us.length := by
  simp [idxToRow, idxDensify]

/-- The reindexing `StMatP` built from a degree `us` and a read's index expressions `idxs` is
    `wellFormed`: `coeffs` is `idxs.length × us.length` and `bias` has length `idxs.length`. This is
    exactly the record `buildStep` constructs (with `us = degUids`, `idxs = rf.2`). -/
theorem reindexing_wellFormed (us : List UID) (idxs : List IdxExpr) :
    (StMatP.mk us.length idxs.length ((idxs.map (idxToRow us)).map (·.1))
      ((idxs.map (idxToRow us)).map (·.2))).wellFormed := by
  simp only [StMatP.wellFormed, List.length_map, beq_self_eq_true, Bool.true_and, Bool.and_true,
    List.all_map, List.all_eq_true, Function.comp_apply]
  intro e _
  simp [idxToRow_fst_length]

/-- A ScanStmt's representative stmt: for `.scan`, the first recurrence stmt (else the first
    base stmt); for `.plain`, the stmt itself. It carries the reads/axes used to build the step. -/
def ScanStmt.repStmt : ScanStmt → Option Stmt
  | .plain s        => some s
  | .scan _ _ b r _ => r.head?.orElse (fun _ => b.head?)
  | .scanPre _ _ _  => none

/-- Is this ScanStmt a `.scan` node? (drives the "scan" op label). -/
def ScanStmt.isScan : ScanStmt → Bool
  | .plain _      => false
  | .scan ..      => true
  | .scanPre _ _ _ => true

/-- Is this ScanStmt a `.scanPre` (recurMorphism) node? (drives the "scan_pre" op label). -/
def ScanStmt.isScanPre : ScanStmt → Bool
  | .scanPre _ _ _ => true
  | _              => false

/-- Is this a `.scan` node flagged affine by `finalizeScans` (Prop 8.7)? Drives the
    "scan_affine" vs "scan" op label. The flag was computed pre-`splitNonlins`. -/
def ScanStmt.isAffineScan : ScanStmt → Bool
  | .scan _ _ _ _ isAff => isAff
  | _                   => false

/-- PASS-1 helper: assign external-name indices `0,1,…` in first-seen order over reads.
    Iterates `stmts` in order, then each stmt's `ScanStmt.reads` in order; assigns the next
    integer the first time a name that `∈ extNames` is seen. -/
def buildExtIndex (extNames : Finset String) (stmts : List ScanStmt)
    : Std.HashMap String Nat :=
  (stmts.foldl (fun (acc : Std.HashMap String Nat × Nat) sc =>
    sc.reads.foldl (fun (m, cnt) nm =>
      if decide (nm ∈ extNames) && !m.contains nm then
        (m.insert nm cnt, cnt + 1)
      else (m, cnt)) acc) ({}, 0)).1

/-- The stmts that contribute reads/axes to a ScanStmt's step. For `.scan`, the base (initial-state)
    stmts followed by the recurrence stmts; for `.plain`, the stmt itself; `.scanPre` carries none
    (its morphism is pre-built). -/
def ScanStmt.stepStmts : ScanStmt → List Stmt
  | .plain s        => [s]
  | .scan _ _ b r _ => b ++ r
  | .scanPre _ _ _  => []

/-- The read factors that become this step's input wires. For `.scan`, ALL base+recur reads with
    **self-reads excluded** (a read whose name the step itself writes is the recurrence, internal to
    the scan generator — not a routing wire); the surviving reads are the initial states (`base`) and
    per-step inputs (`recur`). For `.plain`, the stmt's reads verbatim (a self-read there is a genuine
    cycle, left in so the acyclicity guard rejects it). `.scanPre` has none. -/
def ScanStmt.inputReadFactors (sc : ScanStmt) : List (String × List IdxExpr) :=
  match sc with
  | .plain s        => s.readFactors
  | .scan _ _ b r _ =>
      let ws := sc.writes
      (b.flatMap Stmt.readFactors ++ r.flatMap Stmt.readFactors).filter (fun rf => !ws.contains rf.1)
  | .scanPre _ _ _  => []

/-- The defining stmt of output slot `s` (= `writes[s]`): the LAST stmt in `stepStmts` that writes it
    (for a `.scan`, the recurrence stmt — full output rank, incl. the iteration axis). Used so the
    producer's per-slot output weave and a consumer's input weave derive the SAME axes (conjunct 2). -/
def ScanStmt.slotStmt (sc : ScanStmt) (s : Nat) : Stmt :=
  let nm := sc.outputs.getD s ""
  (sc.stepStmts.filter (fun st => st.lhsName == nm)).getLast?.getD emptyStmt

/-- The retained (LHS/output) axes of a step, in last-writer/slot order: the per-slot defining stmts'
    LHS axes concatenated in `outputs` order, deduplicated by uid. Factored out of `stepDegAxesMulti`
    so the acyclicity/consistency guard and the degree share one source of truth. -/
def ScanStmt.stepRetainedAxes (sc : ScanStmt) : List AxisSpec :=
  dedupByUid ((List.range sc.outputs.length).flatMap (fun s => (sc.slotStmt s).lhsAxes))

/-- A step's combined index space across all `stepStmts`: retained (LHS) axes of every contributing
    stmt, then the contracted (read-but-not-retained) axes, deduplicated by uid. Generalizes
    the per-stmt degree computation to the `.scan` group. -/
def ScanStmt.stepDegAxesMulti (sc : ScanStmt) : List AxisSpec :=
  let ss := sc.stepStmts
  let retained := sc.stepRetainedAxes
  let allRead := ss.flatMap (fun s => s.readFactors.flatMap (fun rf => rf.2.flatMap idxAxes))
  let contracted := allRead.filter (fun a => !(retained.map (·.uid)).contains a.uid)
  dedupByUid (retained ++ contracted)

/-- Canonical degree (Track A): a step's index space (`stepDegAxesMulti`) has distinct uids — one
    column per axis, no duplicates. Immediate from `dedupByUid_uid_nodup` since the degree ends in a
    `dedupByUid`. This is what makes `degUids` a valid column index set for the reindexings. -/
theorem ScanStmt.stepDegAxesMulti_uid_nodup (sc : ScanStmt) :
    ((sc.stepDegAxesMulti).map (·.uid)).Nodup := by
  unfold ScanStmt.stepDegAxesMulti
  exact dedupByUid_uid_nodup _

/-- M2 (`elaborateAffineReindexings`): the affine reindexing artifact for a step, lifted out of
    `buildStep` as the single source of truth. One `StMatP` per input read factor: columns indexed by
    the canonical step degree (`stepDegAxesMulti` uids, `stepDegAxesMulti_uid_nodup`), rows the read's
    `idxToRow` coordinates. `buildStep` produces exactly this (`buildStep_reindexings`, the bridge);
    `route` is unchanged for now (consuming it is M4). `IdxExpr` is affine by construction, so there
    is no non-affine case to reject. -/
def ScanStmt.elaborateReindexings (sc : ScanStmt) : List StMatP :=
  let degUids := sc.stepDegAxesMulti.map (·.uid)
  sc.inputReadFactors.map (fun rf =>
    let rows := rf.2.map (idxToRow degUids)
    StMatP.mk degUids.length rf.2.length (rows.map (·.1)) (rows.map (·.2)))

/-- The M2 artifact is well-formed: every elaborated reindexing has `coeffs` `codLen × domLen` and
    `bias` length `codLen` (from `reindexing_wellFormed`). A property of the artifact itself,
    independent of `route`. -/
theorem ScanStmt.elaborateReindexings_wellFormed (sc : ScanStmt) :
    ∀ m ∈ sc.elaborateReindexings, m.wellFormed := by
  unfold ScanStmt.elaborateReindexings
  intro m hm
  simp only [List.mem_map] at hm
  obtain ⟨rf, _, rfl⟩ := hm
  exact reindexing_wellFormed _ _

/-- Every elaborated reindexing's domain rank equals the canonical degree length (its column count is
    the number of distinct degree axes). A property of the artifact itself. -/
theorem ScanStmt.elaborateReindexings_domLen (sc : ScanStmt) :
    ∀ m ∈ sc.elaborateReindexings, m.domLen = (sc.stepDegAxesMulti.map (·.uid)).length := by
  unfold ScanStmt.elaborateReindexings
  intro m hm
  simp only [List.mem_map] at hm
  obtain ⟨rf, _, rfl⟩ := hm
  rfl

/-- Read-arity well-formedness (Track A #1): every INTERNAL read `rf` (routed to producer step `j`,
    slot `slot`) provides as many index positions as the producer publishes axes. This is an UPSTREAM
    property — `checkReadRanks` pins read arity to the declaration, and the producer's published rank
    (`tensorAxes (slotStmt slot)`) is that declaration's axis count — NOT a routing invariant, so it
    is carried as an explicit hypothesis (cf. the acset `WellShaped` and `topo_bound`'s `hrc`). Its
    full derivation from `checkReadRanks`/`env` is a separate effort (edge cases: repeated LHS axes,
    undeclared intermediates). -/
def ScanStmt.readArityOk (ns : Std.HashMap String (Nat × Nat)) (stmts : List ScanStmt)
    (sc : ScanStmt) : Prop :=
  ∀ rf ∈ sc.inputReadFactors, ∀ j slot, ns[rf.1]? = some (j, slot) →
    rf.2.length = (tensorAxes ((stmts.getD j default).slotStmt slot)).length

/-- The output weave of slot `s` over the step's combined `degree`: fixed on `writes[s]`'s retained
    (LHS) axes, tiled elsewhere. -/
def ScanStmt.slotWeave (sc : ScanStmt) (s : Nat) : WeaveShapeP :=
  let retainedUids := (dedupByUid (sc.slotStmt s).lhsAxes).map (·.uid)
  sc.stepDegAxesMulti.map (fun a =>
    if retainedUids.contains a.uid then WeaveSlotP.fixed (AxisP.mk (some a.name) (SizeExpr.var a.name))
    else WeaveSlotP.tiled)

/-- Consistency guard for coupled scans: every output slot's own (dedup'd) LHS axes must equal the
    step's `stepRetainedAxes` restricted to that slot's uids. Since an output weave masks the SHARED
    step degree (which follows `stepRetainedAxes` order), this is exactly the condition under which
    slot `s`'s published axes come out in `slotStmt s`'s own LHS order. Trivially true for plain and
    single-output scans (one `slotStmt` ⇒ retained = its LHS); can only fail for a coupled scan whose
    outputs disagree on a shared axis's order (e.g. `G[j,l]` vs `H[l,j]`) — invalid input the
    frontend should not emit, which `buildStep` rejects (FAIL LOUD) via `inconsistentScanAxes`. -/
def ScanStmt.outputAxesConsistent (sc : ScanStmt) : Bool :=
  (List.range sc.outputs.length).all (fun s =>
    let Ls := dedupByUid (sc.slotStmt s).lhsAxes
    decide (sc.stepDegAxesMulti.filter (fun a => (Ls.map (·.uid)).contains a.uid) = Ls))

/-- The buildStep well-formedness guard: a step must have at least one true output (nonempty
    `base∩recur` for scans — rejects a degenerate base-only scan) AND its outputs must agree on
    shared-axis order (`outputAxesConsistent`). Both are FAIL-LOUD on invalid input the frontend
    should not emit; on real §12.1 programs both hold. -/
def ScanStmt.stepGuardOk (sc : ScanStmt) : Bool :=
  !sc.outputs.isEmpty && sc.outputAxesConsistent

/-- PASS-1: map each produced tensor name to its (step index, output slot). A step writing several
    names (a coupled scan) assigns slot `0,1,…` in `writes` order. -/
def buildNameToStep (stmts : List ScanStmt) : Std.HashMap String (Nat × Nat) :=
  stmts.zipIdx.foldl (fun m (sc, i) =>
    sc.outputs.zipIdx.foldl (fun m' (nm, s) => m'.insert nm (i, s)) m) {}

/-- Build one `BrBaseP` step and its routing wires for `sc`, given the precomputed maps and the
    full scheduled statement list (`stmts`, used to publish an internal read's producer axes).
    Returns `CompileError.shapeMismatch` for an empty-step `scanPre`, or
    `CompileError.undeclaredName` for an unresolved read. -/
def buildStep (nameToStep : Std.HashMap String (Nat × Nat)) (extIndex : Std.HashMap String Nat)
    (stmts : List ScanStmt)
    (sc : ScanStmt) : Except CompileError (BrBaseP × List Wire) := do
  -- Validate a pre-built (escape-hatch) morphism: its step list must be non-empty.
  match sc with
  | .scanPre nm _ tc =>
      if tc.steps.isEmpty then
        throw (CompileError.shapeMismatch s!"recurMorphism {nm}: empty step morphism" "non-empty ThreadedComposed")
  | _ => pure ()
  -- Step guard: at least one true output, and coupled-scan outputs agree on shared-axis order
  -- (else the shared step degree cannot publish every slot in its own LHS order). FAIL LOUD.
  if ! sc.stepGuardOk then
    throw (if sc.outputs.isEmpty
      then CompileError.emptyScanOutputs "buildStep: scan step has no true outputs (empty base∩recur)"
      else CompileError.inconsistentScanAxes "buildStep: coupled scan outputs disagree on shared axis order")
  -- rep stmt drives ONLY the op label (nonlin/agg/kind); reads/axes come from the whole group.
  let s := sc.repStmt.getD emptyStmt
  let readFactors := sc.inputReadFactors
  -- degree = combined retained ++ contracted across all stepStmts, de-duplicated by uid.
  let degAxes : List AxisSpec := sc.stepDegAxesMulti
  let degree : StObjP := degAxes.map (fun a => AxisP.mk (some a.name) (SizeExpr.var a.name))
  -- internal read ⇒ publish producer step `j`'s slot-`slot` `tensorAxes` (the SAME (j,slot) the wire
  -- below uses, so producer output and consumer input weaves coincide); external ⇒ one fixed slot
  -- per read position.
  let inputWeaves : List WeaveShapeP :=
    readFactors.map (fun rf =>
      match nameToStep[rf.1]? with
      | some (j, slot) => (tensorAxes ((stmts.getD j default).slotStmt slot)).map
                            (fun a => WeaveSlotP.fixed a)
      | none   => (List.range rf.2.length).map (fun pos =>
                    WeaveSlotP.fixed (AxisP.mk (some (rf.1 ++ "_" ++ toString pos))
                      (SizeExpr.var (rf.1 ++ "_" ++ toString pos)))))
  -- one output weave per true output (base∩recur for scans), in `outputs` order.
  let outputWeaves : List WeaveShapeP := (List.range sc.outputs.length).map sc.slotWeave
  -- M4: consume the pre-route artifact (single source of truth) instead of recomputing the rows.
  let reindexings : List StMatP := sc.elaborateReindexings
  let op : BrOp :=
    if sc.isScanPre then .scanPre
    else if sc.isScan then (if sc.isAffineScan then .scanAffine else .scan)
    else match s.nonlin with
      | .relu        => .relu
      | .sigmoid     => .sigmoid
      | .tanh        => .tanh
      | .gelu        => .gelu
      | .leakyrelu   => .leakyrelu
      | .softmax _   => .softmax
      | .normalize _ => .normalize
      | .l2normalize _ => .l2normalize
      | .identity    => match s with
          | .scatter .. => .scatter
          | .assign ..  => match s.agg with
              | .max => .maxreduce
              | .min => .minreduce
              | .sum => .contract
          | .recurMorphism .. => .contract   -- unreachable: scanPre handled above
  let step : BrBaseP := { op, degree, inputWeaves, outputWeaves, reindexings }
  -- Route each read to its producer (step, slot), else to the external sentinel. A name that is
  -- neither produced nor declared external is an unresolved read: FAIL LOUD rather than
  -- silently defaulting to external slot 0 (which masks upstream dataflow errors).
  let wires ← readFactors.mapM (fun rf =>
    match nameToStep[rf.1]? with
    | some (j, slot) => pure (Wire.internal j slot)
    | none   =>
      match extIndex[rf.1]? with
      | some k => pure (Wire.external k)
      | none   => throw (CompileError.undeclaredName rf.1))
  return (step, wires)

/-- Acyclicity guard for Phase 8: every internal read wire points BACKWARD — its producer step
    precedes its consumer (`j < i`). False exactly for a genuine inter-step cycle (`topoSort` source-
    order fallback); scan self-recurrence is NOT a wire (excluded by `inputReadFactors`), so coupled
    scans pass. A cyclic dataflow can't be realized as a finite `Br` morphism, so `routeCore` rejects
    it (FAIL LOUD) — making a successful route imply topological order. -/
def routableInOrder (stmts : List ScanStmt) : Bool :=
  let ns := buildNameToStep stmts
  stmts.zipIdx.all (fun (sc, i) =>
    sc.inputReadFactors.all (fun rf =>
      match ns[rf.1]? with
      | some (j, _) => decide (j < i)
      | none        => true))

/-- M2 (`elaborateAffineReindexings`): the program-level affine reindexing artifact — each step's
    `elaborateReindexings`. A pre-route, `Except`-free compile-time artifact (no routing/cyclicity
    concerns), the single source of truth for reindex rows that `route` is shown to reproduce
    (`routeCore` step reindexings = this; per-step bridge `buildStep_reindexings`). -/
def elaborateAffineReindexings (sp : ScheduledProgram) : List (List StMatP) :=
  sp.stmts.map (·.elaborateReindexings)

/-- Pure core of Phase 8: compute the step list and routing table from a `ScheduledProgram`.
    Computes `nameToStep` and `extIndex` once (PASS 1), then folds `buildStep` over `stmts`
    (PASS 2). Guarded by `routableInOrder`: cyclic dataflow is rejected up front. -/
def routeCore (sp : ScheduledProgram) : Except CompileError (List BrBaseP × List (List Wire)) :=
  if routableInOrder sp.stmts then do
    -- PASS 2: fold buildStep over all stmts, using the PASS-1 maps (`buildNameToStep`/`buildExtIndex`).
    let pairs ← sp.stmts.mapM
      (buildStep (buildNameToStep sp.stmts) (buildExtIndex sp.extNames sp.stmts) sp.stmts)
    return (pairs.map (·.1), pairs.map (·.2))
  else
    throw (CompileError.cyclicDataflow "routeCore: cyclic dataflow (topoSort fallback)")

/-- Phase 8: route the scheduled statements into a `ThreadedComposed`. -/
def route (sp : ScheduledProgram) : FreshM ThreadedComposed := do
  let nExternal := sp.extNames.card
  match routeCore sp with
  | .ok (steps, routing) =>
      let tc : ThreadedComposed := { steps, routing, nExternal }
      -- Validate the domain well-formedness (every external slot referenced + rank agreement) on the
      -- built morphism. FAIL LOUD rather than emit a `tc` the bridge can't realize; this is the
      -- `WellFormed` conjunct-1 invariant carried by construction (see `wf_dom`).
      if tc.wellFormedDom then return tc
      else throw (CompileError.shapeMismatch
        "route: wellFormedDom failed (unreferenced external slot or read-rank mismatch)" "wellFormedDom")
  | .error e             => throw e

end LeanNCD
