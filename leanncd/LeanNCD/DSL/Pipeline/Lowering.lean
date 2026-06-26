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
           extNames := sp.extNames, ctx := sp.ctx }

/-! ## Phase 7 — `schedule`

Dead-code elimination by backward reachability from the program's outputs. A statement is
kept iff it writes a tensor name that is (transitively) needed to compute an output.

NOTE: a full topological re-sort is NOT performed here — for the §12.1 example programs
source order already places producers before consumers, so we keep the surviving stmts in
their original source order. -/

/-- Tensor names a ScanStmt writes (its LHS name(s)). -/
def ScanStmt.writes : ScanStmt → List String
  | .plain s        => [s.lhsName]
  | .scan _ _ b r _ => (b.map Stmt.lhsName ++ r.map Stmt.lhsName).eraseDups
  | .scanPre nm _ _ => [nm]

/-- Tensor names a ScanStmt reads. -/
def ScanStmt.reads : ScanStmt → List String
  | .plain s        => s.readNames
  | .scan _ _ b r _ => (b.flatMap Stmt.readNames ++ r.flatMap Stmt.readNames).eraseDups
  | .scanPre _ _ _  => []

/-- One backward-reachability pass: add names read by any stmt that writes a live name. -/
def liveStep (stmts : List ScanStmt) (live : List String) : List String :=
  (live ++ stmts.flatMap (fun sc =>
    if (sc.writes).any (fun w => live.contains w) then sc.reads else [])).eraseDups

/-- Iterate `liveStep` to a fixpoint. Terminates because `live` grows monotonically and is
    bounded by the finite tensor-name set; `partial` sidesteps the termination proof. -/
partial def liveFix (stmts : List ScanStmt) (live : List String) : List String :=
  let live' := liveStep stmts live
  if live'.length == live.length then live else liveFix stmts live'

/-- Phase 7: keep only the stmts (transitively) needed to compute the program's outputs.

    Output detection: the program's result is the LAST stmt's written name(s). We then UNION
    any other produced names that are read by NO stmt — these are additional results — but
    ONLY when they are still live, which prevents a genuinely dead unread tensor (computed,
    never consumed, not the final result) from masquerading as an output. Concretely we seed
    reachability from the last stmt's writes, take the fixpoint, then fold in unread-produced
    names that already turned out live. -/
def schedule (lp : LinearProgram) : FreshM ScheduledProgram := do
  let produced := lp.stmts.flatMap ScanStmt.writes
  let read     := lp.stmts.flatMap ScanStmt.reads
  -- Primary output(s): the last stmt's written name(s); fall back to all unread-produced.
  let lastOut  := (lp.stmts.reverse.head?.map ScanStmt.writes).getD []
  let outputs  := if lastOut.isEmpty then produced.filter (fun n => ¬ read.contains n) else lastOut
  let live := liveFix lp.stmts outputs.eraseDups
  let liveStmts := lp.stmts.filter (fun sc => (sc.writes).any (fun w => live.contains w))
  -- collect the sizes pinned by `axis … = n` decls (UIDs are canonical by this phase).
  let explicitSizes : Std.HashMap UID Nat := lp.decls.foldl (fun m d => match d with
    | .axis ax (some n) => m.insert ax.uid n
    | _                 => m) {}
  return { decls := lp.decls, stmts := liveStmts, env := lp.env, extNames := lp.extNames,
           ctx := lp.ctx, explicitSizes }

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
        | .iverson _  => none))
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

/-- Express a read coordinate `IdxExpr` as an integer-affine combination of the degree axes
    identified by uids `us` (column order = `us`): returns `(coeff-row, bias)`. -/
def idxToRow (us : List UID) : IdxExpr → (List Int × Int)
  | .axis a      => (us.map (fun u => if u == a.uid then 1 else 0), 0)
  | .const n     => (us.map (fun _ => 0), n)
  | .scale c a   => (us.map (fun u => if u == a.uid then c else 0), 0)
  | .shift a n   => (us.map (fun u => if u == a.uid then 1 else 0), n)
  | .affine n xs => (us.map (fun u => (xs.foldl (fun acc p => if p.2.uid == u then acc + p.1 else acc) 0)), n)

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

/-- Build one `BrBaseP` step and its routing wires for `sc`, given the precomputed maps.
    Returns `CompileError.shapeMismatch` for an empty-step `scanPre`, or
    `CompileError.undeclaredName` for an unresolved read. -/
def buildStep (nameToStep : Std.HashMap String Nat) (extIndex : Std.HashMap String Nat)
    (sc : ScanStmt) : Except CompileError (BrBaseP × List Wire) := do
  -- Validate a pre-built (escape-hatch) morphism: its step list must be non-empty.
  match sc with
  | .scanPre nm _ tc =>
      if tc.steps.isEmpty then
        throw (CompileError.shapeMismatch s!"recurMorphism {nm}: empty step morphism" "non-empty ThreadedComposed")
  | _ => pure ()
  let s := sc.repStmt.getD (.assign "" [] { body := { terms := [] }, nonlin := .identity })
  let lhsAxes := s.lhsAxes
  let lhsUids := lhsAxes.map (·.uid)
  let readFactors := s.readFactors
  let readAxes := readFactors.flatMap (fun rf => rf.2.flatMap idxAxes)
  let contracted := readAxes.filter (fun a => !lhsUids.contains a.uid)
  -- degree = LHS axes ++ contracted, de-duplicated by uid (LHS first).
  let degAxes : List AxisSpec :=
    (lhsAxes ++ contracted).foldl (fun acc a =>
      if acc.any (fun b => b.uid == a.uid) then acc else acc ++ [a]) []
  let contractedUids : List UID := contracted.map (·.uid)
  let degree : StObjP := degAxes.map (fun a => AxisP.mk (some a.name) (SizeExpr.var a.name))
  -- weave per degree axis: contracted ⇒ tiled, else fixed.
  let mkWeave : List WeaveSlotP :=
    degAxes.map (fun a => if contractedUids.contains a.uid then WeaveSlotP.tiled else WeaveSlotP.fixed (AxisP.mk (some a.name) (SizeExpr.var a.name)))
  let inputWeaves : List WeaveShapeP := readFactors.map (fun _ => mkWeave)
  let outputWeaves : List WeaveShapeP := [ mkWeave ]
  let degUids : List UID := degAxes.map (·.uid)
  let reindexings : List StMatP := readFactors.map (fun rf =>
    let rows := rf.2.map (idxToRow degUids)
    { domLen := degUids.length, codLen := rf.2.length,
      coeffs := rows.map (·.1), bias := rows.map (·.2) })
  let op : BrOp :=
    if sc.isScanPre then .scanPre
    else if sc.isScan then (if sc.isAffineScan then .scanAffine else .scan)
    else match s.nonlin with
      | .relu        => .relu
      | .softmax _   => .softmax
      | .normalize _ => .normalize
      | .identity    => match s with
          | .scatter .. => .scatter
          | .assign ..  => match s.agg with
              | .max => .maxreduce
              | .sum => .contract
          | .recurMorphism .. => .contract   -- unreachable: scanPre handled above
  let step : BrBaseP := { op, degree, inputWeaves, outputWeaves, reindexings }
  -- Route each read to its producer step, else to the external sentinel. A name that is
  -- neither produced nor declared external is an unresolved read: FAIL LOUD rather than
  -- silently defaulting to external slot 0 (which masks upstream dataflow errors).
  let wires ← readFactors.mapM (fun rf =>
    match nameToStep[rf.1]? with
    | some j => pure (Wire.internal j 0)
    | none   =>
      match extIndex[rf.1]? with
      | some k => pure (Wire.external k)
      | none   => throw (CompileError.undeclaredName rf.1))
  return (step, wires)

/-- Pure core of Phase 8: compute the step list and routing table from a `ScheduledProgram`.
    Computes `nameToStep` and `extIndex` once (PASS 1), then folds `buildStep` over `stmts`
    (PASS 2). -/
def routeCore (sp : ScheduledProgram) : Except CompileError (List BrBaseP × List (List Wire)) := do
  -- PASS 1: step indices, name→step map, external-name numbering.
  let nameToStep : Std.HashMap String Nat :=
    sp.stmts.zipIdx.foldl (fun m (sc, i) =>
      sc.writes.foldl (fun m' nm => m'.insert nm i) m) {}
  let extIndex := buildExtIndex sp.extNames sp.stmts
  -- PASS 2: fold buildStep over all stmts.
  let pairs ← sp.stmts.mapM (buildStep nameToStep extIndex)
  return (pairs.map (·.1), pairs.map (·.2))

/-- Phase 8: route the scheduled statements into a `ThreadedComposed`. -/
def route (sp : ScheduledProgram) : FreshM ThreadedComposed := do
  let nExternal := sp.extNames.card
  match routeCore sp with
  | .ok (steps, routing) => return { steps, routing, nExternal }
  | .error e             => throw e

end LeanNCD
