-- LeanNCD/DSL/Pipeline/Lowering.lean
-- Phases 6–8 of the tensor-logic DSL back-end. Phase 6 (`splitNonlins`) isolates each
-- nonlinearity (relu/softmax/normalize) into its own step so contraction steps and
-- nonlinearity steps don't mix. Phases 1–5 live in `Structural.lean`.
import LeanNCD.DSL.Pipeline.Structural

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

/-- Split nonlinearities within a `ScanStmt`. For `.scan`, split each stmt in `base`/`recur`
    and flatten back into the base/recur lists, keeping the node coupled. -/
def splitScan (sc : ScanStmt) : FreshM (List ScanStmt) := do
  match sc with
  | .plain s => (← splitStmt s).mapM (fun s' => pure (ScanStmt.plain s'))
  | .scan nm ax base recur =>
      let base'  ← base.flatMapM splitStmt
      let recur' ← recur.flatMapM splitStmt
      return [ ScanStmt.scan nm ax base' recur' ]

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
  | .scan _ _ b r   => (b.map Stmt.lhsName ++ r.map Stmt.lhsName).eraseDups

/-- Tensor names a ScanStmt reads. -/
def ScanStmt.reads : ScanStmt → List String
  | .plain s        => s.readNames
  | .scan _ _ b r   => (b.flatMap Stmt.readNames ++ r.flatMap Stmt.readNames).eraseDups

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
  return { decls := lp.decls, stmts := liveStmts, env := lp.env, extNames := lp.extNames, ctx := lp.ctx }

end LeanNCD
