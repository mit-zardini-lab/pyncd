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

end LeanNCD
