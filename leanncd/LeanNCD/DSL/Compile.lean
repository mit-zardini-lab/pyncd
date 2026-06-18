import LeanNCD.DSL.Pipeline.Structural
import LeanNCD.DSL.Pipeline.Lowering
import LeanNCD.DSL.Elab     -- elabTLProgram, tl_program syntax

/-!
# `TLProgram.compile` + the `tl!{ … }` term macro (Milestone E2a, Task 12)

`TLProgram.compile` chains the eight §12.4 pipeline phases in `FreshM`, each phase
narrowing the structural invariant. The `tl!{ … }` macro runs Stage 1 (parse, via
`elabTLProgram`) and Stage 2 (compile) at elaboration time and embeds the resulting
computable `ThreadedComposed` via its derived `ToExpr`.
-/

namespace LeanNCD
open Lean

/-- The §12.4 pipeline as an explicit `FreshM` do-bind chain (each phase narrows
    the invariant). -/
def TLProgram.compile (p : TLProgram) : FreshM ThreadedComposed := do
  let a ← assignUIDs p
  let b ← resolveDecls a
  let c ← unifyAxes b
  let d ← lowerArith c
  let e ← finalizeScans d
  let f ← splitNonlins e
  let g ← schedule f
  route g

/-- The compile pipeline WITHOUT the final `route` — yields a ScheduledProgram that retains scan
    bodies + lowered ops + decls (dtype). The evaluator consumes this (the routed ThreadedComposed
    collapses scan bodies and can't be evaluated). -/
def TLProgram.compileToScheduled : TLProgram → FreshM ScheduledProgram :=
  assignUIDs >=> resolveDecls >=> unifyAxes >=> lowerArith >=> finalizeScans >=> splitNonlins >=> schedule

/-- Stage 1 (parse) + Stage 2 (compile) at elaboration time; embed the computable
    `ThreadedComposed` presentation via `ToExpr`. -/
elab "tl!{" p:tl_program "}" : term => do
  let prog ← elabTLProgram p.raw
  match TLProgram.compile prog |>.run 0 with
  | .ok tc _    => return Lean.toExpr tc
  | .error e _  => throwError s!"tl! compile error: {repr e}"

end LeanNCD
