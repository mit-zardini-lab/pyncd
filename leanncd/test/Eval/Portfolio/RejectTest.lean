import Eval.Portfolio.Harness
/-!
# Portfolio §13 — Adversarial reject tests

Programs the DSL should refuse. Three sub-kinds:

* **compile-error** — caught via `TLProgram.compile … |>.run 0` (specific `CompileError`).
* **eval-error** — surfaced through `TLProgram.eval` (`assertEvalError`).
* **parse-error** — fail during elaboration of `tlprog!`; these CANNOT be automated (a hard
  parse error fails the build, and `#guard_msgs` does not validate parse-time errors), so they
  are documented as comments below.

Probed against HEAD (2026-07-04). Note: the draft's RJ6 (underdetermined loop axis) and RJ9
(purely-negative index) do **not** actually reject — they return output — so they are not
authored here; see the note at the end.
-/
namespace LeanNCD.Eval
open Std

-- RJ3  predicate output + non-sum aggregation ⇒ CompileError.predicateAgg
run_cmd do
  match TLProgram.compile (tlprog!{ predicate P(i)
                                    P[i] := maxreduce(E[i, j]) }) |>.run 0 with
  | .error (.predicateAgg "P") _ => pure ()
  | .error e _ => throwError s!"RJ3: wrong CompileError: {repr e}"
  | .ok _ _    => throwError "RJ3: expected predicateAgg, compile succeeded"

-- RJ4  softmax with no `·`-marked reduction axis ⇒ eval error
run_cmd (assertEvalError "RJ4 softmax-no-axis"
  (tlprog!{ A[q, s] := softmax(Q[q, d] · K[s, d]) })
  (HashMap.ofList [("Q", tl [2,2] [1,0,0,1]), ("K", tl [2,2] [1,0,0,1])])
  "no output axis is marked")

-- RJ7  an axis unified to two different sizes ⇒ eval size-inconsistency error
run_cmd (assertEvalError "RJ7 size-conflict"
  (tlprog!{ s[] := A[i] · B[i] })
  (HashMap.ofList [("A", tl [3] [1,2,3]), ("B", tl [2] [1,1])])
  "inconsistent")

-- SS4 / RC4  a scan step reading an external at the advancing index l+1 ⇒ causalityViolation
run_cmd do
  match TLProgram.compile (tlprog!{ axis l : ℕ = 3
                                    S[j, 0]    := X[j, 0]
                                    S[j, l +1] := S[j, l] + X[j, l + 1] }) |>.run 0 with
  | .error (.causalityViolation "S") _ => pure ()
  | .error e _ => throwError s!"SS4: wrong CompileError: {repr e}"
  | .ok _ _    => throwError "SS4: expected causalityViolation, compile succeeded"

-- UF1  unknown tensor referenced ONLY via a `.unaryFn` factor (`log(Missing[i])`) ⇒ the same
--   up-front "unknown tensor" eval error as a plain `.read` would give — regression test for
--   `readNames` treating `.unaryFn` exactly like `.read` (§19 test_portfolio discussion).
run_cmd (assertEvalError "UF1 unaryFn-unknown-tensor"
  (tlprog!{ L[] := log(Missing[i]) })
  (HashMap.ofList [("dummy", tl [2] [1,1])])
  "unknown tensor")

-- UF2  a look-ahead scan read HIDDEN inside `log(...)` must still trip the causality guard —
--   regression test for `Stmt.rhsReads` treating `.unaryFn` exactly like `.read` (the guard
--   is computed from index expressions, not raw factor reads, so a careless implementation
--   could let this slip through).
run_cmd do
  match TLProgram.compile (tlprog!{ axis l : ℕ = 3
                                    S[j, 0]    := X[j, 0]
                                    S[j, l +1] := S[j, l] + log(X[j, l + 1]) }) |>.run 0 with
  | .error (.causalityViolation "S") _ => pure ()
  | .error e _ => throwError s!"UF2: wrong CompileError: {repr e}"
  | .ok _ _    => throwError "UF2: expected causalityViolation, compile succeeded"

-- UF3  domain violation: log of a non-positive value fails loud (not a silent NaN/-inf), per
--   this evaluator's fail-loud convention.
run_cmd (assertEvalError "UF3 log-domain-violation"
  (tlprog!{ L[] := log(P[i]) })
  (HashMap.ofList [("P", tl [2] [1, -1])])
  "log domain error")

/-
Parse-level rejects (fail during elaboration of `tlprog!`; not automatable — kept as documentation):

  RJ1  symbolic-coefficient stride:   Y[i] := W[p] · X[s * j]
       → `unexpected token '*'; expected ']'`  (IdxExpr has no symbolic-coeff strides)

  RJ2  floor-division by a variable:   Y[i] := X[i / j]
       → `unexpected token '/'; expected ']'`  (`/` requires a literal divisor)

  RJ5  over-indexed read of an undeclared intermediate — rejected by Task-A guard.

  RJ8  softmax norm axis not among outputs — not constructible via surface syntax (marking a
       slot always places it on the LHS).

  RJ10 scatter with an unsized output axis — `scatterOutShape` fails loud; needs an upstream
       sizing gap that surface syntax does not readily produce.

Dropped from the draft (do NOT reject — return output instead):
  RJ6  scan with no `axis l` pin and no input fixing `l`  → evaluates (0-step / defaulted).
  RJ9  Y[i] := X[i - 5] with a short input               → evaluates (zero-padded), no Issue-D error.
-/

end LeanNCD.Eval
