import LeanNCD.Eval.Eval
/-!
# Portfolio test harness

Shared helpers for the tensor-logic DSL test portfolio (see `docs/test_portfolio.md`).
Every portfolio file imports this module and uses these three assertion helpers, which
correspond to the three test styles in the portfolio:

* `assertEval`   — `[N]` numeric: run on concrete inputs, compare with `approxEq`.
* `assertEvalError` — `[R]`/`[F]` runtime/compile failure surfaced through `eval`.
* `assertCompileError` — `[R]` a specific `CompileError` constructor from the compile pipeline.

Parse-level rejects (`[R]` that fail to elaborate, e.g. symbolic strides) use Lean's
`#guard_msgs` directly in the reject file, since `tlprog!` fails at elaboration and cannot be
caught by `run_cmd`.
-/
namespace LeanNCD.Eval
open Std Lean Elab Command

/-- A dense tensor from a shape and row-major data. -/
def tl (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩

/-- `[N]`: evaluate `prog` on `env`, assert output `key` ≈ `expect`. -/
def assertEval (nm : String) (prog : TLProgram) (env : HashMap String DenseTensor)
    (key : String) (expect : DenseTensor) : CommandElabM Unit := do
  match TLProgram.eval prog env with
  | .error e => throwError s!"{nm}: eval error: {e}"
  | .ok out => match out[key]? with
    | some t =>
        unless t.shape == expect.shape do
          throwError s!"{nm}: shape {t.shape} ≠ expected {expect.shape}"
        unless DenseTensor.approxEq t expect do
          throwError s!"{nm}: got {t.data.toList}, want {expect.data.toList}"
    | none => throwError s!"{nm}: no output '{key}' (have {out.toList.map (·.1)})"

/-- `[R]`/`[F]`: assert `eval` fails and its error string contains `needle`. -/
def assertEvalError (nm : String) (prog : TLProgram) (env : HashMap String DenseTensor)
    (needle : String) : CommandElabM Unit := do
  match TLProgram.eval prog env with
  | .ok _ => throwError s!"{nm}: expected eval failure containing '{needle}', but it succeeded"
  | .error e =>
      unless needle.isEmpty || (e.splitOn needle).length > 1 do
        throwError s!"{nm}: error '{e}' does not contain '{needle}'"

/-- `[R]`: assert the compile pipeline rejects `prog` (any `CompileError`).  Callers that need a
    *specific* constructor should match `TLProgram.compile prog |>.run 0` directly. -/
def assertCompileError (nm : String) (prog : TLProgram) : CommandElabM Unit := do
  match TLProgram.compile prog |>.run 0 with
  | .ok _ _ => throwError s!"{nm}: expected a CompileError, but compile succeeded"
  | .error _ _ => pure ()

/-- Property check: evaluate, then assert `p` holds on output `key` (use for "rows sum to 1",
    "masked entry is 0", etc. — when the doc gives a property rather than a full tensor). -/
def assertEvalPred (nm : String) (prog : TLProgram) (env : HashMap String DenseTensor)
    (key : String) (p : DenseTensor → Bool) (desc : String) : CommandElabM Unit := do
  match TLProgram.eval prog env with
  | .error e => throwError s!"{nm}: eval error: {e}"
  | .ok out => match out[key]? with
    | some t => unless p t do throwError s!"{nm}: property failed ({desc}); got {t.data.toList}"
    | none   => throwError s!"{nm}: no output '{key}'"

/-- Shape-only check: evaluate and assert output `key` has the given shape (for "confirmed
    evals end-to-end" cases where an exact value isn't hand-computed). -/
def assertShape (nm : String) (prog : TLProgram) (env : HashMap String DenseTensor)
    (key : String) (shape : List Nat) : CommandElabM Unit := do
  match TLProgram.eval prog env with
  | .error e => throwError s!"{nm}: eval error: {e}"
  | .ok out => match out[key]? with
    | some t => unless t.shape == shape do throwError s!"{nm}: shape {t.shape} ≠ {shape}"
    | none   => throwError s!"{nm}: no output '{key}'"

/-- Are all `axis`-major rows (last-axis groups) of `t` summing to ≈1? (softmax sanity.) -/
def rowsSumToOne (t : DenseTensor) : Bool :=
  match t.shape.getLast? with
  | none => false
  | some w =>
      if w == 0 then false else
      let n := t.data.size / w
      (List.range n).all (fun r =>
        let s := (List.range w).foldl (fun acc c => acc + t.data.getD (r*w+c) 0.0) 0.0
        Float.abs (s - 1.0) < 1e-6)

end LeanNCD.Eval
