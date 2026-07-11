# E6 — Property-Based Oracles (scan-free harness) — Design

**Status:** approved 2026-07-11. Feeds `writing-plans`.

## Goal

Add a build-time, **bounded-exhaustive** property test that generates small well-formed
*scan-free* `TLProgram`s and checks two **metamorphic laws** of the pipeline —
**reordering-invariance** and **materialization-invariance** — so whole-program semantic bugs
that the ~130 hand-computed point-examples structurally cannot catch are caught mechanically.
This is the E6 exploration from `papers/restructure_suggestions.md`, built as its
highest-value-first *net* rather than proof.

Non-goal: proving correctness. Property tests reduce risk by sampling/enumeration within a
bound; they do not prove the laws.

## Scope

**In:** the scan-free fragment — a generator over small programs, two transforms, an
exact-comparison oracle runner, and a build-failing entry point, plus "test-the-tester"
checks.

**Out (deferred to a follow-up spike):** scan generation and the **scan-unrolling** oracle
(the doc's third, strongest oracle). This harness is designed so scan support slots in later
as a third transform + generator extension without reworking the infrastructure.

## Background facts (verified in the codebase)

- Eval entry: `TLProgram.eval (p : TLProgram) (inputs : HashMap String DenseTensor) : Except EvalError _`
  runs the full pipeline (compile → schedule → eval). The reordering law is *about*
  `topoSort`/`schedule`, so the oracle must go through this full entry, not the inner
  `evalScheduled`/`evalPlain`.
- `DenseTensor = { shape : List Nat, data : Array Float }` (derives `Repr, Inhabited`; **no**
  `Beq`/`DecidableEq`). Output comparison is therefore an explicit shape+`Float`-array check.
- `Plausible` is available in the toolchain, but this design uses **bounded-exhaustive
  enumeration** instead (see Framework), so `Plausible` is not a dependency of the harness.
- Generation targets internal `Stmt`/`RHSExpr`/`Decl`/`TLProgram` values **directly** — no
  round-trip through the surface `tl!{…}` syntax.

## Framework: bounded-exhaustive enumeration (not random)

Enumerate **all** well-formed programs up to a small size bound and check the law on each,
inside a `run_cmd` that **throws on the first counterexample** — failing `lake build` exactly
like every existing `#guard`/`run_cmd` test. Rationale: deterministic + reproducible in CI,
complete *within the bound* (cannot miss a case a random sampler would skip), integrates with
the existing build-failure test convention, and needs no `Gen`/`Sampleable`/`Shrinkable`
plumbing. Enumerating smallest-first makes the first counterexample naturally minimal (the
shrinking Plausible would otherwise provide).

## Components (under `test/Eval/PropertyOracle/`)

### 1. Generator

Bounded-exhaustive enumeration of `(TLProgram, inputs)` pairs. Bounds (tunable so `lake build`
stays fast): **1–3 axes** of **size ≤ 3**, **1–4 statements**.

Program vocabulary (**core + affine reads**):
- **Reads** with affine index expressions per axis: `i`, `i + c`, `k·i`, `k·i + c`, for small
  `k ∈ {1,2}`, `c ∈ {0,1}`.
- **Products** (contraction) of reads over shared axes.
- **Multi-term sums**: an RHS is a sum of ≥1 terms, each a read-term or a product-term (a
  multi-term RHS is required for the materialization law).
- **LHS = plain free axes only** — no affine/scatter LHS (keeps well-formedness tractable;
  no injectivity/overlap guards needed). Affine appears on the *read* side, exercising the
  reindex path.
- No nonlin, no scans (scan is the deferred follow-up).

Well-formedness the generator must guarantee: every read names a declared tensor with matching
arity; free axes on the LHS cover the statement's uncontracted axes; axis sizes are concrete
`Nat` ≤ 3; a statement only reads names that are external inputs or produced by an earlier
statement (so at least one valid schedule exists — though the reordering law then checks all
orders).

Inputs: one **deterministic, non-degenerate** assignment per program — e.g. `data[flat] = flat + 1`
(nonzero, non-constant, so contraction sums and affine reads are distinguishing). Out-of-bounds
affine reads (e.g. `X[i+1]` at the last index) are fine: both original and transformed programs
default identically via the evaluator's `getD 0`, so the laws still hold.

### 2. Transforms

- **`permutations`** (reordering): every permutation of the program's statement list. Since
  `schedule` re-sorts topologically, eval must be invariant under **all** permutations
  (the stronger, cleaner property — no need to detect independence).
- **`materializeSplit`** (materialization): rewrite each multi-term statement
  `Y := t1 + t2 + … + tn` into `T1 := t1; T2 := t2; … ; Tn := tn; Y := T1 + T2 + … + Tn` with
  fresh, non-colliding intermediate names, preserving each term's index structure.

### 3. Oracle runner

For each enumerated `(p, inputs)`, smallest-first:
- **reordering:** `eval` every statement-permutation of `p`; assert each equals `eval p`.
- **materialization:** `eval` `materializeSplit p`; assert it equals `eval p`.

**Comparison — exact:** two results are equal iff both are `.ok` with equal `shape` and
bit-equal `data` arrays (or both the same `.error`). Justification: both sides run the *same*
evaluator with the *same* per-element arithmetic order (the transforms relabel/reorder
statements, they do not reorder arithmetic within a contraction or sum), so any difference is a
genuine discrepancy, not float noise. If exact ever proves spuriously strict, fall back to an
elementwise tolerance `|a-b| ≤ ε` — flagged, not silent.

On the first violation: `throwError` with the minimal offending program (its `Repr`), the input
env, and both eval outputs — so `lake build` fails with an actionable counterexample.

### 4. Entry point

A `run_cmd`/`#guard`-style top-level check in a test module registered in `leanncd/lakefile.toml`'s
`Tests` globs (e.g. `Eval.PropertyOracleTest`), so a violated law fails `lake build`.

## Data flow

```
enumerate (p, inputs)  ──►  per program:
      build variants (all permutations; the split form)
   ─► TLProgram.eval p inputs           (baseline)
   ─► TLProgram.eval variant inputs     (each variant)
   ─► compare exact  ─► ok ? continue : throwError (minimal counterexample)
```

## Testing the tester (so the oracle is not vacuous)

Two unit checks alongside the enumeration:
- **(a) sanity-pass:** a small hand-written program passes both laws.
- **(b) has-teeth:** a deliberately *wrong* transform (e.g. a materialization that drops or
  swaps a term) is *detected* by the comparison — proving the oracle can fail. This guards
  against a silently-vacuous oracle (a law that can never fail is worthless — cf. the project's
  Rule 9).

## Risks / notes

- **Enumeration size:** bounds must be capped so the full enumeration runs quickly under
  `lake build`. Start conservative (e.g. ≤3 statements, ≤2 axes) and widen only if fast.
- **Generator well-formedness is the main effort** (the doc's "the generator is the real work").
  Building it directly on `Stmt`/`Decl` values avoids parser coupling but requires getting the
  arity/axis/declaration invariants right — the "test-the-tester" checks and the existing
  ~130 examples (which must still pass) are the guardrails.
- **Exact float comparison** could, in principle, surface a legitimate reordering of a
  floating-point reduction; if so that is itself a finding (the law would then be only
  approximate), to be adjudicated, not silently tolerated.

## File layout (proposed)

- `test/Eval/PropertyOracle/Gen.lean` — the enumerator (`TLProgram` + inputs).
- `test/Eval/PropertyOracle/Transforms.lean` — `permutations`, `materializeSplit`.
- `test/Eval/PropertyOracle/Oracle.lean` — comparison + runner + the two laws.
- `test/Eval/PropertyOracleTest.lean` — the `run_cmd` entry + the two test-the-tester checks;
  registered in `lakefile.toml` `Tests` globs.

(Exact module split is a plan-time detail; the responsibilities above are the fixed boundaries.)
