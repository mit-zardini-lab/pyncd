# Track A #1 — sound read-arity: reject malformed over-indexing + support diagonal-scatter reads

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans. Steps use `- [ ]` checkboxes. **Read §0 first** — Track B (the design
> spike) gates the diagonal-read tasks; Track A (the guard) is independent and can land immediately.

**Goal:** Make `readArityOk` (internal read arity = producer published rank) *true*, so #1
(`buildStep_reindexings_codLen_eq_inputRank`) becomes unconditional — by (A) rejecting genuinely
malformed over-indexed reads, and (B) making declared tensors written diagonally publish their
*declared* rank so rank-2 reads of them are sound.

**Architecture:** Two separable concerns. **(A)** a `checkReadRanks` extension that rejects reads of a
produced-but-undeclared intermediate at an arity ≠ its produced rank (no declaration justifies a
higher rank → malformed). **(B)** reclassify repeated-index ("diagonal") writes of a *declared* tensor
to a `scatter` that publishes the declared rank, and support *reading* a scatter-produced tensor at
declared rank (new: today no scatter output is ever read). (A) closes the serious soundness gap now;
(B) is the larger feature enabling the `Y[i,i]`-then-read-at-rank-2 case, and needs a design spike.

**Tech stack:** Lean 4 + Mathlib; the LeanNCD DSL pipeline (`Structural.lean`, `Lowering.lean`,
`RouteSpec.lean`) and evaluator (`Eval/Scatter.lean`, `Eval/Shape.lean`, `Eval/Eval.lean`).

## Global constraints

- No `sorry` in landed code; `#print axioms` on new theorems must be `[propext, Classical.choice,
  Quot.sound]` (subset OK).
- Additive where possible; do not regress the proved `route`/`compile_wellFormed` chain.
- Verification ladder: `lean_diagnostic_messages` per edit → `lake env lean <file>` → `lake build`
  (full, incl. `Tests`) at each task's end. Run `lake` from `leanncd/`.
- Behaviour-preserving refactors must keep the full test suite green (it is the regression guard).

---

## §0. Sequencing and the gating unknown

- **Task A (guard)** is independent and lands first — it closes the real soundness gap (silent
  acceptance of over-indexed undeclared intermediates, confirmed: eval silently broadcasts, route
  builds an ill-typed `BrBase`).
- **Tasks B1–B4 (diagonal-scatter reads)** are gated on **Spike B0**, because the pipeline has **no
  precedent for reading a scatter output at declared rank** (verified: the only scatter example
  `Out[2*i,2*j]` is a terminal output; `tensorAxes` publishes affine scatters as rank 0 free-axes and
  diagonal `[free i, free i]` as dedup rank 1 — neither is the declared rank). Resolving how a
  scatter-produced tensor is *published* and *read* at declared rank is a genuine design decision, not
  mechanical code. Do B0 before writing B1–B4.

---

## Task A: reject over-indexed reads of undeclared intermediates

**Files:**
- Modify: `LeanNCD/DSL/Pipeline/Structural.lean` (`checkReadRanks`, ~lines 152–166)
- Test: `test/DSL/Pipeline/StructuralTest.lean`

**Interfaces:**
- Consumes: `ResolvedProgram` (`.stmts`, `.env`, `.extNames`), `Stmt.lhsName`, `Stmt.lhsAxes`,
  `stmtReads : Stmt → List (String × Nat)` (already private in this file), `CompileError.rankMismatch`.
- Produces: `checkReadRanks` additionally throws `rankMismatch nm expected arity` when a read of a
  produced-but-undeclared name uses an arity ≠ that name's produced rank.

Producer rank for an undeclared intermediate = the dedup'd LHS-axis count of the (unique) stmt writing
it. Build a `produced : String → Nat` map from `rp.stmts` (`Stmt.lhsName` ↦ `(dedupByUid
(Stmt.lhsAxes s)).length`), then check reads not in `env` and not external against it.

- [ ] **Step 1 — failing test (offending program rejected).**
  In `StructuralTest.lean`, add (mirroring existing compile-error tests in this file; use the same
  compile entry the other tests use — check the file header for whether it calls `compile`/
  `TLProgram.compile` and returns `Except`):

```lean
-- over-indexed read of an undeclared intermediate T (produced rank 1, read rank 2) is rejected
#guard (TLProgram.compile (tlprog!{
    T[i]   := A[i,j] · B[j]
    Y[i,k] := T[i,k] · C[i,k]
  })).toOption.isNone
-- reading it at the produced rank is still accepted
#guard (TLProgram.compile (tlprog!{
    T[i] := A[i,j] · B[j]
    Y[i] := T[i] · C[i]
  })).toOption.isSome
```

- [ ] **Step 2 — run, verify the first `#guard` FAILS** (today the offending program compiles):
  `lake env lean test/DSL/Pipeline/StructuralTest.lean` → expect the over-index `#guard` to fail
  (reduces to `false`), the accepted one to pass.

- [ ] **Step 3 — implement the guard.** In `checkReadRanks`, after the existing declared/external
  loops, add:

```lean
  -- produced-but-undeclared intermediates: reject reads whose arity ≠ the produced (dedup'd) rank.
  let producedRank : Std.HashMap String Nat :=
    rp.stmts.foldl (fun m s => m.insert s.lhsName (dedupByUid s.lhsAxes).length) {}
  for (nm, arity) in reads do
    if rp.env[nm]?.isNone && !(nm ∈ rp.extNames) then
      match producedRank[nm]? with
      | some r => if arity != r then throw (.rankMismatch nm r arity)
      | none   => pure ()
```

  Note: `dedupByUid`/`Stmt.lhsAxes` live in `Lowering.lean`, which imports `Structural.lean` — so they
  are NOT in scope here. Either (i) move `dedupByUid` + `Stmt.lhsAxes` to a module `Structural.lean`
  imports (e.g. `DSL/Ast.lean`, next to `idxAffineForm`), or (ii) inline the dedup-count locally. Prefer
  (i) — it also lets `Lowering.lean` keep using them. Do the move as the first sub-step, rebuild, then
  add the guard.

- [ ] **Step 4 — run, verify both `#guard`s pass** and `lake build` is green (no regression; the guard
  must not reject any existing example — confirm `CompileExamplesTest`/`EvalExamplesTest` still pass).

- [ ] **Step 5 — commit.** `git add` the two files;
  `git commit -m "feat(track-a): reject over-indexed reads of undeclared intermediates (checkReadRanks)"`.

---

## Spike B0: how is a scatter-produced tensor published + read at declared rank?

**This is a design spike (like S0/S1), not mechanical code.** No task B1–B4 should be written until
its exit criteria are met.

**Question:** For a declared tensor `Y` produced by a diagonal write `Y[i,i] := X[i]` (declared rank
2), how should the pipeline (1) *publish* `Y`'s output at rank 2, and (2) build the *reader's*
reindexing so `Z[a,b] := Y[a,b]` reads the diagonal (`X[a]` if `a==b` else `fill`), and (3) evaluate
it? Today `tensorAxes` publishes the dedup'd rank (1); there is no reader path for scatter outputs.

**Investigate:**
- How `route`/`buildStep` derive a producer's output weave and a consumer's input weave for a normal
  (contract) producer, and what would change for a scatter producer to publish declared rank.
- Whether `evalScatter` already produces a correct rank-2 diagonal tensor when dispatched on slots
  `[free i, free i]` with `outShape` from the decl (`lhsSlotIdx (.free a) = .axis a`, so the output
  coord is `(i,i)` — likely yes for *eval*; the gap is *route*-side publishing + the reader's gather).
- How `inferAxisSizes` would size the reader's extra axis (`b`) — today it fails ("no inferable
  size"); with a rank-2 producer it must take `b`'s size from `Y`'s declared shape.

**Exit criterion:** a concrete design (or a `run_cmd`/scratch proof-of-concept) for producer-side
declared-rank publishing + reader-side reindexing + eval, OR a decision to scope diagonal-reads out
(documenting `Y[i,i]`-then-read as unsupported and having Task A reject *it too* for declared tensors
whose read arity ≠ published rank). Pick one before B1.

### B0 RESULT (2026-07-03, empirically probed)

- **Route already publishes a scatter output at its slot rank.** `Out[2*i,2*j]` routes to a `scatter`
  step with `outputWeaves len = [2]` — so **B2 is essentially free**: reclassifying a diagonal write to
  a scatter would publish rank 2 automatically.
- **But eval CANNOT read a scatter output.** `Out[2*i,2*j]:=X; Z[a,b]:=Out[a,b]` fails at eval:
  `"output axis b has no inferable size (it appears in no read position)"`. `inferAxisSizes` sizes
  axes from read positions; a scatter output carries no size source for the reader's axes. **Reading
  a scatter output at declared rank is unsupported end-to-end — the missing piece is eval-side (B3):
  size the reader's axes from the producer's declared/output shape + gather from the materialized
  tensor.** This is the real cost, and it spans `Eval/Shape.lean` (`inferAxisSizes`) + `Eval/Eval.lean`.
- **The diagonal `Y[i,i]` case is a DECLARED tensor**, so Task A's undeclared-only guard does not catch
  it: `tensor Y(a,b); Y[i,i]:=X[i]; Z[a,b]:=Y[a,b]` currently *compiles* (passes `checkReadRanks` via
  the declaration, 2=2) but is silently broken (producer publishes dedup rank 1; route mismatch + eval
  size error). A second, declared-tensor variant of the same gap.
- **B0 also caught + fixed a Task A bug** (`stmtLhsRank` used free-axis count → falsely rejected
  affine-scatter-output reads; fixed to slot-count for affine LHS, commit `8d771ac`).

**Decision (surfaced to the owner):**
- **Build** the feature: B1 (diagonal→scatter, small) + **B3 (eval read-path for scatter outputs:
  size reader axes from the producer shape + gather — medium, spans `Eval/Shape`+`Eval`)** + B4. Enables
  `Y[i,i]` rank-2 reads soundly.
- **Defer** the feature: document reading-scatter/declared-diagonal-outputs as unsupported, and extend
  the guard to reject the declared-tensor case too (a declared tensor whose producer publishes fewer
  axes than the read arity → reject with a clean error). Smaller; makes `readArityOk` hold by rejection
  rather than by supporting the read. `Y[i,i]` rank-2 reads stay disallowed.

Recommendation: **~~defer~~ BUILD B3 (corrected 2026-07-03).** Empirically, reading a scatter output
over its full output grid (fresh axes) fails ("no inferable size"), while a strided read-back aligned
to the scatter source works. Real decoder/GNN patterns hit the failing case: zero-insertion upsampling
+ conv (U-Net decoders / transposed-conv decomposition), GNN scatter-add then read node features,
max-unpool + conv, and any pointwise op on an upsampled feature map — all read the scatter output over
its full extent. §12.1 masks this only because its one scatter (`upsample`) is a terminal output. So
B3 (propagate a scatter's output shape into `inferAxisSizes` so downstream readers can size their
axes) is load-bearing for realistic multi-layer models, not niche. Build it.

---

## Tasks B1–B4 — FINAL STATUS (2026-07-03)

- **B1 — DONE** (commit `34f1a9c`): diagonal writes (repeated free axis) reclassify to scatter via the
  shared `slotsBecomeScatter` predicate (affine OR repeated-free), used by both `lowerArith` and
  `checkReadRanks`'s `stmtLhsRank` so they agree on the published rank. `Y[i,i]:=X[i]; Z[a,b]:=Y[a,b]`
  evals to the 2×2 diagonal (test in `EvalExamplesTest`). A degenerate-uid test shortcut was fixed
  (matmul used uid=1 for all axes ⇒ i/j collided ⇒ looked diagonal).
- **B2 — was ~free** (B0 finding): a scatter already publishes its output at slot rank; B1's
  reclassification inherits that.
- **B3 — DONE** (commit `57d333f`): `inferAxisSizes` derives scatter output shapes
  (`scatterOutDim`/`scatterOutputShapes`) and sizes downstream reads of scatter outputs — the eval
  read-path for scatter/decoder/GNN patterns. Test in `EvalExamplesTest`.
- **B4 — RESOLVED as "sound operationally; formal capstone deferred" (option 1, owner decision).**
  Soundness is already achieved: **Task A rejects arity-mismatched programs at compile time**, so any
  program that compiles satisfies the arity invariant, and the conditional `#1`
  (`buildStep_reindexings_codLen_eq_inputRank`, hypothesis `readArityOk`) is a correct route-level
  theorem whose hypothesis Task A enforces. Making `#1` *unconditional* (proving `readArityOk` for all
  compiled programs) is a **large multi-pass proof**: it must thread the arity=rank invariant from
  `checkReadRanks` (`ResolvedProgram`) to `readArityOk` (scheduled program) through `unifyAxes`,
  `lowerArith`, `finalizeScans`, `splitNonlins` (which *splits* stmts), and `schedule` — comparable in
  size to `compile_wellFormed`, and `wf_typeMatch` doesn't shortcut it (wire-types vs reindexing-codLen
  indexing). Deferred as its own effort; the cheaper alt (a redundant `readArityOk` validation in
  `routeCore`, mirroring `wellFormedDom`) was declined to avoid disturbing the proved chain for a
  formal-only gain. **Net: #1 stays conditional; soundness holds via Task A.**

---

## Risks

- **B0 is the real cost.** Reading scatter outputs is unimplemented; it may be a sizable feature. If
  B0 shows it's large and no §12.1 example needs it, the pragmatic choice is: land Task A, extend it
  to also reject `Y[i,i]`-then-read-at-declared-rank (documenting diagonal-reads unsupported), and
  make #1 unconditional that way — sound, smaller, defers the feature.
- **Task A's `dedupByUid` move** could ripple (it's imported by `Lowering.lean`); do the move as an
  isolated, green sub-step first.
- **No new axioms**; keep the proved chain intact.

## Self-review notes

- Spec coverage: (A) guard = malformed-input rejection; (B0–B4) = diagonal-read feature + #1
  unconditional. Both concerns from the design discussion are tasked.
- Task A is fully concrete and independently landable. B-tasks are honestly gated on B0 (a genuine
  design unknown), not fabricated — pre-writing their code would be guesswork.
