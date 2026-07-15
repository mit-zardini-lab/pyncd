# E11 — Do UIDs Earn Their Keep? — Design

**Status:** approved 2026-07-15. Feeds `writing-plans`.

## Goal

Resolve the E11 exploration from `papers/restructure_suggestions.md`: axis identity is defined
to be name-based within program scope (§12.1), yet the pipeline mints `Nat` UIDs (`FreshM`),
then runs a coequalizer (`Exec/Context.lean`) and `unifyAxes` to merge UIDs that name-identity
says were never distinct. The investigation's job was to answer, with evidence, the three
honest unknowns the doc lists — not assume an answer:

1. Does any phase genuinely distinguish same-named occurrences?
2. Does `Context`'s coequalizer do work beyond name-unification?
3. What do the bridge/acset layers key on — is `axisUidFor3` evidence that UIDs are load-bearing
   in serialization?

**Answer, with evidence (all three resolved):**

1. **No.** Marks like `freeNorm`/`iterAt` live on the LHS slot (the occurrence/constructor), never
   on `AxisSpec` identity (`DSL/Ast.lean:103-109`). No phase found treats two same-named
   occurrences as permanently distinct after unification — the only place that ever happens is
   one unit test that deliberately bypasses `assignUIDs` to construct that input by hand.
2. **No.** `unifyAxes`'s own doc comment states it outright: *"In the full pipeline `assignUIDs`
   already binds each name to one UID, so this is effectively identity there; the standalone test
   feeds DISTINCT UIDs for one name to genuinely exercise the merge."* (`Structural.lean:252-254`).
   `assignUIDs` mints exactly one fresh UID per distinct axis name (`Structural.lean:78-90`), so
   every bucket `unifyAxes`/`Context.merge` ever groups in a real compiled program is a singleton.
   `finalizeScans`'s UID-keyed scan grouping runs *after* `unifyAxes` in the pipeline
   (`Compile.lean:19-29`) and is itself pre-seeded by matching same *tensor name*
   (`Structural.lean:400-409`) — no divergence from name-identity anywhere downstream.
3. **Not load-bearing for this question.** `axisUidFor3` (`Bridge/AcsetCodec.lean:25`) is an
   unrelated, independently-synthesized `Acset.AxisUID` built from `(step, slot, position)`
   structural coordinates at the serialization layer — not a use of the pipeline's minted `UID`.
   The actual `AxisP` type the bridge/route layer consumes (`DSL/Target.lean:36-38`) has **no
   uid field at all** — only `name` survives past the route boundary. Confirmed independently:
   `CanonicalProgram.ctx` (the value `unifyAxes` produces) is threaded through 4 further phases
   (`lowerArith → finalizeScans → splitNonlins → schedule`) and then simply dropped — `route`
   never reads it, and nothing ever pattern-matches or folds over its contents.

**Outcome: major simplification.** `unifyAxes` and `Exec/Context.lean`'s generic coequalizer are
dead machinery in the compiled pipeline — the same shape as Spike 1a's `liveFix`/`liveStep`
finding. Delete the phase, its type, and the machinery it exists solely to drive.

## Scope

**In:** delete `unifyAxes`, delete `Exec/Context.lean` (`EqClass`/`Context`/`Context.merge`/
`Context.apply`), delete the now-redundant `CanonicalProgram` type (identical to
`ResolvedProgram` once its `ctx` field is gone), drop the `ctx : Context AxisSpec` field from
`LoweredProgram`/`ScanProgram`/`LinearProgram`/`ScheduledProgram`, and update every call site,
test, and doc reference this touches.

**Out — explicitly descoped, not attempted here:** eliminating the `Nat` UID concept from axis
identity entirely (rekeying `Eval`'s `HashMap UID _` dictionaries by `String` name, dropping
`AxisSpec.uid`). That change reaches `dedupByUid`, which `RouteSpec.lean` unfolds definitionally
(`dedupByUid_eq_foldl (xs : List AxisSpec) : dedupByUid xs = xs.foldl dstep [] := rfl`,
`RouteSpec.lean:250-251`, plus lemmas built on it like `dedupByUid_append_filter`) — Constraint 3
in `papers/restructure_suggestions.md` flags exactly this as proof-repair territory that Spike 6
should precede or accompany. Deferred as a separate, explicitly-sequenced follow-on; not part of
this investigation's outcome.

## Background facts (verified in the codebase)

- `FreshM := EStateM CompileError Nat` (`Exec/Uid.lean:40`); `freshUData` mints a fresh `UData`
  (`Exec/Uid.lean:43-46`). The *only* other `freshUData` call site besides `assignUIDs` is
  `splitStmt`'s `%nl`-style compiler-generated intermediate tensor names (`Lowering.lean:36-37`)
  — a legitimate, separate use the doc itself flags as distinct; untouched by this change.
- `TermTraversable`/`traverseUID` (`Exec/Traversable.lean`, instances in `DSL/Traverse.lean`) is
  **not** deleted — `assignUIDs` calls `TermTraversable.traverseUID relabel p` directly
  (`Structural.lean:96`), independent of `Context.apply`. `Exec/Context.lean` only *consumes*
  `Traversable.lean` (`import LeanNCD.Exec.Traversable`, `Context.lean:1-2`); the dependency is
  one-directional.
- Exact signatures: `unifyAxes (rp : ResolvedProgram) : FreshM CanonicalProgram`
  (`Structural.lean:265`); `lowerArith (cp : CanonicalProgram) : FreshM LoweredProgram`
  (`Structural.lean:307`). `ResolvedProgram := { decls, stmts, env, extNames }`
  (`Types.lean:25-29`) — identical fields to `CanonicalProgram` minus `ctx`
  (`Types.lean:31-36`), confirming the merge is lossless.
- `CanonicalProgram`'s only production-code constructor is `unifyAxes` itself
  (`Structural.lean:278-279`); its `.ctx` field is read nowhere downstream (confirmed by
  grepping every `.ctx` occurrence in `LeanNCD/`) — every phase from `lowerArith` through
  `schedule` only ever destructures-and-rebuilds it verbatim, and `route`/`routeCore`/`buildStep`
  never reference it at all.
- `RouteSpec.lean` (908 lines) has zero references to `CanonicalProgram`, `.ctx`, `Context`, or
  `unifyAxes` — confirmed by direct grep. Its 8 theorems are universally quantified over
  `ScheduledProgram` and only ever project `.stmts`/`.extNames`. This change does not cross into
  RouteSpec's definitional-unfold proof-repair territory the way `dedupByUid`/`buildStep` would.

## Changes, by file

1. **`LeanNCD/DSL/Pipeline/Types.lean`** — delete `CanonicalProgram` (lines 31-36); drop
   `ctx : Context AxisSpec` from `LoweredProgram`, `ScanProgram`, `LinearProgram`,
   `ScheduledProgram`; drop `import LeanNCD.Exec.Context`.
2. **`LeanNCD/DSL/Pipeline/Structural.lean`** — delete the `## The unifyAxes phase` doc block,
   `collectAxisNameUID`, and `unifyAxes`; retype `lowerArith : CanonicalProgram → FreshM
   LoweredProgram` to `lowerArith : ResolvedProgram → FreshM LoweredProgram`; drop the
   `ctx := ...` pass-through in `lowerArith`'s and `finalizeScans`'s return values.
3. **`LeanNCD/DSL/Pipeline/Lowering.lean`** — drop the `ctx := ...` pass-through in `splitNonlins`
   (`:62`) and `schedule` (`:165`).
4. **`LeanNCD/DSL/Compile.lean`** — remove the `unifyAxes` step from `TLProgram.compile`'s
   do-chain and from `TLProgram.compileToScheduled`'s `>=>` chain; `checkDtypes` feeds
   `lowerArith` directly.
5. **`LeanNCD/Exec/Context.lean`** — delete the file.
6. **`LeanNCD.lean`** — drop the `import LeanNCD.Exec.Context` line and its module-doc mention of
   the "§7.4 `Context` coequalizer".
7. **Tests:**
   - Delete `test/Exec/ContextTest.lean` and `test/Exec/ContextSpec.lean` outright (they test
     only `Context.lean`'s now-deleted API).
   - `test/DSL/Pipeline/StructuralTest.lean`: delete the `unifyAxes`-merge test (lines 46-67,
     "Two `k` axes with DIFFERENT uids... must unify"). No coverage lost — the invariant it
     exercised (same name ⇒ same uid) is already independently asserted by the `assignUIDs` test
     at lines 4-20 ("i,j,k → exactly 3 distinct uids"). Update the 3 `CanonicalProgram` literals
     (now `ResolvedProgram`, dropping `ctx :=`) and 2 `LoweredProgram` literals (dropping
     `ctx :=`) that feed `lowerArith`/`finalizeScans` directly.
   - `test/DSL/Pipeline/LoweringTest.lean`: drop `ctx := { classes := [] }` from the 7
     `ScanProgram`/`LinearProgram`/`ScheduledProgram` literals (lines 14, 32, 49, 65, 85, 100,
     130, 147).
8. **`papers/restructure_suggestions.md`** — mark E11 ✅ DONE with the finding and outcome (same
   convention as E6/Spike 1); update the checkpoint header to point at the next open item.

## Verification

- `lake build` green (no new sorries; sorry count in the default build unchanged — this touches
  no proof, only `Structural.lean`/`Lowering.lean`/`Compile.lean`/`Types.lean`'s executable
  definitions and `Exec/Context.lean`, which carries no theorems).
- Full test suite green, including the edited `StructuralTest.lean`/`LoweringTest.lean` and the
  ~130 existing hand-computed examples plus the E6 property oracles (regression net for exactly
  this kind of pipeline-shape change).
- No behavior change is possible in principle: every value `unifyAxes` ever produced in the
  compiled pipeline was structurally the identity substitution, so no downstream phase's output
  can differ.

## Risks / notes

- **Confirm no other stray reference.** Before deleting, re-grep for `unifyAxes`/`Context`/
  `EqClass`/`CanonicalProgram` across the whole tree (not just the files listed above) as a final
  check — this design is based on a thorough but not exhaustive-by-tooling search.
- **`ResolvedProgram` reuse.** `lowerArith` now takes `ResolvedProgram` — double check no other
  caller of `lowerArith` (if any exists beyond `Compile.lean` and tests) assumes the
  `CanonicalProgram` type name specifically (e.g. via an explicit type ascription) rather than
  structural fields.
- **Doc drift.** Several doc comments (`Structural.lean:23`, `LeanNCD.lean:40`) mention "by uid"
  traversal or the "§7.4 Context coequalizer" descriptively; sweep for these while editing so the
  post-change doc comments don't describe deleted machinery.
