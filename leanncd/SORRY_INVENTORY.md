# Milestone A — intentional `sorry` inventory

These are SIGNATURE placeholders for data/proof fields the design doc (`papers/leanncd.md` §2.2/§2.3)
elides with `…`. They are discharged in later milestones (B+), not Milestone A.

| File | Field | Section |
| --- | --- | --- |
| `LeanNCD/Base/St.lean` | `St.swap` | §2.2 |
| `LeanNCD/Base/St.lean` | `St.elemental` | §2.2 |
| `LeanNCD/Base/Br.lean` | `Br.swap` | §2.3 |
| `LeanNCD/Base/Br.lean` | `Br.tensorHom` | §2.3 |
| `LeanNCD/Base/Br.lean` | `Br.elemental` | §2.3 |

## Milestone G — `ColoredPROP` morphism-level symmetric-monoidal laws

`ColoredPROP` now carries three genuine `Prop` laws — `tensorHom_id`, `tensorHom_comp`
(bifunctoriality of `tensorHom`) and `swap_swap` (`swap` is an involution). The `St`/`Br`
instances supply them as `-- SIGNATURE` `sorry`s, consistent with the existing `swap`/`tensorHom`/
`elemental` deferrals (St's would need Matrix-block algebra; Br's `tensorHom`/`swap` are themselves
still stubbed).

| File | Field | Note |
| --- | --- | --- |
| `LeanNCD/Base/St.lean` | `St.tensorHom_id` | Matrix-block: `fromBlocks 1 0 0 1` reindexes to `1` |
| `LeanNCD/Base/St.lean` | `St.tensorHom_comp` | Matrix-block interchange |
| `LeanNCD/Base/St.lean` | `St.swap_swap` | swap permutation is an involution (depends on stubbed `St.swap`) |
| `LeanNCD/Base/Br.lean` | `Br.tensorHom_id` | depends on stubbed `Br.tensorHom` |
| `LeanNCD/Base/Br.lean` | `Br.tensorHom_comp` | depends on stubbed `Br.tensorHom` |
| `LeanNCD/Base/Br.lean` | `Br.swap_swap` | depends on stubbed `Br.swap` |

All category and strictness laws (`id_comp`, `comp_id`, `assoc`, `tensor_assoc`, `tensor_unit_l`,
`tensor_unit_r`) for both `St` and `Br` are proved sorry-free (verified via `#print axioms`).
Note: `St.tensorHom` was fully implemented (block-diagonal via `Matrix.fromBlocks`), so it is NOT
in this list — only `Br.tensorHom` remains stubbed.

## Milestone B — intentional `sorry` inventory

Seam strictification (§11) coherences + Prop 8.2, all `-- SIGNATURE`-annotated. The `Category`
instance, the `DGradedColoredPROP` class, `sh_star`, `ev_p`, and `ev_p_naturality` (Eq. 3, via the
`υ_nat` law) are all sorry-free.

| File | Field/lemma | Note |
| --- | --- | --- |
| `LeanNCD/Core/Weave.lean` | `weave_unique` | Prop 8.2; from `ColoredPROP.elemental` + `broadcast_gen` (proof milestone) |

Milestone G discharged most of the seam coherences using the new `ColoredPROP` laws. The
`MonoidalCategory` instance now proves sorry-free: `tensorHom_def`, `id_tensorHom_id`,
`tensorHom_comp_tensorHom`, `whiskerLeft_id`, `id_whiskerRight` (from `tensorHom_id`/`tensorHom_comp`)
and `pentagon`, `triangle` (from functoriality-on-objects of `tensorHom`, via the private
`tensorHom_eqToHom_id`/`tensorHom_id_eqToHom` helpers). The `SymmetricCategory` braiding
`hom_inv_id`/`inv_hom_id` and `symmetry` are proved from `swap_swap`.

The seam coherences that **remain** `-- SIGNATURE` `sorry` (they are independent symmetric-monoidal
coherences of `tensorHom`/`swap` that the bifunctor + `swap_swap` laws do **not** imply):

| File | Field | Note |
| --- | --- | --- |
| `LeanNCD/Seam/Adapter.lean` | `MonoidalCategory.associator_naturality` | tensorHom-associativity coherence |
| `LeanNCD/Seam/Adapter.lean` | `MonoidalCategory.leftUnitor_naturality` | tensorHom left-unit coherence |
| `LeanNCD/Seam/Adapter.lean` | `MonoidalCategory.rightUnitor_naturality` | tensorHom right-unit coherence |
| `LeanNCD/Seam/Adapter.lean` | `SymmetricCategory.braiding_naturality_right` | naturality of swap |
| `LeanNCD/Seam/Adapter.lean` | `SymmetricCategory.braiding_naturality_left` | naturality of swap |
| `LeanNCD/Seam/Adapter.lean` | `SymmetricCategory.hexagon_forward` | hexagon identity for swap |
| `LeanNCD/Seam/Adapter.lean` | `SymmetricCategory.hexagon_reverse` | hexagon identity for swap |

Note: `braiding_naturality_*`/`hexagon_*` previously "closed by `aesop_cat`" only because the
then-`sorry` braiding inverse let the discharger ride on `sorry`; with the inverse now a real proof
(`swap_swap`) they are honestly deferred. `Graded.lean` is fully sorry-free.

## Milestone C — intentional `sorry` inventory

Milestone C adds the §7 Grothendieck split, the §7.2 target actegory / algebra layer, the §8/§9
generic propositions, and the §10.1 flagship `St`/`Br` instance. The mixin class declarations
(`TemporalGraded`/`RouteStructure`/`SymmetryGraded`), the `TargetActegory` class, and the
`Algebra`/`ParaAlgebra` class declarations are **sorry-free**. All `sorry`s below are
`-- SIGNATURE`-annotated. New real code-line `sorry` count: **15** (3 + 1 + 1 + 10).

| File | Field/term | Section | Note |
| --- | --- | --- | --- |
| `LeanNCD/Grothendieck/Split.lean` | `structuralCongruence.instCongruence.comp_left` | §7.1 | stable under precomposition |
| `LeanNCD/Grothendieck/Split.lean` | `structuralCongruence.instCongruence.comp_right` | §7.1 | stable under postcomposition |
| `LeanNCD/Grothendieck/Split.lean` | `structuralCongruence.instCongruence.equivalence` | §7.1 | reflexive + symmetric + transitive |
| `LeanNCD/Grothendieck/Split.lean` | `Dat` | §7.1 / 8.3 | data functor (size-assignments over the structural skeleton) |
| `LeanNCD/Grothendieck/Split.lean` | `grothendieck_split` | 8.3 | Prop 8.3: `C ≌ ∫Dat` (structure/data split) |
| `LeanNCD/Algebra/Target.lean` | `TargetActegory StObj (Mat ℝ) ℝ` instance `actV` | §7.2 | appends ℝ-typed dimensions; composition = matrix multiply over ℝ |
| `LeanNCD/Props/Generic.lean` | `scan_catamorphism` | 8.7 | scan-as-catamorphism (iterate stable under reflexive-prefix restriction) |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.act` | §10.1 | batch lift + reindexing |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.δ` | §10.1 | `[X ⊗ Y, P] ≅ [X,P] ⊗ [Y,P]` |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.δ0` | §10.1 | `[I, P] ≅ I` |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.υ` | §10.1 | `[X, I_St] ≅ X` |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.α` | §10.1 | `[[X,P],Q] ≅ [X, Q ⊗ P]` |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.sh_act` | §10.1 | (Sh-⊛) |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.act_unit_assoc` | §10.1 | actegory triangle + pentagon |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.υ_nat` | §10.1 | unitor naturality |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.dist_coh` | §10.1 | δ/δ0 naturality + interchange |
| `LeanNCD/Instances/StBr.lean` | `instDGradedStBr.broadcast_gen` | §10.1 | every Br morphism factors `lam ; [f,P] ; ρ` |

Notes:

- `Props/Generic.lean` is otherwise sorry-free: `lift_functorial` (8.1) is **PROVED** sorry-free
  (`#print axioms` ⇒ `[propext]` only), and `scan_batches` (8.8) is a **sorry-free re-export** of
  `TemporalGraded.lift_fold_dist`; `weave_subsingleton` (8.2) re-exports `weave_unique`. Only
  `scan_catamorphism` (8.7) is deferred.
- `Instances/StBr.lean`: the `sh` field is **concrete** (`fun a => a.shape`, sorry-free); the 10
  fields above (`act`/`δ`/`δ0`/`υ`/`α`/`sh_act`/`act_unit_assoc`/`υ_nat`/`dist_coh`/`broadcast_gen`)
  are deferred §10.1 content.
- §9 props 8.3/8.4/8.5/8.6: 8.3 is `grothendieck_split` (in `Grothendieck/Split.lean`, not restated);
  8.4 (equivariance), 8.5, and 8.6 (obstruction species) are **OMITTED** — not vacuous, but no class
  field carries faithful content to state yet (EM-machinery / pushout / obstruction datum gated for
  later milestones). They are deliberately not stubbed with a `True`.

## Milestone D — executable seam (zero sorries)

`LeanNCD/Exec/` — the executable side of the §7.4 proposition/computation seam — is **fully
discharged with no `sorry`** (the first fully-executable milestone):

- `Uid.lean`: `UID`/`UData`/`CompileError`/`FreshM`/`freshUData` (Lean core only).
- `Traversable.lean`: `WithUID`, `TermTraversable` class (Lean core only; real per-type instances are Milestone E).
- `Context.lean`: `EqClass`, `Context`, `Context.merge` (largest-UID canonical), `Context.apply`
  (generalized to `(ctx : Context α) (target : β) : β` — the substitution is UID-level, α-independent).

Verified by evaluated `#guard` tests + an LSpec suite (`test/Exec/ContextSpec.lean`). LSpec was
adopted (argumentcomputer/LSpec `main`, rev d3c15b9 — v4.29-targeted but compiles under v4.30; zero
deps, so Mathlib's pins are untouched).

## Milestone E1 — DSL front-end (zero sorries)

`LeanNCD/DSL/` — the tensor-logic DSL front-end (§12) — is **fully executable and `sorry`-free**
(verified: `grep -rn sorry LeanNCD/DSL/` is empty; whole-library `lake build` green). The layer
comprises `SizeExpr` (computable size arithmetic), `Ast` (the typed AST), `Syntax` (the surface
grammar), and `Elab` (value-returning elaborators + the `tlprog!` macro). The five §12.1 example
programs parse via `tlprog!{}` (`test/DSL/ParseExamplesTest.lean`). All eight DSL test modules
(`SizeExprTest`, `AstTest`, `SyntaxTest`, `ParseLayer1Test`, `ParseLayer34Test`, `ParseNaryTest`,
`ParseProgramTest`, `ParseExamplesTest`) elaborate under `lake build` (their `run_cmd`/`#guard`/
`tlprog!` checks fire at elaboration).

Resolved decisions / deviations (feed the §12 doc-consistency pass and the E2 plan):

- `SizeExpr` (computable) replaces `Numeric` in the AST; `SizeExpr.toNumeric` is the noncomputable
  proof-side bridge.
- Elaborators are value-returning (`Syntax → MetaM <value>`), not `Expr`-building.
- `Stmt = assign | scatter` only (the `recurMorphism`/`ThreadedComposed` escape hatch deferred to E2).
- `ScatterOpts.fill : Int` (not `Float`, which lacks `DecidableEq`).
- `tl_idx_expr` generalized to general integer-affine reads; products/sums generalized to n-ary
  (left-recursive). Symbolic-coefficient strides (`s*j`) are unsupported (need `SizeExpr`
  coefficients — future).
- Idents read via `eraseMacroScopes.getString!`; the scan-step LHS token is `+1` (write `l +1`
  spaced).
- E1 simplifications for E2: the `num` LHS base-case uses a placeholder iteration-axis name (E2
  resolves); all `name[…] := rhs` parse to `.assign` (E2's `lowerArith` reclassifies scatter).

Review findings:

- **FIXED — unmasked `softmax(…)` / `normalize(…)` over a sum now parse.** The masked
  `softmax "(" "where" …` rule shared a `softmax (` prefix with the bare token, so the parser
  committed to the `where`-variant on seeing `(` and an unmasked `softmax(A[i] + B[i])` failed
  ("expected 'where'"). Resolved by wrapping the distinguishing lookahead in
  `atomic("(" "where")` (`LeanNCD/DSL/Syntax.lean`), which rewinds cleanly when there is no
  `where` so the bare rule wins and `(sum)` parses at `tl_rhs`. Regression tests in
  `test/DSL/ParseLayer34Test.lean`; masked variants unchanged. Doc updated (§12.3).
- `elabTLProgram`'s child router keys on `child.getKind.getString!` (partial on non-string `Name`
  components; safe for the `tl_decl | tl_stmt` alternation, whose kinds are always string-named).

## Milestone E2a — DSL back-end (zero sorries)

`LeanNCD/DSL/Pipeline/` + `Target.lean`/`Traverse.lean`/`Compile.lean` — the §12.4 compile
pipeline — is **fully executable and `sorry`-free** (`grep -rn sorry LeanNCD/DSL/` empty;
`lake build` green). `TLProgram.compile : TLProgram → FreshM ThreadedComposed` is the 8-phase
Kleisli chain (assignUIDs → resolveDecls → unifyAxes → lowerArith → finalizeScans →
splitNonlins → schedule → route), and `tl!{ … } : ThreadedComposed` compiles all five §12.1
examples at elaboration time (`test/DSL/CompileExamplesTest.lean`).

Resolved decisions / deviations from §12.4 (feed the §12.4 doc-consistency pass and E2b):

- **Computable presentation.** `ThreadedComposed`/`BrBaseP`/`StMatP`/`AxisP`/`WeaveSlotP` are
  first-order, `List`-based, `SizeExpr`/`Int`-valued mirrors of the noncomputable math-tower
  `Br`/`St` types (functions → lists; `Numeric` → `SizeExpr`; `Int` coeffs), all
  `deriving DecidableEq, Repr, Lean.ToExpr` so `tl!{}` can embed the result. The bridge to a
  real `BrMorph` is Milestone E2b.
- **assignUIDs** binds by axis NAME (E1 emits `uid := 0`); `freshNonZero` skips the sentinel 0.
- **resolveDecls** is constructive (never throws): undeclared read names are external inputs
  (the §12.1 examples read `W`/`X`/`Q`/`K` without decls); `extNames` = read-not-produced.
- **unifyAxes** is the §7.4 Context coequalizer (largest-UID canonical); identity in the real
  pipeline (assignUIDs already name-binds) — kept for parity with the CSV path.
- **lowerArith** reclassifies affine-LHS assigns → `Stmt.scatter` (conservative injectivity →
  `overlappingScatter`); affine READS are NOT split into separate Slice/Reindex steps but folded
  into the consuming step's `reindexings` at `route` (where St-maps live in `BrBase`, §2.3);
  `auxStmts := #[]`.
- **finalizeScans** groups iterAt/iterNext by iteration-axis UID into (coupled) `Scan` nodes;
  a pre-pass makes each base case adopt a same-named recurrence's iteration axis (E1 emits scan
  base cases with a placeholder iteration-axis name — the deferred E1 simplification, now resolved).
- **schedule** does backward-reachability DCE; output = last-stmt's written name(s) (single-result-
  at-tail; a genuine multi-output-not-at-tail program would need an explicit outputs field).
- **route** detects contracted axes (tiled weaves), builds `reindexings` via integer-affine
  `idxToRow`, and wires inputs to producer steps or the external sentinel (`step = nExternal`).
- **Out of scope (E2b/later):** the noncomputable presentation→`BrMorph` bridge + Props 8.x;
  `Stmt.recurMorphism`/`ScanStmt.scanPre`; the `ScanAffine` `O(log N)` fast path; numeric
  evaluation (the Algebra `F : C → V`).

Known limitations (final E2a review):

- The `overlappingScatter` dimension-collapse guard keys on `LHSSlot.affine (.const _)`, but the E1
  parser renders a literal LHS coordinate as `LHSSlot.iterAt {name:=""} 0` (a scan base case), so the
  guard is currently unreachable via surface syntax. Guard logic is correct for the AST shape it
  targets; reachability awaits E1 distinguishing a constant output coordinate from a scan-base index.
- Scatter output weaves mark all affine-LHS read axes `.tiled` (`Stmt.lhsAxes` contributes no retained
  axis for `.affine` slots), e.g. upsample `Out[2*i,2*j]` tiles both `i` and `j`. Internally
  consistent for E2a; the E2b bridge must reconcile scatter output weaves.

## Milestone E2b — presentation→BrMorph bridge (signatures + sorry)

`LeanNCD/Bridge/` realizes the E2a computable presentation into the noncomputable math tower
and states the §8 DSL/CSV agreement. INTENTIONALLY signatures + `sorry` (a math-tower bridge,
like §2–§9), verified by elaboration + `#print axioms`. Builds on the Milestone-B+ `Br.tensorHom`
/`Br.swap` `sorry`s (NOT closed here, per the E2b scope decision).

Sorry-free realizations:
- `realizeAxis`, `realizeStObj`, `realizeWeaveSlot`, `realizeWeaveShape`, `intToCoeff`,
  `realizeStMat`, `realizeBrBaseP` — ALL `#print axioms`-verified `[propext, Classical.choice,
  Quot.sound]` (no `sorryAx`). The dependent `reindexings` field typechecked with the real
  `realizeStMat` term, and (post negative-coeff fix below) `realizeStMat`/`realizeBrBaseP` are
  now fully sorry-free, not merely transitively-sorry.

**RESOLVED — negative-coefficient obstruction.** The original `intToNumeric` `sorry`'d negative
coefficients because `StMat` carried `Numeric = MvPolynomial String ℕ` (ℕ-semiring, no inverses).
Fixed in the math tower: `StMat.coeffs`/`bias` now carry the new `Coeff = MvPolynomial String ℤ`
(`LeanNCD/Base/Numeric.lean`), a signed `CommRing` — the `St` laws (`id_comp`/`comp_id`/`comp_assoc`)
re-prove sorry-free over it (their tactics are CommRing-generic). The size type stays `Numeric = ℕ`
(sizes are non-negative); the fix separates the conflated *size* and *coefficient* roles. The bridge's
`intToCoeff : Int → Coeff := MvPolynomial.C` is sorry-free, so look-back offsets (`X[i-1]`) realize
faithfully. (`St`'s `swap`/`tensorHom`/`elemental` remain the pre-existing B+/G sorries — untouched.)

Named obligations (the remaining 6 `sorry`s):
- `realize` (ThreadedComposed→BrMorph) body, `realizeSBr` (SBrInstance→BrMorph) body — the routed-DAG
  threading rests on `Br.tensorHom`/`Br.swap` (B+). `realize` derives real `dom`/`cod` (first step's
  inputs / last step's outputs); full external-input assembly (walk `routing` for `step = nExternal`
  wires) is the documented obligation.
- `fromThreadedComposed` — the §8.2 acset extraction algorithm (acset.md).
- `realize_fromThreadedComposed_agree` (full Σ-equality of the two realized morphisms), `agree_dom`,
  `agree_cod` — the §8 DSL/CSV agreement; faithful statements, sorry-proved.

Documented choice (NOT a sorry) / DEFERRED feedback:
- `weaveToArrayType` defaults `dtype := .reals`. The E2a presentation (`BrBaseP`/`AxisP`) dropped
  `DType`, so predicate (Bool) outputs are not distinguished from real ones. The consumer of `DType`
  is the §7.5 Algebra evaluation functor (`R = ℝ` vs `R = Bool`), which is not yet built, so nothing
  is currently mis-evaluated; FEEDBACK (fix when §7.5 evaluation / Milestone F lands): `route` should
  thread the `DeclEnv` `tensor`/`predicate` tag + `AxisKind` into a dtype/op-semiring field on `BrBaseP`.

Out of scope (later): closing the B+ `Br.tensorHom`/`swap`/`elemental` + G coherences; `recurMorphism`
/`scanPre`; `ScanAffine` fast path; the §7.5 Algebra evaluation functor.

## Milestone F — acset CSV interop (zero sorries)

`LeanNCD/Acset/` is the CSV path of §8 — fully executable and `sorry`-free (`grep -rn sorry
LeanNCD/Acset/` empty; `lake build` green). `Acset.SBrInstance` is the full-fidelity mirror of
the Python `acset/instances.py:SBrInstance` (AxisType/AxisUID, OpTag (10), DataTag (3), the
12-field ArrayRow, 5 tables); sizes are `SizeExpr`, coefficients `Int` (so it is computable,
unlike a `Numeric`-sized version). `Acset.Csv` is the field/row codec; `Acset.Io` is
`writeSBr`/`readSBr`.

Validation:
- `readSBr (writeSBr inst) = .ok inst` (Lean round-trip; `test/Acset/IoTest.lean`).
- **byte-for-byte cross-check against a Python `write_sbr` fixture** (`test/Acset/FixtureTest.lean`
  + `test/Acset/fixtures/`, generated by `acset.csv_io.write_sbr` on
  `tests/test_acset_csv.py:_two_equation_sbr()`): Lean's `readSBr` parses the exact Python CSVs and
  `writeSBr` reproduces them identically (CRLF terminators, exact column order, all encodings). This
  is the genuine cross-language interop proof.

Decisions: CRLF (`\r\n`) line terminator matching Python `csv`; no-quoting CSV (the acset data has no
embedded commas/quotes — documented assumption); sizes only `.lit`/`.var` (compound never serializes);
coeffs always integer. F's `Acset.SBrInstance` SUPERSEDES E2b's minimal placeholder — `realizeSBr`/
`fromThreadedComposed`/the agreement Props (Bridge/) now use it (their E2b `sorry`s unchanged).

## Milestone E2c — recurMorphism/scanPre + ScanAffine (zero sorries)

DSL feature-completion on the E2a pipeline — fully executable, `sorry`-free.
- **recurMorphism escape hatch:** `Stmt.recurMorphism : String → AxisSpec → ThreadedComposed → Stmt`
  (programmatic-only — no `tlprog!` surface syntax). It flows through the pipeline and `finalizeScans`
  turns it into `ScanStmt.scanPre`; `route` emits a single `op="scan_pre"` `BrBaseP` step and VALIDATES
  the pre-built morphism is non-empty (`CompileError.shapeMismatch` on empty `steps`). **Consistent-collapse:**
  the pre-built `ThreadedComposed` is validated but NOT embedded in the flat routed output — symmetric with
  how regular scans already collapse their body (deep scan-body embedding remains a future refinement).
  recurMorphism reads are not introspected (`readNames := []`).
- **ScanAffine:** a scan whose recurrence has no nonlinearity (Prop 8.7, associative/parallel-prefix) is
  detected in `finalizeScans` (BEFORE `splitNonlins` lifts nonlinearities out) via the `isAffine` flag on
  `ScanStmt.scan`, and `route` emits `op="scan_affine"` (vs `"scan"`). The §12.1 coupled scan (relu
  recurrence) stays `op="scan"`. So routed scan steps now carry `op ∈ {scan, scan_affine, scan_pre}`.

## Milestone H — §7.5 Algebra & construct() (signatures + sorry)

`LeanNCD/Algebra/` — the §7.5 Algebra layer — formalized as signatures + `sorry` (math-tower
style, like §2–§9), verified by elaboration + `#print axioms`. Builds ON (does not close) the
B+/G `Br`/`St` coherence sorries.

- **TargetActegory** (`Target.lean`): full actegory coherences in `V` (`υ_V`/`α_V`/`δ_V`/`δ0_V` +
  triangle/pentagon `act_unit_assoc_V`, `υ_nat_V`, `dist_coh_V`), over `[MonoidalCategory V]`,
  transposed from `DGradedColoredPROP`. The `Mat ℝ = FGModuleCat ℝ` instance fields are `sorry`.
- **R = Bool target obligation** (`Target.lean` note): `Mat Bool` doesn't typecheck (`FGModuleCat`
  needs `CommRing`; `Bool` is `CommSemiring`), so the predicate (∧/∃) target needs a separate
  `TargetActegory _ V Bool` over a relations/Bool-semimodule `V` — a deferred obligation.
- **Algebra** (`Algebra.lean`): `F` is strong symmetric monoidal via Mathlib `Functor.Braided`
  (genuine pentagon/unitor/invertibility + braiding laws), `[SymmetricCategory V]`; plus the
  `equivar` coherence laws `equivar_nat`/`equivar_υ`/`equivar_α`/`equivar_δ` (μ-mediated) and
  `F_ev_p` (F preserves the §4.1 evaluation) — all real non-vacuous equations (class fields, no sorry).
- **ParaAlgebra** (`Algebra.lean`): the lightweight `Para(C)→Para(V)` action `paraMap` + its
  μ-mediated `paraMap_eq` + the `weightTie` reparameterization-2-cell law (real equations).
- **Flagship + propositions** (`Construct.lean`): `instAlgebraBrMatR : Algebra StObj BrObj (Mat ℝ) ℝ`
  (Br evaluates into ℝ-modules; all 8 fields `sorry`); `construct_correspondence` (the `F_ev_p` law
  specialized — F realizes construct()'s ℝ-valued read); `semiring_choice_split` (Σ/×-vs-∃/∧ as the
  choice of R, stated as a PROXY via additive idempotency `(1:ℝ)+1≠1 ∧ true+true=true` — the Bool
  target being a deferred obligation; a class-scoped `actV` comparison awaits the Bool category).

Out of scope (later): the executable Algebra *interpreter* (parse→compile→evaluate→numbers; retires
the dtype obstruction); the B+/G `Br`/`St` coherence proofs.
