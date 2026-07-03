# Milestone A — intentional `sorry` inventory

These are SIGNATURE placeholders for data/proof fields the design doc (`papers/leanncd.md` §2.2/§2.3)
elides with `…`. They are discharged in later milestones (B+), not Milestone A.

`St.swap` and `St.elemental` have since been proved (see Milestone G notes below).

| File | Field | Section | Status |
| --- | --- | --- | --- |
| `LeanNCD/Base/Br.lean` | `Br.swap` | §2.3 | **PROVED** (re-presentation) — `swap` is now the first-class `braid` generator of the free strict SMC |
| `LeanNCD/Base/Br.lean` | `Br.tensorHom` | §2.3 | **PROVED** (re-presentation) — `tensorHom` is now the first-class `tensor` generator (no longer `extendRight`/`extendLeft`) |
| `LeanNCD/Base/Br.lean` | `Br.elemental` | §2.3 | **REDUCED** — the quotient→raw-syntax reduction is proved sorry-free; the lone obligation is `brCancelPoint` (point left-cancellation = the normal-form milestone, see Keystone note) |

## Milestone G — `ColoredPROP` morphism-level symmetric-monoidal laws

`ColoredPROP` now carries three genuine `Prop` laws — `tensorHom_id`, `tensorHom_comp`
(bifunctoriality of `tensorHom`) and `swap_swap` (`swap` is an involution). `St` now proves all
five of these fields (plus `swap` and `elemental`) sorry-free. `Br`'s laws remain stubbed because
`Br.tensorHom`/`Br.swap` are themselves still stubbed.

`St.lean` is now **fully sorry-free**: `swap`, `elemental`, `tensorHom_id`, `tensorHom_comp`, and
`swap_swap` are all proved. The remaining open items are all in `Br`.

| File | Field | Note |
| --- | --- | --- |
| `LeanNCD/Base/Br.lean` | `Br.tensorHom_id` | **PROVED** (re-presentation) — `Quot.sound (Rel.tensor_id)` |
| `LeanNCD/Base/Br.lean` | `Br.tensorHom_comp` | **PROVED** (re-presentation) — `Quot.sound (Rel.interchange)` |
| `LeanNCD/Base/Br.lean` | `Br.swap_swap` | **PROVED** (re-presentation) — `Quot.sound (Rel.braid_involution)` |

### Keystone re-presentation (2026-06-19) — `Br` as a free strict symmetric monoidal category

The free-list `BrMorph` (`nil`/`cons`) could **never** satisfy `swap_swap`/`tensorHom_comp`
(`swap a b ; swap b a` is a 2-element list ≠ `nil` by `noConfusion`), and §2.3 rules out a single
concrete record. `Br` is now re-presented as the **free strict symmetric monoidal category** on
`BrBase` generators: `Hom` (raw syntax: `id`/`gen`/`comp`/`tensor`/`braid`) quotiented by `Rel` (the
congruence of the category laws + bifunctor `tensor_id`/interchange + `braid` involution). `BrMorph`
is the quotient; `swap = braid`, `tensorHom = tensor` are first-class, so they no longer decompose
into extended swaps. **`swap_swap`, `tensorHom_comp`, `tensorHom_id`** now hold by `Quot.sound`
(`#print axioms` ⇒ `[propext, Quot.sound]`, no `sorryAx`).

**Cost:** `Br.elemental` regressed from PROVED (cons-injectivity) to a single isolated obligation.
The quotient→raw-syntax **reduction is proved sorry-free** (`elemental` ⟸ `brCancelPoint` via
`Quotient.inductionOn₂` + a manufactured point `brPoint X : BrBase [] X`); ALL the hard content is
localized in `brCancelPoint` (point left-cancellation on `Hom`). That lemma IS true (a generator
participates in no `Rel` constructor, so a leading `gen (brPoint X)` cannot be rewritten away) but
needs a `Rel`-respecting normal form.

**Planned route — NbE / initiality (skeleton validated 2026-06-21).** Interpret `Hom` into a
concrete canonical model `N a b` of free-strict-SMC morphisms (typed string diagrams: a
generator-node set + a color-preserving wiring bijection, up to graph iso), with `eval : Hom → N`,
`quote : N → Hom`, and discharge `brCancelPoint` from three lemmas: `sound` (`Rel f g → eval f =
eval g`), `section_` (`Rel f (quote (eval f))`), and `eval_point_injective` (the empty-domain point
is `N`'s unique input-less node, deletable). `sound` dissolves the congruence closure
(`trans`/`comp_congr` become `Eq`); `section_` — any two node-set sequentializations are `Rel`-equal,
i.e. the `interchange` + braid-naturality content — is the gating several-hundred-line bulk,
alongside defining `N`. In `N`, `interchange`/`∘`-assoc/unit are structural and ALL braid laws
(involution, naturality, hexagon) become `Equiv` facts on the wiring. The reduction
`brCancelPoint ⟸ sound + section_ + eval_point_injective` is machine-checked; the three lemmas +
`N`/`eval`/`quote` remain `sorry`.

**Correction (supersedes earlier framing).** Direct induction on the `Rel` derivation does NOT work,
and the wall is NOT `interchange` per se — it is the `trans` (congruence-closure) case: `trans`
injects an unconstrained intermediate term that must itself be point-prefixed to chain the IHs, which
is circular. The empty domain of `brPoint X` only tames base cases (the `interchange` base case, for
one, is vacuous — a `tensor`-headed term cannot equal a point-prefixed `comp`); it is not a shortcut.
(The old `scratchpad/Spike.lean` is gone; its `assoc`/`braid_involution` fragments were never the
hard part.) Net `Br` sorries: 2 → 1 (`swap_swap` + `tensorHom_comp` cleared; `elemental` →
`brCancelPoint`).

**Symmetry coherences (Adapter):** `braid` naturality is now a `Rel` constructor (`braid_natural`)
and a `ColoredPROP` field (`swap_natural`), discharging `Seam/Adapter.lean`'s two
`braiding_naturality_*` sorries (see Milestone B). The *hexagon* (2 sorries) is still NOT in `Rel` —
it carries `eqToHom` casts across `tensor_assoc`, so it needs the cast calculus; deferred with the
3 unitor/associator naturalities.

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

The seam coherence work proceeds by adding the missing symmetric-monoidal axioms as `ColoredPROP`
fields (`HEq` for the cross-type ones) and discharging the generic Adapter goals from them via the
`natOfHEq` bridge (naturality across `eqToHom` ⟺ `HEq`, `conj_eqToHom_iff_heq`).

**PROVED (2026-06-19 / -20):**
- `braiding_naturality_right`/`_left` — field `ColoredPROP.swap_natural`; sorry-free for `St`
  (matrix identity) and `Br` (`Rel.braid_natural` → `Quot.sound`).
- `leftUnitor_naturality` — field `tensorHom_unit_l` (`HEq (tensorHom (id unit) f) f`); sorry-free
  for `St` (same-type, `[]++a = a` defeq) and `Br` (`Rel.tensor_unitl`).
- `rightUnitor_naturality` — field `tensorHom_unit_r`; sorry-free for `St` (genuine cross-type `HEq`
  across `append_nil`, via `StMat.hext`) and `Br` (`Rel.tensor_unitr` cast-constructor).

- `associator_naturality` — **PROVED sorry-free** (2026-06-20). Field `tensorHom_assoc`
  (`HEq ((f⊗g)⊗h) (f⊗(g⊗h))`), proved for `Br` (`Rel.tensor_assoc_coh`) and for `St` (cross-type
  block-diagonal reassociation across `append_assoc`, via private `StMatAux.block_reassoc` /
  `bias_reassoc` + three `Fin.cast`-`addCases` index-bridge lemmas, fed through `StMat.hext`).
  **`St` is fully sorry-free again** (`#print axioms St` = `[propext, Classical.choice, Quot.sound]`).

**PROVED (2026-06-21):**

- `hexagon_forward` / `hexagon_reverse` — discharged by bridging Mathlib projections
  (`.hom`/`.inv`, `≫`, `whiskerRight/Left`) to `ColoredPROP` primitives (`SmCat.coh`,
  `SmallCategory.comp`, `tensorHom`) via local `have` lemmas, simp-rewriting, then
  `rw [← SmallCategory.assoc, ← SmallCategory.assoc]` to convert Mathlib's right-associated
  `≫`-chain to the left-associated form of `swap_hexagon_fwd/rev`, closed by `exact`.
  Key subtlety: `hg : f ≫ g = SmallCategory.comp f g` requires `f : A ⟶ B` (CategoryTheory `⟶`
  notation, not `SmallCategory.hom`) for `rfl` to elaborate; `eqToHom h = SmCat.coh h` requires
  `cases h; rfl` (not `rfl`—they differ in `Eq.rec` construction).

`Seam/Adapter.lean` is now **fully sorry-free**.

`Graded.lean` is fully sorry-free.

## Milestone C — intentional `sorry` inventory

Milestone C adds the §7 Grothendieck split, the §7.2 target actegory / algebra layer, the §8/§9
generic propositions, and the §10.1 flagship `St`/`Br` instance. The mixin class declarations
(`TemporalGraded`/`RouteStructure`/`SymmetryGraded`), the `TargetActegory` class, and the
`Algebra`/`ParaAlgebra` class declarations are **sorry-free**. All `sorry`s below are
`-- SIGNATURE`-annotated. New real code-line `sorry` count: **15** (3 + 1 + 1 + 10).

| File | Field/term | Section | Note |
| --- | --- | --- | --- |
| `LeanNCD/Grothendieck/Split.lean` | `structuralCongruence.instCongruence.comp_left` | §7.1 | **PROVED** — `True` stub is trivially stable under pre-composition |
| `LeanNCD/Grothendieck/Split.lean` | `structuralCongruence.instCongruence.comp_right` | §7.1 | **PROVED** — `True` stub is trivially stable under post-composition |
| `LeanNCD/Grothendieck/Split.lean` | `structuralCongruence.instCongruence.equivalence` | §7.1 | **PROVED** — `True` stub is trivially reflexive/symmetric/transitive |
| `LeanNCD/Grothendieck/Split.lean` | `Dat` | §7.1 / 8.3 | **BODY FILLED** — Unit-constant functor (`obj _ := Unit`, `map _ := 𝟙 Unit`) under the `True` stub; real body awaits real structural congruence |
| `LeanNCD/Grothendieck/Split.lean` | `grothendieck_split` | 8.3 | Prop 8.3: `C ≌ ∫Dat` (structure/data split) |
| `LeanNCD/Algebra/Target.lean` | `TargetActegory StObj (Mat ℝ) ℝ` instance `actV` | §7.2 | appends ℝ-typed dimensions; composition = matrix multiply over ℝ |
| `LeanNCD/Props/Generic.lean` | `scan_catamorphism` | 8.7 | **PROVED** — needed a new `TemporalGraded.restrict_id` coherence field (the `restrict` analogue of `iotaTo_id`); then `rw [restrict_id]; comp_id`. `Props/Generic.lean` is now fully sorry-free. |
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
faithfully. (`St.lean` is now fully sorry-free; the remaining B+/G open items are all in `Br`.)

**UPDATE (2026-07-01):** `realize` is now **sorry-free** — closed by the multi-output `BrBase`
generalization (`2026-06-26-multioutput-impl-plan.md` Phases C/E), not by closing `Br.tensorHom`/
`Br.swap` (those were already sorry-free per Milestone G's re-presentation by the time this landed).
The remaining work was the routing traversal (`poolAt`/`stepPiece`/`finalPiece` over all output
slots), not the categorical combinators. `RouteSpec.lean` and `LeanNCD/Bridge/Realize.lean` are both
fully sorry-free; `compile_wellFormed` (`Agreement.lean`) is proved sorry-free (Phases B–D of the same
plan; axioms `[propext, Classical.choice, Quot.sound]`, verified via `lean_verify`).

**UPDATE (2026-07-03): the §8.2 acset agreement is now FULLY PROVED — all three obligations closed
(`2026-07-01-acset-agreement-impl-plan.md`, Tasks A–E). `Bridge/AcsetCodec.lean` (new),
`Bridge/SBr.lean`, and the `fromThreadedComposed`/`realizeSBr`/`realize_fromThreadedComposed_agree`/
`agree_dom`/`agree_cod` declarations in `Agreement.lean` are all sorry-free; `lean_verify` on the
agreement ⇒ `[propext, Classical.choice, Quot.sound]` (no `sorryAx`).**
- `fromThreadedComposed` (encode) + `toThreadedComposed` (decode) — a systematic/synthetic
  `ThreadedComposed ↔ Acset.SBrInstance` codec (`AcsetCodec.lean`); axis names stored in a dedicated
  `.normAxis` slot (two-slot encoding) so the round trip is unconditional.
- `toThreadedComposed_fromThreadedComposed` (Task C) — the round trip, under
  `(h : WellFormed) (hs : WellShaped)`. `WellShaped` = `routing.length = steps.length` + per-step
  reindexing-matrix shape (invariants `WellFormed` doesn't carry; every compiled program satisfies
  them). Keystones: the general isolation lemma `filter_flatten_tagged`, `decodeWeaveAt_from`,
  `decodeReindexing_from`.
- `realizeSBr` (Task D) — decode + replay `realize`, gated on `WellFormed`.
- `realize_fromThreadedComposed_agree` (Task E) — reduces to the round trip + proof irrelevance;
  carries `(hs : WellShaped)`, as do `agree_dom`/`agree_cod`.

**(historical, now resolved)** the remaining 2 literal `sorry`s in `Agreement.lean` + 1 in `Bridge/SBr.lean`:
- `realizeSBr` (SBrInstance→BrMorph) body — DONE (Task D).
- `fromThreadedComposed` — the §8.2 acset extraction algorithm (acset.md). DONE (Task A, `AcsetCodec`).
- `realize_fromThreadedComposed_agree` (full Σ-equality of the two realized morphisms) — DONE (Task E).
  `agree_dom`/`agree_cod` follow by `congr_arg`.

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

  **RECORDED OBSTRUCTION (why the `Mat ℝ` instance stays `sorry`, not a proof effort).** A
  *faithful* `actV : (FGModuleCat ℝ × Stᵒᵖ) ⥤ FGModuleCat ℝ` — the intended "append ℝ-typed
  dimensions," i.e. `dim(actV(M,P)) = dim(M) · |P|` with `|P|` the product of `P`'s axis sizes —
  is **mathematically impossible** in `FGModuleCat ℝ`, on two independent grounds:
  1. **Symbolic sizes have no finite dimension.** Axis sizes live in `Numeric = MvPolynomial String ℕ`
     (symbolic, noncomputable); `FGModuleCat ℝ` objects carry a *concrete* finite dimension. There is
     no ℝ-vector space of symbolic dimension `|P|`. (Same root cause as "Numeric is noncomputable",
     here as "no finite-dim module of a symbolic dimension".)
  2. **`δ_V` forces dimension-preservation.** In `FGModuleCat ℝ`, `X ≅ Y ⟺ dim X = dim Y` and `⊗`
     *multiplies* dimension. `δ_V : actV(X⊗Y,P) ≅ actV(X,P) ⊗ actV(Y,P)` then demands
     `dim∘actV` be multiplicative in the `V`-variable: a dimension-scaling lift `f(P)` must satisfy
     `f(P) = f(P)²`, i.e. `f(P) = 1`. So any `actV` obeying `δ_V` preserves dimension; the only
     ℕ-valued non-identity solutions are tensor-powers `M ↦ M^⊗(cᵏ)` keyed on *axis count* — unfaithful
     artifacts, not size-product batching.
  The fallback `actV = id` is **not** a benign stub either: with the intended evaluator
  `F(X) =` free module on `X`'s coordinate space, `dim F(X⊛P) = dim(X)·|P| ≠ dim F(X)`, so the
  `Algebra` equivariance `F(act(X,P)) ≅ actV(F X, P)` would FAIL for `actV = id`. A faithful ℝ-valued
  actegory therefore requires **concrete** (`Nat`) sizes — exactly the regime of the Milestone I
  `DenseTensor` evaluator — and does not belong in the symbolic math tower. Closing these 8 `sorry`s
  is a target-category **redesign decision**, not a proof; deferred deliberately.
- **R = Bool target obligation** (`Target.lean` note): the predicate (∧/∃) target needs the
  `(∨, ∧)` Boolean *semiring* (∨ has no additive inverse → genuinely not a ring). The wrinkle is
  semantic, NOT typechecking: Mathlib's `Bool` *type* is the Boolean *ring* (`+` = XOR, so
  `true + true = false`, and `FGModuleCat Bool` DOES elaborate) — but reusing it computes over XOR,
  the wrong addition for `∃`. So the predicate target needs a separate `TargetActegory _ V Bool`
  over a relations / `(∨,∧)`-semimodule `V` (with `Bool` carrying the `(∨,∧)` semiring, not XOR) —
  a deferred obligation.
- **Algebra** (`Algebra.lean`): `F` is strong symmetric monoidal via Mathlib `Functor.Braided`
  (genuine pentagon/unitor/invertibility + braiding laws), `[SymmetricCategory V]`; plus the
  `equivar` coherence laws `equivar_nat`/`equivar_υ`/`equivar_α`/`equivar_δ` (μ-mediated) and
  `F_ev_p` (F preserves the §4.1 evaluation) — all real non-vacuous equations (class fields, no sorry).
- **ParaAlgebra** (`Algebra.lean`): the lightweight `Para(C)→Para(V)` action `paraMap` + its
  μ-mediated `paraMap_eq` + the `weightTie` reparameterization-2-cell law (real equations).
- **Flagship + propositions** (`Construct.lean`): `instAlgebraBrMatR : Algebra StObj BrObj (Mat ℝ) ℝ`
  (Br evaluates into ℝ-modules; all 8 fields `sorry`); `construct_correspondence` (the `F_ev_p` law
  specialized — F realizes construct()'s ℝ-valued read, `sorry`); `semiring_choice_split` (Σ/×-vs-∃/∧
  as the choice of R, a PROXY via additive idempotency `((1:ℝ)+1 ≠ 1) ∧ (true || true = true)` —
  ℝ's `+` non-idempotent (Σ) vs the `(∨,∧)` semiring's `∨`/`||` idempotent (∃); uses `||` NOT Bool's
  XOR `+`. This is PROVEN (`⟨by norm_num, by decide⟩`), not `sorry`. A class-scoped `actV` comparison
  awaits the deferred Bool target).

Out of scope (later): the executable Algebra *interpreter* (parse→compile→evaluate→numbers; retires
the dtype obstruction); the B+/G `Br`/`St` coherence proofs.

## Milestone I — executable evaluator (zero sorries)

`LeanNCD/Eval/` — the reference `Float`-tensor evaluator for the tensor-logic DSL. Fully
executable, **ZERO `sorry`** (`grep -rn sorry LeanNCD/Eval/` is empty); the eight modules
(`Tensor`/`Shape`/`Gather`/`Contract`/`Nonlin`/`Scatter`/`Scan`/`Eval`) are wired into the
`Tests` lean_lib and exercised by per-module unit tests plus the eleven-example integration
test (`test/Eval/EvalExamplesTest.lean`).

- **`DenseTensor`** (`Tensor.lean`): a row-major `{ shape : List Nat, data : Array Float }`
  Float reference interpreter — `get!`/`set!`/`zeros`/`ofFn`/`allCoords`/`approxEq`.
- **`TLProgram.eval`** (`Eval.lean`): the entry point. It runs `compileToScheduled` and
  evaluates the **pre-route `ScheduledProgram`** — NOT the routed `ThreadedComposed`, which is
  lossy for scans (it keeps only the representative recurrence step, dropping the base case and
  coupled bodies). The `ScheduledProgram` retains the full scan `base`/`recur` stmt lists.
- **dtype-driven combine** (`Contract.lean`): the contraction semiring is read from the decls —
  a `predicate` output contracts in `(∧, ∃)` = `(min, max)` on 0/1 Floats, every other output in
  the ℝ default `(×, Σ)`. Reading the dtype from the decls sidesteps the open `BrBaseP` dtype gap
  (the routed presentation defaults `dtype := .reals`).
- **All ELEVEN example programs compute** with numeric assertions (the 5 §12.1 examples —
  matmul, masked attention, strided conv, upsample, coupled scan — plus the 2 predicate examples
  — masked aggregation, band mask — plus look-back, outer product, contraction+relu, normalize).
- **Affine-output size inference** (`Shape.lean`): `inferAxisSizes` was extended (Milestone I) to
  size output axes that appear ONLY in affine read positions, via the standard "valid"/largest-
  in-range rule `size(a) = ⌊(d-1 - c0 - Σ_{k≠a} cₖ(sizeₖ-1)) / cₐ⌋ + 1`, solved one unknown axis
  per position over a fixpoint. This is what lets the strided-conv (output `i,j`) and look-back
  (`Y[i] := X[i-1]` ⇒ length 5, `Y[0]=0` pad) examples size their purely-affine output axes; for
  a bare axis it reduces to `size(a) = d`, and the conflicting-bare-read check is preserved.
- Scan iteration count `L` is a runtime parameter not present in the program text; the coupled-scan
  example supplies it as the time-extent of the `G` state buffer passed in `inputs`.

**Out of scope (error):** `scanPre`/`recurMorphism` evaluation (the §12.4 escape hatches) and
symbolic-size evaluation — `Numeric = MvPolynomial String ℕ` is noncomputable, so a `SizeExpr`
that does not reduce to a concrete `Nat` from the inputs cannot be evaluated. Both raise an
`EvalError`.
