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
