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
