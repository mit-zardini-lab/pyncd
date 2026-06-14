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
| `LeanNCD/Seam/Adapter.lean` | `MonoidalCategory` coherences (10) | §11 strictification; `tensorHom_def`, `id_tensorHom_id`, `tensorHom_comp_tensorHom`, `whiskerLeft_id`, `id_whiskerRight`, `associator_naturality`, `leftUnitor_naturality`, `rightUnitor_naturality`, `pentagon`, `triangle` (eqToIso data, coherences deferred) |
| `LeanNCD/Seam/Adapter.lean` | `SymmetricCategory` braiding involutivity (2) | `hom_inv_id`/`inv_hom_id`; needs swap involutivity (not a current `ColoredPROP` axiom) |

The 5 `SymmetricCategory` coherences other than the two involutivity fields
(`braiding_naturality_right`/`braiding_naturality_left` etc.) are closed by `aesop_cat` and are NOT
sorries. `Graded.lean` is fully sorry-free. Milestone B adds 13 `sorry`s (1 + 12); the whole library
carries 18 total (5 from Milestone A).
