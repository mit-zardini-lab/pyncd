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
