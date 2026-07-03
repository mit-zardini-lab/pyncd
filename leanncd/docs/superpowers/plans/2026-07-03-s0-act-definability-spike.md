# Spike S0 — `act` definability (Milestone 0.5)

**Question (from `papers/NaperianTypingIntegrationPlan.md` §0.2 / §5 M0.5):**
Can `act : (BrObj × StObjᵒᵖ) ⥤ BrObj` be built as a `Quotient.lift` over `Br.Hom` respecting
every `Rel` constructor, and how large is that proof?

**Exit criterion:** a skeleton `act` with the `Rel`-respecting obligations enumerated + a size
estimate, **or** a named blocker.

**Verdict: SPLIT.**
- **Morphism level (the literal S0 question) — TRACTABLE.** The `Quotient.lift` is well-posed; its
  `Rel`-respecting obligation decomposes into ~21 cases, almost all of which map to an
  *already-proved* `BrMorph` theorem. Size estimate below.
- **Object level — was a NAMED BLOCKER, now RESOLVED (2026-07-03, Option 1 implemented).** The
  object action `act.obj` could not satisfy the **strict** `sh_act` field (`sh*(X ⊛ P) = sh*(X) ⊗ P`,
  originally a `=` in the class) under the natural per-array broadcast for bundles with ≥2 arrays.
  This was a design decision on the class, and it gated the morphism-level work (you cannot state
  `act.map`'s target objects until `act.obj` is fixed). **The `sh_act` field has been relaxed from
  `=` to a canonical iso `≅`** in `LeanNCD/Core/Graded.lean` (build green, no downstream breakage —
  the field had zero proof consumers). See "Object level" below for details.

**Go/no-go:** S0's morphism half is clear, and the object-level design decision it surfaced is now
made and implemented. Track B's remaining gate is **spike S1** (`jointly_monic` circularity); `act`
itself is a bounded, mechanical build (~400–650 lines) with no remaining design blocker.

---

## Target

`DGradedColoredPROP.act : (C × Dᵒᵖ) ⥤ C` with `C = BrObj`, `D = StObj`
(`LeanNCD/Core/Graded.lean:27`; instance hole at `LeanNCD/Instances/StBr.lean:15`).

Notation (`graded_prop.md` §3.1): `X ⊛ P := act(X,P)`; `[f,P] := act(f, 𝟙_P)` (batch lift,
covariant in `C`); `[X,η] := act(𝟙_X, η)` (reindexing, contravariant in `D`).

A morphism of the product category `BrObj × StObjᵒᵖ` (via the §3 seam adapter) is a pair
`(f, η)` with `f : BrMorph X Y` (a `Quotient (setoidHom X Y)`) and `η : P ⟶ Q` in `StObjᵒᵖ` (a
bare `StMat Q.unop P.unop` record — St morphisms are **not** quotiented). So `act.map` is defined by
`Quotient.lift` on the `f` component with `η` threaded as a parameter.

---

## Object level — the blocker (RESOLVED 2026-07-03, Option 1)

**Status: implemented.** `sh_act` in `LeanNCD/Core/Graded.lean` has been changed from a strict `=`
to a canonical iso `≅`. `lake build` is green (8588 jobs); no downstream breakage. Blast radius was
**zero proof consumers** — a re-trace found `sh_act` referenced only by its own field declaration,
its doc comment, and the (still-`sorry`) instance field in `StBr.lean`; nothing projects it. (An
earlier draft of this report guessed the consumers were "the `ev_p` region and `Props/Generic.lean`"
— that was an overcount: `ev_p` is built from `act.map`/`υ`, and `Props/Generic.lean` uses
`act`/`υ`/`weave_unique`; neither touches `sh_act`.) The analysis that motivated the change follows.

### The natural definition and why it fails a strict `sh_act`

`sh` for `Br` is `fun a => a.shape` (`StBr.lean:14`), and `sh_star` (`Core/Graded.lean:11`) folds
`tensor` (= `List.append` for St) over the color list, so `sh_star X` is the **concatenation of every
array's shape** in the bundle `X : List ArrayType`.

The natural per-array broadcast is
```
liftArray P A := { A with shape := A.shape ++ P.unop }
act.obj (X, P) := X.map (liftArray P)      -- a List.map
```
Then `sh_star (act.obj (X,P)) = ⊕ᵢ (sᵢ ++ P)` = `s₁ ++ P ++ s₂ ++ P ++ …` — it contains **|X|
copies of P**, interleaved.

`sh_act` (`Core/Graded.lean:36`) demands the strict equality
`sh_star (act.obj (X,P)) = sh_star X ⊗ P = (s₁ ++ … ++ sₙ) ++ P` — **one** copy of P, at the end.

- For `|X| = 1`: holds (`s₁ ++ P = s₁ ++ P`). ✓
- For `|X| = 0`: `act.obj ([],P) = []`, `sh_star = []`, but RHS `= [] ++ P = P`. ✗
- For `|X| ≥ 2`: LHS has ≥2 copies of P, RHS has 1 — different lists (different multiset even), so
  **no iso/δ freedom can fix it**: `sh_act` is a strict `=` on this specific object. ✗

Prepending P instead of appending does not help (still |X| copies). Appending P to the *last*
array only makes `sh_act` hold for `|X| ≥ 1` but fails `|X| = 0`, is semantically wrong (P must
broadcast to **all** arrays, not just the last), and breaks the clean per-array `δ`.

### Root cause

`BrObj = List ArrayType` stores each array's **full** shape independently; there is no
shared-axis representation. Broadcasting P into a bundle therefore physically duplicates P into
every array, so `sh_star` (which concatenates) necessarily counts P once per array. The strict
`sh_act` equation is only compatible with a single-array-per-bundle world.

### Resolution options (design owner's call — was the blocker)

1. **✅ CHOSEN & IMPLEMENTED — relax `sh_act` to hold up to the symmetric braiding.** Change the
   class field from `=` to a canonical iso `sh_star (act.obj (X,P)) ≅ sh_star X ⊗ P` (the permutation
   that gathers the |X| copies of P to the end). Most faithful to the broadcast semantics. Cost was
   even smaller than projected: a single-field type change in `Core/Graded.lean` (plus its doc
   comment) — **zero proof consumers** to re-check (see the status note at the top of this section;
   the earlier "`ev_p` region, `Props/Generic.lean`" was an overcount).
2. **Redefine `sh_star` for the graded measurement** so it does not linearly concatenate the
   duplicated P's. Larger conceptual change. Rejected.
3. **Restrict the intended `act` object semantics** (e.g. single-array bundles), which conflicts
   with `δ`'s whole point (`act(X⊗Y,P) ≅ act(X,P) ⊗ act(Y,P)` over multi-array `⊗`). Rejected.

Option 1 keeps the per-array `List.map` action (which makes `δ`/`δ0` essentially definitional) and
turned out to localize the cost to the `sh_act` field alone.

---

## Morphism level — enumerated obligations (tractable)

**Structure.** Factor `act.map (f, η) = [f, P] ; [Y, η]` (batch-lift `f` at the source shape `P`,
then reindex the target bundle `Y` from `P` to `Q`). Only `[f,P]` depends on `f`, so the
`Quotient.lift` well-definedness reduces to well-definedness of the batch lift alone:

```
liftAt (P) : Hom X Y → BrMorph (act.obj (X,P)) (act.obj (Y,P))
```
defined by structural recursion on `Hom`, and the obligation
```
Rel f g → liftAt P f = liftAt P g        -- in BrMorph, i.e. Quot.sound-level
```
proved by induction on the `Rel` derivation. `[Y,η]` (the reindex) is a separate, `f`-independent
`BrMorph` built from `η` per array (one `BrBase` with `op = "reindex"`, `reindexings := η`); it is
genuine but bounded content, orthogonal to the `Rel` obligation.

**A `++`-transport tax applies to every `tensor`/`braid` case.** Empirically confirmed
(`lean_run_code`): `List.map f (a ++ c) = List.map f a ++ List.map f c` is **NOT `rfl`** for
variable `a c`. Since `act.obj = List.map (liftArray P)`, the object `act.obj (a++c, P)` is only
*propositionally* (via `List.map_append`) equal to `act.obj (a,P) ++ act.obj (c,P)`. So the
`tensor`/`braid`/`tensor_id`/`interchange`/`braid_natural` cases each need an `eqToHom`/`cast`
across `List.map_append` just to typecheck — this cost is on top of the nominal "cast-crossing"
constructors. Singleton objects (`[]`, `[x]`) *are* defeq (`List.map` on `[]`/`[x]` is `rfl`), so
the `copyW`/`delW` cases are clean.

### The 21 `Rel` constructors

| # | `Rel` constructor | Maps to | Difficulty |
|---|---|---|---|
| 1 | `refl` | `rfl` | trivial |
| 2 | `symm` | `.symm` of IH | trivial |
| 3 | `trans` | `.trans` of IHs | trivial |
| 4 | `comp_congr` | congr of `BrMorph.comp` + IHs | trivial |
| 5 | `tensor_congr` | congr of `BrMorph.tensor` + IHs (+ `map_append` cast) | easy |
| 6 | `id_comp` | `BrMorph.id_comp` | easy |
| 7 | `comp_id` | `BrMorph.comp_id` | easy |
| 8 | `assoc` | `BrMorph.assoc` | easy |
| 9 | `tensor_id` | `BrMorph.tensor_id` + `map_append` cast | medium |
| 10 | `interchange` | `BrMorph.tensor_comp` + `map_append` cast | medium |
| 11 | `braid_involution` | `BrMorph.braid_braid` + `map_append` cast | medium |
| 12 | `braid_natural` | `BrMorph.braid_natural` + `map_append` cast | medium |
| 13 | `tensor_unitl` | `BrMorph.tensor_unitl` | easy |
| 14 | `tensor_unitr` | `BrMorph.tensor_unitr_heq` (HEq template exists) | hard (cast) |
| 15 | `tensor_assoc_coh` | `BrMorph.tensor_assoc_heq` (template exists) | hard (cast) |
| 16 | `braid_hexagon_fwd` | `BrMorph.swap_hexagon_fwd` (template exists) | hard (cast) |
| 17 | `braid_hexagon_rev` | `BrMorph.swap_hexagon_rev` (template exists) | hard (cast) |
| 18 | `copyW_coassoc` | `BrMorph.copyW_coassoc` (singleton, defeq) | easy |
| 19 | `copyW_counitl` | `BrMorph.copyW_counitl` (singleton, defeq) | easy |
| 20 | `copyW_cocomm` | `BrMorph.copyW_cocomm` (singleton, defeq) | easy |
| 21 | `copyW_counitr` | `BrMorph.copyW_counitr_heq` (template exists) | hard (cast) |

**Key observation:** 20 of 21 constructors map to a theorem that is **already proved** on
`BrMorph` (the free-strict-SMC laws that made `Br : ColoredPROP` sorry-free). The batch lift is
essentially an endofunctor-on-generators of the free strict SMC, and the SMC laws are preserved
because they hold identically in the target. Nothing here is *intractable*; the only genuine
friction is `cast`/`HEq` bookkeeping, and every hard case has a directly analogous HEq lemma in
`Base/Br.lean` to copy.

The single non-mechanical case is the **`gen`** leaf of `liftAt` (the batch-lift of one `BrBase`
operation — theory.md Def 11): produce the new `BrBase` with `degree` extended by `P` and the
`reindexings` extended block-diagonally. This is real St-matrix construction but is a definition,
not a `Rel` obligation (it has no equation to discharge inside the well-definedness proof — it just
needs to *exist*).

### Functor laws (`map_id`, `map_comp`)

Beyond well-definedness, `act` as a `Functor` needs:
- `map_id`: `act.map (𝟙_X, 𝟙_P) = 𝟙`. Structural; `liftAt P (id X) = idm` + `[X, 𝟙_P]` reindex is
  identity (needs the reindex-of-identity-η lemma). Easy–medium.
- `map_comp`: `act.map ((f,η₁) ≫ (g,η₂)) = act.map (f,η₁) ≫ act.map (g,η₂)`. With the
  lift-then-reindex convention this reduces to the **interchange/naturality** of reindex past the
  batch lift: `[Z,η₁] ; [g,Q] = [g,P] ; [Y,η₁]` — a genuine naturality obligation (this is the
  `ev_p`-naturality content, `graded_prop.md` Eq. 3). Medium–hard, one real lemma.

---

## Size estimate

| Piece | Estimate |
|---|---|
| ~~resolve the `sh_act` blocker~~ **DONE** (option 1: `=` → `≅`, one field, zero consumers) | ✅ landed |
| `act.obj` + `liftArray` (per-array `List.map` broadcast) | ~20–40 lines |
| `liftAt` definition (with `map_append` casts in tensor/braid) | ~80–120 lines |
| `gen`-leaf batch lift (`BrBase` degree/reindexing extension) | ~40–60 lines |
| `Rel` well-definedness (21 cases; ~8 trivial, ~8 medium, ~5 hard-cast w/ templates) | ~150–250 lines |
| reindex `[X,η]` construction + its identity/composition lemmas | ~60–100 lines |
| `map_id` / `map_comp` (incl. reindex-past-lift naturality) | ~60–120 lines |
| **Total for `act` alone (functor, well-posed, sorry-free)** | **~400–650 lines** |

This is the cost of **`act` only** — it does **not** include the coherence isos `δ`/`δ0`/`υ`/`α`
or `broadcast_gen` (the remaining `StBr.lean` sorries), which sit *on top* of `act`.

---

## Go / no-go for Track B

Per the plan's pivotal test ("if `act` dwarfs Track B's projected savings, stop at Track A"):

- The object-level `sh_act` design blocker is **resolved** (option 1 implemented, build green). `act`
  now has **no remaining design gate** — it is a bounded, mechanical **~400–650-line** build. It is a
  prerequisite that Naperian typing does not shrink (Naperian *reinterprets* `act`, it does not build
  it), but nothing about it is intractable.
- The morphism-level `Rel` obligation — the literal S0 question — is **confirmed tractable** and
  mostly mechanical (20/21 cases reuse proved `BrMorph` laws).
- **Recommendation:** Track B's remaining gate is **spike S1** (jointly-monic circularity). With
  `sh_act` relaxed, building `act` is a well-scoped task that can proceed whenever funded; it is no
  longer blocked. Weigh the ~400–650 lines for `act` (plus the δ/δ0/υ/α/broadcast_gen tower on top)
  against Track B's projected savings before committing — but the decision is now a cost/benefit
  call, not a feasibility one.

`brCancelPoint` and the `Br` free-SMC quotient are **not** on this path (confirmed): the quotient is
sorry-free and every law `liftAt` needs is already proved on `BrMorph`.

---

## Resume pointer — S1 kickoff (next: run spike S1)

**Decision (2026-07-03):** run **S1** next; do **not** build full `act` or commit to Track B until
S1 resolves (building `act` first risks ~400–650 lines on a possibly-illusory payoff). Track A
(compiler hardening) is the ungated fallback / parallel value stream and does *not* help S1.

**S1 question:** is there an **acyclic** proof of `naperian_jointly_monic` (lifted `P`-indexed
families separated by their `ev_p` coordinate evaluations) that does **not** route through
`broadcast_gen`? Exit: acyclic proof/skeleton, or the explicit extra assumption it needs.

**Minimal prep to make S1 runnable (cheap; foundations Track B needs anyway):**
1. Build the small M1 `El`/`NaperianAxis` scaffolding in `Core/Naperian.lean` — enough to *state*
   jointly-monic in terms of `El(P)` + its finite enumeration.
2. Add a **provisional `act.obj`/`ev_p` stub** (tens of lines — the per-array `List.map` object
   action + enough of `ev_p`; NOT the full functorial `act.map`), since `ev_p` is defined via `act`
   (`Core/Graded.lean:90`) and jointly-monic is stated over `ev_p`.
3. Run the **minimal St-level probe**: does a single lifted axis's family separate under point
   evaluations via `St.elemental` (already sorry-free) + `funext`, without touching `broadcast_gen`?

**Read on the three outcomes:** clean + generalizes → positive (Track B fundable); stalls needing
`NaperianFamily` representability of lifted objects → difficulty migrated (medium–hard); collapses
toward Br normal form → the failure mode (Naperian bought nothing; stop at Track A). The encouraging
prior is that `St.elemental` is proved and `ev_p`/`ev_p_naturality` (Eq. 3) already exist sorry-free
in `Core/Graded.lean`; the risk is the representability-of-lifted-objects step.
