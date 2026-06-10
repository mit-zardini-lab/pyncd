# Design: Revamp `leanncd.md` around the `D`-Graded Colored PROP

**Date:** 2026-06-10
**Status:** Approved (design phase)
**Scope:** Rewrite `papers/leanncd.md` — the Lean 4 formalization design document for the NCD categorical framework — to adopt the `D`-graded colored PROP framework of `papers/graded_prop.md`, encoded as a layered tower of typeclasses parameterized by other typeclasses.

This document is a **design/spec**, not the rewrite itself. It records the agreed architecture, the typeclass signatures, the section plan, and the decisions (with rationale) reached during brainstorming. The output artifact is a revised `papers/leanncd.md`.

---

## 1. Motivation and thesis

The current `leanncd.md` predates `graded_prop.md`. It formalizes `St` and `Br` as two independent `PROP` instances and splits everything into "Layer 1 — Mathematical Encoding" and "Layer 2 — Representation" (UIDs, `Context`, names). That split was an artifact of not yet having the vocabulary `graded_prop.md` now supplies.

`graded_prop.md` shows the scattered constructions are **one structure** — a `D`-graded colored PROP — and its central promise is:

> prove the §8 propositions **once**, at the graded-PROP level, and **inherit** them at every instantiation (`St→Br`, the `D=Br` MoE level, swapped-`D` rows).

The Lean realization of "prove once, inherit everywhere" **is** parametricity over a typeclass. The revamp therefore reorganizes `leanncd.md` around a layered typeclass tower, parametric on the index PROP `D` and the operation PROP `C`, so that present work (`St→Br`) and the future extensions sketched in `prop_ideas.md` / `future_ideas.md` are *additive instances and mixins*, never edits to a proven core.

**The intent remains formalizability, not formalization** — definitions are stated as Lean `structure`/`class` data plus named `Prop`-field laws, in the shape a future Lean development transcribes directly. No Lean is *proved* in the document; signatures and proof obligations are exhibited.

---

## 2. Key decisions (with rationale)

### D1 — Discard the Layer 1 / Layer 2 split; one unified categorical development.
The old split was premised on UIDs / sizes / `Context` being non-mathematical "representation." The graded-PROP reframing kills that premise: symbolic sizes are the **fiber of the Grothendieck construction** `∫Dat` (§5/§7), and axis identity/alignment is the **pushout / coequalizer** (§6/§7). Both are categorical. The two-layer separation is therefore dissolved.

### D2 — Keep one *thinner, differently-drawn* seam: **proposition vs. computation**.
The only thing the old boundary actually bought was proof hygiene: "a lemma over `Axis` cannot accidentally mention `UData`." That survives the reframing, but the seam is no longer "is this mathematics?" — it is **"does Lean *prove* this or *compute* this?"** This is the seam `graded_prop.md` itself draws:

- §6: "§6 is a correctness/specification lens, not a composition algorithm … the pushout explains and certifies what `Context` does; it does not replace it." → the coequalizer is the *specification*; union-find + a fresh-name counter is the *implementation*.
- §7: keep the lightweight `Para` encoding and "note the gap explicitly rather than paying for a bicategory only the spec uses."

So: the **propositional core** (PROP/actegory laws, `∫Dat`, pushout-as-colimit, equivariance, weave uniqueness) is stated over UID-free types and proved; the **executable realization** (fresh-UID counter `StateM`, union-find as the *implementation* of the coequalizer, acset tables / CSV, `TermTraversable`) sits on the other side of the seam. Names (`DynamicName`, `Block`) are outside even that — pure display, out of scope.

### D3 — Hybrid foundation (lightweight `ColoredPROP` + Mathlib for the theorems).
`graded_prop.md` §10 reads as a Mathlib shopping list (`MonoidalCategory`, `SymmetricCategory`, `FreeMonoidalCategory`, `Grothendieck`, `Limits.pushout`, `Action`/`Rep`, `Monad.Algebra`). Two tensions:

- A self-contained `ColoredPROP` class keeps the "tensor = list concat, strictness laws = `rfl`" elegance of `St`/`Br` — worth preserving.
- The §8 theorems want Mathlib; hand-rolling Grothendieck/pushouts/monoidal-functors/EM-categories would contradict §10's whole strategy.

**Resolution:** a lightweight `ColoredPROP` typeclass carries the *base definitions and the `St`/`Br` instances*; the **`D`-graded structure and the §8 propositions** are stated against Mathlib, with **one explicit adapter** `ColoredPROP O → MonoidalCategory`. The adapter sits exactly on the proposition/computation seam.

### D4 — Layered mixins, not a fat class.
The core `DGradedColoredPROP D C` is small. Capabilities — `Scan`, `Route`, symmetry/equivariance, `Para` — are **composable mixin classes** layered on top. An instantiation pays only for what it declares. This is what buys future flexibility: new capabilities are new mixins; new domains are new instances; the proven core is never edited. A tall `extends` tower is avoided to prevent typeclass-diamond / resolution-performance pain.

### D5 — Python interop: light inline pointers + one consolidated section (Option B).
One-line "Python counterpart" pointers stay inline where they aid a reader working with the codebase open (e.g. `act` ↔ batch lift, `BrBase` ↔ `Broadcasted`). The *systematic* correspondence tables are consolidated into the dedicated Acset & Interop section (§8), reorganized by the new tower rather than the old two-layer split.

---

## 3. The typeclass tower

Each layer is a class parameterized by the classes below it. `D` (index PROP) and `C` (operation PROP) are **explicit class parameters**, each carrying a `ColoredPROP` instance (see C3 in §6 for why explicit, not `outParam`).

```
ColoredPROP O                                    -- lightweight base; St, Br instances
   ⇣ adapter (the seam)  →  Mathlib MonoidalCategory / SymmetricCategory
DGradedColoredPROP D C   [ColoredPROP D] [ColoredPROP C]   -- core: sh, act, δ, υ, α, axioms
   ├ TemporalGraded   D C   (extends)   -- Scan, Def 3.3–3.5
   ├ RouteStructure   D C   (mixin, STUB)     -- Route, Prop 8.6(ii)      [future_ideas]
   └ SymmetryGraded   D C T (mixin, STUB)     -- equivariance monad, Prop 8.4 [gated; equiv_unif A3]
Algebra D C V   [DGradedColoredPROP D C] [TargetActegory D V]   -- construct()
   └ ParaAlgebra ...        (mixin, STUB)      -- weight tying, pass-as-2-cell  [prop_ideas §7]
```

### 3.1 Base — `ColoredPROP`

Preserved from the current doc, with **one addition**: the elemental separation axiom (`graded_prop.md` §2.1), needed for weave uniqueness.

```lean
class ColoredPROP (ob : Type) extends SmallCategory ob where
  gen       : Type
  toList    : ob → List gen
  ofList    : List gen → ob
  tensor    : ob → ob → ob := fun a b => ofList (toList a ++ toList b)
  unit      : ob := ofList []
  tensor_assoc  : ∀ a b c, tensor (tensor a b) c = tensor a (tensor b c)
  tensor_unit_l : ∀ a, tensor unit a = a
  tensor_unit_r : ∀ a, tensor a unit = a
  swap      : ∀ a b, hom (tensor a b) (tensor b a)
  tensorHom : ∀ {a b c d}, hom a b → hom c d → hom (tensor a c) (tensor b d)
  elemental : ∀ {X Y} (f g : hom X Y),
                (∀ x : hom unit X, comp x f = comp x g) → f = g   -- NEW: (Elem)
```

`St` and `Br` remain instances exactly as in the current doc (St via Mathlib `Matrix` + `MvPolynomial String ℕ`; Br via the free-list `BrMorph`). Strictness laws discharge by `simp [List.append_assoc/nil_append/append_nil]`; Br category laws by `rfl`/list induction; St laws by `Matrix.mul_assoc` + `ring`.

### 3.2 The seam adapter

A single stated adapter turning a `ColoredPROP O` into a Mathlib strict symmetric monoidal category (via `FreeMonoidalCategory (Discrete O)` strictified). Everything **above** the seam (instances, executable defs) uses `ColoredPROP`; everything **below** (the §9 theorems) speaks Mathlib. The adapter *is* the proposition/computation boundary made into a definition.

### 3.3 Core — `DGradedColoredPROP D C`

```lean
class DGradedColoredPROP (D C : Type) [ColoredPROP D] [ColoredPROP C] where
  sh    : ColoredPROP.gen (ob := C) → C        -- shape map on colors; extends to sh* (monoid hom)
  act   : (C ×ᶜ Dᵒᵖ) ⥤ C                        -- lift action (Mathlib functor, via seam)
  δ     : ∀ X Y P, act.obj (tensor X Y, P) ≅ tensor (act.obj (X,P)) (act.obj (Y,P))
  δ0    : ∀ P, act.obj (unit, P) ≅ unit
  υ     : ∀ X, act.obj (X, unit_D) ≅ X
  α     : ∀ X P Q, act.obj (act.obj (X,P), Q) ≅ act.obj (X, tensor Q P)
  -- named Prop-field axioms (one field each):
  sh_act         : ∀ X P, sh* (act.obj (X,P)) = tensor (sh* X) P     -- (Sh-⊛)
  act_unit_assoc : …    -- actegory triangle + pentagon (υ, α coherences) → right D-actegory
  dist_coh       : …    -- δ, δ0 naturality + interchange with υ/α/σ (strong sym. monoidal action)
  broadcast_gen  : …    -- (Broadcast-gen): every C-morphism = [f,P] ; ρ, f degree-trivial
```

**Derived, not fields** (stated as `def`/`theorem` in the doc):
- `ev_p := act.map ⟨𝟙, p⟩ ≫ υ` — a natural transformation by `Functor.map_comp`.
- **Eq. 3** `[f,P] ≫ (Y ⊛ p) = (X ⊛ p) ≫ f` — built in by functoriality of `act`, *not* an axiom. The genuine content is `elemental`.

### 3.4 Weaves as cartesian-lift data

```lean
structure Weave [DGradedColoredPROP D C] {X Y : C} (g : hom X Y) where
  f       : hom X' Y'    -- base op (degree-trivial)
  P       : D
  ρ       : …            -- reindexing assembled from act(id,−) + coherence isos
  factors : g = [f, P] ≫ ρ

theorem weave_unique [DGradedColoredPROP D C] {X Y} (g : hom X Y) :
    Subsingleton (Weave g)         -- Prop 8.2, from elemental + broadcast_gen
```

The target/tiling partition and the `Ω_w` permutation are recovered from `σ`; uniqueness makes `Weave` a *datum*, not a *choice*.

### 3.5 Mixins

```lean
class TemporalGraded (D C : Type) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where
  L          : D                    -- temporal object (Δ₊ / ℕ-graded); prefix inclusions ιₘ
  restrict   : ∀ m, …               -- directed action act(−, ιₘ); NO point-evaluation ev_q
  iterate    : …                    -- finite iteration of a parametric step endofunctor (Def 3.4)
  trace      : …                    -- scanl state history H ⊗ L_{N+1} (Def 3.5)
  lift_fold_dist : …                -- act(Scan,P) ≅ Scan(act(step,P)) for P orthogonal to L
-- Scan := cata(step) is then a DEFINITION; prefix-restriction law is a COROLLARY (Prop 8.7);
-- Scan batches (Prop 8.8) from lift_fold_dist.

class RouteStructure (D C) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where … -- STUB: data-dependent coproduct injection, gate as Para param
class SymmetryGraded (D C : Type) (T : Monad D) [ColoredPROP D] [ColoredPROP C]
    extends DGradedColoredPROP D C where … -- STUB: equivariance via EM-category of T; gated (equiv_unif A3)
```

### 3.6 Algebra (`construct()`) and Para

```lean
class TargetActegory (D V : Type) [ColoredPROP D] where
  actV : (V ×ᶜ Dᵒᵖ) ⥤ V
  …    -- same coherences as §3.3, in V (PyTorch tensors; actV appends dimensions)

structure Algebra (D C V : Type) [DGradedColoredPROP D C] [TargetActegory D V] where
  F        : C ⥤ V                  -- strong symmetric monoidal (Mathlib MonoidalFunctor)
  equivar  : ∀ X P, F.obj (act (X,P)) ≅ actV (F.obj X, P)   -- D-equivariance
  coh      : …                      -- commutes with υ,α,δ; preserves ev_p
-- a morphism of algebras is a MonoidalNatTrans; weight tying collapses params via Δ.

class ParaAlgebra … extends Algebra … where … -- STUB: Para(C)→Para(V) 2-functor; passes-as-2-cells
```

### 3.7 Parametricity property (the payoff)

Every generic lemma reads:

```lean
variable {D C : Type} [ColoredPROP D] [ColoredPROP C] [DGradedColoredPROP D C]
```

and uses **only** class fields/axioms. Therefore `St→Br`, `Br→CMod` (MoE), and `Graph→C` all inherit it with no per-domain proof. `Br`-as-graded-thing and `Br`-as-index occupy different instance positions, so `instance : DGradedColoredPROP Br CMod` does not collide with `instance : DGradedColoredPROP St Br`.

---

## 4. Section plan for the revised `leanncd.md`

1. **Orientation** — the one-structure thesis (`graded_prop.md` §3), the typeclass-tower diagram, the proposition/computation seam stated up front as the organizing principle.
2. **Base: `ColoredPROP`** — the lightweight class (objects = `FreeMonoid O`, tensor = concat, strictness as `rfl`/`simp`, the new `elemental` field) + `St` and `Br` instances. Preserves the current doc's elegance.
3. **The seam adapter** — `ColoredPROP O → Mathlib MonoidalCategory/SymmetricCategory`, stated once.
4. **Core: `DGradedColoredPROP D C`** — fields, the named-law `Prop` fields, derived `ev_p`/Eq. 3; `St→Br` as flagship instance.
5. **Weaves as cartesian-lift data** — `Weave` `Σ`-type + `weave_unique`.
6. **Mixins** — `TemporalGraded` (Scan) in full; `RouteStructure`, `SymmetryGraded` as signed stubs flagged future/gated.
7. **Grothendieck split & composition-as-pushout** — `C ≅ ∫Dat` via `CategoryTheory.Grothendieck` (FreeNumeric → fiber); `Context`/union-find as the *computation* realizing the *coequalizer specification* (UIDs absorbed here). Pure categorical + seam.
8. **Acsets & Python interop** — `S_Br`/`SBrInstance` as the finite presentation of an `∫Dat`-morphism (references `acset.md`, focuses on the Lean encoding); the seam made tangible (acset tables / CSV = executable realization of the `∫Dat` spec); the **consolidated Python correspondence tables**, reorganized by the new tower.
9. **Propositions as generic theorems** — the payoff table: each `graded_prop.md` §8 prop, its Lean statement `variable [DGradedColoredPROP D C]`, the Mathlib machinery discharging it, and what it costs per instance (theorem free; coherence obligations recur). Covers 8.1–8.8.
10. **Instantiation & future extensions** — (a) `D=St, C=Br` today: flagship `instance`, each tower field → pyncd realization. (b) Additive-extension menu: `D=Br` MoE (§9.2), swap-`D` rows graph/`Rep(G)`/`Stoch` (§9.3), mixin roadmap (`RouteStructure`/`SymmetryGraded`/`ParaAlgebra`/corecursion) — each labeled *new instance* or *new mixin*, never a core edit.
11. **Lean formalization notes** — Mathlib coverage map; strictification strategy (develop over `FreeMonoidalCategory`, strictify once so `α/λ/ρ = Iso.refl`); instance-resolution discipline (D/C explicit params, no `outParam` magic); honest caveats (inheritance is for *theorems*, not *obligations* — coherence `Prop` fields recur per instance; mixins-not-tall-tower to avoid diamonds; the Para 2-category gap noted, not paid for).
12. **Appendix — out of scope** — `DynamicName`/LaTeX and `Block` display metadata: semantically transparent, not encoded; one paragraph on why (matches the current doc's refusal to encode `Block`).

---

## 5. What changes from the current doc

| Current | Revised |
| --- | --- |
| "Layer 1 / Layer 2" top-level split | Dissolved → one unified development + proposition/computation seam |
| `St`, `Br` as two independent `PROP` instances | Instances of `ColoredPROP`, plus a single `DGradedColoredPROP St Br` grading instance |
| `PROP` class | `ColoredPROP` (adds `elemental`) |
| No grading / lift / actegory | `DGradedColoredPROP` core class (`sh`, `act`, `δ/δ0/υ/α`, named axioms) |
| Weave described in prose | `Weave` `Σ`-type + `weave_unique` theorem |
| No Scan | `TemporalGraded` mixin; `Scan := cata(step)` as a definition |
| Layer 2: `WithUID`, `TermM`, `Context`, `FreeNumeric` | Absorbed: FreeNumeric → `∫Dat` fiber (§7); UID/`Context`/`TermM` → executable side of the seam (§7); acset realization → §8 |
| Inline per-section "Python counterpart" paragraphs | Light one-line inline pointers + consolidated tables in §8 (Option B) |
| `DynamicName` as Layer 2 §1 | Out-of-scope appendix |
| (none) | Generic §8 propositions; future-extension menu; instance-resolution + caveat notes |

---

## 6. Honest caveats (must appear in the doc, not be hidden)

- **C1 — Inheritance is for theorems, not obligations.** §8's theorems transfer free, but each new `instance` must still discharge the coherence `Prop` fields (actegory triangle/pentagon, dist coherence). "Inherit everywhere" applies to the *proven propositions*, not to the *instance obligations*.
- **C2 — Strictness is the real friction.** Mathlib categories are not strict. Mitigation: develop over `FreeMonoidalCategory` and strictify once so all associators/unitors become `Iso.refl` downstream and PROP equations hold definitionally (`graded_prop.md` §10).
- **C3 — Instance-resolution ergonomics.** With `D` and `C` both free, Lean's instance search needs them determined. Make `D`/`C` explicit (or semi-explicit) class parameters rather than relying on `outParam` unification, so `[DGradedColoredPROP D C]` resolves predictably.
- **C4 — Mixins, not a tall tower.** Keep Scan/Route/Symmetry/Para as composable mixins to avoid diamond and typeclass-performance problems from a deep `extends` chain.
- **C5 — The Para 2-category gap.** Keep the lightweight `Para(V)` 1-category encoding (`Σ (P:V), (P⊗A ⟶ B)` + an explicit reparameterization relation); treat the "2-cell" statements as that relation, and note the gap rather than paying for a full bicategory only the spec uses (`graded_prop.md` §7).
- **C6 — Equivariance is gated.** Prop 8.4's body depends on `SymmetryGraded` and the EM-category machinery; finite-`G` is reachable now (`equivariance_unification.md` A1/A2), the graded-PROP-dependent parts (A3) wait on this very formalization. State the proposition, gate the proof.

---

## 7. Success criteria

The revised `papers/leanncd.md`:

1. Presents the framework as **one** `D`-graded colored PROP development (no Layer 1 / Layer 2).
2. States the **typeclass tower** of §3 with all signatures, and the proposition/computation **seam** as its organizing principle.
3. Makes every `graded_prop.md` §8 proposition a **generic theorem** `variable [DGradedColoredPROP D C]`, with the Mathlib machinery and per-instance cost noted.
4. Shows `St→Br` as the flagship `instance`, and the `D=Br` / swap-`D` / Route / Para directions as **additive instances and mixins** — demonstrably no core edit required.
5. Absorbs FreeNumeric (→`∫Dat`), UID/`Context` (→pushout/computation side), and the acset realization (→§8) into the unified narrative, with consolidated Python correspondence tables and light inline pointers.
6. Carries all six caveats of §6 explicitly.
7. Writes no proved Lean — signatures + named proof obligations only (formalizability, not formalization).

---

## References

- `papers/graded_prop.md` — the `D`-graded colored PROP framework (the structure being adopted); §3 (data/axioms), §5 (Grothendieck), §6 (pushout), §7 (algebras/Para), §8 (propositions), §9 (instantiation), §10 (Lean notes), §11 (CDL).
- `papers/leanncd.md` — the document being revised.
- `papers/acset.md` — schema `S_Br`, `SBrInstance`, the structure/data split, the existing `SBrInstance → Br` Lean-encoding section (§8 references this).
- `papers/equivariance_unification.md` — equivariance closure; Appendix A (Phase 5 Lean reachability: A1/A2 now, A3 gated).
- `papers/prop_ideas.md`, `papers/future_ideas.md` — the future extensions (Route, MoE, Para passes, corecursion) the mixin design must accommodate.
- Mathlib4 — `CategoryTheory.{Monoidal, Grothendieck, Limits, FreeMonoidalCategory, Quotient, Action, Monad.Algebra}`.
