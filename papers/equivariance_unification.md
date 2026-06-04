# Unifying the Three Routes to Equivariance

*Closing the open problem at [graded_prop.md Prop 8.4](graded_prop.md#8-propositions-the-synthesis-organizes) — Phases 1–2 of the attack plan.*

[graded_prop.md](graded_prop.md) records three ways the framework expresses an *equivariant model*, and asks whether they coincide:

- **(a) symmetry monad** ([Prop 8.4](graded_prop.md#8-propositions-the-synthesis-organizes)) — a monad `T` for the symmetry; an equivariant algebra lifts to the Eilenberg–Moore category;
- **(b) base grading** — `D = BG`, the delooping of a group `G` (group convolution; [future_ideas.md Appendix A.2](future_ideas.md#a2-d--group-bg-base-repg-fibers));
- **(c) fiber grading** — `D = Rep(G)` (steerable features; [same appendix](future_ideas.md#a2-d--group-bg-base-repg-fibers)).

This note carries out **Phase 1** (the object-level kernel) and **Phase 2** (lifting to algebras of a graded PROP), reducing the question to a single, classical equivalence and proving the three routes agree under explicit hypotheses. The boundary analysis (Phase 4) is previewed; a Lean target (Phase 5) is noted.

**Result.** For a group (or monoid) `G`, all three routes compute one and the same category — `C`-algebras valued in the `G`-objects of the target — so they coincide. The proof is: *(1)* the three presentations of "a `G`-object" are monoidally equivalent (Phase 1, classical), and *(2)* `Alg(C, −)` preserves monoidal equivalences (Phase 2, standard 2-functoriality). The coincidence is therefore a theorem, with scope exactly the *action (Hopf) monads*; for symmetry monads carrying extra relations the routes provably diverge (§5).

---

## 1. Setup and notation

We use the definitions of [graded_prop.md](graded_prop.md): a `D`-graded colored PROP `C` (Def 3.1) with shape map `sh : O_C → Ob D` and lift action `act : C × Dᵒᵖ ⥤ C`; a **target** `D`-actegory `V` (Def 7.1, symmetric monoidal with a compatible `D`-action `⊛_V`); and an **algebra** `F : C → V` (Def 7.2): a strong symmetric monoidal, `D`-equivariant functor. Write `Alg(C, V)` for the category of algebras and monoidal natural transformations.

**The group and its action monad.** Let `G` be a group (the monoid case is identical wherever invertibility is not invoked). On `V`, the **action monad** is the `G`-fold copower

```text
T_G(X) = G · X = ∐_{g∈G} X ,
```

with unit `η_X : X → G·X` the coprojection at the identity `e`, and multiplication `μ_X : G·(G·X) = (G×G)·X → G·X` induced by the group multiplication `G × G → G`. Concretely:

- `V = Set`: `T_G(X) = G × X`;
- `V = Vect_k`: `T_G(X) = k[G] ⊗ X` (the group-algebra functor, since `⊕_{g} X ≅ k[G] ⊗ X`).

**Standing hypotheses.**

- **(H1)** `G` is a group (or monoid).
- **(H2)** `V` is a cocomplete symmetric monoidal category admitting `G`-indexed copowers, with `⊗` preserving them in each variable.
- **(H3)** `T_G` is a **Hopf monad** on `V` (Bruguières–Lack–Virelizier). For an action monad this is automatic from the group structure: the comultiplication `g ↦ (g,g)` and antipode `g ↦ g⁻¹` make `T_G` Hopf (for `V = Vect`, this is the Hopf-algebra structure of `k[G]`).
- **(H4)** *Compatibility.* The `G`-action commutes with the `D`-lift — i.e. `T_G` distributes over `⊛_V` (the symmetry is "orthogonal" to broadcasting). This makes `V^{T_G}` a `D`-actegory and the Phase-1 equivalence a `D`-actegory equivalence.

The three routes, stated as algebras valued in three targets:

| route | target | reading |
| --- | --- | --- |
| (a) symmetry monad | `V^{T_G}` (Eilenberg–Moore: `T_G`-algebras = `G`-objects of `V`) | `F` lifts through `U : V^{T_G} → V` |
| (b) base grading | `[BG, V]` (functors from the delooping) | each wire carries a `G`-action |
| (c) fiber grading | `Rep(G) = Vect^{T_G}` | the `V = Vect` instance of (a) |

---

## 2. Phase 1 — the object-level kernel

**Theorem 1 (action monad = delooping presheaves).** Under (H1)–(H2),

```text
V^{T_G}  ≃  [BG, V]  ≃  G-objects of V .
```

For a group `G`, `[BG, V]` is the category of objects of `V` equipped with a `G`-action by automorphisms.

**Proof.** By the universal property of the copower, a structure map `a : G·X → X` is the same data as a family `{a_g : X → X}_{g∈G}`, where `a_g = a ∘ (coprojection at g)`. The Eilenberg–Moore laws translate term-by-term:

- *unit* `a ∘ η_X = id_X` ⟺ `a_e = id_X`;
- *associativity* `a ∘ μ_X = a ∘ T_G(a)` ⟺ `a_g ∘ a_h = a_{gh}` for all `g, h`.

Thus `{a_g}` is precisely a monoid homomorphism `G → End_V(X)`, i.e. a functor `BG → V` sending the unique object to `X` and `g` to `a_g`. A morphism of `T_G`-algebras `f : (X, a) → (Y, b)` is a map with `f ∘ a_g = b_g ∘ f` for all `g` — exactly a natural transformation of the corresponding functors. Hence `V^{T_G} ≃ [BG, V]` as categories. When `G` is a group, `a_g ∘ a_{g⁻¹} = a_e = id`, so every `a_g` is invertible: `[BG, V]` is `G`-objects with an action by automorphisms. ∎

**Proposition 1′ (the equivalence is monoidal).** Under (H3), `V^{T_G}` is symmetric monoidal and `V^{T_G} ≃ [BG, V]` is a symmetric monoidal equivalence.

**Proof.** Being a Hopf monad (H3) is exactly the condition under which the Eilenberg–Moore category `V^{T_G}` inherits a monoidal structure for which the forgetful `U : V^{T_G} → V` is strong monoidal: the comultiplication of `T_G` equips `(X,a) ⊗ (Y,b)` with the **diagonal** action `g ↦ a_g ⊗ b_g`, and the antipode supplies duals. On the other side, `[BG, V]` carries the pointwise tensor `(X, a_•) ⊗ (Y, b_•) = (X⊗Y, g ↦ a_g ⊗ b_g)` — the *same* diagonal action. The equivalence of Theorem 1 matches these structures on the nose and preserves the symmetry, hence is symmetric monoidal. ∎

**Instances.** `V = Set`: `Set^{T_G} ≃ [BG, Set] ≃ G\text{-Set}`. `V = Vect_k`: `Vect^{T_G} ≃ [BG, Vect] ≃ Rep(G)` — the `k[G]`-modules. These are textbook; Theorem 1 is the classical identification of the action monad's algebras with the delooping's presheaves.

---

## 3. Phase 2 — lifting to algebras of `C`

The only categorical input beyond Phase 1 is that forming algebras of `C` is a 2-functor in the target.

**Lemma 2 (transport of algebras along a monoidal actegory equivalence).** Let `W ≃ W'` be a symmetric monoidal equivalence of `D`-actegories (i.e. the equivalence and its quasi-inverse are strong symmetric monoidal and commute with the `D`-actions up to coherent iso). Then post-composition induces an equivalence of algebra categories

```text
Alg(C, W)  ≃  Alg(C, W') .
```

**Proof.** An algebra `F : C → W` is a strong symmetric monoidal `D`-equivariant functor. If `E : W → W'` is a strong symmetric monoidal `D`-actegory functor, then `E ∘ F` is again strong symmetric monoidal (composite of strong monoidal functors) and `D`-equivariant (the equivariance isos compose), so `E ∘ −` is a functor `Alg(C, W) → Alg(C, W')`. A quasi-inverse `E'` to `E` induces `E' ∘ −`, and the natural isos `E'E ≅ id`, `EE' ≅ id` post-compose to natural isos of the induced functors; monoidal/actegory coherence of `E, E'` makes these monoidal natural isos. Hence `E ∘ −` is an equivalence. ∎

**The three routes, precisely.**

- **Route (a).** By Prop 8.4 a `T_G`-equivariant algebra is a lift of `F` through `U : V^{T_G} → V`; the category of such lifts *is* `Alg(C, V^{T_G})`.
- **Route (b).** Grading the index by `BG` makes the lift `act(−, g)` act on each wire by `g ∈ G`; an algebra must transport this to a `G`-action on each value, i.e. land in `[BG, V]`. So `Alg(C_{BG}, V) = Alg(C, [BG, V])`. *(This is the faithful unfolding of "graded over `BG`": for a group, building `G` into the source index and valuing in `G`-objects of the target are the same data — a coincidence special to one-object groupoids, and the reason the question has a clean answer.)*
- **Route (c).** The `V = Vect` instance of (a): `Alg(C, Vect^{T_G}) = Alg(C, Rep(G))`.

**Theorem 2 (the three routes coincide).** Under (H1)–(H4),

```text
Alg_a(C, V)  ≃  Alg_b(C, V)  ≃  Alg_c(C, V) ,   all  ≃  Alg(C, V^{T_G}) .
```

**Proof.** Route (a) is `Alg(C, V^{T_G})` by definition. Prop 1′ gives a symmetric monoidal equivalence `V^{T_G} ≃ [BG, V]`, which by (H4) is a `D`-actegory equivalence; Lemma 2 then yields `Alg(C, V^{T_G}) ≃ Alg(C, [BG, V]) = Alg_b(C, V)`. Taking `V = Vect` specializes the same chain to `Alg(C, Vect^{T_G}) = Alg(C, Rep(G)) = Alg_c(C, V)`. ∎

So the symmetry-monad condition (a), the `BG`-graded index (b), and the `Rep(G)`-valued fiber (c) are three presentations of the single category `Alg(C, V^{T_G})` — equivariant `C`-algebras = `C`-algebras valued in `G`-objects of the target.

---

## 4. Corollaries

**Corollary 3 (the open problem, resolved).** Under (H1)–(H4) the Prop 8.4 conjecture holds: the symmetry-monad and `BG`-grading routes to equivariance give equivalent categories of equivariant algebras, and `Rep(G)` is the fiber (`V = Vect`) instance. Geometric deep learning is recovered identically by all three.

**Corollary 4 (St is the translation instance).** Take `G = ℤⁿ` (discrete translations). `[BG, Set]` is `ℤⁿ`-sets and `[BG, Vect]` is `ℤⁿ`-representations; the lift's affine offsets ([future_ideas.md Appendix A.1](future_ideas.md#a1-d--st-axes)) realize the `G`-action, so the translation-equivariant CNN is `Alg(C, V^{T_{ℤⁿ}})` — confirming "St with the translation symmetry = grading over `B(ℤⁿ)`."

### 4.1 Base ⊗ fiber: the action groupoid (Phase 3 completed)

[future_ideas.md Appendix A.2](future_ideas.md#a2-d--group-bg-base-repg-fibers) flagged that a full steerable network grades over *both* a base/spatial role (`BG`, group convolution) and a fiber role (`Rep(G)`), and asked how the two combine. They do **not** combine as two separate gradings to be composed; they are **one `G` acting diagonally on the field category** — translating the base and transforming the fiber at once — and Theorem 2 then applies verbatim. The organizing object is the **action groupoid** `B ⋊ G`.

**Proposition 5 (base–fiber unification).** Let `G` act on a base `B`, and let the field category `V_B := [B, \mathbf{Vect}]` carry the *induced* action monad `T_G^B` with `(T_G^B F)(x) = \bigoplus_{g∈G} F(g^{-1}·x)` (unit = identity component, multiplication = group law). Then:

1. `V_B^{T_G^B} ≃ [B ⋊ G, \mathbf{Vect}]` — the category of `G`-equivariant vector bundles over `B`, i.e. **steerable feature fields**. *(This is Theorem 1 with `V := V_B` and `T := T_G^B`: a `T_G^B`-algebra is a functor out of the action groupoid `B ⋊ G` — objects = points of `B`, morphisms `(g : x → g·x)` — which is exactly an equivariant bundle.)*
2. `Rep(G)` is the **point-base** instance `B = ⋆`: then `⋆ ⋊ G = BG` and `[BG, \mathbf{Vect}] = Rep(G)`.
3. Hence steerable equivariant `C`-algebras are `Alg(C, V_B^{T_G^B})` by **Theorem 2** — the *same* theorem, with the field category as target and the *base* action monad. Base equivariance and fiber equivariance are one `G`-action, not two.

**Proof.** (1) Apply Theorem 1 to `V_B` with the induced action monad; its Eilenberg–Moore algebras are functors `B ⋊ G → \mathbf{Vect}`, and presheaves on the action groupoid are precisely `G`-equivariant bundles over `B` (standard). (2) `⋆ ⋊ G = BG`, and Theorem 1 gives `[BG, \mathbf{Vect}] = \mathbf{Vect}^{T_G} = Rep(G)`. (3) Theorem 2 with `V := V_B`, `T := T_G^B`; `V_B` is a `D`-actegory for any *remaining* (non-spatial) broadcasting, and (H4) holds because `G` acts only on `B`, leaving that broadcasting untouched. ∎

**Remark (the two appendix rows are two extremes of one construction).** Varying the base interpolates between the appendix's `BG` and `Rep(G)` cells:

- `B = ⋆` → `[BG, \mathbf{Vect}] = Rep(G)` — the pure **fiber** (no space): maximal fiber constraint.
- `B = G/H` (a homogeneous space) → `[B ⋊ G, \mathbf{Vect}] ≃ Rep(H)` — feature fields on `G/H` correspond to representations of the stabilizer `H` (Mackey / the standard GDL dictionary).
- `B = G` (free regular action) → `[G ⋊ G, \mathbf{Vect}] ≃ \mathbf{Vect}` — pure **convolution**: a field over the free `G`-space is determined by its value at one point, so the constraint is carried entirely spatially.

So `BG`-base and `Rep(G)`-fiber are the `B = G` (or `G/H`) and `B = ⋆` ends of the single action-groupoid construction `B ⋊ G`; a general steerable net sits in between.

---

## 5. Scope and failure boundary (Phase 4 preview)

Theorem 2 holds **exactly** for symmetries that are group/monoid actions; the hypotheses pinpoint where it breaks.

- **(H1)/(H3) are sharp.** The argument uses, essentially only, that `T` is an **action (Hopf) monad** — that its Eilenberg–Moore category is `[BG, V]`. A monad `T` carrying *relations beyond a bare action* (a non-free `G`-action; a monad whose Lawvere theory is not `BG`; a monad for a richer algebraic structure) has `V^T ⊋ [BG, V]`, and routes (a) and (b) **genuinely diverge**. Thus: *the three routes coincide iff the symmetry monad is the action monad of a group/monoid (iff its presentation is `BG`).* This is the precise closure — a theorem with scope, not a yes/no.
- **(H4) is load-bearing.** If the symmetry acts on the *broadcasting itself* (the `G`-action does not commute with `⊛`), one needs a distributive law `T_G ⊛ ⇒ ⊛ T_G` and `V^{T_G}` is a `D`-actegory only when it exists; absent it, the equivalence need not descend to algebras.
- **Continuous `G`.** For a Lie group, `BG` must be enriched (topological/smooth) and `Vect^{T_G} ≃ Rep(G)` needs compactness/semisimplicity (Peter–Weyl); finite `G` is unconditional, compact `G` needs the representation theory, non-compact `G` may fail (no semisimplicity). The categorical skeleton (Theorems 1–2) is identical; only the analytic hypotheses on the copower/completeness change.

---

## 6. Status of the attack plan

- **Phase 0** (precise statement, comparison functor) — folded into §1 and the route table.
- **Phase 1** (object kernel) — **done**: Theorem 1 + Prop 1′ (classical; full proof above).
- **Phase 2** (algebra lift) — **done**: Lemma 2 + Theorem 2.
- **Phase 3** (reconcile `Rep(G)`, and base ⊗ fiber) — **done**: Cor. 3 (the `V = Vect` instance) plus Prop 5 (§4.1), which unifies base and fiber via the action groupoid `B ⋊ G` — `Rep(G)` is `B = ⋆`, convolution is `B = G/H`.
- **Phase 4** (scope) — characterized in §5; the converse (`V^T ⊋ [BG,V]` for non-action monads) is sketched, not fully written.
- **Phase 5** (Lean) — finite `G` is reachable: Mathlib has `CategoryTheory.Monad.Algebra` (Eilenberg–Moore), `Action`/`Rep G`, and monoidal-functor categories; the target is `Action V G ≃ V^{T_G}` and `Alg(C, −)` preserving it. Continuous `G` is out of reach (no enriched/smooth category theory in Mathlib).

**Net:** the open problem is closed in the affirmative for group/monoid symmetries — it is the classical `V^{T_G} ≃ [BG, V] ≃ Rep(G)` equivalence transported through `Alg(C, −)` — with the failure mode (non-action monads) characterizing its exact scope.

---

## References

- [graded_prop.md](graded_prop.md) — the graded colored PROP framework; Prop 8.4 (the open problem), §9.3 (`D`-menu).
- [future_ideas.md §6.5](future_ideas.md#65-swapping-the-index-d-as-a-dial-across-ml) and [Appendix A](future_ideas.md#appendix-a--index-categories-d-in-detail) — the three routes and the `Rep(G)` detail.
- Mac Lane, *Categories for the Working Mathematician* — monads, Eilenberg–Moore, monadicity.
- Bruguières, Lack, Virelizier, "Hopf monads on monoidal categories", *Advances in Mathematics* 227, 2011 — when `V^T` is monoidal and `U` strong monoidal (H3).
- Standard: `Rep(G) = [BG, Vect] = k[G]\text{-Mod}` (the action monad's modules).
