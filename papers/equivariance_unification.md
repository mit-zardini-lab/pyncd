# Unifying the Three Routes to Equivariance

*Closing the open problem at [graded_prop.md Prop 8.4](graded_prop.md#8-propositions-the-synthesis-organizes) — Phases 1–2 of the attack plan.*

[graded_prop.md](graded_prop.md) records three ways the framework expresses an *equivariant model*, and asks whether they coincide:

- **(a) symmetry monad** ([Prop 8.4](graded_prop.md#8-propositions-the-synthesis-organizes)) — a monad `T` for the symmetry; an equivariant algebra lifts to the Eilenberg–Moore category;
- **(b) base grading** — `D = BG`, the delooping of a group `G` (group convolution; [future_ideas.md Appendix A.2](future_ideas.md#a2-d--group-bg-base-repg-fibers));
- **(c) fiber grading** — `D = Rep(G)` (steerable features; [same appendix](future_ideas.md#a2-d--group-bg-base-repg-fibers)).

This note carries out **Phase 1** (the object-level kernel), **Phase 2** (lifting to algebras of a graded PROP), **Phase 3** (base ⊗ fiber, §4.1), and **Phase 4.1–4.2** (the sharp scope and the symmetry-on-the-broadcasting case, §5): it reduces the question to a single classical equivalence, proves the three routes agree, makes the agreement an "iff" (coincidence ⟺ action monad), and recovers DeepSets as the degree-permutation instance. It also closes **Phase 4.3 for compact `G`** (modulo standard enriched scaffolding — the e3nn/`SO(3)` setting); only **non-compact `G`** (4.3, §5.3) and the Lean formalization (Phase 5) remain open.

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

## 5. Scope and failure boundary (Phase 4)

Theorem 2 holds **exactly** for symmetries that are group/monoid actions. §5.1 makes this an "iff" (4.1); §5.2 handles symmetries that act *on* the broadcasting (4.2); §5.3 is the still-open analytic case (4.3).

### 5.1 The recognition theorem — coincidence ⟺ action monad

**Theorem 6 (recognition).** Fix a group (or monoid) `G` and let `V` be cocomplete symmetric monoidal. For a monad `T` on `V`, the following are equivalent:

1. route (a) with `T` coincides with the `BG`-grading route (b) — i.e. `V^T ≃ [BG, V]` as categories **over `V`** (the equivalence commutes with the forgetful functors to `V`);
2. `T ≅ T_G = G ⊗ (−)` as monads on `V`;
3. `T`'s Lawvere/PROP presentation is `BG`.

Hence the three routes of Theorem 2 coincide **iff** the symmetry monad is an action monad.

**Proof.** (2)⟹(1) is Theorem 1. (1)⟹(2): the forgetful `U : [BG,V] → V` is monadic, with monad `T_G` (left adjoint `T_G = G⊗(−)`, and `[BG,V] = V^{T_G}` by Theorem 1). The forgetful `U_T : V^T → V` is monadic with monad `T` by definition of the Eilenberg–Moore category. An equivalence `Φ : V^T ≃ [BG,V]` **over `V`** (`U ∘ Φ ≅ U_T`) is an equivalence of monadic functors; the monad of a monadic functor is recovered as forgetful-∘-its-left-adjoint and is unique up to isomorphism, so `T ≅ T_G`. (2)⟺(3): `G⊗(−)` is the monad presented by the theory `BG` — one unary operation per element of `G`, relations = the group/monoid law, nothing more. ∎

The phrase **"over `V`"** carries the weight and is exactly what the framework means by "the routes coincide": both `V^T` and `[BG,V]` come with their forgetful functors to `V` (forget the equivariance), and coincidence is an equivalence respecting them. An abstract equivalence of categories is *not* enough — `G`-Set and `H`-Set can be abstractly equivalent for non-isomorphic `G, H`, but never over `Set`.

**Remark (necessity has teeth — counterexamples).** Drop the action-monad hypothesis and the routes provably diverge. The decisive invariant is that **an action monad preserves coproducts** — `T_G(∐_i X_i) ≅ ∐_i T_G(X_i)` and `T_G(∅) = ∅` — being "linear" in its argument.

- *Free-monoid (list) monad* `T_{list}(X) = X^*` on `Set`: `Set^{T_{list}} = Mon` (monoids). It is **not** an action monad — `T_{list}(∅) = \{ε\} = 1 ≠ ∅`, so it fails to preserve the initial object — hence `Mon ≇ [BG,Set] = G\text{-Set}` over `Set` (coproduct of monoids is the free product, not disjoint union). A "`T_{list}`-equivariant" `C`-algebra (valued in monoids) is not any `BG`-graded one.
- *Action-with-a-relation*: the monad for "`G`-action together with a fixed invariant element" (or a commuting idempotent) has `V^T` strictly larger than `[BG,V]` — its algebras carry the `G`-action *plus* the extra structure. This is the framework's intended failure mode: **relations beyond a bare action push `V^T` off the `[BG,V]` locus**, and routes (a)/(b) separate.

### 5.2 Symmetry acting on the broadcasting (dropping H4)

H4 assumed `G` commutes with `⊛`. The interesting failure is when `G` acts *on the degree itself* — set/permutation symmetry. It splits cleanly.

**Proposition 7 (degree-permutation is internal to `D`; DeepSets recovered).** Suppose `G` acts on the degree object `P` by `D`-**automorphisms**, `ρ : G → Aut_D(P)`. Then no symmetry monad is needed: the action is realized by reindexings `act(−, ρ(g)) = [−, ρ(g)]` *already in the framework*, and a broadcasted morphism `g = [f, P]` is `G`-equivariant iff it is fixed by them. For the canonical case `G = S_n` permuting the `n` identical coordinates of a degree `P = A^{⊗ n}` (a set of `n` inputs), `S_n ⊆ Aut_D(P)` is the **symmetric-monoidal symmetry `σ` of the colored PROP itself**, and `act(−, σ)` is automatically natural (functoriality of `act` in `Dᵒᵖ`). Consequently:

- the **tiling** part (the base op broadcast over `P`) is `S_n`-equivariant *unconditionally* — the base op runs identically at each coordinate;
- the **output weave** (the aggregation over the permuted coordinates) is `S_n`-equivariant **iff** it is a symmetric aggregation (sum / mean / max / any `S_n`-invariant reduction).

This **recovers the DeepSets theorem**: a set-function layer is permutation-invariant iff its aggregation is symmetric (`ρ(∐_x φ(x))` form). Equivariance over the degree is not added structure — it is the PROP's own symmetry — and the only real condition lands on the output weave.

*Proof.* `Aut_D(P)` acts on the image of `act(−,P)` through `act(−, σ)` by functoriality of the lift; uniformity of the lift makes the input/tiling side invariant automatically; the output side contracts the `n` coordinates and is `S_n`-invariant iff that contraction is symmetric. ∎

**Proposition 8 (general entanglement ⟺ a distributive law).** If `G` acts on the degree but **not** by `D`-automorphisms, then `V^{T_G}` is a `D`-actegory — and Theorem 2 still lifts — **iff** there is a distributive law `λ : T_G ∘ ⊛ ⇒ ⊛ ∘ T_G` satisfying the Beck axioms. By Beck's theorem (`distributive laws T∘S ⇒ S∘T` ⟺ liftings of `S` to `V^T`), such a `λ` lifts the `⊛`-action to `V^{T_G}`; the Phase-1 monoidal equivalence then upgrades to a `D`-actegory equivalence and Lemma 2 / Theorem 2 go through, giving `Alg_T(C,V) ≃ Alg(C, V^{T_G})`. When no coherent `λ` exists the routes need not agree. So **H4 can be dropped exactly when a distributive law `T_G ⊛ ⇒ ⊛ T_G` exists**; Prop 7 is the special case where `λ` is the canonical symmetry and always exists. ∎

### 5.3 Continuous `G`: compact (closable) vs non-compact (open)

The discrete proofs use the copower `T_G = G·(−)` and finite direct sums; a topological group needs analytic replacements. The dividing line is **compactness**.

**Compact `G` — closable modulo enriched scaffolding (Proposition 9).** Let `G` be a compact Hausdorff group and work enriched over a base of (topological / Hilbert) vector spaces with the relevant completeness. Replace the discrete copower by the **Haar–convolution monad** `T_G` (the measure/convolution Hopf algebra of `G`, integrating against normalized Haar measure), whose algebras are the continuous representations. Then the discrete development transports verbatim:

- **(Peter–Weyl)** `Rep(G)` is semisimple symmetric monoidal; its irreducibles form a small set of generating **colors** and every object is a (Hilbert) direct sum of them — the [Appendix A.2 fiber](future_ideas.md#a2-d--group-bg-base-repg-fibers) picture survives, with the Peter–Weyl decomposition in place of the finite `⊕`.
- **(enriched Theorem 1)** `V^{T_G} ≃ [BG, V]_{cont} ≃ Rep(G)`, monoidally (the convolution algebra is Hopf; tensor of reps = diagonal action), where `BG` is the enriched delooping and `[BG,V]_{cont}` the continuous functors.
- **(enriched Theorems 2 & 6)** `Alg(C, V^{T_G})` is the steerable-equivariant algebra category, and the recognition theorem still holds: `Rep(G) → V` is monadic by enriched Barr–Beck (the convolution algebra has a bounded approximate identity; modules = continuous reps), so coincidence ⟺ `T` is the convolution action monad.

The analytic inputs are exactly: Haar measure (standard for compact `G`), the convolution Hopf algebra and its approximate identity, enriched Barr–Beck monadicity, and completeness of `V` for direct sums. This is the setting GDL actually uses — `SO(2)`, `SO(3)`, `O(3)`, finite point groups (e3nn) — so compact `G` is **closed in practice**, the categorical skeleton being identical to §§2–5.2.

**Non-compact `G` — open; the clean equivalence is a compact phenomenon.** Drop compactness and three things break, each identifiable:

1. **Semisimplicity fails (no Peter–Weyl).** Finite-dimensional reps of e.g. `SL(2,ℝ)` are not completely reducible; unitary reps decompose by a **direct integral** over the unitary dual against Plancherel measure, not a discrete direct sum. "Colors = irreps, objects = `⊕` irreps" is then replaced by *measurable fields of Hilbert spaces / direct integrals* — outside the discrete colored-PROP framework.
2. **No canonical action monad.** Algebraic, smooth (Casselman–Wallach), unitary, and tempered representations give genuinely different categories; there is no single `T_G` whose Eilenberg–Moore category is "the" `Rep(G)`. The answer bifurcates by representation-theoretic context.
3. **Recognition/monadicity may fail.** Without a well-behaved approximate identity, the forgetful from a continuous-rep category need not be monadic over the chosen base, so Theorem 6's argument does not apply.

The abstract `V^{T_G}` (for a *chosen* monad) is always defined, but its identification with a workable, color-generated `Rep(G)` — hence the practical steerable-feature picture — is a **compact-group phenomenon**. Non-compact symmetries need compactification, a restricted finite-dimensional (non-unitary) rep choice, or a **direct-integral generalization of the grading** (index = the unitary dual with Plancherel measure; colors = a measurable field, not a discrete set). Formalizing that generalization is the open residue of Phase 4.

---

## 6. Status of the attack plan

- **Phase 0** (precise statement, comparison functor) — folded into §1 and the route table.
- **Phase 1** (object kernel) — **done**: Theorem 1 + Prop 1′ (classical; full proof above).
- **Phase 2** (algebra lift) — **done**: Lemma 2 + Theorem 2.
- **Phase 3** (reconcile `Rep(G)`, and base ⊗ fiber) — **done**: Cor. 3 (the `V = Vect` instance) plus Prop 5 (§4.1), which unifies base and fiber via the action groupoid `B ⋊ G` — `Rep(G)` is `B = ⋆`, convolution is `B = G/H`.
- **Phase 4** (scope) — **4.1 done** (Theorem 6: coincidence ⟺ action monad, by monadicity + uniqueness, with the free-monoid-monad counterexample, §5.1); **4.2 done** (Prop 7: degree-permutation is the PROP's own symmetry, recovering DeepSets; Prop 8: general entanglement ⟺ a Beck distributive law, §5.2); **4.3 — compact `G` closed** modulo enriched scaffolding (Prop 9: Haar–convolution monad + Peter–Weyl transport the skeleton; the e3nn/`SO(3)` setting), **non-compact `G` open** (semisimplicity, a canonical monad, and monadicity all fail — needs a direct-integral grading over the unitary dual), §5.3.
- **Phase 5** (Lean) — finite `G` is reachable: Mathlib has `CategoryTheory.Monad.Algebra` (Eilenberg–Moore), `Action`/`Rep G`, and monoidal-functor categories; the target is `Action V G ≃ V^{T_G}` and `Alg(C, −)` preserving it. Continuous `G` is out of reach (no enriched/smooth category theory in Mathlib).

**Net:** the open problem is closed in the affirmative for group/monoid symmetries — it is the classical `V^{T_G} ≃ [BG, V] ≃ Rep(G)` equivalence transported through `Alg(C, −)` — with the failure mode (non-action monads) characterizing its exact scope. The base ⊗ fiber composition is unified by the action groupoid (Prop 5); the scope is sharp (Theorem 6); symmetries acting on the broadcasting are handled (Props 7–8, recovering DeepSets); and compact continuous groups are covered modulo standard enriched scaffolding (Prop 9). The sole open residue is **non-compact `G`** — where semisimplicity, a canonical action monad, and monadicity all fail, and a direct-integral grading over the unitary dual would be required.

---

## Appendix — Next steps

Phases 1–4 are done bar the non-compact analytics. The natural continuations, by theme and reachability:

**A. Formalization (Phase 5), layered.**

- **A1 (reachable now).** Formalize Theorem 1 + Prop 1′ for finite `G` in Lean/Mathlib: `Action V G ≃ V^{T_G}` monoidally, via `CategoryTheory.Action`, `Rep`, `Monad.Algebra`, and the monoidal structure on `Action`. Self-contained; Mathlib likely already has most pieces.
- **A2 (reachable now).** Theorem 6 (recognition) for finite `G` — Mathlib has Barr–Beck monadicity; the uniqueness-of-monad argument is short.
- **A3 (gated).** Lemma 2 + Theorem 2 need the graded colored PROP, the actegory, and `Alg(C, −)` formalized first — i.e. [graded_prop.md §10](graded_prop.md#10-lean-formalization-notes)'s own program. A3 waits on that.
- **Out of reach.** Anything continuous (no enriched/smooth category theory in Mathlib).

**B. Close the non-compact residue (4.3).** Develop the direct-integral generalization: an index that is a *measurable field of colors* (the unitary dual `Ĝ` with Plancherel measure) rather than a discrete set, with "objects = direct integrals" replacing "objects = direct sums." This extends the graded-colored-PROP definition ([graded_prop.md §3.1](graded_prop.md#31-data)) to a measure-enriched setting and connects to von Neumann algebras / direct-integral decomposition. Research-grade; the payoff is covering Lorentz / affine / scale symmetries.

**C. Tighten the present results.**

- **C1.** Write the §5.1 counterexamples in full (the free-monoid monad and the action-with-a-relation family) as a proposition, making *necessity* airtight rather than sketched.
- **C2.** Catalog which symmetries-on-the-broadcasting admit the Prop 8 distributive law `T_G ⊛ ⇒ ⊛ T_G` beyond the permutation case (Prop 7) — e.g. gauge / local symmetries — and which do not.

**D. Propagate the closure into the main documents.**

- Promote Theorem 6 (recognition) into [graded_prop.md](graded_prop.md) as a numbered proposition beside Prop 8.4 (which currently only *points* here).
- The group / `Rep(G)` / graph rows of [future_ideas.md §6.5](future_ideas.md#65-swapping-the-index-d-as-a-dial-across-ml) and [Appendix A](future_ideas.md#appendix-a--index-categories-d-in-detail) now have a precise equivariance theorem behind them — annotate them.

**E. Run the same lens on the other `D`-rows.** The recognition + symmetry analysis here was carried out for `D =` group. Each other [§6.5](future_ideas.md#65-swapping-the-index-d-as-a-dial-across-ml) row deserves the same:

- **graph** — GNN node-permutation-equivariance *is* Prop 7 with `G = S_n` acting on the node degree, so the DeepSets recovery extends to "GNN equivariance = symmetric aggregation over neighbours." Worth stating explicitly.
- **metric / `Stoch` / partition** — identify the symmetry monad (if any) per row and whether the routes coincide. This turns §5 into a per-`D` program.

**F. The CDL bridge.** Theorem 6 sharpens [graded_prop.md §11](graded_prop.md#11-relation-to-categorical-deep-learning)'s relation to Categorical Deep Learning: CDL encodes equivariance as monad-algebras; Theorem 6 says *exactly when* that monad route equals the grading route (iff the monad is an action monad). Writing this out makes the two frameworks' equivariance accounts precisely comparable.

**Suggested ordering.** A1/A2 (finite-`G` Lean) and C1 (full counterexample) are immediate and self-contained; D and E are low-effort write-ups propagating what is already proved; B and F are research-grade.

---

## References

- [graded_prop.md](graded_prop.md) — the graded colored PROP framework; Prop 8.4 (the open problem), §9.3 (`D`-menu).
- [future_ideas.md §6.5](future_ideas.md#65-swapping-the-index-d-as-a-dial-across-ml) and [Appendix A](future_ideas.md#appendix-a--index-categories-d-in-detail) — the three routes and the `Rep(G)` detail.
- Mac Lane, *Categories for the Working Mathematician* — monads, Eilenberg–Moore, monadicity.
- Bruguières, Lack, Virelizier, "Hopf monads on monoidal categories", *Advances in Mathematics* 227, 2011 — when `V^T` is monoidal and `U` strong monoidal (H3).
- Beck, "Distributive laws", 1969 — distributive laws `T∘S ⇒ S∘T` ⟺ liftings of `S` to `V^T` (Prop 8).
- Zaheer et al., "Deep Sets", NeurIPS 2017 — permutation-invariant set functions as `ρ(∐ φ)` (recovered by Prop 7).
- Standard: `Rep(G) = [BG, Vect] = k[G]\text{-Mod}` (the action monad's modules); monadicity and uniqueness of the monad of a monadic functor (Theorem 6).
