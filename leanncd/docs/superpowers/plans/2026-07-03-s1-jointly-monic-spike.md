# Spike S1 — `naperian_jointly_monic` circularity (Milestone 1.5)

**Question (from `papers/NaperianTypingIntegrationPlan.md` §0.2 / §5 M1.5):**
Is there an **acyclic** proof of `naperian_jointly_monic` — that lifted `P`-indexed families are
separated by their `ev_p` coordinate evaluations — that does **not** route through `broadcast_gen`?

**Exit criterion:** an acyclic proof/skeleton, or the explicit extra assumption it needs (confirmed
independent of `broadcast_gen`).

**Verdict: POSITIVE — acyclic route established, `broadcast_gen` not needed.** The proof is not
*free* (it reroutes the obligation, it does not vaporize it), but it reroutes it toward
representability / St-slice separation — provable from the already-sorry-free `St.elemental` — and
verifiably away from `broadcast_gen`. The circularity hazard S1 exists to catch is avoided.

---

## What was verified (in the actual LeanNCD setting)

`ev_p` (`Core/Graded.lean:90`) is `act(𝟙_X, p) ; υ_X : X ⊛ P ⟶ X`, indexed by points
`p : P ⟶ Opposite.op (unit_D)` (the concrete `El P` = that hom-set). Joint monicity of the `ev_p`
family is exactly "two maps into `X ⊛ P` agreeing on every slice are equal."

The abstract fact — two morphisms into a product/power are equal iff they agree after every
projection — is `CategoryTheory.Limits.IsLimit.hom_ext` (and `Pi.hom_ext`); it holds in **any**
category and depends only on `[propext, Classical.choice, Quot.sound]`.

Tying it to the real `ev_p`, the following was written against the project, compiled clean, and
axiom-checked (`lake env lean`):

```lean
-- El P = the point hom-set; the ev_p family assembled as a Fan over the discrete El-diagram.
abbrev ElP (P : Dᵒᵖ) := P ⟶ Opposite.op (ColoredPROP.unit : D)

noncomputable def evFan (X : C) (P : Dᵒᵖ) : Limits.Fan (fun _ : ElP P => X) :=
  Limits.Fan.mk (DGradedColoredPROP.act.obj (X, P)) (fun p => ev_p p X)

-- ACYCLIC REDUCTION: representability (lifted object is the El P-power, legs = ev_p) ⇒ jointly monic.
theorem naperian_jointly_monic_of_repr (X : C) (P : Dᵒᵖ)
    (hrepr : Limits.IsLimit (evFan X P))
    {W : C} (f g : W ⟶ DGradedColoredPROP.act.obj (X, P))
    (h : ∀ p : ElP P, f ≫ ev_p p X = g ≫ ev_p p X) : f = g := by
  apply hrepr.hom_ext
  rintro ⟨p⟩
  exact h p
```

`#print axioms naperian_jointly_monic_of_repr` ⇒ **`[propext, Classical.choice, Quot.sound]`** — no
`sorryAx`, nothing from `broadcast_gen`, nothing from any `Br` normal form. (Scratch file removed
after verification per the no-scratch-in-repo-root rule; this skeleton is the durable record and
transplants directly into `Core/Naperian.lean` when Track B proceeds.)

## The honest nuance: reroute, not elimination

`IsLimit.hom_ext` delivers `jointly_monic` **from** the limit structure. But constructing that
`IsLimit (evFan X P)` requires its `uniq` field, which — specialized — *is* `jointly_monic`. So the
reduction does **not** make the obligation vanish; it identifies `jointly_monic` as the
**uniqueness half of the representability (power) structure** of the lifted object.

Why that is nonetheless the positive S1 answer:

- The circularity S1 tests for is specifically **`jointly_monic` ⟵ `broadcast_gen`** (while
  `broadcast_gen` is a thing Naperian hopes `jointly_monic` helps prove). The verified route sources
  `jointly_monic` from the **power/representability structure**, a *different* and independent place.
  So even if `broadcast_gen` is later proved *using* `jointly_monic`, there is no cycle:
  `jointly_monic ← representability`, never `jointly_monic ← broadcast_gen`.
- Representability (existence half = `tabulate`, "assemble a tensor from its `El P` slices") is a
  concrete `Br` weave construction — stacking slices — **not** the general factorization theorem
  `broadcast_gen`. So discharging the residual does not smuggle `broadcast_gen` back in.

## The residual obligation (and why it looks tractable)

What remains to actually *close* `jointly_monic` for the flagship `C = Br` instance:

> Separate two maps `f, g : W ⟶ X ⊛ P` by their `El P`-slices — i.e. prove the `uniq`/joint-monicity
> field of `IsLimit (evFan X P)` for the concrete `act.obj (X,P) = X.map (append P)`.

The route: a `Br` map into a per-array-appended object is carried by its `reindexings` (which are
**`St` morphisms**, `BrBase.reindexings`), and slicing at a point `p` evaluates those `St` reindexings
at `p`. So "separated by slices" reduces to **`St` morphisms separated by point-evaluations = `St`
elementality**, which is `St.elemental` — **already proved sorry-free** (`Base/St.lean:364`). The
encouraging priors from S0 also hold: `ev_p` and `ev_p_naturality` (Eq. 3) already exist sorry-free
in `Core/Graded.lean`.

**This cannot be fully discharged in a spike** because it needs the concrete `act.obj`/`act.map` to
exist (the ~400–650-line S0 build). It does **not** re-introduce `broadcast_gen`, and via the
representability route it stays off the stuck `brCancelPoint` milestone — which was the whole point.
**⚠️ See the "Residual probe" addendum below**, which sharpens this: the `St.elemental` reduction is
clean only for *single-`BrBase`* morphisms; the general case needs the concrete finite-`El`
representability construction (a real, post-solver build), not a cheap `St.elemental` corollary.

## Consequences

- **Track B's value gate is cleared on the circularity question.** `jointly_monic` has a verified
  acyclic route; the projected coherence-sorry reduction is not blocked by a hidden cycle.
- **`brCancelPoint` stays off the path.** The residual routes through `St.elemental` (proved), not
  `Br` elementality — confirming Naperian's central bet.
- **Remaining work is inside the `act`/`NaperianFamily` build**, not a separate obstruction:
  constructing `act.obj`/`act.map` (S0-scoped) and the `tabulate` half of representability, then
  proving the `uniq` field via the `St.elemental` reduction above.

## Recommendation

S1 is a **go**. The next concrete step for Track B is to build the minimal `act.obj`/`act.map` +
`ev_p` for `Br`, then attempt the residual `uniq` proof (St-slice separation) — if that lands,
`IsLimit (evFan)` is constructible and `jointly_monic` closes for real.

---

## Residual probe (2026-07-03) — sharpening the residual cost

Attempted to run the "St-slice-separation" step as a cheap follow-on spike. Two things were checked
in Lean against the real `Br` types (scratch files, since removed):

1. **The object action is trivially definable.** `liftObj P X := X.map (liftArray P)` with
   `liftArray P A := {A with shape := A.shape ++ P}`; `liftObj P (X ++ Y) = liftObj P X ++ liftObj P Y`
   by `simp [List.map_append]`.
2. **`El P` is NOT finitely enumerable symbolically.** `Fintype (BrMorph [] P)` — the point hom-set,
   the index of the `ev_p` family — has **no derivable instance** (`exact?` fails). It is a quotient
   of an inductive with symbolic-size objects.

Two consequences downgrade the earlier optimism that the residual is a *cheap* `St.elemental`
corollary:

- **A faithful `evSlice` is not free — it is the reindex-half of `act`.** Building the slice morphism
  `X ⊛ P ⟶ X` as a `BrBase` requires the full `degree`/`inputWeaves`/`outputWeaves`/`reindexings`
  apparatus (cf. `realizeBrBaseP`), i.e. the `act(id, η)` reindexing machinery. So even *stating* the
  concrete separation lemma needs part of the ~400–650-line `act` build; there is **no faithful
  slice-separation probe that is cheaper than starting that build.**
- **The clean structural route is a CONCRETE / post-solver fact, not symbolic.** Joint monicity via
  the product framing needs `X ⊛ P ≅ ∏_{p ∈ El P} X` (a tensor is the stack of its slices). That
  requires a **finite, enumerated `El P`** — which, per probe (2) and the project's symbolic-vs-
  concrete split, is only available *after* size solving (`AxisPointData`/`FiniteNaperian`). The
  `tabulate`/`lookup` iso that witnesses it is a real construction, not a corollary.
- **The `St.elemental` reduction is real only for single-`BrBase` morphisms.** A single generator's
  content *is* its `St` `reindexings`, so slicing evaluates those and `St.elemental` separates them.
  But a general `W ⟶ X ⊛ P` is a `BrMorph` **composite** (a quotient term); reducing *it* to
  `St.elemental` by induction on the quotient hits the **same `trans`-case wall as `brCancelPoint`**.
  The escape is the representability/product route above — which is exactly why representability, not
  `St.elemental` alone, is the load-bearing residual.

**Refined verdict.** S1's headline stands: the acyclic reduction is verified and `jointly_monic` is
**not** blocked by a `broadcast_gen` cycle, and the representability route keeps it off `brCancelPoint`
too. But the residual is **not a cheap spike** — it is the concrete finite-`El` representability
construction (`tabulate`/`lookup` iso + its laws), sized *with* the `act` build and living in the
post-solver layer. There is **no further cheap probe** that retires this risk short of building
(i) the reindex-half of `act` and (ii) the concrete finite-`El` representability. That is
decision-relevant: Track B's last risk is real work, not something a spike can pre-clear.

**Consequently, the "next spike-sized checkpoint" framing is retracted:** the next step is a
*commitment-sized* build (act reindex-half + concrete finite-`El` representability), not a probe.
Choose it as a funded Track-B increment, not as further de-risking.
