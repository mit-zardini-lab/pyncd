# Lean 4 Encoding of the `D`-Graded Colored PROP Framework

This document describes a Lean 4 formalisation of the NCD categorical framework as a **single structure** — the `D`-graded colored PROP of [graded_prop.md](graded_prop.md) — rather than as two independent categories. `St` and `Br` are one instantiation (`D = St`, `C = Br`); the model level (`D = Br`) and the swapped-`D` rows are others. The encoding is a layered tower of typeclasses parameterised by other typeclasses, so the propositions of [graded_prop.md §8](graded_prop.md#8-propositions-the-synthesis-organizes) are stated once, generically, and inherited at every instantiation.

The intent is **formalizability, not formalization**: definitions are given as Lean `class`/`structure` data plus named `Prop`-field laws, in the shape a Lean development transcribes directly. No proofs are written — signatures and proof obligations only.

## Contents

1. [Orientation: one structure, one seam](#1-orientation-one-structure-one-seam)
2. [The base: `ColoredPROP`](#2-the-base-coloredprop)
3. [The seam adapter into Mathlib](#3-the-seam-adapter-into-mathlib)
4. [The core: `DGradedColoredPROP D C`](#4-the-core-dgradedcoloredprop-d-c)
5. [Weaves as cartesian-lift data](#5-weaves-as-cartesian-lift-data)
6. [Mixins: Scan, Route, Symmetry, Para](#6-mixins-scan-route-symmetry-para)
7. [Grothendieck split and composition as pushout](#7-grothendieck-split-and-composition-as-pushout)
8. [Acsets and Python interop](#8-acsets-and-python-interop)
9. [The propositions as generic theorems](#9-the-propositions-as-generic-theorems)
10. [Instantiation and future extensions](#10-instantiation-and-future-extensions)
11. [Lean formalization notes](#11-lean-formalization-notes)
12. [Appendix: out of scope](#12-appendix-out-of-scope)

## 1. Orientation: one structure, one seam

The current state of the design predates [graded_prop.md](graded_prop.md): it formalised `St` and `Br` as two independent PROP instances and divided everything into a "Layer 1 — Mathematical Encoding" and a "Layer 2 — Representation" (UIDs, `Context`, names). That split was an artifact of not yet having the vocabulary `graded_prop.md` now supplies. The reframing dissolves it. The scattered constructions are recognised as **one structure** — a `D`-graded colored PROP — and the two things the old "representation layer" was carrying turn out to be ordinary categorical data: symbolic sizes are the **fiber of the Grothendieck construction** `∫Dat`, and axis identity/alignment is the **pushout/coequalizer** of composition. Neither is a separate representation layer; both live inside the single development. There is **one** development, parametric on an index PROP `D` and an operation PROP `C`, and "prove the propositions once, inherit them everywhere" becomes, in Lean, plain parametricity over a typeclass.

The encoding is therefore a layered tower of typeclasses, each parameterised by the classes below it:

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

The base `ColoredPROP` carries the lightweight definitions and the `St`/`Br` instances; the core `DGradedColoredPROP D C` adds the grading data and laws; capabilities (`Scan`, `Route`, symmetry, `Para`) are composable mixins, and `Algebra` is the `construct()` functor into a target actegory. An instantiation pays only for the layers it declares, and the proven core is never edited — new domains are new instances, new capabilities are new mixins.

The single seam that remains is **not** the old "is this mathematics?" boundary but a thinner, differently-drawn one — the **proposition/computation** seam: **does Lean *prove* this or *compute* this?** This is the seam [graded_prop.md](graded_prop.md) itself draws. [§6](graded_prop.md#6-composition-as-pushout) is "a correctness/specification lens, not a composition algorithm": the pushout *explains and certifies* what `Context` does, it does not replace it — so the coequalizer is the *specification* and union-find plus a fresh-name counter is the *implementation*. [§7](graded_prop.md#7-algebras-construct-and-the-para-refinement) keeps the lightweight `Para` encoding and tells us to **note the gap explicitly rather than pay for a bicategory only the specification uses**. Read as the organizing principle of this document: the propositional core (PROP/actegory laws, `∫Dat`, pushout-as-colimit, equivariance, weave uniqueness) is stated over UID-free types and proved; the executable realization (fresh-UID counter, union-find, acset tables / CSV) sits on the other side of the seam, realizing the specification without being proved against it line by line. The seam adapter of §3 is exactly this boundary turned into a definition.
