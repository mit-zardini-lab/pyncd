# E1 — `traverseAxes` Prototype (IdxExpr slice) — Design

**Status:** approved 2026-07-15. Feeds `writing-plans`.

## Goal

Resolve the go/no-go decision the doc's own sequencing table requires before Spike 2 can
proceed in its plain shared-functions form: **E1** (the van Laarhoven one-traversal encoding)
must be prototyped first, since — if it works — it replaces Spike 2a/2b entirely and makes
`TermTraversable` (`Exec/Traversable.lean`) its `Id` special case; only if it fails should the
team commit to 2a/2b's plain-functions dedup.

This spec covers the smallest slice that can give a real signal: one `traverseAxes` definition
for `IdxExpr` alone (no mutual recursion, no wiring into production), proven — not just
spot-checked — equivalent to the three existing hand-written functions it would subsume:
`IdxExpr.mapUID` (the remap/rebuild use, `DSL/Traverse.lean:14-19`), `specsIdx` (the
`AxisSpec`-collecting use, `Structural.lean:26-27`), and `idxAxisUIDs` (the UID-collecting use,
`Eval/Contract.lean:7-12`).

**Non-goal:** deciding whether E1 scales to the mutually-recursive `BoolExpr`/`PredArith`
cluster, or to `Factor`/`LHSSlot`/`Stmt`/`Decl`/`TLProgram` — that is a harder, separate risk
this slice cannot surface either way, and is explicitly left to a follow-up spike if this one
succeeds.

## Scope

**In:** a single new isolated test file containing `IdxExpr.traverseAxes`, a minimal local
`Const`-style `Applicative` for collecting into `List α`, and three equivalence theorems.

**Out:** no change to `Structural.lean`, `Contract.lean`, `Traverse.lean`, or
`Exec/Traversable.lean`. `specsIdx` stays `private` in `Structural.lean` — the spike duplicates
its 3-line body locally for comparison rather than changing its visibility. No BoolExpr/Factor/
LHSSlot/Stmt coverage.

## Background facts (verified in the codebase)

- `TermTraversable.traverseUID : (UData → UData) → α → α` (`Exec/Traversable.lean:17-18`) is
  the existing remap abstraction; `IdxExpr.mapUID` (`DSL/Traverse.lean:14-19`) implements it by
  routing every `AxisSpec` occurrence through `AxisSpec.mapUID f` (`DSL/Traverse.lean:8-9`),
  which builds a `UData` from the *whole* `AxisSpec` (`⟨a.uid, some a.name⟩`), applies
  `f : UData → UData`, and writes only the resulting `.uid` back (kind/name preserved from the
  original). This means E1's `g : AxisSpec → f AxisSpec` argument, instantiated at `Id` with
  `g := AxisSpec.mapUID f`, is not a *different* occurrence view from what `mapUID` already
  uses — it is the same wrapping, which substantially de-risks the "two occurrence views must
  be reconciled" concern the doc raises, at least for this slice.
- `specsIdx : IdxExpr → List AxisSpec` (`Structural.lean:26-27`) and
  `idxAxisUIDs : IdxExpr → List UID` (`Eval/Contract.lean:7-12`) are structural twins —
  `idxAxisUIDs` is `specsIdx` post-composed with `(·.uid)` — confirming the doc's claim that the
  `AxisSpec`-valued and UID-valued collector families are the same walk at two projections.
- Mathlib already ships the exact abstraction the doc describes:
  `class Traversable (t : Type u → Type u) extends Functor t` with `traverse`
  (`Mathlib/Control/Traversable/Basic.lean:195`) and a `List` instance
  (`Mathlib/Control/Traversable/Instances.lean`) — the `.affine` case's inner
  `List (Int × AxisSpec)` traversal can reuse this instead of hand-rolling list traversal.
- Mathlib also ships `Functor.Const α` (`Mathlib/Control/Functor.lean:66-`), but its
  `Applicative` instance requires a `Monoid` on `α`; `List`'s natural monoid there would need
  wrapping in `Multiplicative`/`Additive`. Hand-rolling a minimal local `Const`-style type
  scoped to `List`-append is less ceremony and matches the doc's own estimate ("~5 lines").

## The traversal

```lean
def IdxExpr.traverseAxes [Applicative f] (g : AxisSpec → f AxisSpec) : IdxExpr → f IdxExpr
  | .axis a       => .axis <$> g a
  | .const n      => pure (.const n)
  | .scale c a    => .scale c <$> g a
  | .shift a n    => (.shift · n) <$> g a
  | .affine n xs  => .affine n <$> xs.traverse (fun (c, a) => Prod.mk c <$> g a)
```

## The `Const` type

```lean
structure ConstL (α : Type) (β : Type) where run : α
instance : Functor (ConstL (List γ)) where map _ x := ⟨x.run⟩
instance : Applicative (ConstL (List γ)) where
  pure _ := ⟨[]⟩
  seq f x := ⟨f.run ++ (x ()).run⟩
```

## Instantiations and equivalence theorems

```lean
-- remap (subsumes IdxExpr.mapUID)
theorem traverseAxes_id_eq_mapUID (f : UData → UData) (e : IdxExpr) :
    (IdxExpr.traverseAxes (f := Id) (AxisSpec.mapUID f) e) = IdxExpr.mapUID f e

-- collect AxisSpecs (subsumes specsIdx)
theorem traverseAxes_const_eq_specsIdx (e : IdxExpr) :
    (IdxExpr.traverseAxes (f := ConstL (List AxisSpec)) (fun a => ⟨[a]⟩) e).run
      = specsIdx' e   -- specsIdx' : local copy of Structural.lean's private specsIdx, for comparison only

-- collect UIDs (subsumes idxAxisUIDs)
theorem traverseAxes_const_eq_idxAxisUIDs (e : IdxExpr) :
    (IdxExpr.traverseAxes (f := ConstL (List UID)) (fun a => ⟨[a.uid]⟩) e).run
      = idxAxisUIDs e
```

Expected proof shape: `induction e <;> rfl` or a light `simp [IdxExpr.traverseAxes, ...]` per
theorem — `IdxExpr` has no self-recursion outside the `.affine` list case, which routes through
`List`'s existing `Traversable` instance rather than direct recursion, so none of these should
require induction machinery beyond Lean's default case-split.

## File layout

- `test/DSL/TraverseAxesSpike.lean` — the traversal, `ConstL`, the local `specsIdx'` comparison
  copy, and the three theorems. Registered in `lakefile.toml`'s `Tests` globs
  (`DSL.TraverseAxesSpike`) so a failed proof fails `lake build`, same as every other test.
  Imports: `LeanNCD.DSL.Traverse` (`AxisSpec.mapUID`/`IdxExpr.mapUID`), `LeanNCD.Eval.Contract`
  (`idxAxisUIDs`), `Mathlib.Control.Traversable.Instances` (`List`'s `Traversable` instance).

## Success criteria (the go/no-go bar)

**Go** (E1 worth scaling further; revisit before committing to Spike 2a/2b):
- All three theorems prove without structural fighting — no universe issues, no typeclass
  diamond between Mathlib's `Applicative`/`Traversable` hierarchy and `ConstL`, no need for
  `partial` or well-founded-recursion workarounds.
- The reconciliation holds as expected: `g := AxisSpec.mapUID f` at `Id` reproduces `mapUID f`
  without any field-dropping or round-tripping hack.

**No-go** (fall back to Spike 2a/2b for the full dedup effort, at least for now):
- Two or more of the above hit real friction — not "needs a few more lines," but structural
  pain (Lean won't unify `f = Id` transparently, or `ConstL`'s instances fight typeclass
  resolution against Mathlib's own instances in scope).

**Explicitly not decided by this prototype either way:** whether E1 scales to the
mutually-recursive `BoolExpr`/`PredArith` cluster, which is the harder unresolved risk this
narrow slice cannot surface. A clean result here is encouraging but not sufficient on its own
to commit to full E1 across the AST; a bad result here is sufficient to stop and fall back.

## Risks / notes

- `specsIdx` is `private` — the local `specsIdx'` comparison copy in the test file must be kept
  byte-identical to `Structural.lean:26-27`'s body by inspection; if `Structural.lean` changes
  before this spike lands, re-diff before trusting the theorem.
- If Lean's elaborator needs an explicit `(f := Id)` / `(f := ConstL (List AxisSpec))` annotation
  at each call site (likely, since `g`'s codomain alone may not pin `f` down uniquely), that is
  expected and not itself a no-go signal — only genuine unification failure counts.
