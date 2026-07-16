# E1 — `traverseAxes` Prototype (PredArith slice) — Design

**Status:** approved 2026-07-15. Feeds `writing-plans`.

## Goal

Extend the E1 prototype (`docs/superpowers/specs/2026-07-15-e1-traverseaxes-prototype-design.md`,
implemented on branch `worktree-e1-traverseaxes-prototype`) to `PredArith` — the next-smallest
increment after the `IdxExpr` slice, which closed cleanly with three sound equivalence proofs
but never exercised genuine self-recursion (its only "recursion" was through a `List`, routed
via Mathlib's `Traversable` instance).

**Correction to the prior framing:** earlier conversation described the remaining risk as "the
mutually-recursive `BoolExpr`/`PredArith` cluster." That's inaccurate — `PredArith` recurses
only into itself and `IdxExpr`; `BoolExpr` recurses into itself and `PredArith`, one direction
only (`DSL/Ast.lean:49-64`). There is no cycle, so this is not mutual recursion in Lean's
`inductive ... with ...` sense. The genuinely new risks this slice tests are: (1) self-recursion
within one node type (`PredArith.mul`/`.iabs`), and (2) composing a new traversal with an
already-proven one underneath it (`PredArith.embed` delegates to `IdxExpr.traverseAxes`).

## Scope

**In:** extend the existing `test/DSL/TraverseAxesSpike.lean` (not a new file — `PredArith`'s
traversal calls `IdxExpr`'s) with one `PredArith.traverseAxes` definition, a local `specsPred'`
comparison copy (mirroring `specsPred`, private in `Structural.lean:30-31`), and three
equivalence theorems against `PredArith.mapUID` (public, `Traverse.lean:22-26`), `specsPred'`,
and `predAxisUIDs` (public, `Eval/Contract.lean:14-18`). Reuses the existing `ConstL` type
unchanged — it's already generic over any element type.

**Out:** `BoolExpr` itself (deferred pending this result — even a clean `PredArith` result is
only a strong predictor, not a guarantee, since `BoolExpr`'s `.and`/`.or`/`.not` are the same
shape of self-recursion but not yet tested). `Factor`/`LHSSlot`/`Stmt`/`Decl`/`TLProgram`. No
production files touched.

## Background facts (verified in the codebase)

- `PredArith` (`DSL/Ast.lean:49-53`): `embed : IdxExpr → PredArith`, `mul : PredArith →
  PredArith → PredArith`, `iabs : PredArith → PredArith`. Only `.mul`/`.iabs` self-recurse;
  `.embed` bottoms out in `IdxExpr`.
- `PredArith.mapUID` (`Traverse.lean:22-26`): `partial def`, routes `.embed` through
  `IdxExpr.mapUID`, `.mul`/`.iabs` recurse into themselves.
- `specsPred` (`Structural.lean:30-31`, `private`): `.embed e => specsIdx e | .mul a b =>
  specsPred a ++ specsPred b | .iabs a => specsPred a`.
- `predAxisUIDs` (`Eval/Contract.lean:14-18`, public): same shape as `specsPred`, UID-valued.
- The existing `mapUID` family marks `IdxExpr.mapUID`/`PredArith.mapUID`/`BoolExpr.mapUID` all
  `partial`, even though `IdxExpr.mapUID` has no self-recursion at all to justify it — likely
  stylistic/uniform across the file rather than technically required. Whether
  `PredArith.traverseAxes` needs `partial` to satisfy Lean's termination checker (given its
  recursion is genuinely structural, unlike `IdxExpr`'s) is an open question this prototype
  will answer empirically, not something to assume either way.

## The traversal

```lean
def PredArith.traverseAxes [Applicative f] (g : AxisSpec → f AxisSpec) : PredArith → f PredArith
  | .embed e => PredArith.embed <$> IdxExpr.traverseAxes g e
  | .mul a b => PredArith.mul <$> PredArith.traverseAxes g a <*> PredArith.traverseAxes g b
  | .iabs a  => PredArith.iabs <$> PredArith.traverseAxes g a
```

Two things this genuinely tests for the first time (the `IdxExpr` slice couldn't):
1. **Delegation to an already-proven traversal** — `.embed` calls `IdxExpr.traverseAxes`
   directly, not a re-implementation of its logic.
2. **`<*>` (Applicative apply) for a two-child constructor** — `.mul`'s two recursive calls
   must combine via `Seq.seq`. For `ConstL`, `<*>` is already defined as list-append
   (`f <*> x = ⟨f.run ++ (x ()).run⟩`, from the `IdxExpr` slice's existing instance), which
   matches `specsPred`'s own `.mul` case (`specsPred a ++ specsPred b`) structurally — promising,
   but unverified until it actually elaborates and proves.

## Instantiations and equivalence theorems

```lean
theorem traverseAxes_id_eq_predMapUID (f : UData → UData) (e : PredArith) :
    PredArith.traverseAxes (f := Id) (AxisSpec.mapUID f) e = PredArith.mapUID f e

theorem traverseAxes_const_eq_specsPred (e : PredArith) :
    (PredArith.traverseAxes (f := ConstL (List AxisSpec)) (fun a => ⟨[a]⟩) e).run = specsPred' e

theorem traverseAxes_const_eq_predAxisUIDs (e : PredArith) :
    (PredArith.traverseAxes (f := ConstL (List UID)) (fun a => ⟨[a.uid]⟩) e).run = predAxisUIDs e
```

Expected proof shape: `induction e with | embed e => ... | mul a b iha ihb => ... | iabs a iha
=> ...`. Unlike `IdxExpr`'s `.affine` case (which needed a separate list-induction lemma since
the recursion lived in an auxiliary `List`), `PredArith`'s structural induction gives real
inductive hypotheses (`iha`, `ihb`) directly usable via `rw`/`simp` — a reasonable expectation
of a *simpler* proof than the `IdxExpr` slice needed, not a guess with no basis.

## File layout

- `test/DSL/TraverseAxesSpike.lean` (extend, not new) — append `specsPred'`,
  `PredArith.traverseAxes`, and the three theorems after the existing `IdxExpr` content.
  Already registered in `lakefile.toml`'s `Tests` globs from the prior slice — no new
  registration needed.

## Success criteria (the go/no-go bar)

**Go** (worth extending to `BoolExpr` next):
- `PredArith.traverseAxes` compiles as ordinary structural recursion, or with `partial` if
  needed (matching the existing `mapUID` family's style) — either is acceptable; only genuine
  compile failure is a no-go signal here.
- All three theorems close with genuinely sound proofs (real induction using `iha`/`ihb`, not
  vacuous or circular).
- The `.embed` case's delegation to `IdxExpr.traverseAxes` composes cleanly — no re-proving
  `IdxExpr`'s equivalence from scratch inside `PredArith`'s proofs.

**No-go** (stop extending; treat as a data point against scaling E1 further):
- Structural fighting specifically on the `<*>`-based `.mul` case — universe issues, or a proof
  requiring something much more elaborate than the shape above despite the simpler expected
  induction.

**Explicitly not decided by this slice:** `BoolExpr` itself. Even a clean `PredArith` result is
a strong predictor, not a guarantee — `BoolExpr.and`/`.or`/`.not` are the same shape of
self-recursion but untested until attempted directly.

## Risks / notes

- Same maintenance hazard as the `IdxExpr` slice's `specsIdx'`: the local `specsPred'` copy must
  stay byte-identical to `Structural.lean:30-31`'s private `specsPred` by inspection; re-diff if
  that file changes before this lands.
- If `PredArith.traverseAxes` needs `partial`, check whether that forces `IdxExpr.traverseAxes`
  (already `def`, not `partial`, in the existing file) to also need adjustment for the two to
  compose — unlikely, since `partial` is a property of the *defining* function, not something
  that propagates to callees, but worth a quick sanity check if it comes up.
