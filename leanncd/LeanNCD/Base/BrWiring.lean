import LeanNCD.Base.Br

/-! # `BrWiring` — the comonoid-skeleton "wiring" morphisms of `Br`.

Pure-routing `BrMorph`s built only from `id`/`braid`/`copyW`/`delW` (no `gen` nodes): copy a wire,
discard a wire, permute wires. These realize the fan-out/permute/discard plumbing the `realize` fold
needs (realize.md §7). The cast work is confined to the gather primitive `rotateToFront` (and its
inverse); everything above it is definitional.

The headline is `wiring s sel : BrMorph s (sel.map s.get)` — a type-preserving selection where a
repeated source index copies, an absent one discards, and the order is the target layout. -/

namespace LeanNCD
namespace BrMorph

/-- The two list identities behind `rotateToFront`: splitting `l` at position `i`. -/
private theorem rot_dom (l : BrObj) (i : Fin l.length) :
    (l.take i.val ++ [l.get i]) ++ l.drop (i.val + 1) = l := by
  rw [List.append_assoc, List.singleton_append, List.get_eq_getElem,
      ← List.drop_eq_getElem_cons i.isLt, List.take_append_drop]

private theorem rot_cod (l : BrObj) (i : Fin l.length) :
    ([l.get i] ++ l.take i.val) ++ l.drop (i.val + 1) = l.get i :: l.eraseIdx i.val := by
  rw [List.append_assoc, List.singleton_append, ← List.eraseIdx_eq_take_drop_succ]

/-- Gather the wire at position `i` to the front, keeping the rest in order — a single `braid`
    retyped across the split identities. -/
noncomputable def rotateToFront (l : BrObj) (i : Fin l.length) :
    BrMorph l (l.get i :: l.eraseIdx i.val) :=
  cast (congrArg₂ BrMorph (rot_dom l i) (rot_cod l i))
    (tensor (braid (l.take i.val) [l.get i]) (idm (l.drop (i.val + 1))))

/-- Inverse of `rotateToFront`: return the front wire to position `i`. -/
noncomputable def rotateFromFront (l : BrObj) (i : Fin l.length) :
    BrMorph (l.get i :: l.eraseIdx i.val) l :=
  cast (congrArg₂ BrMorph (rot_cod l i) (rot_dom l i))
    (tensor (braid [l.get i] (l.take i.val)) (idm (l.drop (i.val + 1))))

/-- Discard an entire bundle. -/
noncomputable def delAll : (s : BrObj) → BrMorph s []
  | []      => idm []
  | A :: rest => tensor (delW A) (delAll rest)

/-- Emit one copy of the wire at position `i` to the front, leaving the source bundle intact:
    gather it, copy, send the original back. -/
noncomputable def pick (s : BrObj) (i : Fin s.length) : BrMorph s (s.get i :: s) :=
  comp (rotateToFront s i)
    (comp (tensor (copyW (s.get i)) (idm (s.eraseIdx i.val)))
          (tensor (idm [s.get i]) (rotateFromFront s i)))

/-- The wiring morphism: realize a type-preserving selection `sel` of `s`'s wires as a `BrMorph`.
    Repeated index ⇒ copy; absent index ⇒ discard; order ⇒ target layout. Built by emitting each
    target wire via `pick` (source kept intact) and discarding the source at the end — so all casts
    live in `rotateToFront`/`rotateFromFront` and the recursion itself is definitional. -/
noncomputable def wiring (s : BrObj) : (sel : List (Fin s.length)) → BrMorph s (sel.map s.get)
  | []        => delAll s
  | i :: rest => comp (pick s i) (tensor (idm [s.get i]) (wiring s rest))

/-! ### Wiring by element identity

When the wires carry identities `α` (with a typing `f : α → ArrayType`), it is cleaner to select by
identity than by position: the codomain `tgt.map f` is then exact *by construction* — no `idxOf`,
no `getElem` lemmas, no casts. Recursion on the pool with a decidable head-match. -/

variable {α : Type} [DecidableEq α]

/-- Emit one copy of element `w` (found in `pool`) to the front, keeping `pool` intact. Recursion on
    `pool`: copy at the head when `w` matches, else recurse and braid `w` forward. -/
noncomputable def pickBy (f : α → ArrayType) :
    (pool : List α) → (w : α) → w ∈ pool → BrMorph (pool.map f) (f w :: pool.map f)
  | [],      _, hmem => nomatch hmem
  | p :: ps, w, hmem => by
      by_cases hwp : w = p
      · subst hwp
        exact tensor (copyW (f w)) (idm (ps.map f))
      · have hps : w ∈ ps := (List.mem_cons.1 hmem).resolve_left hwp
        exact comp (tensor (idm [f p]) (pickBy f ps w hps))
                   (tensor (braid [f p] [f w]) (idm (ps.map f)))

/-- Wiring by identity: realize a target wire-list `tgt` (each in `pool`) as a `BrMorph` between the
    `f`-typed bundles. Repeated element ⇒ copy, absent ⇒ discard, order ⇒ layout — all definitional. -/
noncomputable def wiringBy (f : α → ArrayType) (pool : List α) :
    (tgt : List α) → (∀ w ∈ tgt, w ∈ pool) → BrMorph (pool.map f) (tgt.map f)
  | [],        _    => delAll (pool.map f)
  | w :: rest, hmem =>
      comp (pickBy f pool w (hmem w (List.mem_cons.2 (Or.inl rfl))))
           (tensor (idm [f w]) (wiringBy f pool rest (fun x hx => hmem x (List.mem_cons.2 (Or.inr hx)))))

end BrMorph
end LeanNCD
