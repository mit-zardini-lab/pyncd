import Mathlib
import LeanNCD.Exec.Traversable

namespace LeanNCD

/-- One equivalence class: a finite set of UIDs with one canonical representative (the member with
    the largest UID — the universal cocone vertex of the §7.3 pushout). -/
structure EqClass (α : Type*) where
  bucket    : Finset UID
  canonical : WithUID α

/-- A context is a disjoint list of equality classes. -/
structure Context (α : Type*) where
  classes : List (EqClass α)

/-- Merge a new class in, unioning with any overlapping existing classes and keeping the
    largest-UID representative (Python's `Context.append_bucket`). -/
def Context.merge (ctx : Context α) (cls : EqClass α) : Context α :=
  let overlaps : EqClass α → Bool := fun c => ! decide (Disjoint c.bucket cls.bucket)
  let overlapping := ctx.classes.filter overlaps
  let merged : EqClass α := overlapping.foldl
    (fun acc c => ⟨acc.bucket ∪ c.bucket,
                   if acc.canonical.uid ≥ c.canonical.uid then acc.canonical else c.canonical⟩)
    cls
  ⟨merged :: ctx.classes.filter (fun c => ! overlaps c)⟩

end LeanNCD
