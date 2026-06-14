import Mathlib
import LeanNCD.Exec.Context

namespace LeanNCD

-- Test-only TermTraversable instances (the real AST instances are Milestone E).
instance : TermTraversable UData := ⟨fun f u => f u⟩
instance : TermTraversable (List UData) := ⟨fun f => List.map (TermTraversable.traverseUID f)⟩

-- traverseUID maps over a single UData (exercises the class + UData instance).
#guard (TermTraversable.traverseUID (fun d => { d with uid := d.uid + 1 }) (⟨5, none⟩ : UData)).uid == 6

-- merge into the empty context just inserts the class.
private def c1 : Context UData :=
  (Context.mk []).merge ⟨{1, 2}, ⟨⟨2, none⟩, 2⟩⟩

#guard c1.classes.length == 1

-- merging an overlapping class unions the buckets and keeps the LARGEST-UID canonical.
private def c2 : Context UData := c1.merge ⟨{2, 3}, ⟨⟨3, none⟩, 3⟩⟩

#guard c2.classes.length == 1
#guard (c2.classes.head?.map (·.bucket) == some ({1, 2, 3} : Finset UID))
#guard (c2.classes.head?.map (·.canonical.uid) == some 3)

-- a non-overlapping class stays separate.
private def c3 : Context UData := c2.merge ⟨{7, 8}, ⟨⟨8, none⟩, 8⟩⟩
#guard c3.classes.length == 2

end LeanNCD
