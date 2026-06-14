import LSpec
import Mathlib
import LeanNCD.Exec.Context

namespace LeanNCD
open LSpec

-- Test-only TermTraversable instance (mirrors ContextTest.lean; the real AST
-- instances arrive in Milestone E). Needed for `Context.apply` on a `UData`.
instance : TermTraversable UData := ⟨fun f u => f u⟩

private def ctxA : Context UData := (Context.mk []).merge ⟨{1, 3}, ⟨⟨3, none⟩, 3⟩⟩

private def merged : Context UData :=
  ((Context.mk []).merge ⟨{1, 2}, ⟨⟨2, none⟩, 2⟩⟩).merge ⟨{2, 3}, ⟨⟨3, none⟩, 3⟩⟩

#lspec group "Context union-find" <|
  test "merge keeps the largest-uid canonical"
      (merged.classes.head?.map (·.canonical.uid) = some 3) $
  test "merge unions overlapping buckets"
      (merged.classes.head?.map (·.bucket) = some ({1, 2, 3} : Finset UID)) $
  test "apply substitutes a member uid"
      ((Context.apply ctxA (⟨1, none⟩ : UData)).uid = 3) $
  test "apply leaves a non-member uid"
      ((Context.apply ctxA (⟨9, none⟩ : UData)).uid = 9)

end LeanNCD
