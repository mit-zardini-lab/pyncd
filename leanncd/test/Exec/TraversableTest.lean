import LeanNCD.Exec.Traversable

namespace LeanNCD

-- TEST: the class and WithUID resolve at the expected types.
#check @WithUID
#check (@TermTraversable.traverseUID : ∀ {α : Type _} [TermTraversable α], (UData → UData) → α → α)

end LeanNCD
