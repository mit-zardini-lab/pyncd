import LeanNCD.Exec.Uid

namespace LeanNCD

universe u

/-- A value paired with the UID of a representative. The `uid` is stored separately from `data`
    because a general `α` need not carry a uid field (for `α = UData` they coincide). -/
structure WithUID (α : Type u) where
  data : α
  uid  : UID

/-- Types whose UID references can be traversed and substituted. One instance per decorated type;
    the real instances (AxisSpec/IdxExpr/…/BrBase/ThreadedComposed) are written in Milestone E.
    Laws (for the instances to satisfy): `traverseUID id = id`,
    `traverseUID (f ∘ g) = traverseUID f ∘ traverseUID g`. -/
class TermTraversable (α : Type u) where
  traverseUID : (UData → UData) → α → α

end LeanNCD
