-- LeanNCD/Bridge/Agreement.lean
import LeanNCD.Bridge.Realize
import LeanNCD.Bridge.SBr

namespace LeanNCD

/-- §8.2 acset extraction (`from_tensor_program`): a ThreadedComposed's tabular twin.
    OBLIGATION (`sorry`): the extraction algorithm (acset.md). -/
noncomputable def fromThreadedComposed (tc : ThreadedComposed) : SBrInstance := sorry

/-- **Prop 8 (DSL/CSV agreement).** The DSL-path realization of `tc` and the CSV-path
    realization of its extracted `SBrInstance` are the SAME `Br` morphism (equal as
    `Σ (dom cod : BrObj), BrMorph dom cod` values — same objects AND same morphism). -/
theorem realize_fromThreadedComposed_agree (tc : ThreadedComposed) :
    realize tc = realizeSBr (fromThreadedComposed tc) := sorry

/-- **Prop 8′ (axis identity on the nose, §7.4).** Both paths share the §7.4 UID coequalizer,
    so the realized domain objects coincide. -/
theorem agree_dom (tc : ThreadedComposed) :
    (realize tc).1 = (realizeSBr (fromThreadedComposed tc)).1 := sorry

/-- Prop 8′ (cod). -/
theorem agree_cod (tc : ThreadedComposed) :
    (realize tc).2.1 = (realizeSBr (fromThreadedComposed tc)).2.1 := sorry

end LeanNCD
