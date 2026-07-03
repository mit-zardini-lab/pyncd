import LeanNCD.Base.Br             -- BrObj, BrMorph
import LeanNCD.Acset.SBrInstance   -- the full-fidelity Acset.SBrInstance (Milestone F)
import LeanNCD.Bridge.Realize       -- realize, ThreadedComposed.WellFormed
import LeanNCD.Bridge.AcsetCodec    -- toThreadedComposed (decode direction, Task B)

namespace LeanNCD

/-- Realize the full-fidelity acset `SBrInstance` (Milestone F; the CSV path's data structure) as a
    `Br` morphism, by DECODING it back to a `ThreadedComposed` (`toThreadedComposed`, Task B) and
    replaying the already-proved DSL-path `realize` fold. Total over any `s`: on a decoded `tc` that
    is `WellFormed` it is `realize tc`; otherwise (malformed/garbage tables — never produced by
    `fromThreadedComposed` of a real program) it falls back to the empty identity morphism.
    `Decidable tc.WellFormed` is supplied classically (`realizeSBr` is `noncomputable` regardless).
    The §8 agreement (`realize_fromThreadedComposed_agree`) shows this coincides with the DSL-path
    `realize` on the nose for compiled programs (via the `toThreadedComposed ∘ fromThreadedComposed`
    round trip). (Supersedes E2b's minimal placeholder SBrInstance.) -/
noncomputable def realizeSBr (s : Acset.SBrInstance) : Σ (dom cod : BrObj), BrMorph dom cod :=
  let tc := AcsetCodec.toThreadedComposed s
  haveI : Decidable tc.WellFormed := Classical.propDecidable _
  if h : tc.WellFormed then realize tc h else ⟨[], [], BrMorph.idm []⟩

end LeanNCD
