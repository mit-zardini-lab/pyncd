/-
# LeanNCD — a Lean 4 encoding of the D-graded colored PROP framework

The library has **two tracks** that meet at a **bridge**. Internalising that split first makes
every individual module easier to read; the import list below is grouped to match it.

## Track 1 — the categorical "math tower"  (NONCOMPUTABLE; the proof world)

The categorical structure and its coherence theorems. Types here carry symbolic, noncomputable
ring values (`Numeric = MvPolynomial String ℕ` for sizes, `Coeff = MvPolynomial String ℤ` for
reindex coefficients), `Fin n →` function fields, and dependent typing — so they are
`noncomputable` and exist only to be reasoned about.

  Base/Category, Base/ColoredPROP  the `ColoredPROP` class (a colored PROP) + `Elemental`
  Base/Numeric                     `Numeric` (sizes) and `Coeff` (signed reindex coefficients)
  Base/St                          `St`, the INDEX prop: objects = axis lists; morphisms =
                                   `StMat`, integer-affine coordinate maps (coeffs + bias)
  Base/Br                          `Br`, the OPERATION prop: `BrMorph` = the free strict
                                   symmetric monoidal category on `BrBase` generators (raw
                                   `Hom` syntax quotiented by `Rel`). St lives INSIDE Br via
                                   each `BrBase`'s dependent `reindexings` field
  Seam/Adapter                     strictifies `ColoredPROP` onto Mathlib's
                                   `MonoidalCategory` / `SymmetricCategory`
  Core/Graded, Core/Weave          `DGradedColoredPROP` (the D-graded class: `sh`, `act`, δ, …)
                                   and Prop 8.2 weave-uniqueness
  Mixins/*, Grothendieck/Split,    the §7–§9 structure: temporal/route/symmetry mixins, the
  Algebra/*, Props/Generic         Grothendieck split, the target actegory + algebra, and the
                                   generic propositions
  Instances/StBr                   the flagship instance `D = St`, `C = Br` — every generic
                                   proposition specializes to `Br` through it

## Track 2 — the executable pipeline  (COMPUTABLE; runs real programs, fully `sorry`-free)

The tensor-logic DSL (§12) and everything that executes. Types here are first-order, `List`-based,
`SizeExpr`/`Int`-valued, and `deriving DecidableEq Repr ToExpr`, so the elaborator can build,
compare, and embed them at elaboration time.

  Exec/*                           UID minting, term traversal, the §7.4 `Context` coequalizer
  DSL/SizeExpr, DSL/Ast,           the front end: computable sizes, the typed AST, the surface
  DSL/Syntax, DSL/Elab             grammar, and value-returning elaborators (`tlprog!{…}`)
  DSL/Target                       the `*P` PRESENTATION types (`AxisP`/`StMatP`/`BrBaseP`/
                                   `ThreadedComposed`) — computable mirrors of the math tower
  DSL/Pipeline/*, DSL/Compile      `TLProgram.compile`: eight transformation phases (+ two
                                   validation passes, `checkReadRanks`/`checkDtypes`) threaded
                                   in `FreshM`, taking source → `ThreadedComposed`
  Eval/*                           a `Float` reference interpreter (`TLProgram.eval`)
  Acset/*                          the §8 CSV path: `SBrInstance` (a byte-for-byte mirror of
                                   Python `acset/instances.py`) + its codec

## The bridge — where the two tracks meet

  Bridge/Realize                   `realize : ThreadedComposed → BrMorph` (presentation →
                                   math tower) plus the sorry-free per-piece realizers
  Bridge/SBr, Bridge/Agreement     `realizeSBr`, `fromThreadedComposed`, and the §8 claim that
                                   the DSL route and the CSV route agree

## Three parallel type families — why they are NOT redundant

The same conceptual object has three representations, one per concern. They share only
`SizeExpr`/`Int` primitives and CANNOT be merged:

  concept           math tower (Base/)          presentation (DSL/Target)   acset (Acset/)
  ───────────────   ─────────────────────────   ─────────────────────────   ──────────────────
  axis              `Axis`   (Numeric size)      `AxisP`  (SizeExpr)         `AxisUID`+`axisSizes`
  reindex map       `StMat`  (Coeff, dependent)  `StMatP` (Int lists)        `SampleRow`s
  one Br morphism   `BrMorph` (Hom/Rel quotient) `ThreadedComposed` (DAG)    `SBrInstance` (tables)

  * Math-tower types are noncomputable (symbolic ring, function fields, dependent indices).
  * Presentation types drop the dependent typing for computability + `ToExpr` embedding.
  * Acset types follow Python's flat RELATIONAL schema (foreign-keyed rows), not the categorical
    shape, for byte-faithful interop.
  The conversions between them (`realize*`, `fromThreadedComposed`, the CSV codec) are genuine
  boundary crossings, not boilerplate to be deduplicated.

## End-to-end data flow

  tlprog!{ … }                              -- surface syntax (§12.1)
    ── parse (DSL/Elab) ──▶  TLProgram      -- typed AST
    ── compile (8 phases) ─▶  ThreadedComposed   -- the routed presentation DAG (the artifact)
         ├─ realize ─────────────▶ BrMorph        -- math tower; §8 agreement   [body deferred]
         ├─ fromThreadedComposed ─▶ SBrInstance ─▶ CSV   -- Python interop       [extraction deferred]
         └─ TLProgram.eval ──────▶ DenseTensor    -- Float reference
  NB: the evaluator consumes the PRE-route `ScheduledProgram` (`compileToScheduled`), which keeps
  full scan bodies; the routed `ThreadedComposed` collapses scans and is lossy to evaluate.

## What is proved vs. deferred

Track 2 (DSL / Pipeline / Eval / Acset) is fully `sorry`-free: you can compile and run real
programs and round-trip CSV against Python today. Track 1 carries DELIBERATE, staged `sorry`s —
the §2–§10 coherences, the `brCancelPoint` free-strict-SMC normal-form milestone, the
`Instances/StBr` §10.1 fields, and the `Bridge` realize/agreement bodies. The categorical meaning
of a runnable artifact is therefore the principal open seam. `SORRY_INVENTORY.md` is the
authoritative, milestone-by-milestone record of exactly what is proved.
-/
import LeanNCD.Base.Category
import LeanNCD.Base.Numeric
import LeanNCD.Base.ColoredPROP
import LeanNCD.Base.St
import LeanNCD.Base.Br
import LeanNCD.Seam.Adapter
import LeanNCD.Core.Graded
import LeanNCD.Core.Weave
import LeanNCD.Mixins.Temporal
import LeanNCD.Mixins.Stubs
import LeanNCD.Grothendieck.Split
import LeanNCD.Algebra.Target
import LeanNCD.Algebra.Algebra
import LeanNCD.Algebra.Construct
import LeanNCD.Props.Generic
import LeanNCD.Instances.StBr
import LeanNCD.Exec.Uid
import LeanNCD.Exec.Traversable
import LeanNCD.Exec.Context
import LeanNCD.DSL.SizeExpr
import LeanNCD.DSL.Ast
import LeanNCD.DSL.Syntax
import LeanNCD.DSL.Elab
import LeanNCD.DSL.Target
import LeanNCD.DSL.Traverse
import LeanNCD.DSL.Pipeline.Types
import LeanNCD.DSL.Pipeline.Structural
import LeanNCD.DSL.Pipeline.Lowering
import LeanNCD.DSL.Pipeline.RouteSpec
import LeanNCD.DSL.Compile
import LeanNCD.Bridge.Realize
import LeanNCD.Bridge.SBr
import LeanNCD.Bridge.Agreement
import LeanNCD.Acset.SBrInstance
import LeanNCD.Acset.Csv
import LeanNCD.Acset.Io
import LeanNCD.Eval.Tensor
import LeanNCD.Eval.Shape
import LeanNCD.Eval.Gather
import LeanNCD.Eval.Contract
import LeanNCD.Eval.Nonlin
import LeanNCD.Eval.Scatter
import LeanNCD.Eval.Scan
import LeanNCD.Eval.Eval
