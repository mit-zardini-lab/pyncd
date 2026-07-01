-- LeanNCD/Bridge/Realize.lean
import LeanNCD.Base.Br          -- Axis, StObj, StMat, ArrayType, DType, Br
import LeanNCD.Base.BrWiring    -- BrMorph.wiring (copy/permute/discard plumbing)
import LeanNCD.DSL.Target       -- AxisP, StMatP, StObjP
import LeanNCD.DSL.SizeExpr     -- SizeExpr.toNumeric

namespace LeanNCD

/-- Realize a presentation axis. Size via the §12.2 bridge `SizeExpr.toNumeric`. SORRY-FREE. -/
noncomputable def realizeAxis (a : AxisP) : Axis :=
  { name := a.name, size := a.size.toNumeric }

/-- SORRY-FREE. -/
noncomputable def realizeStObj (s : StObjP) : StObj := s.map realizeAxis

/-- Coerce an integer reindexing coefficient to `Coeff = MvPolynomial String ℤ`. SORRY-FREE:
    `Coeff` is signed (a `CommRing`), so negative look-back offsets realize faithfully via
    `MvPolynomial.C` (this is the resolution of the former negative-coefficient obstruction —
    `StMat` now carries `Coeff`, not the ℕ-semiring `Numeric`; see §2.2/§7.2). -/
noncomputable def intToCoeff (n : Int) : Coeff := MvPolynomial.C n

/-- A presentation stride map realized into `StMat dom cod`. The list dims must match
    `dom.length`/`cod.length`; out-of-range indices default to `0`, so the realized matrix
    is total and correct WHEN the dims match (no `sorry`). -/
noncomputable def realizeStMat (m : StMatP) (dom cod : StObj) : StMat dom cod :=
  { coeffs := fun (i : Fin cod.length) (j : Fin dom.length) =>
      intToCoeff (((m.coeffs.getD i.val []).getD j.val 0))
    bias   := fun (i : Fin cod.length) => intToCoeff (m.bias.getD i.val 0) }

/-- SORRY-FREE. -/
noncomputable def realizeWeaveSlot : WeaveSlotP → WeaveSlot
  | .fixed a => .fixed (realizeAxis a)
  | .tiled   => .tiled

/-- SORRY-FREE. -/
noncomputable def realizeWeaveShape (w : WeaveShapeP) : WeaveShape := w.map realizeWeaveSlot

/-- One ArrayType per weave. OBLIGATION (dtype): `BrBaseP`/`AxisP` carry no `DType`, so
    `dtype := .reals` (predicate/Bool outputs not distinguished — feedback: `BrBaseP` should
    carry an op-semiring/dtype tag). The SHAPE is real (the weave's target axes). -/
noncomputable def weaveToArrayType (w : WeaveShapeP) : ArrayType :=
  { dtype := .reals, shape := (realizeWeaveShape w).targetAxes }

/-- Realize a presentation base op to a `BrBase` over derived objects. The objects are
    computed from the weaves (not given). The `inputWeaves`/`outputWeaves` functions are
    realized from the lists by `Fin`-indexing (defaulting out-of-range to `[]`). The
    dependent `reindexings : ∀ i, StMat degree (inputWeaves i).targetAxes` is realized
    by indexing `b.reindexings` and passing the field's index `(inputWeaves i).targetAxes`
    as the `cod` argument directly, so the dependent typing holds definitionally. -/
noncomputable def realizeBrBaseP (b : BrBaseP) : Σ (dom cod : BrObj), BrBase dom cod :=
  let dom : BrObj := b.inputWeaves.map weaveToArrayType
  let cod : BrObj := b.outputWeaves.map weaveToArrayType
  ⟨dom, cod, {
    op := b.op.toString
    degree := realizeStObj b.degree
    inputWeaves  := fun i => realizeWeaveShape (b.inputWeaves.getD i.val [])
    outputWeaves := fun i => realizeWeaveShape (b.outputWeaves.getD i.val [])
    reindexings  := fun i =>
      realizeStMat (b.reindexings.getD i.val default) (realizeStObj b.degree)
        ((realizeWeaveShape (b.inputWeaves.getD i.val [])).targetAxes)
  }⟩

/-- The faithful external-input bundle (`realize.md` §6d): for each slot `0 ≤ k < nExternal`, the
    `ArrayType` of its first `Wire.external k` consuming port's input weave. This is the Lean image
    of the Python reference `ThreadedComposed.dom()` (walk `routing`, first occurrence per slot).
    Well-defined under `wellFormedDom`; unreferenced slots (impossible there) fall back to the
    empty-shape type. -/
noncomputable def realizeDom (tc : ThreadedComposed) : BrObj :=
  (List.range tc.nExternal).map fun k =>
    match tc.externalPort k with
    | some (i, j) => weaveToArrayType ((tc.steps.getD i default).inputWeaves.getD j [])
    | none        => weaveToArrayType []

/-! ### The routing plan (combinatorial layer, §12.4 fold)

Pure `Nat`/`Wire`/`List` bookkeeping over the §4 live pool — no morphisms, no dependent types, so it
is `#guard`-testable. `interp` (below) consumes it. -/

/-- Is `w` read by some step *after* `i`? (Then it must be carried past step `i`.) -/
def ThreadedComposed.readByLater (tc : ThreadedComposed) (i : Nat) (w : Wire) : Bool :=
  (List.range tc.steps.length).any fun j => decide (i < j) && (tc.routing.getD j []).contains w

/-- One step of the routing plan: the live `pool` before the step, the wiring `sel` (positions in
    `pool` of `reads ++ carried`), and `nReads = ` how many of `sel` are the step's read inputs. -/
structure WirePlanStep where
  pool   : List Wire
  sel    : List Nat
  nReads : Nat
  deriving Repr

/-- The §4 liveness fold: per-step plans, the final pool, and the final selection into `cod` order
    (the last step's single output). Each step gathers its reads to the front, carries the wires
    still read later, and produces `Wire.internal i 0`. -/
def ThreadedComposed.wirePlan (tc : ThreadedComposed) :
    List WirePlanStep × List Wire × List Nat :=
  let init : List Wire := (List.range tc.nExternal).map Wire.external
  let go : (List Wire × List WirePlanStep) → Nat → (List Wire × List WirePlanStep) :=
    fun (pool, acc) i =>
      let reads   := tc.routing.getD i []
      let carried := pool.filter (tc.readByLater i)
      let sel     := reads.map pool.idxOf ++ carried.map pool.idxOf
      (Wire.internal i 0 :: carried, acc ++ [⟨pool, sel, reads.length⟩])
  let (finalPool, steps) := (List.range tc.steps.length).foldl go (init, [])
  let codWires : List Wire :=
    match tc.steps.length with | 0 => [] | n+1 => [Wire.internal n 0]
  (steps, finalPool, codWires.map finalPool.idxOf)

/-- The `ArrayType` carried by a live wire: an external slot's type is read from its first consuming
    port (matching `realizeDom`); a producer wire `internal j s` is step `j`'s `s`-th output type. -/
noncomputable def ThreadedComposed.wireType (tc : ThreadedComposed) : Wire → ArrayType
  | .external k => match tc.externalPort k with
      | some (i, j) => weaveToArrayType ((tc.steps.getD i default).inputWeaves.getD j [])
      | none        => weaveToArrayType []
  | .internal j s => weaveToArrayType ((tc.steps.getD j default).outputWeaves.getD s [])

/-- `cod` (the realized codomain): the last step's output weaves (by last index; `[]` if no steps).
    Same as the Python oracle `content[-1].cod()`, stated by index to ease the final wiring. -/
noncomputable def ThreadedComposed.codObj (tc : ThreadedComposed) : BrObj :=
  match tc.steps.length with
  | 0     => []
  | m + 1 => (tc.steps.getD m default).outputWeaves.map weaveToArrayType

/-- The output wires of step `j`: `internal j 0, …, internal j (n_j-1)` where `n_j` is the number of
    output weaves (a coupled scan has `n_j > 1`). -/
def ThreadedComposed.outputSlots (tc : ThreadedComposed) (j : Nat) : List Wire :=
  (List.range (tc.steps.getD j default).outputWeaves.length).map (Wire.internal j ·)

/-- The live pool before processing step `i` (monotonic): every earlier step's output slots
    (`outputSlots (i-1) ++ … ++ outputSlots 0`) followed by the externals. A produced wire is never
    dropped, so `reads ⊆ pool` stays easy. Multi-output steps prepend ALL their slots. -/
def ThreadedComposed.poolAt (tc : ThreadedComposed) (i : Nat) : List Wire :=
  (List.range i).foldl (fun p j => tc.outputSlots j ++ p)
    ((List.range tc.nExternal).map Wire.external)

/-- Type-consistency of the routing — the hypothesis `realize` needs (extends `wellFormedDom`):
    every step's read wires carry exactly the types that step consumes (producer ⊳ consumer match),
    and the last step has a single output (so `codObj` is the final pool wire's type). Holds for any
    compiled `tc`; supplied by the agreement layer for the bridge. -/
def ThreadedComposed.WellFormed (tc : ThreadedComposed) : Prop :=
  tc.wellFormedDom = true ∧
  (∀ i, i < tc.steps.length →
    (tc.routing.getD i []).map tc.wireType
      = (tc.steps.getD i default).inputWeaves.map weaveToArrayType) ∧
  -- NOTE(phaseB): weakened from `= 1` to `≥ 1` for multi-output scans (the new compiler shape).
  (∀ i, i < tc.steps.length → (tc.steps.getD i default).outputWeaves.length ≥ 1) ∧
  (∀ i, i < tc.steps.length → ∀ w ∈ tc.routing.getD i [], w ∈ tc.poolAt i)

/-! ### The assembly fold (monotonic pool)

The pool grows monotonically — externals, then every step's output prepended — so a produced wire is
never lost and `reads ⊆ pool` is easy. The per-step type matches come from `WellFormed`. -/

/-- One step as a `BrMorph` (its generator, lifted into the quotient). -/
noncomputable def stepMorph (b : BrBaseP) :
    BrMorph (realizeBrBaseP b).1 (realizeBrBaseP b).2.1 :=
  BrMorph.mk (.gen (realizeBrBaseP b).2.2)

/-- The initial pool's types are exactly `realizeDom`. -/
theorem realizeDom_eq_poolAt_zero (tc : ThreadedComposed) :
    realizeDom tc = (tc.poolAt 0).map tc.wireType := by
  simp only [ThreadedComposed.poolAt, List.range_zero, List.foldl_nil, List.map_map]
  rfl

/-- The pool grows by prepending the new step's output slots. -/
theorem ThreadedComposed.poolAt_succ (tc : ThreadedComposed) (i : Nat) :
    tc.poolAt (i + 1) = tc.outputSlots i ++ tc.poolAt i := by
  simp only [ThreadedComposed.poolAt, List.range_succ, List.foldl_append, List.foldl_cons,
    List.foldl_nil]

/-- A list-of-weaves mapped is the per-index `getD` map over `range length` — lets us equate
    `outputWeaves.map weaveToArrayType` with `(outputSlots j).map wireType` (which maps over the
    index range). -/
private theorem map_eq_range_map_getD {α β : Type*} (l : List α) (f : α → β) (d : α) :
    l.map f = (List.range l.length).map (fun s => f (l.getD s d)) := by
  apply List.ext_getElem
  · simp
  · intro s h1 h2
    simp only [List.getElem_map, List.getElem_range, List.getD_eq_getElem _ _ (by simpa using h1)]

/-- The last-step-output types of `outputSlots j` are exactly `outputWeaves.map weaveToArrayType`. -/
theorem outputSlots_map_wireType (tc : ThreadedComposed) (j : Nat) :
    (tc.outputSlots j).map tc.wireType
      = (tc.steps.getD j default).outputWeaves.map weaveToArrayType := by
  unfold ThreadedComposed.outputSlots
  rw [List.map_map, map_eq_range_map_getD (tc.steps.getD j default).outputWeaves weaveToArrayType []]
  rfl

/-- One step of the fold: from the current pool, gather the step's reads to the front (keeping the
    whole pool live), apply the step beside the carried wires, landing on the next pool's types.
    The two `WellFormed` casts: reads' types = the step's domain, and the step's outputs prepend. -/
noncomputable def stepPiece (tc : ThreadedComposed) (h : tc.WellFormed) (i : Nat)
    (hi : i < tc.steps.length) :
    BrMorph ((tc.poolAt i).map tc.wireType) ((tc.poolAt (i + 1)).map tc.wireType) := by
  -- target = reads ++ pool ; every target wire is in the pool
  have hsub : ∀ w ∈ tc.routing.getD i [] ++ tc.poolAt i, w ∈ tc.poolAt i := fun w hw =>
    (List.mem_append.1 hw).elim (h.2.2.2 i hi w) id
  -- reads' types are exactly the step's domain
  have hdom : (tc.routing.getD i [] ++ tc.poolAt i).map tc.wireType
            = (tc.steps.getD i default).inputWeaves.map weaveToArrayType
              ++ (tc.poolAt i).map tc.wireType := by
    rw [List.map_append, h.2.1 i hi]
  -- the step's outputs prepend to the pool: `poolAt (i+1) = outputSlots i ++ poolAt i`
  have hcod : (tc.steps.getD i default).outputWeaves.map weaveToArrayType ++ (tc.poolAt i).map tc.wireType
            = (tc.poolAt (i + 1)).map tc.wireType := by
    rw [tc.poolAt_succ, List.map_append, outputSlots_map_wireType]
  exact cast (congrArg (BrMorph ((tc.poolAt i).map tc.wireType)) hcod)
    (BrMorph.comp
      (cast (congrArg (BrMorph ((tc.poolAt i).map tc.wireType)) hdom)
        (BrMorph.wiringBy tc.wireType (tc.poolAt i) (tc.routing.getD i [] ++ tc.poolAt i) hsub))
      (BrMorph.tensor (stepMorph (tc.steps.getD i default))
        (BrMorph.idm ((tc.poolAt i).map tc.wireType))))

/-- The fold up to step `i`: a morphism `realizeDom → (poolAt i)`-types, composing the per-step pieces. -/
noncomputable def interpUpto (tc : ThreadedComposed) (h : tc.WellFormed) :
    (i : Nat) → i ≤ tc.steps.length → BrMorph (realizeDom tc) ((tc.poolAt i).map tc.wireType)
  | 0,     _  => cast (congrArg (BrMorph (realizeDom tc)) (realizeDom_eq_poolAt_zero tc))
                   (BrMorph.idm (realizeDom tc))
  | i + 1, hi => BrMorph.comp (interpUpto tc h i (Nat.le_of_succ_le hi))
                              (stepPiece tc h i (Nat.lt_of_succ_le hi))

/-- The final selection: from the full pool, keep the last step's single output (= `cod`). -/
noncomputable def finalPiece (tc : ThreadedComposed) (_h : tc.WellFormed) :
    BrMorph ((tc.poolAt tc.steps.length).map tc.wireType) tc.codObj := by
  by_cases hz : tc.steps.length = 0
  · -- no steps: cod = [], discard the whole (external) pool
    rw [hz]
    have hcod : tc.codObj = [] := by simp [ThreadedComposed.codObj, hz]
    rw [hcod]; exact BrMorph.delAll _
  · -- the last step's output slots are the pool prefix; select them all (= cod)
    set m := tc.steps.length - 1 with hmeq
    have hm : tc.steps.length = m + 1 := by omega
    rw [hm]
    have hmem : ∀ w ∈ tc.outputSlots m, w ∈ tc.poolAt (m + 1) := by
      intro w hw; rw [tc.poolAt_succ]; exact List.mem_append_left _ hw
    have hcod : tc.codObj = (tc.outputSlots m).map tc.wireType := by
      rw [outputSlots_map_wireType]; simp only [ThreadedComposed.codObj, hm]
    exact cast (congrArg (BrMorph ((tc.poolAt (m + 1)).map tc.wireType)) hcod.symm)
      (BrMorph.wiringBy tc.wireType (tc.poolAt (m + 1)) (tc.outputSlots m) hmem)

/-- Realize a routed-DAG presentation (`ThreadedComposed`, §12.4) into ONE `Br` morphism
    `BrMorph dom cod`.

    * `cod = tc.codObj`: the last step's output weaves (the Python `ThreadedComposed.cod()` oracle).
    * `dom = realizeDom tc`: the per-external-slot bundle (the Python `ThreadedComposed.dom()` oracle).

    The body (SORRY-FREE under `WellFormed`): the monotonic-pool fold `interpUpto` threads the per-step
    `realizeBrBaseP` morphisms along `routing` — gathering reads with `BrMorph.wiringBy` (copy/discard
    via the `Δ`/`ε` comonoid) and sequencing with `comp`/`tensor` — and `finalPiece` selects the
    output. `WellFormed tc` supplies the per-step type matches; it holds for any compiled `tc`. -/
noncomputable def realize (tc : ThreadedComposed) (h : tc.WellFormed) :
    Σ (dom cod : BrObj), BrMorph dom cod :=
  ⟨realizeDom tc, tc.codObj,
    BrMorph.comp (interpUpto tc h tc.steps.length (Nat.le_refl _)) (finalPiece tc h)⟩

end LeanNCD
