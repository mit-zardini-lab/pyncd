import LeanNCD.Eval.Contract
import LeanNCD.Eval.Nonlin
import LeanNCD.DSL.Pipeline.Types
namespace LeanNCD.Eval
open Std

/-- All iteration slots `(uid, position)` of a base/recur stmt's slot list, in slot order
    (ascending position). One entry per `.iterAt`/`.iterNext` slot — multi-axis scans yield
    several. -/
def iterSlotPositions (slots : List LHSSlot) : List (UID × Nat) :=
  slots.zipIdx.filterMap (fun (sl, i) => match sl with
    | .iterAt a _ => some (a.uid, i)
    | .iterNext a => some (a.uid, i)
    | _           => none)

/-- Cartesian product of a list of index ranges → list of tuples (each tuple a `List Nat`), in
    reverse-lexicographic order (last axis slowest). This order is a linear extension of the
    componentwise ≤ order, so a cell writing at `+1` on every advancing axis (reading only cells at
    offset ≤ 0 — the causality guarantee) always sees its dependencies already computed. -/
def cartesianList : List (List Nat) → List (List Nat)
  | []      => [[]]
  | r :: rs => (cartesianList rs).flatMap (fun tail => r.map (fun x => x :: tail))

/-- Evaluate ONE stmt with a SET of iteration axes pinned (`seed : uid ↦ value`), over the
    remaining (non-seeded) free axes, returning `(name, slice)` where `slice` has the non-seeded
    free-axis shape. Reads gather from `env`, which holds the partial state at ALL iterations, so a
    read `G[…,l]` works. The seeded axes are pinned via `evalAssignSeeded`. Applies the RHS nonlin.

    The slice's axes are the NON-seeded free slots in slot order (see `evalAssignSeeded`), so the
    softmax/normalize reduction axis is the position of the output slot marked `m.` (the norm flag
    lives on the output slot — see `normAxisUidOf`) within that slice-axis list. This holds uniformly
    whether or not the stmt is itself a scan-state; pinned by the `!seed.contains ·` filter, which
    drops every seeded axis exactly as `evalAssignSeeded` does. -/
def evalStmtSliceSeeded (env : HashMap String DenseTensor) (sizes : HashMap UID Nat)
    (seed : HashMap UID Int) (s : Stmt) : Except EvalError (String × DenseTensor) := do
  match s with
  | .assign nm slots rhs =>
      -- honor the contraction aggregator (KG-scanagg): `maxreduce`/`minreduce` ⇒ tropical max/min, else ℝ sum.
      let c : Combine := match rhs.agg with
        | .max => Combine.max
        | .min => Combine.min
        | .sum => Combine.real
      let (_, slice) ← evalAssignSeeded c.mul c.combine c.unit0 env sizes seed nm slots rhs
      let sliceUids := (slots.filterMap lhsAxisUID?).filter (fun u => ! seed.contains u)
      let pos ← match rhs.nonlin with
        | .identity | .relu => pure 0     -- pointwise: reduction axis irrelevant
        | .softmax _ | .normalize _ => match normAxisUidOf slots with
            | some nu => match sliceUids.findIdx? (· == nu) with
                | some p => pure p
                | none   => throw s!"evalStmtSliceSeeded: marked norm axis of {nm} is not among its slice axes"
            | none    => throw s!"evalStmtSliceSeeded: {nm} applies softmax/normalize but no output axis is marked (·)"
      return (nm, applyNonlin rhs.nonlin pos sliceUids slice)
  | _ => throw "evalStmtSliceSeeded: only assign stmts are supported in scans"

/-- Write a non-iter `slice` into the full state tensor `out`, given the iteration `(position, index)`
    pairs (one per advancing axis of the stmt). The slice's coords are the out-coords with all
    iteration positions removed; rebuild the full coord by inserting each iteration index at its
    position in ASCENDING position order (so earlier insertions don't shift later ones). -/
def writeSliceAtMulti (out : DenseTensor) (iters : List (Nat × Nat)) (slice : DenseTensor) : DenseTensor :=
  let sorted := iters.mergeSort (fun a b => a.1 ≤ b.1)   -- ascending by position
  (DenseTensor.allCoords slice.shape).foldl (fun cur scoord =>
    let ocoord := sorted.foldl (fun acc (pos, idx) => acc.insertIdx pos idx) scoord
    cur.set! ocoord (slice.get! scoord)) out

/-- The LHS tensor name of a (scan-eligible) assign stmt. -/
def stmtName : Stmt → String
  | .assign nm _ _    => nm
  | .scatter nm _ _ _ => nm
  | .recurMorphism nm _ _ => nm

/-- The LHS slots of a (scan-eligible) stmt (empty for `recurMorphism`). -/
def stmtSlots : Stmt → List LHSSlot
  | .assign _ ls _    => ls
  | .scatter _ ls _ _ => ls
  | .recurMorphism _ _ _ => []

/-- The full state-tensor shape for a name produced by base/recur slots: each slot position holds
    the size of its axis (slot order). For an iteration slot this size IS the axis's length `L`
    (`sizes[uid]` equals the declared axis size); for a free slot it is that free axis's size. -/
def stateShape (sizes : HashMap UID Nat) (slots : List LHSSlot) : List Nat :=
  slots.map (fun sl => match lhsAxisUID? sl with
    | some u => (sizes[u]?).getD 0
    | none   => 0)

/-- Evaluate a ScanStmt → the scanned state tensors. Multi-axis (n-D) scans iterate the cartesian
    product of `[0 … L_a − 2]` over every advancing axis. Boundary semantics (zero-default): the
    step writes only fully-advanced cells (every advancing index `+1 ≥ 1`); boundary cells (any
    advancing index `= 0`) keep the zero-allocated state, except where an explicit base stmt pins
    a slice at index 0. -/
def evalScan (env : HashMap String DenseTensor) (sizes : HashMap UID Nat) :
    ScanStmt → Except EvalError (List (String × DenseTensor))
  | .plain _      => .error "evalScan: plain handled by evalScheduled, not here"
  | .scanPre nm _ _ => .error s!"evalScan: scanPre (recurMorphism escape hatch) evaluation unsupported ({nm})"
  | .scan _ axes base recur _ => do
      if axes.isEmpty then .error "evalScan: scan node has no iteration axis" else
      let axUids := axes.map (·.uid)
      let Ls     := axUids.map (fun u => (sizes[u]?).getD 0)   -- per-axis length, in `axes` order
      let stateNames := (base.map stmtName).eraseDups
      -- 1. allocate each state tensor (zeros at full shape) from its base slots.
      let mut work := env
      for s in base do
        match s with
        | .assign nm slots _ => work := work.insert nm (DenseTensor.zeros (stateShape sizes slots))
        | _ => throw "evalScan: base stmts must be assigns"
      -- 2. fill boundaries from base stmts: each base pins a subset of axes to their literal index
      --    and fills that slice over its free axes (e.g. `G[r,0]` fills the c=0 column for all r).
      for s in base do
        let seed : HashMap UID Int := ((stmtSlots s).filterMap (fun
            | .iterAt a n => some (a.uid, n) | _ => none)).foldl (fun m (u, n) => m.insert u n) {}
        let (nm, slice) ← evalStmtSliceSeeded work sizes seed s
        let iters := (iterSlotPositions (stmtSlots s)).map (fun (u, p) => (p, ((seed[u]?).getD 0).toNat))
        work := work.insert nm (writeSliceAtMulti ((work[nm]?).getD (DenseTensor.zeros [])) iters slice)
      -- 3. nested loop over ∏ [0 … L_a − 2]: run the recur list at each tuple; intermediates into
      --    the step env; write final state slices at (position, index+1) per advancing axis.
      let ranges := Ls.map (fun L => List.range (L - 1))
      for tup in cartesianList ranges do          -- tup : one index per axis, in `axes` order
        let seed : HashMap UID Int := (axUids.zip tup).foldl (fun m (u, v) => m.insert u (Int.ofNat v)) {}
        let mut stepEnv := work
        for s in recur do
          let (nm, slice) ← evalStmtSliceSeeded stepEnv sizes seed s
          -- classify by NAME: only the allocated scan states (`stateNames`) are written into `work`.
          -- A per-step intermediate may itself carry an iteration slot (e.g. a `splitNonlins`-lifted
          -- `%nl…` derived from the recurrence), but it is NOT an allocated state — keep it as a raw
          -- slice in the step env only. (This is the crucial distinction from a slot-based test.)
          if stateNames.contains nm then
            -- a state slice: write at (position, currentIndex+1) for each of THIS stmt's advancing axes
            let iters := (iterSlotPositions (stmtSlots s)).map (fun (u, p) => (p, ((seed[u]?).getD 0).toNat + 1))
            let updated := writeSliceAtMulti ((work[nm]?).getD (DenseTensor.zeros [])) iters slice
            work := work.insert nm updated
            stepEnv := stepEnv.insert nm updated
          else
            stepEnv := stepEnv.insert nm slice
      -- 4. return the scanned state tensors
      return stateNames.filterMap (fun nm => (work[nm]?).map (fun t => (nm, t)))

end LeanNCD.Eval
