import LeanNCD.Eval.Contract
import LeanNCD.Eval.Nonlin
import LeanNCD.DSL.Pipeline.Types
namespace LeanNCD.Eval
open Std

/-- The iteration-axis UID of a base/recur stmt's slots (the `.iterAt`/`.iterNext` slot),
    together with its position among the slots. `none` if no iteration slot is present. -/
def iterSlotPos (slots : List LHSSlot) : Option (UID × Nat) :=
  let rec go : Nat → List LHSSlot → Option (UID × Nat)
    | _, []            => none
    | i, sl :: rest    => match sl with
        | .iterAt a _ => some (a.uid, i)
        | .iterNext a => some (a.uid, i)
        | _           => go (i + 1) rest
  go 0 slots

/-- Evaluate ONE stmt at a FIXED iteration value (`iterUID ↦ l`), over the non-iter free axes,
    returning `(name, slice)` where `slice` has the non-iter free-axis shape. Reads gather from
    `env`, which holds the partial state at ALL iterations, so a read `G[…,l]` works. The iteration
    axis is pinned via `evalAssignSeeded`. Applies the RHS nonlin to the produced slice.

    The slice's axes are the NON-iteration free slots in slot order (see `evalAssignSeeded`), so the
    softmax/normalize reduction axis is the position of the output slot marked `m.` (the norm flag
    lives on the output slot — see `normAxisUidOf`) within that slice-axis list. This holds uniformly
    whether or not the stmt is itself a scan-state (has an iteration slot); pinned by the `· != iterUID`
    filter, which drops the iteration axis exactly as `evalAssignSeeded` does. -/
def evalStmtSlice (env : HashMap String DenseTensor) (sizes : HashMap UID Nat)
    (iterUID : UID) (l : Nat) (s : Stmt) : Except EvalError (String × DenseTensor) := do
  match s with
  | .assign nm slots rhs =>
      let seed : HashMap UID Int := ({} : HashMap UID Int).insert iterUID (Int.ofNat l)
      let (_, slice) ← evalAssignSeeded env sizes seed nm slots rhs
      let sliceUids := (slots.filterMap lhsAxisUID?).filter (· != iterUID)
      let pos ← match rhs.nonlin with
        | .identity | .relu => pure 0     -- pointwise: reduction axis irrelevant
        | .softmax _ | .normalize _ => match normAxisUidOf slots with
            | some nu => match sliceUids.findIdx? (· == nu) with
                | some p => pure p
                | none   => throw s!"evalStmtSlice: marked norm axis of {nm} is not among its slice axes"
            | none    => throw s!"evalStmtSlice: {nm} applies softmax/normalize but no output axis is marked (·)"
      return (nm, applyNonlin rhs.nonlin pos sliceUids slice)
  | _ => throw "evalStmtSlice: only assign stmts are supported in scans"

/-- Write a non-iter `slice` into the full state tensor `out` at iteration index `iterIdx`
    (the iteration axis sits at position `iterPos` in `out.shape`). The slice's coords are the
    out-coords with position `iterPos` removed; insert `iterIdx` there to address `out`. -/
def writeSliceAt (out : DenseTensor) (iterPos iterIdx : Nat) (slice : DenseTensor) : DenseTensor :=
  (DenseTensor.allCoords slice.shape).foldl (fun cur scoord =>
    let ocoord := (scoord.take iterPos) ++ [iterIdx] ++ (scoord.drop iterPos)
    cur.set! ocoord (slice.get! scoord)) out

/-- The LHS tensor name of a (scan-eligible) assign stmt. -/
def stmtName : Stmt → String
  | .assign nm _ _    => nm
  | .scatter nm _ _ _ => nm
  | .recurMorphism nm _ _ => nm

/-- The full state-tensor shape for a name produced by base/recur slots: the iteration slot's
    position holds `L`, the other slot positions hold their free-axis sizes (slot order). -/
def stateShape (sizes : HashMap UID Nat) (slots : List LHSSlot) (L : Nat) : List Nat :=
  slots.map (fun sl => match sl with
    | .iterAt _ _ | .iterNext _ => L
    | _ => match lhsAxisUID? sl with
        | some u => (sizes[u]?).getD 0
        | none   => 0)

/-- Evaluate a ScanStmt → the scanned state tensors. -/
def evalScan (env : HashMap String DenseTensor) (sizes : HashMap UID Nat) :
    ScanStmt → Except EvalError (List (String × DenseTensor))
  | .plain _      => .error "evalScan: plain handled by evalScheduled, not here"
  | .scanPre nm _ _ => .error s!"evalScan: scanPre (recurMorphism escape hatch) evaluation unsupported ({nm})"
  | .scan _ ax base recur _ => do
      let L := (sizes[ax.uid]?).getD 0
      -- per state name, find the (iterPos) and full state shape from its base slots
      let stateNames := (base.map stmtName).eraseDups
      -- 1. allocate each state tensor (zeros at full shape) into the working env
      let mut work := env
      let mut iterPosOf : HashMap String Nat := {}
      for s in base do
        match s with
        | .assign nm slots _ =>
            match iterSlotPos slots with
            | some (_, iterPos) =>
                work := work.insert nm (DenseTensor.zeros (stateShape sizes slots L))
                iterPosOf := iterPosOf.insert nm iterPos
            | none => throw s!"evalScan: base stmt for {nm} has no iteration slot"
        | _ => throw "evalScan: base stmts must be assigns"
      -- 2. fill l=0 from base
      for s in base do
        let (nm, slice) ← evalStmtSlice work sizes ax.uid 0 s
        let iterPos := (iterPosOf[nm]?).getD 0
        work := work.insert nm (writeSliceAt ((work[nm]?).getD (DenseTensor.zeros [])) iterPos 0 slice)
      -- 3. for l = 0 … L-2: run the recur list at fixed l; intermediates into the step env;
      --    write final state slices at iterIdx (l+1); update the working env.
      for l in List.range (L - 1) do
        let mut stepEnv := work
        for s in recur do
          let (nm, slice) ← evalStmtSlice stepEnv sizes ax.uid l s
          match iterPosOf[nm]? with
          | some iterPos =>
              -- a state slice: write into the full state tensor at iteration l+1
              let updated := writeSliceAt ((work[nm]?).getD (DenseTensor.zeros [])) iterPos (l + 1) slice
              work := work.insert nm updated
              stepEnv := stepEnv.insert nm updated
          | none =>
              -- an intermediate: keep the raw slice in the step env only
              stepEnv := stepEnv.insert nm slice
      -- 4. return the scanned state tensors
      return stateNames.filterMap (fun nm => (work[nm]?).map (fun t => (nm, t)))

end LeanNCD.Eval
