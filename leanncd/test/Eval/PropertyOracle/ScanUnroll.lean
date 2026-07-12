import Eval.PropertyOracle.Compare
import Eval.PropertyOracle.ScanGen
import LeanNCD.Eval.Scan

/-!
# Scan-unrolling: slice extraction (E6 scan-unrolling oracle, Task 3)

`sliceTensorAtMulti` is the inverse of `LeanNCD.Eval.writeSliceAtMulti`: given a scan's full
state tensor, extract the non-iteration-axis slice at a fixed set of `(position, index)`
iteration coordinates. New code with no existing counterpart in the codebase, so it is verified
directly against its own inverse below before later tasks trust it inside the oracle comparison.
-/
namespace LeanNCD.PropertyOracle
open LeanNCD LeanNCD.Eval

/-- The inverse of `LeanNCD.Eval.writeSliceAtMulti`: extract the non-iteration-axis slice of
    `full` at a fixed set of `(position, index)` iteration coordinates (same coordinate
    bookkeeping as `writeSliceAtMulti`, in reverse). -/
def sliceTensorAtMulti (iters : List (Nat × Nat)) (full : DenseTensor) : DenseTensor :=
  let positions := iters.map Prod.fst
  let sliceShape := full.shape.zipIdx.filterMap (fun (d, p) => if positions.contains p then none else some d)
  let sorted := iters.mergeSort (fun a b => a.1 ≤ b.1)
  DenseTensor.ofFn sliceShape (fun scoord =>
    let ocoord := sorted.foldl (fun acc (pos, idx) => acc.insertIdx pos idx) scoord
    full.get! ocoord)

-- TEST-THE-TESTER: round-trip against `writeSliceAtMulti` (new code, no existing counterpart to
-- lean on — must be verified against its own inverse before the oracle trusts it).
private def rtSlice : DenseTensor := ⟨[3], #[9.0, 8.0, 7.0]⟩
private def rtFull : DenseTensor := writeSliceAtMulti (DenseTensor.zeros [2, 3]) [(0, 1)] rtSlice
#guard denseEq (sliceTensorAtMulti [(0, 1)] rtFull) rtSlice
-- a DIFFERENT position round-trips too (position 0 is not hardcoded correctly by accident):
private def rtSlice2 : DenseTensor := ⟨[2], #[5.0, 6.0]⟩
private def rtFull2 : DenseTensor := writeSliceAtMulti (DenseTensor.zeros [2, 2]) [(1, 1)] rtSlice2
#guard denseEq (sliceTensorAtMulti [(1, 1)] rtFull2) rtSlice2
-- two positions at once (the 2-D grid case, a later task):
private def rtScalar : DenseTensor := ⟨[], #[4.0]⟩
private def rtFull3 : DenseTensor := writeSliceAtMulti (DenseTensor.zeros [2, 2]) [(0, 1), (1, 0)] rtScalar
#guard denseEq (sliceTensorAtMulti [(0, 1), (1, 0)] rtFull3) rtScalar

/-! ## `unrollScan1D` (Task 4): mechanical substitution unroll for single-axis scans -/

/-- Rewrite one read-factor's scan-axis position(s): a read of a name in `stateNames` (a
    self- or cross-state recurrence read) is redirected to that state's PREVIOUS-step leaf
    tensor (dropping the scan-axis position, since the leaf has no scan axis); a read of any
    OTHER (external) tensor at the scan axis is redirected to a literal `.const` at this step. -/
private def substAxisFactor (scanAxis : AxisSpec) (stateNames : List String)
    (prevLeafOf : String → String) (stepIdx : Nat) (f : Factor) : Factor :=
  match f with
  | .read nm idxs =>
      if stateNames.contains nm then
        .read (prevLeafOf nm) (idxs.filter (fun e => match e with
          | .axis a => a.uid != scanAxis.uid
          | _       => true))
      else
        .read nm (idxs.map (fun e => match e with
          | .axis a => if a.uid == scanAxis.uid then .const (Int.ofNat stepIdx) else e
          | other   => other))
  | .unaryFn op nm idxs =>
      .unaryFn op nm (idxs.map (fun e => match e with
        | .axis a => if a.uid == scanAxis.uid then .const (Int.ofNat stepIdx) else e
        | other   => other))
  | .iverson _ => f

/-- Drop the scan-axis LHS slot (the leaf has no scan axis; per the generator's slot-order
    invariant, it is always the last position). -/
private def leafSlots (scanAxis : AxisSpec) (slots : List LHSSlot) : List LHSSlot :=
  slots.filter (fun sl => match sl with
    | .iterAt a _ | .iterNext a => a.uid != scanAxis.uid
    | _ => true)

/-- Unroll a single-axis scan case into a scan-free companion `TLProgram`: one leaf statement
    `Su_<state>_<k>` per (state, step `k`), `k` from `0` to `L-1`, via mechanical substitution —
    NOT by re-running `evalScan`'s own logic, so this stays an independently-derived check. -/
def unrollScan1D (c : ScanCase) : TLProgram :=
  let scanAxis := c.axes.head!
  let L        := c.Ls.head!
  let stateNames := (c.base.filterMap (fun s => match s with | .assign nm _ _ => some nm | _ => none)).eraseDups
  let leafName (nm : String) (k : Nat) : String := s!"Su_{nm}_{k}"
  let step0 : List Stmt := c.base.filterMap (fun s => match s with
    | .assign nm slots rhs => some (.assign (leafName nm 0) (leafSlots scanAxis slots) rhs)
    | _ => none)
  let steps : List Stmt := (List.range (L - 1)).flatMap (fun stepIdx =>
    c.recur.filterMap (fun s => match s with
      | .assign nm slots rhs =>
          let rhs' := { rhs with body := ⟨rhs.body.terms.map (fun t =>
            ⟨t.factors.map (substAxisFactor scanAxis stateNames (fun n => leafName n stepIdx) stepIdx)⟩)⟩ }
          some (.assign (leafName nm (stepIdx + 1)) (leafSlots scanAxis slots) rhs')
      | _ => none))
  { decls := c.prog.decls, stmts := step0 ++ steps }

-- TEST-THE-TESTER: template 1 (linear self-scan), hand-verified against ScanTest.lean's own
-- linear-scan point example (X=[1,2], A=[2,3], L=3 ⇒ S[:,0]=[1,2], S[:,1]=[2,6], S[:,2]=[4,18]).
private def t1case : ScanCase := template1 3 false
private def t1unrolled := unrollScan1D t1case
#guard (TLProgram.eval t1unrolled t1case.inputs).toOption.isSome
run_cmd do
  match TLProgram.eval t1unrolled t1case.inputs with
  | .error e => throwError e
  | .ok env =>
      match env["Su_S_0"]?, env["Su_S_1"]?, env["Su_S_2"]? with
      | some s0, some s1, some s2 =>
          unless denseEq s0 ⟨[2], #[1.0, 2.0]⟩ do throwError s!"Su_S_0 wrong: {repr s0.data}"
          unless denseEq s1 ⟨[2], #[2.0, 6.0]⟩ do throwError s!"Su_S_1 wrong: {repr s1.data}"
          unless denseEq s2 ⟨[2], #[4.0, 18.0]⟩ do throwError s!"Su_S_2 wrong: {repr s2.data}"
      | _, _, _ => throwError "missing Su_S_k"

/-! ## `unrollScan2D` (Task 5): the one 2-D template in scope -/

/-- Unroll the one 2-D template in scope: base `G[r,0] := …` (varies freely over `r`, pins `c`
    at 0) and recur `G[r+1,c+1] := G[r,c] + …[r,c]` (advances both `r` and `c` together). NOT a
    general N-axis walker — specialized to this exact shape, matching the design's explicit
    scope restriction to one curated 2-D template.

    Emits: ONE vector-shaped statement for the `c = 0` column (matching the base's own free-axis
    shape, rather than `Lr` separate scalar leaves), plus one scalar leaf per OTHER grid cell —
    a fully-advanced cell `(ri,ci)` with `ri ≥ 1` reads the diagonal predecessor `(ri-1,ci-1)`
    (from the column vector if `ci-1 = 0`, else from a prior leaf); an unreached boundary cell
    `(0,ci)` with `ci ≥ 1` is the aggregator's zero identity, matching `evalScan`'s zero-default. -/
def unrollScan2D (c : ScanCase) : TLProgram :=
  match c.axes, c.Ls, c.base, c.recur with
  | [r, cc], [Lr, Lc], [.assign stateName _ baseRhs], [.assign _ _ recurRhs] =>
      let col0Name := s!"Su_{stateName}_col0"
      let leaf (ri ci : Nat) : String := s!"Su_{stateName}_{ri}_{ci}"
      let col0 : Stmt := .assign col0Name [.free r] baseRhs
      let substCell (pri pci : Nat) (f : Factor) : Factor :=
        match f with
        | .read nm idxs =>
            if nm == stateName then
              if pci == 0 then .read col0Name [.const (Int.ofNat pri)]
              else .read (leaf pri pci) []
            else
              .read nm (idxs.map (fun e => match e with
                | .axis a =>
                    if a.uid == r.uid then .const (Int.ofNat pri)
                    else if a.uid == cc.uid then .const (Int.ofNat pci)
                    else e
                | other => other))
        | other => other
      let cells : List Stmt := (List.range Lr).flatMap (fun ri =>
        ((List.range (Lc - 1)).map (· + 1)).map (fun ci =>
          if ri == 0 then
            .assign (leaf 0 ci) [] { body := { terms := [] }, nonlin := .identity }
          else
            let rhs' := { recurRhs with body := ⟨recurRhs.body.terms.map (fun t =>
              ⟨t.factors.map (substCell (ri - 1) (ci - 1))⟩)⟩ }
            .assign (leaf ri ci) [] rhs'))
      { decls := c.prog.decls, stmts := [col0] ++ cells }
  | _, _, _, _ =>
      -- unreachable for any case the generator actually produces (template 6 is the only
      -- 2-axis case, and always has exactly this shape); a total (non-panicking) fallback
      -- keeps this function total rather than partial.
      { decls := c.prog.decls, stmts := [] }

-- TEST-THE-TESTER: template 6 (2-D grid-DP), hand-verified against RC6
-- (RecurrenceTest.lean) — G[1,1] = G[0,0]+A[0,0] = 1; all other non-column-0 cells stay 0.
private def t6case : ScanCase := template6
private def t6unrolled := unrollScan2D t6case
#guard (TLProgram.eval t6unrolled t6case.inputs).toOption.isSome
run_cmd do
  match TLProgram.eval t6unrolled t6case.inputs with
  | .error e => throwError e
  | .ok env =>
      match env["Su_G_col0"]?, env["Su_G_1_1"]?, env["Su_G_0_1"]? with
      | some col0, some g11, some g01 =>
          unless denseEq col0 ⟨[2], #[0.0, 0.0]⟩ do throwError s!"Su_G_col0 wrong: {repr col0.data}"
          unless denseEq g11 ⟨[], #[1.0]⟩ do throwError s!"Su_G_1_1 wrong: {repr g11.data}"
          unless denseEq g01 ⟨[], #[0.0]⟩ do throwError s!"Su_G_0_1 wrong: {repr g01.data}"
      | _, _, _ => throwError "missing Su_G_* leaves"

end LeanNCD.PropertyOracle
