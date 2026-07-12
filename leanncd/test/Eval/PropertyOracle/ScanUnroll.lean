import Eval.PropertyOracle.Compare
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

end LeanNCD.PropertyOracle
