import LeanNCD.Eval.Nonlin
namespace LeanNCD.Eval
open Std
private def t1 (xs : List Float) : DenseTensor := ⟨[xs.length], xs.toArray⟩
-- relu:
#guard DenseTensor.approxEq (reluT (t1 [-1, 2, -3, 4])) (t1 [0, 2, 0, 4])
-- softmax of a single row sums to 1 (axisPos 0, no mask):
run_cmd do
  let y := softmaxT 0 [1] none (t1 [1, 2, 3])
  let s := y.data.foldl (· + ·) 0.0
  unless Float.abs (s - 1.0) < 1e-6 do throwError s!"softmax row should sum to 1, got {s}"
  -- monotone: larger logit ⇒ larger prob
  unless y.data[0]! < y.data[1]! && y.data[1]! < y.data[2]! do throwError "softmax not monotone"
-- normalize: [1,1,2] ⇒ [0.25,0.25,0.5]
#guard DenseTensor.approxEq (normalizeT 0 [1] none (t1 [1,1,2])) (t1 [0.25, 0.25, 0.5])
-- masked softmax: a 2×2 with a mask excluding the upper-triangle (toy) still sums to 1 per row over unmasked
run_cmd do
  -- t shape [2,2], axisPos 1 (the second axis), uids [q=1, s=2]; mask: s ≤ q (causal)
  let t : DenseTensor := ⟨[2,2], #[1.0, 2.0, 0.5, 1.5]⟩
  let mask : BoolExpr := .rel .le (.embed (.axis ⟨"s",2,.real none⟩)) (.embed (.axis ⟨"q",1,.real none⟩))
  let y := softmaxT 1 [1,2] (some mask) t
  -- row q=0: only s=0 unmasked ⇒ y[0,0]=1, y[0,1]=0 ; row q=1: both unmasked ⇒ sums to 1
  unless Float.abs (y.get! [0,0] - 1.0) < 1e-6 && Float.abs (y.get! [0,1] - 0.0) < 1e-6 do throwError "masked row0"
  unless Float.abs ((y.get! [1,0] + y.get! [1,1]) - 1.0) < 1e-6 do throwError "masked row1 sum"
end LeanNCD.Eval
