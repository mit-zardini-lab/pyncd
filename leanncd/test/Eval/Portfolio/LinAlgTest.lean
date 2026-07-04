import Eval.Portfolio.Harness
/-!
# Portfolio §2 — Core linear algebra

Numeric cross-checks for contraction, broadcasting, aliasing and multi-axis reductions.
(LA2 matmul is already covered by `Eval.EvalExamplesTest`; not re-authored here.)
-/
namespace LeanNCD.Eval
open Std

-- LA1  mat-vec: contract to a vector
run_cmd (assertEval "LA1 matvec"
  (tlprog!{ y[i] := A[i, j] · x[j] })
  (HashMap.ofList [("A", tl [2,2] [1,2,3,4]), ("x", tl [2] [1,1])])
  "y" (tl [2] [3,7]))

-- LA3  batched matmul (batch axis b shared + free)
run_cmd (assertEval "LA3 batched-matmul"
  (tlprog!{ Y[b, i, j] := A[b, i, k] · X[b, k, j] })
  (HashMap.ofList [("A", tl [1,2,2] [1,0,0,1]), ("X", tl [1,2,2] [5,6,7,8])])
  "Y" (tl [1,2,2] [5,6,7,8]))

-- LA4  outer product (no contraction)
run_cmd (assertEval "LA4 outer"
  (tlprog!{ Y[i, j] := a[i] · b[j] })
  (HashMap.ofList [("a", tl [2] [1,2]), ("b", tl [2] [3,4])])
  "Y" (tl [2,2] [3,4,6,8]))

-- LA5  inner product → scalar (full contraction)
run_cmd (assertEval "LA5 inner"
  (tlprog!{ s[] := x[i] · y[i] })
  (HashMap.ofList [("x", tl [3] [1,2,3]), ("y", tl [3] [1,1,1])])
  "s" (tl [] [6]))

-- LA6  Gram matrix = X Xᵀ (aliasing: same tensor read twice)
run_cmd (assertEval "LA6 gram"
  (tlprog!{ G[i, j] := X[i, k] · X[j, k] })
  (HashMap.ofList [("X", tl [3,2] [1,0, 0,1, 1,1])])
  "G" (tl [3,3] [1,0,1, 0,1,1, 1,1,2]))

-- LA7  bilinear form xᵀWy → scalar (3-factor)
run_cmd (assertEval "LA7 bilinear"
  (tlprog!{ s[] := x[i] · W[i, j] · y[j] })
  (HashMap.ofList [("x", tl [2] [1,1]), ("W", tl [2,2] [1,2,3,4]), ("y", tl [2] [1,0])])
  "s" (tl [] [4]))

-- LA8  Hadamard (all axes free-shared, none contracted)
run_cmd (assertEval "LA8 hadamard"
  (tlprog!{ Z[i, j] := A[i, j] · B[i, j] })
  (HashMap.ofList [("A", tl [2,2] [1,2,3,4]), ("B", tl [2,2] [1,0,0,1])])
  "Z" (tl [2,2] [1,0,0,4]))

-- LA9  multiple contracted axes (j, k)
run_cmd (assertEval "LA9 multi-contract"
  (tlprog!{ s[i] := A[i, j, k] · B[j, k] })
  (HashMap.ofList [("A", tl [2,2,2] [1,1,1,1,1,1,1,1]), ("B", tl [2,2] [1,2,3,4])])
  "s" (tl [2] [10,10]))

end LeanNCD.Eval
