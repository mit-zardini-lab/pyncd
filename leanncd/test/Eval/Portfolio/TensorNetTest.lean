import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §12 — Tensor networks / decomposition

Chain contractions (tensor-train), CP reconstruction, Frobenius inner product, and a
third-order moment (3× aliasing). All `[N]` — hand-computed expected tensors below.
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§12 — Tensor networks / decomposition" <|
-- TN1  tensor-train / chain contraction. Three 2×2 identity cores ⇒ I·I·I = I₂.
test "TN1 tensor-train"
    (evalEqB (tlprog!{ Y[a, d] := G1[a, b] · G2[b, c] · G3[c, d] })
      (HashMap.ofList [("G1", tl [2,2] [1,0,0,1]), ("G2", tl [2,2] [1,0,0,1]),
                   ("G3", tl [2,2] [1,0,0,1])])
      "Y" (tl [2,2] [1,0,0,1])) $

-- TN2  CP reconstruction (shared contracted rank axis `r`, here size 1 ⇒ pure outer product).
-- A=[[1],[2]], B=[[3],[4]], C=[[5],[6]] (all 2×1). T[i,j,k] = A[i,0]·B[j,0]·C[k,0]:
--   T[0,0,0]=15 T[0,0,1]=18 T[0,1,0]=20 T[0,1,1]=24
--   T[1,0,0]=30 T[1,0,1]=36 T[1,1,0]=40 T[1,1,1]=48
test "TN2 CP-reconstruction"
    (evalEqB (tlprog!{ T[i, j, k] := A[i, r] · B[j, r] · C[k, r] })
      (HashMap.ofList [("A", tl [2,1] [1,2]), ("B", tl [2,1] [3,4]), ("C", tl [2,1] [5,6])])
      "T" (tl [2,2,2] [15,18,20,24,30,36,40,48])) $

-- TN3  Frobenius inner product ⟨A,B⟩ (full contraction over 2 axes). A=[[1,2],[3,4]], B=I₂ ⇒
-- 1·1 + 2·0 + 3·0 + 4·1 = 5.
test "TN3 frobenius"
    (evalEqB (tlprog!{ s[] := A[i, j] · B[i, j] })
      (HashMap.ofList [("A", tl [2,2] [1,2,3,4]), ("B", tl [2,2] [1,0,0,1])])
      "s" (tl [] [5])) $

-- TN4  third-order moment M[i,j,k] := X[t,i]·X[t,j]·X[t,k] (3× aliasing, shared contracted t).
-- X=[[1,0],[0,1]] (t=2): only i=j=k=0 (t=0) and i=j=k=1 (t=1) survive ⇒ diagonal 3-tensor
-- M[0,0,0]=M[1,1,1]=1, all else 0.
test "TN4 third-moment"
    (evalEqB (tlprog!{ M[i, j, k] := X[t, i] · X[t, j] · X[t, k] })
      (HashMap.ofList [("X", tl [2,2] [1,0,0,1])])
      "M" (tl [2,2,2] [1,0,0,0,0,0,0,1]))

end LeanNCD.Eval
