import Eval.Portfolio.Harness
/-!
# Portfolio §10 — Losses, reductions & statistics

Sum-of-squares, uncentered covariance, mean via rank-0 `1/n`, marginalization, and the
rank-0 `−1` subtraction trick (ST5). ST6 (cross-entropy) needs a standalone `log` — a known
gap (KG-log) with no expressible program — so it is documented as a comment only.
-/
namespace LeanNCD.Eval
open Std

-- ST1  sum-of-squares (r aliased twice, full contraction): r=[1,-2,2] → 1+4+4 = 9.
run_cmd (assertEval "ST1 sse"
  (tlprog!{ sse[] := r[i] · r[i] })
  (HashMap.ofList [("r", tl [3] [1,-2,2])])
  "sse" (tl [] [9]))

-- ST2  uncentered covariance / scatter matrix: C[i,j]=Σ_k X[k,i]X[k,j], X=[[1,2],[3,4]].
--   C[0,0]=1+9=10, C[0,1]=2+12=14, C[1,0]=2+12=14, C[1,1]=4+16=20.
run_cmd (assertEval "ST2 covariance"
  (tlprog!{ C[i, j] := X[k, i] · X[k, j] })
  (HashMap.ofList [("X", tl [2,2] [1,2, 3,4])])
  "C" (tl [2,2] [10,14, 14,20]))

-- ST3  mean over samples via rank-0 1/n: m[j]=Σ_k X[k,j]·0.5, X=[[2,4],[6,8]].
--   m[0]=(2+6)·0.5=4, m[1]=(4+8)·0.5=6.
run_cmd (assertEval "ST3 mean"
  (tlprog!{ m[j] := X[k, j] · invn[] })
  (HashMap.ofList [("X", tl [2,2] [2,4, 6,8]), ("invn", tl [] [0.5])])
  "m" (tl [2] [4,6]))

-- ST4  probabilistic marginalization = sum out j: p[i]=Σ_j joint[i,j], joint=[[0.1,0.2],[0.3,0.4]].
--   p[0]=0.3, p[1]=0.7.
run_cmd (assertEval "ST4 marginalize"
  (tlprog!{ p[i] := joint[i, j] })
  (HashMap.ofList [("joint", tl [2,2] [0.1,0.2, 0.3,0.4])])
  "p" (tl [2] [0.3,0.7]))

-- ST5  residual via a rank-0 −1 scalar: r[i]=Yhat[i]+(−1)·Y[i], Yhat=[5,3], Y=[2,1].
--   r[0]=5−2=3, r[1]=3−1=2.
run_cmd (assertEval "ST5 residual-sub"
  (tlprog!{ r[i] := Yhat[i] + m1[] · Y[i] })
  (HashMap.ofList [("Yhat", tl [2] [5,3]), ("Y", tl [2] [2,1]), ("m1", tl [] [-1])])
  "r" (tl [2] [3,2]))

-- ST6  cross-entropy `L[] := − Y[i] · log(P[i])` — [F] gap KG-log: no standalone `log`.
--       No expressible program; documented, not authored.

end LeanNCD.Eval
