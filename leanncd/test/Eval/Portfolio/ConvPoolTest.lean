import Eval.Portfolio.Harness
/-!
# Portfolio §5 — Convolution & pooling

Numeric cross-checks for affine (windowed/dilated/strided) reads, max-pooling via
`maxreduce`, avg-pooling as a weighted sum, global sum-pool, and depthwise vs full
multi-channel convolution.  (CV3 strided conv is already covered by
`Eval.EvalExamplesTest`; not re-authored here.)
-/
namespace LeanNCD.Eval
open Std

-- CV1  1-D valid convolution: Y[i] = Σ_p W[p]·X[i+p] = X[i]+X[i+1]
--   X=[1,2,3,4] ⇒ [1+2, 2+3, 3+4] = [3,5,7]
run_cmd (assertEval "CV1 conv1d"
  (tlprog!{ Y[i] := W[p] · X[i + p] })
  (HashMap.ofList [("W", tl [2] [1,1]), ("X", tl [4] [1,2,3,4])])
  "Y" (tl [3] [3,5,7]))

-- CV2  2-D convolution with W=I₂: Y[i,j] = X[i,j] + X[i+1,j+1]
--   X=[[1,2,3],[4,5,6],[7,8,9]] ⇒ [[1+5,2+6],[4+8,5+9]] = [[6,8],[12,14]]
run_cmd (assertEval "CV2 conv2d"
  (tlprog!{ Y[i, j] := W[p, q] · X[i + p, j + q] })
  (HashMap.ofList [("W", tl [2,2] [1,0, 0,1]),
                   ("X", tl [3,3] [1,2,3, 4,5,6, 7,8,9])])
  "Y" (tl [2,2] [6,8, 12,14]))

-- CV4  dilated conv (dilation 2): Y[i] = Σ_p W[p]·X[i+2p] = X[i]+X[i+2]
--   X=[0,1,2,3,4] ⇒ [0+2, 1+3, 2+4] = [2,4,6]
run_cmd (assertEval "CV4 dilated-conv"
  (tlprog!{ Y[i] := W[p] · X[i + 2 * p] })
  (HashMap.ofList [("W", tl [2] [1,1]), ("X", tl [5] [0,1,2,3,4])])
  "Y" (tl [3] [2,4,6]))

-- CV5  max-pool stride-2 window-2: P[i] = max_p X[2i+p], p∈{0,1}.
--   p is otherwise unconstrained (single affine read over two free axes ⇒ solver
--   underdetermined), so it is pinned with `axis p : ℕ = 2` per the DSL reminder.
--   X=[1,3,2,5] ⇒ [max(1,3), max(2,5)] = [3,5]
run_cmd (assertEval "CV5 maxpool"
  (tlprog!{
    axis p : ℕ = 2
    P[i] := maxreduce(X[2 * i + p])
  })
  (HashMap.ofList [("X", tl [4] [1,3,2,5])])
  "P" (tl [2] [3,5]))

-- CV6  avg-pool as weighted sum: P[i] = Σ_p X[2i+p]·w[p]
--   w=[0.5,0.5], X=[2,4,6,8] ⇒ [0.5·2+0.5·4, 0.5·6+0.5·8] = [3,7]
run_cmd (assertEval "CV6 avgpool"
  (tlprog!{ P[i] := X[2 * i + p] · w[p] })
  (HashMap.ofList [("w", tl [2] [0.5,0.5]), ("X", tl [4] [2,4,6,8])])
  "P" (tl [2] [3,7]))

-- CV7  global sum-pool (contract all): s = Σ_i X[i]
--   X=[1,2,3,4] ⇒ 1+2+3+4 = 10
run_cmd (assertEval "CV7 sumpool"
  (tlprog!{ s[] := X[i] })
  (HashMap.ofList [("X", tl [4] [1,2,3,4])])
  "s" (tl [] [10]))

-- CV8  depthwise conv: channel c free on BOTH W and X, only p contracted.
--   Y[c,i] = Σ_p W[c,p]·X[c,i+p]
--   c=0: W[0]=[1,1]  ⇒ [X00+X01, X01+X02] = [1+2, 2+3] = [3,5]
--   c=1: W[1]=[1,-1] ⇒ [X10−X11, X11−X12] = [4−5, 5−6] = [-1,-1]
run_cmd (assertEval "CV8 depthwise"
  (tlprog!{ Y[c, i] := W[c, p] · X[c, i + p] })
  (HashMap.ofList [("W", tl [2,2] [1,1, 1,-1]),
                   ("X", tl [2,3] [1,2,3, 4,5,6])])
  "Y" (tl [2,2] [3,5, -1,-1]))

-- CV9  full multi-channel conv: contracts input-channel ci AND kernel p; out-channel co free.
--   Y[co,i] = Σ_{ci,p} W[co,ci,p]·X[ci,i+p].  W[0]=[[1,0],[0,1]] (indexed [ci,p])
--   ⇒ Y[0,i] = X[0,i] + X[1,i+1] = [1+5, 2+6] = [6,8]
run_cmd (assertEval "CV9 multichannel"
  (tlprog!{ Y[co, i] := W[co, ci, p] · X[ci, i + p] })
  (HashMap.ofList [("W", tl [1,2,2] [1,0, 0,1]),
                   ("X", tl [2,3] [1,2,3, 4,5,6])])
  "Y" (tl [1,2] [6,8]))

end LeanNCD.Eval
