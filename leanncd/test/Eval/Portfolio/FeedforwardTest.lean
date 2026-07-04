import Eval.Portfolio.Harness
/-!
# Portfolio §3 — Feedforward / MLP

Numeric cross-checks for relu MLP layers, the `linear`/`bias` declarations, and the
affine (bias-add) layer path. All expected outputs are hand-computed and shown in comments.
-/
namespace LeanNCD.Eval
open Std

-- FF1  two-layer MLP with intermediate `H`, `linear W_in(f,d), W_out(d,f) bias` decls.
--   X = [[1,2]] (q=1,d=2), W_in = [[1,1],[1,-1]] (f=2,d=2), W_out = I₂ (d=2,f=2).
--   H_pre[0] = [W_in[0]·X[0], W_in[1]·X[0]] = [1·1+1·2, 1·1+(-1)·2] = [3, -1]; relu ⇒ H = [3, 0].
--   Out[0] = [W_out[0]·H[0], W_out[1]·H[0]] = [1·3+0·0, 0·3+1·0] = [3, 0].  (bias decl is metadata;
--   the equations carry no bias term, so the output is the pure two-layer product.)
run_cmd (assertEval "FF1 mlp-2layer"
  (tlprog!{
    linear W_in(f, d), W_out(d, f) bias
    H[q, f]   := relu(W_in[f, d] · X[q, d])
    Out[q, d] := W_out[d, f] · H[q, f]
  })
  (HashMap.ofList [("X", tl [1,2] [1,2]),
                   ("W_in", tl [2,2] [1,1, 1,-1]),
                   ("W_out", tl [2,2] [1,0, 0,1])])
  "Out" (tl [1,2] [3,0]))

-- FF2  relu clamps both negatives.  W=[[1,-1],[-2,1]], x=[1,1].
--   pre = [1·1+(-1)·1, -2·1+1·1] = [0, -1]; relu ⇒ [0, 0].
run_cmd (assertEval "FF2 relu-clamp"
  (tlprog!{ H[i] := relu(W[i, j] · x[j]) })
  (HashMap.ofList [("W", tl [2,2] [1,-1, -2,1]), ("x", tl [2] [1,1])])
  "H" (tl [2] [0,0]))

-- FF3  relu asymmetric.  W=[[1,1],[-1,-1]], x=[2,1].
--   pre = [1·2+1·1, -1·2+(-1)·1] = [3, -3]; relu ⇒ [3, 0].
run_cmd (assertEval "FF3 relu-asym"
  (tlprog!{ H[i] := relu(W[i, j] · x[j]) })
  (HashMap.ofList [("W", tl [2,2] [1,1, -1,-1]), ("x", tl [2] [2,1])])
  "H" (tl [2] [3,0]))

-- FF4  affine layer `Y[i] := W[i,j]·x[j] + b[i]` (bias-add path).
--   NOTE (equation-level summation, §12c): `j` is summed over the WHOLE RHS, so the `j`-less
--   term `b[i]` is broadcast by |j|.  To exercise a clean bias add we pin |j| = 1 (so the
--   broadcast factor is 1):  W=[[2],[3]] (i=2,j=1), x=[5], b=[1,1].
--   Y[i] = W[i,0]·x[0] + b[i] = [2·5+1, 3·5+1] = [11, 16].
run_cmd (assertEval "FF4 affine-bias"
  (tlprog!{ Y[i] := W[i, j] · x[j] + b[i] })
  (HashMap.ofList [("W", tl [2,1] [2, 3]), ("x", tl [1] [5]), ("b", tl [2] [1,1])])
  "Y" (tl [2] [11,16]))

end LeanNCD.Eval
