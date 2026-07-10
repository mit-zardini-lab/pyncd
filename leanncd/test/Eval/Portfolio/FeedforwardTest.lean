import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §3 — Feedforward / MLP

Numeric cross-checks for relu MLP layers, the `linear`/`bias` declarations, and the
affine (bias-add) layer path. All expected outputs are hand-computed and shown in comments.
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§3 — Feedforward / MLP" <|
-- FF1  two-layer MLP with intermediate `H`, `linear W_in(f,d), W_out(d,f) bias` decls.
--   X = [[1,2]] (q=1,d=2), W_in = [[1,1],[1,-1]] (f=2,d=2), W_out = I₂ (d=2,f=2).
--   H_pre[0] = [W_in[0]·X[0], W_in[1]·X[0]] = [1·1+1·2, 1·1+(-1)·2] = [3, -1]; relu ⇒ H = [3, 0].
--   Out[0] = [W_out[0]·H[0], W_out[1]·H[0]] = [1·3+0·0, 0·3+1·0] = [3, 0].  (bias decl is metadata;
--   the equations carry no bias term, so the output is the pure two-layer product.)
test "FF1 mlp-2layer"
    (evalEqB (tlprog!{
    linear W_in(f, d), W_out(d, f) bias
    H[q, f]   := relu(W_in[f, d] · X[q, d])
    Out[q, d] := W_out[d, f] · H[q, f]
  })
      (HashMap.ofList [("X", tl [1,2] [1,2]),
                   ("W_in", tl [2,2] [1,1, 1,-1]),
                   ("W_out", tl [2,2] [1,0, 0,1])])
      "Out" (tl [1,2] [3,0])) $

-- FF2  relu clamps both negatives.  W=[[1,-1],[-2,1]], x=[1,1].
--   pre = [1·1+(-1)·1, -2·1+1·1] = [0, -1]; relu ⇒ [0, 0].
test "FF2 relu-clamp"
    (evalEqB (tlprog!{ H[i] := relu(W[i, j] · x[j]) })
      (HashMap.ofList [("W", tl [2,2] [1,-1, -2,1]), ("x", tl [2] [1,1])])
      "H" (tl [2] [0,0])) $

-- FF3  relu asymmetric.  W=[[1,1],[-1,-1]], x=[2,1].
--   pre = [1·2+1·1, -1·2+(-1)·1] = [3, -3]; relu ⇒ [3, 0].
test "FF3 relu-asym"
    (evalEqB (tlprog!{ H[i] := relu(W[i, j] · x[j]) })
      (HashMap.ofList [("W", tl [2,2] [1,1, -1,-1]), ("x", tl [2] [2,1])])
      "H" (tl [2] [3,0])) $

-- FF5  sigmoid (KG-activation).  W=I₂, x=[-2,2] ⇒ pre=x exactly.  Also the regression test for
--   the `Eval.lean` wildcard hazard: this statement has NO `.`-marked axis anywhere — if the
--   pointwise arm were missing, this would wrongly throw "no output axis is marked" instead of
--   evaluating.  sigmoid(-2)=1/(1+e²)≈0.1192029, sigmoid(2)=1/(1+e⁻²)≈0.8807971.
test "FF5 sigmoid"
    (evalEqB (tlprog!{ H[i] := sigmoid(W[i, j] · x[j]) })
      (HashMap.ofList [("W", tl [2,2] [1,0, 0,1]), ("x", tl [2] [-2,2])])
      "H" (tl [2] [0.11920292202211755, 0.8807970779778823])) $

-- FF6  tanh (KG-activation).  Same pre=[-2,2] setup as FF5 (unmarked, same wildcard-hazard check).
--   tanh(-2)≈-0.9640276, tanh(2)≈0.9640276.
test "FF6 tanh"
    (evalEqB (tlprog!{ H[i] := tanh(W[i, j] · x[j]) })
      (HashMap.ofList [("W", tl [2,2] [1,0, 0,1]), ("x", tl [2] [-2,2])])
      "H" (tl [2] [-0.9640275800758169, 0.9640275800758169])) $

-- FF7  gelu, tanh-approximation (KG-activation).  Same pre=[-2,2] setup.
--   gelu(x) = 0.5·x·(1+tanh(√(2/π)·(x+0.044715x³))): gelu(-2)≈-0.0454023, gelu(2)≈1.9545977.
test "FF7 gelu"
    (evalEqB (tlprog!{ H[i] := gelu(W[i, j] · x[j]) })
      (HashMap.ofList [("W", tl [2,2] [1,0, 0,1]), ("x", tl [2] [-2,2])])
      "H" (tl [2] [-0.045402305912, 1.954597694088])) $

-- FF8  leaky relu, fixed 0.01 negative slope (KG-activation).  Same pre=[-2,2] setup.
--   leakyrelu(-2)=0.01·(-2)=-0.02, leakyrelu(2)=2 (unchanged, ≥0).
test "FF8 leakyrelu"
    (evalEqB (tlprog!{ H[i] := leakyrelu(W[i, j] · x[j]) })
      (HashMap.ofList [("W", tl [2,2] [1,0, 0,1]), ("x", tl [2] [-2,2])])
      "H" (tl [2] [-0.02, 2])) $

-- FF4  affine layer `Y[i] := W[i,j]·x[j] + b[i]` (bias-add path).
--   NOTE (equation-level summation, §12c): `j` is summed over the WHOLE RHS, so the `j`-less
--   term `b[i]` is broadcast by |j|.  To exercise a clean bias add we pin |j| = 1 (so the
--   broadcast factor is 1):  W=[[2],[3]] (i=2,j=1), x=[5], b=[1,1].
--   Y[i] = W[i,0]·x[0] + b[i] = [2·5+1, 3·5+1] = [11, 16].
test "FF4 affine-bias"
    (evalEqB (tlprog!{ Y[i] := W[i, j] · x[j] + b[i] })
      (HashMap.ofList [("W", tl [2,1] [2, 3]), ("x", tl [1] [5]), ("b", tl [2] [1,1])])
      "Y" (tl [2] [11,16]))

end LeanNCD.Eval
