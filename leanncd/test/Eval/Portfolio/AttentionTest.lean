import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §4 — Attention & transformers

Numeric / property cross-checks for the attention family.  AT2 (causal mask) is already
covered by `Eval.EvalExamplesTest` and is not re-authored here.  Softmax/normalize values are
independently hand-computed (max-subtracted softmax; masked entries → 0).
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§4 — Attention & transformers" <|
-- AT1  unmasked self-attention scores → row softmax.  Q=K=I₂ ⇒ scores = I₂.
--   Row0 softmax([1,0]) = [e/(e+1), 1/(e+1)] = [0.73105857863, 0.26894142137];  Row1 mirror.
test "AT1 self-attn softmax"
    (evalEqB (tlprog!{
    tensor A(q, s)
    A[q, s.] := softmax(Q[q, d] · K[s, d])
  })
      (HashMap.ofList [("Q", tl [2,2] [1,0, 0,1]), ("K", tl [2,2] [1,0, 0,1])])
      "A" (tl [2,2] [0.7310585786300049, 0.2689414213699951,
                 0.2689414213699951, 0.7310585786300049])) $

-- AT3  full attention output O = A·V, chaining AT1's A with V=[[1,2],[3,4]].
--   A = [[0.73105857863,0.26894142137],[0.26894142137,0.73105857863]].
--   O[0] = [0.73106·1+0.26894·3, 0.73106·2+0.26894·4] = [1.53788284274, 2.53788284274]
--   O[1] = [0.26894·1+0.73106·3, 0.26894·2+0.73106·4] = [2.46211715726, 3.46211715726]
test "AT3 attn-output"
    (evalEqB (tlprog!{
    tensor A(q, s)
    A[q, s.] := softmax(Q[q, d] · K[s, d])
    O[q, e]  := A[q, s] · V[s, e]
  })
      (HashMap.ofList [("Q", tl [2,2] [1,0, 0,1]), ("K", tl [2,2] [1,0, 0,1]),
                   ("V", tl [2,2] [1,2, 3,4])])
      "O" (tl [2,2] [1.5378828427399902, 2.5378828427399904,
                 2.46211715726001,   3.4621171572600096])) $

-- AT4  multi-head / batched attention (b,h free).  Q=K = I₂ in the last two dims,
--   shape [1,1,2,2] ⇒ A equals AT1 broadcast into the [1,1,·,·] frame.
test "AT4 mha-batched"
    (evalEqB (tlprog!{
    tensor A(b, h, q, s)
    A[b, h, q, s.] := softmax(Q[b, h, q, d] · K[b, h, s, d])
  })
      (HashMap.ofList [("Q", tl [1,1,2,2] [1,0, 0,1]), ("K", tl [1,1,2,2] [1,0, 0,1])])
      "A" (tl [1,1,2,2] [0.7310585786300049, 0.2689414213699951,
                     0.2689414213699951, 0.7310585786300049])) $

-- AT5  cross-attention with distinct query/key lengths.  Q (2×2), K (3×2) ⇒ A is 2×3;
--   property: shape [2,3] and each query row sums to 1 (proper distribution over keys).
test "AT5 cross-attn"
    (evalPredB (tlprog!{
    tensor A(q, s)
    A[q, s.] := softmax(Q[q, d] · K[s, d])
  })
      (HashMap.ofList [("Q", tl [2,2] [1,0, 0,1]), ("K", tl [3,2] [1,0, 0,1, 1,1])])
      "A" (fun t => t.shape == [2,3] && rowsSumToOne t)) $

-- AT6  scaled scores via a rank-0 tensor read `scale[]`.  Q=K=I₂, scale=0.5 ⇒ scores = 0.5·I₂.
--   Row0 softmax([0.5,0]) = [e^0.5/(e^0.5+1), 1/(e^0.5+1)] = [0.62245933120, 0.37754066880]; mirror.
test "AT6 scaled-scores"
    (evalEqB (tlprog!{
    tensor A(q, s)
    A[q, s.] := softmax(Q[q, d] · K[s, d] · scale[])
  })
      (HashMap.ofList [("Q", tl [2,2] [1,0, 0,1]), ("K", tl [2,2] [1,0, 0,1]),
                   ("scale", tl [] [0.5])])
      "A" (tl [2,2] [0.6224593312018546, 0.37754066879814546,
                 0.37754066879814546, 0.6224593312018546])) $

-- AT7  full 1-layer transformer block (attn + FFN + residual + normalize), identity weights,
--   toy sizes SEQ=2,D=2,H=1,K=2 (same block as `EvalExamplesTest` ex.12, whose H output is
--   hand-derived there).  Causal q=0 row is the fixed point [1,0]; q=1 row = [0.13447, 0.86553].
test "AT7 transformer-block"
    (evalEqB (tlprog!{
    Q[q, h, k]       := W_Q[h, k, m] · X[q, m]
    K[s, h, k]       := W_K[h, k, m] · X[s, m]
    V[s, h, k]       := W_V[h, k, m] · X[s, m]
    tensor S(h, q, s)
    S[h, q, s.]      := softmax(where s ≤ q)(Q[q, h, k] · K[s, h, k])
    AttnOut[q, h, k] := S[h, q, s] · V[s, h, k]
    Attn[q, m]       := W_O[m, h, k] · AttnOut[q, h, k]
    tensor A(q, m)
    A[q, m.]         := normalize(Attn[q, m] + X[q, m])
    F[q, d]          := relu(W_in[d, m] · A[q, m])
    Y[q, m]          := W_out[m, d] · F[q, d]
    tensor H(q, m)
    H[q, m.]         := normalize(Y[q, m] + A[q, m])
  })
      (HashMap.ofList [("X",    tl [2,2] [1,0, 0,1]),
                   ("W_Q",  tl [1,2,2] [1,0, 0,1]),
                   ("W_K",  tl [1,2,2] [1,0, 0,1]),
                   ("W_V",  tl [1,2,2] [1,0, 0,1]),
                   ("W_O",  tl [2,1,2] [1,0, 0,1]),
                   ("W_in", tl [2,2] [1,0, 0,1]),
                   ("W_out", tl [2,2] [1,0, 0,1])])
      "H" (tl [2,2] [1.0, 0.0, 0.13447071068499755, 0.8655292893150025])) $

-- AT8  generalized cross-attention: four decoupled axes (q,s,h,g).
--   Q=I₂ (q×h), K=[[1,0],[0,1],[1,1]] (s×h), V=[[2,0],[0,2],[1,1]] (s×g).
--   scores row0=[1,0,1] row1=[0,1,1]; softmax rows ⇒ A; O=A·V (V rows all sum to 2 ⇒ O rows sum to 2).
--   O = [[1.26695639475, 0.73304360525], [0.73304360525, 1.26695639475]].
test "AT8 generalized-cross-attn"
    (evalEqB (tlprog!{
    tensor A(q, s)
    A[q, s.] := softmax(Q[q, h] · K[s, h])
    O[q, g]  := A[q, s] · V[s, g]
  })
      (HashMap.ofList [("Q", tl [2,2] [1,0, 0,1]),
                   ("K", tl [3,2] [1,0, 0,1, 1,1]),
                   ("V", tl [3,2] [2,0, 0,2, 1,1])])
      "O" (tl [2,2] [1.2669563947545546, 0.7330436052454454,
                 0.7330436052454454, 1.2669563947545546])) $

-- AT9  linear (kernelized) attention: reassociate the contraction — sum keys first (M[h,e]),
--   then queries.  Φ=I₂ (for both PhiK and PhiQ), V=[[1,2],[3,4]] ⇒ M=V, O=M=V=[[1,2],[3,4]].
test "AT9 linear-attn"
    (evalEqB (tlprog!{
    M[h, e] := PhiK[s, h] · V[s, e]
    O[q, e] := PhiQ[q, h] · M[h, e]
  })
      (HashMap.ofList [("PhiK", tl [2,2] [1,0, 0,1]), ("PhiQ", tl [2,2] [1,0, 0,1]),
                   ("V", tl [2,2] [1,2, 3,4])])
      "O" (tl [2,2] [1,2, 3,4])) $

-- AT10  bilinear attention: learned qᵀWk scoring (3-factor contraction inside softmax).
--   Q=K=I₂, W=I₂ ⇒ scores = I₂ = AT1's scores; rows = [0.73105857863, 0.26894142137] / mirror.
test "AT10 bilinear-attn"
    (evalEqB (tlprog!{
    tensor S(q, s)
    S[q, s.] := softmax(Q[q, a] · W[a, b] · K[s, b])
  })
      (HashMap.ofList [("Q", tl [2,2] [1,0, 0,1]), ("W", tl [2,2] [1,0, 0,1]),
                   ("K", tl [2,2] [1,0, 0,1])])
      "S" (tl [2,2] [0.7310585786300049, 0.2689414213699951,
                 0.2689414213699951, 0.7310585786300049])) $

-- AT11  grouped / multi-query attention: K has no head axis `h` ⇒ shared across heads (GQA/MQA).
--   Q (h=2,q=2,d=2), K (s=2,d=2).  Property: A shape [2,2,2] and each (h,q) row sums to 1.
test "AT11 gqa"
    (evalPredB (tlprog!{
    tensor A(h, q, s)
    A[h, q, s.] := softmax(Q[h, q, d] · K[s, d])
  })
      (HashMap.ofList [("Q", tl [2,2,2] [1,0, 0,1,  1,0, 0,1]), ("K", tl [2,2] [1,0, 0,1])])
      "A" (fun t => t.shape == [2,2,2] && rowsSumToOne t)) $

-- AT12  sparse attention (Longformer local window ∨ global token s=0).  Q=K=I₃, raw scores = I₃.
--   Row0: keep s∈{0,1} (|q-s|≤1), mask s=2 ⇒ softmax([1,0]) = [0.73106,0.26894,0].
--   Row1: keep all (|q-s|≤1) ⇒ softmax([0,1,0]) = [0.21194,0.57612,0.21194].
--   Row2: |q-s|≤1 keeps {1,2}, s=0 kept as global ⇒ softmax([0,0,1]) = [0.21194,0.21194,0.57612].
--   So A[0,2]=0 (out of window) and A[2,0]>0 (global token attended from q=2).
test "AT12 sparse-attn"
    (evalEqB (tlprog!{
    tensor A(q, s)
    A[q, s.] := softmax(where |q - s| ≤ 1 ∨ s = 0)(Q[q, d] · K[s, d])
  })
      (HashMap.ofList [("Q", tl [3,3] [1,0,0, 0,1,0, 0,0,1]),
                   ("K", tl [3,3] [1,0,0, 0,1,0, 0,0,1])])
      "A" (tl [3,3] [0.7310585786300049, 0.2689414213699951, 0.0,
                 0.21194155761708544, 0.5761168847658291, 0.21194155761708544,
                 0.21194155761708544, 0.21194155761708544, 0.5761168847658291]))

end LeanNCD.Eval
