import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §8 — Graph / message passing (GNN) + §8b — Consuming scatter outputs

§8 (GN1–GN4): dense/predicate message passing, node degree, max-aggregation.
  GN5 (degree-normalized D⁻¹AX, KG-div) and GN6 (edge-list scatter-add, KG-gather) are known
  gaps — left unauthored (see the `--` note at the end).

§8b (SC1–SC8): reading the output of a scatter statement in *later* statements — the
  decoder/GNN readback pattern closed by commits 57d333f / fc10d70 (B3). Doubles as a
  regression suite. Shared base is the 2× upsample `Out[2*i,2*j] := X[i,j]` with X = [[1,2],[3,4]]
  (⇒ Out is 4×4 with the input values at even coords, 0 elsewhere).
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§8 & §8b — GNN & scatter" <|
/- ## §8 Graph / message passing -/

-- GN1  dense message passing A·X. A = adjacency [[0,1],[1,0]] (swap), X = [[1,2],[3,4]].
--   H[i,f] = Σⱼ A[i,j]·X[j,f]  ⇒  row i gets neighbour j's features: [[3,4],[1,2]].
test "GN1 dense-message-passing"
    (evalEqB (tlprog!{ H[i, f] := A[i, j] · X[j, f] })
      (HashMap.ofList [("A", tl [2,2] [0,1, 1,0]), ("X", tl [2,2] [1,2, 3,4])])
      "H" (tl [2,2] [3,4, 1,2])) $

-- GN2  message passing over a PREDICATE adjacency. edge booleanizes to the same 0/1 matrix,
--   so H is identical to GN1: [[3,4],[1,2]].
test "GN2 predicate-message-passing"
    (evalEqB (tlprog!{ predicate edge(i, j)
            H[i, f] := edge[i, j] · X[j, f] })
      (HashMap.ofList [("edge", tl [2,2] [0,1, 1,0]), ("X", tl [2,2] [1,2, 3,4])])
      "H" (tl [2,2] [3,4, 1,2])) $

-- GN3  node degree = contract over neighbours. edge = [[0,1,1],[1,0,0]] ⇒ deg = [2,1].
test "GN3 degree"
    (evalEqB (tlprog!{ predicate edge(i, j)
            deg[i] := edge[i, j] })
      (HashMap.ofList [("edge", tl [2,3] [0,1,1, 1,0,0])])
      "deg" (tl [2] [2,1])) $

-- GN4  max-aggregation GNN (GraphSAGE-max). edge = [[1,1],[0,1]], X = [[1,2],[3,4]].
--   H[i,f] = maxⱼ edge[i,j]·X[j,f].
--     i=0: max(X[0], X[1]) = max([1,2],[3,4]) = [3,4].
--     i=1: max(0·X[0], X[1]) = max([0,0],[3,4]) = [3,4].
--   ⇒ [[3,4],[3,4]] (sum-agg would give [4,6] for row 0, so this genuinely exercises max).
test "GN4 max-aggregation"
    (evalEqB (tlprog!{ predicate edge(i, j)
            H[i, f] := maxreduce(edge[i, j] · X[j, f]) })
      (HashMap.ofList [("edge", tl [2,2] [1,1, 0,1]), ("X", tl [2,2] [1,2, 3,4])])
      "H" (tl [2,2] [3,4, 3,4])) $

/- ## §8b Consuming scatter outputs -/

-- SC1  reduce a scatter output to a scalar. total = Σ Out = Σ X = 1+2+3+4 = 10.
test "SC1 reduce-to-scalar"
    (evalEqB (tlprog!{ tensor Out(i, j)
            Out[2 * i, 2 * j] := X[i, j]
            total[] := Out[a, b] })
      (HashMap.ofList [("X", tl [2,2] [1,2, 3,4])])
      "total" (tl [] [10])) $

-- SC2  elementwise square of a scatter output. Nonzeros at even coords 1,2,3,4 ⇒ squares
--   1,4,9,16 in place (Σ = 30); structural zeros square to 0.
test "SC2 elementwise-square"
    (evalEqB (tlprog!{ tensor Out(i, j)
            Out[2 * i, 2 * j] := X[i, j]
            Sq[a, b] := Out[a, b] · Out[a, b] })
      (HashMap.ofList [("X", tl [2,2] [1,2, 3,4])])
      "Sq" (tl [4,4] [1,0,4,0, 0,0,0,0, 9,0,16,0, 0,0,0,0])) $

-- SC3  relu on a scatter output. X = [[1,-2],[-3,4]] placed at even coords; negatives clamp:
--   (0,0)=1, (0,2)=-2→0, (2,0)=-3→0, (2,2)=4 ⇒ 1,0,0,4 at evens.
test "SC3 relu"
    (evalEqB (tlprog!{ tensor Out(i, j)
            Out[2 * i, 2 * j] := X[i, j]
            R[a, b] := relu(Out[a, b]) })
      (HashMap.ofList [("X", tl [2,2] [1,-2, -3,4])])
      "R" (tl [4,4] [1,0,0,0, 0,0,0,0, 0,0,4,0, 0,0,0,0])) $

-- SC4  contraction consuming a scatter output. W = ones(4×2) ⇒ each column of Z = row-sums of
--   Out = [3,0,7,0] (row0 sums 1+2, row2 sums 3+4). Z is 4×2.
test "SC4 matmul"
    (evalEqB (tlprog!{ tensor Out(i, j)
            Out[2 * i, 2 * j] := X[i, j]
            Z[a, c] := Out[a, b] · W[b, c] })
      (HashMap.ofList [("X", tl [2,2] [1,2, 3,4]), ("W", tl [4,2] [1,1, 1,1, 1,1, 1,1])])
      "Z" (tl [4,2] [3,3, 0,0, 7,7, 0,0])) $

-- SC5  affine/strided read of a scatter output. Wk = [1,1] ⇒ Y[a,b] = Out[a,b] + Out[a+1,b];
--   valid extent 3×4. Y[0]=[1,0,2,0], Y[1]=Y[2]=[3,0,4,0].
test "SC5 conv"
    (evalEqB (tlprog!{ tensor Out(i, j)
            Out[2 * i, 2 * j] := X[i, j]
            Y[a, b] := Wk[p] · Out[a + p, b] })
      (HashMap.ofList [("X", tl [2,2] [1,2, 3,4]), ("Wk", tl [2] [1,1])])
      "Y" (tl [3,4] [1,0,2,0, 3,0,4,0, 3,0,4,0])) $

-- SC6  scatter reading another scatter (stacked upsampling). Out2[2a,2b] = Out[a,b], so the
--   nonzeros 1,2,3,4 (at Out coords (0,0),(0,2),(2,0),(2,2)) land at Out2 coords
--   (0,0),(0,4),(4,0),(4,4) in an 8×8.
test "SC6 scatter-of-scatter"
    (evalEqB (tlprog!{ tensor Out(i, j)
            tensor Out2(a, b)
            Out[2 * i, 2 * j] := X[i, j]
            Out2[2 * a, 2 * b] := Out[a, b] })
      (HashMap.ofList [("X", tl [2,2] [1,2, 3,4])])
      "Out2" (tl [8,8]
    [1,0,0,0,2,0,0,0,
     0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,
     3,0,0,0,4,0,0,0,
     0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0])) $

-- SC7  diagonal-write scatter consumed in a contraction (row scaling). D = diag(v) = [[2,0],[0,3]];
--   M = ones ⇒ Y[i,j] = Σₖ D[i,k]·M[k,j] = vᵢ ⇒ [[2,2],[3,3]].
test "SC7 diag-scatter-then-matmul"
    (evalEqB (tlprog!{ tensor D(i, j)
            D[i, i] := v[i]
            Y[i, j] := D[i, k] · M[k, j] })
      (HashMap.ofList [("v", tl [2] [2,3]), ("M", tl [2,2] [1,1, 1,1])])
      "Y" (tl [2,2] [2,2, 3,3])) $

-- SC8  softmax over a scatter output. A scatter's padding is genuine 0.0, so softmax treats
--   those cells as real entries: an all-zero row (rows 1,3) becomes uniform [¼,¼,¼,¼], not
--   undefined. Property check: every row sums to 1.
test "SC8 softmax-over-scatter"
    (evalPredB (tlprog!{ tensor Out(i, j)
            tensor P(a, b)
            Out[2 * i, 2 * j] := X[i, j]
            P[a, b.] := softmax(Out[a, b]) })
      (HashMap.ofList [("X", tl [2,2] [1,2, 3,4])])
      "P" rowsSumToOne)

/-
GN5 / GN6 — known gaps [F], left unauthored (would be red tests):
  GN5  degree-normalized  Ĥ = D⁻¹ A X  — needs per-node division  (KG-div).
  GN6  edge-list scatter-add  Msg[dst[e]] += X[src[e]]  — data-dependent gather/scatter with an
       index tensor (KG-gather); indices must be affine.
-/

end LeanNCD.Eval
