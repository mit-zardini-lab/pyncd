import Eval.Portfolio.Harness
/-!
# Portfolio §6 — Normalization

L1 `normalize` and `softmax` over the marked reduction axis: exact distributions,
numerical stability under large logits, masked renormalization, and a softmax over a
NON-last axis (column softmax / axial attention).
-/
namespace LeanNCD.Eval
open Std

-- NM1  L1 normalize over the marked axis s: Y[q,s] = A[q,s] / Σ_s A[q,s]
--   A=[[1,3],[2,2]] ⇒ [[1/4,3/4],[2/4,2/4]] = [[0.25,0.75],[0.5,0.5]]
run_cmd (assertEval "NM1 normalize"
  (tlprog!{ Y[q, s.] := normalize(A[q, s]) })
  (HashMap.ofList [("A", tl [2,2] [1,3, 2,2])])
  "Y" (tl [2,2] [0.25,0.75, 0.5,0.5]))

-- NM2  plain softmax → distribution: Y[q,s] = softmax_s(A[q,s])
--   row0: softmax([0,0]) = [0.5,0.5]
--   row1: softmax([0,ln3]) = [1,3]/(1+3) = [0.25,0.75]
run_cmd (assertEval "NM2 softmax"
  (tlprog!{ Y[q, s.] := softmax(A[q, s]) })
  (HashMap.ofList [("A", tl [2,2] [0, 0, 0, Float.log 3])])
  "Y" (tl [2,2] [0.5,0.5, 0.25,0.75]))

-- NM3  numerical stability: softmax([1000,1001]) must not overflow.
--   Max-subtraction ⇒ softmax([1000,1001]) = softmax([0,1]) = [1/(1+e), e/(1+e)]
--                                            ≈ [0.26894, 0.73106].
--   Property test: both entries finite (no NaN/∞) AND equal to the stable reference.
run_cmd (assertEvalPred "NM3 softmax-stable"
  (tlprog!{ Y[q, s.] := softmax(A[q, s]) })
  (HashMap.ofList [("A", tl [1,2] [1000, 1001])])
  "Y"
  (fun t =>
    let e := Float.exp 1.0
    let a := 1.0 / (1.0 + e)
    let b := e / (1.0 + e)
    t.data.size == 2
      && Float.isFinite (t.data.getD 0 0.0) && Float.isFinite (t.data.getD 1 0.0)
      && Float.abs (t.data.getD 0 0.0 - a) < 1e-6
      && Float.abs (t.data.getD 1 0.0 - b) < 1e-6)
  "stable softmax([1000,1001]) = [1/(1+e), e/(1+e)], finite")

-- NM4  masked L1 normalize: mask keeps s ≠ 0, renormalizes the survivors, masked → 0.
--   A=[[1,2,3],[4,1,1]]:
--     row0 survivors [2,3] ⇒ [0, 2/5, 3/5] = [0,0.4,0.6]
--     row1 survivors [1,1] ⇒ [0, 1/2, 1/2] = [0,0.5,0.5]
--   Property test: the masked column (s=0) is exactly 0, and each row sums to 1.
run_cmd (assertEvalPred "NM4 masked-normalize"
  (tlprog!{ Y[q, s.] := normalize(where s ≠ 0)(A[q, s]) })
  (HashMap.ofList [("A", tl [2,3] [1,2,3, 4,1,1])])
  "Y"
  (fun t => t.shape == [2,3]
    && Float.abs (t.data.getD 0 1.0) < 1e-9   -- Y[0,0] masked → 0
    && Float.abs (t.data.getD 3 1.0) < 1e-9   -- Y[1,0] masked → 0
    && rowsSumToOne t)
  "masked s=0 entries are 0; each row renormalized to sum 1")

-- NM5  softmax over a NON-last axis (dot on slot 0 ⇒ reduce over s): A[s,q] = softmax_s(scores).
--   scores S[s,q] = Σ_d Q[q,d]·K[s,d]; Q=K=I₂ ⇒ S = I₂ (S[s,q]=[s=q]).
--   For each column q, softmax over s of [S[0,q],S[1,q]]:
--     q=0: softmax([1,0]) = [e/(1+e), 1/(1+e)] ≈ [0.731,0.269]
--     q=1: softmax([0,1]) = [1/(1+e), e/(1+e)] ≈ [0.269,0.731]
--   ⇒ A = [[0.731,0.269],[0.269,0.731]]; each COLUMN (not row) sums to 1.
run_cmd (assertEvalPred "NM5 column-softmax"
  (tlprog!{ A[s., q] := softmax(Q[q, d] · K[s, d]) })
  (HashMap.ofList [("Q", tl [2,2] [1,0, 0,1]), ("K", tl [2,2] [1,0, 0,1])])
  "A"
  (fun t => match t.shape with
    | [n, w] =>
        (List.range w).all (fun c =>
          let sm := (List.range n).foldl (fun acc r => acc + t.data.getD (r*w+c) 0.0) 0.0
          Float.abs (sm - 1.0) < 1e-6)
    | _ => false)
  "each column sums to 1 (softmax over non-last axis s)")

end LeanNCD.Eval
