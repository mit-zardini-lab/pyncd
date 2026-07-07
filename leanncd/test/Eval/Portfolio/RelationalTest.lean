import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §9 — Relational / logic (tensor-logic's logic side)

Iverson-predicate tensors, boolean-semiring joins, and mask selection.
(RL5 band-mask is already covered elsewhere; not re-authored here.)
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§9 — Relational / logic" <|
-- RL1  pure-Iverson identity: sizes come from `axis` decls only (no read binds i,j)
test "RL1 identity"
    (evalEqB (tlprog!{ axis i : ℕ = 3, j : ℕ = 3
            I[i, j] := [i = j] })
      (HashMap.ofList [])
      "I" (tl [3,3] [1,0,0, 0,1,0, 0,0,1])) $

-- RL2  relational join / composition (boolean semiring product): swap ∘ swap = identity.
--   R[i,k] = Σ_j E1[i,j]·E2[j,k]; E1=E2=swap → R[0,0]=E1[0,1]E2[1,0]=1, R[1,1]=E1[1,0]E2[0,1]=1.
test "RL2 join"
    (evalEqB (tlprog!{ predicate E1(i, j), E2(j, k)
            R[i, k] := E1[i, j] · E2[j, k] })
      (HashMap.ofList [("E1", tl [2,2] [0,1, 1,0]), ("E2", tl [2,2] [0,1, 1,0])])
      "R" (tl [2,2] [1,0, 0,1])) $

-- RL3  strict-upper-triangular selection via Iverson `[i < j]`.
--   A=[[1,2,3],[4,5,6],[7,8,9]] keep i<j → [[0,2,3],[0,0,6],[0,0,0]].
test "RL3 strict-upper"
    (evalEqB (tlprog!{ U[i, j] := A[i, j] · [i < j] })
      (HashMap.ofList [("A", tl [3,3] [1,2,3, 4,5,6, 7,8,9])])
      "U" (tl [3,3] [0,2,3, 0,0,6, 0,0,0])) $

-- RL4  length-2 reachability (path count): E aliased twice, E=[[0,1],[0,0]].
--   P2[i,k]=Σ_j E[i,j]E[j,k]; only E[0,1]=1 and E[1,·]=0 ⇒ no 2-paths ⇒ all zero.
test "RL4 path-count"
    (evalEqB (tlprog!{ P2[i, k] := E[i, j] · E[j, k] })
      (HashMap.ofList [("E", tl [2,2] [0,1, 0,0])])
      "P2" (tl [2,2] [0,0, 0,0])) $

-- RL6  compound `∧` mask: local window i ≤ j ∧ j ≤ i+2 over an all-ones 4×4.
--   keep j∈{i,i+1,i+2}: row0 {0,1,2}, row1 {1,2,3}, row2 {2,3}, row3 {3}.
test "RL6 window-and"
    (evalEqB (tlprog!{ S[i, j] := A[i, j] · [i ≤ j ∧ j ≤ i + 2] })
      (HashMap.ofList [("A", tl [4,4] [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])])
      "S" (tl [4,4] [1,1,1,0, 0,1,1,1, 0,0,1,1, 0,0,0,1])) $

-- RL7  `ieq`/`imul` builtins: keep where 2·i = j over an all-ones 3×5.
--   row0 keep j=0, row1 keep j=2, row2 keep j=4.
test "RL7 ieq-imul"
    (evalEqB (tlprog!{ M[i, j] := A[i, j] · [ieq(imul(i, 2), j)] })
      (HashMap.ofList [("A", tl [3,5] [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1])])
      "M" (tl [3,5] [1,0,0,0,0, 0,0,1,0,0, 0,0,0,0,1])) $

-- RL8  negated mask (exclude self): A=[[1,2],[3,4]], zero the diagonal → [[0,2],[3,0]].
test "RL8 negate-diag"
    (evalEqB (tlprog!{ M[i, j] := A[i, j] · [¬(i = j)] })
      (HashMap.ofList [("A", tl [2,2] [1,2, 3,4])])
      "M" (tl [2,2] [0,2, 3,0]))

end LeanNCD.Eval
