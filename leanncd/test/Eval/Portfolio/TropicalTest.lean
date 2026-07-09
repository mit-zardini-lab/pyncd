import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §11 — Tropical / max-times & min-times semirings

`maxreduce`/`minreduce` combined with products for best/worst-path reachability.
(TR1/TR2/TR4 basic max-reductions are already covered elsewhere; not re-authored here.)
`minreduce` is the tropical min-times contraction `(×, min, +∞)` — the min analog of `maxreduce`;
added 2026-07-08 to close the min-*aggregation* half of KG-min. TR5 (min-*plus* shortest path)
still needs an additive within-term combine (`+` instead of `×`), which `minreduce` does not
provide — that half of KG-min remains open.
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§11 — Tropical / max-times & min-times semirings" <|
-- TR3  max-times "best path" 2-hop reliability: R[i,k]=max_j P[i,j]·P[j,k], P=[[0,0.9],[0.8,0]].
--   R[0,0]=max(0·0, 0.9·0.8)=0.72,  R[0,1]=max(0·0.9, 0.9·0)=0,
--   R[1,0]=max(0.8·0, 0·0.8)=0,     R[1,1]=max(0.8·0.9, 0·0)=0.72.
test "TR3 best-path"
    (evalEqB (tlprog!{ R[i, k] := maxreduce(P[i, j] · P[j, k]) })
      (HashMap.ofList [("P", tl [2,2] [0,0.9, 0.8,0])])
      "R" (tl [2,2] [0.72,0, 0,0.72])) $

-- TR6  basic min-reduction (min analog of TR1): C[i] = min_k A[i,k].
--   A=[[3,1,4],[1,5,9]] ⇒ C=[min(3,1,4), min(1,5,9)] = [1,1].
test "TR6 min-reduction"
    (evalEqB (tlprog!{ C[i] := minreduce(A[i, k]) })
      (HashMap.ofList [("A", tl [2,3] [3,1,4, 1,5,9])])
      "C" (tl [2] [1,1])) $

-- TR7  min-times combine (min analog of TR2): C[i] = min_k A[i,k]·B[k].
--   A=[[3,1,4],[2,2,2]], B=[2,5,1] ⇒ products row0=[6,5,4]→4, row1=[4,10,2]→2 ⇒ C=[4,2].
test "TR7 min-times"
    (evalEqB (tlprog!{ C[i] := minreduce(A[i, k] · B[k]) })
      (HashMap.ofList [("A", tl [2,3] [3,1,4, 2,2,2]), ("B", tl [3] [2,5,1])])
      "C" (tl [2] [4,2]))

-- TR5  min-plus shortest path (Viterbi/Bellman step) — still an [F] gap: `minreduce` supplies the
--       `min` aggregation, but min-plus needs `+` as the within-term combine (no expressible
--       program yet). Documented, not authored.

end LeanNCD.Eval
