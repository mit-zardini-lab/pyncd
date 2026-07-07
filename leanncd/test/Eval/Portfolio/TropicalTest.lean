import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §11 — Tropical / max-times semiring

`maxreduce` combined with products for best-path (max-times) reachability.
(TR1/TR2/TR4 basic max-reductions are already covered elsewhere; not re-authored here.)
TR5 (min-plus shortest path) needs a `min` aggregation and additive combine — a known gap
(KG-min) with no expressible program — so it is documented as a comment only.
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§11 — Tropical / max-times semiring" <|
-- TR3  max-times "best path" 2-hop reliability: R[i,k]=max_j P[i,j]·P[j,k], P=[[0,0.9],[0.8,0]].
--   R[0,0]=max(0·0, 0.9·0.8)=0.72,  R[0,1]=max(0·0.9, 0.9·0)=0,
--   R[1,0]=max(0.8·0, 0·0.8)=0,     R[1,1]=max(0.8·0.9, 0·0)=0.72.
test "TR3 best-path"
    (evalEqB (tlprog!{ R[i, k] := maxreduce(P[i, j] · P[j, k]) })
      (HashMap.ofList [("P", tl [2,2] [0,0.9, 0.8,0])])
      "R" (tl [2,2] [0.72,0, 0,0.72]))

-- TR5  min-plus shortest path (Viterbi/Bellman step) — [F] gap KG-min: no `min` agg, no `+`
--       combine (min-plus semiring). No expressible program; documented, not authored.

end LeanNCD.Eval
