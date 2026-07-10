import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §15 — Adversarial: tricky-but-valid edge cases (should pass)

Legal-but-tricky corners: zero-pad boundary reads (EC1, EC2), size-1 axes (EC3, EC4),
diagonal read/trace/write (EC5, EC6, EC7), high-rank contraction (EC10, EC14), n-ary
products (EC11), affine reindex (EC12), precedence (EC13) and per-term contraction
scoping in a multi-term sum (EC15).

EC8 (`Out[2*i,2*j]:=X[i,j]` upsample) is `[✔]` — already covered by `EvalExamplesTest`.
EC9 (`Out[i+j]:=X[i,j]`) does NOT parse — LHS slots are single-axis affine only (`i+num`,
`2*i`); a two-axis-into-one-slot write is rejected (KG-reshape). Comment only.
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§15 — Adversarial edge cases" <|
-- EC1  boundary read → zero-pad.  X=[10,20,30]; shift 1 ⇒ output axis spans the valid-range
--   domain (len n+1=4): i=0 zero-pads (i−1 out of range), i≥1 reads X[i−1] ⇒ [0,10,20,30].
test "EC1 lookback-1"
    (evalEqB (tlprog!{ Y[i] := X[i - 1] })
      (HashMap.ofList [("X", tl [3] [10,20,30])])
      "Y" (tl [4] [0,10,20,30])) $

-- EC2  deep look-back (shift 3).  X=[1,2,3]; output len n+3=6: i<3 zero-pad, then X[i−3].
--   ⇒ [0,0,0,1,2,3].
test "EC2 lookback-3"
    (evalEqB (tlprog!{ Y[i] := X[i - 3] })
      (HashMap.ofList [("X", tl [3] [1,2,3])])
      "Y" (tl [6] [0,0,0,1,2,3])) $

-- EC3  size-1 contracted axis (rank-1 outer via matmul).  A 2×1, B 1×2.
--   A=[[1],[2]], B=[[3,4]] ⇒ Y[i,j]=A[i,0]·B[0,j] = [[3,4],[6,8]].
test "EC3 size1-contract"
    (evalEqB (tlprog!{ Y[i, j] := A[i, k] · B[k, j] })
      (HashMap.ofList [("A", tl [2,1] [1,2]), ("B", tl [1,2] [3,4])])
      "Y" (tl [2,2] [3,4, 6,8])) $

-- EC4  size-1 free axis i.  A 1×3, B 3.  A=[[1,2,3]], B=[1,1,1] ⇒ Y[0]=1+2+3=6 ⇒ [6].
test "EC4 size1-free"
    (evalEqB (tlprog!{ Y[i] := A[i, j] · B[j] })
      (HashMap.ofList [("A", tl [1,3] [1,2,3]), ("B", tl [3] [1,1,1])])
      "Y" (tl [1] [6])) $

-- EC5  diagonal read (repeated axis on RHS).  M=[[1,2],[3,4]] ⇒ d[i]=M[i,i]=[1,4].
test "EC5 diagonal-read"
    (evalEqB (tlprog!{ d[i] := M[i, i] })
      (HashMap.ofList [("M", tl [2,2] [1,2, 3,4])])
      "d" (tl [2] [1,4])) $

-- EC6  trace (diagonal read + full contraction).  M above ⇒ t=M[0,0]+M[1,1]=1+4=5.
test "EC6 trace"
    (evalEqB (tlprog!{ t[] := M[i, i] })
      (HashMap.ofList [("M", tl [2,2] [1,2, 3,4])])
      "t" (tl [] [5])) $

-- EC7  diagonal write (scatter reclassification; needs the `tensor D(i,j)` decl).
--   v=[5,6] ⇒ D[i,i]=v[i], off-diagonal 0 ⇒ [[5,0],[0,6]].
test "EC7 diagonal-write"
    (evalEqB (tlprog!{
    tensor D(i, j)
    D[i, i] := v[i]
  })
      (HashMap.ofList [("v", tl [2] [5,6])])
      "D" (tl [2,2] [5,0, 0,6])) $

-- EC8  [✔] upsample scatter `Out[2*i,2*j]:=X[i,j]` — already in EvalExamplesTest; not re-authored.

-- EC9  ~~`Out[i+j]:=X[i,j]`~~ does NOT parse — LHS slots are single-axis affine only
--   (`i+num`, `2*i`); `i+j` (two axes → one slot) is rejected. See KG-reshape (§14).

-- EC10  high-rank contraction (4 free + 1 contracted).  Y[i,j,k,l]=Σ_m A[i,j,m]·B[m,k,l].
--   A (i=2,j=1,m=2)=[[[1,2]],[[3,4]]], B (m=2,k=1,l=2)=[[[1,0]],[[0,1]]] ⇒
--     Y[i,0,0,l]=Σ_m A[i,0,m]B[m,0,l]:  Y[0]=[1,2], Y[1]=[3,4]  (B acts as identity on l)
--   ⇒ shape [2,1,1,2], data [1,2,3,4].
test "EC10 high-rank"
    (evalEqB (tlprog!{ Y[i, j, k, l] := A[i, j, m] · B[m, k, l] })
      (HashMap.ofList [("A", tl [2,1,2] [1,2, 3,4]), ("B", tl [2,1,2] [1,0, 0,1])])
      "Y" (tl [2,1,1,2] [1,2, 3,4])) $

-- EC11  n-ary product, all rank-0 scalar reads (doc shorthand `Z := A·B·C·D`).
--   A=2,B=3,C=4,D=5 ⇒ Z=2·3·4·5=120.
test "EC11 nary-product"
    (evalEqB (tlprog!{ Z[] := A[] · B[] · C[] · D[] })
      (HashMap.ofList [("A", tl [] [2]), ("B", tl [] [3]), ("C", tl [] [4]), ("D", tl [] [5])])
      "Z" (tl [] [120])) $

-- EC12  general integer-affine reindex  Y[i]:=X[2*i + 3*j - 1]  (j is RHS-only ⇒ summed).
--   The multi-term affine read leaves both loop axes unconstrained by X's shape alone, so both
--   are pinned (i,j:ℕ=2); out-of-range positions zero-pad.  X=[1,2,3,4,5,6]:
--     Y[i]=Σ_j X[2i+3j−1]:  Y[0]=X[−1]+X[2]=0+3=3;  Y[1]=X[1]+X[4]=2+5=7  ⇒  Y=[3,7].
test "EC12 affine-reindex"
    (evalEqB (tlprog!{
    axis i : ℕ = 2, j : ℕ = 2
    Y[i] := X[2 * i + 3 * j - 1]
  })
      (HashMap.ofList [("X", tl [6] [1,2,3,4,5,6])])
      "Y" (tl [2] [3, 7])) $

-- EC13  precedence: `·` binds tighter than `+`; two contractions summed, each contracted
--   independently over its OWN dummy axis.  (Parens are not surface syntax; the bare form
--   relies on precedence, which is exactly the property tested.)
--   `k` (size 2) and `l` (size 3) are DIFFERENT sizes, so the test also confirms the two
--   terms don't share a joint contraction range (per-term contraction scoping; see the
--   §12c callout in the portfolio doc and EC15).
--   A=[[1,2],[3,4]] (i,k=2×2), B=I₂ (k,j) ⇒ A·B = A = [[1,2],[3,4]].
--   C=ones(2,3) (i,l), D=ones(3,2) (l,j) ⇒ C·D[i,j]=Σ_l 1·1=3 (l size 3) ⇒ [[3,3],[3,3]].
--   Y=(A·B)+(C·D)=[[4,5],[6,7]].
test "EC13 precedence"
    (evalEqB (tlprog!{ Y[i, j] := A[i, k] · B[k, j] + C[i, l] · D[l, j] })
      (HashMap.ofList [("A", tl [2,2] [1,2, 3,4]), ("B", tl [2,2] [1,0, 0,1]),
    ("C", tl [2,3] [1,1,1, 1,1,1]), ("D", tl [3,2] [1,1, 1,1, 1,1])])
      "Y" (tl [2,2] [4,5, 6,7])) $

-- EC14  rank-5 full contraction (many contracted axes at once).
--   A shape [2,2,1,1,1] data [1,2,3,4] ⇒ s=Σ all = 10.
test "EC14 rank5-contract"
    (evalEqB (tlprog!{ s[] := A[i, j, k, l, m] })
      (HashMap.ofList [("A", tl [2,2,1,1,1] [1,2,3,4])])
      "s" (tl [] [10])) $

-- EC15  per-term contraction scoping in a multi-term sum.  Y[i]:=u[]·a[i] + W[i,k]·v[k].
--   `k` appears only in the second term, so it is contracted WITHIN that term alone; the
--   k-less first term `u·a[i]` is added in exactly once, unaffected by |k|:
--     Y[i] = u·a[i] + Σ_k W[i,k]·v[k].
--   u=10, a=[1,2], W=[[1,1],[1,1]], v=[1,1]:
--     Y[0]=10·1 + (1+1)=12;  Y[1]=10·2 + (1+1)=22  ⇒  Y=[12,22].
--   (Regression test: an earlier whole-equation contraction scoping summed every term over the
--   union of axes across the RHS, which silently broadcast k-less terms by |k|. See
--   `termAxisUIDs` in `Eval/Contract.lean`.)
--   NOTE: the doc's canonical form of this example names the second RHS tensor `c[k]`; that
--   spelling was renamed to `v[k]` here because `c[` is a GLOBALLY reserved token — Mathlib's
--   `Equiv.Perm` cycle notation (`Mathlib.GroupTheory.Perm.Cycle.Concrete`) declares `c[…]` as
--   permutation-cycle syntax, and this project transitively imports mathlib, so any tensor
--   named exactly `c` followed by `[` fails to parse project-wide. Not a DSL bug; `cc`, `c2`,
--   and uppercase `C` are all unaffected. See the "Naming hazard" note in the portfolio doc.
test "EC15 per-term-contraction"
    (evalEqB (tlprog!{ Y[i] := u[] · a[i] + W[i, k] · v[k] })
      (HashMap.ofList [("a", tl [2] [1,2]), ("u", tl [] [10]),
    ("W", tl [2,2] [1,1, 1,1]), ("v", tl [2] [1,1])])
      "Y" (tl [2] [12, 22]))

end LeanNCD.Eval
