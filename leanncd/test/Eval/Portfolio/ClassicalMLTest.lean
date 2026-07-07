import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §12c — Classical ML, probabilistic & RL

Numeric cross-checks for pairwise distance (CM1), factorization machines (CM2),
value iteration / Bellman backup (CM3), power iteration / HMM-forward (CM4),
matrix factorization (CM5) and VAE reparameterization (CM6).

Several are **multi-statement by necessity** (equation-level summation, see the callout
after §12c): a single RHS sum can't mix terms with different contracted axes, so each
contraction is materialized into its own intermediate whose index set matches the LHS,
then the intermediates (all sharing the same free axes) are combined.

CM7 (argmin / nearest-centroid), CM8 (linear solve / inverse) and CM9 (sigmoid/tanh
activations) are `[F]` gaps — no argmin, no solve, only relu — so they are comments only.
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§12c — Classical ML, probabilistic & RL" <|
-- CM1  pairwise squared distance ‖xᵢ−xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² − 2 xᵢ·xⱼ.
--   X=[[0,0],[3,4]], one=[1,1], m2=−2.
--   sq[i]=Σ_d X[i,d]² ⇒ sq=[0,25].  cross[i,j]=Σ_d X[i,d]X[j,d] ⇒ [[0,0],[0,25]].
--   D[i,j]=sq[i]·one[j] + one[i]·sq[j] + m2·cross[i,j]  (i,j both free ⇒ no summation):
--     D[0,0]=0+0−0=0;  D[0,1]=0+25−0=25;  D[1,0]=25+0−0=25;  D[1,1]=25+25−2·25=0.
test "CM1 pairwise-sqdist"
    (evalEqB (tlprog!{
    sq[i] := X[i, d] · X[i, d]
    cross[i, j] := X[i, d] · X[j, d]
    D[i, j] := sq[i] · one[j] + one[i] · sq[j] + m2[] · cross[i, j]
  })
      (HashMap.ofList [("X", tl [2,2] [0,0, 3,4]), ("one", tl [2] [1,1]), ("m2", tl [] [-2])])
      "D" (tl [2,2] [0,25, 25,0])) $

-- CM2  factorization machine 2nd order = ½((Σ)² − Σ()²).
--   V=[[1],[2]] (i=2,f=1), x=[1,1], hp=0.5, hm=−0.5.
--   sqsum[f]=Σ_{i,j} V[i,f]x[i]V[j,f]x[j]=(Σ_i V[i,f]x[i])²=(1+2)²=9   (fresh dummy j).
--   sumsq[f]=Σ_i V[i,f]x[i]V[i,f]x[i]=Σ_i(V[i,f]x[i])²=1²+2²=5         (reused dummy i).
--   fm[f]=hp·sqsum+hm·sumsq=0.5·9−0.5·5=4.5−2.5=2.
test "CM2 factorization-machine"
    (evalEqB (tlprog!{
    sqsum[f] := V[i, f] · x[i] · V[j, f] · x[j]
    sumsq[f] := V[i, f] · x[i] · V[i, f] · x[i]
    fm[f] := hp[] · sqsum[f] + hm[] · sumsq[f]
  })
      (HashMap.ofList [("V", tl [2,1] [1,2]), ("x", tl [2] [1,1]), ("hp", tl [] [0.5]), ("hm", tl [] [-0.5])])
      "fm" (tl [1] [2])) $

-- CM3  value iteration / Bellman backup.  R=I₂, γ=0.9, P=½ uniform (2×2×2), V=[1,1].
--   EV[s,a]=Σ_s2 γ·P[s,a,s2]·V[s2]=0.9·(0.5·1+0.5·1)=0.9  (all entries).
--   Q[s,a]=R[s,a]+EV[s,a]=[[1.9,0.9],[0.9,1.9]]  (s,a free ⇒ no summation).
--   Vn[s]=max_a Q[s,a]=[1.9,1.9].
test "CM3 value-iteration"
    (evalEqB (tlprog!{
    EV[s, a] := gamma[] · P[s, a, s2] · V[s2]
    Q[s, a] := R[s, a] + EV[s, a]
    Vn[s] := maxreduce(Q[s, a])
  })
      (HashMap.ofList [("R", tl [2,2] [1,0, 0,1]), ("gamma", tl [] [0.9]),
    ("P", tl [2,2,2] [0.5,0.5, 0.5,0.5, 0.5,0.5, 0.5,0.5]), ("V", tl [2] [1,1])])
      "Vn" (tl [2] [1.9, 1.9])) $

-- CM4  power iteration / Markov / HMM-forward (contraction INSIDE a scan).
--   p0=[1,0], M=swap=[[0,1],[1,0]].  Step: p[j,l+1]=Σ_i p[i,l]·M[i,j].
--     l=0: [1,0];  l=1: [0,1];  l=2: [1,0]  ⇒  p (shape [j=2, l=3]) = [[1,0,1],[0,1,0]].
test "CM4 power-iteration"
    (evalEqB (tlprog!{
    axis l : ℕ = 3
    p[j, 0] := p0[j]
    p[j, l +1] := p[i, l] · M[i, j]
  })
      (HashMap.ofList [("p0", tl [2] [1,0]), ("M", tl [2,2] [0,1, 1,0])])
      "p" (tl [2,3] [1,0,1, 0,1,0])) $

-- CM5  matrix factorization (recommender dot-product model)  Rhat[u,i]=Σ_f P[u,f]·Q[i,f].
--   P=[[1,2]] (u=1,f=2), Q=[[3,4],[5,6]] (i=2,f=2).
--     Rhat[0,0]=1·3+2·4=11;  Rhat[0,1]=1·5+2·6=17  ⇒  Rhat=[[11,17]].
test "CM5 matrix-factorization"
    (evalEqB (tlprog!{ Rhat[u, i] := P[u, f] · Q[i, f] })
      (HashMap.ofList [("P", tl [1,2] [1,2]), ("Q", tl [2,2] [3,4, 5,6])])
      "Rhat" (tl [1,2] [11, 17])) $

-- CM6  VAE reparameterization  z[i]=μ[i]+σ[i]·ε[i]  (Hadamard σ⊙ε then add; i free ⇒ no sum).
--   mu=[1,2], sigma=[0.5,2], eps=[2,1]:  z[0]=1+0.5·2=2;  z[1]=2+2·1=4  ⇒  z=[2,4].
test "CM6 vae-reparam"
    (evalEqB (tlprog!{ z[i] := mu[i] + sigma[i] · eps[i] })
      (HashMap.ofList [("mu", tl [2] [1,2]), ("sigma", tl [2] [0.5,2]), ("eps", tl [2] [2,1])])
      "z" (tl [2] [2, 4]))

-- CM7  [F] k-means assignment / nearest-centroid — no `argmin` (only max VALUE, not its index). (KG-min)
-- CM8  [F] closed-form linear regression β=(XᵀX)⁻¹Xᵀy — no linear solve / inverse. (KG-solve)
-- CM9  [F] logistic regression / GRU-LSTM gates — only `relu`; no `σ`/`tanh`. (KG-activation)

end LeanNCD.Eval
