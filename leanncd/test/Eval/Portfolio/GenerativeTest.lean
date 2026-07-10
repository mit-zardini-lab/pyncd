import Eval.Portfolio.Harness
import LSpec
/-!
# Portfolio §12b — Advanced & generative domains

Diffusion (DF*), mixture-of-experts (ME*), state-space models (SS*), positional encodings
(PE*), and contrastive/metric learning (CL*). Only the **expressible** (confirmed) entries
are authored; the transcendental / gather / L2 / log gaps are `[F]` and left as comments.
-/
namespace LeanNCD.Eval
open Std LSpec

#lspec group "§12b — Advanced & generative domains" <|
/- ### Diffusion -/

-- DF1  forward process Xt[i] := α·x0[i] + β·eps[i]. α=0.6, β=0.8, x0=[1,0], ε=[0,1]:
--   Xt[0]=0.6·1+0.8·0=0.6, Xt[1]=0.6·0+0.8·1=0.8.
test "DF1 forward-process"
    (evalEqB (tlprog!{ Xt[i] := alpha[] · x0[i] + beta[] · eps[i] })
      (HashMap.ofList [("alpha", tl [] [0.6]), ("beta", tl [] [0.8]),
                   ("x0", tl [2] [1,0]), ("eps", tl [2] [0,1])])
      "Xt" (tl [2] [0.6,0.8])) $

-- DF2  DDPM posterior mean Xp[i] := c1·Xt[i] + c2·e[i] with c2<0.
-- c1=0.5, c2=−0.5, Xt=[2,4], e=[1,1]:
--   Xp[0]=0.5·2 + (−0.5)·1 = 1.0−0.5 = 0.5
--   Xp[1]=0.5·4 + (−0.5)·1 = 2.0−0.5 = 1.5
test "DF2 ddpm-posterior"
    (evalEqB (tlprog!{ Xp[i] := c1[] · Xt[i] + c2[] · e[i] })
      (HashMap.ofList [("c1", tl [] [0.5]), ("c2", tl [] [-0.5]),
                   ("Xt", tl [2] [2,4]), ("e", tl [2] [1,1])])
      "Xp" (tl [2] [0.5,1.5])) $

-- DF3  denoising loss sse := r·r with r[i] := eps[i] + (−1)·ehat[i] (the ST5 subtraction trick).
-- eps=[0.5,0.2], ehat=[0.1,0.4], m1=−1:
--   r=[0.5−0.1, 0.2−0.4]=[0.4, −0.2]; sse = 0.4² + (−0.2)² = 0.16 + 0.04 = 0.20.
test "DF3 denoising-loss"
    (evalEqB (tlprog!{
    r[i] := eps[i] + m1[] · ehat[i]
    sse[] := r[i] · r[i]
  })
      (HashMap.ofList [("eps", tl [2] [0.5,0.2]), ("ehat", tl [2] [0.1,0.4]), ("m1", tl [] [-1])])
      "sse" (tl [] [0.20])) $

-- DF4  sinusoidal timestep embedding `te[2*i] := sin(Arg[i])` with `Arg[i] := t[]·omega_pow[i]`.
--   `ω^i` itself is NOT computable inside the DSL (KG-idxvalue: index arithmetic yields
--   booleans only) — it must be supplied as a precomputed input tensor. Given that, this is an
--   ordinary materialized product (like DF1) feeding an inline `sin(...)` factor into an
--   upsample-style affine scatter (EC8-style; odd slots zero-default).
--   t=1, omega_pow=[ω^0,ω^1]=[1,0.1] ⇒ Arg=[1,0.1] ⇒ te=[sin 1, 0, sin 0.1, 0]
--   ≈ [0.8414710, 0, 0.0998334, 0].
test "DF4 sinusoidal-embedding"
    (evalEqB (tlprog!{
    axis i : ℕ = 2
    Arg[i] := t[] · omega_pow[i]
    te[2*i] := sin(Arg[i])
  })
      (HashMap.ofList [("t", tl [] [1]), ("omega_pow", tl [2] [1,0.1])])
      "te" (tl [4] [0.8414709848078965, 0, 0.09983341664682815, 0])) $

/- ### Mixture of Experts -/

-- ME1  gating g[t,e.] := softmax(X·Wg) — property: each router row sums to 1.
test "ME1 gating"
    (evalPredB (tlprog!{
    tensor g(t, e)
    g[t, e.] := softmax(X[t, d] · Wg[d, e])
  })
      (HashMap.ofList [("X", tl [1,2] [1,0]), ("Wg", tl [2,2] [1,0,0,1])])
      "g" rowsSumToOne) $

-- ME2  batched expert MLP Y[t,e,f] := relu(We[e,f,d]·X[t,d]). Expert axis `e` is free.
-- We (2×1×2) = [[[1,0]],[[0,−1]]], X=[[2,3]] (t=1):
--   Y[0,0,0]=relu(1·2+0·3)=relu(2)=2
--   Y[0,1,0]=relu(0·2+(−1)·3)=relu(−3)=0   (relu clamps the negative expert)
test "ME2 expert-mlp"
    (evalEqB (tlprog!{ Y[t, e, f] := relu(We[e, f, d] · X[t, d]) })
      (HashMap.ofList [("We", tl [2,1,2] [1,0,0,-1]), ("X", tl [1,2] [2,3])])
      "Y" (tl [1,2,1] [2,0])) $

-- ME3  combine Out[t,f] := g[t,e]·Y[t,e,f], chained on ME1+ME2 — property: shape [1,1].
test "ME3 combine"
    (evalShapeB (tlprog!{
    tensor g(t, e)
    g[t, e.] := softmax(X[t, d] · Wg[d, e])
    Y[t, e, f] := relu(We[e, f, d] · X[t, d])
    Out[t, f] := g[t, e] · Y[t, e, f]
  })
      (HashMap.ofList [("X", tl [1,2] [1,0]), ("Wg", tl [2,2] [1,0,0,1]),
                   ("We", tl [2,1,2] [1,0,0,1])])
      "Out" [1,1]) $

-- ME4  top-k routing — [F] KG-gather (no argmax / top-k gather). Not authored.

/- ### State-space / sequence models -/

-- SS1  linear recurrence h[j,l+1] := A[j,k]·h[k,l] + B[j]·u[l], reading input at CURRENT index l.
-- A=[[1]], B=[1], h0=[1], u=[1,1,1] (k size 1 ⇒ no equation-level broadcast); axis l = 3:
--   h[0]=1, h[1]=1·1+1·u[0]=2, h[2]=1·2+1·u[1]=3 ⇒ h=[1,2,3].
test "SS1 linear-recurrence"
    (evalEqB (tlprog!{
    axis l : ℕ = 3
    h[j, 0]    := h0[j]
    h[j, l +1] := A[j, k] · h[k, l] + B[j] · u[l]
  })
      (HashMap.ofList [("h0", tl [1] [1]), ("A", tl [1,1] [1]), ("B", tl [1] [1]),
                   ("u", tl [3] [1,1,1])])
      "h" (tl [1,3] [1,2,3])) $

-- SS2  output map y[j,l] := C[j,k]·h[k,l] off the scan state. An `l`-indexed statement written
-- inside the SS1 program is swallowed as a per-step scan intermediate (a scan's outputs are only
-- names written by BOTH base and recur — see Lowering.ScanStmt.outputs), so it is authored
-- standalone consuming the MATERIALIZED scan state h=[1,2,3] (l is then an ordinary free axis).
-- C=[[1]] ⇒ y = h = [1,2,3].
test "SS2 output-map"
    (evalEqB (tlprog!{ y[j, l] := C[j, k] · h[k, l] })
      (HashMap.ofList [("h", tl [1,3] [1,2,3]), ("C", tl [1,1] [1])])
      "y" (tl [1,3] [1,2,3])) $

-- SS3  diagonal SSM h[j,l+1] := a[j]·h[j,l] + B[j]·u[l] (elementwise, no contraction).
-- a=[2, 0.5], B=[1,1], h0=[1,1], u=[1,1,1]; axis l = 3:
--   j=0: 1, 2·1+1=3, 2·3+1=7       ⇒ [1,3,7]
--   j=1: 1, 0.5·1+1=1.5, 0.5·1.5+1=1.75 ⇒ [1,1.5,1.75]
test "SS3 diagonal-ssm"
    (evalEqB (tlprog!{
    axis l : ℕ = 3
    h[j, 0]    := h0[j]
    h[j, l +1] := a[j] · h[j, l] + B[j] · u[l]
  })
      (HashMap.ofList [("h0", tl [2] [1,1]), ("a", tl [2] [2,0.5]), ("B", tl [2] [1,1]),
                   ("u", tl [3] [1,1,1])])
      "h" (tl [2,3] [1,3,7,1,1.5,1.75])) $

-- SS4  input at advancing index (…+ B[j]·u[l+1]) — [R] causalityViolation, covered in RejectTest.

/- ### Positional encodings -/

-- PE1  learned additive Out[p,d] := X[p,d] + PE[p,d]. X=[[1,2],[3,4]], PE=[[10,20],[30,40]]:
--   ⇒ [[11,22],[33,44]].
test "PE1 additive-pe"
    (evalEqB (tlprog!{ Out[p, d] := X[p, d] + PE[p, d] })
      (HashMap.ofList [("X", tl [2,2] [1,2,3,4]), ("PE", tl [2,2] [10,20,30,40])])
      "Out" (tl [2,2] [11,22,33,44])) $

-- PE2  sinusoidal pe[p,2i] := sin(…) — [F] KG-trig. Not authored.
-- PE3  RoPE rotation — [F] KG-trig (sin/cos + paired rotation). Not authored.

/- ### Contrastive / metric learning -/

-- CL1  similarity matrix S[i,j] := Z1[i,d]·Z2[j,d]. Z1=Z2=I₂ ⇒ identity.
test "CL1 similarity"
    (evalEqB (tlprog!{ S[i, j] := Z1[i, d] · Z2[j, d] })
      (HashMap.ofList [("Z1", tl [2,2] [1,0,0,1]), ("Z2", tl [2,2] [1,0,0,1])])
      "S" (tl [2,2] [1,0,0,1])) $

-- CL2  InfoNCE softmax P[i,j.] := softmax(S[i,j]), chained on CL1 — property: rows sum to 1.
test "CL2 infonce-softmax"
    (evalPredB (tlprog!{
    S[i, j] := Z1[i, d] · Z2[j, d]
    tensor P(i, j)
    P[i, j.] := softmax(S[i, j])
  })
      (HashMap.ofList [("Z1", tl [2,2] [1,0,0,1]), ("Z2", tl [2,2] [1,0,0,1])])
      "P" rowsSumToOne) $

-- CL3  cosine similarity via L2-normalize-then-dot-product (KG-l2norm).
--   Z1=[[3,4]] (‖·‖₂=5) → normalized [0.6,0.8]; Z2=[[1,0]] (already unit) → unchanged.
--   S[i,j] = Z1n·Z2n = 0.6·1+0.8·0 = 0.6.
test "CL3 cosine-similarity"
    (evalEqB (tlprog!{
    Z1n[i, d.] := l2normalize(Z1[i, d])
    Z2n[j, d.] := l2normalize(Z2[j, d])
    S[i, j] := Z1n[i, d] · Z2n[j, d]
  })
      (HashMap.ofList [("Z1", tl [1,2] [3,4]), ("Z2", tl [1,2] [1,0])])
      "S" (tl [1,1] [0.6])) $

-- CL3b  all-zero row L2-normalizes silently to zero (‖x‖₂=0), matching `normalize`/`softmax`'s
--   existing convention (SC8's precedent) — not a domain error, unlike `log`/`sqrt`/`recip`.
test "CL3b l2normalize-zero-row"
    (evalEqB (tlprog!{ Y[i, d.] := l2normalize(X[i, d]) })
      (HashMap.ofList [("X", tl [1,2] [0,0])])
      "Y" (tl [1,2] [0,0])) $

-- CL4  InfoNCE loss `L[] := m1[]·log(P[i, i])` (diagonal read, EC5-style, wrapped in `log`;
--   ST5's `−1` trick for the leading minus). P given directly (not chained through CL1/CL2's
--   softmax) to keep the hand-computed ground truth simple: P=[[0.5,0.5],[0.25,0.75]], diagonal
--   = [0.5, 0.75]. L = −(ln 0.5 + ln 0.75) = −(−0.6931472 + −0.2876821) = 0.9808293.
test "CL4 infonce-loss"
    (evalEqB (tlprog!{ L[] := m1[] · log(P[i, i]) })
      (HashMap.ofList [("P", tl [2,2] [0.5,0.5, 0.25,0.75]), ("m1", tl [] [-1])])
      "L" (tl [] [0.9808292530117262]))

end LeanNCD.Eval
