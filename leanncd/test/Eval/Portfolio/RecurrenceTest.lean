import Eval.Portfolio.Harness
/-!
# Portfolio §7 — Recurrence & scans

Scans, cumulative sums, and the two confirmed silent-wrong scan gaps.
(RC1 coupled scan is already covered by `Eval.EvalExamplesTest`; not re-authored here.)

* RC2/RC3 — numeric `[N]`.
* RC4 — reject `[R]`: a scan step reading an external at the advancing index `l + 1` is
  rejected with `CompileError.causalityViolation` (same shape as SS4 in `RejectTest`).
* RC5/RC6 — silent-wrong gaps `[F]`: these pin the *current* (wrong) evaluator output so the
  test flips red when the gap is fixed. Flagged KG-scanagg (RC5) and KG-2dscan (RC6).
-/
namespace LeanNCD.Eval
open Std

-- RC2  simple RNN: single scan, self-recurrence. 1 feature, W = 1, X0 = 1.
--   S₀ = 1; S₁ = relu(1·1) = 1; S₂ = relu(1·1) = 1  ⇒  S = [1,1,1].
run_cmd (assertEval "RC2 rnn"
  (tlprog!{ axis l : ℕ = 3
            S[j, 0]    := X0[j]
            S[j, l +1] := relu(S[j, l] · W[j, k]) })
  (HashMap.ofList [("X0", tl [1] [1]), ("W", tl [1,1] [1])])
  "S" (tl [1,3] [1,1,1]))

-- RC3  prefix-sum via a triangular Iverson mask (cumulative sum without a scan).
--   C[i] = Σⱼ X[j]·[j ≤ i];  X = [1,2,3]  ⇒  C = [1, 1+2, 1+2+3] = [1,3,6].
run_cmd (assertEval "RC3 prefix-sum"
  (tlprog!{ axis i : ℕ = 3, j : ℕ = 3
            C[i] := X[j] · [j ≤ i] })
  (HashMap.ofList [("X", tl [3] [1,2,3])])
  "C" (tl [3] [1,3,6]))

-- RC4  REJECT: a scan step reading an external at the advancing index l+1 ⇒ causalityViolation
--   (the compiler treats any next-index read as a future dependency). Asserted exactly like SS4.
run_cmd do
  match TLProgram.compile (tlprog!{ axis l : ℕ = 3
                                    S[j, 0]    := X[j, 0]
                                    S[j, l +1] := S[j, l] + X[j, l + 1] }) |>.run 0 with
  | .error (.causalityViolation "S") _ => pure ()
  | .error e _ => throwError s!"RC4: wrong CompileError: {repr e}"
  | .ok _ _    => throwError "RC4: expected causalityViolation, compile succeeded"

-- RC5  KNOWN GAP (KG-scanagg): `maxreduce` inside a scan step is SILENTLY summed (agg dropped).
--   X0 = 2, W = [1,3].  Correct max-semantics: Mₗ₊₁ = max(Mₗ·1, Mₗ·3) = 3·Mₗ ⇒ [2,6,18].
--   Current (wrong) output: the step SUMS ⇒ Mₗ₊₁ = Mₗ·(1+3) = 4·Mₗ ⇒ [2,8,32]. Pinning the
--   wrong value; flip to [2,6,18] when KG-scanagg is fixed.
run_cmd (assertEval "RC5 maxreduce-in-scan (KG-scanagg, current wrong)"
  (tlprog!{ axis l : ℕ = 3
            M[j, 0]    := X0[j]
            M[j, l +1] := maxreduce(M[j, l] · W[j, k]) })
  (HashMap.ofList [("X0", tl [1] [2]), ("W", tl [1,2] [1,3])])
  "M" (tl [1,3] [2,8,32]))

-- RC6  KNOWN GAP (KG-2dscan): a 2-D / nested recurrence collapses to a 1-D scan over one axis;
--   the base case on the other axis is overwritten. Base G = 0, A = ones(2×2).
--   Correct 2-D DP (G[r+1,c+1] = G[r,c] + A[r,c]) is [[0,0],[0,1]]. Current (wrong) output is
--   [[0,1],[0,1]] (collapses to a 1-D scan). Pinning the wrong value; flip when KG-2dscan is fixed.
run_cmd (assertEval "RC6 2d-scan (KG-2dscan, current wrong)"
  (tlprog!{ axis r : ℕ = 2, c : ℕ = 2
            G[r, 0]       := Z[r]
            G[r +1, c +1] := G[r, c] + A[r, c] })
  (HashMap.ofList [("Z", tl [2] [0,0]), ("A", tl [2,2] [1,1,1,1])])
  "G" (tl [2,2] [0,1, 0,1]))

end LeanNCD.Eval
