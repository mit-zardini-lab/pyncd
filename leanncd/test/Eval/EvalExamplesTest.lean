import LeanNCD.Eval.Eval
/-!
# End-to-end evaluation examples (Milestone I integration test)

Eleven tensor-logic programs are parse-compiled (`tlprog!{…}`), run on concrete `Float`
input tensors via `TLProgram.eval`, and asserted against hand-computed numbers
(`DenseTensor.approxEq`, or exact equality for placement). A failure here means an
`Eval/` evaluator module has a bug — this doubles as the integration test for the whole
`Eval/` stack (Shape/Gather/Contract/Nonlin/Scatter/Scan/Eval).

Coverage: the five §12.1 examples (matmul, masked attention, strided conv, upsample,
coupled scan) + the two predicate examples (masked aggregation, band mask) + four extra
examples (look-back, outer product, contraction+relu, normalize).
-/
namespace LeanNCD.Eval
open Std

private def tensorOf (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩

/- 1. Matmul `Y[i,j] := W[i,k]·X[k,j]` — W (2×3), X (3×2) ⇒ known 2×2. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "W" (tensorOf [2,3] [1,2,3, 4,5,6])).insert "X" (tensorOf [3,2] [1,0, 0,1, 1,1])
  match TLProgram.eval (tlprog!{ Y[i,j] := W[i,k] · X[k,j] }) env with
  | .error e => throwError s!"matmul: {e}"
  | .ok out => match out["Y"]? with
    | some Y => unless DenseTensor.approxEq Y (tensorOf [2,2] [4,5, 10,11]) do
        throwError s!"matmul wrong: {repr Y.data}"
    | none => throwError "matmul: no Y"

/- 2. Masked (causal) attention `A[q,s] := softmax(where s ≤ q)(Q[q,d]·K[s,d])`.
    Q = K = I₂. Row q=0: only s=0 unmasked ⇒ A[0]=[1,0]. Row q=1: scores [0,1] ⇒
    softmax = [e⁰, e¹]/(e⁰+e¹) ≈ [0.2689, 0.7311]. Asserts each q-row sums to 1 over
    unmasked s and masked (s>q) entries are exactly 0. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "Q" (tensorOf [2,2] [1,0, 0,1])).insert "K" (tensorOf [2,2] [1,0, 0,1])
  match TLProgram.eval (tlprog!{
    tensor A : (q : ℝ, s : norm)
    A[q, s] := softmax(where s ≤ q)(Q[q, d] · K[s, d])
  }) env with
  | .error e => throwError s!"attn: {e}"
  | .ok out => match out["A"]? with
    | some A =>
        -- masked entry (q=0, s=1, s>q) is exactly 0
        unless A.get! [0,1] == 0.0 do throwError s!"attn: masked entry ≠ 0: {repr A.data}"
        -- each q-row sums to 1 over unmasked s
        let row0 := A.get! [0,0] + A.get! [0,1]
        let row1 := A.get! [1,0] + A.get! [1,1]
        unless Float.abs (row0 - 1.0) < 1e-6 && Float.abs (row1 - 1.0) < 1e-6 do
          throwError s!"attn: row sums ≠ 1: {repr A.data}"
        -- numeric check of the unmasked second row
        unless DenseTensor.approxEq A (tensorOf [2,2] [1.0, 0.0, 0.2689414213699951, 0.7310585786300049]) do
          throwError s!"attn wrong: {repr A.data}"
    | none => throwError "attn: no A"

/- 3. Strided convolution `Y[i,j] := W[p,r]·X[i+p, 2*j+r]`. W = I₂ (kernel), X (3×5).
    With W identity, `Y[i,j] = X[i,2j] + X[i+1,2j+1]`. Output 2×2 (valid-conv extents,
    inferred from the affine read positions): [[6,10],[16,20]]. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "W" (tensorOf [2,2] [1,0, 0,1])).insert "X"
      (tensorOf [3,5] [0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14])
  match TLProgram.eval (tlprog!{ Y[i,j] := W[p,r] · X[i + p, 2 * j + r] }) env with
  | .error e => throwError s!"conv: {e}"
  | .ok out => match out["Y"]? with
    | some Y => unless DenseTensor.approxEq Y (tensorOf [2,2] [6,10, 16,20]) do
        throwError s!"conv wrong: shape={repr Y.shape} {repr Y.data}"
    | none => throwError "conv: no Y"

/- 4. Upsample 2× `Out[2*i, 2*j] := X[i,j]` (affine scatter write). X (2×2) ⇒ 4×4 with
    the input values at the even coordinates and 0 elsewhere. Exact placement. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (tensorOf [2,2] [1,2, 3,4])).insert "_L" (tensorOf [1] [0])
  match TLProgram.eval (tlprog!{
    tensor Out : (i : ℝ[2 * m], j : ℝ[2 * n])
    Out[2 * i, 2 * j] := X[i, j]
  }) env with
  | .error e => throwError s!"upsample: {e}"
  | .ok out => match out["Out"]? with
    | some Y =>
        unless DenseTensor.approxEq Y (tensorOf [4,4] [1,0,2,0, 0,0,0,0, 3,0,4,0, 0,0,0,0]) do
          throwError s!"upsample wrong: shape={repr Y.shape} {repr Y.data}"
    | none => throwError "upsample: no Out"

/- 5. Coupled scan (G, H share the iteration axis `l`). All weights = 1, one feature.
    The iteration count L is supplied as the time-extent of the `G` state buffer passed
    in `inputs` (L is a runtime parameter, not present in the program text; sized here
    via the input shape `[1,L]`). Steps: G₀=1,H₀=2; G₁=relu(1+2)=3,H₁=relu(2+1)=3;
    G₂=relu(3+3)=6,H₂=6 ⇒ G=[1,3,6], H=[2,3,6]. Asserts the first two steps. -/
run_cmd do
  let e0 : HashMap String DenseTensor := {}
  let env := ((((((e0.insert "X" (tensorOf [1] [1.0])).insert "Y" (tensorOf [1] [2.0])).insert "W_G"
      (tensorOf [1,1] [1.0])).insert "U" (tensorOf [1,1] [1.0])).insert "W_H"
      (tensorOf [1,1] [1.0])).insert "V" (tensorOf [1,1] [1.0])).insert "G" (tensorOf [1,3] [0,0,0])
  match TLProgram.eval (tlprog!{
    G[j, 0]    := X[j]
    G[j, l +1] := relu(G[j, l] · W_G[j, k] + H[j, l] · U[j, k])
    H[j, 0]    := Y[j]
    H[j, l +1] := relu(H[j, l] · W_H[j, k] + G[j, l] · V[j, k])
  }) env with
  | .error e => throwError s!"scan: {e}"
  | .ok out => match out["G"]?, out["H"]? with
    | some G, some H =>
        unless DenseTensor.approxEq G (tensorOf [1,3] [1,3,6]) do
          throwError s!"scan G wrong: {repr G.data}"
        unless DenseTensor.approxEq H (tensorOf [1,3] [2,3,6]) do
          throwError s!"scan H wrong: {repr H.data}"
    | _, _ => throwError "scan: no G/H"

/- 6. Masked aggregation over a predicate `Result[] := F[t,i]·F[t,j]·edge[i,j]`
    (every index contracted ⇒ scalar). edge[0,1]=edge[1,0]=1 (others 0), F (2×2).
    Result = Σ_t 2·F[t,0]·F[t,1] = 2(1·2) + 2(3·4) = 4 + 24 = 28. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "F" (tensorOf [2,2] [1,2, 3,4])).insert "edge" (tensorOf [2,2] [0,1, 1,0])
  match TLProgram.eval (tlprog!{
    predicate edge : (i : ℕ, j : ℕ)
    Result[] := F[t, i] · F[t, j] · edge[i, j]
  }) env with
  | .error e => throwError s!"maskedAgg: {e}"
  | .ok out => match out["Result"]? with
    | some R => unless DenseTensor.approxEq R (tensorOf [] [28]) do
        throwError s!"maskedAgg wrong: shape={repr R.shape} {repr R.data}"
    | none => throwError "maskedAgg: no Result"

/- 7. Band (Iverson) mask `Band[i,j] := A[i,j]·[|i - j| ≤ 1]` — tridiagonal mask on A (3×3):
    zeros out the [0,2] and [2,0] corners, keeps the rest. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "A" (tensorOf [3,3] [1,2,3, 4,5,6, 7,8,9])).insert "_L" (tensorOf [1] [0])
  match TLProgram.eval (tlprog!{ Band[i, j] := A[i, j] · [|i - j| ≤ 1] }) env with
  | .error e => throwError s!"band: {e}"
  | .ok out => match out["Band"]? with
    | some B => unless DenseTensor.approxEq B (tensorOf [3,3] [1,2,0, 4,5,6, 0,8,9]) do
        throwError s!"band wrong: {repr B.data}"
    | none => throwError "band: no Band"

/- 8. Look-back `Y[i] := X[i-1]`. X = [10,20,30,40]. The output axis `i` ranges over the
    largest in-range domain of the shifted read (valid-range size inference) ⇒ length 5;
    `Y[0]=0` (out-of-range zero-pad), `Y[i]=X[i-1]` for i≥1: [0,10,20,30,40]. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (tensorOf [4] [10,20,30,40])).insert "_L" (tensorOf [1] [0])
  match TLProgram.eval (tlprog!{ Y[i] := X[i - 1] }) env with
  | .error e => throwError s!"lookback: {e}"
  | .ok out => match out["Y"]? with
    | some Y => unless DenseTensor.approxEq Y (tensorOf [5] [0,10,20,30,40]) do
        throwError s!"lookback wrong: shape={repr Y.shape} {repr Y.data}"
    | none => throwError "lookback: no Y"

/- 9. Outer product `Y[i,j] := A[i]·B[j]` (broadcasting; no contracted axis). A=[1,2],
    B=[10,20,30] ⇒ Y (2×3) = [[10,20,30],[20,40,60]]. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "A" (tensorOf [2] [1,2])).insert "B" (tensorOf [3] [10,20,30])
  match TLProgram.eval (tlprog!{ Y[i, j] := A[i] · B[j] }) env with
  | .error e => throwError s!"outer: {e}"
  | .ok out => match out["Y"]? with
    | some Y => unless DenseTensor.approxEq Y (tensorOf [2,3] [10,20,30, 20,40,60]) do
        throwError s!"outer wrong: {repr Y.data}"
    | none => throwError "outer: no Y"

/- 10. Contraction + relu `Y[i] := relu(W[i,k]·X[k])`. W = [[1,-2],[-1,1]], X = [3,1] ⇒
    pre-activation W·X = [1·3-2·1, -1·3+1·1] = [1, -2]; relu clips ⇒ [1, 0]. -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "W" (tensorOf [2,2] [1,-2, -1,1])).insert "X" (tensorOf [2] [3,1])
  match TLProgram.eval (tlprog!{ Y[i] := relu(W[i, k] · X[k]) }) env with
  | .error e => throwError s!"crelu: {e}"
  | .ok out => match out["Y"]? with
    | some Y => unless DenseTensor.approxEq Y (tensorOf [2] [1, 0]) do
        throwError s!"crelu wrong: {repr Y.data}"
    | none => throwError "crelu: no Y"

/- 11. Normalize `Y[q,s] := normalize(A[q,s])` over the `norm` axis `s`. A = [[1,3],[2,2]] ⇒
    each q-row divided by its sum: [[0.25,0.75],[0.5,0.5]] (each row sums to 1, ∝ A). -/
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "A" (tensorOf [2,2] [1,3, 2,2])).insert "_L" (tensorOf [1] [0])
  match TLProgram.eval (tlprog!{
    tensor Y : (q : ℝ, s : norm)
    Y[q, s] := normalize(A[q, s])
  }) env with
  | .error e => throwError s!"normalize: {e}"
  | .ok out => match out["Y"]? with
    | some Y =>
        unless DenseTensor.approxEq Y (tensorOf [2,2] [0.25, 0.75, 0.5, 0.5]) do
          throwError s!"normalize wrong: {repr Y.data}"
        let row0 := Y.get! [0,0] + Y.get! [0,1]
        unless Float.abs (row0 - 1.0) < 1e-6 do throwError s!"normalize: row ≠ 1: {repr Y.data}"
    | none => throwError "normalize: no Y"

end LeanNCD.Eval
