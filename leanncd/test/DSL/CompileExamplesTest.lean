import LeanNCD.DSL.Compile
namespace LeanNCD

/-! # §12.1 acceptance test — all five example programs compile end-to-end via `tl!{}`.

Each `tl!{…}` must ELABORATE (i.e. `TLProgram.compile` returns `.ok`, not a `CompileError`),
and each `#guard` asserts a structural fact calibrated to what the pipeline actually produces. -/

-- 1. Matmul `Y[i,j] := W[i,k] · X[k,j]`. One contraction step; W,X external; the single
--    output weave has exactly one contracted (`.tiled`) axis — the summed `k`.
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).nExternal == 2
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).steps.length == 1
#guard
  let tc := tl!{ Y[i,j] := W[i,k] · X[k,j] }
  (tc.steps.head!.outputWeaves.head!.filter (· == WeaveSlotP.tiled)).length == 1

-- 2. Causal masked attention. After `splitNonlins` the masked-softmax statement becomes a
--    linear pre-activation step + a `softmax` step ⇒ ≥2 steps, one of which is `op == "softmax"`.
#guard
  let tc := tl!{
    tensor A(q, s)
    A[q, s.] := softmax(where s ≤ q)(Q[q, d] · K[s, d])
  }
  tc.steps.length ≥ 2
#guard
  let tc := tl!{
    tensor A(q, s)
    A[q, s.] := softmax(where s ≤ q)(Q[q, d] · K[s, d])
  }
  tc.steps.any (fun s => s.op == BrOp.softmax)

-- 3. Strided convolution `Y[i,j] := W[p,r] · X[i+p, 2*j+r]`. The stride 2 is absorbed into the
--    read's reindexing matrix: some step has a coeff row containing `2`.
#guard
  let tc := tl!{ Y[i, j] := W[p, r] · X[i + p, 2 * j + r] }
  tc.steps.any (fun s => s.reindexings.any (fun m => m.coeffs.any (fun row => row.contains 2)))

-- 4. Upsample 2× `Out[2*i, 2*j] := X[i,j]`. The affine LHS is reclassified to a scatter ⇒
--    some step has `op == "scatter"`.
#guard
  let tc := tl!{
    tensor Out(i, j)
    Out[2 * i, 2 * j] := X[i, j]
  }
  tc.steps.any (fun s => s.op == BrOp.scatter)

-- 5. Coupled scan: G and H recur over the shared iteration axis `l` ⇒ ONE coupled scan step
--    (`op == "scan"`), and X,Y,W_G,U,W_H,V are the six external inputs.
#guard
  let tc := tl!{
    G[j, 0]    := X[j]
    G[j, l +1] := relu(G[j, l] · W_G[j, k] + H[j, l] · U[j, k])
    H[j, 0]    := Y[j]
    H[j, l +1] := relu(H[j, l] · W_H[j, k] + G[j, l] · V[j, k])
  }
  tc.steps.any (fun s => s.op == BrOp.scan || s.op == BrOp.scanAffine)
#guard
  let tc := tl!{
    G[j, 0]    := X[j]
    G[j, l +1] := relu(G[j, l] · W_G[j, k] + H[j, l] · U[j, k])
    H[j, 0]    := Y[j]
    H[j, l +1] := relu(H[j, l] · W_H[j, k] + G[j, l] · V[j, k])
  }
  tc.nExternal ≥ 4

end LeanNCD
