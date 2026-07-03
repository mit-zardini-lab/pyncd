import LeanNCD.DSL.Compile
import LeanNCD.Bridge.AcsetCodec

/-! # Acset codec round-trip regression (Task F)

Computational witness of the Task C theorem `AcsetCodec.toThreadedComposed_fromThreadedComposed`:
decoding the encoded acset tables recovers the original `ThreadedComposed` on the nose, checked by
`decide` over the §12.1 example programs. A `#guard` failure means the encode/decode pair drifted out
of sync (fires at `lake build`). -/

namespace LeanNCD
open LeanNCD.AcsetCodec

-- 1. Matmul (single contraction step, two external inputs).
#guard toThreadedComposed (fromThreadedComposed (tl!{ Y[i,j] := W[i,k] · X[k,j] }))
        = tl!{ Y[i,j] := W[i,k] · X[k,j] }

-- 2. Causal masked attention (multi-step: linear pre-activation + softmax).
#guard toThreadedComposed (fromThreadedComposed (tl!{
          tensor A(q, s)
          A[q, s.] := softmax(where s ≤ q)(Q[q, d] · K[s, d]) }))
        = tl!{
          tensor A(q, s)
          A[q, s.] := softmax(where s ≤ q)(Q[q, d] · K[s, d]) }

-- 3. Strided convolution (nontrivial reindexing coefficients — exercises the matrix round trip).
#guard toThreadedComposed (fromThreadedComposed (tl!{ Y[i, j] := W[p, r] · X[i + p, 2 * j + r] }))
        = tl!{ Y[i, j] := W[p, r] · X[i + p, 2 * j + r] }

-- 4. Upsample 2× (affine-LHS scatter).
#guard toThreadedComposed (fromThreadedComposed (tl!{
          tensor Out(i, j)
          Out[2 * i, 2 * j] := X[i, j] }))
        = tl!{
          tensor Out(i, j)
          Out[2 * i, 2 * j] := X[i, j] }

-- 5. Coupled scan (one multi-output scan step, six external inputs).
#guard toThreadedComposed (fromThreadedComposed (tl!{
          G[j, 0]    := X[j]
          G[j, l +1] := relu(G[j, l] · W_G[j, k] + H[j, l] · U[j, k])
          H[j, 0]    := Y[j]
          H[j, l +1] := relu(H[j, l] · W_H[j, k] + G[j, l] · V[j, k]) }))
        = tl!{
          G[j, 0]    := X[j]
          G[j, l +1] := relu(G[j, l] · W_G[j, k] + H[j, l] · U[j, k])
          H[j, 0]    := Y[j]
          H[j, l +1] := relu(H[j, l] · W_H[j, k] + G[j, l] · V[j, k]) }

end LeanNCD
