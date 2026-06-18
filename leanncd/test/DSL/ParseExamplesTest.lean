import LeanNCD.DSL.Elab

namespace LeanNCD

-- 1. Matmul (k contracted)
private def matmul : TLProgram := tlprog!{ Y[i, j] := W[i, k] · X[k, j] }
#guard matmul.decls.length == 0
#guard matmul.stmts.length == 1

-- 2. Causal masked attention (norm axis marked `s.` + Iverson mask via softmax-where)
private def attn : TLProgram := tlprog!{
  tensor A(q, s)
  A[q, s.] := softmax(where s ≤ q)(Q[q, d] · K[s, d])
}
#guard attn.decls.length == 1
#guard attn.stmts.length == 1

-- 3. Strided convolution (affine reads; CONCRETE stride 2 — symbolic strides are unsupported, see note)
private def conv : TLProgram := tlprog!{ Y[i, j] := W[p, r] · X[i + p, 2 * j + r] }
#guard conv.stmts.length == 1

-- 4. Upsample 2× (affine scatter write — parsed as assign in E1; E2 reclassifies)
private def upsample : TLProgram := tlprog!{
  tensor Out(i, j)
  Out[2 * i, 2 * j] := X[i, j]
}
#guard upsample.decls.length == 1
#guard upsample.stmts.length == 1

-- 5. Coupled scan (G, H share iteration axis l; scan-step LHS written spaced as `l +1`)
private def coupled : TLProgram := tlprog!{
  G[j, 0]    := X[j]
  G[j, l +1] := relu(G[j, l] · W_G[j, k] + H[j, l] · U[j, k])
  H[j, 0]    := Y[j]
  H[j, l +1] := relu(H[j, l] · W_H[j, k] + G[j, l] · V[j, k])
}
#guard coupled.stmts.length == 4

end LeanNCD
