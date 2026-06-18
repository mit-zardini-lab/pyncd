import LeanNCD.DSL.Elab

namespace LeanNCD

-- The term macro: parse a whole program into a TLProgram value (embedded via ToExpr).
private def prog : TLProgram := tlprog!{
  tensor A : (q, s)
  A[q, s.] := softmax(where s ≤ q)(Q[q, d] · K[s, d])
}

#guard prog.decls.length == 1
#guard prog.stmts.length == 1

-- a scan-step LHS (`l +1`, spaced) and a base-case (`0`) parse:
private def scanprog : TLProgram := tlprog!{
  G[j, 0]   := X[j]
  G[j, l +1] := relu(G[j, l] · W[j, k])
}
#guard scanprog.stmts.length == 2

end LeanNCD
