import LeanNCD.DSL.Elab

/-!
# Predicate parsing tests (Milestone E1 front-end)

The five §12.1 examples in `ParseExamplesTest.lean` do not exercise predicates, so these
two programs cover the two predicate-bearing surface forms:

1. a `predicate` DECLARATION (a Bool-typed tensor) read as a factor — drawn from
   `docs/dsl_examples.md` Example 2 (masked aggregation over a graph adjacency); and
2. an Iverson-bracket factor `[bool_expr]` whose predicate arithmetic uses `|·|` (iabs)
   inside a comparison — a tridiagonal band mask.

All assertions parse via `tlprog!{}` (Stage 1 only); no compilation is involved.
-/

namespace LeanNCD

/-- Bool helpers with explicit `Decl →` / `Factor →` types so the `.predicate`/`.iverson`
    constructor patterns elaborate (a bare `.iverson` in a nested match cannot infer its type). -/
private def Decl.isPredicate : Decl → Bool | .predicate .. => true | _ => false
private def Factor.isIverson : Factor → Bool | .iverson _ => true | _ => false

-- Test 1 — predicate declaration + masked aggregation (docs/dsl_examples.md Example 2).
-- `edge(i,j)` is a Bool-typed 5×5 adjacency predicate; it gates a doubly-contracted
-- feature product, and every index is contracted (scalar `Result[]`).
private def maskedAgg : TLProgram := tlprog!{
  predicate edge : (i, j)
  Result[] := F[t, i] · F[t, j] · edge[i, j]
}
#guard maskedAgg.decls.length == 1
#guard maskedAgg.stmts.length == 1
-- the declaration is genuinely a `predicate` (Bool-typed), not a `tensor`
#guard (maskedAgg.decls.head?.map Decl.isPredicate).getD false
-- the equation is a 3-factor Einstein product: F · F · edge
#guard (match maskedAgg.stmts.head? with
        | some (.assign _ _ r) => (r.body.terms.head?.map (·.factors.length)) == some 3
        | _ => false)

-- Test 2 — Iverson-bracket predicate: a tridiagonal band mask `[|i - j| ≤ 1]`.
-- Exercises the `[bool_expr]` factor and the PredArith `|·|` (iabs) inside a `≤` comparison.
private def band : TLProgram := tlprog!{ Band[i, j] := A[i, j] · [|i - j| ≤ 1] }
#guard band.stmts.length == 1
-- a 2-factor product: the data read `A` and the Iverson mask
#guard (match band.stmts.head? with
        | some (.assign _ _ r) => (r.body.terms.head?.map (·.factors.length)) == some 2
        | _ => false)
-- one of the factors is an Iverson bracket `[…]`
#guard (match band.stmts.head? with
        | some (.assign _ _ r) => (r.body.terms.head?.map (·.factors.any Factor.isIverson)).getD false
        | _ => false)

end LeanNCD
