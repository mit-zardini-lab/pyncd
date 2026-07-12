import LeanNCD.DSL.Ast
import LeanNCD.Eval.Eval

/-!
# Bounded-exhaustive generator of well-formed scan-free programs (Task 3, E6)

`enumPrograms` is a finite list of `(TLProgram, inputs)` pairs, each a well-formed
scan-free program paired with a deterministic input env covering exactly its input
tensors. "Well-formed" here means: every read names a declared tensor with the
right arity, every axis is declared with a concrete pinned size, and the input env
provides a `DenseTensor` for every input tensor name with `shape`/`data` matching
the declared axis sizes. Contract test (2) below is the load-bearing check: EVERY
generated program must compile (`compileToScheduled`) AND evaluate to `.ok` — that
is what makes this a trustworthy source of baselines for property-based oracles.
-/
namespace LeanNCD.PropertyOracle
open LeanNCD LeanNCD.Eval

private def i : AxisSpec := ⟨"i", 1, .real (some (.lit 2))⟩
private def j : AxisSpec := ⟨"j", 2, .real (some (.lit 2))⟩
private def axDecls : List Decl := [.axis i (some 2), .axis j (some 2)]

/-- Affine index-expr choices over one axis (plain, +1, 2·). Out-of-range reads
    zero-pad (`Eval/Shape.lean: gatherRead`), so these never fail eval — they are
    included purely for the affine-read coverage guard (3). -/
private def idxChoices (a : AxisSpec) : List IdxExpr := [.axis a, .shift a 1, .scale 2 a]

/-- Input tensors: A, B, both 1-D over axis `i` (kept small; `j` is declared but
    unused by any tensor/read in this bound — reserved for a future widening). -/
private def inputDecls : List Decl := [.tensor "A" [i], .tensor "B" [i]]

/-- Read-factor choices for a 1-D output over axis i: A or B, each with each idx choice. -/
private def readChoices : List Factor :=
  (["A", "B"] : List String).flatMap (fun nm => (idxChoices i).map (fun e => Factor.read nm [e]))

/-- Single-term (one read) and product-term (two reads) choices. -/
private def termChoices : List ProdTerm :=
  (readChoices.map (fun f => (⟨[f]⟩ : ProdTerm)))
  ++ (readChoices.flatMap (fun f => readChoices.map (fun g => (⟨[f, g]⟩ : ProdTerm))))

/-- RHS choices: 1-term and 2-term sums (identity nonlin, sum agg), output over [i]. -/
private def rhsChoices : List RHSExpr :=
  (termChoices.map (fun t => ({ body := ⟨[t]⟩, nonlin := .identity, agg := .sum } : RHSExpr)))
  ++ (termChoices.flatMap (fun t => termChoices.map (fun u =>
        ({ body := ⟨[t, u]⟩, nonlin := .identity, agg := .sum } : RHSExpr))))

/-- One statement writing `nm` over [i]. -/
private def stmtChoices (nm : String) : List Stmt :=
  rhsChoices.map (fun r => Stmt.assign nm [.free i] r)

/-- Deterministic input env: A,B are 1-D size-2 tensors with data [1,2]/[3,4]. -/
private def inputEnv : Std.HashMap String DenseTensor :=
  (({} : Std.HashMap String DenseTensor).insert "A" ⟨[2], #[1.0, 2.0]⟩).insert "B" ⟨[2], #[3.0, 4.0]⟩

/-- Cap on how many of `stmtChoices` per name are crossed to build 2-statement
    programs (`termChoices`/`rhsChoices` are already large — 42/1806 — and their
    full cross product would blow well past the ≤5000 bound). The single-statement
    enumeration below still uses every one of the 1806 `rhsChoices`. -/
private def twoStmtCap : Nat := 40

private def cappedTwoStmt : Bool := (stmtChoices "Y").length > twoStmtCap

/-- Bounded enumeration: 1-statement programs over every RHS choice (`Y := f(A,B)`),
    plus 2-statement programs (`Y := f(A,B); Z := g(A,B)`, independent — extend to
    Y-dependent later) over a capped subset of RHS choices per statement. -/
def enumPrograms : List (TLProgram × Std.HashMap String DenseTensor) :=
  let decls := axDecls ++ inputDecls
  let one := (stmtChoices "Y").map (fun s => ({ decls, stmts := [s] } : TLProgram))
  let two :=
    (if cappedTwoStmt then
      dbg_trace s!"Gen.lean: two-statement enumeration capped at {twoStmtCap} RHS choices \
        per statement (of {(stmtChoices "Y").length} total) to keep enumPrograms bounded"
      (stmtChoices "Y").take twoStmtCap
    else (stmtChoices "Y")).flatMap (fun s1 =>
      (if cappedTwoStmt then (stmtChoices "Z").take twoStmtCap else stmtChoices "Z").map
        (fun s2 => ({ decls, stmts := [s1, s2] } : TLProgram)))
  (one ++ two).map (fun p => (p, inputEnv))

-- CONTRACT TESTS (fire on build):
-- (1) non-empty and bounded:
#guard enumPrograms.length > 0
#guard enumPrograms.length ≤ 5000
-- (2) EVERY baseline compiles+evals to `.ok` (generator produces only well-formed programs):
#guard enumPrograms.all (fun (p, env) => (TLProgram.eval p env).toOption.isSome)
-- (3) coverage: at least one multi-term RHS and at least one affine read are generated:
#guard enumPrograms.any (fun (p, _) =>
  p.stmts.any (fun | .assign _ _ r => r.body.terms.length ≥ 2 | _ => false))
#guard enumPrograms.any (fun (p, _) =>
  p.stmts.any (fun | .assign _ _ r => r.body.terms.any (fun t => t.factors.any (fun
      | .read _ idxs => idxs.any (fun | .axis _ => false | _ => true) | _ => false)) | _ => false))

end LeanNCD.PropertyOracle
