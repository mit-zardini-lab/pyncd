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

/-- Input tensors: A, B, both 1-D over axis `i` (kept small). `P` is a 2-D tensor over
    `[i, j]`, used only by the contraction programs below — it is what finally puts the
    `j` axis to work. -/
private def inputDecls : List Decl := [.tensor "A" [i], .tensor "B" [i]]
private def pDecl : Decl := .tensor "P" [i, j]

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

/-- Deterministic input env: A,B are 1-D size-2 tensors with data [1,2]/[3,4]; P is a
    2-D size-(2×2) tensor with data [1,2,3,4] (used only by the contraction programs). -/
private def inputEnv : Std.HashMap String DenseTensor :=
  (((({} : Std.HashMap String DenseTensor).insert "A" ⟨[2], #[1.0, 2.0]⟩).insert "B" ⟨[2], #[3.0, 4.0]⟩).insert "P" ⟨[2, 2], #[1.0, 2.0, 3.0, 4.0]⟩)

/-- Cap on how many of `stmtChoices` per name are crossed to build 2-statement
    programs (`termChoices`/`rhsChoices` are already large — 42/1806 — and their
    full cross product would blow well past the ≤5000 bound). The single-statement
    enumeration below still uses every one of the 1806 `rhsChoices`. -/
private def twoStmtCap : Nat := 40

private def cappedTwoStmt : Bool := (stmtChoices "Y").length > twoStmtCap

/-! ## Y-dependent 2-statement programs (Task 5, E6 review — the key reordering fix)

The `two` programs above are INDEPENDENT (`Y := f(A,B); Z := g(A,B)`): permuting them to
`[Z; Y]` still evaluates identically, so `checkLaws`'s reordering law never exercises real
reordering. `yDepPrograms` instead builds `Y := f(A,B); Z := g(...Y...)` where `Z`'s RHS READS
the tensor `Y` produced by the first statement — a genuine producer→consumer dependency.
Permuting the statement list to `[Z; Y]` now forces `schedule`/`topoSort`
(`LeanNCD/DSL/Pipeline/Lowering.lean`) to reorder producer-before-consumer to get the same
answer, so the reordering law does real work on these programs. -/

/-- Cap on how many `stmtChoices "Y"` (producer statements) are crossed with `zRhsChoices`
    below, kept small since this is purely additive to the existing bound. -/
private def yDepCap : Nat := 20

/-- A product term reading the produced intermediate `Y` at index `e`. -/
private def yReadTerm (e : IdxExpr) : ProdTerm := ⟨[Factor.read "Y" [e]]⟩

/-- RHS choices for `Z`: each reads `Y` alone, or `Y` summed with one A/B read term
    (3 idx choices × (1 self + 6 partner reads) = 21 — small and bounded). -/
private def zRhsChoices : List RHSExpr :=
  (idxChoices i).flatMap (fun e =>
    ({ body := ⟨[yReadTerm e]⟩, nonlin := .identity, agg := .sum } : RHSExpr) ::
    readChoices.map (fun f =>
      { body := ⟨[yReadTerm e, ⟨[f]⟩]⟩, nonlin := .identity, agg := .sum }))

/-- Y-dependent programs: `yDepCap` (20) first-statement (`Y`) choices × `zRhsChoices`
    (21) = 420 programs, each `[Y := f(A,B); Z := g(...Y...)]`. -/
private def yDepPrograms : List TLProgram :=
  ((stmtChoices "Y").take yDepCap).flatMap (fun s1 =>
    zRhsChoices.map (fun r2 =>
      ({ decls := axDecls ++ inputDecls, stmts := [s1, Stmt.assign "Z" [.free i] r2] } :
        TLProgram)))

/-! ## A genuine-contraction program (Task 5 — exercises materialization beyond the linear/sum
fragment, and puts the previously-unused `j` axis to work)

`C[i] := P[i,j]` (`agg := .sum`) reads the 2-D input `P` over both `i` and `j`, but its LHS
only retains `i` — `j` is CONTRACTED (summed away), matching the Route phase's own definition
(`Lowering.lean`: "Contracted axes = read axes whose uid is NOT among the LHS axis uids"). -/

/-- A product term reading `P` at `(e, j)` — `j` fixed at a plain read (the contracted axis),
    `e` ranging over `idxChoices i` (the retained axis's read expression). -/
private def pTerm (e : IdxExpr) : ProdTerm := ⟨[Factor.read "P" [e, .axis j]]⟩

/-- 3 contraction programs (one per `idxChoices i` variant on the retained axis). -/
private def contractPrograms : List TLProgram :=
  (idxChoices i).map (fun e =>
    ({ decls := axDecls ++ inputDecls ++ [pDecl],
       stmts := [Stmt.assign "C" [.free i]
         ({ body := ⟨[pTerm e]⟩, nonlin := .identity, agg := .sum } : RHSExpr)] } :
      TLProgram))

/-! ## A multi-term contraction program (Task 5b — materialization must bite on a CONTRACTING
statement)

`contractPrograms` above are all SINGLE-term (`C[i] := P[i,j]`), so `materializeSplit`
(`terms.length ≥ 2` is its split trigger, `Transforms.lean`) never touches them — materialization
was never exercised on a contracting statement. `contractMultiTermPrograms` fixes this:
`C[i] := P[i,j] + A[i]` has TWO terms — `P[i,j]` contracts `j` (read but absent from the LHS
`[i]`), `A[i]` is a plain read — so `materializeSplit` splits it into `T0[i] := P[i,j]` (itself a
contracting statement) and `T1[i] := A[i]`, summed. -/

/-- A product term reading `A` at `e` (shares `e` with `pTerm` so both terms of a program use the
    same output index). -/
private def aTerm (e : IdxExpr) : ProdTerm := ⟨[Factor.read "A" [e]]⟩

/-- 3 multi-term contraction programs (one per `idxChoices i` variant), each 2-term: the
    contracting `P[e,j]` plus the plain `A[e]`. -/
private def contractMultiTermPrograms : List TLProgram :=
  (idxChoices i).map (fun e =>
    ({ decls := axDecls ++ inputDecls ++ [pDecl],
       stmts := [Stmt.assign "C" [.free i]
         ({ body := ⟨[pTerm e, aTerm e]⟩, nonlin := .identity, agg := .sum } : RHSExpr)] } :
      TLProgram))

/-- Bounded enumeration: 1-statement programs over every RHS choice (`Y := f(A,B)`), 2-statement
    INDEPENDENT programs (`Y := f(A,B); Z := g(A,B)`) over a capped subset of RHS choices per
    statement, Y-DEPENDENT 2-statement programs (`yDepPrograms`), genuine-contraction programs
    (`contractPrograms`), and multi-term contraction programs (`contractMultiTermPrograms`). -/
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
  (one ++ two ++ yDepPrograms ++ contractPrograms ++ contractMultiTermPrograms).map
    (fun p => (p, inputEnv))

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
-- (3b) Y-dependent coverage: some program has a statement that reads a name produced by an
-- earlier statement (the reordering law is no longer vacuous):
#guard enumPrograms.any (fun (p, _) =>
  (p.stmts.foldl (fun (acc : Bool × List String) s =>
      let (found, produced) := acc
      (found || s.readNames.any produced.contains, produced ++ [s.lhsName]))
    (false, [])).1)
-- (3c) contraction coverage: some program has a statement that reads an axis absent from its
-- own LHS (a genuinely contracted axis, per the Route phase's definition):
#guard enumPrograms.any (fun (p, _) =>
  p.stmts.any (fun s =>
    let lhsUids := (Stmt.lhsAxes s).map AxisSpec.uid
    (Stmt.readFactors s).any (fun (_, idxs) =>
      idxs.any (fun e => (idxAffineForm e).2.any (fun (_, u) => !lhsUids.contains u)))))
-- (3d) materialization-bites-on-contraction coverage (Task 5b): some program has a statement
-- that is BOTH multi-term (`terms.length ≥ 2`, so `materializeSplit` will split it) AND has a
-- term reading an axis absent from the statement's own LHS (a genuinely contracted axis) — i.e.
-- a statement whose split intermediate is itself a contracting statement.
#guard enumPrograms.any (fun (p, _) =>
  p.stmts.any (fun s =>
    let lhsUids := (Stmt.lhsAxes s).map AxisSpec.uid
    let isMultiTerm : Bool := match s with
      | .assign _ _ r | .scatter _ _ r _ => r.body.terms.length ≥ 2
      | _ => false
    isMultiTerm && (Stmt.readFactors s).any (fun (_, idxs) =>
      idxs.any (fun e => (idxAffineForm e).2.any (fun (_, u) => !lhsUids.contains u)))))

end LeanNCD.PropertyOracle

