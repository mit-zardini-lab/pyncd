import LeanNCD.DSL.Ast
import LeanNCD.Eval.Eval

/-!
# Curated scan-template generator (E6 scan-unrolling oracle, Task 1/2)

`enumScanCases` is a small, curated (not combinatorial) family of six scan templates, each
authored directly as `Stmt`/`Decl` values (same convention as `test/Eval/ScanTest.lean`), varied
over a few parameters (scan length `L`, coefficient signs, aggregator). Scan well-formedness is
materially tighter than the scan-free fragment (causality, matching base/recur names, full-axis-
set coupling), so a curated family is lower-risk than combinatorial enumeration here.

INVARIANT (load-bearing for `ScanUnroll`'s slicing, added in a later task): in every generated
statement, scan-axis LHS slots are always the LAST slot position(s).
-/
namespace LeanNCD.PropertyOracle
open LeanNCD LeanNCD.Eval

/-- One curated scan test case: the full program + its deterministic inputs, plus the scan's
    own structure (axes, per-axis lengths, base/recur statements) as authored — since this
    generator builds `Stmt` values directly, it already knows this grouping and doesn't need to
    re-derive it the way `finalizeScans` does at compile time. -/
structure ScanCase where
  prog   : TLProgram
  inputs : Std.HashMap String DenseTensor
  axes   : List AxisSpec
  Ls     : List Nat
  base   : List Stmt
  recur  : List Stmt

-- ===== Template 1: linear self-scan  S[j,l+1] := S[j,l]·A[j] =====
private def j1 : AxisSpec := ⟨"j", 201, .real (some (.lit 2))⟩
private def l1 (L : Nat) : AxisSpec := ⟨"l", 202, .nat (some (.lit L))⟩

private def template1 (L : Nat) (Aneg : Bool) : ScanCase :=
  let l := l1 L
  let base : Stmt := .assign "S" [.free j1, .iterAt l 0]
    { body := { terms := [{ factors := [.read "X0" [.axis j1]] }] }, nonlin := .identity }
  let recur : Stmt := .assign "S" [.free j1, .iterNext l]
    { body := { terms := [{ factors := [.read "S" [.axis j1, .axis l], .read "A" [.axis j1]] }] },
      nonlin := .identity }
  let aVals : Array Float := if Aneg then #[-2.0, 3.0] else #[2.0, 3.0]
  let inputs : Std.HashMap String DenseTensor :=
    (({} : Std.HashMap String DenseTensor).insert "X0" ⟨[2], #[1.0, 2.0]⟩).insert "A" ⟨[2], aVals⟩
  { prog := { decls := [.axis j1 (some 2), .axis l (some L), .tensor "X0" [j1], .tensor "A" [j1]],
              stmts := [base, recur] },
    inputs := inputs, axes := [l], Ls := [L], base := [base], recur := [recur] }

-- ===== Template 2: nonlin self-scan  S[j,l+1] := relu(S[j,l]·A[j]) =====
private def j2 : AxisSpec := ⟨"j", 211, .real (some (.lit 2))⟩
private def l2 (L : Nat) : AxisSpec := ⟨"l", 212, .nat (some (.lit L))⟩

private def template2 (L : Nat) (Aneg : Bool) : ScanCase :=
  let l := l2 L
  let base : Stmt := .assign "S" [.free j2, .iterAt l 0]
    { body := { terms := [{ factors := [.read "X0" [.axis j2]] }] }, nonlin := .identity }
  let recur : Stmt := .assign "S" [.free j2, .iterNext l]
    { body := { terms := [{ factors := [.read "S" [.axis j2, .axis l], .read "A" [.axis j2]] }] },
      nonlin := .relu }
  let aVals : Array Float := if Aneg then #[-1.0, -1.0] else #[1.0, 2.0]
  let inputs : Std.HashMap String DenseTensor :=
    (({} : Std.HashMap String DenseTensor).insert "X0" ⟨[2], #[1.0, 1.0]⟩).insert "A" ⟨[2], aVals⟩
  { prog := { decls := [.axis j2 (some 2), .axis l (some L), .tensor "X0" [j2], .tensor "A" [j2]],
              stmts := [base, recur] },
    inputs := inputs, axes := [l], Ls := [L], base := [base], recur := [recur] }

-- ===== Template 3: coupled 2-state  G[l+1]:=G[l]+H[l]; H[l+1]:=G[l] =====
private def l3 (L : Nat) : AxisSpec := ⟨"l", 222, .nat (some (.lit L))⟩

private def template3 (L : Nat) : ScanCase :=
  let l := l3 L
  let baseG : Stmt := .assign "G" [.iterAt l 0]
    { body := { terms := [{ factors := [.read "C" []] }] }, nonlin := .identity }
  let baseH : Stmt := .assign "H" [.iterAt l 0]
    { body := { terms := [{ factors := [.read "C" []] }] }, nonlin := .identity }
  let recurG : Stmt := .assign "G" [.iterNext l]
    { body := { terms := [{ factors := [.read "G" [.axis l]] }, { factors := [.read "H" [.axis l]] }] },
      nonlin := .identity }
  let recurH : Stmt := .assign "H" [.iterNext l]
    { body := { terms := [{ factors := [.read "G" [.axis l]] }] }, nonlin := .identity }
  let inputs : Std.HashMap String DenseTensor := (({} : Std.HashMap String DenseTensor).insert "C" ⟨[], #[1.0]⟩)
  { prog := { decls := [.axis l (some L), .tensor "C" []],
              stmts := [baseG, baseH, recurG, recurH] },
    inputs := inputs, axes := [l], Ls := [L], base := [baseG, baseH], recur := [recurG, recurH] }

/-- Templates 1–3 only; Task 2 extends this into the full six-template `enumScanCases`. -/
def partialScanCases : List ScanCase :=
  ([2, 3].flatMap (fun L => [true, false].map (fun neg => template1 L neg))) ++
  ([2, 3].flatMap (fun L => [true, false].map (fun neg => template2 L neg))) ++
  ([2, 3].map template3)

-- CONTRACT TESTS (fire on build):
#guard partialScanCases.length == 10   -- 4 (template1) + 4 (template2) + 2 (template3)
#guard partialScanCases.all (fun c => (TLProgram.eval c.prog c.inputs).toOption.isSome)

end LeanNCD.PropertyOracle
