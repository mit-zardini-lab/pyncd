-- test/DSL/Pipeline/StructuralTest.lean
import LeanNCD.DSL.Pipeline.Structural
namespace LeanNCD
-- matmul: Y[i,j] := W[i,k]·X[k,j].  After assignUIDs: no axis uid is 0, and the
-- two `k` occurrences share one uid while i,j,k are pairwise distinct.
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 0, kind := .real none }
  let p : TLProgram := { decls := [], stmts := [
    .assign "Y" [ .free (ax "i"), .free (ax "j") ]
      { body := { terms := [ { factors := [
            .read "W" [ .axis (ax "i"), .axis (ax "k") ],
            .read "X" [ .axis (ax "k"), .axis (ax "j") ] ] } ] },
        nonlin := .identity } ] }
  match assignUIDs p |>.run 0 with
  | .ok lp _ =>
      let uids := lp.stmts.flatMap Stmt.uids
      unless uids.all (· ≠ 0) do throwError s!"found zero UID: {uids}"
      -- i,j,k → exactly 3 distinct uids
      unless uids.eraseDups.length == 3 do throwError s!"expected 3 distinct axis uids, got {uids.eraseDups}"
  | .error e _ => throwError s!"assignUIDs errored: {repr e}"

-- matmul Y[i,j] := W[i,k]·X[k,j]: W,X are external inputs; Y is produced (not external).
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 0, kind := .real none }
  let p : TLProgram := { decls := [], stmts := [
    .assign "Y" [ .free (ax "i"), .free (ax "j") ]
      { body := { terms := [ { factors := [
            .read "W" [ .axis (ax "i"), .axis (ax "k") ],
            .read "X" [ .axis (ax "k"), .axis (ax "j") ] ] } ] },
        nonlin := .identity } ] }
  match (assignUIDs p >>= resolveDecls) |>.run 0 with
  | .ok rp _ =>
      unless decide ("W" ∈ rp.extNames) && decide ("X" ∈ rp.extNames) do
        throwError "W,X should be external"
      unless ¬ decide ("Y" ∈ rp.extNames) do throwError "Y must not be external (it is produced)"
  | .error e _ => throwError s!"resolveDecls errored (should never throw): {repr e}"

-- a `tensor` decl lands in env.
run_cmd do
  let p : TLProgram := { decls := [ .tensor "A" [] ], stmts := [
    .assign "A" [] { body := { terms := [] }, nonlin := .identity } ] }
  match (assignUIDs p >>= resolveDecls) |>.run 0 with
  | .ok rp _ => unless rp.env.contains "A" do throwError "env missing declared A"
  | .error e _ => throwError s!"errored: {repr e}"
end LeanNCD
