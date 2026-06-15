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

-- Two `k` axes with DIFFERENT uids (7 and 3) must unify to the canonical (max = 7);
-- i,j keep their own uids. (Feeds resolveDecls directly with distinct uids — skips
-- assignUIDs, which would otherwise pre-bind the two k's.)
run_cmd do
  let i : AxisSpec := { name := "i", uid := 5, kind := .real none }
  let j : AxisSpec := { name := "j", uid := 4, kind := .real none }
  let k1 : AxisSpec := { name := "k", uid := 7, kind := .real none }
  let k2 : AxisSpec := { name := "k", uid := 3, kind := .real none }
  let lp : LabeledProgram := { decls := [], stmts := [
    .assign "Y" [ .free i, .free j ]
      { body := { terms := [ { factors := [
            .read "W" [ .axis i, .axis k1 ],
            .read "X" [ .axis k2, .axis j ] ] } ] },
        nonlin := .identity } ] }
  match (resolveDecls lp >>= unifyAxes) |>.run 0 with
  | .ok cp _ =>
      let prog : TLProgram := { decls := cp.decls, stmts := cp.stmts }
      let kUIDs := ((collectAxisNameUID prog).filter (·.1 == "k")).map (·.2) |>.eraseDups
      unless kUIDs == [7] do throwError s!"k should unify to [7], got {kUIDs}"
      let iUIDs := ((collectAxisNameUID prog).filter (·.1 == "i")).map (·.2) |>.eraseDups
      unless iUIDs == [5] do throwError s!"i should stay [5], got {iUIDs}"
  | .error e _ => throwError s!"errored: {repr e}"

-- lowerArith
-- Upsample: Out[2*i, 2*j] := X[i,j] — affine LHS ⇒ reclassified to Stmt.scatter, injective (no error).
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 1, kind := .real none }
  let upsample : Stmt := .assign "Out"
    [ .affine (.scale 2 (ax "i")), .affine (.scale 2 (ax "j")) ]
    { body := { terms := [ { factors := [ .read "X" [ .axis (ax "i"), .axis (ax "j") ] ] } ] },
      nonlin := .identity }
  let cp : CanonicalProgram :=
    { decls := [], stmts := [upsample], env := {}, extNames := ∅, ctx := { classes := [] } }
  match lowerArith cp |>.run 0 with
  | .ok lp _ =>
      match lp.stmts with
      | [ .scatter "Out" _ _ _ ] => pure ()
      | _ => throwError s!"expected one Stmt.scatter for Out, got {repr lp.stmts}"
  | .error e _ => throwError s!"lowerArith errored: {repr e}"

-- Plain matmul assign is left as Stmt.assign.
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 1, kind := .real none }
  let mm : Stmt := .assign "Y" [ .free (ax "i"), .free (ax "j") ]
    { body := { terms := [ { factors := [ .read "W" [.axis (ax "i"), .axis (ax "k")],
                                            .read "X" [.axis (ax "k"), .axis (ax "j")] ] } ] },
      nonlin := .identity }
  let cp : CanonicalProgram :=
    { decls := [], stmts := [mm], env := {}, extNames := ∅, ctx := { classes := [] } }
  match lowerArith cp |>.run 0 with
  | .ok lp _ => match lp.stmts with
                | [ .assign "Y" _ _ ] => pure ()
                | _ => throwError "matmul assign should remain Stmt.assign"
  | .error e _ => throwError s!"errored: {repr e}"

-- A collapsing constant LHS coord without reduce ⇒ overlappingScatter.
run_cmd do
  let collapse : Stmt := .assign "Z" [ .affine (.const 0) ]
    { body := { terms := [ { factors := [ .read "X" [ .const 0 ] ] } ] }, nonlin := .identity }
  let cp : CanonicalProgram :=
    { decls := [], stmts := [collapse], env := {}, extNames := ∅, ctx := { classes := [] } }
  match lowerArith cp |>.run 0 with
  | .error (.overlappingScatter "Z") _ => pure ()
  | .error e _ => throwError s!"wrong error: {repr e}"
  | .ok _ _ => throwError "expected overlappingScatter for collapsing const LHS"
end LeanNCD
