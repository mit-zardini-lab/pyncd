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

-- Plain matmul assign is left as Stmt.assign. (Distinct uids per axis, as `assignUIDs` produces —
-- i/j must NOT share a uid, else the LHS `Y[i,j]` looks like a diagonal `Y[i,i]` write.)
run_cmd do
  let ax (nm : String) : AxisSpec :=
    { name := nm, uid := (if nm == "i" then 1 else if nm == "j" then 2 else 3), kind := .real none }
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

-- Coupled scan: G and H both recur over `l` (uid 9) ⇒ ONE ScanStmt.scan whose recur list
-- has BOTH G and H steps; each has a base case (no missingBaseCase).
run_cmd do
  let l : AxisSpec := { name := "l", uid := 9, kind := .nat none }
  let j : AxisSpec := { name := "j", uid := 1, kind := .real none }
  let rhs (nm : String) : RHSExpr :=
    { body := { terms := [ { factors := [ .read nm [ .axis j, .axis l ] ] } ] }, nonlin := .relu }
  let gBase : Stmt := .assign "G" [ .free j, .iterAt l 0 ] { body := { terms := [] }, nonlin := .identity }
  let gRec  : Stmt := .assign "G" [ .free j, .iterNext l ] (rhs "G")
  let hBase : Stmt := .assign "H" [ .free j, .iterAt l 0 ] { body := { terms := [] }, nonlin := .identity }
  let hRec  : Stmt := .assign "H" [ .free j, .iterNext l ] (rhs "H")
  let lp : LoweredProgram := { decls := [], stmts := [gBase, gRec, hBase, hRec], env := {}, extNames := ∅, ctx := { classes := [] } }
  match finalizeScans lp |>.run 0 with
  | .ok sp _ =>
      let scans := sp.stmts.filterMap (fun | .scan _ _ b r _ => some (b, r) | .plain _ => none | .scanPre _ _ _ => none)
      match scans with
      | [(base, recur)] =>
          unless recur.length == 2 do throwError s!"coupled scan recur should have 2 steps, got {recur.length}"
          unless base.length == 2 do throwError s!"coupled scan should have 2 base steps, got {base.length}"
      | _ => throwError s!"expected exactly one coupled ScanStmt.scan, got {scans.length}"
  | .error e _ => throwError s!"finalizeScans errored: {repr e}"

-- A recurrence with no matching base case ⇒ missingBaseCase.
run_cmd do
  let l : AxisSpec := { name := "l", uid := 9, kind := .nat none }
  let orphan : Stmt := .assign "S" [ .iterNext l ]
    { body := { terms := [ { factors := [ .read "S" [ .axis l ] ] } ] }, nonlin := .identity }
  let lp : LoweredProgram := { decls := [], stmts := [orphan], env := {}, extNames := ∅, ctx := { classes := [] } }
  match finalizeScans lp |>.run 0 with
  | .error (.missingBaseCase "S") _ => pure ()
  | .error e _ => throwError s!"wrong error: {repr e}"
  | .ok _ _ => throwError "expected missingBaseCase for orphan recurrence"
-- checkReadRanks: declared tensor read with matching rank passes through.
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 0, kind := .real none }
  let p : TLProgram := {
    decls := [.tensor "W" [ax "i", ax "k"]],
    stmts := [.assign "Y" [.free (ax "i")]
      { body := { terms := [{ factors := [
            .read "W" [.axis (ax "i"), .axis (ax "k")] ] }] },
        nonlin := .identity }] }
  match (assignUIDs p >>= resolveDecls >>= checkReadRanks) |>.run 0 with
  | .ok _ _    => pure ()
  | .error e _ => throwError s!"should not error on correct rank: {repr e}"

-- checkReadRanks: declared tensor read with wrong rank ⇒ rankMismatch.
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 0, kind := .real none }
  let p : TLProgram := {
    decls := [.tensor "W" [ax "i", ax "k"]],   -- declared rank 2
    stmts := [.assign "Y" [.free (ax "i")]
      { body := { terms := [{ factors := [
            .read "W" [.axis (ax "i")] ] }] },  -- read rank 1
        nonlin := .identity }] }
  match (assignUIDs p >>= resolveDecls >>= checkReadRanks) |>.run 0 with
  | .error (.rankMismatch "W" 2 1) _ => pure ()
  | .error e _                       => throwError s!"wrong error: {repr e}"
  | .ok _ _                          => throwError "expected rankMismatch for wrong-rank read"

-- checkReadRanks: external tensor with consistent read arity passes through.
run_cmd do
  let ax : AxisSpec := { name := "i", uid := 1, kind := .real none }
  let readX n : Factor := .read "X" (List.replicate n (.axis ax))
  let mkStmt (nm : String) (n : Nat) : Stmt :=
    .assign nm [] { body := { terms := [{ factors := [readX n] }] }, nonlin := .identity }
  let rp : ResolvedProgram := {
    decls := [], env := {}, extNames := insert "X" ∅,
    stmts := [mkStmt "A" 2, mkStmt "B" 2] }
  match checkReadRanks rp |>.run 0 with
  | .ok _ _    => pure ()
  | .error e _ => throwError s!"consistent external reads should not error: {repr e}"

-- checkReadRanks: external tensor with conflicting read arities ⇒ rankMismatch.
run_cmd do
  let ax : AxisSpec := { name := "i", uid := 1, kind := .real none }
  let readX n : Factor := .read "X" (List.replicate n (.axis ax))
  let mkStmt (nm : String) (n : Nat) : Stmt :=
    .assign nm [] { body := { terms := [{ factors := [readX n] }] }, nonlin := .identity }
  let rp : ResolvedProgram := {
    decls := [], env := {}, extNames := insert "X" ∅,
    stmts := [mkStmt "A" 2, mkStmt "B" 1] }   -- first read rank 2, second rank 1
  match checkReadRanks rp |>.run 0 with
  | .error (.rankMismatch "X" 2 1) _ => pure ()
  | .error e _                       => throwError s!"wrong error: {repr e}"
  | .ok _ _                          => throwError "expected rankMismatch for inconsistent external reads"

-- checkReadRanks: over-indexed read of a produced-but-undeclared intermediate ⇒ rankMismatch
-- (Track A #1 guard: T produced rank 1, read at arity 2, no declaration to justify the higher rank).
run_cmd do
  let i : AxisSpec := { name := "i", uid := 0, kind := .real none }
  let k : AxisSpec := { name := "k", uid := 1, kind := .real none }
  let rp : ResolvedProgram := {
    decls := [], env := {}, extNames := insert "A" ∅,
    stmts := [
      .assign "T" [.free i]
        { body := { terms := [{ factors := [.read "A" [.axis i]] }] }, nonlin := .identity },
      .assign "Y" [.free i, .free k]
        { body := { terms := [{ factors := [.read "T" [.axis i, .axis k]] }] }, nonlin := .identity } ] }
  match checkReadRanks rp |>.run 0 with
  | .error (.rankMismatch "T" 1 2) _ => pure ()
  | .error e _                       => throwError s!"wrong error: {repr e}"
  | .ok _ _                          => throwError "expected rankMismatch for over-indexed intermediate read"

-- checkReadRanks: reading the produced intermediate at its produced rank passes.
run_cmd do
  let i : AxisSpec := { name := "i", uid := 0, kind := .real none }
  let rp : ResolvedProgram := {
    decls := [], env := {}, extNames := insert "A" ∅,
    stmts := [
      .assign "T" [.free i]
        { body := { terms := [{ factors := [.read "A" [.axis i]] }] }, nonlin := .identity },
      .assign "Y" [.free i]
        { body := { terms := [{ factors := [.read "T" [.axis i]] }] }, nonlin := .identity } ] }
  match checkReadRanks rp |>.run 0 with
  | .ok _ _    => pure ()
  | .error e _ => throwError s!"correct-arity intermediate read should pass: {repr e}"

-- checkReadRanks: reading an affine-LHS (→ scatter) producer at its SLOT rank passes the guard —
-- the produced rank of an affine LHS is its slot count (e.g. Out[2*i,2*i] ⇒ 2), not its free-axis
-- count (0). Regression for the guard's scatter-rank handling.
run_cmd do
  let i : AxisSpec := { name := "i", uid := 0, kind := .real none }
  let a : AxisSpec := { name := "a", uid := 2, kind := .real none }
  let b : AxisSpec := { name := "b", uid := 3, kind := .real none }
  let rp : ResolvedProgram := {
    decls := [], env := {}, extNames := insert "X" ∅,
    stmts := [
      .assign "Out" [.affine (.scale 2 i), .affine (.scale 2 i)]
        { body := { terms := [{ factors := [.read "X" [.axis i]] }] }, nonlin := .identity },
      .assign "Z" [.free a, .free b]
        { body := { terms := [{ factors := [.read "Out" [.axis a, .axis b]] }] }, nonlin := .identity } ] }
  match checkReadRanks rp |>.run 0 with
  | .ok _ _    => pure ()
  | .error e _ => throwError s!"reading affine-scatter output at slot rank should pass: {repr e}"

-- checkDtypes: real-kinded axis in iterAt slot ⇒ iterAxisNotNat.
run_cmd do
  let l : AxisSpec := { name := "l", uid := 1, kind := .real none }  -- ← real, not nat
  let s : Stmt := .assign "H" [.iterAt l 0]
    { body := { terms := [] }, nonlin := .identity }
  let rp : ResolvedProgram := { decls := [], env := {}, extNames := ∅, stmts := [s] }
  match checkDtypes rp |>.run 0 with
  | .error (.iterAxisNotNat "l") _ => pure ()
  | .error e _                     => throwError s!"wrong error: {repr e}"
  | .ok _ _                        => throwError "expected iterAxisNotNat for real-kinded iterAt axis"

-- checkDtypes: nat-kinded axis in iterAt slot passes through.
run_cmd do
  let l : AxisSpec := { name := "l", uid := 1, kind := .nat none }
  let s : Stmt := .assign "H" [.iterAt l 0]
    { body := { terms := [] }, nonlin := .identity }
  let rp : ResolvedProgram := { decls := [], env := {}, extNames := ∅, stmts := [s] }
  match checkDtypes rp |>.run 0 with
  | .ok _ _    => pure ()
  | .error e _ => throwError s!"nat iterAt should not error: {repr e}"

-- checkDtypes: nat-kinded axis in freeNorm slot ⇒ normAxisNotReal.
run_cmd do
  let m : AxisSpec := { name := "m", uid := 2, kind := .nat none }  -- ← nat, not real
  let s : Stmt := .assign "S" [.freeNorm m]
    { body := { terms := [] }, nonlin := .softmax none }
  let rp : ResolvedProgram := { decls := [], env := {}, extNames := ∅, stmts := [s] }
  match checkDtypes rp |>.run 0 with
  | .error (.normAxisNotReal "m") _ => pure ()
  | .error e _                      => throwError s!"wrong error: {repr e}"
  | .ok _ _                         => throwError "expected normAxisNotReal for nat-kinded freeNorm axis"

-- checkDtypes: predicate tensor with non-identity nonlin ⇒ predicateNonlin.
run_cmd do
  let ax : AxisSpec := { name := "i", uid := 1, kind := .real none }
  let s : Stmt := .assign "P" [.free ax]
    { body := { terms := [] }, nonlin := .relu }  -- ← relu on a predicate
  let env : DeclEnv := ({} : Std.HashMap String Decl).insert "P" (.predicate "P" [ax])
  let rp : ResolvedProgram := { decls := [], env, extNames := ∅, stmts := [s] }
  match checkDtypes rp |>.run 0 with
  | .error (.predicateNonlin "P") _ => pure ()
  | .error e _                      => throwError s!"wrong error: {repr e}"
  | .ok _ _                         => throwError "expected predicateNonlin for relu on predicate output"

-- checkDtypes: predicate tensor with identity nonlin passes through.
run_cmd do
  let ax : AxisSpec := { name := "i", uid := 1, kind := .real none }
  let s : Stmt := .assign "P" [.free ax]
    { body := { terms := [] }, nonlin := .identity }
  let env : DeclEnv := ({} : Std.HashMap String Decl).insert "P" (.predicate "P" [ax])
  let rp : ResolvedProgram := { decls := [], env, extNames := ∅, stmts := [s] }
  match checkDtypes rp |>.run 0 with
  | .ok _ _    => pure ()
  | .error e _ => throwError s!"predicate with identity should not error: {repr e}"

end LeanNCD
