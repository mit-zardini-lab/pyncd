import LeanNCD.DSL.Pipeline.Lowering
namespace LeanNCD
-- Masked attention: A[q,s] := softmax(where s≤q)(Q[q,d]·K[s,d]) splits into a linear step
-- (identity nonlin) + a softmax step (carries the mask).
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 1, kind := .real none }
  let mask : BoolExpr := .rel .le (.embed (.axis (ax "s"))) (.embed (.axis (ax "q")))
  let attn : Stmt := .assign "A" [ .free (ax "q"), .free (ax "s") ]
    { body := { terms := [ { factors := [ .read "Q" [.axis (ax "q"), .axis (ax "d")],
                                            .read "K" [.axis (ax "s"), .axis (ax "d")] ] } ] },
      nonlin := .softmax (some mask) }
  let sp : ScanProgram :=
    { decls := [], stmts := [ .plain attn ], env := {}, extNames := ∅, ctx := { classes := [] } }
  match splitNonlins sp |>.run 0 with
  | .ok lp _ =>
      let stmts := lp.stmts.filterMap (fun | .plain s => some s | .scan .. => none)
      -- exactly two stmts: a linear (identity) and a softmax-carrying step
      unless stmts.length == 2 do throwError s!"expected 2 stmts after split, got {stmts.length}"
      let nlins := stmts.map (fun | .assign _ _ r => r.nonlin | .scatter _ _ r _ => r.nonlin)
      unless nlins.any (· == .identity) do throwError "missing linear (identity) step"
      unless nlins.any (fun n => match n with | .softmax (some _) => true | _ => false) do
        throwError "missing masked-softmax step"
  | .error e _ => throwError s!"splitNonlins errored: {repr e}"

-- A plain identity assign is unchanged (one stmt out).
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 1, kind := .real none }
  let mm : Stmt := .assign "Y" [ .free (ax "i") ]
    { body := { terms := [ { factors := [ .read "X" [.axis (ax "i")] ] } ] }, nonlin := .identity }
  let sp : ScanProgram :=
    { decls := [], stmts := [ .plain mm ], env := {}, extNames := ∅, ctx := { classes := [] } }
  match splitNonlins sp |>.run 0 with
  | .ok lp _ => unless lp.stmts.length == 1 do throwError "identity assign should not split"
  | .error e _ => throwError s!"errored: {repr e}"

-- DCE: `Dead` is computed but never read and is not the output ⇒ dropped. `Y` (output)
-- and its dependency `T` survive.
run_cmd do
  let ax (nm : String) : AxisSpec := { name := nm, uid := 1, kind := .real none }
  let rd (nm : String) : RHSExpr :=
    { body := { terms := [ { factors := [ .read nm [ .axis (ax "i") ] ] } ] }, nonlin := .identity }
  let tStmt   : ScanStmt := .plain (.assign "T" [ .free (ax "i") ] (rd "X"))      -- T := X
  let deadS   : ScanStmt := .plain (.assign "Dead" [ .free (ax "i") ] (rd "X"))   -- Dead := X (unused)
  let yStmt   : ScanStmt := .plain (.assign "Y" [ .free (ax "i") ] (rd "T"))      -- Y := T (output)
  let lp : LinearProgram := { decls := [], stmts := [tStmt, deadS, yStmt], env := {}, extNames := ∅, ctx := { classes := [] } }
  match schedule lp |>.run 0 with
  | .ok sp _ =>
      let names := sp.stmts.flatMap ScanStmt.writes
      unless names.contains "Y" && names.contains "T" do throwError s!"Y,T must survive; got {names}"
      unless ¬ names.contains "Dead" do throwError s!"Dead must be eliminated; got {names}"
  | .error e _ => throwError s!"schedule errored: {repr e}"
end LeanNCD
