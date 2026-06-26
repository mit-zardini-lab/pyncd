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
      let stmts := lp.stmts.filterMap (fun | .plain s => some s | .scan .. => none | .scanPre .. => none)
      -- exactly two stmts: a linear (identity) and a softmax-carrying step
      unless stmts.length == 2 do throwError s!"expected 2 stmts after split, got {stmts.length}"
      let nlins := stmts.map (fun | .assign _ _ r => r.nonlin | .scatter _ _ r _ => r.nonlin | .recurMorphism .. => .identity)
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

-- Matmul Y[i,j] := W[i,k]·X[k,j]: W,X external ⇒ nExternal=2; one step; k contracted ⇒
-- exactly one .tiled slot in the output weave; W,X reindexings present.
run_cmd do
  let ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
  let i := ax "i" 1; let j := ax "j" 2; let k := ax "k" 3
  let mm : Stmt := .assign "Y" [ .free i, .free j ]
    { body := { terms := [ { factors := [ .read "W" [.axis i, .axis k], .read "X" [.axis k, .axis j] ] } ] },
      nonlin := .identity }
  let sp : ScheduledProgram := { decls := [], stmts := [ .plain mm ], env := {}, extNames := (insert "W" (insert "X" (∅ : Finset String))), ctx := { classes := [] }, explicitSizes := {} }
  match route sp |>.run 0 with
  | .ok tc _ =>
      unless tc.nExternal == 2 do throwError s!"nExternal should be 2, got {tc.nExternal}"
      unless tc.steps.length == 1 do throwError s!"expected 1 step, got {tc.steps.length}"
      let step := tc.steps.head!
      -- output weave: i,j fixed, k tiled ⇒ exactly one tiled
      let nTiled := step.outputWeaves.head!.filter (fun | .tiled => true | .fixed _ => false) |>.length
      unless nTiled == 1 do throwError s!"expected exactly one contracted (tiled) axis, got {nTiled}"
      unless step.reindexings.length == 2 do throwError s!"expected 2 input reindexings, got {step.reindexings.length}"
      -- both inputs route to externals
      unless tc.routing.head!.all (fun w => match w with | .external _ => true | .internal .. => false) do throwError "matmul inputs should be external"
  | .error e _ => throwError s!"route errored: {repr e}"

-- Strided read: Y[i] := X[2*i] ⇒ the X reindexing row has coefficient 2.
run_cmd do
  let ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
  let i := ax "i" 1
  let conv : Stmt := .assign "Y" [ .free i ]
    { body := { terms := [ { factors := [ .read "X" [ .scale 2 i ] ] } ] }, nonlin := .identity }
  let sp : ScheduledProgram := { decls := [], stmts := [ .plain conv ], env := {}, extNames := (insert "X" (∅ : Finset String)), ctx := { classes := [] }, explicitSizes := {} }
  match route sp |>.run 0 with
  | .ok tc _ =>
      let sm := tc.steps.head!.reindexings.head!
      unless sm.coeffs == [[2]] do throwError s!"expected strided coeff [[2]], got {sm.coeffs}"
  | .error e _ => throwError s!"route errored: {repr e}"

-- Unresolved read: `Ghost` is neither produced by a step nor declared external ⇒ `route` must
-- FAIL LOUD with `undeclaredName`, not silently route it to external slot 0 (the former
-- `(extIndex …).getD 0` fallback, which masked upstream dataflow errors).
run_cmd do
  let ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
  let i := ax "i" 1
  let s : Stmt := .assign "Y" [ .free i ]
    { body := { terms := [ { factors := [ .read "Ghost" [ .axis i ] ] } ] }, nonlin := .identity }
  let sp : ScheduledProgram := { decls := [], stmts := [ .plain s ], env := {}, extNames := (∅ : Finset String), ctx := { classes := [] }, explicitSizes := {} }
  match route sp |>.run 0 with
  | .ok tc _ => throwError s!"expected route to reject unresolved read, got {tc.steps.length} step(s)"
  | .error (.undeclaredName nm) _ => unless nm == "Ghost" do throwError s!"wrong undeclared name: {nm}"
  | .error e _ => throwError s!"expected undeclaredName, got {repr e}"
end LeanNCD
