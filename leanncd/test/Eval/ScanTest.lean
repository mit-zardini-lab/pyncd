import LeanNCD.Eval.Scan
namespace LeanNCD.Eval
open Std
private def ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
private def tensorOf (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩

-- LINEAR scan, single state: S[j,0] := X[j]; S[j,l+1] := S[j,l] · A[j]   (elementwise, no contraction)
--   X = [1, 10] (j=0,1), A = [2, 3], L = 3  ⇒  S[:,0]=[1,10], S[:,1]=[2,30], S[:,2]=[4,90].
run_cmd do
  let j := ax "j" 1; let l := ax "l" 9
  let X := tensorOf [2] [1, 10]; let A := tensorOf [2] [2, 3]
  let env : HashMap String DenseTensor := (({} : HashMap String DenseTensor).insert "X" X).insert "A" A
  let sizes := (({} : HashMap UID Nat).insert 1 2).insert 9 3   -- j↦2, l↦3
  let base : Stmt := .assign "S" [.free j, .iterAt l 0] { body := { terms := [{ factors := [.read "X" [.axis j]] }] }, nonlin := .identity }
  let recur : Stmt := .assign "S" [.free j, .iterNext l] { body := { terms := [{ factors := [.read "S" [.axis j, .axis l], .read "A" [.axis j]] }] }, nonlin := .identity }
  match evalScan env sizes (.scan "S" l [base] [recur] false) with
  | .error e => throwError e
  | .ok outs =>
      match outs.find? (·.1 == "S") with
      | some (_, S) =>
          unless DenseTensor.approxEq S (tensorOf [2,3] [1,2,4, 10,30,90]) do throwError s!"linear scan wrong: {repr S.data}"
      | none => throwError "no S"

-- relu scan, single state: S[j,0]:=X[j]; S[j,l+1] := relu(S[j,l]·A[j])  with A negative ⇒ clamped to 0
--   X=[1], A=[-1], L=2 ⇒ S[:,0]=[1], S[:,1]=relu(1·-1)=relu(-1)=0.
run_cmd do
  let j := ax "j" 1; let l := ax "l" 9
  let X := tensorOf [1] [1]; let A := tensorOf [1] [-1]
  let env : HashMap String DenseTensor := (({} : HashMap String DenseTensor).insert "X" X).insert "A" A
  let sizes := (({} : HashMap UID Nat).insert 1 1).insert 9 2
  let base : Stmt := .assign "S" [.free j, .iterAt l 0] { body := { terms := [{ factors := [.read "X" [.axis j]] }] }, nonlin := .identity }
  let recur : Stmt := .assign "S" [.free j, .iterNext l] { body := { terms := [{ factors := [.read "S" [.axis j, .axis l], .read "A" [.axis j]] }] }, nonlin := .relu }
  match evalScan env sizes (.scan "S" l [base] [recur] false) with
  | .error e => throwError e
  | .ok outs => match outs.find? (·.1 == "S") with
    | some (_, S) => unless DenseTensor.approxEq S (tensorOf [1,2] [1, 0]) do throwError s!"relu scan wrong: {repr S.data}"
    | none => throwError "no S"

-- COUPLED scan: two states G,H sharing l, each recur reading both.
--   G[0]:=1; H[0]:=1; G[l+1]:=G[l]+H[l]; H[l+1]:=G[l]  (Fibonacci-ish), L=4 (scalar, single axis l)
--   G: 1, 1+1=2, 2+1=3, 3+2=5  => [1,2,3,5]
--   H: 1, 1,     2,     3       => [1,1,2,3]
run_cmd do
  let l := ax "l" 9
  let one := tensorOf [] [1]
  let env : HashMap String DenseTensor := (({} : HashMap String DenseTensor).insert "C" one)
  let sizes := (({} : HashMap UID Nat).insert 9 4)
  let baseG : Stmt := .assign "G" [.iterAt l 0] { body := { terms := [{ factors := [.read "C" []] }] }, nonlin := .identity }
  let baseH : Stmt := .assign "H" [.iterAt l 0] { body := { terms := [{ factors := [.read "C" []] }] }, nonlin := .identity }
  let recurG : Stmt := .assign "G" [.iterNext l] { body := { terms := [{ factors := [.read "G" [.axis l]] }, { factors := [.read "H" [.axis l]] }] }, nonlin := .identity }
  let recurH : Stmt := .assign "H" [.iterNext l] { body := { terms := [{ factors := [.read "G" [.axis l]] }] }, nonlin := .identity }
  match evalScan env sizes (.scan "G" l [baseG, baseH] [recurG, recurH] false) with
  | .error e => throwError e
  | .ok outs =>
      match outs.find? (·.1 == "G"), outs.find? (·.1 == "H") with
      | some (_, G), some (_, H) =>
          unless DenseTensor.approxEq G (tensorOf [4] [1,2,3,5]) do throwError s!"coupled G wrong: {repr G.data}"
          unless DenseTensor.approxEq H (tensorOf [4] [1,1,2,3]) do throwError s!"coupled H wrong: {repr H.data}"
      | _, _ => throwError "missing G or H"

-- plain errors (handled by evalScheduled, not evalScan)
run_cmd do
  match evalScan {} {} (.plain (.assign "x" [] { body := { terms := [] }, nonlin := .identity })) with
  | .error _ => pure ()
  | .ok _ => throwError "expected plain to error"

end LeanNCD.Eval
