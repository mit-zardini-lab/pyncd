import LeanNCD.DSL.SizeExpr
import LeanNCD.DSL.Elab
import LeanNCD.DSL.Syntax

namespace LeanNCD

open SizeExpr

-- DecidableEq: structurally distinct size exprs compare unequal; equal ones equal.
#guard decide (SizeExpr.mul (.var "m") (.lit 2) = SizeExpr.mul (.var "m") (.lit 2))
#guard ! decide (SizeExpr.var "m" = SizeExpr.var "n")

-- ToExpr exists (used by the tl! macro).
#check (inferInstance : Lean.ToExpr SizeExpr)

-- DecidableEq for new constructors.
#guard decide (SizeExpr.sub (.var "n") (.lit 1) = SizeExpr.sub (.var "n") (.lit 1))
#guard ! decide (SizeExpr.sub (.var "n") (.lit 1) = SizeExpr.sub (.var "n") (.lit 2))
#guard decide (SizeExpr.div (.var "n") 2 = SizeExpr.div (.var "n") 2)
#guard ! decide (SizeExpr.div (.var "n") 2 = SizeExpr.div (.var "n") 3)

-- eval: basic arithmetic
#guard SizeExpr.eval (fun _ => 0) (.lit 7) = 7
#guard SizeExpr.eval (fun s => if s == "n" then 10 else 0) (.add (.var "n") (.lit 3)) = 13
#guard SizeExpr.eval (fun s => if s == "n" then 10 else 0) (.mul (.var "n") (.lit 3)) = 30

-- eval: sub (saturating Nat.sub)
#guard SizeExpr.eval (fun s => if s == "n" then 5 else 0) (.sub (.var "n") (.lit 2)) = 3
#guard SizeExpr.eval (fun _ => 0) (.sub (.lit 2) (.lit 5)) = 0   -- saturates at 0

-- eval: div (floor division)
#guard SizeExpr.eval (fun s => if s == "n" then 10 else 0) (.div (.var "n") 2) = 5
#guard SizeExpr.eval (fun s => if s == "n" then 11 else 0) (.div (.var "n") 2) = 5  -- floor
#guard SizeExpr.eval (fun s => if s == "n" then 10 else 0) (.div (.var "n") 3) = 3

-- eval: pooling formula (n - k) / s + 1 with n=10, k=3, s=2 → (10-3)/2+1 = 4
#guard SizeExpr.eval (fun s => match s with | "n" => 10 | "k" => 3 | _ => 0)
  (SizeExpr.add (SizeExpr.div (SizeExpr.sub (.var "n") (.var "k")) 2) (.lit 1)) = 4

-- Parse round-trips: tl_size syntax produces the expected SizeExpr trees.
run_cmd do
  let e ← Lean.Elab.Command.liftTermElabM do
    LeanNCD.elabTLSize (← `(tl_size| n - 1))
  unless e == SizeExpr.sub (.var "n") (.lit 1) do
    throwError s!"expected sub(var n, lit 1), got {repr e}"

run_cmd do
  let e ← Lean.Elab.Command.liftTermElabM do
    LeanNCD.elabTLSize (← `(tl_size| n / 2))
  unless e == SizeExpr.div (.var "n") 2 do
    throwError s!"expected div(var n, 2), got {repr e}"

-- Pooling formula: (n - k) / 2 + 1 parses to add(div(sub(n,k), 2), 1)
run_cmd do
  let e ← Lean.Elab.Command.liftTermElabM do
    LeanNCD.elabTLSize (← `(tl_size| (n - k) / 2 + 1))
  unless e == SizeExpr.add (SizeExpr.div (SizeExpr.sub (.var "n") (.var "k")) 2) (.lit 1) do
    throwError s!"unexpected shape: {repr e}"

-- Division by zero guard fires.
run_cmd do
  let ok ← try
    let _ ← Lean.Elab.Command.liftTermElabM do
      LeanNCD.elabTLSize (← `(tl_size| n / 0))
    pure false
  catch _ => pure true
  unless ok do throwError "expected division-by-zero error but got none"

end LeanNCD
