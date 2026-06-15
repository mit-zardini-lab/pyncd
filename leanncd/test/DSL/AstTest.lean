import LeanNCD.DSL.Ast

namespace LeanNCD

-- Build the matmul AST by hand: Y[i,j] := W[i,k] · X[k,j] (k contracted).
private def axI : AxisSpec := ⟨"i", 0, .real none⟩
private def axJ : AxisSpec := ⟨"j", 0, .real none⟩
private def axK : AxisSpec := ⟨"k", 0, .real none⟩
private def matmul : TLProgram :=
  { decls := []
    stmts := [ .assign "Y" [.free axI, .free axJ]
                 { body := { terms := [ { factors :=
                      [ .read "W" [.axis axI, .axis axK], .read "X" [.axis axK, .axis axJ] ] } ] }
                   nonlin := .identity } ] }

#guard matmul.stmts.length == 1
#guard decide (matmul = matmul)                       -- DecidableEq over the whole AST
#check (inferInstance : Lean.ToExpr TLProgram)         -- ToExpr on the whole AST

end LeanNCD
