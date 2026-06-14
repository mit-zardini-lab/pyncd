import LeanNCD.DSL.Syntax

namespace LeanNCD

-- The categories + rules PARSE (quotation elaborates to Syntax). No elaborator yet.
#check (`(tl_decl| tensor A : (i : ℝ)) : Lean.MacroM _)
#check (`(tl_stmt| Y[i, j] := W[i, k] · X[k, j]) : Lean.MacroM _)
#check (`(tl_bool_expr| s ≤ q) : Lean.MacroM _)
#check (`(tl_rhs| softmax(where s ≤ q)(Q[q, d] · K[s, d])) : Lean.MacroM _)

end LeanNCD
