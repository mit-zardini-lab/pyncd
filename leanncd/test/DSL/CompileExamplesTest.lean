import LeanNCD.DSL.Compile
namespace LeanNCD
-- Matmul compiles to a ThreadedComposed with 2 external inputs (W, X) and ≥1 step.
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).nExternal == 2
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).steps.length ≥ 1
end LeanNCD
