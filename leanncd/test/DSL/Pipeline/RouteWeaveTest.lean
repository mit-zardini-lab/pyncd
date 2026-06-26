import LeanNCD.DSL.Compile
namespace LeanNCD
-- BEFORE refactor this elaborates against the current `route`; it must still hold AFTER.
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).steps.length == 1
#guard (tl!{ Y[i,j] := W[i,k] · X[k,j] }).routing == [[Wire.external 0, Wire.external 1]]
#guard (tl!{
    H[i,k] := W1[k,d] · X[i,d]
    Y[i,j] := relu(W2[j,k] · H[i,k])
  }).routing == [[Wire.external 0, Wire.external 1],
                 [Wire.external 2, Wire.internal 0 0],
                 [Wire.internal 1 0]]
end LeanNCD
