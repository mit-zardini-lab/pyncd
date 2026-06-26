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

-- Fixed-axis NAME list of a presentation weave (computable surrogate of weaveToArrayType shape).
def fixedNames (w : WeaveShapeP) : List (Option String) :=
  w.filterMap fun s => match s with | .fixed a => some a.name | .tiled => none

-- After the fix: the H-read input weave of step 1 must have targetAxes = H's output axes [i,k],
-- NOT the degenerate [i,j]. The H wire is routing[1][1] = internal 0 0.
#guard
  let tc := tl!{
    H[i,k] := W1[k,d] · X[i,d]
    Y[i,j] := relu(W2[j,k] · H[i,k])
  }
  fixedNames ((tc.steps.getD 1 default).inputWeaves.getD 1 [])
    == fixedNames ((tc.steps.getD 0 default).outputWeaves.getD 0 [])
end LeanNCD
