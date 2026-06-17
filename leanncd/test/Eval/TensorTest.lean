import LeanNCD.Eval.Tensor
namespace LeanNCD.Eval
open DenseTensor
#guard DenseTensor.sizeOf [2,3] == 6
#guard strides [2,3,4] == [12,4,1]
#guard flatIdx [2,3] [1,2] == 5
#guard (zeros [2,3]).data.size == 6
-- set! then get! round-trips:
#guard ((zeros [2,3]).set! [1,2] 7.0).get! [1,2] == 7.0
-- ofFn agrees with get! and is row-major consistent with flatIdx:
#guard (ofFn [2,2] (fun c => Float.ofNat (flatIdx [2,2] c))).get! [1,1] == 3.0
#guard (ofFn [2,3] (fun c => Float.ofNat (c.headD 0))).get! [1,2] == 1.0
-- approxEq: reflexive true; perturbation false:
#guard approxEq (ofFn [3] (fun _ => 1.0)) (ofFn [3] (fun _ => 1.0))
#guard ! approxEq (ofFn [3] (fun _ => 1.0)) ((ofFn [3] (fun _ => 1.0)).set! [0] 2.0)
-- allCoords enumerates the full space in row-major order:
#guard allCoords [2,2] == [[0,0],[0,1],[1,0],[1,1]]
end LeanNCD.Eval
