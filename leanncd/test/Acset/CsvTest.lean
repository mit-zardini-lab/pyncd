import LeanNCD.Acset.Csv
namespace LeanNCD.Acset

-- table round-trip:
#guard parseTable (renderTable ["a","b"] [["1","2"],["3","4"]]) == [["a","b"],["1","2"],["3","4"]]
#guard renderTable ["x"] [] == "x\r\n"
#guard renderTable ["a","b"] [["1","2"]] == "a,b\r\n1,2\r\n"

-- AxisType:
#guard encodeAxisType .normAxis == "NormAxis"
#guard decodeAxisType "NatAxis" == AxisType.natAxis
#guard decodeAxisType "bogus" == AxisType.rawAxis

-- UID:
#guard encodeUID ⟨.normAxis, 3⟩ == "NormAxis:3"
#guard encodeUID ⟨.natAxis, 0⟩ == "NatAxis:0"
#guard decodeUID "NormAxis:3" == .ok (⟨.normAxis, 3⟩ : AxisUID)
#guard decodeUID "7" == .ok (⟨.rawAxis, 7⟩ : AxisUID)   -- untagged ⇒ RawAxis
#guard (decodeUID "RawAxis:abc").toOption == none       -- non-Nat id ⇒ error

-- size:
#guard encodeSize (.lit 5) == .ok "5"
#guard encodeSize (.var "2") == .ok "?2"
#guard (encodeSize (.add (.lit 1) (.lit 2))).toOption == none   -- compound never serializes
#guard decodeSize "?2" == .ok (SizeExpr.var "2")
#guard decodeSize "5" == .ok (SizeExpr.lit 5)
#guard (decodeSize "?xyz") == .ok (SizeExpr.var "xyz")
#guard (decodeSize "nope").toOption == none             -- non-Nat literal ⇒ error

-- int (negatives):
#guard encodeInt (-1) == "-1"
#guard encodeInt 42 == "42"
#guard decodeInt "-1" == .ok (-1 : Int)
#guard decodeInt "42" == .ok (42 : Int)
#guard (decodeInt "x").toOption == none

-- bool:
#guard encodeBoolOpt none == "" ∧ encodeBoolOpt (some true) == "true" ∧ encodeBoolOpt (some false) == "false"
#guard decodeBoolOpt "" == none ∧ decodeBoolOpt "true" == some true ∧ decodeBoolOpt "false" == some false
#guard encodeReqBool true == "true" ∧ encodeReqBool false == "false"
#guard decodeReqBool "true" == true ∧ decodeReqBool "" == false ∧ decodeReqBool "false" == false

-- name / op-string:
#guard encodeName none == "" ∧ encodeName (some "Y") == "Y"
#guard decodeName "" == none ∧ decodeName "Y" == some "Y"

-- optag (incl. multi-word value) + error on unknown:
#guard encodeOpTag .identity == "identity"
#guard encodeOpTag .softmax == "softmax"
#guard encodeOpTag .maskedSoftmax == "masked_softmax"
#guard encodeOpTag .maskedNormalize == "masked_normalize"
#guard encodeOpTag .elementwise == "elementwise"
#guard encodeOpTag .normalize == "normalize"
#guard encodeOpTag .embedding == "embedding"
#guard encodeOpTag .addition == "addition"
#guard encodeOpTag .weightedTriangularLower == "weighted_triangular_lower"
#guard encodeOpTag .linear == "linear"
#guard encodeOpTagOpt none == ""
#guard encodeOpTagOpt (some .maskedSoftmax) == "masked_softmax"
#guard encodeOpTagOpt (some .weightedTriangularLower) == "weighted_triangular_lower"
#guard decodeOpTagOpt "masked_softmax" == .ok (some OpTag.maskedSoftmax)
#guard decodeOpTagOpt "weighted_triangular_lower" == .ok (some OpTag.weightedTriangularLower)
#guard decodeOpTagOpt "" == .ok (none : Option OpTag)
#guard (decodeOpTagOpt "bogus").toOption == none

-- datatag:
#guard encodeDataTag .reals == "reals" ∧ encodeDataTag .natural == "natural" ∧ encodeDataTag .bool == "bool"
#guard decodeDataTag "reals" == .ok DataTag.reals
#guard decodeDataTag "natural" == .ok DataTag.natural
#guard decodeDataTag "bool" == .ok DataTag.bool
#guard (decodeDataTag "bogus").toOption == none

end LeanNCD.Acset
