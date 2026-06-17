import LeanNCD.Acset.Io
namespace LeanNCD.Acset
-- helper to fetch a file's content:
private def fileOf (pairs : List (String × String)) (nm : String) : String :=
  (pairs.find? (·.1 == nm)).map (·.2) |>.getD "<missing>"

-- a small instance: Y[i] := X[i] with a norm axis + identity op + a sample with negative offset.
def inst : SBrInstance :=
  { axisSizes := [(⟨.rawAxis, 0⟩, .lit 5), (⟨.normAxis, 1⟩, .var "2")],
    equations := [{ equationIdx := 0, lhsName := some "Y" }],
    arrays    := [{ equationIdx := 0, slot := 0, name := some "Y", isInput := false,
                    operatorTag := some .maskedSoftmax, normAxis := some ⟨.normAxis,1⟩,
                    datatypeTag := .reals, maxValue := none, bias := none,
                    elementwiseFn := none, opPredicate := some "P", wireLabel := none }],
    arrayAxes := [{ equationIdx := 0, arraySlot := 0, axisUid := ⟨.rawAxis,0⟩,
                    isTarget := true, position := 0 }],
    samples   := [{ equationIdx := 0, reindexingSlot := 1, srcUid := ⟨.rawAxis,0⟩,
                    tgtUid := ⟨.rawAxis,0⟩, coeff := 1, offset := -1 }] }

-- five files in order:
#guard (writeSBr inst).map (·.1) ==
  ["axis_sizes.csv","equations.csv","arrays.csv","array_axes.csv","samples.csv"]
-- arrays.csv header is the EXACT Python column line:
#guard ((fileOf (writeSBr inst) "arrays.csv").splitOn "\r\n")[0]! ==
  "equation_idx,slot,name,is_input,operator_tag,norm_axis,datatype_tag,max_value,bias,elementwise_fn,op_predicate,wire_label"
-- the one arrays data row encodes exactly (note empty fields for none):
#guard ((fileOf (writeSBr inst) "arrays.csv").splitOn "\r\n")[1]! ==
  "0,0,Y,false,masked_softmax,NormAxis:1,reals,,,,P,"
-- samples row with negative offset:
#guard ((fileOf (writeSBr inst) "samples.csv").splitOn "\r\n")[1]! == "0,1,RawAxis:0,RawAxis:0,1,-1"
-- axis_sizes with a FreeNumeric (?2):
#guard ((fileOf (writeSBr inst) "axis_sizes.csv").splitOn "\r\n")[2]! == "NormAxis:1,?2"
-- every file ends with \r\n:
#guard ((writeSBr inst).all (fun p => p.2.endsWith "\r\n"))

-- ── Task 4: round-trip property  readSBr (writeSBr inst) = .ok inst ──

#guard readSBr (writeSBr inst) == .ok inst

-- empty instance round-trips:
def emptyInst : SBrInstance := { axisSizes := [], equations := [], arrays := [], arrayAxes := [], samples := [] }
#guard readSBr (writeSBr emptyInst) == .ok emptyInst

-- a richer instance: two equations, an input + output array, multiple axes/samples, negatives, ?id sizes
def inst2 : SBrInstance :=
  { axisSizes := [(⟨.rawAxis,0⟩, .lit 8), (⟨.natAxis,2⟩, .var "5")],
    equations := [{ equationIdx := 0, lhsName := some "T" }, { equationIdx := 1, lhsName := some "Y" }],
    arrays := [{ equationIdx := 1, slot := 0, name := some "Y", isInput := false,
                 operatorTag := some .linear, normAxis := none, datatypeTag := .natural,
                 maxValue := some (.lit 3), bias := some true, elementwiseFn := some "relu",
                 opPredicate := none, wireLabel := some "w0" },
               { equationIdx := 1, slot := 1, name := some "X", isInput := true,
                 operatorTag := none, normAxis := none, datatypeTag := .reals,
                 maxValue := none, bias := none, elementwiseFn := none, opPredicate := none, wireLabel := none }],
    arrayAxes := [{ equationIdx := 1, arraySlot := 0, axisUid := ⟨.rawAxis,0⟩, isTarget := true, position := 0 },
                  { equationIdx := 1, arraySlot := 1, axisUid := ⟨.natAxis,2⟩, isTarget := false, position := 1 }],
    samples := [{ equationIdx := 1, reindexingSlot := 1, srcUid := ⟨.rawAxis,0⟩, tgtUid := ⟨.natAxis,2⟩, coeff := 2, offset := -3 }] }
#guard readSBr (writeSBr inst2) == .ok inst2

-- a missing file errors:
#guard (readSBr [("axis_sizes.csv","axis_uid,size\r\n")]).toOption == none
end LeanNCD.Acset
