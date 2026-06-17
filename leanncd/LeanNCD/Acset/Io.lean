import LeanNCD.Acset.Csv

/-!
# Serialize an `SBrInstance` to the five acset CSV tables (Milestone F)

`writeSBr` mirrors Python `acset/csv_io.py`: five `(filename, csv-text)` pairs in a fixed order,
columns in the EXACT Python field order, every row `\r\n`-terminated (via `renderTable`).
Fully executable, zero `sorry`.
-/

namespace LeanNCD.Acset

/-- Encode an `Option SizeExpr` column ("" when none; sizes here are `.lit`/`.var` so total). -/
private def encSizeOpt (m : Option SizeExpr) : String :=
  match m with | none => "" | some s => (encodeSize s).toOption.getD ""

private def axisSizesRows (inst : SBrInstance) : List (List String) :=
  inst.axisSizes.map (fun (u, sz) => [encodeUID u, (encodeSize sz).toOption.getD ""])

private def equationRows (inst : SBrInstance) : List (List String) :=
  inst.equations.map (fun e => [toString e.equationIdx, encodeName e.lhsName])

private def arrayRows (inst : SBrInstance) : List (List String) :=
  inst.arrays.map (fun a =>
    [ toString a.equationIdx, toString a.slot, encodeName a.name, encodeReqBool a.isInput,
      encodeOpTagOpt a.operatorTag, (a.normAxis.map encodeUID).getD "", encodeDataTag a.datatypeTag,
      encSizeOpt a.maxValue, encodeBoolOpt a.bias, encodeName a.elementwiseFn,
      encodeName a.opPredicate, encodeName a.wireLabel ])

private def arrayAxisRows (inst : SBrInstance) : List (List String) :=
  inst.arrayAxes.map (fun aa =>
    [ toString aa.equationIdx, toString aa.arraySlot, encodeUID aa.axisUid,
      encodeReqBool aa.isTarget, toString aa.position ])

private def sampleRows (inst : SBrInstance) : List (List String) :=
  inst.samples.map (fun s =>
    [ toString s.equationIdx, toString s.reindexingSlot, encodeUID s.srcUid, encodeUID s.tgtUid,
      encodeInt s.coeff, encodeInt s.offset ])

/-- Serialize an `SBrInstance` to the five `(filename, content)` pairs, in Python order. -/
def writeSBr (inst : SBrInstance) : List (String × String) :=
  [ ("axis_sizes.csv", renderTable ["axis_uid", "size"] (axisSizesRows inst)),
    ("equations.csv",  renderTable ["equation_idx", "lhs_name"] (equationRows inst)),
    ("arrays.csv",     renderTable
        ["equation_idx","slot","name","is_input","operator_tag","norm_axis",
         "datatype_tag","max_value","bias","elementwise_fn","op_predicate","wire_label"]
        (arrayRows inst)),
    ("array_axes.csv", renderTable
        ["equation_idx","array_slot","axis_uid","is_target","position"] (arrayAxisRows inst)),
    ("samples.csv",    renderTable
        ["equation_idx","reindexing_slot","src_uid","tgt_uid","coeff","offset"] (sampleRows inst)) ]

/-!
## Deserialize the five CSV tables back into an `SBrInstance` (Task 4)

`readSBr` is the inverse of `writeSBr`: it parses each table, drops the header, and decodes
each data row. The CORE property is `readSBr (writeSBr inst) = .ok inst`. Fully executable.
-/

/-- Fetch a file's content from the pairs, or error if absent. -/
private def lookup (pairs : List (String × String)) (nm : String) : Except CsvError String :=
  match pairs.find? (·.1 == nm) with
  | some (_, c) => .ok c
  | none        => .error s!"missing file: {nm}"

/-- Decode a plain (header-less) Nat field. -/
private def decodeNat (s : String) : Except CsvError Nat :=
  match s.toNat? with
  | some n => .ok n
  | none   => .error s!"expected Nat, got '{s}'"

/-- `"" ⇒ none`; else `some (← decodeUID s)`. -/
private def decodeUidOpt (s : String) : Except CsvError (Option AxisUID) :=
  if s.isEmpty then .ok none else (some <$> decodeUID s)

/-- `"" ⇒ none`; else `some (← decodeSize s)`. -/
private def decodeSizeOpt (s : String) : Except CsvError (Option SizeExpr) :=
  if s.isEmpty then .ok none else (some <$> decodeSize s)

/-- Data rows of a CSV table (header dropped). -/
private def dataRows (content : String) : List (List String) := (parseTable content).drop 1

private def rowToAxisSize : List String → Except CsvError (AxisUID × SizeExpr)
  | [u, sz] => do return (← decodeUID u, ← decodeSize sz)
  | r => .error s!"axis_sizes.csv: expected 2 fields, got {r.length}"

private def rowToEquation : List String → Except CsvError EquationRow
  | [eq, lhs] => do return { equationIdx := ← decodeNat eq, lhsName := decodeName lhs }
  | r => .error s!"equations.csv: expected 2 fields, got {r.length}"

private def rowToArray : List String → Except CsvError ArrayRow
  | [eq, slot, name, isInp, op, norm, dt, mx, bias, efn, opred, wlab] => do
      return { equationIdx := ← decodeNat eq, slot := ← decodeNat slot, name := decodeName name,
               isInput := decodeReqBool isInp, operatorTag := ← decodeOpTagOpt op,
               normAxis := ← decodeUidOpt norm, datatypeTag := ← decodeDataTag dt,
               maxValue := ← decodeSizeOpt mx, bias := decodeBoolOpt bias,
               elementwiseFn := decodeName efn, opPredicate := decodeName opred,
               wireLabel := decodeName wlab }
  | r => .error s!"arrays.csv: expected 12 fields, got {r.length}"

private def rowToArrayAxis : List String → Except CsvError ArrayAxisRow
  | [eq, slot, uid, isTgt, pos] => do
      return { equationIdx := ← decodeNat eq, arraySlot := ← decodeNat slot,
               axisUid := ← decodeUID uid, isTarget := decodeReqBool isTgt,
               position := ← decodeNat pos }
  | r => .error s!"array_axes.csv: expected 5 fields, got {r.length}"

private def rowToSample : List String → Except CsvError SampleRow
  | [eq, slot, src, tgt, coeff, offset] => do
      return { equationIdx := ← decodeNat eq, reindexingSlot := ← decodeNat slot,
               srcUid := ← decodeUID src, tgtUid := ← decodeUID tgt,
               coeff := ← decodeInt coeff, offset := ← decodeInt offset }
  | r => .error s!"samples.csv: expected 6 fields, got {r.length}"

/-- Parse the five acset CSV tables back into an `SBrInstance` (inverse of `writeSBr`). -/
def readSBr (pairs : List (String × String)) : Except CsvError SBrInstance := do
  let axisSizes ← (dataRows (← lookup pairs "axis_sizes.csv")).mapM rowToAxisSize
  let equations ← (dataRows (← lookup pairs "equations.csv")).mapM rowToEquation
  let arrays    ← (dataRows (← lookup pairs "arrays.csv")).mapM rowToArray
  let arrayAxes ← (dataRows (← lookup pairs "array_axes.csv")).mapM rowToArrayAxis
  let samples   ← (dataRows (← lookup pairs "samples.csv")).mapM rowToSample
  return { axisSizes, equations, arrays, arrayAxes, samples }

end LeanNCD.Acset
