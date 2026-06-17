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

end LeanNCD.Acset
