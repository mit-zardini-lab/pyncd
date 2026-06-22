import LeanNCD.Acset.SBrInstance

/-!
# CSV codec for acset interop (Milestone F)

Byte-faithful with Python `acset/csv_io.py`. The acset CSV data contains **no embedded
commas or quotes** — UIDs are `Type:id`, sizes are ints or `?id`, enum values and names are
identifiers — so split-on-comma with NO quoting/unquoting is a faithful inverse of Python's
`csv.writer`/`csv.DictReader` for this data. This assumption is documented and relied upon.
Python `csv` terminates every row (including the last) with `\r\n`; we match that exactly.
-/

namespace LeanNCD.Acset

abbrev CsvError := String

/-- Python `csv` terminates every row (incl. the last) with `\r\n`. Match it byte-for-byte. -/
def crlf : String := "\r\n"

def renderRow (fields : List String) : String := String.intercalate "," fields

/-- header + data rows, each followed by `\r\n`. -/
def renderTable (header : List String) (rows : List (List String)) : String :=
  String.join ((header :: rows).map (fun r => renderRow r ++ crlf))

/-- Split into rows on `\n` (dropping a trailing `\r` and empty trailing line), then fields on `,`.
    No unquoting — the acset data contains no commas/quotes (documented assumption). -/
def parseTable (s : String) : List (List String) :=
  (s.splitOn "\n").filterMap (fun line =>
    let line := if line.endsWith "\r" then (line.dropEnd 1).toString else line
    if line.isEmpty then none else some (line.splitOn ","))

-- ── field codecs ──

def encodeAxisType : AxisType → String
  | .rawAxis => "RawAxis"
  | .normAxis => "NormAxis"
  | .natAxis => "NatAxis"

/-- Map tag string → `AxisType`; unknown/absent ⇒ `rawAxis` (Python `_UID_TYPE_BY_NAME.get`). -/
def decodeAxisType (s : String) : AxisType :=
  if s == "NormAxis" then .normAxis
  else if s == "NatAxis" then .natAxis
  else .rawAxis

def encodeUID (u : AxisUID) : String := encodeAxisType u.type ++ ":" ++ toString u.id

/-- Split on the (single) `:`; parse the id with `toNat?`. No `:` ⇒ untagged ⇒ `rawAxis`. -/
def decodeUID (s : String) : Except CsvError AxisUID :=
  let mk (ty : AxisType) (idStr : String) : Except CsvError AxisUID :=
    match idStr.toNat? with
    | some n => .ok ⟨ty, n⟩
    | none   => .error s!"decodeUID: bad id {idStr}"
  match s.splitOn ":" with
  | [idStr]      => mk .rawAxis idStr            -- backwards-compat untagged ⇒ RawAxis
  | [tag, idStr] => mk (decodeAxisType tag) idStr
  | _            => .error s!"decodeUID: malformed {s}"

def encodeInt (n : Int) : String := toString n

def decodeInt (s : String) : Except CsvError Int :=
  match s.toInt? with
  | some n => .ok n
  | none   => .error s!"decodeInt: bad int {s}"

/-- `.lit n ⇒ toString n`; `.var v ⇒ "?"++v`; compound (`.add`/`.mul`) never serializes. -/
def encodeSize : SizeExpr → Except CsvError String
  | .lit n => .ok (toString n)
  | .var v => .ok ("?" ++ v)
  | .add _ _ => .error "encodeSize: compound SizeExpr (add) is not serializable"
  | .sub _ _ => .error "encodeSize: compound SizeExpr (sub) is not serializable"
  | .mul _ _ => .error "encodeSize: compound SizeExpr (mul) is not serializable"
  | .div _ _ => .error "encodeSize: compound SizeExpr (div) is not serializable"

/-- Leading `?` ⇒ `.var` (free numeric); else `.lit` of the parsed Nat. -/
def decodeSize (s : String) : Except CsvError SizeExpr :=
  if s.startsWith "?" then
    .ok (.var (s.drop 1).toString)
  else
    match s.toNat? with
    | some n => .ok (.lit n)
    | none   => .error s!"decodeSize: bad literal {s}"

def encodeName : Option String → String
  | none => ""
  | some s => s

def decodeName (s : String) : Option String :=
  if s.isEmpty then none else some s

def encodeBoolOpt : Option Bool → String
  | none => ""
  | some true => "true"
  | some false => "false"

def decodeBoolOpt (s : String) : Option Bool :=
  if s.isEmpty then none
  else if s == "true" then some true
  else some false

def encodeReqBool : Bool → String
  | true => "true"
  | false => "false"

def decodeReqBool (s : String) : Bool := s == "true"

def encodeOpTag : OpTag → String
  | .identity => "identity"
  | .softmax => "softmax"
  | .maskedSoftmax => "masked_softmax"
  | .maskedNormalize => "masked_normalize"
  | .elementwise => "elementwise"
  | .normalize => "normalize"
  | .embedding => "embedding"
  | .addition => "addition"
  | .weightedTriangularLower => "weighted_triangular_lower"
  | .linear => "linear"

def encodeOpTagOpt : Option OpTag → String
  | none => ""
  | some t => encodeOpTag t

/-- `"" ⇒ none`; a known value ⇒ `some tag`; unknown nonempty value ⇒ error (Python `OpTag(s)`). -/
def decodeOpTagOpt (s : String) : Except CsvError (Option OpTag) :=
  if s.isEmpty then .ok none
  else if s == "identity" then .ok (some .identity)
  else if s == "softmax" then .ok (some .softmax)
  else if s == "masked_softmax" then .ok (some .maskedSoftmax)
  else if s == "masked_normalize" then .ok (some .maskedNormalize)
  else if s == "elementwise" then .ok (some .elementwise)
  else if s == "normalize" then .ok (some .normalize)
  else if s == "embedding" then .ok (some .embedding)
  else if s == "addition" then .ok (some .addition)
  else if s == "weighted_triangular_lower" then .ok (some .weightedTriangularLower)
  else if s == "linear" then .ok (some .linear)
  else .error s!"decodeOpTagOpt: unknown OpTag value {s}"

def encodeDataTag : DataTag → String
  | .reals => "reals"
  | .natural => "natural"
  | .bool => "bool"

/-- known value ⇒ ok; else error (Python `DataTag(s)`). -/
def decodeDataTag (s : String) : Except CsvError DataTag :=
  if s == "reals" then .ok .reals
  else if s == "natural" then .ok .natural
  else if s == "bool" then .ok .bool
  else .error s!"decodeDataTag: unknown DataTag value {s}"

end LeanNCD.Acset
