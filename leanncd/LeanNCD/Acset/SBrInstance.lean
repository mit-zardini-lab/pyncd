import LeanNCD.DSL.SizeExpr     -- SizeExpr (axis sizes)

namespace LeanNCD.Acset

/-- UID type tag (Python's RawAxis / NormAxis / NatAxis). -/
inductive AxisType | rawAxis | normAxis | natAxis
  deriving DecidableEq, Repr, Inhabited

/-- A typed axis UID: serialized `f'{type}:{id}'`. -/
structure AxisUID where
  type : AxisType
  id   : Nat
  deriving DecidableEq, Repr, Inhabited

/-- Base operation tag (Python OpTag.value strings). -/
inductive OpTag
  | identity | softmax | maskedSoftmax | maskedNormalize | elementwise
  | normalize | embedding | addition | weightedTriangularLower | linear
  deriving DecidableEq, Repr, Inhabited

/-- Scalar datatype (Python DataTag.value strings). -/
inductive DataTag | reals | natural | bool
  deriving DecidableEq, Repr, Inhabited

structure EquationRow where
  equationIdx : Nat
  lhsName     : Option String          -- DynamicName serialized form, opaque (§13)
  deriving DecidableEq, Repr, Inhabited

structure ArrayRow where
  equationIdx   : Nat
  slot          : Nat
  name          : Option String
  isInput       : Bool
  operatorTag   : Option OpTag
  normAxis      : Option AxisUID
  datatypeTag   : DataTag
  maxValue      : Option SizeExpr
  bias          : Option Bool
  elementwiseFn : Option String
  opPredicate   : Option String
  wireLabel     : Option String
  deriving DecidableEq, Repr, Inhabited

structure ArrayAxisRow where
  equationIdx : Nat
  arraySlot   : Nat
  axisUid     : AxisUID
  isTarget    : Bool
  position    : Nat
  deriving DecidableEq, Repr, Inhabited

structure SampleRow where
  equationIdx    : Nat
  reindexingSlot : Nat
  srcUid         : AxisUID
  tgtUid         : AxisUID
  coeff          : Int
  offset         : Int
  deriving DecidableEq, Repr, Inhabited

/-- The full-fidelity acset presentation of one ∫Dat-morphism (mirrors
    `acset/instances.py:SBrInstance`). Computable: sizes in `SizeExpr`, coeffs in `Int`. -/
structure SBrInstance where
  axisSizes : List (AxisUID × SizeExpr)
  equations : List EquationRow
  arrays    : List ArrayRow
  arrayAxes : List ArrayAxisRow
  samples   : List SampleRow
  deriving DecidableEq, Repr, Inhabited

end LeanNCD.Acset
