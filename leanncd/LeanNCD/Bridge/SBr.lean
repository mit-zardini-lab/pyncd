import LeanNCD.Base.Br          -- BrObj, BrMorph
import LeanNCD.Base.Numeric     -- Numeric
import LeanNCD.Exec.Uid         -- UID

namespace LeanNCD

/-- §8.1 acset row types — a SELF-CONTAINED minimal version (acset.md has the full schemas;
    this is sufficient to state realization + the §8 agreement). -/
structure EquationRow where
  op : String
  outputArray : Nat
  deriving Repr, DecidableEq

structure ArrayRow where
  slot : Nat
  isInput : Bool
  datatypeTag : String
  opPredicate : Bool
  deriving Repr, DecidableEq

structure ArrayAxisRow where
  array : Nat
  axisUID : UID
  isTarget : Bool
  position : Nat
  deriving Repr, DecidableEq

structure SampleRow where
  src : UID
  tgt : UID
  coeff : Int
  offset : Int
  deriving Repr, DecidableEq

/-- The four-table acset presentation of one `∫Dat`-morphism (§8.1). `axisSizes` is the
    `Dat`-part (each axis-UID's symbolic size); the other four tables are the `C♯` connectivity.
    Minimal/self-contained; see acset.md for the complete row schemas. -/
structure SBrInstance where
  equations  : List EquationRow
  arrays     : List ArrayRow
  arrayAxes  : List ArrayAxisRow
  samples    : List SampleRow
  axisSizes  : List (UID × Numeric)

/-- Realize an `SBrInstance` as a `Br` morphism (the CSV path). OBLIGATION (`sorry`):
    reconstructing the routed diagram from the four tables rests on the same `Br` glue
    (`Br.tensorHom`/`Br.swap`, Milestone-B+ sorries) as the DSL-path `realize`. -/
noncomputable def realizeSBr (s : SBrInstance) : Σ (dom cod : BrObj), BrMorph dom cod := sorry

end LeanNCD
