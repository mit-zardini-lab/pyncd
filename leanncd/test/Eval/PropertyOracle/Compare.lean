import LeanNCD.Eval.Eval
import LeanNCD.DSL.Ast

namespace LeanNCD.PropertyOracle
open LeanNCD.Eval

/-- Exact tensor equality: same shape and bit-equal Float data. -/
def denseEq (x y : DenseTensor) : Bool :=
  x.shape == y.shape && x.data == y.data

/-- The tensor names a program produces (LHS of every `.assign`), in order, deduped. -/
def producedNames (p : LeanNCD.TLProgram) : List String :=
  (p.stmts.filterMap (fun | LeanNCD.Stmt.assign nm _ _ => some nm | _ => none)).eraseDups

/-- Two eval results agree on a set of names: both `.ok` with `denseEq`-equal tensors at every
    name (extra keys — e.g. materialization intermediates — are ignored), or the same `.error`. -/
def evalAgreesOn (names : List String)
    (a b : Except String (Std.HashMap String DenseTensor)) : Bool :=
  match a, b with
  | .ok ea, .ok eb =>
      names.all (fun n => match ea[n]?, eb[n]? with
        | some ta, some tb => denseEq ta tb
        | _, _ => false)
  | .error sa, .error sb => sa == sb
  | _, _ => false

-- TESTS (fire on build):
private def t2 : DenseTensor := ⟨[2], #[1.0, 2.0]⟩
private def t2' : DenseTensor := ⟨[2], #[1.0, 2.0]⟩
private def t2diff : DenseTensor := ⟨[2], #[1.0, 9.0]⟩
private def t3 : DenseTensor := ⟨[3], #[1.0, 2.0, 3.0]⟩
#guard denseEq t2 t2'            -- equal
#guard ! denseEq t2 t2diff       -- differing data
#guard ! denseEq t2 t3           -- differing shape
#guard evalAgreesOn ["Y"] (.ok (({} : Std.HashMap String DenseTensor).insert "Y" t2)) (.ok (({} : Std.HashMap String DenseTensor).insert "Y" t2'))
#guard ! evalAgreesOn ["Y"] (.ok (({} : Std.HashMap String DenseTensor).insert "Y" t2)) (.ok (({} : Std.HashMap String DenseTensor).insert "Y" t2diff))
-- extra keys ignored (materialization adds intermediates):
#guard evalAgreesOn ["Y"]
  (.ok (({} : Std.HashMap String DenseTensor).insert "Y" t2))
  (.ok (((({} : Std.HashMap String DenseTensor).insert "Y" t2').insert "T1" t3)))
#guard ! evalAgreesOn ["Y"] (.ok (({} : Std.HashMap String DenseTensor).insert "Y" t2)) (.error "boom")
#guard evalAgreesOn [] (.error "e") (.error "e")

end LeanNCD.PropertyOracle
