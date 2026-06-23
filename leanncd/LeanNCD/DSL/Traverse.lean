-- LeanNCD/DSL/Traverse.lean
import LeanNCD.DSL.Ast
import LeanNCD.Exec.Traversable

namespace LeanNCD

/-- Apply a UID remap to a single AxisSpec (name is display-only; preserved). -/
def AxisSpec.mapUID (f : UData → UData) (a : AxisSpec) : AxisSpec :=
  { a with uid := (f ⟨a.uid, some a.name⟩).uid }

instance : TermTraversable AxisSpec where
  traverseUID f a := AxisSpec.mapUID f a

partial def IdxExpr.mapUID (f : UData → UData) : IdxExpr → IdxExpr
  | .axis a       => .axis (AxisSpec.mapUID f a)
  | .const n      => .const n
  | .scale n a    => .scale n (AxisSpec.mapUID f a)
  | .shift a n    => .shift (AxisSpec.mapUID f a) n
  | .affine n xs  => .affine n (xs.map (fun (c, a) => (c, AxisSpec.mapUID f a)))
instance : TermTraversable IdxExpr where traverseUID := IdxExpr.mapUID

partial def PredArith.mapUID (f : UData → UData) : PredArith → PredArith
  | .embed e  => .embed (IdxExpr.mapUID f e)
  | .mul a b  => .mul (PredArith.mapUID f a) (PredArith.mapUID f b)
  | .iabs a   => .iabs (PredArith.mapUID f a)
instance : TermTraversable PredArith where traverseUID := PredArith.mapUID

partial def BoolExpr.mapUID (f : UData → UData) : BoolExpr → BoolExpr
  | .rel op a b => .rel op (PredArith.mapUID f a) (PredArith.mapUID f b)
  | .and a b    => .and (BoolExpr.mapUID f a) (BoolExpr.mapUID f b)
  | .or  a b    => .or  (BoolExpr.mapUID f a) (BoolExpr.mapUID f b)
  | .not a      => .not (BoolExpr.mapUID f a)
  | .ieq a b    => .ieq (PredArith.mapUID f a) (PredArith.mapUID f b)
instance : TermTraversable BoolExpr where traverseUID := BoolExpr.mapUID

def Nonlin.mapUID (f : UData → UData) : Nonlin → Nonlin
  | .identity      => .identity
  | .relu          => .relu
  | .softmax m     => .softmax (m.map (BoolExpr.mapUID f))
  | .normalize m   => .normalize (m.map (BoolExpr.mapUID f))

def Factor.mapUID (f : UData → UData) : Factor → Factor
  | .read nm es => .read nm (es.map (IdxExpr.mapUID f))
  | .iverson b  => .iverson (BoolExpr.mapUID f b)

def ProdTerm.mapUID (f : UData → UData) (p : ProdTerm) : ProdTerm :=
  { factors := p.factors.map (Factor.mapUID f) }
def SumExpr.mapUID (f : UData → UData) (s : SumExpr) : SumExpr :=
  { terms := s.terms.map (ProdTerm.mapUID f) }
def RHSExpr.mapUID (f : UData → UData) (r : RHSExpr) : RHSExpr :=
  { body := SumExpr.mapUID f r.body, nonlin := Nonlin.mapUID f r.nonlin, agg := r.agg }

def LHSSlot.mapUID (f : UData → UData) : LHSSlot → LHSSlot
  | .free a     => .free (AxisSpec.mapUID f a)
  | .freeNorm a => .freeNorm (AxisSpec.mapUID f a)
  | .iterAt a n => .iterAt (AxisSpec.mapUID f a) n
  | .iterNext a => .iterNext (AxisSpec.mapUID f a)
  | .affine e   => .affine (IdxExpr.mapUID f e)

def Decl.mapUID (f : UData → UData) : Decl → Decl
  | .tensor nm ax        => .tensor nm (ax.map (AxisSpec.mapUID f))
  | .predicate nm ax     => .predicate nm (ax.map (AxisSpec.mapUID f))
  | .linear nm ax b      => .linear nm (ax.map (AxisSpec.mapUID f)) b
  | .axis ax n           => .axis (AxisSpec.mapUID f ax) n
instance : TermTraversable Decl where traverseUID := Decl.mapUID

def Stmt.mapUID (f : UData → UData) : Stmt → Stmt
  | .assign nm ls r      => .assign nm (ls.map (LHSSlot.mapUID f)) (RHSExpr.mapUID f r)
  | .scatter nm ls r o   => .scatter nm (ls.map (LHSSlot.mapUID f)) (RHSExpr.mapUID f r) o
  | .recurMorphism nm ax tc => .recurMorphism nm (AxisSpec.mapUID f ax) tc
instance : TermTraversable Stmt where traverseUID := Stmt.mapUID

instance : TermTraversable TLProgram where
  traverseUID f p := { decls := p.decls.map (Decl.mapUID f), stmts := p.stmts.map (Stmt.mapUID f) }

end LeanNCD
