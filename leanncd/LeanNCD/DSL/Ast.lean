import LeanNCD.DSL.SizeExpr
import LeanNCD.Exec.Uid   -- reuse the canonical `UID := Nat`; do NOT redefine it (duplicate-def error)
import LeanNCD.DSL.Target

namespace LeanNCD

inductive AxisKind
  | real : Option SizeExpr → AxisKind
  | nat  : Option SizeExpr → AxisKind
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

structure AxisSpec where
  name : String
  uid  : UID
  kind : AxisKind
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

inductive Decl
  | tensor    : String → List AxisSpec → Decl
  | predicate : String → List AxisSpec → Decl
  | linear    : String → List AxisSpec → (bias : Bool) → Decl
  | axis      : AxisSpec → Option Nat → Decl   -- `axis l : ℕ = 3`: declares an axis's dtype + optional pinned size
  deriving DecidableEq, Repr, Lean.ToExpr

inductive IdxExpr
  | axis   : AxisSpec → IdxExpr
  | const  : Int → IdxExpr
  | scale  : Int → AxisSpec → IdxExpr
  | shift  : AxisSpec → Int → IdxExpr
  | affine : Int → List (Int × AxisSpec) → IdxExpr
  deriving DecidableEq, Repr, Lean.ToExpr

/-- Normalize an `IdxExpr` to canonical integer-affine form `(c0, [(coef, uid)])`. The single
    affine-lowering primitive shared by the compile path (`idxToRow`, dense over a degree basis) and
    the eval size solver (`Eval/Shape.lean`, sparse) — see `idxDensify`. -/
def idxAffineForm : IdxExpr → Int × List (Int × UID)
  | .axis a      => (0, [(1, a.uid)])
  | .const n     => (n, [])
  | .scale c a   => (0, [(c, a.uid)])
  | .shift a n   => (n, [(1, a.uid)])
  | .affine n xs => (n, xs.map (fun (c, a) => (c, a.uid)))

/-- Densify a sparse affine coefficient list over a degree basis `us` (column order = `us`): the
    `u`-th entry is the sum of the coefficients whose uid is `u`. The bridge from `idxAffineForm`
    (basis-free) to the dense row `idxToRow` needs. -/
def idxDensify (cf : List (Int × UID)) (us : List UID) : List Int :=
  us.map (fun u => cf.foldl (fun acc p => if p.2 == u then acc + p.1 else acc) 0)

inductive PredArith
  | embed : IdxExpr → PredArith
  | mul   : PredArith → PredArith → PredArith
  | iabs  : PredArith → PredArith
  deriving DecidableEq, Repr, Lean.ToExpr

inductive RelOp | lt | le | eq | ne | ge | gt
  deriving DecidableEq, Repr, Lean.ToExpr

inductive BoolExpr
  | rel  : RelOp → PredArith → PredArith → BoolExpr
  | and  : BoolExpr → BoolExpr → BoolExpr
  | or   : BoolExpr → BoolExpr → BoolExpr
  | not  : BoolExpr → BoolExpr
  | ieq  : PredArith → PredArith → BoolExpr
  deriving DecidableEq, Repr, Lean.ToExpr

inductive Nonlin
  | identity  : Nonlin
  | relu      : Nonlin
  | sigmoid   : Nonlin
  | tanh      : Nonlin
  | gelu      : Nonlin
  | leakyrelu : Nonlin
  | softmax   : Option BoolExpr → Nonlin
  | normalize : Option BoolExpr → Nonlin
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

-- Reduction operation for contraction (sum is standard; max/min are tropical).
inductive AggOp | sum | max | min
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

-- Unary transcendental functions applicable inline to a single factor's read (`log(P[i])`).
inductive UnaryOp | log | exp | sin | cos | sqrt
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

inductive Factor
  | read    : String → List IdxExpr → Factor
  | iverson : BoolExpr → Factor
  | unaryFn : UnaryOp → String → List IdxExpr → Factor
  deriving DecidableEq, Repr, Lean.ToExpr

structure ProdTerm where factors : List Factor
  deriving DecidableEq, Repr, Lean.ToExpr
structure SumExpr  where terms   : List ProdTerm
  deriving DecidableEq, Repr, Lean.ToExpr
structure RHSExpr  where
  body   : SumExpr
  nonlin : Nonlin
  agg    : AggOp := .sum   -- contraction operation (sum = standard; max = tropical)
  deriving DecidableEq, Repr, Lean.ToExpr

inductive LHSSlot
  | free     : AxisSpec → LHSSlot
  | freeNorm : AxisSpec → LHSSlot   -- a free output axis marked (`m.`) as the softmax/normalize reduction axis
  | iterAt   : AxisSpec → Int → LHSSlot
  | iterNext : AxisSpec → LHSSlot
  | affine   : IdxExpr → LHSSlot
  deriving DecidableEq, Repr, Lean.ToExpr

structure ScatterOpts where
  fill   : Int := 0
  reduce : Option String := none
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

inductive Stmt
  | assign  : String → List LHSSlot → RHSExpr → Stmt
  | scatter : String → List LHSSlot → RHSExpr → ScatterOpts → Stmt
  | recurMorphism : String → AxisSpec → ThreadedComposed → Stmt
    -- escape hatch (§12.2): tensor name, iteration axis, a pre-built step morphism (programmatic-only)
  deriving DecidableEq, Repr, Lean.ToExpr

structure TLProgram where
  decls : List Decl
  stmts : List Stmt
  deriving DecidableEq, Repr, Lean.ToExpr, Inhabited

end LeanNCD
