import LeanNCD.DSL.Ast
import LeanNCD.DSL.Syntax

/-!
# Tensor-logic DSL elaborators — lower layers 1–2.5 (Milestone E1, Task E1.3)

Value-returning `Syntax → MetaM <value>` elaborators that construct AST values
directly (NOT `Expr`-building). This is possible because `SizeExpr` and the AST
inductives are all computable + `DecidableEq` + `Repr`.

Layers covered: `tl_size`, `tl_axis_kind`, `tl_axis_spec`, `tl_shape`, `tl_decl`,
`tl_idx_expr` (general integer-affine), `tl_pred_term`.
-/

namespace LeanNCD
open Lean Elab Meta

partial def elabTLSize : Syntax → MetaM SizeExpr
  | `(tl_size| $n:num)          => return .lit n.getNat
  | `(tl_size| $x:ident)        => return .var x.getId.eraseMacroScopes.getString!
  | `(tl_size| $a:tl_size * $b) => return .mul (← elabTLSize a) (← elabTLSize b)
  | `(tl_size| $a:tl_size + $b) => return .add (← elabTLSize a) (← elabTLSize b)
  | `(tl_size| ($a:tl_size))    => elabTLSize a
  | _                           => throwUnsupportedSyntax

partial def elabTLAxisKind : Syntax → MetaM AxisKind
  | `(tl_axis_kind| ℝ)          => return .real none
  | `(tl_axis_kind| ℝ[ $s ])    => return .real (some (← elabTLSize s))
  | `(tl_axis_kind| ℕ)          => return .nat none
  | `(tl_axis_kind| ℕ[ $s ])    => return .nat (some (← elabTLSize s))
  | `(tl_axis_kind| norm)       => return .norm none
  | `(tl_axis_kind| norm[ $s ]) => return .norm (some (← elabTLSize s))
  | _                           => throwUnsupportedSyntax

partial def elabTLAxisSpec : Syntax → MetaM AxisSpec
  | `(tl_axis_spec| $x:ident : $k:tl_axis_kind) =>
      return { name := x.getId.eraseMacroScopes.getString!, uid := 0, kind := (← elabTLAxisKind k) }
  | _ => throwUnsupportedSyntax

partial def elabTLShape : Syntax → MetaM (List AxisSpec)
  | `(tl_shape| ( $specs,* )) => specs.getElems.toList.mapM elabTLAxisSpec
  | _ => throwUnsupportedSyntax

partial def elabTLDecl : Syntax → MetaM Decl
  | `(tl_decl| tensor $x:ident : $sh:tl_shape)    => return .tensor x.getId.eraseMacroScopes.getString! (← elabTLShape sh)
  | `(tl_decl| predicate $x:ident : $sh:tl_shape) => return .predicate x.getId.eraseMacroScopes.getString! (← elabTLShape sh)
  | `(tl_decl| linear $x:ident : $i:tl_shape → $o:tl_shape) =>
      return .linear x.getId.eraseMacroScopes.getString! (← elabTLShape i) (← elabTLShape o) false
  | `(tl_decl| linear $x:ident : $i:tl_shape → $o:tl_shape bias) =>
      return .linear x.getId.eraseMacroScopes.getString! (← elabTLShape i) (← elabTLShape o) true
  | _ => throwUnsupportedSyntax

/-- A placeholder `AxisSpec` for an index-expression axis reference.
    `uid` is assigned in Stage 2 (E2's `resolveDecls`); `kind` is resolved there too. -/
private def idxAxis (name : String) : AxisSpec :=
  { name := name, uid := 0, kind := .real none }

/-- An accumulated integer-affine read: a constant plus a list of `(coeff, axis)` terms. -/
private structure AffineAcc where
  const : Int := 0
  terms : List (Int × AxisSpec) := []

/-- Collect the (signed) terms of a `tl_idx_expr` into an `AffineAcc`.
    `sign` threads the surrounding `+`/`-` polarity (so `i - p` negates `p`). -/
partial def collectIdxTerms (sign : Int) (acc : AffineAcc) : Syntax → MetaM AffineAcc
  | `(tl_idx_expr| $n:num) =>
      return { acc with const := acc.const + sign * (n.getNat : Int) }
  | `(tl_idx_expr| $x:ident) =>
      return { acc with terms := acc.terms ++ [(sign, idxAxis x.getId.eraseMacroScopes.getString!)] }
  | `(tl_idx_expr| $n:num * $x:ident) =>
      return { acc with terms := acc.terms ++ [(sign * (n.getNat : Int), idxAxis x.getId.eraseMacroScopes.getString!)] }
  | `(tl_idx_expr| $a:tl_idx_expr + $b:tl_idx_expr) => do
      collectIdxTerms sign (← collectIdxTerms sign acc a) b
  | `(tl_idx_expr| $a:tl_idx_expr - $b:tl_idx_expr) => do
      collectIdxTerms (-sign) (← collectIdxTerms sign acc a) b
  | `(tl_idx_expr| ($a:tl_idx_expr)) => collectIdxTerms sign acc a
  | _ => throwUnsupportedSyntax

/-- Build an `IdxExpr` from a `tl_idx_expr`. Single-term shapes use the simple
    ctors (`.axis`/`.const`/`.scale`/`.shift`); general multi-term sums use `.affine`.
    Symbolic-coefficient strides (`ident * ident`) are not parsed and are out of scope. -/
partial def elabTLIdxExpr (stx : Syntax) : MetaM IdxExpr := do
  let acc ← collectIdxTerms 1 {} stx
  match acc.const, acc.terms with
  | c, []            => return .const c
  | 0, [(1, ax)]     => return .axis ax
  | 0, [(k, ax)]     => return .scale k ax
  | c, [(1, ax)]     => return .shift ax c
  | c, terms         => return .affine c terms

partial def elabTLPredTerm : Syntax → MetaM PredArith
  | `(tl_pred_term| $e:tl_idx_expr)    => return .embed (← elabTLIdxExpr e)
  | `(tl_pred_term| imul( $a , $b ))   => return .mul (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_pred_term| | $a |)            => return .iabs (← elabTLPredTerm a)
  | `(tl_pred_term| ($a:tl_pred_term)) => elabTLPredTerm a
  | _ => throwUnsupportedSyntax

/-! ## Layers 3–4: predicates, nonlinearities, factors, products, sums, RHS (Task E1.4) -/

partial def elabTLBoolExpr : Syntax → MetaM BoolExpr
  | `(tl_bool_expr| $a:tl_pred_term < $b:tl_pred_term)  => return .rel .lt (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_bool_expr| $a:tl_pred_term ≤ $b:tl_pred_term)  => return .rel .le (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_bool_expr| $a:tl_pred_term = $b:tl_pred_term)  => return .rel .eq (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_bool_expr| $a:tl_pred_term ≠ $b:tl_pred_term)  => return .rel .ne (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_bool_expr| $a:tl_pred_term > $b:tl_pred_term)  => return .rel .gt (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_bool_expr| $a:tl_pred_term ≥ $b:tl_pred_term)  => return .rel .ge (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_bool_expr| $a:tl_bool_expr ∧ $b:tl_bool_expr)  => return .and (← elabTLBoolExpr a) (← elabTLBoolExpr b)
  | `(tl_bool_expr| $a:tl_bool_expr ∨ $b:tl_bool_expr)  => return .or (← elabTLBoolExpr a) (← elabTLBoolExpr b)
  | `(tl_bool_expr| ¬ $a:tl_bool_expr)                  => return .not (← elabTLBoolExpr a)
  | `(tl_bool_expr| ieq( $a , $b ))                     => return .ieq (← elabTLPredTerm a) (← elabTLPredTerm b)
  | `(tl_bool_expr| ($b:tl_bool_expr))                  => elabTLBoolExpr b
  | _ => throwUnsupportedSyntax

partial def elabTLNonlin : Syntax → MetaM Nonlin
  | `(tl_nonlin| relu)                          => return .relu
  | `(tl_nonlin| softmax)                       => return .softmax none
  | `(tl_nonlin| softmax( where $b ))           => return .softmax (some (← elabTLBoolExpr b))
  | `(tl_nonlin| normalize)                     => return .normalize none
  | `(tl_nonlin| normalize( where $b ))         => return .normalize (some (← elabTLBoolExpr b))
  | _ => throwUnsupportedSyntax

partial def elabTLFactor : Syntax → MetaM Factor
  | `(tl_factor| $name:ident [ $idxs,* ]) =>
      return .read (name.getId.eraseMacroScopes.getString!) (← idxs.getElems.toList.mapM elabTLIdxExpr)
  | `(tl_factor| [ $b:tl_bool_expr ]) => return .iverson (← elabTLBoolExpr b)
  | _ => throwUnsupportedSyntax

/-- Collect the factor list of a `tl_prod_term`. The `·` rule is left-recursive
    and n-ary (`tl_prod_term · tl_factor`); flatten left-recursively into a list. -/
partial def prodFactors : Syntax → MetaM (List Factor)
  | `(tl_prod_term| $p:tl_prod_term · $f:tl_factor) => return (← prodFactors p) ++ [(← elabTLFactor f)]
  | `(tl_prod_term| $f:tl_factor)                   => return [(← elabTLFactor f)]
  | _ => throwUnsupportedSyntax

partial def elabTLProdTerm (stx : Syntax) : MetaM ProdTerm :=
  return { factors := (← prodFactors stx) }

/-- Collect the product-term list of a `tl_sum_expr`. The `+` rule is left-recursive
    and n-ary (`tl_sum_expr + tl_prod_term`); flatten left-recursively into a list. -/
partial def sumTerms : Syntax → MetaM (List ProdTerm)
  | `(tl_sum_expr| $s:tl_sum_expr + $p:tl_prod_term) => return (← sumTerms s) ++ [(← elabTLProdTerm p)]
  | `(tl_sum_expr| $p:tl_prod_term)                  => return [(← elabTLProdTerm p)]
  | _ => throwUnsupportedSyntax

partial def elabTLSumExpr (stx : Syntax) : MetaM SumExpr :=
  return { terms := (← sumTerms stx) }

partial def elabTLRHS : Syntax → MetaM RHSExpr
  | `(tl_rhs| $nl:tl_nonlin ( $s:tl_sum_expr )) =>
      return { body := (← elabTLSumExpr s), nonlin := (← elabTLNonlin nl) }
  | `(tl_rhs| $s:tl_sum_expr) =>
      return { body := (← elabTLSumExpr s), nonlin := .identity }
  | _ => throwUnsupportedSyntax

end LeanNCD
