import Lean

/-!
# Tensor-logic DSL surface grammar (Milestone E1, Task E1.2)

This module declares the 17 `declare_syntax_cat` categories and transcribes the
surface grammar rules from `papers/leanncd.md` §12.3.  It defines *only* the
grammar; the elaborators that consume it are added in later tasks.

Precedence annotations are added where the flat binary rules of §12.3 would be
ambiguous, so that the standard binding holds:
* `*` binds tighter than `+` (in `tl_size`);
* `·` (product) binds tighter than `+` (sum);
* comparisons bind tighter than `∧`, which binds tighter than `∨`.
All binary operators are left-associative.
-/

namespace LeanNCD
open Lean

declare_syntax_cat tl_size
declare_syntax_cat tl_axis_kind
declare_syntax_cat tl_axis_spec
declare_syntax_cat tl_shape
declare_syntax_cat tl_decl
declare_syntax_cat tl_idx_expr
declare_syntax_cat tl_pred_term
declare_syntax_cat tl_rel_op
declare_syntax_cat tl_bool_expr
declare_syntax_cat tl_nonlin
declare_syntax_cat tl_factor
declare_syntax_cat tl_prod_term
declare_syntax_cat tl_sum_expr
declare_syntax_cat tl_rhs
declare_syntax_cat tl_lhs_slot
declare_syntax_cat tl_stmt
declare_syntax_cat tl_program

-- Layer 1: sizes (bracket holds a tl_size term elaborating to Numeric, §2.1)
-- `*` (prec 70) binds tighter than `+` (prec 65); both left-associative.
syntax:max num                       : tl_size
syntax:max ident                     : tl_size
syntax:70 tl_size:70 " * " tl_size:71 : tl_size
syntax:65 tl_size:65 " + " tl_size:66 : tl_size
syntax:max "(" tl_size ")"           : tl_size

-- Layer 1: axis kinds
syntax "ℝ"                   : tl_axis_kind
syntax "ℝ[" tl_size "]"      : tl_axis_kind
syntax "ℕ"                   : tl_axis_kind
syntax "ℕ[" tl_size "]"      : tl_axis_kind
syntax "norm"                : tl_axis_kind
syntax "norm[" tl_size "]"   : tl_axis_kind

syntax ident ":" tl_axis_kind : tl_axis_spec
syntax "(" tl_axis_spec,* ")" : tl_shape

syntax "tensor"    ident ":" tl_shape                      : tl_decl
syntax "predicate" ident ":" tl_shape                      : tl_decl
syntax "linear"    ident ":" tl_shape "→" tl_shape         : tl_decl
syntax "linear"    ident ":" tl_shape "→" tl_shape " bias" : tl_decl

-- Layer 2: index expressions — GENERALIZED to general integer-affine sums (E1.3).
-- A `tl_idx_expr` is a left-associative `+`/`-` sum of terms, where each term is a
-- bare `num`, a bare `ident`, or a literal-coefficient product `num "*" ident`.
-- This subsumes the former single-term forms (`ident`, `num`, `num*ident`,
-- `ident±num`, `num*ident+num`) and admits general reads like `i + p`, `2*j + r`.
-- NOTE: symbolic-coefficient strides `ident "*" ident` (e.g. `s * j`) are NOT
-- representable in the integer-coefficient `IdxExpr` and are out of scope.
-- `*` (prec 70) binds tighter than `+`/`-` (prec 65); both left-associative.
syntax:70 num "*" ident          : tl_idx_expr  -- literal-coefficient term
syntax:max ident                 : tl_idx_expr
syntax:max num                   : tl_idx_expr
syntax:65 tl_idx_expr:65 " + " tl_idx_expr:66 : tl_idx_expr
syntax:65 tl_idx_expr:65 " - " tl_idx_expr:66 : tl_idx_expr  -- look-back (n < 0)
syntax:max "(" tl_idx_expr ")"   : tl_idx_expr

-- Layer 2.5: predicate arithmetic
syntax:max tl_idx_expr                           : tl_pred_term
syntax "imul(" tl_pred_term "," tl_pred_term ")" : tl_pred_term
syntax "|" tl_pred_term "|"                       : tl_pred_term  -- iabs; value, not bool
syntax:max "(" tl_pred_term ")"                  : tl_pred_term

-- Layer 3: predicates
-- Comparisons (prec 50) bind tighter than `∧` (prec 35), which binds tighter
-- than `∨` (prec 30); `¬` (prec 40) binds tighter than `∧`/`∨`.
syntax:50 tl_pred_term:51 "<"  tl_pred_term:51   : tl_bool_expr
syntax:50 tl_pred_term:51 "≤"  tl_pred_term:51   : tl_bool_expr
syntax:50 tl_pred_term:51 "="  tl_pred_term:51   : tl_bool_expr
syntax:50 tl_pred_term:51 "≠"  tl_pred_term:51   : tl_bool_expr
syntax:50 tl_pred_term:51 ">"  tl_pred_term:51   : tl_bool_expr
syntax:50 tl_pred_term:51 "≥"  tl_pred_term:51   : tl_bool_expr
syntax:35 tl_bool_expr:35 "∧" tl_bool_expr:36    : tl_bool_expr
syntax:30 tl_bool_expr:30 "∨" tl_bool_expr:31    : tl_bool_expr
syntax:40 "¬" tl_bool_expr:40                     : tl_bool_expr
syntax "ieq(" tl_pred_term "," tl_pred_term ")"  : tl_bool_expr
syntax:max "(" tl_bool_expr ")"                  : tl_bool_expr

-- Layer 4: RHS
syntax ident "[" tl_idx_expr,* "]"     : tl_factor
syntax "[" tl_bool_expr "]"            : tl_factor

-- `·` (product) binds tighter than `+` (sum); both left-associative.
syntax:70 tl_factor:70 " · " tl_factor:71 : tl_prod_term
syntax:max tl_factor                       : tl_prod_term

syntax:65 tl_prod_term:65 " + " tl_prod_term:66 : tl_sum_expr
syntax:max tl_prod_term                          : tl_sum_expr

syntax "relu"                                    : tl_nonlin
syntax "softmax"                                 : tl_nonlin
syntax "softmax"   "(" "where" tl_bool_expr ")"  : tl_nonlin
syntax "normalize"                               : tl_nonlin
syntax "normalize" "(" "where" tl_bool_expr ")"  : tl_nonlin

syntax tl_nonlin "(" tl_sum_expr ")"   : tl_rhs
syntax tl_sum_expr                      : tl_rhs

-- Layer 5: statements
syntax ident "[" tl_lhs_slot,* "]" ":=" tl_rhs : tl_stmt

syntax:60 num "*" ident "+" num : tl_lhs_slot
syntax:55 num "*" ident         : tl_lhs_slot
syntax:55 ident "+1"            : tl_lhs_slot
syntax:55 ident "+" num         : tl_lhs_slot
syntax:max ident                : tl_lhs_slot
syntax:max num                  : tl_lhs_slot

-- Layer 6
syntax (tl_decl <|> tl_stmt)* : tl_program

end LeanNCD
