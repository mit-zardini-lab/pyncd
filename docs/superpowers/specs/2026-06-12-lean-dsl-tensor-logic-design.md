# Lean DSL for Tensor Logic: Design Spec

## What this is

A new §12 ("The tensor-logic DSL") in `papers/leanncd.md`, inserted between the current §11 (Lean formalization notes) and §12 (Appendix: out of scope, which renumbers to §13). The section presents a complete DSL embedding in Lean 4 following the syntax-category + elaboration pattern from the Lean 4 metaprogramming book (ch. 8): BNF grammar → inductive AST types → `declare_syntax_cat`/`syntax` rules → `elabXxx : Syntax → MetaM Expr` functions → semantic compiler `TLProgram.compile : TLProgram → TermM ThreadedComposed`.

## Two-stage pipeline (key insight from ch. 8)

The IMP chapter distinguishes two stages that the prior spec collapsed into one:

**Stage 1 — Lean elaboration** (`MetaM`): Concrete syntax (`Syntax`) → typed `TLProgram` value. This is done by `elabTLProgram : Syntax → MetaM Expr`, which walks the syntax tree produced by Lean's parser and uses `mkAppM` to build a Lean `Expr` representing a `TLProgram`. No fresh UIDs are minted here; the elaborator is pure syntax-to-AST.

**Stage 2 — Semantic compilation** (`TermM`): `TLProgram` value → `ThreadedComposed` morphism. This is `TLProgram.compile`, which mints fresh UIDs, runs axis unification, lowers index arithmetic, finalizes scans, etc. It runs in `TermM` (§7.4) because it needs the fresh-name counter.

The entry point wires the two stages:

```lean
elab "tl!{" p:tl_program "}" : term => do
  let prog ← elabTLProgram p          -- Stage 1: Syntax → TLProgram  (MetaM)
  mkAppM ``TLProgram.compile #[prog]  -- Stage 2: TLProgram → ThreadedComposed  (TermM)
```

## Structure (four subsections)

1. **§12.1 BNF grammar** — surface language
2. **§12.2 Abstract syntax** — Lean inductive types
3. **§12.3 Concrete syntax and elaboration** — syntax categories, `syntax` rules, `elabXxx` signatures, entry point
4. **§12.4 Semantic compilation** — `TLProgram.compile` phase table + Python correspondence table

## Placement changes

- §12 Appendix → §13 Appendix
- Table of contents: add `12. [The tensor-logic DSL](#12-the-tensor-logic-dsl)` and renumber the appendix entry

---

## §12.1 BNF Grammar (full)

Extends Domingos' tensor-logic notation (implicit Σ over contracted axes, Einstein product) with: axis typing (ℝ/ℕ/norm), tensor declarations, Iverson predicates, nonlinearities with optional masks, affine index arithmetic (Slice/Reindex/Scatter), and temporal recursion (Scan).

```
-- Layer 1: Axis specifications and declarations
decl        ::= 'tensor'    name ':' shape
              | 'predicate' name ':' shape
              | 'linear'    name ':' in_shape '→' out_shape ['bias']

shape       ::= '(' ')'
              | '(' axis_spec (',' axis_spec)* ')'

axis_spec   ::= name ':' axis_kind

axis_kind   ::= 'ℝ'           -- symbolic-size real axis          (RawAxis)
              | 'ℝ[' n ']'    -- concrete-size real axis
              | 'ℕ'           -- symbolic-size discrete axis       (NatAxis)
              | 'ℕ[' n ']'    -- concrete-size discrete axis
              | 'norm'        -- normalization axis, symbolic       (NormAxis)
              | 'norm[' n ']' -- normalization axis, concrete

-- Layer 2: Index expressions (strictly affine; n ∈ ℤ)
idx_expr    ::= axis_name                      -- plain axis: free or contracted
              | n                              -- constant coordinate             (Slice)
              | n '*' axis_name               -- scaling
              | axis_name '+' n               -- shift  (look-back on RHS: n < 0)
              | n '*' axis_name '+' n         -- general affine                   (Reindex)
              | '(' idx_expr ')'              -- grouping (for clarity)

-- Layer 2.5: Predicate arithmetic (extends idx_expr with non-affine products)
-- Only valid inside bool_expr; forbidden in tensor index slots.
pred_term   ::= idx_expr
              | 'imul(' pred_term ',' pred_term ')'   -- non-affine integer product
              | '(' pred_term ')'

-- Layer 3: Iverson predicates
bool_expr   ::= pred_term rel_op pred_term
              | bool_expr '∧' bool_expr
              | bool_expr '∨' bool_expr
              | '¬' bool_expr
              | '|' pred_term '|'             -- integer absolute value            (iabs)
              | 'ieq(' pred_term ',' pred_term ')' -- integer equality predicate
              | '(' bool_expr ')'             -- grouping

rel_op      ::= '<' | '≤' | '=' | '≠' | '>' | '≥'

-- Layer 4: RHS expressions (Domingos core + nonlinearities + predicates)
rhs         ::= nonlin '(' sum_expr ')'
              | sum_expr

sum_expr    ::= prod_term ('+' prod_term)*

prod_term   ::= factor ('·' factor)*           -- implicit Σ over contracted axes

factor      ::= name '[' idx_expr (',' idx_expr)* ']'  -- tensor read
              | '[' bool_expr ']'                       -- Iverson bracket

nonlin      ::= 'relu'
              | 'softmax'
              | 'softmax'   '(' 'where' bool_expr ')'   -- masked softmax
              | 'normalize'
              | 'normalize' '(' 'where' bool_expr ')'   -- masked normalize

-- Layer 5: Statements
stmt        ::= assign | base_case | recur_step | scatter_write

assign      ::= name '[' axis_name (',' axis_name)* ']' ':=' rhs

-- l+1 and 0 may appear in any position; iteration axis l is identified by the l+1 slot
base_case   ::= name '[' base_slot_list ']'  ':=' rhs
recur_step  ::= name '[' recur_slot_list ']' ':=' rhs

base_slot_list  ::= (axis_name ',')* n (',' axis_name)*          -- n is base value (usually 0)
recur_slot_list ::= (axis_name ',')* axis_name '+' '1' (',' axis_name)*

-- Affine LHS: every slot is a (possibly affine) output coordinate
scatter_write ::= name '[' affine_slot (',' affine_slot)* ']' ':=' rhs
                    ['fill' n] ['reduce' 'sum']

affine_slot ::= axis_name
              | n '*' axis_name
              | axis_name '+' n
              | n '*' axis_name '+' n

-- Layer 6: Programs
program     ::= decl* stmt+
```

**Contracted axes** are implicit: any `axis_name` appearing in a `prod_term` but absent from the LHS is summed over — Domingos' convention, unchanged.

**Coupled scans** require no special syntax: two `recur_step` stmts for different tensor names whose iteration axis (the `axis_name` in `axis_name '+' '1'`) carries the same UID are automatically grouped into a coupled `Scan` (`n_states > 1`) by the semantic compiler.

**Semantic constraints** enforced by the compiler, not the grammar:

- `l+1` on the RHS of a `recur_step` is a causality violation and is rejected, where `l` is that step's iteration axis; look-ahead reads on non-iteration axes are permitted
- Scatter with overlapping writes requires `reduce sum`
- A `recur_step` without a matching `base_case` for the same name is an error
- A `linear`-declared weight must multiply exactly one activation factor

### Five representative examples

```
-- Matmul (Domingos base: k is contracted)
Y[i,j] := W[i,k] · X[k,j]

-- Causal masked attention (norm axis + Iverson mask)
tensor A : (q : ℝ, s : norm)
A[q,s] := softmax(where s ≤ q)(Q[q,d] · K[s,d])

-- Strided convolution (affine Reindex reads)
Y[i,j] := W[p,r] · X[i+p, s*j+r]

-- Upsample 2× (affine Scatter write)
tensor Out : (i : ℝ[2*m], j : ℝ[2*n])
Out[2*i, 2*j] := X[i,j]

-- Coupled scan: G and H share iteration axis l (coupled Scan, n_states=2)
G[j, 0]   := X[j]
G[j, l+1] := relu(G[j,l] · W_G[j,k] + H[j,l] · U[j,k])
H[j, 0]   := Y[j]
H[j, l+1] := relu(H[j,l] · W_H[j,k] + G[j,l] · V[j,k])
```

---

## §12.2 Abstract syntax

Direct formalization of the BNF layers as Lean inductive types. `UID` and `Numeric` from §2.1/§7.4; `TermM` from §7.4.

```lean
-- Layer 1
inductive AxisKind
  | real   : Option Numeric → AxisKind   -- ℝ axis; Python: RawAxis
  | nat    : Option Numeric → AxisKind   -- ℕ axis; Python: NatAxis
  | norm   : Option Numeric → AxisKind   -- normalization axis; Python: NormAxis

structure AxisSpec where
  name : String
  uid  : UID       -- identity key for Context coequalizer (§7.4); assigned in Stage 2
  kind : AxisKind
-- Python: RawAxis / NatAxis / NormAxis

inductive Decl
  | tensor    : String → List AxisSpec → Decl
  | predicate : String → List AxisSpec → Decl
  | linear    : String → (inAxes outAxes : List AxisSpec) → (bias : Bool) → Decl
-- Python: TensorDeclaration (TensorKind.TENSOR / PREDICATE / LINEAR)
```

```lean
-- Layer 2
inductive IdxExpr
  | axis   : AxisSpec → IdxExpr                          -- free or contracted axis
  | const  : ℤ → IdxExpr                                -- constant coordinate (Slice)
  | scale  : ℤ → AxisSpec → IdxExpr                     -- n * a
  | shift  : AxisSpec → ℤ → IdxExpr                     -- a + n  (n < 0 = look-back)
  | affine : ℤ → List (ℤ × AxisSpec) → IdxExpr          -- n + Σ cᵢ·aᵢ (general Reindex)
  -- Note: '(' idx_expr ')' is a surface-syntax grouping, not a constructor;
  --       elabTLIdxExpr recurses into the inner expression.
-- Python: RawAxis / int / IversonBinOp (monkey-patched arithmetic)
```

```lean
-- Layer 2.5: Predicate arithmetic (extends IdxExpr with non-affine products)
inductive PredArith
  | embed : IdxExpr → PredArith          -- lift any affine expression
  | mul   : PredArith → PredArith → PredArith   -- imul; non-affine product
  -- Note: '(' pred_term ')' is surface grouping, not a constructor.
-- Python: IversonBinOp('*', …) / imul
```

```lean
-- Layer 3
inductive RelOp | lt | le | eq | ne | ge | gt

inductive BoolExpr
  | rel  : RelOp → PredArith → PredArith → BoolExpr
  | and  : BoolExpr → BoolExpr → BoolExpr
  | or   : BoolExpr → BoolExpr → BoolExpr
  | not  : BoolExpr → BoolExpr
  | iabs : PredArith → BoolExpr                          -- |e| (iabs)
  | ieq  : PredArith → PredArith → BoolExpr
  -- Note: '(' bool_expr ')' is a surface-syntax grouping, not a constructor;
  --       elabTLBoolExpr recurses into the inner expression.
-- Python: IversonBinOp / IversonUnaryOp / ieq / iabs / inot
```

```lean
-- Layer 4
inductive Nonlin
  | identity  : Nonlin
  | relu      : Nonlin
  | softmax   : Option BoolExpr → Nonlin
  | normalize : Option BoolExpr → Nonlin
-- Python: ops.Identity / ops.ReLU / ops.SoftMax / ops.Normalize

inductive Factor
  | read    : String → List IdxExpr → Factor    -- name[e₁,...,eₙ]
  | iverson : BoolExpr → Factor                 -- [P]
-- Python: IndexedTensor / IversonBinOp as a multiplicative factor

structure ProdTerm where factors : List Factor   -- Python: RHSExpression.factors
structure SumExpr  where terms   : List ProdTerm -- Python: SumExpr.terms
structure RHSExpr  where body : SumExpr; nonlin : Nonlin
```

```lean
-- Layer 5
inductive LHSSlot
  | free     : AxisSpec → LHSSlot              -- ordinary free axis
  | iterAt   : AxisSpec → ℤ → LHSSlot          -- l = n  (base case; n usually 0)
  | iterNext : AxisSpec → LHSSlot              -- l + 1  (recurrence step)
  | affine   : IdxExpr → LHSSlot              -- affine output slot (Scatter)
-- Python: __setitem__ dispatch on index type

structure ScatterOpts where
  fill   : Float := 0.0
  reduce : Option String := none    -- none = injective required; 'sum' = accumulate
-- Python: TL._scatter_opts

inductive Stmt
  | assign        : String → List LHSSlot → RHSExpr → Stmt
  | scatter       : String → List LHSSlot → RHSExpr → ScatterOpts → Stmt
  | recurMorphism : String → AxisSpec → Expr → Stmt
  -- escape hatch: pre-built step morphism when the recurrence step is too complex
  -- for a single equation. String = tensor name, AxisSpec = iteration axis,
  -- Expr : ThreadedComposed = the step morphism term. Syntax TBD.
  -- Python: tl.Name.recur(l, morphism)
-- LHSSlot variants distinguish assign / base_case / recur_step
-- Python: TL._register_entry / _register_iter_base / _register_iter_recur / _register_scatter

structure TLProgram where
  decls : List Decl
  stmts : List Stmt
-- Python: TL registry (_declarations + _entries + _pending_iter)
```

---

## §12.3 Concrete syntax and elaboration

Following the IMP language pattern of the Lean 4 metaprogramming book (ch. 8): one `declare_syntax_cat` per BNF layer, `syntax` rules transcribing each production, and a `partial def elabXxx : Syntax → MetaM Expr` function per category.

### Syntax categories

```lean
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
```

### Representative syntax rules (one per BNF layer)

```lean
-- Layer 1: axis kinds
syntax "ℝ"              : tl_axis_kind
syntax "ℝ[" num "]"     : tl_axis_kind
syntax "ℕ"              : tl_axis_kind
syntax "ℕ[" num "]"     : tl_axis_kind
syntax "norm"           : tl_axis_kind
syntax "norm[" num "]"  : tl_axis_kind

syntax ident ":" tl_axis_kind : tl_axis_spec
syntax "(" tl_axis_spec,* ")" : tl_shape

syntax "tensor"    ident ":" tl_shape                             : tl_decl
syntax "predicate" ident ":" tl_shape                             : tl_decl
syntax "linear"    ident ":" tl_shape "→" tl_shape               : tl_decl
syntax "linear"    ident ":" tl_shape "→" tl_shape " bias"       : tl_decl

-- Layer 2: index expressions
syntax ident                     : tl_idx_expr  -- axis
syntax num                       : tl_idx_expr  -- constant (Slice)
syntax num "*" ident             : tl_idx_expr  -- scaling
syntax ident "+" num             : tl_idx_expr  -- shift (positive)
syntax ident "-" num             : tl_idx_expr  -- look-back (n < 0)
syntax num "*" ident "+" num     : tl_idx_expr  -- general affine
syntax "(" tl_idx_expr ")"       : tl_idx_expr  -- grouping

-- Layer 2.5: predicate arithmetic
syntax tl_idx_expr                                          : tl_pred_term  -- embed affine
syntax "imul(" tl_pred_term "," tl_pred_term ")"           : tl_pred_term  -- non-affine product
syntax "(" tl_pred_term ")"                                : tl_pred_term  -- grouping

-- Layer 3: predicates
syntax tl_pred_term "<"  tl_pred_term : tl_bool_expr
syntax tl_pred_term "≤"  tl_pred_term : tl_bool_expr
syntax tl_pred_term "="  tl_pred_term : tl_bool_expr
syntax tl_pred_term "≠"  tl_pred_term : tl_bool_expr
syntax tl_pred_term ">"  tl_pred_term : tl_bool_expr
syntax tl_pred_term "≥"  tl_pred_term : tl_bool_expr
syntax tl_bool_expr "∧" tl_bool_expr : tl_bool_expr
syntax tl_bool_expr "∨" tl_bool_expr : tl_bool_expr
syntax "¬" tl_bool_expr              : tl_bool_expr
syntax "|" tl_pred_term "|"           : tl_bool_expr
syntax "ieq(" tl_pred_term "," tl_pred_term ")" : tl_bool_expr
syntax "(" tl_bool_expr ")"                   : tl_bool_expr  -- grouping

-- Layer 4: RHS
syntax ident "[" tl_idx_expr,* "]"  : tl_factor   -- tensor read
syntax "[" tl_bool_expr "]"          : tl_factor   -- Iverson bracket

syntax tl_factor " · " tl_factor    : tl_prod_term  -- product
syntax tl_factor                     : tl_prod_term  -- single factor

syntax tl_prod_term " + " tl_prod_term : tl_sum_expr
syntax tl_prod_term                     : tl_sum_expr

-- tl_nonlin (masked variants carry the where-clause inside their own parens)
syntax "relu"                                              : tl_nonlin
syntax "softmax"                                           : tl_nonlin
syntax "softmax"   "(" "where" tl_bool_expr ")"           : tl_nonlin
syntax "normalize"                                         : tl_nonlin
syntax "normalize" "(" "where" tl_bool_expr ")"           : tl_nonlin

-- tl_rhs: nonlin applied to a sum, or bare sum
syntax tl_nonlin "(" tl_sum_expr ")"                      : tl_rhs
syntax tl_sum_expr                                         : tl_rhs

-- Layer 5: statements
syntax ident "[" tl_lhs_slot,* "]" ":=" tl_rhs : tl_stmt
-- tl_lhs_slot handles free / iterAt / iterNext / affine via their own syntax rules:
syntax ident                 : tl_lhs_slot   -- free axis
syntax num                   : tl_lhs_slot   -- base case (iterAt, n=0 typically)
syntax ident "+1"            : tl_lhs_slot   -- recurrence step (iterNext)
syntax num "*" ident         : tl_lhs_slot   -- affine scatter slot
syntax ident "+" num         : tl_lhs_slot   -- affine scatter slot (shift)
syntax num "*" ident "+" num : tl_lhs_slot   -- affine scatter slot (general)

-- Layer 6: program = sequence of decls then stmts
syntax (tl_decl <|> tl_stmt)* : tl_program
```

### Elaboration function signatures

One `partial def elabXxx : Syntax → MetaM Expr` per syntax category, following the IMP chapter pattern. `partial` is required because Lean's termination checker cannot verify that syntax consumption decreases.

```lean
partial def elabTLAxisKind  : Syntax → MetaM Expr   -- → AxisKind
partial def elabTLAxisSpec  : Syntax → MetaM Expr   -- → AxisSpec  (uid left as 0; assigned in Stage 2)
partial def elabTLShape     : Syntax → MetaM Expr   -- → List AxisSpec
partial def elabTLDecl      : Syntax → MetaM Expr   -- → Decl
partial def elabTLIdxExpr   : Syntax → MetaM Expr   -- → IdxExpr
partial def elabTLPredTerm  : Syntax → MetaM Expr   -- → PredArith
partial def elabTLRelOp     : Syntax → MetaM Expr   -- → RelOp
partial def elabTLBoolExpr  : Syntax → MetaM Expr   -- → BoolExpr
partial def elabTLNonlin    : Syntax → MetaM Expr   -- → Nonlin
partial def elabTLFactor    : Syntax → MetaM Expr   -- → Factor
partial def elabTLProdTerm  : Syntax → MetaM Expr   -- → ProdTerm
partial def elabTLSumExpr   : Syntax → MetaM Expr   -- → SumExpr
partial def elabTLRHS       : Syntax → MetaM Expr   -- → RHSExpr
partial def elabTLLHSSlot   : Syntax → MetaM Expr   -- → LHSSlot
partial def elabTLStmt      : Syntax → MetaM Expr   -- → Stmt
partial def elabTLProgram   : Syntax → MetaM Expr   -- → TLProgram
-- Each uses mkAppM ``Constructor #[...] to build the typed Expr; throwUnsupportedSyntax on mismatch.
```

### Entry point

```lean
/-- Write a tensor program between tl!{ ... } to produce a ThreadedComposed morphism. -/
elab "tl!{" p:tl_program "}" : term => do
  let prog ← elabTLProgram p          -- Stage 1: Syntax → TLProgram  (MetaM)
  mkAppM ``TLProgram.compile #[prog]  -- Stage 2: TLProgram → ThreadedComposed  (TermM, reduced at kernel time)
```

The `elab` command separates the two stages cleanly: `elabTLProgram` runs in `MetaM` and produces a Lean `Expr` of type `TLProgram`; the `mkAppM` wraps it in a call to `TLProgram.compile`, which runs in `TermM` when Lean evaluates the resulting term. UID minting and axis unification happen in Stage 2, not Stage 1, so the elaborator is pure syntax-walking with no side effects.

---

## §12.4 Semantic compilation

```lean
/-- Lower a TLProgram to a ThreadedComposed morphism. Runs in TermM to mint
    fresh UIDs for synthetic intermediates introduced during index-arithmetic
    lowering and nonlinearity splitting. -/
def TLProgram.compile : TLProgram → TermM ThreadedComposed
-- Python: TL.to_morphism()
```

The compilation is a typed pipeline. Each phase boundary carries a more
constrained type, so invariants that Python documents in comments are enforced
by construction:

```text
TLProgram
  →[assignUIDs]    LabeledProgram        -- every AxisSpec has a fresh UID
  →[resolveDecls]  ResolvedProgram       -- DeclEnv built; external names identified; bias materialized
  →[unifyAxes]     CanonicalProgram      -- axis UIDs are canonical (pure)
  →[lowerArith]    LoweredProgram        -- no const/affine IdxExprs in reads; Scatter fill initialized
  →[finalizeScans] ScanProgram           -- no bare iterAt/iterNext LHSSlots
  →[splitNonlins]  LinearProgram         -- no nonlinearity in RHSExpr.nonlin
  →[schedule]      ScheduledProgram      -- live stmts in reverse-topological order
  →[route]         ThreadedComposed
```

| Phase | What it does | Key Lean idiom |
|---|---|---|
| **assignUIDs** | Traverses `decls` and `stmts`; mints a fresh UID for each `AxisSpec` via `freshUData` (§7.4). Produces `LabeledProgram` where every axis occurrence is tagged. | `StateT UID TermM` |
| **resolveDecls** | Builds `DeclEnv : HashMap String Decl` from `LabeledProgram.decls`. Validates: `linear` weight appears in exactly one product factor; every declared name has a consistent shape across stmts. `linear ... bias:=true` emits a fresh bias-add stmt (WriterT). Marks each name as external (declared) or internal (produced by a stmt) — this distinction drives routing in the final phase. Predicate-typed names are recorded here so the routing phase can emit ∃/∧ contraction rather than Σ. | `WriterT (DList Stmt) Id` for bias stmts; pure `DeclEnv` output |
| **unifyAxes** | Collects `(uid_a, uid_b)` pairs from positional matching of axis occurrences across stmts; computes transitive closure; returns a canonical-representative map and `CanonicalProgram` with all UIDs normalized. **Pure function** — the full program is known statically, so no incremental update loop is needed. | `HashMap UID UID` (transitive closure); pure |
| **lowerArith** | `IdxExpr.const n` reads → fresh `Slice` intermediate stmt; `IdxExpr.affine` reads → fresh `Reindex` intermediate stmt; affine `LHSSlot`s → `Scatter` (injectivity checked; `ScatterOpts.reduce = some "sum"` required for non-injective maps). Scatter with a non-zero fill value also emits a fill-initialization stmt before the scatter write. All auxiliary stmts are **emitted into a writer**, not registered into a global. | `WriterT (DList Stmt) (StateT UID Id)` |
| **finalizeScans** | Groups stmts by name + iteration axis UID; pairs `iterAt`/`iterNext` slots into `Scan` nodes; stmts sharing the same iteration-axis UID across different tensor names are grouped into a coupled `Scan` (`n_states > 1`). `Stmt.recurMorphism` stubs supply the step morphism directly, bypassing equation lowering for that scan state. Forward references resolve without pre-declaration because the full stmt list is available. Validates: every `recur_step` has a matching `base_case`; no `l+1` on the RHS for the same iteration axis. | Pure `List Stmt → List ScanStmt` |
| **splitNonlins** | Lifts `relu`/`softmax`/`normalize` out of `RHSExpr.nonlin` into a separate composed step, emitting the new step into the writer. Masked softmax also emits an alignment-permutation step computed from the `where` mask. | `WriterT (DList ScanStmt) Id` |
| **schedule** | Backward reachability BFS from the output name simultaneously determines liveness (DCE) and produces a valid reverse-topological order. Two passes in Python; one in Lean because the BFS visit order is already a reverse topo order. | Pure `String → List ScanStmt → List ScanStmt` |
| **route** | For each stmt: detects contracted axes (axes appearing in a `ProdTerm` but absent from the LHS), builds a `Broadcasted` morphism (using ∃/∧ contraction for predicate-typed outputs per `DeclEnv`, Σ otherwise). Assigns each external input and produced intermediate an index slot; builds `ThreadedComposed.routing` and `n_external`. Automatic associative-scan detection (pure syntactic check on recurrence `IdxExpr`) selects `ScanAffine` where applicable. | Pure `List ScanStmt → DeclEnv → Context → ThreadedComposed` |

The result is a `ThreadedComposed` morphism in `Br` — a finite presentation of an `∫Dat`-morphism in the sense of §8.1.

### Python correspondence table

| Lean DSL | Python DSL | Notes |
|---|---|---|
| `tensor Name : shape` | `tl.Name.tensor(*axes)` | shape declaration |
| `predicate Name : shape` | `tl.Name.predicate(*axes)` | Bool-typed |
| `linear Name : in → out [bias]` | `tl.Name.linear(out_axes=…, in_axes=…, bias=…)` | weight declaration |
| `Name[i,j] := rhs` | `tl.Name[i,j] = rhs` | normal assignment |
| `Name[0, j] := rhs` | `tl.Name[j, 0] = rhs` | scan base case |
| `Name[l+1, j] := rhs` | `tl.Name[j, l+1] = rhs` | scan recurrence step |
| `Name[2*i] := rhs` | `tl.Name[2*i] = rhs` | affine Scatter write |
| `A[i,k] · B[k,j]` | `tl.A[i,k] * tl.B[k,j]` | Einstein product; k contracted |
| `A[i] + B[i]` | `tl.A[i] + tl.B[i]` | elementwise sum |
| `[i < j]` | `i < j` (Iverson via monkey-patch) | Iverson bracket |
| `relu(…)` | `relu(…)` | ReLU nonlinearity |
| `softmax(where P)(…)` | `softmax(…, where=P)` | masked softmax |
| `normalize(where P)(…)` | `normalize(…, where=P)` | masked normalize |
| `X[n]` | `tl.X[n]` (int index) | Slice — constant read |
| `X[i + n]` | `tl.X[i + n]` (affine expr) | Reindex — affine read |
| `Y[n*i] := …` | `tl.Y[n*i] = …` (affine LHS) | Scatter — affine write |
| `Stmt.recurMorphism name axis morphism` | `tl.name.recur(l, morphism)` | escape hatch; step morphism supplied as a term (syntax TBD) |
| `elabTLProgram` (Stage 1) | — | `Syntax → MetaM Expr`; no Python analogue |
| `TLProgram.compile` (Stage 2) | `tl.to_morphism()` | `TLProgram → TermM ThreadedComposed` |

---

## Files changed

- `papers/leanncd.md`: insert §12 before the appendix; renumber appendix to §13; update table of contents
