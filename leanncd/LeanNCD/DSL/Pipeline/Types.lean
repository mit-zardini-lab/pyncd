import LeanNCD.DSL.Ast
import LeanNCD.DSL.Target
import LeanNCD.Exec.Context
import Std.Data.HashMap

namespace LeanNCD
open Std

/-- Declaration environment built by resolveDecls (`String` has BEq+Hashable). -/
abbrev DeclEnv := HashMap String Decl

/-- A statement after finalizeScans grouped iterAt/iterNext pairs into Scan nodes.
    `scanPre` carries a pre-built step morphism (the `Stmt.recurMorphism` escape hatch, E2c);
    the trailing `Bool` on `scan` is the ScanAffine flag. -/
inductive ScanStmt
  | plain   : Stmt → ScanStmt
  | scan    : String → AxisSpec → List Stmt → List Stmt → Bool → ScanStmt  -- final Bool = isAffine
  | scanPre : String → AxisSpec → ThreadedComposed → ScanStmt              -- recurMorphism case
  deriving Inhabited

structure LabeledProgram where
  decls : List Decl
  stmts : List Stmt           -- every AxisSpec.uid is fresh & non-zero

structure ResolvedProgram where
  decls      : List Decl
  stmts      : List Stmt
  env        : DeclEnv
  extNames   : Finset String  -- externally declared (input) tensor names
  extraStmts : Array Stmt     -- bias-add stmts for `linear … bias`

structure CanonicalProgram where
  decls    : List Decl
  stmts    : List Stmt
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec  -- canonical axis equivalence classes

structure LoweredProgram where
  decls    : List Decl
  stmts    : List Stmt         -- no const/affine IdxExprs in reads
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec
  auxStmts : Array Stmt        -- Slice/Reindex/Scatter intermediates

structure ScanProgram where
  decls    : List Decl
  stmts    : List ScanStmt     -- iterAt/iterNext grouped
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec

structure LinearProgram where
  decls    : List Decl
  stmts    : List ScanStmt     -- no nonlinearity in RHSExpr.nonlin
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec

structure ScheduledProgram where
  decls    : List Decl
  stmts    : List ScanStmt     -- live stmts, topological order: producers precede consumers
  env      : DeclEnv
  extNames : Finset String
  ctx      : Context AxisSpec
  explicitSizes : HashMap UID Nat  -- axis sizes pinned by `axis … = n` decls (seed for inferAxisSizes)

example : ScanStmt := .plain (.assign "x" [] { body := { terms := [] }, nonlin := .identity })

end LeanNCD
