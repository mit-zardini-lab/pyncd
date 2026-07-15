import LeanNCD.DSL.Ast
import LeanNCD.DSL.Target
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
  | scan    : String → List AxisSpec → List Stmt → List Stmt → Bool → ScanStmt  -- axis list, final Bool = isAffine
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

structure LoweredProgram where
  decls    : List Decl
  stmts    : List Stmt         -- no const/affine IdxExprs in reads
  env      : DeclEnv
  extNames : Finset String

structure ScanProgram where
  decls    : List Decl
  stmts    : List ScanStmt     -- iterAt/iterNext grouped
  env      : DeclEnv
  extNames : Finset String

structure LinearProgram where
  decls    : List Decl
  stmts    : List ScanStmt     -- no nonlinearity in RHSExpr.nonlin
  env      : DeclEnv
  extNames : Finset String

structure ScheduledProgram where
  decls    : List Decl
  stmts    : List ScanStmt     -- live stmts, topological order: producers precede consumers
  env      : DeclEnv
  extNames : Finset String
  explicitSizes : HashMap UID Nat  -- axis sizes pinned by `axis … = n` decls (seed for inferAxisSizes)

example : ScanStmt := .plain (.assign "x" [] { body := { terms := [] }, nonlin := .identity })

end LeanNCD
