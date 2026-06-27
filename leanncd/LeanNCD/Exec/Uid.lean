namespace LeanNCD

/-- Human-readable, LaTeX-renderable display name for an axis/array. Display-only and out of scope
    (§13) — a thin `String` wrapper kept so `UData.name` typechecks. -/
abbrev DynamicName := String

/-- A unique identity for a symbolic axis. Carries no semantic content — only equality/inequality
    of two UIDs matters. -/
abbrev UID := Nat

/-- A UID paired with its optional display name. -/
structure UData where
  uid  : UID
  name : Option DynamicName := none
  deriving Repr, DecidableEq, Inhabited

/-- Errors that `FreshM` compilation can throw (the §7.4 / §12 validation failures). -/
inductive CompileError
  | shapeMismatch        : String → String → CompileError   -- expected shape, actual shape
  | missingBaseCase      : String → CompileError            -- tensor name missing iterAt stmt
  | causalityViolation   : String → CompileError            -- l+1 on RHS for iteration axis l
  | overlappingScatter   : String → CompileError            -- non-injective scatter w/o reduce sum
  | linearWeightAmbiguous : String → CompileError           -- linear weight in ≠1 product factors
  | undeclaredName       : String → CompileError            -- name used but not declared
  | rankMismatch         : String → Nat → Nat → CompileError -- tensor name, expected rank, actual rank
  | iterAxisNotNat       : String → CompileError            -- axis used in iterAt/iterNext but kind is ℝ
  | normAxisNotReal      : String → CompileError            -- axis used in freeNorm but kind is ℕ
  | predicateNonlin      : String → CompileError            -- predicate tensor with non-identity nonlin
  | predicateAgg         : String → CompileError            -- predicate tensor with non-sum aggregation
  | cyclicDataflow       : String → CompileError            -- cyclic dataflow (topoSort cycle fallback)
  deriving Repr, DecidableEq

/-- Combined error + UID-counter monad (`EStateM ε σ α = σ → Result ε σ α`, Lean core). Mints fresh
    UIDs and validates structure; the executable side of the seam (it computes, proves nothing). -/
abbrev FreshM := EStateM CompileError Nat

/-- Mint a fresh UID (a counter, not random ints — reproducible and testable). -/
def freshUData : FreshM UData := do
  let n ← get
  set (n + 1)
  return ⟨n, none⟩

end LeanNCD
