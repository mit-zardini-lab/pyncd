namespace LeanNCD.Eval

/-- Errors any evaluator phase can raise (shared across the Eval/ modules). -/
abbrev EvalError := String

/-- A dense row-major Float tensor. Invariant: `data.size = (shape.foldl (·*·) 1)`. -/
structure DenseTensor where
  shape : List Nat
  data  : Array Float
  deriving Repr, Inhabited

namespace DenseTensor

/-- number of elements of a shape. -/
def sizeOf (shape : List Nat) : Nat := shape.foldl (· * ·) 1

/-- all-zeros tensor of a shape. -/
def zeros (shape : List Nat) : DenseTensor := ⟨shape, Array.replicate (sizeOf shape) 0.0⟩

/-- row-major strides: strides[k] = product of dims AFTER position k.
    e.g. shape [2,3,4] ⇒ [12,4,1]. -/
def strides (shape : List Nat) : List Nat :=
  (shape.foldr (fun d (acc : Nat × List Nat) => (acc.1 * d, acc.1 :: acc.2)) (1, [])).2

/-- flat index of a multi-index `coord` (must have the same length as `shape`). -/
def flatIdx (shape coord : List Nat) : Nat :=
  ((strides shape).zip coord).foldl (fun a (s, c) => a + s * c) 0

/-- read element at `coord` (out-of-range defaults to `0.0`; `Array.get!` is gone in v4.30). -/
def get! (t : DenseTensor) (coord : List Nat) : Float := t.data.getD (flatIdx t.shape coord) 0.0

/-- write element at `coord`. -/
def set! (t : DenseTensor) (coord : List Nat) (v : Float) : DenseTensor :=
  ⟨t.shape, t.data.set! (flatIdx t.shape coord) v⟩

/-- enumerate every multi-index of a shape, row-major order (reused by ofFn + later tasks). -/
partial def allCoords (shape : List Nat) : List (List Nat) :=
  match shape with
  | []      => [[]]
  | d :: ds =>
      let rest := allCoords ds
      (List.range d).flatMap (fun i => rest.map (fun c => i :: c))

/-- build a tensor from a coordinate → value function (row-major). -/
def ofFn (shape : List Nat) (f : List Nat → Float) : DenseTensor :=
  ⟨shape, ((allCoords shape).map f).toArray⟩

/-- elementwise approximate equality (tolerance) — for tests; Float isn't exactly comparable. -/
def approxEq (a b : DenseTensor) (ε : Float := 1e-6) : Bool :=
  a.shape == b.shape && a.data.size == b.data.size &&
    (a.data.zip b.data).all (fun (x, y) => Float.abs (x - y) < ε)

end DenseTensor
end LeanNCD.Eval
