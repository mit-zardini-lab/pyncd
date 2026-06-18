import LeanNCD.Eval.Tensor
import LeanNCD.Eval.Gather
namespace LeanNCD.Eval
open Std

/-- relu: elementwise max(0, x). -/
def reluT (t : DenseTensor) : DenseTensor := ⟨t.shape, t.data.map (fun x => max 0.0 x)⟩

/-- Build the `coord : HashMap UID Int` for a full multi-index `c`, pairing each axis position's
    UID with its coordinate value (so `evalBool` can see the coordinate for masking). -/
private def coordMap (axisUids : List UID) (c : List Nat) : HashMap UID Int :=
  (axisUids.zip c).foldl (fun m (u, v) => m.insert u (v : Int)) ({} : HashMap UID Int)

/-- Group the tensor's coordinates into "rows" along `axisPos`: each row is the list of full
    coordinates that agree on all axes except `axisPos` (varying that axis 0..dim-1). Returns a
    list of rows, each a list of `(coord, flatIdx)`. -/
private def rowsAlong (axisPos : Nat) (shape : List Nat) : List (List (List Nat × Nat)) :=
  let coords := DenseTensor.allCoords shape
  -- row key = coord with position `axisPos` removed.
  let keyed := coords.map (fun c => (c.eraseIdx axisPos, c, DenseTensor.flatIdx shape c))
  -- group by key, preserving first-seen order of keys.
  let keys := (keyed.map (·.1)).foldl
    (fun acc k => if acc.contains k then acc else acc ++ [k]) []
  keys.map (fun k =>
    (keyed.filter (fun e => e.1 == k)).map (fun e => (e.2.1, e.2.2)))

/-- Apply a per-row normalization `f` (given the list of (value, masked?) entries it returns the
    list of output values, in the same order) along `axisPos`, with optional mask. -/
private def perRow (axisPos : Nat) (axisUids : List UID) (mask? : Option BoolExpr)
    (f : List (Float × Bool) → List Float) (t : DenseTensor) : DenseTensor :=
  let rows := rowsAlong axisPos t.shape
  rows.foldl (fun acc row =>
    let entries : List (Float × Bool) := row.map (fun (c, fi) =>
      let x := acc.data.getD fi 0.0
      let masked := match mask? with
        | none => false
        | some b => ! evalBool (coordMap axisUids c) b
      (x, masked))
    let ys := f entries
    -- write outputs back at each flatIdx.
    ((row.zip ys).foldl (fun (cur : DenseTensor) ((_, fi), y) => ⟨cur.shape, cur.data.set! fi y⟩) acc))
    t

/-- softmax along `axisPos`, with an optional mask. For each row: masked entries are excluded
    (treated as -∞ ⇒ exp 0); `y = exp(x - rowMax) / Σ exp(x - rowMax)` over unmasked entries.
    The `mask?`/`axisUids` let `evalBool` see the coordinate (axis-UID → Int) for masking. -/
def softmaxT (axisPos : Nat) (axisUids : List UID) (mask? : Option BoolExpr)
    (t : DenseTensor) : DenseTensor :=
  perRow axisPos axisUids mask? (fun entries =>
    let unmasked := entries.filterMap (fun (x, m) => if m then none else some x)
    let m := match unmasked with
      | []      => 0.0
      | x :: xs => xs.foldl (fun a b => max a b) x
    let es := entries.map (fun (x, masked) => if masked then 0.0 else Float.exp (x - m))
    let s := es.foldl (· + ·) 0.0
    (entries.zip es).map (fun ((_, masked), e) =>
      if masked || s == 0.0 then 0.0 else e / s)) t

/-- normalize along `axisPos` (+ optional mask): y = x / Σ x over unmasked entries in the row. -/
def normalizeT (axisPos : Nat) (axisUids : List UID) (mask? : Option BoolExpr)
    (t : DenseTensor) : DenseTensor :=
  perRow axisPos axisUids mask? (fun entries =>
    let s := entries.foldl (fun a (x, masked) => if masked then a else a + x) 0.0
    entries.map (fun (x, masked) =>
      if masked || s == 0.0 then 0.0 else x / s)) t

/-- Dispatch: apply a Nonlin. `axisPos`/`axisUids` describe the norm axis (ignored by relu/identity). -/
def applyNonlin (nl : Nonlin) (axisPos : Nat) (axisUids : List UID)
    (t : DenseTensor) : DenseTensor :=
  match nl with
  | .identity      => t
  | .relu          => reluT t
  | .softmax m     => softmaxT axisPos axisUids m t
  | .normalize m   => normalizeT axisPos axisUids m t

end LeanNCD.Eval
