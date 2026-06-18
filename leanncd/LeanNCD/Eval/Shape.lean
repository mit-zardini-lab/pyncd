import LeanNCD.Eval.Tensor
import LeanNCD.DSL.Ast
import Std.Data.HashMap

namespace LeanNCD.Eval
open Std

/-- The LHS slots of a statement (assign/scatter; `recurMorphism` has none here). -/
def Stmt.lhsSlots : Stmt → List LHSSlot
  | .assign _ ls _    => ls
  | .scatter _ ls _ _ => ls
  | .recurMorphism _ _ _ => []

/-- All `(name, [idxExprs])` reads in a stmt's RHS (assign/scatter; recurMorphism has none here). -/
def Stmt.readsOf : Stmt → List (String × List IdxExpr)
  | .assign _ _ r | .scatter _ _ r _ =>
      r.body.terms.flatMap (fun t => t.factors.filterMap (fun
        | .read nm es => some (nm, es)
        | .iverson _  => none))
  | .recurMorphism _ _ _ => []

/-- Normalize an `IdxExpr` to a canonical integer-affine form `(c0, [(coef, uid)])`. -/
def idxAffineForm : IdxExpr → Int × List (Int × UID)
  | .axis a      => (0, [(1, a.uid)])
  | .const n     => (n, [])
  | .scale c a   => (0, [(c, a.uid)])
  | .shift a n   => (n, [(1, a.uid)])
  | .affine n xs => (n, xs.map (fun (c, a) => (c, a.uid)))

/-- Infer axis-UID → concrete size from the input tensors + read positions.
    For a read `name[e₁,…,eₘ]` whose input `env[name]` has shape `[d₁,…,dₘ]`, each `eᵢ`
    is treated as the integer-affine map `c0 + Σ cₖ·aₖ`. A bare `.axis a` (the common case)
    binds `a.uid ↦ dᵢ`. For a richer affine position with exactly ONE unknown axis `a`
    (coefficient `cₐ > 0`) and all other axes already sized, we infer `a`'s size by the
    standard "valid"/largest-in-range rule: the maximal source index `c0 + Σ cₖ·(sizeₖ-1)`
    must stay `< dᵢ`, so `size(a) = ⌊(dᵢ-1 - c0 - Σ_{k≠a} cₖ·(sizeₖ-1)) / cₐ⌋ + 1`.
    (For a bare axis this reduces to `size(a) = dᵢ`; for `X[i-1]` it gives `dᵢ+1`; for a
    stride-`c` read `X[c·j]` it gives `⌊(dᵢ-1)/c⌋+1`. This is what lets the strided-conv
    and look-back examples size their purely-affine output axes.)
    A position with ≥2 still-unknown axes is left for a later fixpoint pass.
    Conflicting sizes for one UID ⇒ error. We iterate to a fixpoint so inference order
    (e.g. a kernel axis sizing before the dotted output axis) does not matter.
    `seed` pre-binds axes pinned by `axis … = n` decls; inference treats them as already
    known (and a later read implying a different size conflicts, as for any bound UID). -/
def inferAxisSizes (seed : HashMap UID Nat) (env : HashMap String DenseTensor)
    (stmts : List Stmt) : Except EvalError (HashMap UID Nat) := do
  -- collect every (affine-form, dim) read position once
  let positions : List ((Int × List (Int × UID)) × Nat) := stmts.flatMap (fun s =>
    (Stmt.readsOf s).flatMap (fun (nm, es) =>
      match env[nm]? with
      | none   => []
      | some t => (es.zip t.shape).map (fun (e, d) => (idxAffineForm e, d))))
  let mut sizes : HashMap UID Nat := seed
  -- fixpoint: repeat passes until no new UID is bound (bounded by #positions iterations)
  for _ in List.range (positions.length + 1) do
    let mut changed := false
    for ((c0, terms), d) in positions do
      -- partition into known-sized vs unknown axes
      let unknown := terms.filter (fun (_, u) => ! (sizes.contains u))
      match unknown with
      | [(coef, u)] =>
          if coef > 0 then
            -- Σ over the OTHER (known) axes of cₖ·(sizeₖ-1)
            let known := terms.filter (fun (_, v) => sizes.contains v)
            let other := known.foldl (fun acc (c, v) =>
              acc + c * (Int.ofNat ((sizes[v]?).getD 0) - 1)) 0
            let numer := (Int.ofNat d - 1) - c0 - other
            let sz := (numer / coef + 1)
            let szN := (max sz 0).toNat
            match sizes[u]? with
            | some d' => if d' != szN then throw s!"axis size conflict for uid {u}: {d'} vs {szN}"
            | none    => sizes := sizes.insert u szN; changed := true
      | [] =>
          -- fully-known: a bare single-axis position `name[a]` must match its dim exactly
          -- (preserves the original conflicting-bare-read detection).
          match c0, terms with
          | 0, [(1, u)] =>
              match sizes[u]? with
              | some d' => if d' != d then throw s!"axis size conflict for uid {u}: {d'} vs {d}"
              | none    => pure ()
          | _, _ => pure ()
      | _ => pure ()   -- ≥2 unknowns: defer to the next fixpoint pass
    unless changed do break
  return sizes

/-- The axis UID of an LHS slot, if it has one (`affine` slots do not). -/
def lhsAxisUID? : LHSSlot → Option UID
  | .free a     => some a.uid
  | .freeNorm a => some a.uid
  | .iterAt a _ => some a.uid
  | .iterNext a => some a.uid
  | .affine _   => none

/-- The UID of the slot marked (`m.`) as the softmax/normalize reduction axis, if any.
    This is how the reduction axis is identified for a stmt (the norm flag moved off the
    tensor decl onto the output slot); `none` means no axis was marked. -/
def normAxisUidOf (slots : List LHSSlot) : Option UID :=
  slots.findSome? (fun | .freeNorm a => some a.uid | _ => none)

/-- The output shape: the size of each LHS slot's axis (free/iterAt/iterNext), in slot order.
    `affine` slots (scatter) are handled in Task 6 — `0` placeholder here. -/
def outputShape (sizes : HashMap UID Nat) (slots : List LHSSlot) : List Nat :=
  slots.map (fun sl => match lhsAxisUID? sl with
    | some u => (sizes[u]?).getD 0
    | none   => 0)

end LeanNCD.Eval
