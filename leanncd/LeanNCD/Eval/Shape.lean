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

structure SizeConstraint where
  coeffs : List (Int × UID)
  rhs : Int
  source : String

private structure AffinePosition where
  coeffs : List (Int × UID)
  const : Int
  dim : Nat
  source : String

private structure RatRow where
  coeffs : Array Rat
  rhs : Rat
  sources : List String := []
  deriving Inhabited

private inductive SolveFailureKind
  | inconsistent
  | underdetermined
  | nonIntegral
  | nonPositive

private structure SolveDiagnostic where
  kind : SolveFailureKind
  unconstrained : List UID := []
  offendingUid? : Option UID := none
  detail? : Option String := none
  sourceRefs : List String := []
  remediation : List String := []
  mlHints : List String := []

private def insertStringSortedUnique (acc : List String) (s : String) : List String :=
  match acc with
  | [] => [s]
  | x :: xs =>
      if s == x then
        acc
      else if s < x then
        s :: acc
      else
        x :: insertStringSortedUnique xs s

private def mlHintsOfConstraints (constraints : List SizeConstraint) : List String :=
  let hints0 : List String := []
  let hints1 := if constraints.length ≥ 2 then
      insertStringSortedUnique hints0 "ml-hint: coupled affine constraints"
    else
      hints0
  let hints2 := if constraints.any (fun c => c.coeffs.length ≥ 2) then
      insertStringSortedUnique hints1 "ml-hint: multi-axis window constraint"
    else
      hints1
  if constraints.any (fun c => c.coeffs.any (fun (coef, _) => coef.natAbs > 1)) then
    insertStringSortedUnique hints2 "ml-hint: stride/dilation-like coefficients"
  else
    hints2

private def mergeStringLists (xs ys : List String) : List String :=
  ys.foldl insertStringSortedUnique xs

private def uidConstraintCount (constraints : List SizeConstraint) (u : UID) : Nat :=
  constraints.foldl (fun acc c => if c.coeffs.any (fun (_, v) => v == u) then acc + 1 else acc) 0

private def uidConstraintSources (constraints : List SizeConstraint) (u : UID) : List String :=
  constraints.foldl
    (fun acc c => if c.coeffs.any (fun (_, v) => v == u) then insertStringSortedUnique acc c.source else acc)
    []

private def renderUnderdeterminedDetail (constraints : List SizeConstraint) (freeUids : List UID)
    (rank vars : Nat) : String :=
  let uidParts := freeUids.map
    (fun u => s!"uid {u}: in {uidConstraintCount constraints u} equation(s), sources={uidConstraintSources constraints u}")
  s!"rank={rank}, vars={vars}, {uidParts}"

private def remediationOfDiagnostic (d : SolveDiagnostic) : List String :=
  match d.kind with
  | .inconsistent =>
      [ "verify cited tensor dimensions and affine offsets are mutually consistent"
      , "if mismatch is intentional, split equations across distinct output axes" ]
  | .underdetermined =>
      [ "bind unconstrained axes with explicit axis declarations"
      , "add independent affine reads so rank matches variable count" ]
  | .nonIntegral =>
      [ "adjust stride/dilation-like coefficients to satisfy divisibility"
      , "shift constants or input extents so inferred size is integral" ]
  | .nonPositive =>
      [ "increase effective input extent or reduce negative shifts"
      , "ensure inferred output window size stays strictly positive" ]

private def renderSolveDiagnostic (d : SolveDiagnostic) : String :=
  let base := match d.kind with
    | .inconsistent => "affine size system inconsistent"
    | .underdetermined => s!"affine size system underdetermined (unconstrained uids: {d.unconstrained})"
    | .nonIntegral =>
        match d.offendingUid? with
        | some u => s!"affine size system non-integral (uid {u})"
        | none   => "affine size system non-integral"
    | .nonPositive =>
        match d.offendingUid? with
        | some u => s!"affine size system non-positive (uid {u})"
        | none   => "affine size system non-positive"
  let withDetail := match d.detail? with
    | some txt => s!"{base} ({txt})"
    | none => base
  let withSources := if d.sourceRefs.isEmpty then
      withDetail
    else
      s!"{withDetail}; sources: {d.sourceRefs}"
  let remediation := if d.remediation.isEmpty then remediationOfDiagnostic d else d.remediation
  let withRemediation := if remediation.isEmpty then
      withSources
    else
      s!"{withSources}; actions: {remediation}"
  if d.mlHints.isEmpty then
    withRemediation
  else
    s!"{withRemediation}; ml-hints: {d.mlHints}"

private def addCoeff (acc : List (Int × UID)) (coef : Int) (u : UID) : List (Int × UID) :=
  match acc with
  | [] => if coef == 0 then [] else [(coef, u)]
  | (c, v) :: rest =>
      if v == u then
        let c' := c + coef
        if c' == 0 then rest else (c', v) :: rest
      else
        (c, v) :: addCoeff rest coef u

private def insertCoeffSorted (acc : List (Int × UID)) (entry : Int × UID) : List (Int × UID) :=
  match acc with
  | [] => [entry]
  | hd :: tl =>
      if entry.2 < hd.2 then
        entry :: acc
      else
        hd :: insertCoeffSorted tl entry

private def sortCoeffs (coeffs : List (Int × UID)) : List (Int × UID) :=
  coeffs.foldl (fun acc entry => insertCoeffSorted acc entry) []

private def normalizeCoeffs (coeffs : List (Int × UID)) : List (Int × UID) :=
  sortCoeffs (coeffs.foldl (fun acc (coef, u) => addCoeff acc coef u) [])

private def upperEnvelopeCoeffs (coeffs : List (Int × UID)) : List (Int × UID) :=
  normalizeCoeffs (coeffs.filterMap (fun (coef, u) => if coef > 0 then some (coef, u) else none))

private def insertSolvedSize (sizes : HashMap UID Nat) (u : UID) (sz : Nat) :
    Except EvalError (HashMap UID Nat) :=
  match sizes[u]? with
  | some d' =>
      if d' == sz then
        return sizes
      else
        throw s!"axis size conflict for uid {u}: {d'} vs {sz}"
  | none => return sizes.insert u sz

private def canonicalizeConstraint (constraint : SizeConstraint) : SizeConstraint :=
  let coeffs0 := normalizeCoeffs constraint.coeffs
  let g := coeffs0.foldl (fun acc (coef, _) => Nat.gcd acc coef.natAbs) constraint.rhs.natAbs
  let g := if g == 0 then 1 else g
  let div := Int.ofNat g
  let coeffs1 := if g == 1 then coeffs0 else coeffs0.map (fun (coef, u) => (coef / div, u))
  let rhs1 := if g == 1 then constraint.rhs else constraint.rhs / div
  let lead : Int := match coeffs1.find? (fun (coef, _) => coef != 0) with
    | some (coef, _) => coef
    | none => rhs1
  let negLead := match lead with
    | Int.negSucc _ => true
    | _ => false
  let (coeffs2, rhs2) := if negLead then
      (coeffs1.map (fun (coef, u) => (-coef, u)), -rhs1)
    else
      (coeffs1, rhs1)
  { constraint with coeffs := sortCoeffs coeffs2, rhs := rhs2 }

private def mkConstraint (pos : AffinePosition) : SizeConstraint :=
  let rhs := Int.ofNat pos.dim - pos.const + pos.coeffs.foldl (fun acc (coef, _) => acc + coef) 0 - 1
  canonicalizeConstraint { coeffs := pos.coeffs, rhs, source := pos.source }

private def substituteKnownSizes (sizes : HashMap UID Nat) (constraint : SizeConstraint) : SizeConstraint :=
  let (rhs, coeffsRev) := constraint.coeffs.foldl
    (fun (state : Int × List (Int × UID)) (coef, u) =>
      match sizes[u]? with
      | some sz => (state.1 - coef * Int.ofNat sz, state.2)
      | none    => (state.1, (coef, u) :: state.2))
    (constraint.rhs, [])
  canonicalizeConstraint { constraint with rhs, coeffs := normalizeCoeffs coeffsRev.reverse }

private def ratOfInt (n : Int) : Rat := Rat.normalize n

private def ratRowScale (factor : Rat) (row : RatRow) : RatRow :=
  { coeffs := row.coeffs.map (fun q => factor * q), rhs := factor * row.rhs, sources := row.sources }

private def ratRowSub (row pivot : RatRow) (factor : Rat) : RatRow :=
  let coeffs := Id.run do
    if row.coeffs.size ≠ pivot.coeffs.size then
      panic! s!"ratRowSub: row/pivot size mismatch ({row.coeffs.size} ≠ {pivot.coeffs.size})"
    let mut out := Array.mkEmpty row.coeffs.size
    for i in List.range row.coeffs.size do
      out := out.push (row.coeffs[i]! - factor * pivot.coeffs[i]!)
    return out
  { coeffs, rhs := row.rhs - factor * pivot.rhs, sources := mergeStringLists row.sources pivot.sources }

private def swapRows (rows : Array RatRow) (i j : Nat) : Array RatRow :=
  if i == j then
    rows
  else
    let ri := rows[i]!
    let rj := rows[j]!
    (rows.set! i rj).set! j ri

private def findPivotRow? (rows : Array RatRow) (start col : Nat) : Option Nat :=
  (List.range rows.size).drop start |>.find? (fun i => rows[i]!.coeffs[col]! != 0)

private def rowIsZero (row : RatRow) : Bool :=
  row.coeffs.all (fun q => q == 0)

private def sameEquation (a b : RatRow) : Bool :=
  a.rhs == b.rhs && a.coeffs == b.coeffs

private def findEquivalentRow? (rows : Array RatRow) (target : RatRow) : Option Nat :=
  (List.range rows.size).find? (fun i => sameEquation rows[i]! target)

private def insertUIDSorted (acc : List UID) (u : UID) : List UID :=
  match acc with
  | [] => [u]
  | x :: xs =>
      if u == x then
        acc
      else if u < x then
        u :: acc
      else
        x :: insertUIDSorted xs u

private def sortedVarsOfConstraints (constraints : List SizeConstraint) : List UID :=
  constraints.foldl
    (fun acc constraint =>
      constraint.coeffs.foldl (fun acc' (_, u) => insertUIDSorted acc' u) acc)
    []

private def insertConstraintSorted (acc : List SizeConstraint) (constraint : SizeConstraint) : List SizeConstraint :=
  let key := s!"{constraint.source}|{constraint.rhs}|{repr constraint.coeffs}"
  match acc with
  | [] => [constraint]
  | hd :: tl =>
      let keyH := s!"{hd.source}|{hd.rhs}|{repr hd.coeffs}"
      if key < keyH then
        constraint :: acc
      else
        hd :: insertConstraintSorted tl constraint

private def sortConstraints (constraints : List SizeConstraint) : List SizeConstraint :=
  constraints.foldl (fun acc c => insertConstraintSorted acc c) []

private def solveSizeConstraints (constraints : List SizeConstraint) :
    Except EvalError (HashMap UID Nat) := do
  let constraints := sortConstraints (constraints.map canonicalizeConstraint)
  let mlHints := mlHintsOfConstraints constraints
  let vars := sortedVarsOfConstraints constraints
  if vars.isEmpty then
    return {}
  let varIndex := Id.run do
    let mut idx : HashMap UID Nat := {}
    for i in List.range vars.length do
      idx := idx.insert vars[i]! i
    return idx
  let rows := Id.run do
    let mut out := Array.mkEmpty constraints.length
    for constraint in constraints do
      let mut coeffs := Array.replicate vars.length (0 : Rat)
      for (coef, u) in constraint.coeffs do
        let i := (varIndex[u]?).get!
        coeffs := coeffs.set! i (coeffs[i]! + ratOfInt coef)
      let row : RatRow := { coeffs, rhs := ratOfInt constraint.rhs, sources := [constraint.source] }
      match findEquivalentRow? out row with
      | some i =>
          let merged := { row with sources := mergeStringLists out[i]!.sources row.sources }
          out := out.set! i merged
      | none =>
          out := out.push row
    return out
  let mut matrix := rows
  let mut pivotRow := 0
  let mut pivots : Array (Nat × Nat) := #[]
  for col in List.range vars.length do
    if pivotRow < matrix.size then
      match findPivotRow? matrix pivotRow col with
      | none => pure ()
      | some found =>
          matrix := swapRows matrix pivotRow found
          let pivot := matrix[pivotRow]!
          let lead := pivot.coeffs[col]!
          let pivot := ratRowScale ((1 : Rat) / lead) pivot
          matrix := matrix.set! pivotRow pivot
          for row in List.range matrix.size do
            if row != pivotRow then
              let coeff := matrix[row]!.coeffs[col]!
              if coeff != 0 then
                matrix := matrix.set! row (ratRowSub matrix[row]! pivot coeff)
          pivots := pivots.push (pivotRow, col)
          pivotRow := pivotRow + 1
  for row in matrix do
    if rowIsZero row && row.rhs != 0 then
      throw (renderSolveDiagnostic
        { kind := .inconsistent
        , detail? := some s!"reduced witness: 0 = {row.rhs}"
        , sourceRefs := row.sources
        , mlHints })
  let pivotColSet := pivots.foldl (fun acc (_, col) => acc.insert col) ({} : HashSet Nat)
  let freeCols := (List.range vars.length).filter (fun i => !(pivotColSet.contains i))
  if !freeCols.isEmpty then
    let freeUids := freeCols.map (fun i => vars[i]!)
    let refs := freeUids.foldl (fun acc u => mergeStringLists acc (uidConstraintSources constraints u)) []
    throw (renderSolveDiagnostic
      { kind := .underdetermined
      , unconstrained := freeUids
      , detail? := some (renderUnderdeterminedDetail constraints freeUids pivots.size vars.length)
      , sourceRefs := refs
      , mlHints })
  -- Phase 1: floor all RREF solutions (padded semantics: non-divisible cases floor down)
  let mut solved : HashMap UID Nat := {}
  for (rowIdx, col) in pivots do
    let value := matrix[rowIdx]!.rhs
    let flooredNum : Int := Rat.num value / Int.ofNat (Rat.den value)
    if flooredNum <= 0 then
      let hints := insertStringSortedUnique mlHints "ml-hint: offset/window yields non-positive extent"
      throw (renderSolveDiagnostic
        { kind := .nonPositive
        , offendingUid? := some vars[col]!
        , detail? := some s!"value={value}, reduced row={rowIdx}, col={col}"
        , sourceRefs := matrix[rowIdx]!.sources
        , mlHints := hints })
    solved := solved.insert vars[col]! flooredNum.toNat
  -- Phase 2: verify floored solution satisfies all original constraints as inequalities
  for constraint in constraints do
    let lhs := constraint.coeffs.foldl
      (fun acc (coef, u) => acc + coef * Int.ofNat ((solved[u]?).get!)) 0
    if lhs > constraint.rhs then
      let hints := insertStringSortedUnique mlHints "ml-hint: output extent divisibility mismatch"
      throw (renderSolveDiagnostic
        { kind := .nonIntegral
        , detail? := some s!"floored solution violates constraint: {lhs} > {constraint.rhs} ({constraint.source})"
        , sourceRefs := [constraint.source]
        , mlHints := hints })
  return solved

/-- Infer axis-UID → concrete size from the input tensors + read positions.
    For a read `name[e₁,…,eₘ]` whose input `env[name]` has shape `[d₁,…,dₘ]`, each `eᵢ`
    is treated as the integer-affine map `c0 + Σ cₖ·aₖ`. A bare `.axis a` (the common case)
    binds `a.uid ↦ dᵢ`. For a richer affine position with one or more unknown axes, we build
    the upper-envelope constraint `Σ max(cₖ,0)·sizeₖ = d - c0 + Σ max(cₖ,0) - 1` and pass all
    remaining unknowns to an exact RREF solver over ℚ. Non-integral RREF solutions are floored
    and verified against all original constraints as inequalities (padded semantics: the maximal
    valid index need not sit exactly at `d-1`). This is the unified floor-then-verify convention.
    Conflicting sizes for one UID ⇒ error. We iterate to a fixpoint so inference order
    (e.g. a kernel axis sizing before the dotted output axis) does not matter.
    `seed` pre-binds axes pinned by `axis … = n` decls; inference treats them as already
    known (and a later read implying a different size conflicts, as for any bound UID).

    **Known gap (Issue H)**: when ALL axes in a multi-term read are already sized (fully-known
    position), the max-index `c0 + Σ max(cₖ,0)·(sizeₖ-1)` may exceed `dim-1`. Under padded
    semantics this is valid (out-of-range reads return 0) but is often surprising. A non-fatal
    warning is emitted for such positions; the second component of the return pair collects all
    such warnings. Only bare-axis positions (`name[a]`) receive a hard conflict check. -/
def inferAxisSizes (seed : HashMap UID Nat) (env : HashMap String DenseTensor)
    (stmts : List Stmt) : Except EvalError (HashMap UID Nat × List String) := do
  -- collect every (affine-form, dim) read position once
  let positions : List AffinePosition := stmts.flatMap (fun s =>
    (Stmt.readsOf s).flatMap (fun (nm, es) =>
      match env[nm]? with
      | none   => []
      | some t => (es.zip t.shape).map (fun (e, d) =>
          let (const, coeffs) := idxAffineForm e
          { coeffs := normalizeCoeffs coeffs, const, dim := d, source := s!"{nm}[{repr e}]" })))
  let mut sizes : HashMap UID Nat := seed
  let mut warns : List String := []
  -- fixpoint: build upper-envelope constraints for all unknown positions, solve jointly,
  -- then loop until no new sizes are learned.
  for _ in List.range (positions.length + 1) do
    let mut deferred : List SizeConstraint := []
    for pos in positions do
      let maxCoeffs := upperEnvelopeCoeffs pos.coeffs
      let unknown := maxCoeffs.filter (fun (_, u) => ! (sizes.contains u))
      match unknown with
      | [] =>
          match pos.const, pos.coeffs with
          | 0, [(1, u)] =>
              -- bare single-axis: hard conflict check
              match sizes[u]? with
              | some d' =>
                  if d' != pos.dim then
                    throw s!"axis size conflict for uid {u}: {d'} vs {pos.dim}"
              | none => pure ()
          | _, _ =>
              -- Issue H: fully-known multi-term (or negative-only) read.
              -- Under padded semantics this is valid, but warn if max-index ≥ dim.
              let maxIdx : Int := pos.const + maxCoeffs.foldl
                (fun acc (coef, u) => acc + coef * (Int.ofNat ((sizes[u]?).getD 0) - 1)) 0
              if maxIdx >= Int.ofNat pos.dim then
                warns := insertStringSortedUnique warns
                  s!"padded-access warning: {pos.source} max-index {maxIdx} ≥ dim {pos.dim}; out-of-range reads will be zero-padded"
      | _ =>
          deferred := deferred ++ [mkConstraint { pos with coeffs := maxCoeffs }]
    let remaining := (deferred.map (substituteKnownSizes sizes)).filter (fun c => !c.coeffs.isEmpty)
    if remaining.isEmpty then
      break
    let solved ← solveSizeConstraints remaining
    let mut solverChanged := false
    for (u, sz) in solved.toList do
      let next ← insertSolvedSize sizes u sz
      if !(sizes.contains u) then
        solverChanged := true
      sizes := next
    unless solverChanged do break
  -- Issue D: axes that appear in reads but only with non-positive upper-envelope
  -- coefficients are invisible to the solver. Detect and fail loud.
  let uidMaxCoeff : HashMap UID Int := positions.foldl (fun acc pos =>
    pos.coeffs.foldl (fun acc' (coef, u) =>
      acc'.insert u (max coef ((acc'[u]?).getD coef))) acc) {}
  let negUids := uidMaxCoeff.toList.foldl
    (fun acc (u, c) => if c <= 0 && !sizes.contains u then insertUIDSorted acc u else acc) []
  if let u :: _ := negUids then
    let srcs := positions.filterMap (fun pos =>
      if pos.coeffs.any (fun (_, v) => v == u) then some pos.source else none)
    throw s!"axis uid {u} is purely negatively constrained (appears only with non-positive coefficients in all reads; sources: {srcs}); add an explicit axis declaration"
  return (sizes, warns)

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
