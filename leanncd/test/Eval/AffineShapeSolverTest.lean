import LeanNCD.Eval.Eval

namespace LeanNCD.Eval
open Std

private def ax (nm : String) (u : Nat) : AxisSpec := { name := nm, uid := u, kind := .real none }
private def tensorOf (shape : List Nat) (xs : List Float) : DenseTensor := ⟨shape, xs.toArray⟩

-- Two affine read equations determine both output axes exactly:
--   X[i + j]   with X:[7]  => i + j = 8
--   U[i + 2*j] with U:[9]  => i + 2j = 11
-- so i = 5, j = 3.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [7])).insert "U" (DenseTensor.zeros [9])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(1, i), (1, j)]],
                                      .read "U" [.affine 0 [(1, i), (2, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError e
  | .ok (sizes, _) =>
      unless sizes[i.uid]? == some 5 && sizes[j.uid]? == some 3 do
        throwError s!"wrong solved sizes: {sizes[i.uid]?}, {sizes[j.uid]?}"
      unless outputShape sizes (Stmt.lhsSlots stmt) == [5, 3] do
        throwError s!"wrong output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"

-- A conv-like padded read: W[k] constrains the kernel axis; X[2h+k-1] then constrains h
-- jointly via the solver (unified floor-then-verify route).
run_cmd do
  let h := ax "h" 1
  let k := ax "k" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [8])).insert "W" (DenseTensor.zeros [3])
  let stmt : Stmt := .assign "Y" [.free h]
    { body := { terms := [{ factors := [.read "W" [.axis k], .read "X" [.affine (-1) [(2, h), (1, k)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError e
  | .ok (sizes, _) =>
      unless sizes[h.uid]? == some 4 && sizes[k.uid]? == some 3 do
        throwError s!"wrong conv-like sizes: {sizes[h.uid]?}, {sizes[k.uid]?}"
      unless outputShape sizes (Stmt.lhsSlots stmt) == [4] do
        throwError s!"wrong conv-like output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"

-- One affine equation over two free axes remains underdetermined.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor := ({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [7])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(1, i), (1, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e =>
      unless e.contains "underdetermined" do
        throwError s!"expected underdetermined error, got: {e}"
      unless e.contains "unconstrained uids: [2]" do
        throwError s!"expected sorted unconstrained UID list, got: {e}"
      unless e.contains "rank=1, vars=2" do
        throwError s!"expected underdetermined rank/vars detail, got: {e}"
      unless e.contains "sources:" do
        throwError s!"expected underdetermined source provenance, got: {e}"
      unless e.contains "actions:" do
        throwError s!"expected underdetermined remediation actions, got: {e}"
      unless e.contains "add independent affine reads" do
        throwError s!"expected underdetermined remediation guidance, got: {e}"
      unless e.contains "ml-hint: multi-axis window constraint" do
        throwError s!"expected ml diagnostic hint, got: {e}"
  | .ok (sizes, _) => throwError s!"expected underdetermined failure, got sizes {sizes.toList}"

-- Row order in factors should not change solved sizes.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [7])).insert "U" (DenseTensor.zeros [9])
  let stmtXU : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(1, i), (1, j)]],
                                      .read "U" [.affine 0 [(1, i), (2, j)]]] }] },
      nonlin := .identity }
  let stmtUX : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "U" [.affine 0 [(1, i), (2, j)]],
                                      .read "X" [.affine 0 [(1, i), (1, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmtXU], inferAxisSizes {} env [stmtUX] with
  | .ok (s1, _), .ok (s2, _) =>
      unless s1[i.uid]? == s2[i.uid]? && s1[j.uid]? == s2[j.uid]? do
        throwError s!"factor-order instability: {s1.toList} vs {s2.toList}"
  | .error e, _ => throwError s!"factor-order test first solve failed: {e}"
  | _, .error e => throwError s!"factor-order test second solve failed: {e}"

-- Duplicate affine terms should canonicalize deterministically.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [6])).insert "U" (DenseTensor.zeros [5])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(1, i), (1, i), (1, j)]],
                                      .read "U" [.affine 0 [(1, i), (1, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected duplicate-term success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[i.uid]? == some 2 && sizes[j.uid]? == some 4 do
        throwError s!"wrong duplicate-term sizes: {sizes[i.uid]?}, {sizes[j.uid]?}"

-- Two incompatible affine equations should surface as inconsistent.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [7])).insert "U" (DenseTensor.zeros [8])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(1, i), (1, j)]],
                                      .read "U" [.affine 0 [(1, i), (1, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e =>
      unless e.contains "inconsistent" do
        throwError s!"expected inconsistent error, got: {e}"
      unless e.contains "reduced witness: 0 =" do
        throwError s!"expected inconsistent reduced witness detail, got: {e}"
      unless e.contains "sources:" do
        throwError s!"expected inconsistent source provenance, got: {e}"
      unless e.contains "actions:" do
        throwError s!"expected inconsistent remediation actions, got: {e}"
      unless e.contains "verify cited tensor dimensions and affine offsets" do
        throwError s!"expected inconsistent remediation guidance, got: {e}"
  | .ok (sizes, _) => throwError s!"expected inconsistent failure, got sizes {sizes.toList}"

-- Non-integral RREF solutions are floored under padded semantics and verified as inequalities.
-- X[i+2j] with X:[5] → i+2j=7; U[i+4j] with U:[8] → i+4j=12.
-- RREF: j=5/2 (floor→2), i=2 (exact). Verify: 2+4=6≤7 ✓, 2+8=10≤12 ✓.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [5])).insert "U" (DenseTensor.zeros [8])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(1, i), (2, j)]],
                                      .read "U" [.affine 0 [(1, i), (4, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected non-integral floor success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[i.uid]? == some 2 && sizes[j.uid]? == some 2 do
        throwError s!"wrong floored non-integral sizes: {sizes[i.uid]?}, {sizes[j.uid]?}"
      unless outputShape sizes (Stmt.lhsSlots stmt) == [2, 2] do
        throwError s!"wrong floored non-integral output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"

-- Positive-only sizes are required for inferred axes.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [1])).insert "U" (DenseTensor.zeros [2])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(1, i), (1, j)]],
                                      .read "U" [.affine 0 [(1, i), (2, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e =>
      unless e.contains "non-positive" do
        throwError s!"expected non-positive error, got: {e}"
      unless e.contains "reduced row=" do
        throwError s!"expected non-positive reduced-row detail, got: {e}"
      unless e.contains "sources:" do
        throwError s!"expected non-positive source provenance, got: {e}"
      unless e.contains "actions:" do
        throwError s!"expected non-positive remediation actions, got: {e}"
      unless e.contains "ensure inferred output window size stays strictly positive" do
        throwError s!"expected non-positive remediation guidance, got: {e}"
      unless e.contains "ml-hint: offset/window yields non-positive extent" do
        throwError s!"expected non-positive ml hint, got: {e}"
  | .ok (sizes, _) => throwError s!"expected non-positive failure, got sizes {sizes.toList}"

-- Signed affine form: the solver uses the upper envelope of `3 - i + j`,
-- so this read constrains `j` while `U[i]` constrains `i`.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [8])).insert "U" (DenseTensor.zeros [4])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 3 [(-1, i), (1, j)]],
                                      .read "U" [.axis i]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected signed-affine success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[i.uid]? == some 4 && sizes[j.uid]? == some 5 do
        throwError s!"wrong signed-affine sizes: {sizes[i.uid]?}, {sizes[j.uid]?}"
      unless outputShape sizes (Stmt.lhsSlots stmt) == [4, 5] do
        throwError s!"wrong signed-affine output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"

-- End-to-end signed-affine evaluation under padded semantics.
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (tensorOf [8] [0,1,2,3,4,5,6,7])).insert "U"
      (tensorOf [4] [10,20,30,40])
  match TLProgram.eval (tlprog!{ Y[i, j] := X[j - i + 3] · U[i] }) env with
  | .error e => throwError s!"signed eval failed: {e}"
  | .ok out => match out["Y"]? with
    | some y =>
        unless y.shape == [4,5] do
          throwError s!"signed eval wrong shape: {repr y.shape}"
    | none => throwError "signed eval: no Y"

-- End-to-end evaluation uses the new inferred shape for a multi-equation affine program.
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (tensorOf [7] [0, 1, 2, 3, 4, 5, 6])).insert "U"
      (tensorOf [9] [0, 10, 20, 30, 40, 50, 60, 70, 80])
  match TLProgram.eval (tlprog!{ Y[i, j] := X[i + j] + U[i + 2 * j] }) env with
  | .error e => throwError s!"affine eval failed: {e}"
  | .ok out => match out["Y"]? with
    | some y =>
        unless DenseTensor.approxEq y (tensorOf [5, 3]
            [0, 21, 42,
             11, 32, 53,
             22, 43, 64,
             33, 54, 75,
             44, 65, 86]) do
          throwError s!"affine eval wrong: shape={repr y.shape} {repr y.data}"
    | none => throwError "affine eval: no Y"

-- End-to-end ML-shaped padded window: kernel width 3, stride 2, left shift 1.
run_cmd do
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "W" (tensorOf [3] [1, 10, 100])).insert "X"
      (tensorOf [8] [1, 2, 3, 4, 5, 6, 7, 8])
  match TLProgram.eval (tlprog!{ Y[h] := W[k] · X[2 * h + k - 1] }) env with
  | .error e => throwError s!"window eval failed: {e}"
  | .ok out => match out["Y"]? with
    | some y =>
        unless DenseTensor.approxEq y (tensorOf [4] [210, 432, 654, 876]) do
          throwError s!"window eval wrong: shape={repr y.shape} {repr y.data}"
    | none => throwError "window eval: no Y"

-- Phase 2D: 2D padded-window family should infer all output/kernel axes.
run_cmd do
  let h := ax "h" 1
  let w := ax "w" 2
  let kh := ax "kh" 3
  let kw := ax "kw" 4
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [10, 12])).insert "K"
      (DenseTensor.zeros [3, 5])
  let stmt : Stmt := .assign "Y" [.free h, .free w]
    { body := { terms := [{ factors := [.read "K" [.axis kh, .axis kw],
                                      .read "X" [.affine 0 [(1, h), (1, kh)], .affine 0 [(1, w), (1, kw)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected 2D padded-window success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[h.uid]? == some 8 && sizes[w.uid]? == some 8 do
        throwError s!"wrong 2D output sizes: {sizes[h.uid]?}, {sizes[w.uid]?}"
      unless sizes[kh.uid]? == some 3 && sizes[kw.uid]? == some 5 do
        throwError s!"wrong 2D kernel sizes: {sizes[kh.uid]?}, {sizes[kw.uid]?}"
      unless outputShape sizes (Stmt.lhsSlots stmt) == [8, 8] do
        throwError s!"wrong 2D output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"

-- Phase 2D: ND padded-window family (3D) should remain stable.
run_cmd do
  let t := ax "t" 1
  let h := ax "h" 2
  let w := ax "w" 3
  let kt := ax "kt" 4
  let kh := ax "kh" 5
  let kw := ax "kw" 6
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [7, 8, 9])).insert "W"
      (DenseTensor.zeros [2, 3, 4])
  let stmt : Stmt := .assign "Y" [.free t, .free h, .free w]
    { body := { terms := [{ factors := [.read "W" [.axis kt, .axis kh, .axis kw],
                                      .read "X" [.affine 0 [(1, t), (1, kt)],
                                                 .affine 0 [(1, h), (1, kh)],
                                                 .affine 0 [(1, w), (1, kw)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected 3D padded-window success, got: {e}"
  | .ok (sizes, _) =>
      unless outputShape sizes (Stmt.lhsSlots stmt) == [6, 6, 6] do
        throwError s!"wrong 3D output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"
      unless sizes[kt.uid]? == some 2 && sizes[kh.uid]? == some 3 && sizes[kw.uid]? == some 4 do
        throwError s!"wrong 3D kernel sizes: {sizes[kt.uid]?}, {sizes[kh.uid]?}, {sizes[kw.uid]?}"

-- Unified solver handles mixed-dimension constraints (bare-axis + conv-like + multi-unknown).
run_cmd do
  let h := ax "h" 1
  let i := ax "i" 2
  let j := ax "j" 3
  let k := ax "k" 4
  let env : HashMap String DenseTensor :=
    (((( {} : HashMap String DenseTensor).insert "D" (DenseTensor.zeros [8])).insert "X" (DenseTensor.zeros [7])).insert "U"
      (DenseTensor.zeros [9])).insert "K" (DenseTensor.zeros [3])
  let stmt : Stmt := .assign "Y" [.free h, .free i, .free j]
    { body := { terms := [{ factors := [.read "K" [.axis k],
                                      .read "D" [.affine (-1) [(2, h), (1, k)]],
                                      .read "X" [.affine 0 [(1, i), (1, j)]],
                                      .read "U" [.affine 0 [(1, i), (2, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected mixed-path inference success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[h.uid]? == some 4 && sizes[i.uid]? == some 5 && sizes[j.uid]? == some 3 do
        throwError s!"wrong mixed-path output sizes: {sizes[h.uid]?}, {sizes[i.uid]?}, {sizes[j.uid]?}"
      unless sizes[k.uid]? == some 3 do
        throwError s!"wrong mixed-path kernel size: {sizes[k.uid]?}"
      unless outputShape sizes (Stmt.lhsSlots stmt) == [4, 5, 3] do
        throwError s!"wrong mixed-path output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"

-- Phase 2E: redundant equalities should be pruned without changing the solve result.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    (({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [7])).insert "U" (DenseTensor.zeros [9])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [
        { factors := [.read "X" [.affine 0 [(1, i), (1, j)]], .read "U" [.affine 0 [(1, i), (2, j)]]] },
        { factors := [.read "X" [.affine 0 [(1, i), (1, j)]], .read "U" [.affine 0 [(1, i), (2, j)]]] },
        { factors := [.read "X" [.affine 0 [(1, i), (1, j)]], .read "U" [.affine 0 [(1, i), (2, j)]]] }
      ] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected redundant-equalities success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[i.uid]? == some 5 && sizes[j.uid]? == some 3 do
        throwError s!"wrong redundant-equalities sizes: {sizes[i.uid]?}, {sizes[j.uid]?}"
      unless outputShape sizes (Stmt.lhsSlots stmt) == [5, 3] do
        throwError s!"wrong redundant-equalities output shape: {outputShape sizes (Stmt.lhsSlots stmt)}"

-- Issue D: an axis with only negative upper-envelope coefficients in all reads is invisible
-- to the solver and must be declared explicitly (e.g. axis i = n).
run_cmd do
  let i := ax "i" 1
  let env : HashMap String DenseTensor :=
    ({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [5])
  let stmt : Stmt := .assign "Y" [.free i]
    { body := { terms := [{ factors := [.read "X" [.affine 3 [(-1, i)]]] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e =>
      unless e.contains "purely negatively constrained" do
        throwError s!"expected Issue-D negatively-constrained error, got: {e}"
      unless e.contains "uid 1" do
        throwError s!"expected uid in Issue-D error, got: {e}"
  | .ok (sizes, _) => throwError s!"expected Issue-D failure, got sizes {sizes.toList}"

-- Issue D with seed: an explicit axis declaration resolves the ambiguity.
run_cmd do
  let i := ax "i" 1
  let env : HashMap String DenseTensor :=
    ({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [5])
  let stmt : Stmt := .assign "Y" [.free i]
    { body := { terms := [{ factors := [.read "X" [.affine 3 [(-1, i)]]] }] },
      nonlin := .identity }
  match inferAxisSizes (({} : HashMap UID Nat).insert i.uid 4) env [stmt] with
  | .error e => throwError s!"expected Issue-D seeded success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[i.uid]? == some 4 do
        throwError s!"wrong Issue-D seeded size: {sizes[i.uid]?}"

-- Issue A regression: old fast-path would floor k=floor(7/3)+1=3 from U[3k], then pass k=3
-- into the 2-variable solver for (i,j), yielding 2i+j=10 and i+2j=12 → 3i=8 (non-integral
-- error). The unified route solves all three together: k=10/3, then 2i+j=9, i+2j=12 → i=2,
-- j=5 (both integral). Floor k to 3. Verify: 4+5+9=18≤19 ✓, 9≤10 ✓, 2+10=12≤12 ✓.
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let k := ax "k" 3
  let env : HashMap String DenseTensor :=
    ((({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [14])).insert "U"
       (DenseTensor.zeros [8])).insert "V" (DenseTensor.zeros [10])
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors :=
        [ .read "X" [.affine 0 [(2, i), (1, j), (3, k)]]
        , .read "U" [.scale 3 k]
        , .read "V" [.affine 0 [(1, i), (2, j)]] ] }] },
      nonlin := .identity }
  match inferAxisSizes {} env [stmt] with
  | .error e => throwError s!"expected Issue-A regression success, got: {e}"
  | .ok (sizes, _) =>
      unless sizes[i.uid]? == some 2 && sizes[j.uid]? == some 5 && sizes[k.uid]? == some 3 do
        throwError s!"wrong Issue-A sizes: i={sizes[i.uid]?} j={sizes[j.uid]?} k={sizes[k.uid]?}"

-- Issue H: fully-known multi-term reads are not bounds-checked (padded semantics: out-of-range = 0).
-- Axes i=4 and j=3 are provided via seed (explicit axis declarations). The read X[2*i+j] with
-- X:[6] is then fully-known: max-index = 2*(4-1)+(3-1) = 8 ≥ 6. Since all coefficients are
-- equalities in the solver, Issue H can only fire when sizes arrive via the seed rather than
-- from other solver constraints (a solver constraint on X[2*i+j] would be i+j-consistent
-- tight-fit and never produce max-index ≥ dim).
run_cmd do
  let i := ax "i" 1
  let j := ax "j" 2
  let env : HashMap String DenseTensor :=
    ({} : HashMap String DenseTensor).insert "X" (DenseTensor.zeros [6])
  let seed : HashMap UID Nat := (({} : HashMap UID Nat).insert i.uid 4).insert j.uid 3
  let stmt : Stmt := .assign "Y" [.free i, .free j]
    { body := { terms := [{ factors := [.read "X" [.affine 0 [(2, i), (1, j)]]] }] },
      nonlin := .identity }
  match inferAxisSizes seed env [stmt] with
  | .error e => throwError s!"expected Issue-H warning success, got: {e}"
  | .ok (sizes, warns) =>
      unless sizes[i.uid]? == some 4 && sizes[j.uid]? == some 3 do
        throwError s!"wrong Issue-H sizes: i={sizes[i.uid]?} j={sizes[j.uid]?}"
      unless warns.any (fun w => w.contains "padded-access") do
        throwError s!"expected padded-access warning in Issue-H, got warns={warns}"

end LeanNCD.Eval
