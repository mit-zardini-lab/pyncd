import LeanNCD.DSL.Ast

namespace LeanNCD.PropertyOracle

/-- Insert `x` at every position of `ys` (`ys.length + 1` results, one per insertion point). -/
def insertEverywhere {α} (x : α) : List α → List (List α)
  | []      => [[x]]
  | y :: ys => (x :: y :: ys) :: (insertEverywhere x ys).map (y :: ·)

/-- All permutations of a list (small n only — used on ≤4 statements). Structurally recursive on
    the head/tail split, so termination is immediate; no `partial` needed. -/
def perms {α} : List α → List (List α)
  | []      => [[]]
  | x :: xs => (perms xs).flatMap (insertEverywhere x)

/-- One program per permutation of the statement list; decls unchanged. -/
def programPermutations (p : TLProgram) : List TLProgram :=
  (perms p.stmts).map (fun ss => { p with stmts := ss })

/-- The read-slot axis list for a produced intermediate: read it back over the LHS's own free axes,
    so `Tk` is consumed with the same index structure it was written with. -/
private def lhsReadIdxs (lhs : List LHSSlot) : List IdxExpr :=
  lhs.filterMap (fun | .free a => some (.axis a) | .freeNorm a => some (.axis a)
                     | .iterAt a _ => some (.axis a) | .iterNext a => some (.axis a)
                     | .affine e => some e)

/-- First name of the form `__mz{k}_{nm}` (k = 0, 1, 2, …) not already in `taken`. Since the
    `taken.length + 1` candidates `cand 0, …, cand taken.length` are pairwise distinct strings,
    pigeonhole guarantees one of them is absent from `taken`; the `none` branch is unreachable. -/
private def freshName (taken : List String) (nm : String) : String :=
  let cand (k : Nat) : String := s!"__mz{k}_{nm}"
  match (List.range (taken.length + 1)).find? (fun k => !taken.contains (cand k)) with
  | some k => cand k
  | none => cand taken.length

/-- Split one multi-term `.assign` into per-term intermediates + a final assign summing reads of
    those intermediates over the same LHS axes. `taken0` is the set of names already in use
    (pre-existing program names ∪ names minted earlier in this call); each minted name is checked
    against — and added to — this set, so mints are disjoint from the whole program, not just
    from a call-local counter. Returns the replacement statements and the updated taken set. -/
private def splitOne (taken0 : List String) (nm : String) (lhs : List LHSSlot) (rhs : RHSExpr) :
    List Stmt × List String :=
  let n := rhs.body.terms.length
  let (interNames, taken1) := (List.range n).foldl
    (fun (acc, taken) _ =>
      let tname := freshName taken nm
      (acc ++ [tname], tname :: taken))
    ([], taken0)
  let interStmts := (rhs.body.terms.zip interNames).map (fun (term, tname) =>
    Stmt.assign tname lhs { rhs with body := ⟨[term]⟩ })
  let sumTerms := interNames.map (fun tn => (⟨[.read tn (lhsReadIdxs lhs)]⟩ : ProdTerm))
  let finalStmt := Stmt.assign nm lhs { rhs with body := ⟨sumTerms⟩ }
  (interStmts ++ [finalStmt], taken1)

/-- Split every multi-term `.assign` into named per-term intermediates + a final sum of reads.
    Single-term statements (and non-`.assign` statements) are unchanged. Minted names are fresh
    w.r.t. every name already produced by `p` as well as every name minted earlier in the same
    call, so applying `materializeSplit` to its own output (or to a program that already contains
    `__mz`-named tensors) cannot collide. -/
def materializeSplit (p : TLProgram) : TLProgram :=
  let existing := p.stmts.filterMap (fun | .assign nm _ _ => some nm | _ => none)
  let step : (List Stmt × List String) → Stmt → (List Stmt × List String) :=
    fun (out, taken) s =>
      match s with
      | .assign nm lhs rhs =>
          if rhs.body.terms.length ≥ 2 then
            let (newStmts, taken') := splitOne taken nm lhs rhs
            (out ++ newStmts, taken')
          else (out ++ [s], taken)
      | other => (out ++ [other], taken)
  let (out, _) := p.stmts.foldl step ([], existing)
  { p with stmts := out }

-- TESTS (fire on build) — hand-built tiny program:
private def ax (n : String) (u : Nat) : AxisSpec := ⟨n, u, .real (some (.lit 2))⟩
private def rdT (nm : String) : ProdTerm := ⟨[.read nm [.axis (ax "i" 1)]]⟩
private def twoTerm : Stmt := .assign "Y" [.free (ax "i" 1)] ⟨⟨[rdT "A", rdT "B"]⟩, .identity, .sum⟩
private def prog : TLProgram := { decls := [], stmts := [twoTerm] }

#guard (perms [1, 2, 3]).length == 6
#guard (perms ([] : List Nat)).length == 1
#guard (perms [1, 2, 3, 4]).length == 24
#guard (perms [1, 2, 3]).all (fun l => l.length == 3)
#guard (perms [1, 2, 3]).all (fun l => [1, 2, 3].all (fun x => l.contains x))

#guard (programPermutations { decls := [], stmts := [twoTerm, twoTerm] }).length == 2
#guard (materializeSplit prog).stmts.length == 3       -- 2 intermediates + 1 final sum
-- final statement still writes "Y":
#guard ((materializeSplit prog).stmts.getLast?.map (fun | .assign nm _ _ => nm | _ => "")) == some "Y"
-- single-term statement is unchanged:
private def oneTerm : Stmt := .assign "Z" [.free (ax "i" 1)] ⟨⟨[rdT "A"]⟩, .identity, .sum⟩
#guard (materializeSplit { decls := [], stmts := [oneTerm] }).stmts.length == 1

-- freshness: re-applying materializeSplit to its own output must not collide with the
-- `__mz0_Y`/`__mz1_Y` names the first application already minted:
#guard
  let names := (materializeSplit (materializeSplit prog)).stmts.filterMap
    (fun | .assign nm _ _ => some nm | _ => none)
  names.eraseDups.length == names.length

end LeanNCD.PropertyOracle
