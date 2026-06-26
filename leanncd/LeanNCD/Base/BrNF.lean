import LeanNCD.Base.Br
import Mathlib

/-! # `BrNF` — WORK IN PROGRESS: NbE / free-strict-SMC normal-form model for `brCancelPoint`.

  ⚠️ EXPLORATORY, NOT LOAD-BEARING. This module is the in-progress proof of `Br.brCancelPoint` (hence
  `Br` elementality). Nothing imports it; it is **off every critical path**. `brCancelPoint` was
  demoted to the opt-in `Elemental BrObj` instance (see `Base/ColoredPROP.lean` / leanncd.md §2), so
  `Br : ColoredPROP` is sorry-free regardless of this file. Promoted in-tree (2026-06-22) so the
  progress is version-controlled and resumable; carries deferred `sorry`s by design.

  ## The model
  `N a b` (here `NData`) is a concrete model of free-strict-SMC morphisms: a node list (generator
  occurrences) + a color-preserving **wiring bijection** on ports. Ports are defined by *recursion on
  the node list* (`codPorts`/`domPorts`), so empty/singleton cases collapse via `Empty`/`sumEmpty`
  with NO `Fin 1` cast hacks and NO `List.get` — this killed the dependent-cast friction of earlier
  representations. The braid wiring is `Equiv.Perm`-based (faithful by construction).

  ## Status (batch-verified `lake build`)
  SORRY-FREE: representation (`Node`/`codPorts`/`domPorts`/`NData`); `nId`, `nGen` (full, `rfl`
  colors); `codPortsAppend`/`domPortsAppend`/`finSplit`/`finJoin`; ALL FIVE SMC op WIRES
  (`nId`/`nGen`/`nBraid`/`nTensor`/`nComp` — composition is just reassociations, NO GoI trace); `eval`
  (the syntax→model interpreter, on the five SMC constructors); `NData.ext'`; the `@[simp]` lemmas.
  DEFERRED `sorry`s (6): `nComp.color`/`nTensor.color`/`nBraid.color` (proof-irrelevant
  well-formedness, off the `sound` path since `color : Prop`); `nComp_nId_left` (the first SMC
  law — ~90% mechanized, see its doc-comment for the `erw`-driven recipe and the residual); and `eval`
  on the CD comonoid generators `copyW`/`delW` — the linear *bijective-wiring* `NData` cannot model
  copy/discard (one-in-two-out is not a bijection); a non-bijective CD model is out of scope and off
  the `brCancelPoint` path (the realize-side `Br` copy/discard laws are sorry-free in `Br.lean`).

  ## Remaining route to `brCancelPoint`
  category/SMC laws (technique demonstrated in `nComp_nId_left`) → `sound : Rel f g → eval f = eval g`
  (induction on `Rel`, each case = one law) → the deep `section_` (Joyal–Street coherence) +
  `eval_point_injective` → assemble `brCancelPoint` (skeleton in the old `_scratch_nf.lean`). -/

namespace LeanNCD
namespace BrNF

structure Node where
  dom : BrObj
  cod : BrObj
  op  : BrBase dom cod

/-- All output (cod) ports of a node list, as a nested sum. -/
def codPorts : List Node → Type
  | [] => Empty
  | x :: xs => Fin x.cod.length ⊕ codPorts xs

/-- All input (dom) ports of a node list, as a nested sum. -/
def domPorts : List Node → Type
  | [] => Empty
  | x :: xs => Fin x.dom.length ⊕ domPorts xs

instance : IsEmpty (codPorts []) := inferInstanceAs (IsEmpty Empty)
instance : IsEmpty (domPorts []) := inferInstanceAs (IsEmpty Empty)

def codColor : (nodes : List Node) → codPorts nodes → ArrayType
  | [], e => nomatch e
  | x :: _, .inl k => x.cod.get k
  | _ :: xs, .inr q => codColor xs q

def domColor : (nodes : List Node) → domPorts nodes → ArrayType
  | [], e => nomatch e
  | x :: _, .inl k => x.dom.get k
  | _ :: xs, .inr q => domColor xs q

/-- Output ports of a diagram `a → b`: an external input or a node-output. -/
def OutPort (a : BrObj) (nodes : List Node) : Type := Fin a.length ⊕ codPorts nodes
/-- Input ports: a node-input or an external output. -/
def InPort (b : BrObj) (nodes : List Node) : Type := domPorts nodes ⊕ Fin b.length

def outColor (a : BrObj) (nodes : List Node) : OutPort a nodes → ArrayType
  | .inl k => a.get k
  | .inr q => codColor nodes q

def inColor (b : BrObj) (nodes : List Node) : InPort b nodes → ArrayType
  | .inl q => domColor nodes q
  | .inr k => b.get k

/-- The model datum: nodes + a color-preserving wiring bijection. -/
structure NData (a b : BrObj) where
  nodes : List Node
  wire  : OutPort a nodes ≃ InPort b nodes
  color : ∀ p, inColor b nodes (wire p) = outColor a nodes p

/-- Identity: no nodes; ports are `a` on each side, wired straight. -/
def nId (a : BrObj) : NData a a where
  nodes := []
  wire := (Equiv.sumEmpty _ _).trans (Equiv.emptySum _ _).symm
  color := by
    rintro (k | e)
    · rfl
    · exact isEmptyElim e

/-- Cod-ports of an appended node list split as a sum (by recursion on the first list). -/
def codPortsAppend : (xs ys : List Node) → codPorts (xs ++ ys) ≃ codPorts xs ⊕ codPorts ys
  | [], _ => (Equiv.emptySum _ _).symm
  | _ :: xs, ys =>
      (Equiv.sumCongr (Equiv.refl _) (codPortsAppend xs ys)).trans (Equiv.sumAssoc _ _ _).symm

/-- Dom-ports of an appended node list split as a sum. -/
def domPortsAppend : (xs ys : List Node) → domPorts (xs ++ ys) ≃ domPorts xs ⊕ domPorts ys
  | [], _ => (Equiv.emptySum _ _).symm
  | _ :: xs, ys =>
      (Equiv.sumCongr (Equiv.refl _) (domPortsAppend xs ys)).trans (Equiv.sumAssoc _ _ _).symm

/-- Split the input ports of an object append `Fin (a++c).length ≃ Fin a.length ⊕ Fin c.length`. -/
def finSplit (a c : BrObj) : Fin (a ++ c).length ≃ Fin a.length ⊕ Fin c.length :=
  (finCongr (show (a ++ c).length = a.length + c.length by simp)).trans finSumFinEquiv.symm

/-- Join: `Fin b.length ⊕ Fin d.length ≃ Fin (b++d).length`. -/
def finJoin (b d : BrObj) : Fin b.length ⊕ Fin d.length ≃ Fin (b ++ d).length :=
  finSumFinEquiv.trans (finCongr (show b.length + d.length = (b ++ d).length by simp))

/-- Tensor: place `f` and `g` in parallel — concatenate node lists, run the two wirings on disjoint
    blocks (`f` on `a`/`Nf`→`b`, `g` on `c`/`Ng`→`d`), reshuffling blocks via `sumSumSumComm`. -/
def nTensor {a b c d : BrObj} (f : NData a b) (g : NData c d) : NData (a ++ c) (b ++ d) where
  nodes := f.nodes ++ g.nodes
  wire :=
    -- OutPort (a++c) (Nf++Ng) ≃ (A ⊕ C) ⊕ (Cf ⊕ Cg)
    (Equiv.sumCongr (finSplit a c) (codPortsAppend f.nodes g.nodes)).trans <|
    -- ≃ (A ⊕ Cf) ⊕ (C ⊕ Cg)  = OutPort a Nf ⊕ OutPort c Ng
    (Equiv.sumSumSumComm _ _ _ _).trans <|
    -- ≃ (Df ⊕ B) ⊕ (Dg ⊕ D)  = InPort b Nf ⊕ InPort d Ng
    (Equiv.sumCongr f.wire g.wire).trans <|
    -- ≃ (Df ⊕ Dg) ⊕ (B ⊕ D)
    (Equiv.sumSumSumComm _ _ _ _).trans <|
    -- ≃ domPorts (Nf++Ng) ⊕ Fin (b++d).length = InPort (b++d) (Nf++Ng)
    (Equiv.sumCongr (domPortsAppend f.nodes g.nodes).symm (finJoin b d))
  color := by sorry

/-- Composition: concatenate node lists, glue wirings along the shared boundary `b` (single pass —
    `b` is `f`'s output and `g`'s input, so it's reassociations, no trace). -/
def nComp {a b c : BrObj} (f : NData a b) (g : NData b c) : NData a c where
  nodes := f.nodes ++ g.nodes
  wire :=
    -- OutPort a (Nf++Ng) ≃ A ⊕ (Cf ⊕ Cg)
    (Equiv.sumCongr (Equiv.refl _) (codPortsAppend f.nodes g.nodes)).trans <|
    -- A ⊕ (Cf ⊕ Cg) ≃ (A ⊕ Cf) ⊕ Cg
    (Equiv.sumAssoc _ _ _).symm.trans <|
    -- (A ⊕ Cf) ⊕ Cg ≃ (Df ⊕ B) ⊕ Cg
    (Equiv.sumCongr f.wire (Equiv.refl _)).trans <|
    -- (Df ⊕ B) ⊕ Cg ≃ Df ⊕ (B ⊕ Cg)
    (Equiv.sumAssoc _ _ _).trans <|
    -- Df ⊕ (B ⊕ Cg) ≃ Df ⊕ (Dg ⊕ C)
    (Equiv.sumCongr (Equiv.refl _) g.wire).trans <|
    -- Df ⊕ (Dg ⊕ C) ≃ (Df ⊕ Dg) ⊕ C
    (Equiv.sumAssoc _ _ _).symm.trans <|
    -- (Df ⊕ Dg) ⊕ C ≃ InPort c (Nf++Ng)
    (Equiv.sumCongr (domPortsAppend f.nodes g.nodes) (Equiv.refl _)).symm
  color := by sorry

/-- Braiding: no nodes; the wiring is the block-swap permutation `Fin (a++b).len ≃ Fin (b++a).len`
    (the keystone `braidPerm`, here with the `List.length_append` casts). -/
def nBraid (a b : BrObj) : NData (a ++ b) (b ++ a) where
  nodes := []
  wire :=
    (Equiv.sumEmpty _ _).trans <|
    (finCongr (show (a ++ b).length = a.length + b.length by simp)).trans <|
    finSumFinEquiv.symm.trans <|
    (Equiv.sumComm _ _).trans <|
    finSumFinEquiv.trans <|
    (finCongr (show b.length + a.length = (b ++ a).length by simp)).trans <|
    (Equiv.emptySum _ _).symm
  color := by sorry

/-- A single generator node: ports `d` (inputs) and `c` (outputs), wired straight through. -/
def nGen {d c : BrObj} (op : BrBase d c) : NData d c where
  nodes := [⟨d, c, op⟩]
  wire := (Equiv.sumCongr (Equiv.refl _) (Equiv.sumEmpty _ _)).trans
            (Equiv.sumCongr (Equiv.sumEmpty _ _) (Equiv.refl _)).symm
  color := by
    rintro (k | (p | e))
    · rfl
    · rfl
    · exact isEmptyElim e

/-- Extensionality for `NData`: equal nodes + heterogeneously-equal wires ⟹ equal (color is a
    `Prop`, so proof-irrelevant). -/
theorem NData.ext' {a b : BrObj} :
    ∀ (F G : NData a b), F.nodes = G.nodes → HEq F.wire G.wire → F = G
  | ⟨_, _, _⟩, ⟨_, _, _⟩, hn, hw => by cases hn; cases hw; rfl

/-- Projection `rfl`-lemmas: expose the `.nodes`/`.wire` of the `where`-defined ops so `simp` can
    drive the reduction. The bare structure projection of a `where`-def does NOT unfold under `simp`
    (the displayed composite was only pretty-printing), and leaving `.nodes` stuck also blocks the
    inner `codPorts`/`IsEmpty`/`sumEmpty` reductions — this was the law-proving blocker; these
    `@[simp]` lemmas give `simp` usable rewrites, after which the `Equiv.*_apply` lemmas fire. -/
@[simp] theorem nId_nodes (a : BrObj) : (nId a).nodes = [] := rfl

@[simp] theorem nComp_nodes {a b c : BrObj} (f : NData a b) (g : NData b c) :
    (nComp f g).nodes = f.nodes ++ g.nodes := rfl

@[simp] theorem nComp_wire {a b c : BrObj} (f : NData a b) (g : NData b c) :
    (nComp f g).wire =
      ((Equiv.sumCongr (Equiv.refl _) (codPortsAppend f.nodes g.nodes)).trans <|
       (Equiv.sumAssoc _ _ _).symm.trans <|
       (Equiv.sumCongr f.wire (Equiv.refl _)).trans <|
       (Equiv.sumAssoc _ _ _).trans <|
       (Equiv.sumCongr (Equiv.refl _) g.wire).trans <|
       (Equiv.sumAssoc _ _ _).symm.trans <|
       (Equiv.sumCongr (domPortsAppend f.nodes g.nodes) (Equiv.refl _)).symm) := rfl

set_option linter.unusedSimpArgs false in   -- the `repeat'` loop leaves some leaf-lemmas unused per branch
/-- Left identity — the `Rel.id_comp` case of `sound`. Gluing `nId` on the left collapses to the
    original wiring.
    PROGRESS / SHARP RECIPE (the law-proving technique, ~90% mechanized; residual `sorry`):
    1. `NData.ext' _ _ rfl` + `heq_of_eq` + `ext p` reduces to a per-port wire equality.
    2. The `@[simp] nComp_wire`/`nComp_nodes`/`nId_nodes` lemmas above expose the composite and
       reduce `(nId a).nodes → []` (a bare `where`-def projection does NOT unfold under `simp`, and a
       stuck `.nodes` also stalls the inner `codPorts`/`IsEmpty`/`sumEmpty` — that was the blocker).
    3. KEY: the composite's `Equiv.trans` heads are **defeq-but-not-syntactic**, so `rw`/`simp` with
       `Equiv.trans_apply` will NOT fire ("pattern not found" / "no progress"); **`erw [Equiv.trans_apply]`
       (up-to-defeq match) DOES** — `repeat' first | erw [Equiv.trans_apply] | simp only [<leaf
       apply-lemmas: sumCongr_apply, sumAssoc_(symm_)apply_*, sumEmpty_apply_inl, emptySum_symm_apply,
       Sum.map_inl/inr, coe_refl, sumCongr_symm, symm_symm, refl_symm>]` peels all `trans` and most
       leaves.
    RESIDUAL: stalls on inside-out reduction ordering + `(nId a).wire` being itself a defeq-`trans`
    nested under `sumCongr` (needs a second `erw`+`simp` round / `conv` innermost-first). Closable;
    deferred — this is `Br`-elementality-adjacent WIP, off every critical path (see header). -/
theorem nComp_nId_left {a b : BrObj} (F : NData a b) : nComp (nId a) F = F := by
  refine NData.ext' _ _ rfl ?_
  apply heq_of_eq
  ext p
  rcases p with k | q <;>
    (simp only [nComp_wire, nComp_nodes, nId_nodes, List.nil_append, codPortsAppend, domPortsAppend]
     repeat' first
       | erw [Equiv.trans_apply]
       | simp only [Equiv.sumCongr_apply, Sum.map_inl, Sum.map_inr, Equiv.coe_refl, id_eq,
           Equiv.sumAssoc_apply_inl_inl, Equiv.sumAssoc_apply_inl_inr, Equiv.sumAssoc_apply_inr,
           Equiv.sumAssoc_symm_apply_inl, Equiv.sumAssoc_symm_apply_inr_inl,
           Equiv.sumAssoc_symm_apply_inr_inr, Equiv.sumEmpty_apply_inl, Equiv.emptySum_symm_apply,
           Equiv.emptySum_apply_inr, Equiv.sumCongr_symm, Equiv.symm_symm, Equiv.refl_symm])
  all_goals sorry

/-- Interpret raw syntax into the model: the unique strict-SMC functor extending `gen ↦ nGen`. -/
def eval : {a b : BrObj} → Hom a b → NData a b
  | _, _, .id a => nId a
  | _, _, .gen op => nGen op
  | _, _, .comp f g => nComp (eval f) (eval g)
  | _, _, .tensor f g => nTensor (eval f) (eval g)
  | _, _, .braid a b => nBraid a b
  | _, _, .copyW _ => sorry  -- CD copy: unmodeled by the linear bijective-wiring `NData` (see header)
  | _, _, .delW _  => sorry  -- CD discard: ditto — needs a non-bijective (CD) model, out of scope

end BrNF
end LeanNCD
