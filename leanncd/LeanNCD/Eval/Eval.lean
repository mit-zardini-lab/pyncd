import LeanNCD.Eval.Scan        -- (transitively brings Contract/Nonlin/Gather/Shape/Tensor)
import LeanNCD.Eval.Scatter
import LeanNCD.DSL.Compile
namespace LeanNCD.Eval
open Std

/-- Find the `norm`-axis position + the per-position axis-UID list for an output, from the decls,
    so softmax/normalize know which axis to reduce. Returns (axisPos, axisUids) or none.

    The decl's declared axes are positional with the output's LHS slots, so the `norm`-kind axis's
    index in the decl is the slot position to reduce over; `axisUids` is the output tensor's
    per-position axis UIDs taken from the slots. -/
def normAxisInfo (decls : List Decl) (nm : String) (slots : List LHSSlot) : Option (Nat × List UID) := do
  let d ← decls.find? (fun d => declName d == nm)
  let axes := declAxes d
  let pos ← axes.findIdx? (fun a => match a.kind with | .norm _ => true | _ => false)
  let axisUids := slots.filterMap lhsAxisUID?
  return (pos, axisUids)

/-- The declared output shape for a scatter `nm[slots] := …`, computed from the inferred source
    axis sizes (`sizes`): each affine slot's output extent is its affine map applied to the source
    sizes — `c0 + Σ cᵢ·size(aᵢ)` for an `.affine`, `c·size(a)` for a `.scale` (e.g. upsample
    `Out[2·i]` ⇒ `2·size(i)`), `size(a)+c` for a `.shift`, `n+1` for a `.const n`, `size(a)` for a
    bare axis. (`sizes[u]` defaults to 0 for an unseen UID.) -/
def scatterOutShape (sizes : HashMap UID Nat) (slots : List LHSSlot) : List Nat :=
  slots.map (fun sl =>
    let idx := lhsSlotIdx sl
    match idx with
    | .axis a      => (sizes[a.uid]?).getD 0
    | .const n     => (n + 1).toNat
    | .scale c a   => (c * Int.ofNat ((sizes[a.uid]?).getD 0)).toNat
    | .shift a c   => (Int.ofNat ((sizes[a.uid]?).getD 0) + c).toNat
    | .affine c0 xs =>
        (xs.foldl (fun acc (c, a) => acc + c * Int.ofNat ((sizes[a.uid]?).getD 0)) c0).toNat)

/-- Evaluate one `.plain` stmt → (name, tensor). -/
def evalPlain (decls : List Decl) (env : HashMap String DenseTensor) (sizes : HashMap UID Nat)
    (s : Stmt) : Except EvalError (String × DenseTensor) := do
  match s with
  | .assign nm slots rhs =>
      let (_, pre) ← evalAssignDtyped decls env sizes nm slots rhs    -- contract (dtype-aware)
      if rhs.nonlin == Nonlin.identity then return (nm, pre)
      else
        let (axisPos, axisUids) := (normAxisInfo decls nm slots).getD (0, [])
        return (nm, applyNonlin rhs.nonlin axisPos axisUids pre)
  | .scatter nm slots rhs opts =>
      let outShape := scatterOutShape sizes slots
      evalScatter env sizes nm slots rhs opts outShape
  | .recurMorphism nm _ _ => .error s!"evalPlain: recurMorphism (escape hatch) unsupported ({nm})"

/-- Evaluate a ScheduledProgram on concrete inputs → the full env (inputs + computed). -/
def evalScheduled (sched : ScheduledProgram) (inputs : HashMap String DenseTensor) :
    Except EvalError (HashMap String DenseTensor) := do
  -- gather ALL underlying stmts (plain + scan base/recur) to infer axis sizes from the inputs:
  let allStmts : List Stmt := sched.stmts.flatMap (fun
    | .plain s => [s] | .scan _ _ b r _ => b ++ r | .scanPre _ _ _ => [])
  let sizes ← inferAxisSizes inputs allStmts
  let mut env := inputs
  for sc in sched.stmts do
    match sc with
    | .plain s =>
        let (nm, t) ← evalPlain sched.decls env sizes s
        env := env.insert nm t
    | .scan .. =>
        let outs ← evalScan sched.decls env sizes sc
        for (nm, t) in outs do env := env.insert nm t
    | .scanPre nm _ _ => throw s!"evalScheduled: scanPre unsupported ({nm})"
  return env

/-- The DSL evaluator entry point: parse-compiled program + inputs → outputs. -/
def TLProgram.eval (p : TLProgram) (inputs : HashMap String DenseTensor) :
    Except EvalError (HashMap String DenseTensor) :=
  match p.compileToScheduled |>.run 0 with
  | .ok sched _ => evalScheduled sched inputs
  | .error e _  => .error s!"compile failed: {repr e}"

end LeanNCD.Eval
