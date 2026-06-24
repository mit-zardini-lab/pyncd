import LeanNCD.Eval.Scan        -- (transitively brings Contract/Nonlin/Gather/Shape/Tensor)
import LeanNCD.Eval.Scatter
import LeanNCD.DSL.Compile
namespace LeanNCD.Eval
open Std

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
        -- the reduction axis is the slot marked `m.` (norm flag now lives on the output slot).
        let axisUids := slots.filterMap lhsAxisUID?
        let axisPos ← match rhs.nonlin, normAxisUidOf slots with
          | .relu, _   => pure 0     -- relu is pointwise: the axis is irrelevant
          | _, some nu => match axisUids.findIdx? (· == nu) with
              | some p => pure p
              | none   => throw s!"evalPlain: marked norm axis of {nm} is not among its output axes"
          | _, none    => throw s!"evalPlain: {nm} applies softmax/normalize but no output axis is marked (·)"
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
  let (sizes, warns) ← inferAxisSizes sched.explicitSizes inputs allStmts
  for w in warns do dbg_trace w
  let mut env := inputs
  for sc in sched.stmts do
    match sc with
    | .plain s =>
        let (nm, t) ← evalPlain sched.decls env sizes s
        env := env.insert nm t
    | .scan .. =>
        let outs ← evalScan env sizes sc
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
