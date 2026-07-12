import Eval.PropertyOracle.Compare
import Eval.PropertyOracle.ScanGen
import Eval.PropertyOracle.ScanUnroll
import LeanNCD.Eval.Eval

/-!
# Scan-unrolling oracle runner (E6, Task 6)

`checkScanLaw`/`runAllScans` evaluate each generated `ScanCase` and its unrolled companion
through the full `TLProgram.eval` pipeline, then compare grid-cell-by-grid-cell via
`sliceTensorAtMulti` + `denseEq`. Kept in its own file, separate from the scan-free harness's
`Oracle.lean`, so a violation is unambiguously attributed to this law.
-/
namespace LeanNCD.PropertyOracle
open LeanNCD LeanNCD.Eval

/-- Compare a single-axis scan's full state tensor against its unrolled per-step leaves, cell
    by cell. `none` if every state agrees at every step, else the first mismatch message. -/
private def compareScan1D (c : ScanCase) (scanEnv unrollEnv : Std.HashMap String DenseTensor)
    (stateNames : List String) : Option String :=
  let L := c.Ls.head!
  stateNames.findSome? (fun nm =>
    (List.range L).findSome? (fun k =>
      match scanEnv[nm]? with
      | none => some s!"scan output missing state {nm}"
      | some full =>
          let slice := sliceTensorAtMulti [(full.shape.length - 1, k)] full
          let leafName := s!"Su_{nm}_{k}"
          match unrollEnv[leafName]? with
          | none      => some s!"unrolled leaf {leafName} missing"
          | some leaf =>
              if denseEq slice leaf then none
              else some s!"SCAN-UNROLL law violated (1-D) at {nm}[{k}]: scan={repr slice.data} unrolled={repr leaf.data}"))

/-- Compare the one 2-D template's full state tensor against its unrolled column/grid-cell
    leaves, cell by cell. -/
private def compareScan2D (c : ScanCase) (scanEnv unrollEnv : Std.HashMap String DenseTensor)
    (stateNames : List String) : Option String :=
  match c.Ls with
  | [Lr, Lc] =>
      stateNames.findSome? (fun nm =>
        (List.range Lr).findSome? (fun ri =>
          (List.range Lc).findSome? (fun ci =>
            match scanEnv[nm]? with
            | none => some s!"scan output missing state {nm}"
            | some full =>
                let slice := sliceTensorAtMulti [(0, ri), (1, ci)] full
                if ci == 0 then
                  match unrollEnv[s!"Su_{nm}_col0"]? with
                  | none => some s!"unrolled col0 for {nm} missing"
                  | some col0 =>
                      let cell := sliceTensorAtMulti [(0, ri)] col0
                      if denseEq slice cell then none
                      else some s!"SCAN-UNROLL law violated (2-D) at {nm}[{ri},0]: scan={repr slice.data} unrolled={repr cell.data}"
                else
                  match unrollEnv[s!"Su_{nm}_{ri}_{ci}"]? with
                  | none => some s!"unrolled leaf Su_{nm}_{ri}_{ci} missing"
                  | some leaf =>
                      if denseEq slice leaf then none
                      else some s!"SCAN-UNROLL law violated (2-D) at {nm}[{ri},{ci}]: scan={repr slice.data} unrolled={repr leaf.data}")))
  | _ => some "compareScan2D: expected exactly 2 scan axes"

/-- Check the scan-unrolling law on one case. `none` if OK, else a counterexample message. -/
def checkScanLaw (c : ScanCase) : Option String :=
  let stateNames := (c.base.filterMap (fun s => match s with | .assign nm _ _ => some nm | _ => none)).eraseDups
  match TLProgram.eval c.prog c.inputs with
  | .error e => some s!"scan case did not evaluate (generator well-formedness gap): {e}\n{repr c.prog}"
  | .ok scanEnv =>
      let unrolled := if c.axes.length == 1 then unrollScan1D c else unrollScan2D c
      match TLProgram.eval unrolled c.inputs with
      | .error e => some s!"unrolled companion did not evaluate: {e}\n{repr unrolled}"
      | .ok unrollEnv =>
          if c.axes.length == 1 then compareScan1D c scanEnv unrollEnv stateNames
          else compareScan2D c scanEnv unrollEnv stateNames

/-- Run the law over the whole generator; `none` if all pass, else the first failure message. -/
def runAllScans : Option String :=
  enumScanCases.findSome? checkScanLaw

-- TEST-THE-TESTER (a): every generated case passes.
#guard runAllScans.isNone

-- TEST-THE-TESTER (b): the oracle HAS TEETH — a bogus "unrolled" program for template1's L=3
-- case, with a term dropped from the final step, must be caught by `sliceTensorAtMulti`+`denseEq`.
private def bogusT1 : ScanCase := template1 3 false
private def bogusUnrolled : TLProgram :=
  { (unrollScan1D bogusT1) with
    stmts := (unrollScan1D bogusT1).stmts.map (fun s => match s with
      | .assign nm slots rhs =>
          if nm == "Su_S_2" then .assign nm slots { rhs with body := { terms := [] } }
          else .assign nm slots rhs
      | other => other) }
#guard
  match TLProgram.eval bogusT1.prog bogusT1.inputs, TLProgram.eval bogusUnrolled bogusT1.inputs with
  | .ok scanEnv, .ok unrollEnv =>
      match scanEnv["S"]?, unrollEnv["Su_S_2"]? with
      | some full, some leaf =>
          ! denseEq (sliceTensorAtMulti [(full.shape.length - 1, 2)] full) leaf
      | _, _ => false
  | _, _ => false

end LeanNCD.PropertyOracle
