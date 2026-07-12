import Eval.PropertyOracle.ScanOracle

/-!
# E6 scan-unrolling property oracle — entry point (Task 7)

Registers the scan-unrolling law as a build-failing check, completing the deferred third E6
metamorphic law (reordering + materialization already landed in `PropertyOracleTest.lean`).
Kept as its own entry so a violation is unambiguously attributed to this law, not the
scan-free harness's REORDERING/MATERIALIZATION checks.
-/
namespace LeanNCD.PropertyOracle

-- THE SCAN-UNROLLING ORACLE: every generated scan case obeys the law, else fail the build with
-- a minimal counterexample.
run_cmd do
  match runAllScans with
  | none => pure ()
  | some msg => throwError s!"E6 SCAN-UNROLL property oracle FAILED:\n{msg}"

end LeanNCD.PropertyOracle
