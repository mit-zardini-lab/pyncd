import LeanNCD.Exec.Uid

namespace LeanNCD

-- freshUData mints a strictly increasing counter starting at 0.
private def twoFresh : FreshM (UID × UID) := do
  let a ← freshUData
  let b ← freshUData
  return (a.uid, b.uid)

#guard (match twoFresh.run 0 with | .ok r _ => r == (0, 1) | .error _ _ => false)

-- A thrown CompileError surfaces as `.error` through FreshM.run.
private def failing : FreshM Unit := throw (CompileError.undeclaredName "x")
#guard (match failing.run 0 with | .ok _ _ => false | .error _ _ => true)

-- CompileError has Repr (used in error messages).
#guard (toString (repr (CompileError.shapeMismatch "a" "b"))).length > 0

end LeanNCD
