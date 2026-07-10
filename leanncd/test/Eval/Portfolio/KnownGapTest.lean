import Eval.Portfolio.Harness
/-!
# Portfolio §14 — Known gaps / expected-fail (documentation)

Most gaps are "the DSL cannot express feature X", which has no single rejecting program to
assert. They are catalogued here; the gaps that DO have a concrete failing program carry a live
regression test (linked below) that will flip when the gap closes.

## Rejecting gaps — live in RejectTest
* **KG (predicate agg)** — `RJ3` (`predicateAgg`).
* **KG-scanprojection** — a per-step scan read-out written INSIDE the scan block (e.g.
  `y[j,l] := C[j,k]·h[k,l]` alongside a `h` base/recur pair, in the same `tlprog!`) is rejected
  at compile time (`scanProjectionUnsupported`) rather than silently discarded. `finalizeScans`
  classifies it `isInter` (a per-step intermediate, same bucket as genuine same-step scratch)
  because it reads scan state `h`; the new check additionally looks at whether the statement's
  OWN LHS references the component's iteration axis — the signal that distinguishes "track this
  across every step" (not materialized) from "recomputed and discarded every step" (scratch).
  Not a missing primitive: the fully general workaround is to write the same statement standalone
  *after* the scan, reading the fully materialized state — already supported and tested (SS2,
  `GenerativeTest`). Live: `UF5` (`RejectTest`).

## Parse-level gaps (cannot be automated — hard parse errors fail the build)
* **KG-reshape** — multi-axis-into-one LHS slot: `Out[i + j] := X[i,j]`, `Out[2*i + k] := …`
  → `unexpected identifier; expected numeral`. No reshape / flatten / Kronecker / im2col; the
  overlapping-accumulate scatter is likewise unreachable via surface syntax.
* **KG-datamask** — data-driven mask: `softmax(where edge[i,j] > 0)(…)`
  → `unexpected token '['`; `where` clauses are index predicates only.

## Missing-primitive gaps (no expressible program at all)
* **KG-min** (partial) — `min` aggregation is supported (`minreduce`; live: `TR6`/`TR7`/`RC7`).
  Still missing: additive within-term combine ⇒ min-plus (shortest path, Viterbi, DTW), and
  argmin (k-means assignment).
* **KG-gather** — no data-dependent gather/scatter with an index tensor ⇒ embedding lookup by id,
  edge-list scatter-add, top-k.
* **KG-idxvalue** — index arithmetic yields booleans only (via Iverson), never a numeric tensor
  value ⇒ ALiBi / relative-position numeric bias.
* **KG-solve** — no linear solve / inverse / eigendecomposition / determinant ⇒ closed-form
  regression, PCA, spectral GCN, normalizing-flow log-det.
* **KG-functor** — axes are flat `Fin n`; no structured/applicative-functor index spaces (trees,
  functions) ⇒ generalized transformers over non-sequence data.
* **KG-recur** — `recurMorphism`/`scanPre` escape hatch is AST-only (eval unsupported).

## Confirmed NON-gaps (resolved during probing — recorded so they aren't re-litigated)
* **KG-sub** — no `−` operator, but subtraction works via a rank-0 `−1` scalar
  (`r[i] := a[i] + neg[]·b[i]`). Live: `ST5` in `StatsLossTest`.
* **KG-scale** — scalar scaling works via a rank-0 tensor read `scale[]`. Live: `AT6` /
  `DF1` (`GenerativeTest`).
* **KG-log** — `log`/`exp` are available inline on a single factor's read via `Factor.unaryFn`
  (e.g. `log(P[i])`), not just as a whole-statement `Nonlin`. Live: `ST6` (`StatsLossTest`),
  `CL4` (`GenerativeTest`).
* **KG-trig** — `sin`/`cos` are likewise available via `Factor.unaryFn`. Live: `DF4`
  (`GenerativeTest`, `sin`). Turning *index* arithmetic into a numeric value (e.g. `ω^i`) is
  still open — see `KG-idxvalue`.
* **KG-sqrt** — `sqrt` is likewise available via `Factor.unaryFn`; fails loud (`EvalError`) on
  a negative input rather than propagating `NaN`. Live: `CM1b` (`ClassicalMLTest`).
* **KG-activation** — `sigmoid`/`tanh`/`gelu`/`leakyrelu` are available as new `Nonlin` variants,
  the same whole-statement mechanism `relu` already uses (not `Factor.unaryFn` — these apply to
  a statement's full contracted output, not a single factor). `gelu` uses the tanh-approximation
  formula (no `Float.erf` in this toolchain); `leakyrelu` has a fixed `0.01` negative slope, not
  parameterized. Live: `FF5`–`FF8` (`FeedforwardTest`), `CM9a` (`ClassicalMLTest`, logistic
  regression). The full multi-gate GRU/LSTM composite is not authored (`CM9b`).
* **KG-div** — a friendly infix `/` operator is available (`H[i,f] / deg[i]`), desugaring to
  `Factor.unaryFn .recip` — the same mechanism as `log`/`sin`/`sqrt`, just with an operator
  instead of a keyword. RHS restricted to a bare read, same as the other `unaryFn` forms; fails
  loud (`EvalError`) on division by exactly zero rather than propagating `inf`. Live: `GN5`
  (`GnnScatterTest`, degree-normalized GNN propagation).
* **KG-l2norm** — `l2normalize` is a new `Nonlin` variant (`y = x/‖x‖₂`), the same row-wise
  reduction mechanism (`perRow`) as `normalize`/`softmax`, not `Factor.unaryFn` (this reduces
  over a marked axis, it doesn't transform one already-affine value). An all-zero row silently
  normalizes to zero, matching `normalize`/`softmax`'s convention (see SC8's precedent) — not
  the fail-loud convention used by `log`/`sqrt`/`recip`. Live: `CL3` (`GenerativeTest`, cosine
  similarity), `CL3b` (all-zero row).
* **KG-multiout** — was never actually a missing primitive: `schedule`
  (`DSL/Pipeline/Lowering.lean`) was eliminating any produced-but-unread, non-tail statement as
  dead code — indistinguishable, using only read/unread status, from a legitimate second output.
  Fixed by no longer eliminating produced-but-unread top-level statements (every produced name is
  now a potential output; the caller ignores keys it doesn't need). Live: the multi-output
  regression test in `EvalExamplesTest` (`Total`/`Peak`, two independent, non-chained results).
-/
namespace LeanNCD.Eval.Portfolio.KnownGap
-- documentation-only module (no assertions)
end LeanNCD.Eval.Portfolio.KnownGap
