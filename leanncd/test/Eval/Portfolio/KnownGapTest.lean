import Eval.Portfolio.Harness
/-!
# Portfolio §14 — Known gaps / expected-fail (documentation)

Most gaps are "the DSL cannot express feature X", which has no single rejecting program to
assert. They are catalogued here; the gaps that DO have a concrete failing program carry a live
regression test (linked below) that will flip when the gap closes.

## Rejecting gaps — live in RejectTest
* **KG (predicate agg)** — `RJ3` (`predicateAgg`).

## Parse-level gaps (cannot be automated — hard parse errors fail the build)
* **KG-reshape** — multi-axis-into-one LHS slot: `Out[i + j] := X[i,j]`, `Out[2*i + k] := …`
  → `unexpected identifier; expected numeral`. No reshape / flatten / Kronecker / im2col; the
  overlapping-accumulate scatter is likewise unreachable via surface syntax.
* **KG-datamask** — data-driven mask: `softmax(where edge[i,j] > 0)(…)`
  → `unexpected token '['`; `where` clauses are index predicates only.

## Missing-primitive gaps (no expressible program at all)
* **KG-l2norm** — only L1 `normalize`; no `√Σx²` ⇒ cosine similarity, RMSNorm.
* **KG-min** (partial) — `min` aggregation is supported (`minreduce`; live: `TR6`/`TR7`/`RC7`).
  Still missing: additive within-term combine ⇒ min-plus (shortest path, Viterbi, DTW), and
  argmin (k-means assignment).
* **KG-div** — no per-element tensor division (only whole-axis L1 `normalize`) ⇒ D⁻¹AX GNN
  normalization, layer-norm variance division.
* **KG-gather** — no data-dependent gather/scatter with an index tensor ⇒ embedding lookup by id,
  edge-list scatter-add, top-k.
* **KG-idxvalue** — index arithmetic yields booleans only (via Iverson), never a numeric tensor
  value ⇒ ALiBi / relative-position numeric bias.
* **KG-solve** — no linear solve / inverse / eigendecomposition / determinant ⇒ closed-form
  regression, PCA, spectral GCN, normalizing-flow log-det.
* **KG-functor** — axes are flat `Fin n`; no structured/applicative-functor index spaces (trees,
  functions) ⇒ generalized transformers over non-sequence data.
* **KG-multiout** — outputs must be at the tail statement; no genuine multi-output-not-at-tail.
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
-/
namespace LeanNCD.Eval.Portfolio.KnownGap
-- documentation-only module (no assertions)
end LeanNCD.Eval.Portfolio.KnownGap
