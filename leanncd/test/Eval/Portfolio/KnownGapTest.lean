import Eval.Portfolio.Harness
/-!
# Portfolio §14 — Known gaps / expected-fail (documentation)

Most gaps are "the DSL cannot express feature X", which has no single rejecting program to
assert. They are catalogued here; the gaps that DO have a concrete failing program carry a live
regression test (linked below) that will flip when the gap closes.

## Silent-wrong (soundness) gaps — live regression tests elsewhere
* **KG-scanagg** — `maxreduce` inside a scan step is silently summed. Live test: `RC5` in
  `RecurrenceTest` (asserts the current wrong output `[2,8,32]`).
* **KG-2dscan** — a 2-D/nested recurrence collapses to a 1-D scan. Live test: `RC6` in
  `RecurrenceTest` (asserts the current wrong output `[[0,1],[0,1]]`).
* **Equation-level summation** — a multi-term RHS with heterogeneous contracted axes broadcasts
  the shorter terms. Live test: `EC15` in `EdgeCaseTest`.

## Rejecting gaps — live in RejectTest
* **KG (predicate agg)** — `RJ3` (`predicateAgg`).

## Parse-level gaps (cannot be automated — hard parse errors fail the build)
* **KG-reshape** — multi-axis-into-one LHS slot: `Out[i + j] := X[i,j]`, `Out[2*i + k] := …`
  → `unexpected identifier; expected numeral`. No reshape / flatten / Kronecker / im2col; the
  overlapping-accumulate scatter is likewise unreachable via surface syntax.
* **KG-datamask** — data-driven mask: `softmax(where edge[i,j] > 0)(…)`
  → `unexpected token '['`; `where` clauses are index predicates only.

## Missing-primitive gaps (no expressible program at all)
* **KG-log** — no standalone `log`/`exp` (only relu/softmax/normalize) ⇒ cross-entropy,
  log-likelihood, GELU.
* **KG-trig** — no `sin`/`cos` ⇒ sinusoidal positional encodings, RoPE, diffusion time embeds.
* **KG-l2norm** — only L1 `normalize`; no `√Σx²` ⇒ cosine similarity, RMSNorm.
* **KG-sqrt** — no `sqrt` ⇒ true Euclidean distance, std / batch-norm scaling.
* **KG-min** — no `min` aggregation / additive-combine ⇒ min-plus (shortest path, Viterbi, DTW),
  argmin (k-means assignment).
* **KG-div** — no per-element tensor division (only whole-axis L1 `normalize`) ⇒ D⁻¹AX GNN
  normalization, layer-norm variance division.
* **KG-gather** — no data-dependent gather/scatter with an index tensor ⇒ embedding lookup by id,
  edge-list scatter-add, top-k.
* **KG-activation** — only relu ⇒ sigmoid/tanh (LSTM/GRU gates, logistic regression), leaky-relu
  (GAT), gelu.
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
-/
namespace LeanNCD.Eval.Portfolio.KnownGap
-- documentation-only module (no assertions)
end LeanNCD.Eval.Portfolio.KnownGap
