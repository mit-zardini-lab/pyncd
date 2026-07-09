# Tensor-Logic DSL Test Portfolio (proposal for review)

**Status:** Draft for review — this is the *catalog* of example programs we intend to turn
into Lean test cases. Nothing here is implemented yet. Review, cut, and annotate before we
author the actual `.lean` tests.

**Target:** the Lean DSL under `leanncd/` — surface grammar in
[`DSL/Syntax.lean`](../LeanNCD/DSL/Syntax.lean), evaluator in [`Eval/`](../LeanNCD/Eval/),
entry points `tlprog!{…}` (parse), `tl!{…}` (parse+compile), `TLProgram.eval env` (run on
concrete `DenseTensor`).

---

## Contents

- [1. Goals & how to read this](#1-goals--how-to-read-this)
- [2. Core linear algebra](#2-core-linear-algebra)
- [3. Feedforward / MLP](#3-feedforward--mlp)
- [4. Attention & transformers](#4-attention--transformers)
- [5. Convolution & pooling](#5-convolution--pooling)
- [6. Normalization](#6-normalization)
- [7. Recurrence & scans](#7-recurrence--scans)
- [8. Graph / message passing (GNN)](#8-graph--message-passing-gnn)
- [8b. Consuming scatter outputs (upsampling decoders / GNN readback)](#8b-consuming-scatter-outputs-upsampling-decoders--gnn-readback)
- [9. Relational / logic (tensor-logic's logic side)](#9-relational--logic-tensor-logics-logic-side)
- [10. Losses, reductions & statistics](#10-losses-reductions--statistics)
- [11. Tropical / max-times semiring](#11-tropical--max-times-semiring)
- [12. Tensor networks / decomposition](#12-tensor-networks--decomposition)
- [12b. Advanced & generative domains](#12b-advanced--generative-domains)
- [12c. Classical ML, probabilistic & RL](#12c-classical-ml-probabilistic--rl)
- [13. Adversarial — reject tests (assert an error)](#13-adversarial--reject-tests-assert-an-error)
- [14. Adversarial — known gaps / expected-fail](#14-adversarial--known-gaps--expected-fail)
- [15. Adversarial — tricky-but-valid edge cases (should pass)](#15-adversarial--tricky-but-valid-edge-cases-should-pass)
- [16. Coverage matrix (feature × where exercised)](#16-coverage-matrix-feature--where-exercised)
- [17. Decisions (resolved) & follow-ups](#17-decisions-resolved--follow-ups)
- [18. What needs fixing](#18-what-needs-fixing-as-of-2026-07-07)

---

## 1. Goals & how to read this

We already have ~13 end-to-end examples in
[`test/Eval/EvalExamplesTest.lean`](../test/Eval/EvalExamplesTest.lean) plus parse/compile
tests. This portfolio broadens coverage across ML use cases and deliberately probes the DSL's
limits.

Per the agreed scope:

- **Both test styles, numeric-weighted.** Most examples run on a tiny dataset and assert the
  output with `DenseTensor.approxEq`; a minority assert compile-time structure with `#guard`
  when that's the interesting property.
- **Ground truth is hand-computed and shown in-doc.** Every `[N]` entry lists its input
  tensors and the worked-out expected output right here, so review = checking arithmetic.
- **Adversarial cases in three flavors:** reject-tests, known-gaps, and tricky-but-valid.

**Resolved decisions (2026-07-04):**

- **Volume:** author the full catalog (~130 entries incl. the advanced/creative domains below) — do not tighten.
- **File layout:** split by domain — one test file per section.
- **Expected-fail mechanism:** each `[R]`/`[F]` case asserts the **exact** `.error` inside a
  `run_cmd do` (match on the specific `CompileError`/`EvalError`; `throwError` otherwise). No
  reliance on a `#expect_failure`-style macro (Lean has none).
- **Test framework (2026-07-07):** adopted a **hybrid LSpec strategy**. Numeric/property/shape
  `[N]`/`[E]` assertions use `#lspec group "§N — …"` with named `test "ID" (evalEqB/evalPredB/evalShapeB …)` entries (LSpec reports all failures in a group, not just the first). Exact `CompileError`/`EvalError` constructor matches stay as `run_cmd do` — LSpec cannot catch elaboration-layer errors. Parse-level rejects remain documentation comments. `Harness.lean` exposes both layers from one import.
- **Advanced/generative domains added** (§12b): diffusion, mixture-of-experts, state-space
  models, positional encodings, contrastive — all core programs probed against HEAD.

**Implemented (2026-07-04, updated 2026-07-07):** all 17 files below are written, registered in `lakefile.toml`,
and build green via `lake build Tests`. 105 assertions total; every numeric value was hand-computed
independently. Migrated to a **hybrid LSpec strategy** (2026-07-07): 14 runtime test files use
`#lspec group` with named `test` entries for grouped, named failure reporting; `RejectTest` and `KnownGapTest`
are unchanged; `RecurrenceTest` is hybrid (RC4 stays `run_cmd do`, remainder in `#lspec group`).

| File | Section | IDs implemented | IDs skipped ([✔]/[F]/parse-fail) |
|------|---------|------------------|-----------------------------------|
| [`Harness.lean`](../test/Eval/Portfolio/Harness.lean) | — | shared helpers — `run_cmd` layer: `assertEval`, `assertEvalError`, `assertCompileError`, `assertEvalPred`, `assertShape`; LSpec layer: `evalEqB`, `evalPredB`, `evalShapeB`; shared: `rowsSumToOne`, `tl`; `import LSpec`, `open LSpec` | — |
| [`LinAlgTest.lean`](../test/Eval/Portfolio/LinAlgTest.lean) | §2 | LA1, LA3–LA9 | LA2 `[✔]` |
| [`FeedforwardTest.lean`](../test/Eval/Portfolio/FeedforwardTest.lean) | §3 | FF1–FF4 | — |
| [`AttentionTest.lean`](../test/Eval/Portfolio/AttentionTest.lean) | §4 | AT1, AT3–AT12 | AT2 `[✔]` |
| [`ConvPoolTest.lean`](../test/Eval/Portfolio/ConvPoolTest.lean) | §5 | CV1, CV2, CV4–CV9 | CV3 `[✔]` |
| [`NormTest.lean`](../test/Eval/Portfolio/NormTest.lean) | §6 | NM1–NM5 | — |
| [`RecurrenceTest.lean`](../test/Eval/Portfolio/RecurrenceTest.lean) | §7 | RC2–RC8 (RC4 reject via `run_cmd do`; RC5 asserts correct output — KG-scanagg fixed; RC6 asserts correct output — KG-2dscan fixed; RC6-compile checks (via `run_cmd do`) that the 2-D scan lowers to a single well-formed rank-2 Br `.scan` step; RC7 `minreduce`-in-scan; RC8 3-D nested scan) — **hybrid file** | RC1 `[✔]` |
| [`GnnScatterTest.lean`](../test/Eval/Portfolio/GnnScatterTest.lean) | §8, §8b | GN1–GN4, SC1–SC8 | GN5/GN6 `[F]` (comment) |
| [`RelationalTest.lean`](../test/Eval/Portfolio/RelationalTest.lean) | §9 | RL1–RL4, RL6–RL8 | RL5 `[✔]` |
| [`StatsLossTest.lean`](../test/Eval/Portfolio/StatsLossTest.lean) | §10 | ST1–ST5 | ST6 `[F]` (comment) |
| [`TropicalTest.lean`](../test/Eval/Portfolio/TropicalTest.lean) | §11 | TR3, TR6, TR7 (`minreduce` added) | TR1/TR2/TR4 `[✔]`; TR5 `[F]` (min-plus half, comment) |
| [`TensorNetTest.lean`](../test/Eval/Portfolio/TensorNetTest.lean) | §12 | TN1–TN4 | — |
| [`GenerativeTest.lean`](../test/Eval/Portfolio/GenerativeTest.lean) | §12b | DF1–DF3, ME1–ME3, SS1–SS3, PE1, CL1–CL2 | DF4/PE2/PE3 `[F]` trig; ME4 `[F]` gather; CL3 `[F]` l2norm; CL4 `[F]` log; SS4 → `RejectTest` (comments) |
| [`ClassicalMLTest.lean`](../test/Eval/Portfolio/ClassicalMLTest.lean) | §12c | CM1–CM6 | CM7–CM9 `[F]` (comment) |
| [`RejectTest.lean`](../test/Eval/Portfolio/RejectTest.lean) | §13 | RJ3, RJ4, RJ7, SS4/RC4 — **unchanged, `run_cmd do` only** (exact `CompileError`/`EvalError` constructor matches) | RJ1/RJ2/RJ5/RJ8/RJ10 (parse-fail or unconstructible — comment); RJ6/RJ9 dropped (don't reject) |
| [`KnownGapTest.lean`](../test/Eval/Portfolio/KnownGapTest.lean) | §14 | documentation only — cross-references all 19 `KG-*` gaps to their live regression test (if any) | — |
| [`EdgeCaseTest.lean`](../test/Eval/Portfolio/EdgeCaseTest.lean) | §15 | EC1–EC7, EC10–EC15 | EC8 `[✔]`; EC9 parse-fail (comment) |

Run the whole portfolio with `cd leanncd && lake build Tests` (builds the full pre-existing suite
too) or target one file, e.g. `lake build Eval.Portfolio.AttentionTest`. LSpec groups emit named
`✓ ∃: ID` lines per test case; `run_cmd` failures surface as build errors as before.

### Legend

| Tag | Meaning | Assertion |
|-----|---------|-----------|
| `[N]` | Numeric eval | `TLProgram.eval` → `approxEq` against the stated expected tensor |
| `[S]` | Structural | `#guard` on the compiled `tl!{…}` (step count, op, shape) |
| `[E]` | Edge case (should pass) | usually `[N]`; probes a legal-but-tricky corner |
| `[R]` | Reject | assert a specific `CompileError`/`EvalError` is thrown |
| `[F]` | Known gap / expected-fail | documents a construct that does *not* work today |
| `[✔]` | Already covered | exists in current tests; listed for the coverage map, not re-authored |

### DSL syntax reminders (gotchas that bit us before)

- Products use `·` (U+00B7 middle dot), **not** `*`. `*` only appears in `num*ident` index/size terms.
- Softmax/normalize **must** mark their reduction axis with a trailing dot on the LHS slot: `A[q, s.]`.
- Scan-step LHS is written **spaced**: `G[j, l +1]` (not `l+1`).
- Boolean tensors are declared `predicate P(i, j)`; masks are Iverson factors `[bool]`.
- Pin otherwise-unconstrained axes (e.g. a loop bound) with `axis l : ℕ = N`.
- Repeated free axis (`Y[i,i]`) or affine LHS (`Out[2*i, 2*j]`) ⇒ the write is reclassified to a **scatter**.
- Confirmed evaluator semantics used in expected outputs below: `relu = max(0,x)`;
  `softmax` is max-subtracted, masked entries → 0; `normalize` is **L1** (`x / Σx` over the marked axis);
  `maxreduce` unit = `-∞`; contraction sums every RHS-only axis.
- **Naming hazard:** never name a tensor/predicate exactly `c` — `c[` is a globally-reserved
  token from Mathlib's `Equiv.Perm` cycle notation (`c[0,1,2]`), so `c[k]` fails to parse with
  `unexpected token 'c['`. `cc`, `c2`, and uppercase `C` are all unaffected — see EC15.
- **Look-back reads are `n+shift` long, not `n`:** `X[i-1]` on a length-`n` input yields a
  length-`n+1` output (leading zero-pad), per the "maximal padded extent" shape-solver rule —
  see EC1/EC2 (this corrects an arithmetic error in an earlier draft of this doc).

---

## 2. Core linear algebra

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| LA1 | `y[i] := A[i,j] · x[j]` | A=`[[1,2],[3,4]]`, x=`[1,1]` → y=`[3,7]` | mat-vec; contract to vector | `[N]` |
| LA2 | `Y[i,j] := W[i,k] · X[k,j]` | — | matmul | `[✔]` |
| LA3 | `Y[b,i,j] := A[b,i,k] · X[b,k,j]` | A=`[[[1,0],[0,1]]]` (1×2×2), X=`[[[5,6],[7,8]]]` → Y=`[[[5,6],[7,8]]]` | **batched** matmul (batch axis shared+free) | `[N]` |
| LA4 | `Y[i,j] := a[i] · b[j]` | a=`[1,2]`, b=`[3,4]` → `[[3,4],[6,8]]` | outer product (no contraction) | `[N]` |
| LA5 | `s[] := x[i] · y[i]` | x=`[1,2,3]`, y=`[1,1,1]` → `6` | inner product → scalar (full contraction) | `[N]` |
| LA6 | `G[i,j] := X[i,k] · X[j,k]` | X=`[[1,0],[0,1],[1,1]]` (3×2) → `[[1,0,1],[0,1,1],[1,1,2]]` | **aliasing** (same tensor twice); Gram = X Xᵀ | `[N]` `[E]` |
| LA7 | `s[] := x[i] · W[i,j] · y[j]` | x=`[1,1]`, W=`[[1,2],[3,4]]`, y=`[1,0]` → `4` | bilinear form (3-factor scalar) | `[N]` |
| LA8 | `Z[i,j] := A[i,j] · B[i,j]` | A=`[[1,2],[3,4]]`, B=`[[1,0],[0,1]]` → `[[1,0],[0,4]]` | Hadamard (all axes free-shared, none contracted) | `[N]` `[E]` |
| LA9 | `s[i] := A[i,j,k] · B[j,k]` | A=`ones(2,2,2)`, B=`[[1,2],[3,4]]` → `[10,10]` | **multiple** contracted axes (j,k) | `[N]` `[E]` |

## 3. Feedforward / MLP

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| FF1 | `H[q,f] := relu(W_in[f,d] · X[q,d])` `Out[q,d] := W_out[d,f] · H[q,f]` (with `linear W_in(f,d), W_out(d,f) bias`) | small 1×2 → hand-check | 2-layer MLP with intermediate `H`; `linear`+`bias` decls | `[N]` |
| FF2 | `H[i] := relu(W[i,j] · x[j])` | W=`[[1,-1],[-2,1]]`, x=`[1,1]` → pre=`[0,-1]` → `[0,0]` | relu clamps negatives (both go ≤0) | `[N]` |
| FF3 | `H[i] := relu(W[i,j] · x[j])` | W=`[[1,1],[-1,-1]]`, x=`[2,1]` → pre=`[3,-3]` → `[3,0]` | relu asymmetric | `[N]` |
| FF4 | affine layer `Y[i] := W[i,j]·x[j] + b[i]` | check bias add path | bias term as separate product/sum | `[N]` `[E]` |

## 4. Attention & transformers

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| AT1 | `A[q, s.] := softmax(Q[q,d]·K[s,d])` | Q=K=I₂ → row-wise softmax of scores `[[1,0],[0,1]]` | **unmasked** self-attention scores→softmax | `[N]` |
| AT2 | `A[q, s.] := softmax(where s ≤ q)(Q[q,d]·K[s,d])` | Q=K=I₂ → `[[1,0],[0.2689,0.7311]]` | causal mask; masked entry exactly 0 | `[✔]` |
| AT3 | `O[q,e] := A[q,s]·V[s,e]` (chain from AT1's A) | attention·values | full attention output (values mix) | `[N]` |
| AT4 | `A[b,h,q,s.] := softmax(Q[b,h,q,d]·K[b,h,s,d])` | 1×1×2×2 → same as AT1 | **multi-head/batched** attention (b,h free) | `[N]` `[E]` |
| AT5 | cross-attention: `A[q, s.] := softmax(Q[q,d]·K[s,d])` with `q`≠`s` lengths | Q 2×2, K 3×2 → 2×3 rows sum to 1 | distinct query/key sequence lengths | `[N]` `[E]` |
| AT6 | scaled scores `A[q,s.] := softmax(Q[q,d]·K[s,d]·scale[])` | scale=`[0.5]` (rank-0 tensor) | scalar scaling via **rank-0** tensor read `scale[]` | `[N]` `[E]` |
| AT7 | full 1-layer transformer block (attn + MLP + residual path) | small pinned dims | end-to-end composite; residual add | `[N]` (large) |
| AT8 | **generalized cross-attention** `A[q, s.] := softmax(Q[q,h]·K[s,h])` `O[q,g] := A[q,s]·V[s,g]` | Q=I₂, K=`[[1,0],[0,1],[1,1]]`, V=`[[2,0],[0,2],[1,1]]` → A rows sum to 1, O=`[[1.267,0.733],[0.733,1.267]]` | **confirmed** — four fully-decoupled axes: query-pos `q`, key/val-pos `s`, score-feature `h` (Q/K only), value-feature `g` (K/V only, `h`≠`g` as roles). Q/K may come from a different source than V | `[N]` `[E]` |
| AT9 | **linear attention** (O(N), kernelized) `M[h,e] := PhiK[s,h]·V[s,e]` `O[q,e] := PhiQ[q,h]·M[h,e]` | Φ=I₂, V=`[[1,2],[3,4]]` → O=`[[1,2],[3,4]]` | **confirmed** — contraction **reassociation**: sum keys first (`M[h,e]`) so cost is O(N) not O(N²); relu as the feature map φ | `[N]` `[E]` |
| AT10 | **bilinear attention** `S[q, s.] := softmax(Q[q,a]·W[a,b]·K[s,b])` | I₂ inputs → rows sum to 1 | **confirmed** — learned bilinear scoring `qᵀWk` (3-factor contraction inside softmax) | `[N]` |
| AT11 | **grouped/multi-query attention** `A[h, q, s.] := softmax(Q[h,q,d]·K[s,d])` | K has no `h` → shared across heads | **confirmed** — K/V **broadcast across heads** (GQA/MQA) | `[N]` `[E]` |
| AT12 | **sparse attention** (Longformer local+global) `A[q, s.] := softmax(where \|q−s\| ≤ 1 ∨ s = 0)(Q[q,d]·K[s,d])` | Q=K=I₃ → `A[0,2]=0` (out of window), global token `s=0` attended from every `q` (`A[2,0]>0`) | **confirmed** — first use of `∨` in a mask; local window OR a global token | `[N]` `[E]` |

> Attention variants that hit gaps: **GAT** (graph attention) needs a *data-driven* masked
> softmax (`where edge[i,j]`) — not expressible, masks are index-predicates only (KG-datamask);
> **additive/Bahdanau** needs `tanh` (KG-activation); **ALiBi / relative-position numeric bias**
> needs an index-derived numeric value `−m·|q−s|` — index arithmetic only yields booleans via
> Iverson, not tensor values (KG-idxvalue).

> **AT8 (generalized cross-attention)** is the einsum realization of the Glaive "Generalized
> Transformers from Applicative Functors" `attention` combinator
> (`fmap softmax (queries ·mulMMT· keys) ·mulMM· values`), where `queries : f(h)`,
> `keys : i(h)`, `values : i(g)`, `output : f(g)`. The flat DSL captures the full *axis
> decoupling* (q, s, h, g independent; Q/K cross-source from V). What it **cannot** express is
> the article's deeper generality — the index spaces `f, g, h, i` being arbitrary *applicative
> functors* (trees, functions, nested containers) rather than flat `Fin n` axes. That
> structured-index generalization is a known gap: **KG-functor** (§14) — a good pointer to the
> Naperian/`act` typing work in `papers/NaperianTypingIntegrationPlan.md`.

> Note AT6/residuals: because the RHS sum has **no tensor subtraction** and no scalar literals,
> "scale by 1/√d" and "x + sublayer(x)" both rely on carrying a helper tensor. AT6 tests that a
> rank-0 tensor read is legal; if it isn't, this becomes an `[F]` (see KG-scale).

## 5. Convolution & pooling

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| CV1 | `Y[i] := W[p] · X[i+p]` | W=`[1,1]`, X=`[1,2,3,4]` → `[3,5,7]` | 1-D valid convolution (affine read) | `[N]` |
| CV2 | `Y[i,j] := W[p,q] · X[i+p, j+q]` | W=I₂, X=`[[1,2,3],[4,5,6],[7,8,9]]` → `[[6,8],[12,14]]` | 2-D convolution | `[N]` |
| CV3 | `Y[i,j] := W[p,r] · X[i+p, 2*j+r]` | conv with stride 2 | strided conv (literal stride in read) | `[✔]` |
| CV4 | `Y[i] := W[p] · X[i + 2*p]` | W=`[1,1]`, X=`[0,1,2,3,4]` → `[2,4,6]` (X[i]+X[i+2]) | **dilated** conv (dilation 2) | `[N]` `[E]` |
| CV5 | `P[i] := maxreduce(X[2*i + p])` | X=`[1,3,2,5]`, p∈{0,1} → `[3,5]` | **max-pool** stride-2 window-2 | `[N]` |
| CV6 | `P[i] := X[2*i+p] · w[p]` | w=`[0.5,0.5]`, X=`[2,4,6,8]` → `[3,7]` | **avg-pool** as weighted sum | `[N]` |
| CV7 | `s[] := X[i]` | X=`[1,2,3,4]` → `10` | global sum-pool (contract all) | `[N]` `[E]` |
| CV8 | **depthwise / grouped conv** `Y[c,i] := W[c,p]·X[c,i+p]` | W=`[[1,1],[1,-1]]`, X=`[[1,2,3],[4,5,6]]` → `[[3,5],[-1,-1]]` | **confirmed** — channel `c` is **free on both** W and X (not contracted); per-channel kernel | `[N]` `[E]` |
| CV9 | **full multi-channel conv** `Y[co,i] := W[co,ci,p]·X[ci,i+p]` | W=`[[[1,0],[0,1]]]` (1×2×2), X=`[[1,2,3],[4,5,6]]` → `[[6,8]]` | **confirmed** — contracts **input-channel `ci` AND kernel `p` together**, output-channel `co` free (contrast CV8 depthwise) | `[N]` `[E]` |

## 6. Normalization

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| NM1 | `Y[q, s.] := normalize(A[q,s])` | A=`[[1,3],[2,2]]` → `[[0.25,0.75],[0.5,0.5]]` | L1 normalize over marked axis | `[✔]`-ish |
| NM2 | `Y[q, s.] := softmax(A[q,s])` | A=`[[0,0],[0,ln3]]` → `[[.5,.5],[.25,.75]]` | plain softmax → distribution | `[N]` |
| NM3 | `Y[q, s.] := softmax(A[q,s])` | A=`[[1000,1001]]` → `[[0.2689,0.7311]]` | **numerical stability** (max-subtraction; no overflow) | `[N]` `[E]` |
| NM4 | `Y[q,s.] := normalize(where s ≠ 0)(A[q,s])` | masked entry → 0, rest renormalized | masked L1 normalize | `[N]` `[E]` |
| NM5 | **softmax over a non-last axis** (column softmax / axial attn) `A[s., q] := softmax(Q[q,d]·K[s,d])` | Q=K=I₂ → `[[0.731,0.269],[0.269,0.731]]`, each **column** sums to 1 | **confirmed** — reduction axis marked at slot **position 0**, not the last; exercises `axisPos ≠ last` | `[N]` `[E]` |

## 7. Recurrence & scans

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| RC1 | coupled scan `G/H` (relu step) | all-ones → G=`[1,3,6]`, H=`[2,3,6]` | coupled scan; `axis l : ℕ = 3` pin | `[✔]` |
| RC2 | simple RNN `S[j,0]:=X0[j]` `S[j,l +1] := relu(S[j,l]·W[j,k])` | 1-feature, W=1 | single scan, self-recurrence | `[N]` |
| RC3 | prefix-sum via mask `C[i] := X[j] · [j ≤ i]` | X=`[1,2,3]`, `axis i,j = 3` → `[1,3,6]` | **cumulative sum without a scan** (triangular Iverson) | `[N]` `[E]` |
| RC4 | `S[j, l +1] := S[j, l] + X[j, l + 1]` (scan step reads external at advancing index) | `X=[[1,2,3]]` | **CONFIRMED GAP**: compile → `causalityViolation "S"` | `[R]` |
| RC5 | `M[j, l +1] := maxreduce(M[j,l] · W[j,k])` (`maxreduce` in a scan step) | `X=[2], W=[[1,3]]` → `[2,6,18]` (max-step `M×3`) | **FIXED (KG-scanagg)**: the scan step now honors the `maxreduce` agg (tropical max); was silently summed to `[2,8,32]` before the fix | `[N]` |
| RC6 | **2D / nested recurrence** (grid-DP / PixelRNN) `G[r +1, c +1] := G[r,c] + A[r,c]` | base G=0, A=ones(2×2) → `[[0,0],[0,1]]` (only the fully-advanced cell `G[1,1]` is written; boundary cells `r=0`/`c=0` keep the zero-default) | **FIXED (KG-2dscan)**: both advancing iteration axes are now tracked through the scan; was silently collapsing to a 1-D scan (`[[0,1],[0,1]]`, overwriting the second axis's base case) before the fix | `[N]` |
| RC7 | `M[j, l +1] := minreduce(M[j,l] · W[j,k])` (`minreduce` in a scan step) | `X=[2], W=[[2,3]]` → `[2,4,8]` (min-step `M×2`) | **min-agg in a scan** (KG-min; mirrors RC5) — the scan step honors `minreduce` via the KG-scanagg plumbing | `[N]` |
| RC8 | **3-D nested scan** `G[a +1, b +1, d +1] := G[a,b,d] + T[a,b,d]` | axes a,b,d size 2; base S=0, T=ones(2×2×2) → 2×2×2 tensor with a single `1` at `[1,1,1]`, all boundary cells `0` | **generality of n-D support** (KG-2dscan): confirms multi-axis scans generalize past the 2-D case (RC6) to 3 axes, with the same zero-default boundary semantics | `[N]` |

> RC4/RC5/RC6 were the "verify" cases — confirmed against HEAD (2026-07-04); RC5 and RC6 fixed
> 2026-07-08.
> **RC4**: a scan step reading an external at the advancing index `l + 1` is rejected with
> `CompileError.causalityViolation` (the compiler treats any next-index read as a future
> dependency, even for non-state externals). Author as a reject-test asserting that error.
> **RC5 (KG-scanagg, FIXED 2026-07-08)**: `maxreduce` inside a scan step used to be silently
> ignored — the step performed a sum contraction and returned `2·(1+3)=8` rather than
> `max(2·1,2·3)=6`. Fixed by threading the aggregator's `Combine` through `evalStmtSlice` /
> `evalAssignSeeded` (`Eval/Scan.lean`, `Eval/Contract.lean`) so a `maxreduce` step reduces with
> tropical max `(×, max, −∞)`; RC5 now asserts the correct `[2,6,18]`.
> **RC6/RC8 (KG-2dscan, FIXED 2026-07-08)**: multi-axis (n-D) recurrences now evaluate correctly
> — positional base recovery plus nested evaluation track every advancing axis, and a scan step
> writes only fully-advanced cells (boundary cells keep their zero-allocated/base value —
> **zero-default** boundary semantics). RC6 (2-D) now asserts the correct `[[0,0],[0,1]]`
> (previously collapsed to a 1-D scan giving `[[0,1],[0,1]]`); RC8 (3-D) confirms the fix
> generalizes to 3 axes. `RC6-compile` additionally checks at the compile level that the 2-D scan
> lowers to a single well-formed Br step with `.op = .scan` and `degree.length == 2`, i.e. both
> iteration axes survive lowering.

## 8. Graph / message passing (GNN)

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| GN1 | `H[i,f] := A[i,j] · X[j,f]` | A=adjacency `[[0,1],[1,0]]`, X=`[[1,2],[3,4]]` → `[[3,4],[1,2]]` | dense message passing (A·X) | `[N]` |
| GN2 | `H[i,f] := edge[i,j] · X[j,f]` (`predicate edge`) | edge booleanizes | message passing over a **predicate** adjacency | `[N]` |
| GN3 | `deg[i] := edge[i,j]` | edge=`[[0,1,1],[1,0,0]]` → `[2,1]` | node degree = contract over neighbors | `[N]` |
| GN4 | `H[i,f] := maxreduce(edge[i,j] · X[j,f])` | max-aggregation GNN (GraphSAGE-max) | `maxreduce` neighbor aggregation | `[N]` `[E]` |
| GN5 | degree-normalized `Ĥ = D⁻¹ A X` | needs division by `deg[i]` | **no per-node division** ⇒ expected gap | `[F]` (KG-div) |
| GN6 | edge-list scatter-add `Msg[dst[e]] += X[src[e]]` | index tensors `src/dst` | data-dependent gather/scatter | `[F]` (KG-gather) |

## 8b. Consuming scatter outputs (upsampling decoders / GNN readback)

Reading the output of a scatter statement in *later* statements — the decoder/GNN pattern added
by commits `57d333f`/`fc10d70` (B3). This was **unsupported end-to-end before B3**, so the whole
block doubles as a regression suite. All eight consumers **probed against HEAD (2026-07-04)** and
numerically correct. The shared base is a 2× upsample `Out[2*i,2*j] := X[i,j]` with X=`[[1,2],[3,4]]`
(⇒ `Out` is 4×4 with the input values at even coordinates, 0 elsewhere).

| ID | Consumer of the scatter output `Out` | Data → Expected | Probes | Tag |
|----|--------------------------------------|-----------------|--------|-----|
| SC1 | reduce `total[] := Out[a,b]` | → `10` (Σ of X) | contract a scatter output to a scalar | `[N]` |
| SC2 | elementwise square `Sq[a,b] := Out[a,b]·Out[a,b]` | → `[1,0,4,0; 0…; 9,0,16,0; 0…]` (Σ=30) | pointwise op on a scatter output | `[N]` |
| SC3 | `R[a,b] := relu(Out[a,b])` (X=`[[1,−2],[−3,4]]`) | negatives clamped → `1,0,0,4` at evens | nonlinearity on a scatter output | `[N]` |
| SC4 | matmul `Z[a,c] := Out[a,b]·W[b,c]` (W=ones 4×2) | → col = row-sums `[3,0,7,0]` | **contraction** consuming a scatter output | `[N]` |
| SC5 | conv `Y[a,b] := Wk[p]·Out[a+p,b]` (Wk=`[1,1]`) | shape `[3,4]`, `Y[0]=[1,0,2,0]`, `Y[1]=Y[2]=[3,0,4,0]` | **affine/strided read** of a scatter output (shape solver over a scattered tensor) | `[N]` `[E]` |
| SC6 | scatter-of-scatter `Out2[2*a,2*b] := Out[a,b]` | 8×8, values `1,2,3,4` at `(0,0),(0,4),(4,0),(4,4)` | a scatter **reading another scatter** (stacked upsampling) | `[N]` `[E]` |
| SC7 | diagonal-scatter then matmul `D[i,i]:=v[i]`; `Y[i,j]:=D[i,k]·M[k,j]` | v=`[2,3]`, M=ones → `[[2,2],[3,3]]` | a **diagonal-write** scatter consumed in a contraction (row scaling) | `[N]` `[E]` |
| SC8 | `P[a, b.] := softmax(Out[a,b])` | rows sum to 1; all-zero rows → uniform `[¼,¼,¼,¼]` | softmax over a scatter output (its structural 0s are real values, not masked) | `[N]` `[E]` |

> SC8 is a subtle one worth calling out in review: a scatter output's padding is genuine `0.0`,
> so `softmax`/`normalize` treat those cells as real entries (an all-zero row becomes uniform,
> not undefined). If a decoder wants padding excluded from a softmax, it needs an explicit mask —
> the scatter fill value doesn't propagate as "masked".

## 9. Relational / logic (tensor-logic's logic side)

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| RL1 | `I[i,j] := [i = j]` with `axis i:ℕ=3, j:ℕ=3` | → 3×3 identity | pure-Iverson tensor; **sizes from decls only** (no read binds i,j) | `[N]` `[E]` |
| RL2 | `R[i,k] := E1[i,j] · E2[j,k]` (`predicate`) | relational **join** / composition; counts 2-paths | boolean semiring product | `[N]` |
| RL3 | `U[i,j] := E[i,j] · [i < j]` | strict-upper-triangular selection | Iverson **selection** predicate | `[N]` |
| RL4 | `P2[i,k] := E[i,j] · E[j,k]` | E=`[[0,1],[0,0]]` → path counts | length-2 reachability (path count) | `[N]` |
| RL5 | band mask `Band[i,j] := A[i,j] · [\|i − j\| ≤ 1]` | tri-diagonal keep | `iabs` predicate arithmetic | `[✔]` |
| RL6 | `S[i,j] := A[i,j] · [i ≤ j ∧ j ≤ i + 2]` | local window | **compound** boolean (`∧`) mask | `[N]` `[E]` |
| RL7 | `M[i,j] := A[i,j] · [ieq(imul(i,2), j)]` | keep where `2i = j` | `ieq`/`imul` predicate builtins | `[N]` `[E]` |
| RL8 | **negated mask** (exclude-self) `M[i,j] := A[i,j] · [¬(i = j)]` | A=`[[1,2],[3,4]]` → `[[0,2],[3,0]]` | **confirmed** — first use of `¬`; zeroes the diagonal | `[N]` `[E]` |

## 10. Losses, reductions & statistics

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| ST1 | `sse[] := r[i] · r[i]` | r=`[1,-2,2]` → `9` | sum-of-squares (given residual `r`) | `[N]` |
| ST2 | `C[i,j] := X[k,i] · X[k,j]` | X=`[[1,2],[3,4]]` (2 samples×2 feat) → `[[10,14],[14,20]]` | uncentered covariance / scatter matrix | `[N]` |
| ST3 | `m[j] := X[k,j] · invn[]` | X=`[[2,4],[6,8]]`, invn=`[0.5]` → `[4,6]` | mean over samples via rank-0 `1/n` | `[N]` `[E]` |
| ST4 | `p[i] := joint[i,j]` | marginalize out `j` | probabilistic marginalization = sum | `[N]` |
| ST5 | residual via `−1` scalar `r[i] := Yhat[i] + m1[]·Y[i]` (`m1=−1`) | Yhat=`[5,3]`, Y=`[2,1]` → `[3,2]` | **CONFIRMED**: no `−` operator, but subtraction works via a rank-0 `−1` (⇒ MSE `sse[]:=r[i]·r[i]` is expressible) | `[N]` `[E]` |
| ST6 | cross-entropy `L[] := − Y[i] · log(P[i])` | — | **no standalone `log`** (the `−` part is fine via ST5's trick) | `[F]` (KG-log) |

## 11. Tropical / max-times semiring

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| TR1 | `C[i] := maxreduce(A[i,k])` | A=`[[3,1,4],[1,5,9]]` → `[4,9]` | basic max-reduction | `[✔]` |
| TR2 | `C[i] := maxreduce(A[i,k] · B[k])` | max of products | max-times combine | `[✔]` |
| TR3 | `R[i,k] := maxreduce(P[i,j] · P[j,k])` | P=reliability `[[0,0.9],[0.8,0]]` → most-reliable 2-hop | max-times "best path" | `[N]` `[E]` |
| TR4 | all-negative input max | A=`[[-2,-5,-1]]` → `[-1]` | `-∞` unit doesn't pollute | `[✔]` |
| TR6 | `C[i] := minreduce(A[i,k])` | A=`[[3,1,4],[1,5,9]]` → `[1,1]` | **basic min-reduction** (min analog of TR1); tropical min `(×, min, +∞)`, added 2026-07-08 | `[N]` |
| TR7 | `C[i] := minreduce(A[i,k] · B[k])` | A=`[[3,1,4],[2,2,2]]`, B=`[2,5,1]` → `[4,2]` | **min-times combine** (min analog of TR2) | `[N]` |
| TR5 | min-*plus* shortest path (Viterbi/Bellman step) | — | `minreduce` supplies the `min` agg, but min-plus still needs `+` as the within-term combine ⇒ **partial gap** | `[F]` (KG-min, min-plus half) |

## 12. Tensor networks / decomposition

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| TN1 | `Y[a,d] := G1[a,b] · G2[b,c] · G3[c,d]` | 3 identity cores → identity | tensor-train / chain contraction | `[N]` |
| TN2 | `T[i,j,k] := A[i,r] · B[j,r] · C[k,r]` | rank-1 factors → outer product | **CP reconstruction** (shared contracted `r`) | `[N]` `[E]` |
| TN3 | `s[] := A[i,j] · B[i,j]` | A=`[[1,2],[3,4]]`, B=I₂ → `5` | Frobenius inner product (full contract, 2 axes) | `[N]` |
| TN4 | **third-order moment** (method of moments) `M[i,j,k] := X[t,i]·X[t,j]·X[t,k]` | X=`[[1,0],[0,1]]` (t=2) → diagonal 3-tensor `M[0,0,0]=M[1,1,1]=1` | **confirmed** — **3× aliasing** of one tensor (Gram is 2×) with a shared contracted `t`, rank-3 output | `[N]` `[E]` |

## 12b. Advanced & generative domains

Core programs of each domain were **probed against HEAD (2026-07-04)** so tags are grounded.
Recurring finding: the DSL's missing scalar/transcendental ops (subtraction workaround, no
`log`/`sin`) determine which parts are expressible.

### Diffusion

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| DF1 | forward process `Xt[i] := alpha[]·x0[i] + beta[]·eps[i]` | α=`0.6`, β=`0.8`, x0=`[1,0]`, ε=`[0,1]` → `[0.6,0.8]` | **confirmed** — rank-0 scalar scaling + 2-term sum | `[N]` |
| DF2 | DDPM posterior mean `Xp[i] := c1[]·Xt[i] + c2[]·e[i]` (`c2<0`) | another scaled combo | reverse-step linear combination (uses `−` scalar) | `[N]` |
| DF3 | denoising loss `sse[] := r[i]·r[i]` with `r[i]:=eps[i]+m1[]·ehat[i]` | ε,ε̂ small → hand-check | **expressible** via ST5 subtraction trick + sum-of-squares | `[N]` |
| DF4 | sinusoidal timestep embedding `te[t,2i]:=sin(t·ω^i)` | — | **no `sin`/`cos`** | `[F]` (KG-trig) |

### Mixture of Experts

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| ME1 | gating `g[t, e.] := softmax(X[t,d]·Wg[d,e])` | rows sum to 1 | softmax router over experts | `[N]` |
| ME2 | batched expert MLP `Y[t,e,f] := relu(We[e,f,d]·X[t,d])` | — | rank-3 expert weight, expert axis `e` free | `[N]` |
| ME3 | combine `Out[t,f] := g[t,e]·Y[t,e,f]` (chained on ME1/ME2) | **confirmed** evals end-to-end (shape `[1,1]`) | gated weighted sum over experts | `[N]` |
| ME4 | top-k routing (keep k highest-gated experts) | — | **no argmax / top-k gather** | `[F]` (KG-gather) |

### State-space / sequence models (SSM, S4/Mamba-style)

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| SS1 | linear recurrence `h[j,l +1] := A[j,k]·h[k,l] + B[j]·u[l]` | A=`[[1]]`, B=`[1]`, u=`[1,1,1]` → h=`[1,2,3]` | **confirmed** — scan reading input at the **current** index `u[l]` | `[N]` |
| SS2 | output map `y[j,l] := C[j,k]·h[k,l]` (chained on SS1) | — | per-step output projection off the scan state | `[N]` |
| SS3 | diagonal SSM `h[j,l +1] := a[j]·h[j,l] + B[j]·u[l]` | diagonal `A` (Mamba-ish) | elementwise (diagonal) state transition | `[N]` `[E]` |
| SS4 | input at advancing index `… + B[j]·u[l + 1]` | — | **rejected** — `causalityViolation` (cf. RC4); convention matters | `[R]` |

### Positional encodings

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| PE1 | learned additive `Out[p,d] := X[p,d] + PE[p,d]` | X=`[[1,2],[3,4]]`, PE=`[[10,20],[30,40]]` → `[[11,22],[33,44]]` | **confirmed** — elementwise add of same-shape reads | `[N]` |
| PE2 | sinusoidal `pe[p,2i] := sin(p·ω^i)`, `pe[p,2i+1]:=cos(…)` | — | **no `sin`/`cos`** | `[F]` (KG-trig) |
| PE3 | RoPE rotation of `(q_2i, q_2i+1)` by `θ_p` | — | needs `sin`/`cos` + paired rotation | `[F]` (KG-trig) |

### Contrastive / metric learning

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| CL1 | similarity matrix `S[i,j] := Z1[i,d]·Z2[j,d]` | Z1=Z2=I₂ → `[[1,0],[0,1]]` | pairwise dot-product similarity (matmul) | `[N]` |
| CL2 | `P[i, j.] := softmax(S[i,j])` (chained) | rows sum to 1 | InfoNCE softmax over the negatives | `[N]` |
| CL3 | cosine similarity (L2-normalize embeddings first) | — | **only L1 `normalize`, no L2** | `[F]` (KG-l2norm) |
| CL4 | InfoNCE loss `L := − log P[i,i]` | — | **no `log`** (subtraction fine via ST5) | `[F]` (KG-log) |

## 12c. Classical ML, probabilistic & RL

Less-obvious use cases, all probed against HEAD. Several are **multi-statement by necessity**
because of the equation-level summation rule (see the callout after this section) — a single
RHS sum can't mix terms with different contracted axes, so each contraction is materialized into
its own intermediate first.

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| CM1 | **pairwise squared distance** (k-NN / RBF kernel numerator): `sq[i]:=X[i,d]·X[i,d]`; `cross[i,j]:=X[i,d]·X[j,d]`; `D[i,j] := sq[i]·one[j] + one[i]·sq[j] + m2[]·cross[i,j]` | X=`[[0,0],[3,4]]`, one=`[1,1]`, m2=`−2` → `[[0,25],[25,0]]` | **confirmed** — broadcast (`sq[i]·one[j]`) + subtraction scalar + materialized cross term | `[N]` `[E]` |
| CM2 | **factorization machine** 2nd-order: `sqsum[f]:=V[i,f]·x[i]·V[j,f]·x[j]`; `sumsq[f]:=V[i,f]·x[i]·V[i,f]·x[i]`; `fm[f]:=hp[]·sqsum[f]+hm[]·sumsq[f]` | V=`[[1],[2]]`, x=`[1,1]`, hp=`0.5`, hm=`−0.5` → `[2]` | **confirmed** — the `½((Σ)²−Σ()²)` identity; distinguishes **reused dummy `i`** (Σ()²) from **fresh dummy `j`** (Σ)² | `[N]` `[E]` |
| CM3 | **value iteration / Bellman backup**: `EV[s,a]:=gamma[]·P[s,a,s2]·V[s2]`; `Q[s,a]:=R[s,a]+EV[s,a]`; `Vn[s]:=maxreduce(Q[s,a])` | R=I₂, γ=`0.9`, P=`½` unif, V=`[1,1]` → `[1.9,1.9]` | **confirmed** — `maxreduce` over actions + discounted expectation (rank-0 γ). *Iterated* VI hits KG-scanagg (maxreduce-in-scan) | `[N]` |
| CM4 | **power iteration / Markov / HMM-forward (sum-product)**: `p[j,0]:=p0[j]`; `p[j,l +1]:=p[i,l]·M[i,j]` | p0=`[1,0]`, M=swap → p=`[[1,0,1],[0,1,0]]` | **confirmed** — **contraction inside a scan** works correctly (contrast RC5: `maxreduce`-in-scan does not) | `[N]` |
| CM5 | **matrix factorization** `Rhat[u,i] := P[u,f]·Q[i,f]` | small | recommender dot-product model | `[N]` |
| CM6 | **VAE reparameterization** `z[i] := mu[i] + sigma[i]·eps[i]` | Hadamard + add | elementwise `μ+σ⊙ε`; the KL term needs `log` (KG-log) | `[N]` `[E]` |
| CM7 | k-means assignment / nearest-centroid (argmin over centroids) | — | **no `argmin`** (only `max` value, no index) | `[F]` (KG-min) |
| CM8 | closed-form linear regression `β = (XᵀX)⁻¹Xᵀy` | XᵀX, Xᵀy expressible; the solve is not | **no linear solve / inverse** | `[F]` (KG-solve) |
| CM9 | logistic regression / GRU-LSTM gates (`σ`, `tanh`) | — | **only relu** activation | `[F]` (KG-activation) |

> **⚠ Equation-level summation (a semantic gotcha, confirmed 2026-07-04).** Tensor logic sums
> over *every* index variable not on the LHS, applied to the **whole RHS** — not per product
> term. So `Y[i] := a[i] + b[i,k]·c[k]` sums `k` over the entire equation and **broadcasts the
> `k`-less term** `a[i]` by `|k|`. This silently gave wrong results in first drafts of CM1
> (`[[0,50],[50,50]]` instead of `[[0,25],[25,0]]`) and CM3 (`[2.9,2.9]` instead of
> `[1.9,1.9]`). **Safe pattern:** materialize each differently-contracted subexpression into its
> own intermediate whose index set matches the LHS, then sum intermediates that all share the
> same free axes. This is arguably the *defined* Datalog/Einstein-at-equation-scope semantics,
> not a bug — but it is a sharp edge worth a dedicated test (EC15) and a prominent doc note.

## 13. Adversarial — reject tests (assert an error)

Each asserts a **specific** `CompileError`/`EvalError` (or a parse failure). These lock in the
DSL's intended boundaries; if a future change accepts them silently, the test fails.

| ID | Prog | Expected failure | Tag |
|----|------|------------------|-----|
| RJ1 | `Y[i] := W[p] · X[s * j]` | parse error — symbolic-coefficient stride `s*j` not in `IdxExpr` | `[R]` |
| RJ2 | `Y[i] := X[i / j]` | parse error — `/` requires a **literal** divisor | `[R]` |
| RJ3 | `predicate P(i)` … `P[i] := maxreduce(edge[i,j])` | `checkDtypes` → `predicateAgg` (bool output + non-sum agg) | `[R]` |
| RJ4 | `A[q,s] := softmax(Q[q,d]·K[s,d])` (no `.` marker) | eval error — softmax with no marked reduction axis | `[R]` |
| RJ5 | over-indexed read of an undeclared intermediate (e.g. `T[i,j,k]` where `T` was written rank-2) | Task-A rejection of over-indexed undeclared read | `[R]` (not automated) |
| ~~RJ6~~ | ~~scan step with no input and no `axis l:ℕ=N` pin~~ | **DROPPED — does NOT reject** (probed: evaluates to a 0-step/defaulted result, no error) | — |
| RJ7 | `s[] := A[i] · B[i]` with A length 3, B length 2 | eval error — affine size system inconsistent (`0 = -1`) — **confirmed** | `[R]` |
| RJ8 | `A[q, d.] := softmax(...)` where `d` is contracted (not an output axis) | not constructible via surface syntax (marking always places the slot on the LHS) | `[R]` (not automated) |
| ~~RJ9~~ | ~~`Y[i] := X[i - 5]` short input~~ | **DROPPED — does NOT reject** (probed: evaluates, zero-padded; no Issue-D error at this size) | — |
| RJ10 | scatter whose output axis is unsized | `scatterOutShape` fail-loud on unsized axis | `[R]` (not readily constructible) |

> **Implemented (`RejectTest.lean`):** RJ3 (`predicateAgg`, compile), RJ4 (eval), RJ7 (eval),
> and SS4/RC4 (`causalityViolation`, compile). RJ1/RJ2 fail at **parse time** — a hard parse
> error fails the build and `#guard_msgs` does not validate parse errors, so they are kept as
> documentation comments in the reject file rather than live tests. RJ6/RJ9 were dropped after
> probing showed they don't reject. RJ5/RJ8/RJ10 aren't readily constructible via surface syntax.

## 14. Adversarial — known gaps / expected-fail

Documented constructs that a user might reasonably expect but that do **not** work today.
Author as expected-fail so that (a) the gap is visible and (b) the test flips green the day the
gap is closed — a built-in regression alarm. Cross-referenced above by `KG-*`.

| ID | Gap | Where it bites | Tag |
|----|-----|----------------|-----|
| KG-sub | no `−` operator in the RHS sum — **but** subtraction is achievable via a rank-0 `−1` scalar (`a[i] + neg[]·b[i]`, *confirmed* — see ST5). Only a *soft* gap: the ergonomic form is missing, the capability is not | residuals, MSE, centering | `[F]` (soft) |
| KG-log | standalone `log`/`exp` (only relu/softmax/normalize) | cross-entropy, log-likelihood, GELU | `[F]` |
| KG-trig | no `sin`/`cos`/transcendental fns | sinusoidal PE, RoPE, diffusion timestep embeddings | `[F]` |
| KG-l2norm | only **L1** `normalize` (`x/Σx`); no L2 / `√Σx²` | cosine similarity, RMSNorm, unit-norm embeddings | `[F]` |
| KG-functor | axes are flat `Fin n` only — no **structured/applicative-functor index spaces** (trees, functions, nested containers) | generalized transformers over non-sequence data (parsing, image→function); see AT8 & `NaperianTypingIntegrationPlan.md` | `[F]` |
| KG-reshape | LHS index slots are **single-axis affine only** (`2*i`, `i+num`); no multi-axis-into-one-slot (`i+j`, `2*i+k`) — *confirmed parse-fail* | reshape/flatten, Kronecker product, im2col, overlapping-accumulate scatter (also unreachable via surface syntax) | `[F]` |
| KG-datamask | `where` masks are **index predicates only** — cannot read a data tensor (`where edge[i,j] > 0`) — *confirmed parse-fail* | GAT / any graph-structured masked softmax over a runtime adjacency | `[F]` |
| KG-solve | no linear **solve / inverse / eigendecomp / determinant** (only contraction) | closed-form regression, PCA, spectral GCN, normalizing-flow log-det | `[F]` |
| KG-activation | only **relu** (+ softmax/normalize) — no `sigmoid`/`tanh`/`gelu`/`leakyrelu` | LSTM/GRU gates, logistic regression, GAT, GELU transformers | `[F]` |
| KG-idxvalue | index arithmetic yields **booleans only** (via Iverson) — never a numeric tensor value | ALiBi, relative-position numeric bias, distance-decay weights | `[F]` |
| KG-sqrt | no `sqrt` | true Euclidean distance (vs squared), std / RMSNorm, unit-norm (cf. KG-l2norm) | `[F]` |
| KG-min | **min aggregation FIXED 2026-07-08** (`minreduce`, tropical min-times `(×, min, +∞)` — see TR6/TR7, RC7). Still open: **additive within-term combine** (min-*plus* semiring, `+` instead of `×`) and **argmin** (index output, cf. KG-idxvalue) | shortest path, Viterbi, DTW (need min-plus); k-means (needs argmin) | `[F]` (partial) |
| KG-div | division by a tensor / per-element reciprocal (only whole-axis `normalize`) | D⁻¹AX GNN norm, layernorm variance, softmax-free attention | `[F]` |
| KG-gather | data-dependent gather/scatter with an **index tensor** (indices must be affine) | embedding lookup by ids, edge-list scatter-add, top-k | `[F]` |
| KG-scale | free scalar literal on the RHS (e.g. `· 0.5`) | 1/√d attention scaling (workaround: rank-0 tensor, see AT6) | `[F]`? |
| KG-multiout | multiple outputs **not** at the tail statement | multi-head split, aux losses | `[F]` |
| KG-recur | `recurMorphism` / `scanPre` escape hatch (AST-only) | programmatic recurrences | `[F]` |
| ~~KG-scanagg~~ | **FIXED 2026-07-08** — `maxreduce` inside a scan step now reduces with tropical max (was silently summed). Scan steps thread the agg's `Combine` through `evalStmtSlice`/`evalAssignSeeded`. Live test: RC5 (now `[N]`, asserts `[2,6,18]`) | max recurrences (running-max); *argmax* still blocked by KG-min (no index output) | ✓ fixed |
| ~~KG-2dscan~~ | **FIXED 2026-07-08** — n-D scans (2-D and beyond) are now supported: positional base recovery plus nested evaluation track every advancing axis, and a scan step writes only fully-advanced cells (zero-default boundary semantics). Was: 2-D/nested recurrences silently collapsed to a 1-D scan (two advancing axes reduced to one; no error). Live tests: RC6 (2-D, asserts `[[0,0],[0,1]]`), RC8 (3-D generality), and `RC6-compile` (compile-level check that the 2-D scan lowers to a well-formed rank-2 Br `.scan` step) | grid-DP, PixelRNN, edit-distance/NW, CTC alignment tables | ✓ fixed |

> Verification note: some of these may already be partially addressed (e.g. reading scatter
> outputs across layers was **recently closed** by commits `57d333f`/`fc10d70` — that one belongs
> in §15 as a passing regression test, not here). Each `KG-*` needs a 30-second confirmation
> against `HEAD` before we commit to "expected-fail", so we don't ship a red test for something
> that actually works.

> **Design direction — general semiring notation (planned, not yet implemented).** The remaining
> tropical/semiring gaps (KG-min's min-*plus* half, and future max-plus / log-sum-exp) should be
> closed by adopting a **general semiring selector** rather than adding one bespoke `AggOp`
> keyword per combination. The intended surface form names both operations explicitly — an
> aggregate `⊕` and a within-term combine `⊗`, e.g. `D[i,k] := agg(min, +)(D[i,j] · W[j,k])`
> — where inside the form `·` denotes `⊗` and the reduction/`+` across terms denotes `⊕`. This
> subsumes the current `maxreduce`/`minreduce` (= `agg(max, ·)` / `agg(min, ·)`) and covers
> min-plus `agg(min, +)`, max-plus `agg(max, +)`, min-times `agg(min, ·)`, etc. Adopt this
> notation **where appropriate** when the min-plus work is scheduled. Implementation note: a
> non-`×` within-term combine requires the product accumulator to start at the `⊗`-identity
> (`0` for `+`, `1` for `·`), so `Combine` needs a `⊗`-unit field (today the product init is
> hardcoded to `1.0`).

## 15. Adversarial — tricky-but-valid edge cases (should pass)

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| EC1 | `Y[i] := X[i - 1]` | X=`[10,20,30]` → `[0,10,20,30]` (shape **4**, not 3) | boundary read → **zero-pad**; **CORRECTED**: the shape solver's "maximal padded extent" gives length `n+shift` (confirmed against HEAD; the original `[0,10,20]`/length-3 draft was wrong) | `[N]` `[E]` |
| EC2 | `Y[i] := X[i - 3]` | X=`[10,20,30]` → `[0,0,0,10,20,30]` (shape **6**) | multi-step look-back + padding; same `n+shift` rule as EC1 | `[N]` `[E]` |
| EC3 | `Y[i,j] := A[i,k] · B[k,j]` with `k` size 1 | A 2×1, B 1×2 | **size-1** contracted axis (rank-1 outer via matmul) | `[N]` `[E]` |
| EC4 | `Y[i] := A[i,j] · B[j]` with a **size-1** free axis `i` | 1×3, 3 → 1 | singleton free dim | `[N]` `[E]` |
| EC5 | `d[i] := M[i,i]` | M=`[[1,2],[3,4]]` → `[1,4]` | **diagonal read** (repeated axis on RHS) — *confirmed valid* | `[N]` `[E]` |
| EC6 | `t[] := M[i,i]` | M above → `5` | trace (diagonal read + full contraction) — *confirmed valid* | `[N]` `[E]` |
| EC7 | `D[i,i] := v[i]` | v=`[5,6]` → `[[5,0],[0,6]]` | **diagonal write** (scatter reclassification) | `[N]` `[E]` |
| EC8 | `Out[2*i,2*j] := X[i,j]` | 2×2 → 4×4 placement | upsample scatter | `[✔]` |
| EC9 | ~~`Out[i + j] := X[i,j]`~~ | — | **CONFIRMED does not parse** — LHS slots are single-axis affine only (`i + num`, `2*i`); `i + j` (two axes → one slot) is rejected. See KG-reshape (§14); overlapping-accumulate scatter is unreachable via surface syntax | `[F]` |
| EC10 | `Y[i,j,k,l] := A[i,j,m] · B[m,k,l]` | small | **high-rank** (4 free + 1 contracted) tensor contraction | `[N]` `[E]` |
| EC11 | `Z := A · B · C · D` (n-ary product, all scalar reads) | → product | n-ary `·` associativity | `[N]` `[E]` |
| EC12 | `Y[i] := X[2*i + 3*j - 1]` chained affine | multi-term affine read index; **CORRECTION**: both `i` and `j` are unconstrained by any input shape (the read is the only occurrence of either), so this needs `axis i : ℕ = 2, j : ℕ = 2` pins to evaluate; `j` is RHS-only so it's summed | general integer-affine reindex | `[N]` |
| EC13 | `Y[i,j] := A[i,k]·B[k,j] + C[i,l]·D[l,j]` (**no parens** — `(…)` around a sub-product is **not** surface syntax; parenthesizing a `tl_prod_term` fails to parse) | small, size-1 `k`/`l` (to avoid the EC15 equation-level-summation confound) | `·` binds tighter than `+`; two contractions summed | `[N]` `[E]` |
| EC14 | `s[] := A[i,j,k,l,m]` | rank-5 full contraction | many contracted axes at once | `[N]` `[E]` |
| EC15 | **equation-level summation** `Y[i] := a[i]·u[] + W[i,k]·v[k]` | `k` is summed over the *whole* RHS ⇒ `a[i]·u[]` broadcast by `\|k\|` | pins the gotcha behavior (see §12c callout); asserts the *observed* semantics so a future per-term change is caught. **Naming hazard found while authoring:** a tensor/predicate named exactly `c` immediately followed by `[` (e.g. `c[k]`) fails to parse — NOT a DSL bug, but a **global token collision**: Mathlib's `Equiv.Perm` cycle notation (`Mathlib.GroupTheory.Perm.Cycle.Concrete`) registers `c[` as a single reserved token (`c[0,1,2]` = a permutation cycle), and Lean's tokenizer binds it greedily ahead of `ident` + `[`. Confirmed scoped to exactly the identifier `c` (verified `cc`, `c2`, uppercase `C` all parse fine). Avoid naming any tensor/predicate `c` in DSL programs | `[N]` `[E]` |

---

## 16. Coverage matrix (feature × where exercised)

| DSL feature | Existing | New in this portfolio |
|-------------|----------|-----------------------|
| Contraction (einsum) | matmul | LA1,LA5,LA7,LA9,TN* |
| Broadcasting / outer | — | LA4, TN2 |
| Hadamard (no contraction) | — | LA8 |
| Aliasing (same tensor twice) | — | LA6, ST2 |
| relu | scan, MLP | FF2, FF3 |
| softmax (± mask) | causal attn | AT1, AT4, AT5, NM2, NM3 |
| normalize (± mask) | normalize | NM1, NM4 |
| maxreduce / minreduce | max-pool | CV5, GN4, TR3 (max); TR6, TR7, RC7 (min) |
| Iverson / predicates | band, masked-agg | RL1–RL7, RC3 |
| Affine reads (conv/dilation/look-back) | strided conv, look-back | CV1,CV2,CV4, EC1,EC2,EC12 |
| Scatter (affine/diagonal write) | upsample | EC7, EC9 |
| Scans / recurrence | coupled scan | RC2, RC4, RC5 (maxreduce-in-scan) |
| `linear` + `bias` | parse only | FF1, FF4 |
| Rank-0 tensors / scalar scale | — | AT6, ST3, DF1, DF2, ST5 |
| Size inference from decls | — | RL1 |
| Scan reading input at index `l` | — | SS1, SS3, CM4 (SS4 = reject at `l+1`) |
| Contraction inside a scan | coupled scan | CM4 (sum), RC5 (`maxreduce`, fixed); RC6 (2-D) and RC8 (3-D) now work — n-D scans fixed (KG-2dscan) |
| Attention variants | causal attn | AT8 (generalized), AT9 (linear), AT10 (bilinear), AT11 (GQA), AT12 (sparse ∨) |
| Consuming scatter outputs (B3) | — | §8b SC1–SC8 (reduce/pointwise/relu/matmul/conv/scatter-of-scatter/diag/softmax) |
| Softmax/normalize over non-last axis | — | NM5 |
| `∨` / `¬` in masks | — | AT12 (∨), RL8 (¬) |
| Full vs depthwise conv | — | CV9 (contract channel+kernel) vs CV8 (channel free) |
| 3× aliasing / high-order moment | — | TN4 |
| Broadcast + subtraction-scalar patterns | — | CM1, CM2, CM3, DF*, ST5 |
| Equation-level summation gotcha | — | EC15 (+ §12c callout; CM1/CM3 need materialization) |
| Diffusion / MoE / SSM / PE / contrastive | — | §12b (DF*, ME*, SS*, PE*, CL*) |
| Classical ML / probabilistic / RL | — | §12c (CM1–CM9) |
| **Reject** paths | (a few) | RJ1–RJ10, SS4 |
| **Known gaps** | — | KG-* (§14): sub(soft), log, trig, l2norm, sqrt, min, div, gather, multiout, recur, scanagg, functor, reshape, datamask, solve, activation, idxvalue |

## 17. Decisions (resolved) & follow-ups

All reviewer questions are resolved (2026-07-04):

1. **Volume** — author the full catalog; do not tighten. (~130 entries incl. §12b, §12c.)
2. **File layout** — split by domain; one file per section (see the list in §1).
3. **Expected-fail** — assert the exact `.error` in a `run_cmd do`.
4. **"verify" entries** — probed against HEAD: EC5/EC6 valid `[N]`; RC4/SS4 reject
   (`causalityViolation`); RC5 confirmed silent-wrong → **KG-scanagg**.
5. **Added domains** — diffusion, MoE, SSM, positional, contrastive (§12b), all probed.

Follow-ups for the implementation plan:

- ~~**Surface the RC5/KG-scanagg soundness issue**~~ **RESOLVED 2026-07-08.** The scan step now
  honors the aggregator: `evalStmtSlice`/`evalAssignSeeded` thread the agg's `Combine`, so a
  `maxreduce` step reduces with tropical max. RC5 asserts `[2,6,18]`.
- ~~**KG-2dscan (RC6) remains a silent-wrong trap**~~ **RESOLVED 2026-07-08.** n-D scans are now
  supported: every advancing axis is tracked via positional base recovery plus nested
  evaluation, and a scan step writes only fully-advanced cells (zero-default boundary
  semantics). RC6 (2-D) asserts `[[0,0],[0,1]]`; RC8 (3-D) confirms the fix generalizes.
- **Equation-level summation (§12c callout)** is the other silent-wrong trap: a multi-term RHS
  with heterogeneous contracted axes broadcasts the shorter terms. Whether this is intended
  Datalog semantics or should warn/reject deserves a decision from the language owners; either
  way EC15 pins current behavior. Consider a lint that flags a `+` whose terms have unequal
  free-axis sets.
- **`[R]`/`[F]` needs the exact error constructors.** Before authoring, enumerate the actual
  `CompileError`/`EvalError` variants each reject case throws (we've confirmed
  `causalityViolation`, `predicateAgg`; the parse-failure cases (RJ1, RJ2) fail at
  *elaboration*, so they need a different harness — likely a `#guard`-style negative parse test
  or documentation-only, since a parse error can't be caught by `run_cmd`).
- **Read-vs-LHS look-ahead asymmetry** (`l + 1` in reads vs `l +1` in scan-LHS) is a real
  gotcha worth its own tiny `[S]` test + a note in the DSL docs.

---

## 18. What needs fixing (as of 2026-07-07)

`lake build Tests` exits 0, but that does **not** mean the DSL is correct in all cases. Some
tests deliberately pin *wrong* evaluator output as regression alarms. This section distinguishes
what genuinely needs to be fixed from what is an accepted limitation.

### A. Soundness bugs — silent wrong output, no error raised ⚠

These are the most dangerous issues: the evaluator accepts the program, produces output, but the
output is mathematically wrong. The test suite pins the current wrong value so the test turns red
the moment the bug is fixed — treat them as **failing tests in disguise**.

(Both scan soundness bugs that used to live in this table are now fixed. **KG-scanagg** —
`maxreduce` silently summed inside a scan — fixed 2026-07-08. **KG-2dscan** — a 2-D/nested
recurrence collapsing to a 1-D scan (two advancing axes reduced to one, no error) — also fixed
2026-07-08: n-D scans are now supported via positional base recovery plus nested evaluation, with
zero-default boundary semantics. See §14 / §17. RC5, RC6, and RC8 now assert the correct output.
No soundness bugs are currently open in this table.)

### B. Semantic sharp edge — silent unexpected behavior, arguably by design

| ID | Issue | Live test | Recommended action |
|----|-------|-----------|--------------------|
| **Equation-level summation** | A multi-term RHS sums *every* free variable not on the LHS across the **whole** equation, not per product term — so a `k`-less term like `a[i]` in `Y[i] := a[i] + W[i,k]·v[k]` is broadcast by `|k|`. This silently gave wrong results in CM1 and CM3 first drafts. | `EC15` in `EdgeCaseTest` (pins current behavior) | Decide whether this is intended Datalog semantics or a bug. If intended, add a **lint warning** whenever `+` joins terms with unequal free-axis sets. If unintended, fix the evaluator and update EC15. |

### C. Missing primitives — features the DSL cannot express at all

These are **not** bugs in the evaluator — the evaluator is correct for what it accepts. They are
gaps in the DSL's expressive power. Each is documented in §14 (`KG-*`). No live test can cover
them (there is no expressible program to test). Priority order for practical ML coverage:

| Priority | Gap | Blocks |
|----------|-----|--------|
| High | **KG-log** — no `log`/`exp` | Cross-entropy loss, log-likelihood, GELU, InfoNCE |
| High | **KG-activation** — only `relu`; no `sigmoid`/`tanh` | LSTM/GRU gates, logistic regression, GAT |
| High | **KG-gather** — no data-dependent gather/scatter | Embedding lookup, edge-list scatter-add, top-k routing |
| High | **KG-div** — no per-element division | Degree-normalized GNN (D⁻¹AX), LayerNorm variance |
| Medium | **KG-trig** — no `sin`/`cos` | Sinusoidal PE, RoPE, diffusion timestep embeddings |
| Medium | **KG-l2norm** — only L1 `normalize` | Cosine similarity, RMSNorm, unit-norm embeddings |
| Medium | **KG-sqrt** — no `sqrt` | Euclidean distance, std dev, batch-norm scaling |
| Medium | **KG-min** (partial) — `min` aggregation DONE (`minreduce`, 2026-07-08); still missing additive within-term combine (min-plus) and argmin | Shortest path, Viterbi, DTW (min-plus); k-means argmin |
| Lower | **KG-solve** — no linear solve/inverse | Closed-form regression, PCA, normalizing-flow log-det |
| Lower | **KG-idxvalue** — index arithmetic yields booleans only | ALiBi, relative-position numeric bias |
| Lower | **KG-datamask** — `where` clauses are index predicates only | GAT data-driven masked softmax |
| Lower | **KG-reshape** — no multi-axis-into-one-slot LHS | Reshape/flatten, Kronecker, im2col |
| Lower | **KG-functor** — axes are flat `Fin n` only | Generalized transformers over non-sequence data |
| Lower | **KG-multiout** — outputs must be at the tail statement | Multi-head split, auxiliary losses |
| Lower | **KG-recur** — `recurMorphism`/`scanPre` AST-only | Programmatic recurrences |

### D. How to track progress

- **A (soundness bugs):** `RC5`/KG-scanagg fixed 2026-07-08 (RC5 now asserts correct output).
  `RC6`/`RC8`/KG-2dscan also fixed 2026-07-08 (RC6 now asserts `[[0,0],[0,1]]`; RC8 confirms the
  fix generalizes to 3 axes). No soundness bugs remain open in §A.
- **B (sharp edge):** a language-owner decision on equation-level summation semantics; update
  `EC15` accordingly.
- **C (missing primitives):** each closed gap should have a new `[N]` test added and its `KG-*`
  entry moved from §14 to "Confirmed NON-gaps" at the bottom of §14.
