# Tensor-Logic DSL Test Portfolio

**Status:** Implemented — this is the *catalog* of example programs backing the Lean test
cases under `leanncd/test/Eval/Portfolio/`.

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
- [17. Follow-ups](#17-follow-ups)
- [18. What needs fixing](#18-what-needs-fixing)
- [19. Design notes — deferred fixes (not yet implemented)](#19-design-notes--deferred-fixes-not-yet-implemented)
  - [19.1 KG-gather — data-dependent gather/scatter](#191-kg-gather--data-dependent-gatherscatter)
  - [19.2 General semiring notation — min-plus](#192-general-semiring-notation--min-plus)

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

**Design decisions:**

- **Volume:** the full catalog (~130 entries incl. the advanced/creative domains below).
- **File layout:** split by domain — one test file per section.
- **Expected-fail mechanism:** each `[R]`/`[F]` case asserts the **exact** `.error` inside a
  `run_cmd do` (match on the specific `CompileError`/`EvalError`; `throwError` otherwise). No
  reliance on a `#expect_failure`-style macro (Lean has none).
- **Test framework:** a **hybrid LSpec strategy**. Numeric/property/shape
  `[N]`/`[E]` assertions use `#lspec group "§N — …"` with named `test "ID" (evalEqB/evalPredB/evalShapeB …)` entries (LSpec reports all failures in a group, not just the first). Exact `CompileError`/`EvalError` constructor matches stay as `run_cmd do` — LSpec cannot catch elaboration-layer errors. Parse-level rejects remain documentation comments. `Harness.lean` exposes both layers from one import.
- **Advanced/generative domains** (§12b): diffusion, mixture-of-experts, state-space
  models, positional encodings, contrastive.

All 17 files below are implemented, registered in `lakefile.toml`, and build green via
`lake build Tests`. 105 assertions total; every numeric value was hand-computed independently.
14 runtime test files use `#lspec group` with named `test` entries for grouped, named failure
reporting; `RejectTest` and `KnownGapTest` are unchanged; `RecurrenceTest` is hybrid (RC4 stays
`run_cmd do`, remainder in `#lspec group`).

| File | Section | IDs implemented | IDs skipped ([✔]/[F]/parse-fail) |
|------|---------|------------------|-----------------------------------|
| [`Harness.lean`](../test/Eval/Portfolio/Harness.lean) | — | shared helpers — `run_cmd` layer: `assertEval`, `assertEvalError`, `assertCompileError`, `assertEvalPred`, `assertShape`; LSpec layer: `evalEqB`, `evalPredB`, `evalShapeB`; shared: `rowsSumToOne`, `tl`; `import LSpec`, `open LSpec` | — |
| [`LinAlgTest.lean`](../test/Eval/Portfolio/LinAlgTest.lean) | §2 | LA1, LA3–LA9 | LA2 `[✔]` |
| [`FeedforwardTest.lean`](../test/Eval/Portfolio/FeedforwardTest.lean) | §3 | FF1–FF8 (FF5–FF8: `sigmoid`/`tanh`/`gelu`/`leakyrelu`) | — |
| [`AttentionTest.lean`](../test/Eval/Portfolio/AttentionTest.lean) | §4 | AT1, AT3–AT12 | AT2 `[✔]` |
| [`ConvPoolTest.lean`](../test/Eval/Portfolio/ConvPoolTest.lean) | §5 | CV1, CV2, CV4–CV9 | CV3 `[✔]` |
| [`NormTest.lean`](../test/Eval/Portfolio/NormTest.lean) | §6 | NM1–NM5 | — |
| [`RecurrenceTest.lean`](../test/Eval/Portfolio/RecurrenceTest.lean) | §7 | RC2–RC10 (RC4/RC9 reject via `run_cmd do`; RC5 asserts correct `maxreduce`-in-scan output; RC6 asserts correct 2-D scan output; RC6-compile checks (via `run_cmd do`) that the 2-D scan lowers to a single well-formed rank-2 Br `.scan` step; RC7 `minreduce`-in-scan; RC8 3-D nested scan; RC9 rejects a heterogeneous coupled multi-axis scan (`inconsistentScanAxes`); RC10 multi-axis `maxreduce`) — **hybrid file** | RC1 `[✔]` |
| [`GnnScatterTest.lean`](../test/Eval/Portfolio/GnnScatterTest.lean) | §8, §8b | GN1–GN5 (GN5: friendly `/` operator), SC1–SC8 | GN6 `[F]` (comment) |
| [`RelationalTest.lean`](../test/Eval/Portfolio/RelationalTest.lean) | §9 | RL1–RL4, RL6–RL8 | RL5 `[✔]` |
| [`StatsLossTest.lean`](../test/Eval/Portfolio/StatsLossTest.lean) | §10 | ST1–ST6 (ST6 uses inline `log(...)`) | — |
| [`TropicalTest.lean`](../test/Eval/Portfolio/TropicalTest.lean) | §11 | TR3, TR6, TR7 (`minreduce` added) | TR1/TR2/TR4 `[✔]`; TR5 `[F]` (min-plus half, comment) |
| [`TensorNetTest.lean`](../test/Eval/Portfolio/TensorNetTest.lean) | §12 | TN1–TN4 | — |
| [`GenerativeTest.lean`](../test/Eval/Portfolio/GenerativeTest.lean) | §12b | DF1–DF4, ME1–ME3, SS1–SS3, PE1, CL1–CL4, CL3b (DF4/CL4 use inline `sin(...)`/`log(...)`; CL3/CL3b use `l2normalize`) | PE2/PE3 `[F]` (comment, see note); ME4 `[F]` gather; SS4 → `RejectTest` (comment) |
| [`ClassicalMLTest.lean`](../test/Eval/Portfolio/ClassicalMLTest.lean) | §12c | CM1–CM6, CM1b (`sqrt(...)`) | CM7–CM9 `[F]` (comment) |
| [`RejectTest.lean`](../test/Eval/Portfolio/RejectTest.lean) | §13 | RJ3, RJ4, RJ7, SS4/RC4, UF1–UF3 (`.unaryFn` regression: unknown-tensor, causality-violation, domain-violation) — **`run_cmd do` only** (exact `CompileError`/`EvalError` constructor matches) | RJ1/RJ2/RJ5/RJ8/RJ10 (parse-fail or unconstructible — comment); RJ6/RJ9 (don't reject) |
| [`KnownGapTest.lean`](../test/Eval/Portfolio/KnownGapTest.lean) | §14 | documentation only — cross-references all 19 `KG-*` gaps to their live regression test (if any) | — |
| [`EdgeCaseTest.lean`](../test/Eval/Portfolio/EdgeCaseTest.lean) | §15 | EC1–EC7, EC10–EC15 | EC8 `[✔]`; EC9 parse-fail (comment) |

Run the whole portfolio with `cd leanncd && lake build Tests` (builds the full pre-existing suite
too) or target one file, e.g. `lake build Eval.Portfolio.AttentionTest`. LSpec groups emit named
`✓ ∃: ID` lines per test case; `run_cmd` failures surface as build errors.

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
  see EC1/EC2.

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
| FF5 | `H[i] := sigmoid(W[i,j] · x[j])` | W=I₂, x=`[-2,2]` → `[0.1192029, 0.8807971]` | new pointwise activation (KG-activation); no `.`-marked axis (regression test for the pointwise dispatch) | `[N]` |
| FF6 | `H[i] := tanh(W[i,j] · x[j])` | W=I₂, x=`[-2,2]` → `[-0.9640276, 0.9640276]` | pointwise activation (KG-activation) | `[N]` |
| FF7 | `H[i] := gelu(W[i,j] · x[j])` | W=I₂, x=`[-2,2]` → `[-0.0454023, 1.9545977]` | pointwise activation, tanh-approximation formula (KG-activation; no `Float.erf` available) | `[N]` |
| FF8 | `H[i] := leakyrelu(W[i,j] · x[j])` | W=I₂, x=`[-2,2]` → `[-0.02, 2]` | pointwise activation, fixed `0.01` negative slope (KG-activation) | `[N]` |
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
| AT8 | **generalized cross-attention** `A[q, s.] := softmax(Q[q,h]·K[s,h])` `O[q,g] := A[q,s]·V[s,g]` | Q=I₂, K=`[[1,0],[0,1],[1,1]]`, V=`[[2,0],[0,2],[1,1]]` → A rows sum to 1, O=`[[1.267,0.733],[0.733,1.267]]` | four fully-decoupled axes: query-pos `q`, key/val-pos `s`, score-feature `h` (Q/K only), value-feature `g` (K/V only, `h`≠`g` as roles). Q/K may come from a different source than V | `[N]` `[E]` |
| AT9 | **linear attention** (O(N), kernelized) `M[h,e] := PhiK[s,h]·V[s,e]` `O[q,e] := PhiQ[q,h]·M[h,e]` | Φ=I₂, V=`[[1,2],[3,4]]` → O=`[[1,2],[3,4]]` | contraction **reassociation**: sum keys first (`M[h,e]`) so cost is O(N) not O(N²); relu as the feature map φ | `[N]` `[E]` |
| AT10 | **bilinear attention** `S[q, s.] := softmax(Q[q,a]·W[a,b]·K[s,b])` | I₂ inputs → rows sum to 1 | learned bilinear scoring `qᵀWk` (3-factor contraction inside softmax) | `[N]` |
| AT11 | **grouped/multi-query attention** `A[h, q, s.] := softmax(Q[h,q,d]·K[s,d])` | K has no `h` → shared across heads | K/V **broadcast across heads** (GQA/MQA) | `[N]` `[E]` |
| AT12 | **sparse attention** (Longformer local+global) `A[q, s.] := softmax(where \|q−s\| ≤ 1 ∨ s = 0)(Q[q,d]·K[s,d])` | Q=K=I₃ → `A[0,2]=0` (out of window), global token `s=0` attended from every `q` (`A[2,0]>0`) | first use of `∨` in a mask; local window OR a global token | `[N]` `[E]` |

> Attention variants that hit gaps: **GAT** (graph attention) needs a *data-driven* masked
> softmax (`where edge[i,j]`) — not expressible, masks are index-predicates only (KG-datamask);
> **additive/Bahdanau** needs `tanh`, now available (FF6) — not itself a gap anymore, though GAT's
> masked-softmax half above still is; **ALiBi / relative-position numeric bias**
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
| CV8 | **depthwise / grouped conv** `Y[c,i] := W[c,p]·X[c,i+p]` | W=`[[1,1],[1,-1]]`, X=`[[1,2,3],[4,5,6]]` → `[[3,5],[-1,-1]]` | channel `c` is **free on both** W and X (not contracted); per-channel kernel | `[N]` `[E]` |
| CV9 | **full multi-channel conv** `Y[co,i] := W[co,ci,p]·X[ci,i+p]` | W=`[[[1,0],[0,1]]]` (1×2×2), X=`[[1,2,3],[4,5,6]]` → `[[6,8]]` | contracts **input-channel `ci` AND kernel `p` together**, output-channel `co` free (contrast CV8 depthwise) | `[N]` `[E]` |

## 6. Normalization

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| NM1 | `Y[q, s.] := normalize(A[q,s])` | A=`[[1,3],[2,2]]` → `[[0.25,0.75],[0.5,0.5]]` | L1 normalize over marked axis | `[✔]`-ish |
| NM2 | `Y[q, s.] := softmax(A[q,s])` | A=`[[0,0],[0,ln3]]` → `[[.5,.5],[.25,.75]]` | plain softmax → distribution | `[N]` |
| NM3 | `Y[q, s.] := softmax(A[q,s])` | A=`[[1000,1001]]` → `[[0.2689,0.7311]]` | **numerical stability** (max-subtraction; no overflow) | `[N]` `[E]` |
| NM4 | `Y[q,s.] := normalize(where s ≠ 0)(A[q,s])` | masked entry → 0, rest renormalized | masked L1 normalize | `[N]` `[E]` |
| NM5 | **softmax over a non-last axis** (column softmax / axial attn) `A[s., q] := softmax(Q[q,d]·K[s,d])` | Q=K=I₂ → `[[0.731,0.269],[0.269,0.731]]`, each **column** sums to 1 | reduction axis marked at slot **position 0**, not the last; exercises `axisPos ≠ last` | `[N]` `[E]` |

## 7. Recurrence & scans

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| RC1 | coupled scan `G/H` (relu step) | all-ones → G=`[1,3,6]`, H=`[2,3,6]` | coupled scan; `axis l : ℕ = 3` pin | `[✔]` |
| RC2 | simple RNN `S[j,0]:=X0[j]` `S[j,l +1] := relu(S[j,l]·W[j,k])` | 1-feature, W=1 | single scan, self-recurrence | `[N]` |
| RC3 | prefix-sum via mask `C[i] := X[j] · [j ≤ i]` | X=`[1,2,3]`, `axis i,j = 3` → `[1,3,6]` | **cumulative sum without a scan** (triangular Iverson) | `[N]` `[E]` |
| RC4 | `S[j, l +1] := S[j, l] + X[j, l + 1]` (scan step reads external at advancing index) | `X=[[1,2,3]]` | compile → `causalityViolation "S"` | `[R]` |
| RC5 | `M[j, l +1] := maxreduce(M[j,l] · W[j,k])` (`maxreduce` in a scan step) | `X=[2], W=[[1,3]]` → `[2,6,18]` (max-step `M×3`) | the scan step honors the `maxreduce` agg (tropical max) | `[N]` |
| RC6 | **2D / nested recurrence** (grid-DP / PixelRNN) `G[r +1, c +1] := G[r,c] + A[r,c]` | base G=0, A=ones(2×2) → `[[0,0],[0,1]]` (only the fully-advanced cell `G[1,1]` is written; boundary cells `r=0`/`c=0` keep the zero-default) | both advancing iteration axes are tracked through the scan | `[N]` |
| RC7 | `M[j, l +1] := minreduce(M[j,l] · W[j,k])` (`minreduce` in a scan step) | `X=[2], W=[[2,3]]` → `[2,4,8]` (min-step `M×2`) | **min-agg in a scan** (KG-min; mirrors RC5) — the scan step honors `minreduce` via the KG-scanagg plumbing | `[N]` |
| RC8 | **3-D nested scan** `G[a +1, b +1, d +1] := G[a,b,d] + T[a,b,d]` | axes a,b,d size 2; base S=0, T=ones(2×2×2) → 2×2×2 tensor with a single `1` at `[1,1,1]`, all boundary cells `0` | **generality of n-D support** (KG-2dscan): confirms multi-axis scans generalize past the 2-D case (RC6) to 3 axes, with the same zero-default boundary semantics | `[N]` |
| RC9 | **heterogeneous coupled multi-axis scan** — `H[j, c +1]` (advances over `{c}`) coupled via shared `c` with `G[r +1, c +1]` (over `{r,c}`) | compile → `inconsistentScanAxes "H"` | **FAIL-LOUD guard (design §5)**: coupled state tensors advancing over *different* axis sets are rejected, not silently mis-addressed. Homogeneous couplings (RC1, both over `l`) and single-tensor multi-axis scans (RC6/RC8) are unaffected | `[R]` |
| RC10 | **multi-axis scan with a tropical step** `G[r +1, c +1] := maxreduce(G[r,c] · W[r,c,k])` | Z=`[2,5]` (c=0 column base), W[0,0,:]=`[1,3]` → `[[2,0],[5,6]]` (`G[1,1]=max(2·1,2·3)=6`, not sum `8`) | **KG-scanagg × KG-2dscan interaction**: `maxreduce` agg inside a 2-D scan (agg-in-scan + multi-axis together); tropical max honored under zero-default boundaries | `[N]` `[E]` |

> **RC4**: a scan step reading an external at the advancing index `l + 1` is rejected with
> `CompileError.causalityViolation` (the compiler treats any next-index read as a future
> dependency, even for non-state externals).
> **RC5**: `maxreduce` inside a scan step reduces with tropical max `(×, max, −∞)`, via the
> aggregator's `Combine` threaded through `evalStmtSlice`/`evalAssignSeeded` (`Eval/Scan.lean`,
> `Eval/Contract.lean`); asserts `[2,6,18]`.
> **RC6/RC8**: multi-axis (n-D) recurrences evaluate correctly — positional base recovery plus
> nested evaluation track every advancing axis, and a scan step writes only fully-advanced cells
> (boundary cells keep their zero-allocated/base value — **zero-default** boundary semantics).
> RC6 (2-D) asserts `[[0,0],[0,1]]`; RC8 (3-D) confirms this generalizes to 3 axes. `RC6-compile`
> additionally checks at the compile level that the 2-D scan lowers to a single well-formed Br
> step with `.op = .scan` and `degree.length == 2`, i.e. both iteration axes survive lowering.
> **RC9/RC10:** RC9 pins the fail-loud guard (design §5) — a *heterogeneous* coupled multi-axis
> scan (coupled tensors advancing over different axis sets) is rejected with
> `inconsistentScanAxes` rather than silently mis-addressed. RC10 exercises a `maxreduce` step
> inside a 2-D scan, confirming tropical aggregation and multi-axis nesting compose (tropical max
> honored, zero-default boundaries).

## 8. Graph / message passing (GNN)

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| GN1 | `H[i,f] := A[i,j] · X[j,f]` | A=adjacency `[[0,1],[1,0]]`, X=`[[1,2],[3,4]]` → `[[3,4],[1,2]]` | dense message passing (A·X) | `[N]` |
| GN2 | `H[i,f] := edge[i,j] · X[j,f]` (`predicate edge`) | edge booleanizes | message passing over a **predicate** adjacency | `[N]` |
| GN3 | `deg[i] := edge[i,j]` | edge=`[[0,1,1],[1,0,0]]` → `[2,1]` | node degree = contract over neighbors | `[N]` |
| GN4 | `H[i,f] := maxreduce(edge[i,j] · X[j,f])` | max-aggregation GNN (GraphSAGE-max) | `maxreduce` neighbor aggregation | `[N]` `[E]` |
| GN5 | degree-normalized `Ĥ[i,f] := H[i,f] / deg[i]` (chained on `H[i,f] := A[i,j]·X[j,f]`, `deg[i] := A[i,j]`) | A=`[[0,1,1],[1,0,0]]`, X=`[[1,2],[3,4],[5,6]]` → H=`[[8,10],[1,2]]`, deg=`[2,1]` → Ĥ=`[[4,5],[1,2]]` | friendly infix `/` operator (KG-div), per-node division by degree | `[N]` |
| GN6 | edge-list scatter-add `Msg[dst[e]] += X[src[e]]` | index tensors `src/dst` | data-dependent gather/scatter | `[F]` (KG-gather) |

## 8b. Consuming scatter outputs (upsampling decoders / GNN readback)

Reading the output of a scatter statement in *later* statements — the decoder/GNN pattern. This
block doubles as a regression suite. All eight consumers are numerically correct. The shared
base is a 2× upsample `Out[2*i,2*j] := X[i,j]` with X=`[[1,2],[3,4]]`
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
| RL8 | **negated mask** (exclude-self) `M[i,j] := A[i,j] · [¬(i = j)]` | A=`[[1,2],[3,4]]` → `[[0,2],[3,0]]` | first use of `¬`; zeroes the diagonal | `[N]` `[E]` |

## 10. Losses, reductions & statistics

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| ST1 | `sse[] := r[i] · r[i]` | r=`[1,-2,2]` → `9` | sum-of-squares (given residual `r`) | `[N]` |
| ST2 | `C[i,j] := X[k,i] · X[k,j]` | X=`[[1,2],[3,4]]` (2 samples×2 feat) → `[[10,14],[14,20]]` | uncentered covariance / scatter matrix | `[N]` |
| ST3 | `m[j] := X[k,j] · invn[]` | X=`[[2,4],[6,8]]`, invn=`[0.5]` → `[4,6]` | mean over samples via rank-0 `1/n` | `[N]` `[E]` |
| ST4 | `p[i] := joint[i,j]` | marginalize out `j` | probabilistic marginalization = sum | `[N]` |
| ST5 | residual via `−1` scalar `r[i] := Yhat[i] + m1[]·Y[i]` (`m1=−1`) | Yhat=`[5,3]`, Y=`[2,1]` → `[3,2]` | no `−` operator, but subtraction works via a rank-0 `−1` (⇒ MSE `sse[]:=r[i]·r[i]` is expressible) | `[N]` `[E]` |
| ST6 | cross-entropy `L[] := m1[]·Y[i]·log(P[i])` (`log(...)` inline on a factor, ST5's `−1` trick for the minus) | Y=`[1,0]`, P=`[0.5,0.5]` → `ln 2 ≈ 0.6931472` | inline unary transcendental function on a single factor (`Factor.unaryFn`) | `[N]` |

## 11. Tropical / max-times semiring

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| TR1 | `C[i] := maxreduce(A[i,k])` | A=`[[3,1,4],[1,5,9]]` → `[4,9]` | basic max-reduction | `[✔]` |
| TR2 | `C[i] := maxreduce(A[i,k] · B[k])` | max of products | max-times combine | `[✔]` |
| TR3 | `R[i,k] := maxreduce(P[i,j] · P[j,k])` | P=reliability `[[0,0.9],[0.8,0]]` → most-reliable 2-hop | max-times "best path" | `[N]` `[E]` |
| TR4 | all-negative input max | A=`[[-2,-5,-1]]` → `[-1]` | `-∞` unit doesn't pollute | `[✔]` |
| TR6 | `C[i] := minreduce(A[i,k])` | A=`[[3,1,4],[1,5,9]]` → `[1,1]` | **basic min-reduction** (min analog of TR1); tropical min `(×, min, +∞)` | `[N]` |
| TR7 | `C[i] := minreduce(A[i,k] · B[k])` | A=`[[3,1,4],[2,2,2]]`, B=`[2,5,1]` → `[4,2]` | **min-times combine** (min analog of TR2) | `[N]` |
| TR5 | min-*plus* shortest path (Viterbi/Bellman step) | — | `minreduce` supplies the `min` agg, but min-plus still needs `+` as the within-term combine ⇒ **partial gap** | `[F]` (KG-min, min-plus half) |

## 12. Tensor networks / decomposition

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| TN1 | `Y[a,d] := G1[a,b] · G2[b,c] · G3[c,d]` | 3 identity cores → identity | tensor-train / chain contraction | `[N]` |
| TN2 | `T[i,j,k] := A[i,r] · B[j,r] · C[k,r]` | rank-1 factors → outer product | **CP reconstruction** (shared contracted `r`) | `[N]` `[E]` |
| TN3 | `s[] := A[i,j] · B[i,j]` | A=`[[1,2],[3,4]]`, B=I₂ → `5` | Frobenius inner product (full contract, 2 axes) | `[N]` |
| TN4 | **third-order moment** (method of moments) `M[i,j,k] := X[t,i]·X[t,j]·X[t,k]` | X=`[[1,0],[0,1]]` (t=2) → diagonal 3-tensor `M[0,0,0]=M[1,1,1]=1` | **3× aliasing** of one tensor (Gram is 2×) with a shared contracted `t`, rank-3 output | `[N]` `[E]` |

## 12b. Advanced & generative domains

Recurring finding: the DSL's scalar/transcendental ops (subtraction workaround, inline
`log`/`sin`/`cos`/`sqrt`/`exp` via `Factor.unaryFn`) determine which parts are expressible.
`log`/`sin` are no longer missing (see DF4, CL4) — what remains a gap is turning *index*
arithmetic into a numeric value (KG-idxvalue): `ω^i` itself must be supplied as a precomputed
input tensor, not computed from the loop index `i` inside the DSL.

### Diffusion

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| DF1 | forward process `Xt[i] := alpha[]·x0[i] + beta[]·eps[i]` | α=`0.6`, β=`0.8`, x0=`[1,0]`, ε=`[0,1]` → `[0.6,0.8]` | rank-0 scalar scaling + 2-term sum | `[N]` |
| DF2 | DDPM posterior mean `Xp[i] := c1[]·Xt[i] + c2[]·e[i]` (`c2<0`) | another scaled combo | reverse-step linear combination (uses `−` scalar) | `[N]` |
| DF3 | denoising loss `sse[] := r[i]·r[i]` with `r[i]:=eps[i]+m1[]·ehat[i]` | ε,ε̂ small → hand-check | **expressible** via ST5 subtraction trick + sum-of-squares | `[N]` |
| DF4 | sinusoidal timestep embedding `Arg[i]:=t[]·omega_pow[i]`; `te[2*i]:=sin(Arg[i])` | t=`1`, omega_pow=`[1,0.1]` (ω⁰,ω¹, precomputed) → te=`[sin 1, 0, sin 0.1, 0]` ≈ `[0.8414710,0,0.0998334,0]` | inline `sin(...)` on a materialized read, feeding an affine (EC8-style) scatter; `ω^i` itself supplied precomputed (KG-idxvalue still applies to the exponentiation) | `[N]` `[E]` |

### Mixture of Experts

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| ME1 | gating `g[t, e.] := softmax(X[t,d]·Wg[d,e])` | rows sum to 1 | softmax router over experts | `[N]` |
| ME2 | batched expert MLP `Y[t,e,f] := relu(We[e,f,d]·X[t,d])` | — | rank-3 expert weight, expert axis `e` free | `[N]` |
| ME3 | combine `Out[t,f] := g[t,e]·Y[t,e,f]` (chained on ME1/ME2) | evals end-to-end (shape `[1,1]`) | gated weighted sum over experts | `[N]` |
| ME4 | top-k routing (keep k highest-gated experts) | — | **no argmax / top-k gather** | `[F]` (KG-gather) |

### State-space / sequence models (SSM, S4/Mamba-style)

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| SS1 | linear recurrence `h[j,l +1] := A[j,k]·h[k,l] + B[j]·u[l]` | A=`[[1]]`, B=`[1]`, u=`[1,1,1]` → h=`[1,2,3]` | scan reading input at the **current** index `u[l]` | `[N]` |
| SS2 | output map `y[j,l] := C[j,k]·h[k,l]` (chained on SS1) | — | per-step output projection off the scan state | `[N]` |
| SS3 | diagonal SSM `h[j,l +1] := a[j]·h[j,l] + B[j]·u[l]` | diagonal `A` (Mamba-ish) | elementwise (diagonal) state transition | `[N]` `[E]` |
| SS4 | input at advancing index `… + B[j]·u[l + 1]` | — | **rejected** — `causalityViolation` (cf. RC4); convention matters | `[R]` |
| UF5 | same `y[j,l] := C[j,k]·h[k,l]` written INSIDE the SS1 scan block instead | — | **rejected** — `scanProjectionUnsupported` (§14 KG-scanprojection); write it standalone (SS2) instead | `[R]` |

SS2 must be written standalone, *after* the scan — writing the identical statement inside the
scan block is a compile-time error (`scanProjectionUnsupported`, `UF5`), not a silent no-op. The
scan classifies any statement that reads scan state as a per-step intermediate; if that
statement's own LHS also references the iteration axis, whether the user wants it tracked across
every step (unsupported — not materialized) or recomputed as same-step scratch is ambiguous, so
it's rejected rather than guessed at. See §14 KG-scanprojection.

### Positional encodings

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| PE1 | learned additive `Out[p,d] := X[p,d] + PE[p,d]` | X=`[[1,2],[3,4]]`, PE=`[[10,20],[30,40]]` → `[[11,22],[33,44]]` | elementwise add of same-shape reads | `[N]` |
| PE2 | sinusoidal `pe[p,2i] := sin(p·ω^i)`, `pe[p,2i+1]:=cos(…)` | — | likely closeable by the same DF4 pattern (materialize the argument, wrap in `sin`/`cos`) — **not yet verified/authored** | `[F]` |
| PE3 | RoPE rotation of `(q_2i, q_2i+1)` by `θ_p` | — | needs `sin`/`cos` (now available) **+** paired rotation combining two scattered slots — the rotation structure, not the trig, is the remaining gap | `[F]` |

### Contrastive / metric learning

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| CL1 | similarity matrix `S[i,j] := Z1[i,d]·Z2[j,d]` | Z1=Z2=I₂ → `[[1,0],[0,1]]` | pairwise dot-product similarity (matmul) | `[N]` |
| CL2 | `P[i, j.] := softmax(S[i,j])` (chained) | rows sum to 1 | InfoNCE softmax over the negatives | `[N]` |
| CL3 | cosine similarity `Z1n[i,d.] := l2normalize(Z1[i,d])`; `Z2n[j,d.] := l2normalize(Z2[j,d])`; `S[i,j] := Z1n[i,d]·Z2n[j,d]` | Z1=`[[3,4]]` (‖·‖₂=5 → `[0.6,0.8]`), Z2=`[[1,0]]` (unit) → `S=[[0.6]]` | new `l2normalize` `Nonlin` (KG-l2norm), row-wise `x/‖x‖₂` | `[N]` |
| CL3b | all-zero row `Y[i,d.] := l2normalize(X[i,d])` | X=`[[0,0]]` → `Y=[[0,0]]` | `‖x‖₂=0` normalizes silently to zero, matching `normalize`/`softmax`'s convention (cf. SC8) | `[N]` `[E]` |
| CL4 | InfoNCE loss `L[] := m1[]·log(P[i, i])` (diagonal read wrapped inline in `log`) | P=`[[0.5,0.5],[0.25,0.75]]` → `−(ln 0.5 + ln 0.75) ≈ 0.9808293` | inline `log(...)` on a diagonal (repeated-axis) read | `[N]` |

## 12c. Classical ML, probabilistic & RL

Less-obvious use cases. Several decompose a computation into multiple statements (each
contraction materialized into its own intermediate) purely for clarity — this matches the
natural structure of the underlying method, not a DSL requirement: a single RHS sum can freely
mix terms with different contracted axes (see the callout after this section).

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| CM1 | **pairwise squared distance** (k-NN / RBF kernel numerator): `sq[i]:=X[i,d]·X[i,d]`; `cross[i,j]:=X[i,d]·X[j,d]`; `D[i,j] := sq[i]·one[j] + one[i]·sq[j] + m2[]·cross[i,j]` | X=`[[0,0],[3,4]]`, one=`[1,1]`, m2=`−2` → `[[0,25],[25,0]]` | broadcast (`sq[i]·one[j]`) + subtraction scalar + materialized cross term | `[N]` `[E]` |
| CM1b | **true Euclidean distance** = `sqrt(squared distance)`, chained on CM1: `Dist[i,j] := sqrt(D[i,j])` | D=`[[0,25],[25,0]]` (3-4-5 triangle) → `Dist=[[0,5],[5,0]]` exactly | inline `sqrt(...)` (KG-sqrt) on a materialized read | `[N]` |
| CM2 | **factorization machine** 2nd-order: `sqsum[f]:=V[i,f]·x[i]·V[j,f]·x[j]`; `sumsq[f]:=V[i,f]·x[i]·V[i,f]·x[i]`; `fm[f]:=hp[]·sqsum[f]+hm[]·sumsq[f]` | V=`[[1],[2]]`, x=`[1,1]`, hp=`0.5`, hm=`−0.5` → `[2]` | the `½((Σ)²−Σ()²)` identity; distinguishes **reused dummy `i`** (Σ()²) from **fresh dummy `j`** (Σ)² | `[N]` `[E]` |
| CM3 | **value iteration / Bellman backup**: `EV[s,a]:=gamma[]·P[s,a,s2]·V[s2]`; `Q[s,a]:=R[s,a]+EV[s,a]`; `Vn[s]:=maxreduce(Q[s,a])` | R=I₂, γ=`0.9`, P=`½` unif, V=`[1,1]` → `[1.9,1.9]` | `maxreduce` over actions + discounted expectation (rank-0 γ). *Iterated* VI needs `maxreduce`-in-scan (see RC5) | `[N]` |
| CM4 | **power iteration / Markov / HMM-forward (sum-product)**: `p[j,0]:=p0[j]`; `p[j,l +1]:=p[i,l]·M[i,j]` | p0=`[1,0]`, M=swap → p=`[[1,0,1],[0,1,0]]` | **contraction inside a scan** works correctly | `[N]` |
| CM5 | **matrix factorization** `Rhat[u,i] := P[u,f]·Q[i,f]` | small | recommender dot-product model | `[N]` |
| CM6 | **VAE reparameterization** `z[i] := mu[i] + sigma[i]·eps[i]` | Hadamard + add | elementwise `μ+σ⊙ε`; the KL term needs `log` (KG-log) | `[N]` `[E]` |
| CM7 | k-means assignment / nearest-centroid (argmin over centroids) | — | **no `argmin`** (only `max` value, no index) | `[F]` (KG-min) |
| CM8 | closed-form linear regression `β = (XᵀX)⁻¹Xᵀy` | XᵀX, Xᵀy expressible; the solve is not | **no linear solve / inverse** | `[F]` (KG-solve) |
| CM9a | logistic regression `p[i] := sigmoid(W[i,j]·x[j] + b[i])` (`sigmoid` wraps a full sum, same as `relu` already does) | W=`[[2]]`, x=`[1]`, b=`[-1]` → pre=`1` → `p = sigmoid(1) ≈ 0.7310586` | inline whole-statement `sigmoid` (KG-activation) | `[N]` |
| CM9b | full GRU-LSTM gate composite (multi-gate cell-state combination) | — | `sigmoid`/`tanh` are now available (see CM9a, FF5–FF8); the multi-gate composite isn't authored | `[F]` |

> **Per-term contraction scoping.** A `+`-joined RHS contracts each product term
> **independently**, over only the axes that term itself mentions — not the union of axes
> across the whole equation. So in `Y[i] := a[i] + W[i,k]·v[k]`, the `k`-less term `a[i]` is
> added in exactly once; `k` is contracted only within the second term, where it actually
> appears. Two terms using different dummy names (e.g. `A[i,k]·B[k,j] + C[i,l]·D[l,j]`, EC13)
> each contract over their own axis independently, even when `k` and `l` have different sizes.
> See `termAxisUIDs` in `Eval/Contract.lean`, and the regression tests EC13/EC15.

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
| RJ6 | scan step with no input and no `axis l:ℕ=N` pin | does not reject — evaluates to a 0-step/defaulted result, no error | — |
| RJ7 | `s[] := A[i] · B[i]` with A length 3, B length 2 | eval error — affine size system inconsistent (`0 = -1`) | `[R]` |
| RJ8 | `A[q, d.] := softmax(...)` where `d` is contracted (not an output axis) | not constructible via surface syntax (marking always places the slot on the LHS) | `[R]` (not automated) |
| RJ9 | `Y[i] := X[i - 5]` short input | does not reject — evaluates, zero-padded; no Issue-D error at this size | — |
| RJ10 | scatter whose output axis is unsized | `scatterOutShape` fail-loud on unsized axis | `[R]` (not readily constructible) |

> **Live in `RejectTest.lean`:** RJ3 (`predicateAgg`, compile), RJ4 (eval), RJ7 (eval),
> and SS4/RC4 (`causalityViolation`, compile). RJ1/RJ2 fail at **parse time** — a hard parse
> error fails the build and `#guard_msgs` does not validate parse errors, so they are kept as
> documentation comments in the reject file rather than live tests. RJ6/RJ9 do not reject.
> RJ5/RJ8/RJ10 aren't readily constructible via surface syntax.

## 14. Adversarial — known gaps / expected-fail

Documented constructs that a user might reasonably expect but that do **not** work today.
Author as expected-fail so that (a) the gap is visible and (b) the test flips green the day the
gap is closed — a built-in regression alarm. Cross-referenced above by `KG-*`.

| ID | Gap | Where it bites | Tag |
|----|-----|----------------|-----|
| KG-sub | no `−` operator in the RHS sum — **but** subtraction is achievable via a rank-0 `−1` scalar (`a[i] + neg[]·b[i]` — see ST5). Only a *soft* gap: the ergonomic form is missing, the capability is not | residuals, MSE, centering | `[F]` (soft) |
| KG-functor | axes are flat `Fin n` only — no **structured/applicative-functor index spaces** (trees, functions, nested containers) | generalized transformers over non-sequence data (parsing, image→function); see AT8 & `NaperianTypingIntegrationPlan.md` | `[F]` |
| KG-reshape | LHS index slots are **single-axis affine only** (`2*i`, `i+num`); no multi-axis-into-one-slot (`i+j`, `2*i+k`) — parse-fail | reshape/flatten, Kronecker product, im2col, overlapping-accumulate scatter (also unreachable via surface syntax) | `[F]` |
| KG-datamask | `where` masks are **index predicates only** — cannot read a data tensor (`where edge[i,j] > 0`) — parse-fail | GAT / any graph-structured masked softmax over a runtime adjacency | `[F]` |
| KG-solve | no linear **solve / inverse / eigendecomp / determinant** (only contraction) | closed-form regression, PCA, spectral GCN, normalizing-flow log-det | `[F]` |
| KG-idxvalue | index arithmetic yields **booleans only** (via Iverson) — never a numeric tensor value; blocks computing `ω^i`/index-derived biases inside the DSL (they must be supplied as precomputed input tensors, e.g. DF4) | ALiBi, relative-position numeric bias, distance-decay weights | `[F]` |
| KG-min | min aggregation (`minreduce`, tropical min-times `(×, min, +∞)` — see TR6/TR7, RC7) is supported. Still open: **additive within-term combine** (min-*plus* semiring, `+` instead of `×`) and **argmin** (index output, cf. KG-idxvalue) | shortest path, Viterbi, DTW (need min-plus); k-means (needs argmin) | `[F]` (partial) |
| KG-gather | data-dependent gather/scatter with an **index tensor** (indices must be affine) — see [§19.1](#191-kg-gather--data-dependent-gatherscatter) for a fix design & proof-burden analysis | embedding lookup by ids, edge-list scatter-add, top-k | `[F]` |
| KG-scale | free scalar literal on the RHS (e.g. `· 0.5`) | 1/√d attention scaling (workaround: rank-0 tensor, see AT6) | `[F]`? |
| KG-recur | `recurMorphism` / `scanPre` escape hatch (AST-only) | programmatic recurrences | `[F]` |
| KG-scanprojection | a per-step scan read-out (e.g. `y[j,l] := C[j,k]·h[k,l]`) written **inside** the same scan block as its state `h` — rejected at compile time (`scanProjectionUnsupported`), not silently dropped. Not a missing capability: the identical statement written standalone *after* the scan (reading the materialized state) already works — see SS2 (§12b) | any attempt to read out per-step scan state inline instead of as a separate post-scan statement | `[R]` |

> **Design direction — general semiring notation (planned, not yet implemented).** The remaining
> tropical/semiring gaps (KG-min's min-*plus* half, and future max-plus / log-sum-exp) should be
> closed by adopting a **general semiring selector** — an aggregate `⊕` and a within-term
> combine `⊗` named explicitly, e.g. `D[i,k] := agg(min, +)(D[i,j] · W[j,k])` — rather than one
> bespoke `AggOp` keyword per combination. See [§19.2](#192-general-semiring-notation--min-plus)
> for the full pipeline-stage fix design and proof-burden analysis.

## 15. Adversarial — tricky-but-valid edge cases (should pass)

| ID | Prog | Data → Expected | Probes | Tag |
|----|------|-----------------|--------|-----|
| EC1 | `Y[i] := X[i - 1]` | X=`[10,20,30]` → `[0,10,20,30]` (shape **4**, not 3) | boundary read → **zero-pad**; the shape solver's "maximal padded extent" gives length `n+shift` | `[N]` `[E]` |
| EC2 | `Y[i] := X[i - 3]` | X=`[10,20,30]` → `[0,0,0,10,20,30]` (shape **6**) | multi-step look-back + padding; same `n+shift` rule as EC1 | `[N]` `[E]` |
| EC3 | `Y[i,j] := A[i,k] · B[k,j]` with `k` size 1 | A 2×1, B 1×2 | **size-1** contracted axis (rank-1 outer via matmul) | `[N]` `[E]` |
| EC4 | `Y[i] := A[i,j] · B[j]` with a **size-1** free axis `i` | 1×3, 3 → 1 | singleton free dim | `[N]` `[E]` |
| EC5 | `d[i] := M[i,i]` | M=`[[1,2],[3,4]]` → `[1,4]` | **diagonal read** (repeated axis on RHS) — valid | `[N]` `[E]` |
| EC6 | `t[] := M[i,i]` | M above → `5` | trace (diagonal read + full contraction) — valid | `[N]` `[E]` |
| EC7 | `D[i,i] := v[i]` | v=`[5,6]` → `[[5,0],[0,6]]` | **diagonal write** (scatter reclassification) | `[N]` `[E]` |
| EC8 | `Out[2*i,2*j] := X[i,j]` | 2×2 → 4×4 placement | upsample scatter | `[✔]` |
| EC9 | `Out[i + j] := X[i,j]` | — | does not parse — LHS slots are single-axis affine only (`i + num`, `2*i`); `i + j` (two axes → one slot) is rejected. See KG-reshape (§14); overlapping-accumulate scatter is unreachable via surface syntax | `[F]` |
| EC10 | `Y[i,j,k,l] := A[i,j,m] · B[m,k,l]` | small | **high-rank** (4 free + 1 contracted) tensor contraction | `[N]` `[E]` |
| EC11 | `Z := A · B · C · D` (n-ary product, all scalar reads) | → product | n-ary `·` associativity | `[N]` `[E]` |
| EC12 | `Y[i] := X[2*i + 3*j - 1]` chained affine | multi-term affine read index; both `i` and `j` are unconstrained by any input shape (the read is the only occurrence of either), so this needs `axis i : ℕ = 2, j : ℕ = 2` pins to evaluate; `j` is RHS-only so it's summed | general integer-affine reindex | `[N]` |
| EC13 | `Y[i,j] := A[i,k]·B[k,j] + C[i,l]·D[l,j]` (**no parens** — `(…)` around a sub-product is **not** surface syntax; parenthesizing a `tl_prod_term` fails to parse) | A=`[[1,2],[3,4]]` (2×2), B=I₂ (2×2) ⇒ A·B=A; C=ones(2,3), D=ones(3,2) ⇒ C·D=`[[3,3],[3,3]]` (`l` size 3) → Y=`[[4,5],[6,7]]` | `·` binds tighter than `+`; two contractions, each over its **own** dummy axis (`k` size 2, `l` size 3 — different sizes, confirming the terms don't share a joint contraction range) | `[N]` `[E]` |
| EC14 | `s[] := A[i,j,k,l,m]` | rank-5 full contraction | many contracted axes at once | `[N]` `[E]` |
| EC15 | **per-term contraction scoping** `Y[i] := u[]·a[i] + W[i,k]·v[k]` | `k` appears only in the second term, so only that term is contracted over `k`; the `k`-less first term `u·a[i]` is added in exactly once | regression test for per-term (not whole-equation) contraction scoping. **Naming hazard:** a tensor/predicate named exactly `c` immediately followed by `[` (e.g. `c[k]`) fails to parse — not a DSL bug, but a **global token collision**: Mathlib's `Equiv.Perm` cycle notation (`Mathlib.GroupTheory.Perm.Cycle.Concrete`) registers `c[` as a single reserved token (`c[0,1,2]` = a permutation cycle), and Lean's tokenizer binds it greedily ahead of `ident` + `[`. Scoped to exactly the identifier `c` (`cc`, `c2`, uppercase `C` all parse fine). Avoid naming any tensor/predicate `c` in DSL programs | `[N]` `[E]` |

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
| maxreduce / minreduce | max-pool | CV5, GN4, TR3 (max); TR6, TR7, RC7 (min); RC5 (max-in-1-D-scan), RC10 (max-in-multi-axis-scan) |
| Iverson / predicates | band, masked-agg | RL1–RL7, RC3 |
| Affine reads (conv/dilation/look-back) | strided conv, look-back | CV1,CV2,CV4, EC1,EC2,EC12 |
| Scatter (affine/diagonal write) | upsample | EC7, EC9 |
| Scans / recurrence | coupled scan | RC2, RC4, RC5 (maxreduce-in-scan) |
| `linear` + `bias` | parse only | FF1, FF4 |
| Rank-0 tensors / scalar scale | — | AT6, ST3, DF1, DF2, ST5 |
| Size inference from decls | — | RL1 |
| Scan reading input at index `l` | — | SS1, SS3, CM4 (SS4 = reject at `l+1`) |
| Contraction inside a scan | coupled scan | CM4 (sum), RC5 (`maxreduce`); RC6 (2-D) and RC8 (3-D) (n-D scans) |
| Attention variants | causal attn | AT8 (generalized), AT9 (linear), AT10 (bilinear), AT11 (GQA), AT12 (sparse ∨) |
| Consuming scatter outputs (B3) | — | §8b SC1–SC8 (reduce/pointwise/relu/matmul/conv/scatter-of-scatter/diag/softmax) |
| Softmax/normalize over non-last axis | — | NM5 |
| `∨` / `¬` in masks | — | AT12 (∨), RL8 (¬) |
| Full vs depthwise conv | — | CV9 (contract channel+kernel) vs CV8 (channel free) |
| 3× aliasing / high-order moment | — | TN4 |
| Broadcast + subtraction-scalar patterns | — | CM1, CM2, CM3, DF*, ST5 |
| Per-term contraction scoping | — | EC13 (independent per-term dummy axes of different sizes), EC15 (`k`-less term unaffected by `\|k\|`) |
| Inline unary transcendental functions (`log`/`exp`/`sin`/`cos`/`sqrt`) | — | ST6 (`log`), CL4 (`log` on a diagonal read), DF4 (`sin`), CM1b (`sqrt`) — `Factor.unaryFn`, fails loud on domain violation |
| Pointwise activations (`sigmoid`/`tanh`/`gelu`/`leakyrelu`) | relu | FF5–FF8, CM9a (`sigmoid` on a full sum, logistic regression) — new `Nonlin` variants |
| Friendly division (`/`) | — | GN5 (degree-normalized GNN) — desugars to `Factor.unaryFn .recip`, fails loud on divide-by-zero |
| L2-normalize (`l2normalize`) | normalize (L1) | CL3 (cosine similarity), CL3b (all-zero row → silent zero) — new `Nonlin` variant, same `perRow` mechanism as `normalize`/`softmax` |
| Diffusion / MoE / SSM / PE / contrastive | — | §12b (DF*, ME*, SS*, PE*, CL*) |
| Classical ML / probabilistic / RL | — | §12c (CM1–CM9b, CM1b) |
| **Reject** paths | (a few) | RJ1–RJ10, SS4, UF1–UF5 |
| **Known gaps** | — | KG-* (§14): sub(soft), min, gather, recur, functor, reshape, datamask, solve, idxvalue |
| Multi-output programs (schedule bug fix) | — | `EvalExamplesTest` (Total/Peak, KG-multiout) — `schedule` no longer eliminates produced-but-unread top-level statements |
| In-scan per-step projection (KG-scanprojection) | — | UF5 — rejected at compile time (`scanProjectionUnsupported`); not a missing primitive, see SS2 for the standalone workaround |

## 17. Follow-ups

- **`[R]`/`[F]` needs the exact error constructors.** Confirmed so far: `causalityViolation`,
  `predicateAgg`. The parse-failure cases (RJ1, RJ2) fail at *elaboration*, so they need a
  different harness — likely a `#guard`-style negative parse test or documentation-only, since a
  parse error can't be caught by `run_cmd`.
- **Read-vs-LHS look-ahead asymmetry** (`l + 1` in reads vs `l +1` in scan-LHS) is a real
  gotcha worth its own tiny `[S]` test + a note in the DSL docs.

---

## 18. What needs fixing

`lake build Tests` exits 0, but that does **not** mean the DSL is correct in all cases. Some
tests deliberately pin *wrong* evaluator output as regression alarms. This section distinguishes
what genuinely needs to be fixed from what is an accepted limitation.

### A. Soundness bugs — silent wrong output, no error raised ⚠

These are the most dangerous issues: the evaluator accepts the program, produces output, but the
output is mathematically wrong. The test suite pins the current wrong value so the test turns red
the moment the bug is fixed — treat them as **failing tests in disguise**.

No soundness bugs are currently open in this table.

### B. Semantic sharp edge — silent unexpected behavior, arguably by design

No semantic sharp edges are currently open in this table. Contraction is scoped per product
term — see the §12c callout and EC13/EC15.

### C. Missing primitives — features the DSL cannot express at all

These are **not** bugs in the evaluator — the evaluator is correct for what it accepts. They are
gaps in the DSL's expressive power. Each is documented in §14 (`KG-*`). No live test can cover
them (there is no expressible program to test). Priority order for practical ML coverage:

| Priority | Gap | Blocks |
|----------|-----|--------|
| High | **KG-gather** — no data-dependent gather/scatter | Embedding lookup, edge-list scatter-add, top-k routing |
| Medium | **KG-min** (partial) — `min` aggregation (`minreduce`) is supported; still missing additive within-term combine (min-plus) and argmin | Shortest path, Viterbi, DTW (min-plus); k-means argmin |
| Lower | **KG-solve** — no linear solve/inverse | Closed-form regression, PCA, normalizing-flow log-det |
| Lower | **KG-idxvalue** — index arithmetic yields booleans only | ALiBi, relative-position numeric bias |
| Lower | **KG-datamask** — `where` clauses are index predicates only | GAT data-driven masked softmax |
| Lower | **KG-reshape** — no multi-axis-into-one-slot LHS | Reshape/flatten, Kronecker, im2col |
| Lower | **KG-functor** — axes are flat `Fin n` only | Generalized transformers over non-sequence data |
| Lower | **KG-recur** — `recurMorphism`/`scanPre` AST-only | Programmatic recurrences |

### D. How to track progress

- **A (soundness bugs):** none currently open.
- **B (sharp edge):** none currently open.
- **C (missing primitives):** each closed gap should have a new `[N]` test added and its `KG-*`
  entry moved from §14 to "Confirmed NON-gaps" at the bottom of §14.

---

## 19. Design notes — deferred fixes (not yet implemented)

Analysis only for both subsections below — **holding off on implementation for now.**

### 19.1 KG-gather — data-dependent gather/scatter

Data-dependent gather/scatter (`Msg[dst[e]] += X[src[e]]`, embedding lookup by id, top-k
routing) is blocked because `IdxExpr` (`DSL/Ast.lean:25-31`) only admits integer-affine index
expressions (`.axis`, `.const`, `.scale`, `.shift`, `.affine`) — there's no case for "use
another tensor's value as an index." Below is what a fix would touch, stage by stage, and why
the cost differs sharply depending which of two independent Lean tracks in this repo you land
it in.

#### What it would take, pipeline stage by stage

1. **`DSL/Ast.lean`** — add a new `IdxExpr` case, e.g. `.gather : String → List IdxExpr →
   IdxExpr`. `idxAffineForm : IdxExpr → Int × List (Int × UID)` (lines 36-41) is currently
   **total** over the 5 affine constructors; a `.gather` arm has no valid `(const, coeffs)` —
   the signature must become `Option (Int × List (Int × UID))`, returning `none` for `.gather`.
2. **`DSL/Syntax.lean`** — not just a semantic rejection: the grammar itself has no production
   for it. `tl_idx_expr` only accepts `num`, `ident`, `num*ident`, and affine `+`/`-`
   combinations; the `ident[...]` bracket form exists one level up, at `tl_factor`. A new
   `tl_idx_expr` production plus an elaborator case is needed before `X[src[e]]` even parses.
3. **`Eval/Shape.lean`** (`inferAxisSizes`, lines 425-497) — becomes `filterMap` instead of
   `map` at the two `idxAffineForm e` call sites (lines 433, 449): a `.gather` read simply
   contributes no size equation for the axis it indexes into. The fixpoint loop, upper-envelope
   constraint builder, RREF solve, and the Issue-D/Issue-H checks are all untouched.
4. **`Stmt.readsOf`** (`Eval/Shape.lean:15-20`) — needs a new case to recurse into a
   `.gather`'s inner read (e.g. `src[e]`) so that the inner read's own axis (`e`) still gets
   sized normally; today `readsOf` never looks inside index expressions, because there's never
   been anything to recurse into.
5. **Consequence, not a new failure mode:** the gathered-into axis (e.g. `X`'s row axis) can no
   longer be sized from that occurrence. It must be inferable elsewhere (another affine read,
   or an explicit `axis ... = n` pin) or the solver fails loud — via the *existing* Issue-D
   guard (`Shape.lean:493-496`, "appears only with non-positive coefficients... add an explicit
   axis declaration") and the "no inferable size" guard already in `evalAssignWith`. No new
   failure-mode design needed.
6. **`Eval/Gather.lean` / `Eval/Contract.lean`** — `gather` needs to evaluate the inner read to
   a `Float`, round it to an index, and decide an out-of-range policy (zero-pad, matching the
   existing affine-read convention, is the natural default). `termAxisUIDs`/`idxAxisUIDs` need
   a `.gather` case, contributing the *inner* index's UIDs — not the gathered tensor's own axis.
7. **`Eval/Scatter.lean`** — `scatterOutShape` currently requires an affine LHS to bound output
   shape at compile time. A dynamic-indexed scatter LHS (`Msg[dst[e]]`) breaks that; output
   shape would need an explicit `axis` size pin instead of inference — the same escape hatch
   already used elsewhere for underdetermined scatter output axes (cf. RJ10).

#### Proof-burden verdict

Two independent Lean tracks exist in this repo, confirmed disjoint by import search — the
fix's cost differs sharply between them:

- **`Eval/*`** (the runtime interpreter: `Shape`, `Gather`, `Contract`, `Scatter`, `Tensor`) —
  `DenseTensor` is a plain runtime record (`{shape : List Nat, data : Array Float}`), not
  dependently typed; everything is `Except EvalError`. **Zero theorems/lemmas anywhere in
  `Eval/`.** Every change above is ordinary interpreter code — no proof cost.
- **`DSL/Pipeline/Lowering.lean` + `DSL/Target.lean`** — a separate compile-time backend that
  lowers `IdxExpr` into affine reindexing matrices (`StMatP`, and a dependently-typed
  `StMatP' (domLen codLen : Nat)`). This track carries real proofs — `reindexing_wellFormed`,
  `elaborateReindexings_wellFormed`, `StMatP'.toStMatP_wellFormed` — built on the same
  `idxAffineForm` primitive, with a load-bearing comment stating outright that *"`IdxExpr` is
  affine by construction, so there is no non-affine case to reject."* Option-ifying
  `idxAffineForm` (step 1 above) reaches this track too, since it's shared. The existing affine
  well-formedness proofs don't need reworking — but every call site needs an explicit new
  branch to reject `.gather` at compile time.

**Two implementation options, if/when this is picked up:**

- **Cheap:** support `.gather` in `Eval/` only; reject it explicitly in `Lowering` (one new
  error branch, no proof rework) — treating it like `recurMorphism` already is, an `Eval`-only
  escape hatch not compiled to `StMatP`. Gather-using programs simply can't be compiled to the
  matrix backend.
- **Expensive:** generalize `Lowering`/`Target` to represent non-affine reindexing, which has
  no fixed-width row by construction — a genuine proof-engineering project, not a mechanical
  extension.

### 19.2 General semiring notation — min-plus

This expands the §14 "Design direction — general semiring notation" stub into a concrete,
file-grounded plan. The gap it closes is the remaining half of **KG-min**: `minreduce` already
gives tropical **min-times** `(×, min, +∞)` (TR6/TR7, RC7), but **min-plus** `(+, min, +∞)` —
the semiring shortest-path / Viterbi / Bellman-step actually needs — is still `[F]` (TR5, §11):
`minreduce` supplies the `min` **aggregate** (⊕), but the within-term **combine** (⊗) is
hardcoded to multiplication. The fix generalizes `AggOp` from three flat named reductions into
an explicit `(⊕, ⊗)` pair, e.g. `D[i,k] := agg(min, +)(D[i,j] · W[j,k])` — same surface `·`
between factors, but `agg` reinterprets what "combine" and "reduce" mean arithmetically.

#### What it would take, pipeline stage by stage

1. **`DSL/Ast.lean:74,89`** — `AggOp` is currently a flat `inductive AggOp | sum | max | min`,
   each implicitly bundling an aggregate *and* an assumed `⊗ = ×`. It needs to carry an
   independent `(⊕, ⊗)` pair instead (e.g. two small enums packaged in a structure), so `min`
   aggregation can pair with either `×` (today's `minreduce`) or `+` (min-plus).
2. **`DSL/Syntax.lean:140-145`** — add a new `agg(⊕,⊗)` `tl_agg` production alongside the
   existing bare `maxreduce`/`minreduce` keyword productions. Those keywords stay as sugar for
   `agg(max,×)`/`agg(min,×)`, so no existing test needs to change.
3. **`DSL/Elab.lean:180-183`** (`elabTLAgg`) — extend to build the richer `AggOp` from the new
   `agg(⊕,⊗)` syntax form (currently it only matches the two bare keywords).
4. **`DSL/Pipeline/Structural.lean:237`** — the predicate-dtype guard
   (`unless rhs.agg == .sum do throw (.predicateAgg nm)`) compares against a literal `.sum`
   today; it needs to check "is this the `(sum, ×)` identity combo" instead, with a decision on
   whether any non-`(sum,×)` combo is legal for a `predicate` output (almost certainly not —
   same rejection as today, just re-expressed).
5. **`DSL/Pipeline/Lowering.lean:546-550`** (`Stmt.agg` → `BrOp` tag selection) — this match
   keys **only** on the ⊕ component today to pick `.maxreduce`/`.minreduce`/`.contract`.
   `BrOp` (`Target.lean:73-84`) needs **no new variant** — confirmed by grep that `BrOp` is
   never referenced anywhere in `Base/Br*.lean`, and no theorem in `Target.lean`/`Lowering.lean`
   mentions `op`/`BrOp`/`AggOp` in its statement. So this is a purely mechanical relabeling:
   destructure the new `AggOp` shape, keep picking the `BrOp` tag from the ⊕ half alone.
6. **`Eval/Contract.lean`** — `Combine` (currently `{mul, combine, unit0}`, lines 106-109) needs
   a 4th field, the ⊗-identity (e.g. `unitOtimes`); `combineFor` (134-140) needs to derive both
   `mul` and `unitOtimes` from the ⊗ choice (today `Combine.max`/`.min` hardcode
   `mul := (·*·)`). The literal `prod := 1.0` hardcoded in `evalAssignWith`/`evalAssignSeeded`
   (lines 92, 185) becomes `prod := c.unitOtimes`. Natural accompanying cleanup: pass the whole
   `Combine` struct into those two functions instead of three loose `(mul, combine, unit0)`
   params, since a 4th loose parameter would make an already-wide signature noisier.
7. **`Eval/Scan.lean:39-42`** — the same `Combine`-from-`rhs.agg` selection used for scan steps
   needs the identical generalization, so `agg(min,+)` composes with a scan step exactly the way
   `maxreduce`-in-scan already does (RC5/RC10).
8. **Test:** TR5 (§11) flips from a documented `[F]` gap to a live `[N]` test, e.g.
   `D[i,k] := agg(min, +)(D[i,j] · W[j,k])` for a min-plus shortest-path step.

#### Proof-burden verdict

**Confirmed proof-free on both tracks** — a sharp contrast with KG-gather (§19.1). Grepping
`Base/Br*.lean` (the category-theory proof track — `Br.lean`, `BrNF.lean`, `BrWiring.lean`)
shows **zero references to `BrOp` at all**: that track proves things about categorical
wiring/composition structure, and never inspects which reduction or combine a step performs.
Within `DSL/Target.lean`/`DSL/Pipeline/Lowering.lean` themselves, the only theorem
(`StMatP'.toStMatP_wellFormed`, `Target.lean:51`) is entirely about reindexing-matrix shape and
never mentions `op`/`BrOp`/`AggOp`. So unlike KG-gather — where changing `idxAffineForm`'s
signature ripples into the compile-time well-formedness proofs — generalizing `AggOp` to a
semiring pair touches **zero existing theorems** anywhere in the repo. It's a data-model
extension (`AggOp`) + a grammar addition + an arithmetic generalization in
`Eval/Contract.lean`/`Eval/Scan.lean`, fully contained to non-proof-bearing code. This is the
cheaper of the two deferred gaps analyzed so far, on both the interpreter side (as with gather)
*and* the compile/proof side (unlike gather).
