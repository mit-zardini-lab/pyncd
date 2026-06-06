# Iteration in Tensor Logic

This document records the design decisions for expressing iterative recurrences
in pyncd's tensor logic DSL. It covers motivation, syntax, and the compiler
consistency checks that make the design statically safe. §6 sketches how these
decisions map onto the `TL` and `ConstructedModule` machinery, and §7 gives the
diagrammatic (string-diagram) notation.

---

## Contents

- [1. Motivation](#1-motivation)
- [2. Design Decisions](#2-design-decisions)
  - [2.1 Declaration on the tensor, not the axis](#21-declaration-on-the-tensor-not-the-axis)
  - [2.2 Base case as a tensor equation](#22-base-case-as-a-tensor-equation)
  - [2.3 Explicit `l+1` on the LHS](#23-explicit-l1-on-the-lhs)
  - [2.4 Semantics of `l` on the RHS](#24-semantics-of-l-on-the-rhs)
  - [2.5 Iteration bounds from the axis declaration](#25-iteration-bounds-from-the-axis-declaration)
  - [2.6 Coupled recurrences use Jacobi-style updates](#26-coupled-recurrences-use-jacobi-style-updates)
- [3. Full Syntax Summary](#3-full-syntax-summary)
- [4. Compiler Consistency Checks](#4-compiler-consistency-checks)
  - [4.1 Axis must carry a concrete size](#41-axis-must-carry-a-concrete-size)
  - [4.2 Base case literal must match the axis lower bound](#42-base-case-literal-must-match-the-axis-lower-bound)
  - [4.3 Exactly one base case and one recurrence per iterative tensor](#43-exactly-one-base-case-and-one-recurrence-per-iterative-tensor)
  - [4.4 `l+1` on the RHS is a causality violation](#44-l1-on-the-rhs-is-a-causality-violation)
  - [4.5 `l` is not a contraction axis within a recurrence equation](#45-l-is-not-a-contraction-axis-within-a-recurrence-equation)
  - [4.6 Coupled recurrences must share the same axis](#46-coupled-recurrences-must-share-the-same-axis)
  - [4.7 Non-recurrent tensors must supply the full `l` dimension](#47-non-recurrent-tensors-must-supply-the-full-l-dimension)
- [5. Out of Scope (Static Case)](#5-out-of-scope-static-case)
- [6. DSL Changes and Br Morphism Representation](#6-dsl-changes-and-br-morphism-representation)
  - [6.1 Required DSL changes](#61-required-dsl-changes)
  - [6.2 Block vs Scan: two categorical cases](#62-block-vs-scan-two-categorical-cases)
  - [6.3 Step body extraction](#63-step-body-extraction)
  - [6.4 Compilation to PyTorch](#64-compilation-to-pytorch)
  - [6.5 Complexities](#65-complexities)
  - [6.6 Visual rendering](#66-visual-rendering)
- [7. Diagrammatic Notation: a String-Diagram Realization of the ScanBox](#7-diagrammatic-notation-a-string-diagram-realization-of-the-scanbox)
  - [7.1 Why a scan diagram is not a broadcast](#71-why-a-scan-diagram-is-not-a-broadcast)
  - [7.2 The rolled ScanBox](#72-the-rolled-scanbox)
  - [7.3 Unrolled semantics (fixed N)](#73-unrolled-semantics-fixed-n)
  - [7.4 Correspondence to the compiled module](#74-correspondence-to-the-compiled-module)
  - [7.5 Conventions](#75-conventions)
  - [7.6 Implementation status in tsncd — and what remains](#76-implementation-status-in-tsncd--and-what-remains)
- [Appendix: Categorical Status of `Scan` in Br](#appendix-categorical-status-of-scan-in-br)
  - [A.1 The core axioms `Scan` satisfies](#a1-the-core-axioms-scan-satisfies)
  - [A.2 The structural property `Scan` fails — the batch lift independence axiom](#a2-the-structural-property-scan-fails--the-batch-lift-independence-axiom)
  - [A.3 Why `Block` does not already cover `Scan`](#a3-why-block-does-not-already-cover-scan)
  - [A.4 Interaction with weaves and reindexings](#a4-interaction-with-weaves-and-reindexings)
  - [A.5 Summary](#a5-summary)

---

## 1. Motivation

A broad class of computations — gradient descent, RNNs, dynamic programming,
Viterbi, belief propagation — is defined by a recurrence relation: the value of
a tensor at step `l+1` depends on its own value at step `l`.

A naive reading of the DSL suggests writing this directly as a tensor equation:

```python
tl.H[i, l+1] = tl.H[i, l] + lr * tl.Grad[i, l]
```

This does not work, for two compounding reasons.

The first is representational: einops contraction strings have no notion of
shifted axes. `l` and `l+1` are the same axis offset by one position, but
einops treats axis labels as opaque symbols — there is no string that means
"output's `l`-axis is input's `l`-axis shifted by one." The existing
`bc_signature()` machinery would have to emit something like
`... i l, ... i l -> ... i l` (treating `l` as a free axis, since it appears on
both sides), which is a simple elementwise copy — not a recurrence.

The second reason is fundamental: even if shifted-axis einsums were
representable, einsum evaluates all output indices simultaneously from a fixed
set of input tensors. To compute `H[i, l+1]` for all `l` at once, you would
need `H[i, l]` for all `l` to already exist as a concrete tensor. But for
`l > 0`, `H[i, l]` is itself part of the output being computed. The dependency
is inherently sequential — step `l+1` cannot begin until step `l` is complete
— and no static tensor contraction can encode this ordering.

The `Block` morphism handles a special case, but is subject to a strict
`dom = cod` constraint: the body morphism `f` must have the same domain and
codomain so that `f ∘ f ∘ … ∘ f` type-checks. A recurrence step naturally has
domain `(H_l, Grad_l)` — the current state together with a per-step external
input — and codomain `H_{l+1}`, the updated state alone. These are not equal,
so `Block` cannot type-check the recurrence directly. Absorbing `Grad` as a
parameter to recover `dom = cod` does not help either, because `Block` applies
the *same* morphism at every step: a per-step varying input such as `Grad[:, l]`
cannot be accommodated (§6.2 analyses both cases and introduces `Scan`).

The goal is to extend pyncd's DSL with a first-class notation for recurrences
that:

- fits the existing equation model without introducing a parallel syntax,
- makes the iterative structure statically visible (not inferred from runtime
  control flow),
- allows the compiler to validate the recurrence before any tensor is
  allocated, and
- composes naturally with the existing Bool semiring / Iverson predicate
  machinery.

---

## 2. Design Decisions

### 2.1 Declaration on the tensor, not the axis

The iterative semantics belong to the tensor being defined recursively, not to
the index variable. An axis `l` may appear in several tensors simultaneously
with different roles: as the iteration variable in a recurrent tensor, and as a
plain slice index in a pre-loaded input tensor. Attaching the declaration to
the tensor preserves this distinction.

The chosen syntax mirrors the existing `.predicate()` declaration:

```python
tl.Out.predicate(i, j)        # marks Out as Bool-typed over (i, j)
tl.H.iteration_axis(l)        # marks H as iteratively defined over l
```

Just as `.predicate()` does not change what `l` means to other tensors,
`.iteration_axis(l)` does not change the type of `l` itself. `l` remains a
plain `real_axis` and may appear freely in non-recurrent equations.

A special `iter_axis` type is therefore not needed and not introduced.

This stands in contrast to Domingos' tensor logic, where the iteration axis is
marked with a `*` annotation on the axis itself — making the iterative role a
property of the index variable rather than of the tensor being defined. The
pyncd approach avoids the ambiguity that arises when the same axis appears in
multiple tensors with different roles, and keeps the declaration co-located with
the tensor it governs.

### 2.2 Base case as a tensor equation

The initial condition is expressed as a tensor equation with a literal integer
at the `l` slot:

```python
tl.H[i, 0] = tl.X[i]
```

This means "the 0-th slice of `H` along `l` equals `X`." It is a first-class
assignment in the `TL` registry, stored alongside the recurrence entry. The
initial condition can be an arbitrary tensor equation — a linear map, a
normalisation, or a copy — without requiring a dedicated `init=` parameter on
`iteration_axis`.

The compiler identifies the base case by detecting a literal integer at the `l`
slot of the LHS, and the recurrence by detecting `l+1` at that slot.

### 2.3 Explicit `l+1` on the LHS

The recurrence equation uses `l+1` on the LHS to mark the step being defined:

```python
tl.H[i, l+1] = tl.H[i, l] + lr * tl.Grad[i, l]
```

This is the standard mathematical convention for difference equations
(`H_{l+1} = f(H_l)`) made explicit at the DSL level. Requiring it — rather
than inferring "next step" from context — keeps the iterative structure visible
at the definition site and allows the compiler to reject equations that use `l`
on the LHS without an offset (which would be an ambiguous write to an arbitrary
slice).

`l+1` is implemented by `RawAxis.__add__(int)`, which returns a typed
`IterNextRef` object that `__setitem__` recognises. It is only valid at the `l`
slot of the LHS of a tensor declared with `.iteration_axis(l)`.

The symmetric operation `RawAxis.__sub__(int)` returns an `IterPrevRef` for use
on the RHS, intended to support two-term recurrences that read two previous
steps. **This case is design-only — the type exists, but compiler and morphism
support are not yet implemented (§5).** The remainder of this subsection
describes the *planned* rule:

```python
tl.H[i, l+1] = tl.H[i, l] + tl.H[i, l-1]   # Fibonacci-style — design only (§5)
```

A two-term recurrence would require two base case equations (one for `l=0` and
one for `l=1`). The planned compiler rule infers that the inductive step must
start at `l=1` — not `l=0` — because reading `H[i, l-1]` at `l=0` would
reference the non-existent slice `H[:, -1]`. The rule is: the recurrence starts
at `l = max_lookback`, where `max_lookback` is the largest offset among all
`IterPrevRef` references on the RHS (here, 1), and exactly `max_lookback` base
case equations must be present.

```python
l = real_axis('l', 10)
tl.H[i, l+1] = tl.H[i, l] + tl.H[i, l-1]   # design only (§5); no bound predicate needed
```

Multi-step look-back is out of scope for the current static design (§5); the
`IterPrevRef` type anticipates it.

### 2.4 Semantics of `l` on the RHS

Within a recurrence equation, plain `l` on the RHS has two different roles
depending on which tensor it indexes:

| Tensor | Role of `l` on RHS | Memory contract |
| --- | --- | --- |
| Recurrent (`H`) | read current step — the running state from the previous iteration | one slice in memory at a time |
| Non-recurrent (`Grad`) | slice index — read the `l`-th entry of a pre-loaded tensor | full `l` dimension must be pre-loaded by the caller |

The distinction is determined by the declaration: any tensor that appears with
`l` on the RHS but has not been declared with `.iteration_axis(l)` is treated
as a pre-loaded input. No additional annotation is needed on the non-recurrent
tensor.

### 2.5 Iteration bounds from the axis declaration

The iteration count is determined by the size declared on the axis itself:

```python
l = real_axis('l', 10)    # l ∈ {0, …, 9} — 10 steps
tl.H.iteration_axis(l)
tl.H[i, 0] = tl.X[i]
tl.H[i, l+1] = tl.H[i, l] + lr * tl.Grad[i, l]
```

This is consistent with how every other axis is declared in pyncd:
`real_axis('i', 16)` gives `i` sixteen values; `real_axis('l', 10)` gives `l`
ten values. The iterative nature of `l` is declared on the tensor via
`iteration_axis`; the range of `l` is a property of the axis itself.

`iteration_axis` therefore takes no bounds parameter. The compiler reads the
step count directly from `l._size._value`. The lower bound is 0 by convention
and is cross-checked against the base case literal (§4.2).

Dynamic iteration counts (where `L` is determined at runtime by an input
tensor's shape) are out of scope and require a separate mechanism.

### 2.6 Coupled recurrences use Jacobi-style updates

When two tensors are both declared with `.iteration_axis(l)`, their updates are
treated as simultaneous: all RHS reads within a step use the values from step
`l`, and all writes produce the values at step `l+1`.

```python
l = real_axis('l', 10)
tl.H.iteration_axis(l)
tl.G.iteration_axis(l)
tl.H[i, l+1] = tl.H[i, l] + tl.G[i, l]
tl.G[i, l+1] = tl.G[i, l] * tl.H[i, l]   # uses H[i, l], not H[i, l+1]
```

This follows directly from the `l` convention: `l` on the RHS always means
"the value at the current step," regardless of whether that tensor's own
recurrence has already been evaluated. Gauss-Seidel ordering (where a later
equation in the same step reads an already-updated value) is not expressible
and not supported. This is a deliberate restriction: it keeps the semantics
independent of equation order and makes the update well-defined.

---

## 3. Full Syntax Summary

```python
# Declare axes
i = real_axis('i', 16)
l = real_axis('l', 10)                     # l ∈ {0, …, 9} — 10 steps

# Declare iterative tensor
tl.H.iteration_axis(l)

# Base case (literal 0 at the l slot)
tl.H[i, 0] = tl.X[i]

# Inductive step
tl.H[i, l+1] = tl.H[i, l] + lr * tl.Grad[i, l]
```

`X` is a pre-loaded input of shape `(I,)`. `Grad` is a pre-loaded input of
shape `(I, 10)` — the full `l` dimension, which the caller must supply. The
step count is read directly from `l._size._value = 10`.
`H` is the running state. The compiled module:

1. Initialises `H[:, 0]` from `X`.
2. For `l` in `[0, 9]`: computes `H[:, l+1]` from `H[:, l]` and `Grad[:, l]`.
3. Returns `H` of shape `(I, 11)` — initial state plus one slice per step.

Whether the full history or only the final state is returned is addressed in
§6.4.

---

## 4. Compiler Consistency Checks

The following checks can all be performed statically, before any tensor is
allocated or any module is constructed.

### 4.1 Axis must carry a concrete size

For a tensor declared with `.iteration_axis(l)`, the axis `l` must be a
`real_axis` with a concrete integer size — i.e. `l._size` must be an `Integer`,
not an unsized placeholder. The step count is `l._size._value` and the
iteration range is `[0, l._size._value - 1]`. An unsized `l` is a static error.

### 4.2 Base case literal must match the axis lower bound

The iteration range starts at 0 by convention. The base case equation must
have the literal `0` at the `l` slot. A mismatch — e.g. a base case at `l=2`
— leaves steps `0` and `1` undefined and is rejected. For two-term recurrences
using `IterPrevRef`, a second base case at `l=1` must also be present; the
compiler derives the required number of base cases from the maximum look-back
offset among all `IterPrevRef` references in the recurrence equation.

### 4.3 Exactly one base case and one recurrence per iterative tensor

Each tensor declared with `.iteration_axis(l)` must have exactly:

- one equation whose LHS has a literal integer at the `l` slot (base case), and
- one equation whose LHS has `l+1` at the `l` slot (inductive step).

Zero, or more than one, of either is an error.

### 4.4 `l+1` on the RHS is a causality violation

Within any recurrence equation over `l`, an expression `H[..., l+1]` on the
RHS reads a step that has not yet been computed. This is a static error. The
compiler scans all RHS index tuples for `IterNextRef` objects and rejects any
that appear.

### 4.5 `l` is not a contraction axis within a recurrence equation

`l` is the iteration variable. Within the recurrence equation it appears either
as a positional read of the running state (`H[i, l]`) or as a slice index of a
pre-loaded tensor (`Grad[i, l]`). It must not appear as a contracted axis (one
that is present on the RHS but absent from the LHS without an offset). If `l`
appears in a tensor on the RHS but not on the LHS in any form, the compiler
rejects the equation and reports that `l` cannot be contracted within a
recurrence.

### 4.6 Coupled recurrences must share the same axis

If two tensors are both declared with `.iteration_axis(l)`, they must reference
the identical `l` axis object — same `uid` and same `_size`. Declaring two
separate axes with the same name but different sizes and coupling them is
rejected. Because the iteration count is encoded in the axis declaration, this
check reduces to a single `uid` comparison rather than requiring bound
extraction from multiple equations.

### 4.7 Non-recurrent tensors must supply the full `l` dimension

Any non-recurrent input tensor indexed with `l` in a recurrence equation must
have a concrete size along `l` equal to `l._size._value`. The compiler verifies
this against the axis size at signature construction time.

---

## 5. Out of Scope (Static Case)

The following are deferred and not addressed by this design:

**Dynamic iteration count.** When the number of steps is determined at runtime
by an input tensor's shape, the Iverson predicate cannot express it. A
`steps=` parameter on `iteration_axis`, or inferring `L` from a sized input
axis, would handle this case but is left for a later design iteration.

**Multi-step look-back.** Recurrences of the form `H[i, l+1] = f(H[i, l], H[i, l-1])`
require reading two previous steps. The DSL types (`IterPrevRef`, base case
counting) are sketched in §2.3, but compiler and morphism support for this case
are not part of the current implementation.

**Non-sequential iteration order.** The compiled execution always steps `l`
from lower to upper in unit increments. Reverse-mode, stride-2, or arbitrary
permutation orderings are not expressible.

**Coupled recurrences with subset outputs.** When two or more tensors are
coupled under `.iteration_axis(l)` and a downstream equation in the same `TL`
session consumes only a subset of the coupled outputs, the chaining is
incorrect: `ConstructedComposed` feeds the full output tuple to the next
module, which expects a single tensor. Coupled scans must therefore be
terminal in their `TL` session. Routing individual coupled outputs into
further equations is not yet supported.

---

## 6. DSL Changes and Br Morphism Representation

This section sketches what implementing the design requires at the DSL level and
how iterated recurrences map into the Br categorical framework. It also
identifies the points of significant complexity.

### 6.1 Required DSL changes

**New index reference types.** `RawAxis.__add__(int)` must return a typed
`IterNextRef(axis, offset)` object rather than raising an error or returning an
integer. `__setitem__` on `TensorProxy` recognises `IterNextRef` at an axis slot
and records a recurrence equation. Symmetrically, `RawAxis.__sub__(int)` returns
`IterPrevRef(axis, offset)` for use on the RHS in two-term recurrences.

**Literal integer at the iteration axis slot.** `TensorProxy.__setitem__`
currently expects `RawAxis` objects as indices. It must be extended to accept a
plain `int` at the iteration axis slot and record the resulting equation as a
base case. The iteration slot is identified by cross-referencing the index
tuple with `_iteration_axes[name]` — the position of the recurrence axis in
the declared shape. An `int` at that position signals a base case:
`_register_entry()` strips the iteration slot from the LHS indices (reducing
shape from `(i, l)` to `(i,)` for the slice), builds a regular
`_build_rhs_morphism()` morphism over the stripped indices, and stores the
result in `_pending_iter[name]['base_case']` rather than `_entries`. No
extension to `_build_rhs_morphism()` or `bc_signature()` is required — the
base case is an ordinary tensor equation over the non-iteration axes.

**`TensorProxy.iteration_axis(l)`.** A new method analogous to `.predicate()`.
It records `l` as the iteration axis for that tensor on the `TL` registry.

**Multiple entries and deferred assembly.** The `TL` registry currently builds
and stores at most one morphism entry per LHS tensor name. Iterative tensors
require two: the base case and the recurrence. `Scan` assembly requires both
simultaneously — the base case morphism establishes the initial-state type and
the recurrence provides the step body — so neither assignment alone has
sufficient information to finalize the morphism. Iterative entries are
therefore stored in a separate `_pending_iter: dict[str, dict]` dict keyed by
tensor name rather than in `_entries`. `to_morphism()` calls `_finalize_iter()`
before collecting `_entries`. `_finalize_iter()` groups all iterative tensors by
iteration axis uid. Tensors that share an axis uid are dispatched to
`_finalize_iter_group`, which emits a single coupled `Scan(n_states=k)` entry
with a synthetic lhs name (`__coupled_G_H` for states G and H). Tensors with
a unique uid take the existing single-state path: `_finalize_iter()` runs the
§4 consistency checks, assembles the step body via the rewriting pass, wraps it
in `Scan`, and appends the resulting morphism to `_entries`. The existing
`_entries`/`_name_to_axes` infrastructure is then used unchanged for
serialization and downstream composition.

**Axis tracking for iterative tensors.** `_name_to_axes[H]` must hold the
full logical shape `(i, l)` — not the slice shape `(i,)` produced by the step
body or the base case — so that downstream equations referencing `tl.H[i, l]`
receive the correct axes for unification. The base case registration does not
update `_name_to_axes[H]`. When the recurrence entry `H[i, l+1] = ...` is
stored in `_pending_iter`, `_name_to_axes[H]` is set immediately to the full
shape: `decl.shape` from `_declarations[H]` if declared, or `(i, l)` inferred
from the recurrence LHS by replacing `IterNextRef(l, 1)` with `l`. If a
downstream equation references `H[i, l]` before the recurrence entry is
processed, `_name_to_axes[H]` will be absent and no unification is attempted —
consistent with existing behavior for forward references.

**Step count from axis size.** The iteration count is read directly from
`l._size._value`. No predicate extraction is required. The compiler validates
that `l` carries a concrete `Integer` size at `_register_entry()` time (i.e.,
when the equation is assigned).

### 6.2 Block vs Scan: two categorical cases

The appropriate Br morphism depends on whether the recurrence body refers to any
non-recurrent tensor indexed by `l`.

**Pure state recurrence** — the RHS contains only the recurrent tensor and
constants:

```python
l = real_axis('l', 10)
tl.H[i, l+1] = f(tl.H[i, l])
```

The step body `f: H_slice → H_slice` has `dom = cod`. This fits the existing
`Block` morphism directly. The compiled module is `nn.Sequential` of `N` copies
of the step body, which is already supported.

**Recurrence with per-step external inputs** — the RHS also reads a pre-loaded
tensor at slice `l`:

```python
l = real_axis('l', 10)
tl.H[i, l+1] = tl.H[i, l] + lr * tl.Grad[i, l]
```

The natural step body takes `(H_slice, Grad_slice) → H_next_slice`, so
`dom = (State, InputSlice) ≠ cod = State`.

One might recover `dom = cod` by augmenting the state to carry the full `Grad`
tensor and a step counter, with the step body performing
`Grad_full[:, l_counter]` to extract the current slice. This is technically
valid but introduces dynamic integer indexing and counter increment — neither
of which is expressible as a `TensorEquation` einsum. New operator types would
be required in Br, and the complexity would exceed that of adding `Scan`
directly. The augmented state also inflates every step of the `nn.Sequential`
chain with a full tensor and a bookkeeping scalar that have no mathematical
meaning at the step level.

`Scan` is the honest abstraction: `dom ≠ cod` at the step level is resolved by
lifting the domain and codomain to the outer level
(`(StateInit, InputSequence) → StateSequence`), where the shape mismatch is
natural. A new `Scan` morphism is required (see Appendix for its categorical
status in Br and its interaction with weaves and reindexings; §6.6 for
rendering):

```python
@dataclass(frozen=True)
class ScanAffine:
    A_morphism: object | None    # per-step A factor; None means identity
    b_morphism: object | None    # per-step bias; None means zero
    state_in_axes: tuple         # contracted state axes (non-empty iff matrix recurrence)
    a_positions: tuple[int, ...] # indices into step_xs selecting A_module inputs
    b_positions: tuple[int, ...] # indices into step_xs selecting b_module inputs

@dataclass(frozen=True)
class Scan(fd.Term):
    step: object          # uncoupled: morphism (H_state, *xs) → H_next
                          # coupled:   tuple[morphism, ...], one per state
    base: object          # uncoupled: morphism *xs → H_0
                          # coupled:   tuple[morphism, ...], one per state
    N: nm.Numeric         # nm.Integer; N._value is the step count
    axis: sc.RawAxis      # recurrence axis l
    affine: ScanAffine | None = None  # affine decomposition for associative_scan (uncoupled only)
    n_states: int = 1     # number of coupled state tensors; 1 = uncoupled
    step_state_deps: tuple[tuple[int, ...], ...] = ()
                          # coupled only: for each step morphism, the indices (into
                          # the canonical states tuple) it reads, in morphism-domain order.
                          # Empty tuple = all states in canonical order 0..n_states-1.
```

`Scan` is directly analogous to the functional programming primitives `foldl`
and `scanl`. `foldl f z [x_0, …, x_{N-1}]` computes
`s_{l+1} = f(s_l, x_l)` starting from `s_0 = z` and returns only the final
accumulator `s_N`; `scanl` returns the full sequence `[s_0, …, s_N]`. `Scan`
uses `scanl` semantics — always returning the full state history — for the
reasons given in §6.4.

`Block` is then the degenerate fold over a sequence of `N` unit inputs:
`foldl f z (repeat () N)` — the step body ignores its second argument, giving
`dom(step) = cod(step) = State`. The two morphisms unify as fold/scan with
different input sequences; `Block` is the special case where no per-step
external input exists.

`Scan` has domain `(BaseInputs..., InputSequence...)` and codomain
`StateSequence` (full state history; see §6.4). The `base` morphism is stored
directly inside `Scan` and run at the start of `ConstructedScan.forward`; there
is no external `Composed(BaseCase, Scan)` wrapper. All caller-supplied tensors
— both the initial-condition inputs and the per-step sequences — are passed
together to `ConstructedScan.forward`, which splits at `n_base`:

```text
X ──────────────────────────────────────────────────┐
                                          Scan ──► H
Grad ───────────────────────────────────────────────┘
       (base and step both contained within Scan)
```

`Block` is the correct morphism for pure-state recurrences (no per-step
external inputs), but currently `_finalize_iter` always emits `Scan`. Block
emission is deferred as a future optimisation. `Scan` is a new sibling
morphism and requires a new `ConstructedScan` registered in `torch_compile`.

### 6.3 Step body extraction

The central algorithmic challenge is deriving the step body morphism from the
recurrence equation. Given:

```text
H[i, l+1] = H[i, l] + lr * Grad[i, l]
```

the compiler must produce a step body morphism via a single rewriting pass.
The pass is valid for any recurrence equation regardless of the operator, the
number of recurrent tensors, or the number of non-recurrent tensors indexed by
`l`. Every tensor reference `T[..., l, ...]` becomes `T_local[...]` with the
`l` slot removed. The role of each stripped reference is determined by the
tensor's `iteration_axis` declaration and the form of the `l` reference:

- `l+1` on the LHS → output proxy
- plain `l` on a tensor declared with `.iteration_axis(l)` → state input
- plain `l` on any other tensor → per-step input

This classification determines the input and output weave structure of the step
body for any recurrence. For the running example it produces:

- Input weaves: `[Weave(H_state, (i,)), Weave(Grad_slice, (i,))]`
- Output weaves: `[Weave(H_out, (i,))]`
- Step body morphism: a `Composed(ProductOfMorphisms(H_term, Grad_term), AdditionOp_br)`
  — the additive RHS `H[i, l] + lr * Grad[i, l]` strips `l` from each factor
  and compiles via the existing `SumExpr` path: each stripped term becomes a
  `Broadcasted(TensorEquation, ...)` and the addition step is a
  `Broadcasted(AdditionOp, ...)` over the shared `(i,)` degree.

### 6.4 Compilation to PyTorch

`ConstructedScan.__init__` builds two compiled sub-modules and records the key
scalars:

- `self.step_module` — a `ConstructedModule` compiled from the step body
  extracted in §6.3, mapping `(H_state, *non_state_inputs) → H_next` with `l`
  stripped from all indices. Determined entirely by the §6.3 rewriting pass.
- `self.base_module` — a `ConstructedModule` compiled from the base case
  morphism, mapping the initial-condition inputs to `H_0`. The base case is
  held inside `ConstructedScan` and evaluated at the start of `forward`, not in
  an external `Composed` chain.
- `self.N` — the step count `l._size._value`, fixed at construction time.
- `self.n_base` — the number of caller-supplied base-case inputs. Derived from
  `len(self.base_module._caller_positions)` when the base module has
  pre-materialised Iverson buffers; otherwise `len(target.base.dom())`.

`ConstructedScan.forward(*xs)` receives all caller inputs combined:
`xs[:n_base]` go to `base_module`; `xs[n_base:]` are the per-step sequence
tensors, each with N as the **last** dimension (l-last convention). The
sequential loop path:

```python
def forward(self, *xs):
    base_xs  = xs[:self.n_base]
    step_xs  = xs[self.n_base:]
    H0 = base_module(*base_xs)        # initial state
    if self._has_affine:
        return self._assoc_scan_forward(H0, step_xs)
    return self._run_loop(H0, step_xs)

def _run_loop(self, H, step_xs):
    outputs = [H]
    for l_idx in range(self.N):
        sliced = tuple(x[..., l_idx] for x in step_xs)  # l-last: slice trailing dim
        H = step_module(H, *sliced)
        outputs.append(H)
    return torch.stack(outputs, dim=-1)   # (*state, N+1)
```

The loop is wrapped with `torch._dynamo.disable` to prevent `torch.compile`
from unrolling it.

**Affine fast path.** When `Scan.affine is not None`, `ConstructedScan` uses
`torch._higher_order_ops.associative_scan` instead of the sequential loop.
All N copies of `A_l` and `b_l` are computed in one batched pass using
`torch.vmap` over the N dimension, then prefix-composed via the associative
combine function `(A₂, b₂) ∘ (A₁, b₁) = (A₂·A₁, A₂·b₁ + b₂)`. The result
is applied to `H0` to produce the full state sequence:

```python
def _assoc_scan_forward(self, H0, step_xs):
    # Batch over N via vmap; A_all/b_all have shape (N, *out).
    A_all = vmap(A_module)(step_xs[a_positions, ...].movedim(-1, 0))
    b_all = vmap(b_module)(step_xs[b_positions, ...].movedim(-1, 0))

    def combine(s1, s2):
        A1, b1 = s1; A2, b2 = s2
        if is_matrix:
            return einsum('ij,jk->ik', A2, A1), einsum('ij,j->i', A2, b1) + b2
        return A2 * A1, A2 * b1 + b2

    A_prefix, b_prefix = associative_scan(combine, (A_all, b_all), dim=0, combine_mode='generic')
    H_seq = (einsum('nij,j->ni', A_prefix, H0) + b_prefix) if is_matrix else (A_prefix * H0 + b_prefix)
    return cat([H0.unsqueeze(0), H_seq], dim=0).movedim(0, -1)  # (*state, N+1)
```

`combine_mode='generic'` is used in all cases (scalar and matrix) because
`'pointwise'` requires CUDA tensors. `is_matrix` is `True` when
`len(affine.state_in_axes) > 0` — i.e. the state has contracted (non-tiled)
axes.

All return the full state history (`scanl` semantics, shape `(*state, N+1)`).
In the ML context pyncd targets, `scanl` semantics are the common case:
sequence models need `H[:, l]` at every position, attention mechanisms contract
over the full state sequence, and DP algorithms require traceback through all
intermediate states. Even when the forward pass uses only the final state,
backpropagation through time requires all intermediates to be stored anyway.
Final-state-only output can be recovered as a compiler optimisation when use
analysis confirms the full history is never consumed, but it does not block
correctness.

**Coupled path (n_states > 1).** When `Scan.n_states > 1`, `ConstructedScan`
builds parallel lists of compiled sub-modules:

- `self.step_modules` — `nn.ModuleList` of per-state step morphisms in
  canonical (sorted-by-name) order.
- `self.base_modules` — `nn.ModuleList` of per-state base-case morphisms in
  the same order.
- `self.n_bases` — number of caller inputs consumed by each base morphism.
- `self.step_state_deps` — for each step morphism, the indices (into the
  canonical states tuple) of the states it reads, in morphism-domain order.
  Derived from `Scan.step_state_deps` at construction time.
- `self.n_step_xs` — number of per-step inputs consumed by each step morphism.
  Computed as `len(_caller_positions) - len(step_state_deps[k])`, correctly
  excluding the state inputs that are passed separately at runtime.

`forward(*xs)` for the coupled path splits `xs` as follows:

```text
xs[:n_bases[0]]                 — base inputs for state 0 (first in canonical order)
xs[n_bases[0]:...]              — base inputs for state 1, etc.
xs[sum(n_bases):...]            — per-step inputs for state 0
xs[sum(n_bases)+n_step_xs[0]:...] — per-step inputs for state 1, etc.
```

`_run_loop_coupled` applies Jacobi semantics: at each step `l`, every step
module receives the **old** values of its required states (read from
`states[j] for j in step_state_deps[k]`) and its own sliced per-step inputs,
then all next-states are materialised simultaneously before any is stored back.

```python
def _run_loop_coupled(self, states, step_xs_per_state):
    histories = [[s] for s in states]
    for l_idx in range(self.N):
        sliced = tuple(tuple(x[..., l_idx] for x in xs) for xs in step_xs_per_state)
        new_states = tuple(
            to_tuple(mod(*[states[j] for j in self.step_state_deps[k]], *sliced[k]))[0]
            for k, mod in enumerate(self.step_modules)
        )
        states = new_states
        for k, s in enumerate(states):
            histories[k].append(s)
    return tuple(torch.stack(hist, dim=-1) for hist in histories)
```

The return value is a `tuple[Tensor, ...]` of shape `(*state_k, N+1)` per
state, in canonical (sorted-by-name) order. `ConstructedComposed.forward` uses
`to_tuple` on sub-module outputs, so the tuple propagates correctly through
downstream composition chains.

### 6.5 Complexities

**Step body extraction is the core difficulty.** The `l` axis appears in the
recurrence entry's RHS factors but must be absent from the step body's weave
types. Implementing this requires a rewriting pass that renames each factor's
tensor reference to a stripped proxy, removes the `l` slot from its index
tuple, and feeds the result into `_build_rhs_morphism()` or
`_build_sum_morphism()` to build the step body morphism without the iteration
axis. This is substantially more involved than any existing morphism
construction.

*Resolution (implemented).* The step body is produced by a static rewriting pass
inside `_finalize_iter()`: `_strip_iter_axis_from_value` removes the `l` slot
from each RHS factor's index tuple, classifying recurrent-tensor references as
state-input proxies and non-recurrent references as per-step-input proxies. The
`l+1` reference on the LHS becomes the output proxy. The stripped RHS feeds into
`_build_step_morph()`, which delegates to the existing `_build_rhs_morphism()` /
`_build_sum_morphism()` paths — no new compilation path is required. Additive
recurrence bodies produce a `Composed(ProductOfMorphisms, AdditionOp Broadcasted)`
step morphism as described in §6.3.

**`Block` vs `Scan` determination requires tensor classification.** The compiler
must inspect all RHS tensor references in the recurrence equation, classify each
as recurrent (state) or non-recurrent (per-step input), and select the
appropriate morphism. A recurrence equation with only recurrent tensors on the
RHS fits `Block`; any non-recurrent tensor indexed by `l` forces `Scan`. The
classification must account for tensors that appear in both the base case and
the recurrence with different roles.

*Resolution (implemented).* The classification logic is implemented in `_is_pure_state_recurrence`
in `_finalize_iter`. However, `_finalize_iter` currently always emits `Scan` for
all recurrences, including pure-state ones. Emitting `Block` for pure-state
recurrences is deferred as an optimisation.

**Coupled recurrences require a multi-output step body.** When `H` and `G` are
both declared with `.iteration_axis(l)`, the equations are typically
cross-dependent: in the §2.6 example, `H_next` reads `G_state` and `G_next`
reads `H_state`. A `ProductOfMorphisms` step body cannot represent this because
it partitions inputs — each factor receives an exclusive slice of the domain
with no shared inputs. The step body must be a single morphism that takes all
state inputs and per-step inputs together and returns a tuple of next states.

*Resolution (implemented).* `_finalize_iter` groups all tensors sharing the same
iteration axis uid and dispatches to `_finalize_iter_group`, which emits one
`Scan(n_states=k)` entry with a synthetic `__coupled_G_H` lhs name. Each
state's step is a separate morphism (built by the existing `_build_step_morph`
path) with state proxies stripped and renamed; `step_state_deps` records which
canonical state indices each morphism reads and in what order, so that per-step
input counts and domain routing are correct even when a step morphism is
independent of some other states in the group. `ConstructedScan._run_loop_coupled`
applies Jacobi semantics: every step module sees the OLD states before any is
replaced, and the forward method returns a tuple of history tensors (one per
state) that `ConstructedComposed` propagates via its existing `to_tuple`
handling. Known limitation: downstream TL equations that consume only a subset
of coupled outputs cannot be chained in the same `TL` session.

**`torch.compile` transparency.** A Python `for` loop in `ConstructedScan._run_loop`
is not FX-graph-transparent. `torch.compile` will attempt to trace through it,
either unrolling (safe for small `N`, produces large graphs for large `N`) or
treating it as a loop (graph break).

*Resolution (implemented).* `_run_loop` is wrapped with `torch._dynamo.disable`
to produce a predictable graph break rather than unbounded loop unrolling. For
affine recurrences, `_finalize_iter` runs `_recognize_affine` before morphism
construction and stores a `ScanAffine` on the `Scan` term when the recurrence
is **affine in the state**: at most one additive term contains the state tensor,
that term's operator is `Identity` (no nonlinearity), and the state factor
appears exactly once. `ConstructedScan` detects `_has_affine` and takes the
`_assoc_scan_forward` path, which uses `torch._higher_order_ops.associative_scan`
with `combine_mode='generic'` (required on CPU; `'pointwise'` is not used).

`_recognize_affine` runs in `_finalize_iter()` on the raw `RHSExpression` or
`SumExpr`. It partitions additive terms by whether any factor carries
`name == state_name`; rejects if more than one state-bearing term exists or the
operator is non-identity. If affine, it extracts: (i) the `A_l` sub-expression
— built with `lhs_indices = step_out + state_in_axes` so that contracted state
axes survive as free output axes rather than being summed away; (ii) the `b_l`
sub-expression — all remaining terms, or `None` if absent. `state_in_axes` is
filtered against `step_out_uids` to exclude tiled (free) axes that coincidentally
appear in both state and output. The result is stored as `Scan.affine`; position
lists `a_positions` / `b_positions` index into the ordered non-state step inputs
to route the correct tensors to `A_module` and `b_module`.

**Recurrence axis contamination in the shared context.** The `TL` instance
carries a single `_ctx` that accumulates axis unifications across all
assignments. If the recurrence axis `l` is unified with another axis by an
unrelated equation in the same `TL` — for example, because a non-iterative
tensor is also indexed by `l` and gets its axes merged in the shared context —
then the step body extraction pass may inherit that unification when it calls
`_ctx.apply()` on the stripped factors. This can distort size information or
merge stripped axes with the recurrence axis in ways the step body did not
intend.

*Resolution (implemented).* Step body extraction creates a filtered copy of `_ctx` by
removing every equivalence class that contains the recurrence axis uid:
`step_ctx = _ctx.without(l.uid)`. The stripped factors are built using
`step_ctx` rather than `_ctx`, preventing `l` from participating in step-body
unification. Non-iterative equations that used `l` as a free axis had their
entries built with the full `_ctx` before filtering and are unaffected.
`Context.without(uid)` is a helper defined in `data_structure/Term.py` that
returns a new `Context` containing only the groups from the original that do not
include the given uid.

**Composition with the base case morphism.** The full `Scan` pipeline must
evaluate the base case morphism (`X → H_0`) and then run the recurrence
(`(H_0, Grad) → H_sequence`). In Br, this could require a `Composed` or
`ProductOfMorphisms` wrapper that routes `X` through the base case and passes
`Grad` directly to `Scan`.

*Resolution (implemented).* Rather than an external wrapper, the `base`
morphism is stored directly in `Scan` and evaluated inside
`ConstructedScan.forward`. All caller inputs are passed together as `*xs`;
`ConstructedScan` splits at `n_base` (the number of base-case inputs, inferred
from `base_module._caller_positions` when the base module has pre-materialised
Iverson buffers). This avoids external `ProductOfMorphisms` routing plumbing
at the cost of making `ConstructedScan` responsible for the split.

### 6.6 Visual rendering

`Block` is rendered by `BlockBox` as a padded rectangle with an optional title
annotation and `[` `]` brackets annotated with the repetition count. `Scan`
cannot reuse this convention unchanged because two things differ:

1. *The step body's wire types do not match `Scan`'s outer wire types.* The
   step body diagram shows wires of type `([a, S], [b, X])` on the left and
   `[a, S]` on the right. `Scan`'s outer left wires are `([a, S], [b, X⊗L])`
   and its outer right wire is `[a, S⊗L']`. The L and L' axes exist only at
   the `Scan` level — they do not appear inside the step body box. A `ScanBox`
   must render the outer L and L' axis labels on the enclosing bracket, not on
   the inner wires.

2. *The state feedback is the defining visual content.* In `Block`, the wires
   pass straight through because `dom = cod`. In `Scan`, the state wire is
   carried from the step body's right side back to its left side at the next
   step — a feedback loop. This needs a distinct visual marker: a curved or
   looped arrow on the state wire inside the bracket, absent from `Block`.

A `ScanBox` should therefore draw the same padded rectangle and bracket
notation as `BlockBox` (reusing `block_padding`, `draw_left_bracket`,
`draw_right_bracket`), augment the bracket annotation with the step count N
and the recurrence axis label L, and add a feedback arrow on the state wire
within the bracket to visually distinguish it from `Block`.

---

## 7. Diagrammatic Notation: a String-Diagram Realization of the ScanBox

§6.6 specifies the `ScanBox` rendering rules in prose. This section gives the figures
for those rules (drawn in Mermaid in the string-diagram grammar of
[weavesWiresMorphisms.pdf](weavesWiresMorphisms.pdf)) and applies them to the
depth-recursive MLP of
[dsl_examples.md § Example 4](../docs/dsl_examples.md#example-4--deep-mlp). Consistent
with [Appendix A.4](#a4-interaction-with-weaves-and-reindexings), the recurrence axis
is drawn as **feedback**, never as a tiling weave.

### 7.1 Why a scan diagram is not a broadcast

Every string diagram in [weavesWiresMorphisms.pdf](weavesWiresMorphisms.pdf) is an
acyclic, left-to-right DAG of boxes and wires. A scan is the first construct with a
genuine sequential dependency, and §6.6 together with A.4 pin down exactly what the
diagram must show that a broadcast does not:

- **The recurrence axis `l` is not a tiling weave** (A.4, input side). The per-step
  input slice is read by a *dynamic* element access `⟨l|` that advances each
  iteration, not by a fixed affine reindexing. So `l` cannot be drawn as the paper's
  broadcast tile; the per-step weights `W` are a *sequence indexed by `l`*, sliced
  dynamically — not parallel independent tiles.
- **The output history is sequential** (A.4, output side). The history axis `L'` is
  created by the iteration; it has no counterpart in the step body's signature and so
  cannot be a weave at all.
- **Feedback is the defining content** (§6.6). Because `dom ≠ cod` and the state is
  carried output→input, the state wire loops back — a curved feedback arrow absent
  from `Block`.

The cycle is therefore **quarantined inside the labelled `ScanBox`**, so the outer
diagram stays the acyclic DAG the construction-rule algebra (§2) requires. The
recurrence axis `l` and step count `N` annotate the box bracket, not the inner wires
(§6.6, pt 1). Following the Appendix, `l`'s axis/object form is written `L` (with
`|L| = N`) and the length-`N+1` state-history axis is `L'`.

The feedback is categorically a **trace** — independent structure beyond the tiling
and reindexing that St/Br provide. A reindexing acts on already-materialised data,
but the state history is produced one slice at a time and never exists as a weave to
reindex; and the carry violates the batch-lift independence that *defines* a
broadcast (A.2). `Scan` is therefore a new construction rule (A.3, A.5), and the
carry is drawn as a feedback loop — distinct from the reindexing hexagons that mark
slices (§7.2).

### 7.2 The rolled ScanBox

The figure uses the paper's string-diagram grammar throughout; only the carry is new.

```mermaid
flowchart LR
    %% weaves grammar: wires carry axis types; named morphisms are solid boxes;
    %% slices/reindexings are hexagons; inputs are left-pointing element flags;
    %% the Scan is a Block (teal backdrop + title). Carry = the one non-standard trace.
    x>"⟨x∣"] -- "q, d₀" --> Win["L_Wᵢₙ"]

    subgraph SCAN["Scan · l · N = 3"]
        direction LR
        Wsrc(["W"]) -- "d_h, d_h ⊗ l" --> SLW{{"⟨l∣"}}
        SLW -. "d_h, d_h" .-> MUL["W[l] ·"]
        MUL == "q, d_h" ==> RELU["relu"]
        RELU == "q, d_h" ==> CARRY(("⟲"))
        CARRY == "carry" ==> MUL
    end

    Win == "q, d_h" ==> MUL
    RELU == "q, d_h ⊗ L′" ==> SEL{{"⟨·,3∣"}}
    SEL -- "q, d_h" --> Wout["L_Wₒᵤₜ"]
    Wout -- "q, c" --> SM[/"softmax ▷"\]
    SM -- "q, c." --> y(["y"])

    classDef block fill:#d6f5ee,stroke:#2a9d8f,stroke-width:1px;
    class SCAN block;
    classDef carry fill:#ffe0e0,stroke:#cc0000,stroke-width:2px;
    class CARRY carry;
    classDef morph fill:#cfd0ff,stroke:#333,stroke-width:1px;
    class Win,MUL,RELU,Wout morph;
    classDef pict fill:#cfe8ff,stroke:#333;
    class SM pict;
    classDef io fill:#eef,stroke:#333;
    class x,y,Wsrc io;
```

Every mark is a primitive from [weavesWiresMorphisms.pdf](weavesWiresMorphisms.pdf)
except the carry:

| Mark | Meaning | Weaves PDF |
| --- | --- | --- |
| labelled wire (`q, d_h`) | object — an axis / shape (a weave) | §3.1; Fig 1 |
| solid box (`L_Wᵢₙ`, `W[l] ·`, `relu`) | named morphism (operation) | §3.1, *Morphisms* |
| left flag `⟨x∣` | element (a `1 → X` value) | §3.2.1, *Elements* |
| hexagon `⟨l∣`, `⟨·,3∣` | reindexing, incl. a slice | Def 10; Figs 2–3 |
| triangle `softmax ▷` | operation pictogram | §4.2; Fig 10 |
| teal backdrop + title | **Block** — repeated structure | §3.1, *Blocks* |
| ⟲ feedback (red) | the carry — a **trace** | *new* — §7.1 |

Reading it: `⟨x∣` is the input element; `L_Wᵢₙ` projects `(q, d₀) → (q, d_h)` and
injects the carry at `l = 0`. Inside the **Block**, the weight sequence `W : (d_h,
d_h) ⊗ l` is sliced by the reindexing `⟨l∣` (dynamic, A.4 — dotted) and fed to the
linear morphism `W[l] ·`, then `relu`; the **⟲ carry** returns the state for the
next step. The block emits the full history over `L′` (`scanl`, §6.4); the hexagon
`⟨·,3∣` selects the final slice (a reindexing on the materialised history), which
feeds `L_Wₒᵤₜ` and the `softmax ▷` pictogram to give `(q, c.)`.

### 7.3 Unrolled semantics (fixed N)

For a statically known `N`, the scan unrolls into a `Composed` of `N` steps — the one
expressibility the appendix grants without a new primitive ("Expressible via
`Composed` for fixed N", A.5). The rolled box (the definition) and this chain (its
fixed-`N` meaning) denote the same module.

```mermaid
flowchart LR
    x>"⟨x∣"] -- "q, d₀" --> Win["L_Wᵢₙ"]
    Win == "q, d_h" ==> M0["W[0] ·"]
    M0 == "q, d_h" ==> R0["relu"]
    R0 == "q, d_h" ==> M1["W[1] ·"]
    M1 == "q, d_h" ==> R1["relu"]
    R1 == "q, d_h" ==> M2["W[2] ·"]
    M2 == "q, d_h" ==> R2["relu"]
    R2 == "q, d_h" ==> Wout["L_Wₒᵤₜ"]
    Wout -- "q, c" --> SM[/"softmax ▷"\]
    SM -- "q, c." --> y(["y"])
    W0(["W[0]"]) -. "d_h, d_h" .-> M0
    W1(["W[1]"]) -. "d_h, d_h" .-> M1
    W2(["W[2]"]) -. "d_h, d_h" .-> M2

    classDef morph fill:#cfd0ff,stroke:#333,stroke-width:1px;
    class Win,M0,R0,M1,R1,M2,R2,Wout morph;
    classDef pict fill:#cfe8ff,stroke:#333;
    class SM pict;
    classDef io fill:#eef,stroke:#333;
    class x,y,W0,W1,W2 io;
```

With no carry to close, the scan is an ordinary horizontal **composition** of step
morphisms (thick wires still mark the state spine `(q, d_h)`); each step's weight
slice enters from below. Weight-tying versus independent weights is read straight off
the figure: a single shared `W` feeding every step is tied; distinct `W[0..2]` (as here)
are independent.

### 7.4 Correspondence to the compiled module

The rolled box maps 1:1 onto the structure §6.4 compiles for Example 4:

```mermaid
flowchart LR
    subgraph TC["ConstructedThreadedComposed"]
        direction LR
        A["L_Wᵢₙ<br/>Linear 100→64"]
        subgraph CS["ConstructedScan"]
            direction TB
            BM["base_module<br/>(l=0 injection)"]
            SM["step_module<br/>· W[l] · ▸ relu ▸"]
        end
        B["L_Wₒᵤₜ ▸ softmax<br/>Linear 64→10"]
        A --> CS --> B
    end
```

- **`ScanBox` ↔ `ConstructedScan`**,
- **base injection ↔ `base_module`**,
- **box interior (step + feedback) ↔ `step_module`**.

The diagram, the recurrence equations (§3), and the IR all share the same three
top-level pieces.

### 7.5 Conventions

- **Carry-wire styling.** The carry is drawn as a **thick (bold) wire**; colour is
  reserved for dense diagrams. It is the one wire that must never be mistaken for
  ordinary data flow, and the bold form reads without colour.
- **Output.** The box **emits the full history** `h[q, d_h, 0:N]` over `L'` (`Scan`
  returns `scanl`, §6.4); a downstream **reindexing** extracts the final slice when
  only that is needed. The history axis `L'` annotates the bracket, never an inner
  wire (A.4: `L'` is not a weave). Final-slice selection is a reindexing (a hexagon
  on the materialised history), separate from the scan itself.

When later layers consume every state (e.g. attention over recurrence depth), the
reindexing is dropped and the full `L'` history wire feeds downstream directly.

### 7.6 Implementation status in tsncd — and what remains

A first `ScanBox` renderer is implemented in tsncd. It reuses `BlockBox` wholesale
(teal backdrop + title + bracket) and adds exactly one new mark — the **carry arc**
(`CategoryRenderer.ts`, dispatched on `instanceof cat.Scan`; the `Scan` term lives in
`ProductCategory.ts`). The live render of Example 4 is in
[dsl_examples.md § 4](../docs/dsl_examples.md#example-4--deep-mlp). All changes are
additive and gated on `instanceof Scan`, so non-scan diagrams render byte-identically
(verified).

> [!IMPORTANT]
> **What is done, and what remains for a *general, robust* scan renderer.**
>
> **Working today** — single-state (`n_states = 1`), non-affine scans with an
> *arbitrary step body*. The body is delegated to the existing `display_category`, so
> any step morphism renders (verified across three different bodies: `Linear▸ReLU`, a
> contraction `Σ`, and an additive `Σ + Σ` step). The Block scaffold and carry are
> body-agnostic.
>
> **Not yet general — three concrete gaps:**
>
> 1. **Affine scans don't serialise.** `ScanAffine` (the associative-scan
>    decomposition) is a plain dataclass, not a `Term`, so `export_to_json` throws on
>    any recurrence the compiler recognises as affine (linear RNN, EMA, gradient
>    descent…). *Fix (small):* make `ScanAffine` a `Term`, or null `affine` in a
>    visualisation-only export.
> 2. **Coupled scans (`n_states > 1`) render only the first state.** The `ScanBox`
>    `body` getter returns `steps[0]`, so the other states' step morphisms — and their
>    carries — are dropped. *Fix (moderate):* stack all step morphisms in the box and
>    draw one carry per state, driven by `n_states` / `step_state_deps`.
> 3. **The carry is a generic over-the-box arc**, routed off the box rectangle rather
>    than wired to the state's codomain→domain anchors. Fine for one state; imprecise
>    when the step has extra per-step inputs and structurally wrong for multiple
>    states. *Fix (localised):* attach the carry to the actual state anchors (needs a
>    reliable "which wire is the state" rule).
>
> Until #1–#3 land, treat the renderer as a **single-state, non-affine** ScanBox.

---

## Appendix: Categorical Status of `Scan` in Br

### A.1 The core axioms `Scan` satisfies

Let `a` and `b` be Br datatypes (e.g. `Reals()`), let `S, X ∈ Ob(St)` be
shapes, and let `step: ([a, S], [b, X]) → [a, S]` be the **step morphism** —
a Br morphism taking a state of type `[a, S]` and a per-step input of type
`[b, X]` and producing an updated state of the same type. Let `N ≥ 1` be the
step count and `L` the **recurrence axis** with `|L| = N`; let `L'` be a
fresh axis with `|L'| = N+1`. The input sequence stacks the `N` per-step
inputs along `L`, giving type `[b, X⊗L]`. In the running example,
`a = b = ℝ`, `S = X = (i,)`, and `step` is the update
`H[i] ↦ H[i] + lr · Grad[i]`.

`Scan(step, N)` has:

- domain `([a, S], [b, X⊗L])` — initial state and full input sequence
- codomain `[a, S⊗L']` — full state history over `N+1` positions

Both are valid Br objects — products of typed arrays — so `Scan` is a
well-typed morphism.

The core product-category axioms hold:

**Identity and associativity.** `Scan` is a construction rule with definite
`dom()` and `cod()`. It composes with other Br morphisms before and after,
and sequential composition is associative because the semantics are a pure
function. ✓

**Bifunctoriality** `(f;g) ⊗ (h;k) = (f⊗h);(g⊗k)`. `Scan` operates on a
disjoint wire bundle from `h`, so the monoidal product interleaves with
composition correctly. ✓

**Elemental property.** Elements of `Scan`'s domain and codomain are tuples
of array values; the element-determination property is unaffected. ✓

`Scan` therefore fits the ProdCategory grammar as a new construction rule:

```python
type ProdCategory[L, M: Morphism] = (
    M
    | Rearrangement[L]
    | Composed[L, ProdCategory[L, M]]
    | ProductOfMorphisms[L, ProdCategory[L, M]]
    | Block[L, ProdCategory[L, M]]
    | Scan[L, ProdCategory[L, M]]   # new
)
```

### A.2 The structural property `Scan` fails — the batch lift independence axiom

Br carries a key structural axiom for the batch lift (theory.md Eq. 3):

$$[f, P] \mathbin{;} [Y, q] = [X, q] \mathbin{;} f \qquad \forall\, q : \mathbf{1} \to P$$

This says slicing the output at position `q` after applying `f` equals slicing
the input at `q` first, then applying `f`. It guarantees position-independence:
every `q ∈ P` is computed from only its own input slice.

`Scan` cannot satisfy this for L because of the sequential dependency. A
`Broadcasted` morphism with degree L would require (Def. 11, Eq. 2) that the
output at each position l depends only on the l-th input slices — the
morphism factors into `|El(L)|` independent copies of the step body, with no
data flow between positions. But `H_l` depends on `H_0, …, H_{l-1}`: the
output at position l is a function of ALL preceding positions, not just the
l-th input slice. `Scan` is therefore not expressible as a `Broadcasted` root
morphism with L as part of the degree.

For any **fixed** N, `Scan(step, N)` can be unrolled into a `Composed` of N
distinct `Broadcasted` morphisms — one per step, each using a different fixed
reindexing into the Grad sequence. What cannot be expressed within the
existing grammar is `Scan` for variable N: no single construction rule
produces a morphism parameterised by N without unrolling.

Batch lifting over axes **orthogonal** to L is well-defined: `[Scan(step,N), P]`
for a batch axis P independent of L runs the scan independently for each `p ∈ P`,
and each such run satisfies the independence axiom over P. The sequential
dependency is within each trajectory over L, not between batch positions
(see §A.4 for the weave/reindexing treatment).

### A.3 Why `Block` does not already cover `Scan`

`Block` is a degenerate fold — `foldl f z (repeat () N)` where the step
ignores its second argument, giving `dom(step) = cod(step) = State`. This is
why theory.md describes `Block` as "transparent to the categorical semantics":
it passes `dom()` and `cod()` through from the body unchanged.

`Scan` changes the type. `dom(step) = ([a, S], [b, X])` and `cod(step) = [a, S]`,
whereas `dom(Scan(step, N)) = ([a, S], [b, X⊗L])` and
`cod(Scan(step, N)) = [a, S⊗L']`. The input sequence `[b, X⊗L]` and the
output history `[a, S⊗L']` have no counterpart in `step`'s signature. This
type change is the formal signal that `Scan` is a new combinator, not a
special case of `Block`.

One might try to express `Scan` as `Composed([step_0, step_1, …, step_{N-1}])`
— one copy of `step` per time step, each with a different fixed Grad slice
baked in via a `StrideMorphism` reindexing. This works for fixed N and gives
O(N) morphisms, but it loses the structural invariant that all steps are the
same operation. `Scan(step, N)` is the compact, structurally-aware form.

### A.4 Interaction with weaves and reindexings

The weave/reindexing apparatus is designed for static, uniform, parallel
broadcasting: a tiling axis is one where every degree position `p ∈ P` receives
its own independent input slice via a fixed affine reindexing `η: P → Q`,
with no data flow between positions. The recurrence axis L fits neither
requirement.

**Within the step body.** `step` is an ordinary Br morphism with its own
weaves and reindexings. The state axes S and per-step input axes X are target
axes for the step body; any additional axes the step body broadcasts over are
tiling axes with the usual static reindexings. Nothing here is special to
`Scan`.

**The recurrence axis L.** L cannot appear in any weave or reindexing of
`Scan` itself, for two reasons:

- *Input side.* At step `l`, `Scan` reads `[b, X⊗L]` via the element
  morphism `⟨l|: 1 → L`. This is a dynamic reindexing — `l` advances each
  iteration. A `Broadcasted` reindexing must be a fixed affine map determined
  at morphism-construction time; it cannot depend on the running step index.

- *Output side.* The output history axis L' is created by the sequential
  iteration — states are accumulated one at a time. A tiling axis in a
  `Broadcasted` morphism is pre-existing in both input and output weaves and
  drives which slice is written where. L' has no counterpart in `step`'s
  signature and so cannot be described by a weave at all.

L therefore sits outside the weave/reindexing framework and is managed by
`Scan`'s sequential iteration logic.

**Batch lifting over orthogonal axes P.** For a batch axis P independent of
L, `[Scan(step, N), P]` applies the scan independently for each `p ∈ P`.
The step body becomes `[step, P]`, and the standard machinery applies: each
input weave gains TILED entries for the P axes, and each reindexing extends
from `η_i: P_step → Q_i` to `id_P ⊗ η_i: P⊗P_step → P⊗Q_i`. The
independence axiom holds over P — different batch positions have no data flow
between them.

| Axis | Role in weave | Reindexing |
| --- | --- | --- |
| State axes S | Target in step body | — |
| Per-step input axes X | Target in step body | — |
| Recurrence axis L | Not a weave axis — sequential dependency | Not a reindexing — dynamic element access |
| Output history axis L' | Not a weave axis — created by iteration | — |
| Orthogonal batch axis P | TILED in lifted step body | `id_P ⊗ η_i` |

### A.5 Summary

| Property | Holds for `Scan`? |
| --- | --- |
| Well-typed domain and codomain in Br | ✓ |
| Identity and associativity axioms | ✓ |
| Bifunctoriality with `ProductOfMorphisms` | ✓ |
| Elemental determination property | ✓ |
| Batch lift independence over recurrence axis L | ✗ |
| Expressible as `Broadcasted` root morphism | ✗ |
| Expressible via `Composed` for fixed N | ✓ (O(N) unrolling) |
| Expressible via single construction rule for variable N | ✗ |
| Batch lift well-defined over orthogonal axes P | ✓ |
| Recurrence axis L expressible as tiling axis | ✗ |
| Recurrence axis L accessible via static reindexing | ✗ |
| Step body weaves and reindexings unaffected | ✓ |
| Orthogonal batch axes extend via standard weave/reindexing | ✓ |

`Scan` satisfies the core categorical axioms and can be added to the
ProdCategory grammar as a valid construction rule. It cannot be expressed as
a `Broadcasted` root morphism for two independent reasons: on the input side,
the element access `⟨l| : 1 → L` is dynamic and cannot be expressed as a
fixed affine reindexing; on the output side, the state history `[a, S⊗L']`
is accumulated sequentially, so output at position l depends on all preceding
positions, violating the independence property (Def. 11). For any fixed N it
can be unrolled into a `Composed` of N operations, but this loses the uniform
N-parameterised structure and cannot serve as a general recurrence primitive.
`Scan` must therefore be a new construction rule rather than derived from
existing Br operations.
