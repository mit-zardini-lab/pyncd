from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import data_structure.Term as fd
import data_structure.BroadcastedCategory as bc
import data_structure.ProductCategory as pc
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc
import data_structure.Operators as ops
from data_structure.TensorLogic import TensorEquation, TensorProgram
from data_structure.TensorExpr import (
    TensorRef, IversonBinOp, IversonUnaryOp,
    IversonExpr,
    ieq, imul, iabs, inot,
)
from data_structure.AxisAnnotations import NormAxis, NatAxis


# ---------------------------------------------------------------------------
# Tensor-level declarations
# ---------------------------------------------------------------------------

class TensorKind(Enum):
    TENSOR    = 'tensor'
    PREDICATE = 'predicate'  # Bool-typed; axes no longer promoted to PredAxis
    LINEAR    = 'linear'     # weight of a Linear/affine layer (ops.Linear box)


@dataclass
class TensorDeclaration:
    """Positional shape declaration for a named tensor."""
    kind:  TensorKind
    shape: tuple[sc.RawAxis, ...]
    bias:  bool = False                          # LINEAR only: affine (Wx + b) vs linear


# ---------------------------------------------------------------------------
# Scan morphism types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanAffine:
    """Affine decomposition of a recurrence: H[l+1] = A_l · H[l] + b_l.

    A_morphism: produces A_l for a single step; built with lhs = step_out +
                state_in_axes so the state contraction axis k survives as a
                free output axis.  None means identity.
    b_morphism: produces b_l for a single step.  None means zero bias.
    state_in_axes: index tuple of the state factor (e.g. (k,)), used by
                   ConstructedScan to build the combine and final-apply einsum.
    a_positions: indices into the Scan's non-state step_xs for A_module inputs.
    b_positions: indices into the Scan's non-state step_xs for b_module inputs.
    """
    A_morphism: object | None
    b_morphism: object | None
    state_in_axes: tuple[sc.RawAxis, ...]
    a_positions: tuple[int, ...]
    b_positions: tuple[int, ...]


@dataclass(frozen=True)
class Scan(fd.Term):
    """Iterative scan over N steps.

    For n_states == 1 (uncoupled):
      step: single step-body morphism: (H_state, *non_state_inputs) -> H_next
      base: single base-case morphism: (*init_inputs) -> H_0
    For n_states > 1 (coupled, Jacobi-style):
      step: tuple of step morphisms, one per state, each taking
            (*all_states_in_canonical_order, *own_per_step_inputs) -> state_k_next
      base: tuple of base morphisms, one per state
      affine is always None for coupled groups.

    N:    number of recurrence steps (not counting the base case).
    axis: the recurrence RawAxis (l).
    affine: ScanAffine decomposition for associative_scan lowering, or None.
    n_states: number of coupled state tensors (1 = uncoupled).
    """
    step: object
    base: object
    N: nm.Numeric
    axis: sc.RawAxis
    affine: ScanAffine | None = None
    n_states: int = 1
    # For coupled scans: which state indices (into the canonical states tuple)
    # each step morphism expects, in the order they appear in its domain.
    # Length == n_states; empty tuple means "all states in order 0..n-1".
    step_state_deps: tuple[tuple[int, ...], ...] = ()
    # Uncoupled only: the axis position of the recurrence axis l within each
    # per-step input, as the user declared it (e.g. W[l, ...] -> 0).  The runtime
    # moves that axis to last before slicing.  Empty = assume l-last for all.
    step_x_l_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class Slice(fd.Term):
    """Select fixed integer coordinates from an input tensor — a constant index
    read such as ``X[..., 3]`` (design note papers/index_arithmetic.md, P1).

    positions[k] is the axis position in the input's own index order and
    indices[k] the coordinate to select there.  Realised at runtime as repeated
    ``tensor.select(pos, idx)`` (a slice = an St element reindexing).
    """
    positions: tuple[int, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class Reindex(fd.Term):
    """Affine gather (P2): ``X_g[out_axes] = X[ rows ]``.

    For input-axis ``p`` of ``X``, ``rows[p] = (const_p, ((k, coeff), ...))`` gives
    the read coordinate ``const_p + Σ coeff·out_axes[k]``.  Realises a non-iteration
    affine index read — shift/stride/dilation/convolution/diagonal — as an St affine
    reindexing.  ``out_axes`` are the fan-out axes; their sizes give the output shape.
    ``in_axes`` are the storage axes of the original tensor, used for display.
    """
    in_axes: fd.Prod[sc.RawAxis]
    out_axes: fd.Prod[sc.RawAxis]
    rows: fd.Prod  # tuple[tuple[int, tuple[tuple[int, int], ...]], ...]


@dataclass(frozen=True)
class Scatter(fd.Term):
    """Affine scatter (P2): ``out[ rows ] = value``, zero-filled elsewhere.

    The value is indexed by ``in_axes``; for output slot ``s``,
    ``rows[s] = (const_s, ((k, coeff), ...))`` places it at output coordinate
    ``const_s + Σ coeff·in_axes[k]``.  ``out_shape`` is the (inferred) output size
    per slot.  Uncovered coordinates take ``fill`` (zero default, P3).  An
    injective map writes directly; a non-injective map (overlapping writes) is
    rejected at build time unless ``reduce`` declares a combine ('sum'), in which
    case overlapping writes accumulate (design note §4.4, P3).
    """
    in_axes: fd.Prod[sc.RawAxis]
    out_shape: fd.Prod[int]
    rows: fd.Prod
    fill: float = 0.0
    reduce: str | None = None


def _scatter_injective(in_sizes: tuple[int, ...], rows: tuple) -> bool:
    """True iff the affine scatter map is injective over the value domain.

    Enumerates the (small) input coordinate box and checks that no two distinct
    input coordinates map to the same output tuple.  ``rows[s] = (const, ((k, c), ...))``
    gives output slot ``s`` as ``const + Σ c·coord[k]``.  Used to reject overlapping
    writes at build time unless a reduction is declared (design note §4.4, P3).
    """
    from itertools import product
    seen: set[tuple[int, ...]] = set()
    for coord in product(*[range(s) for s in in_sizes]):
        out = tuple(
            const + sum(c * coord[k] for k, c in terms)
            for const, terms in rows
        )
        if out in seen:
            return False
        seen.add(out)
    return True


def _topo_sort_entries(entries):
    """Stably order entries so every producer precedes its consumers.

    An entry consumes a name iff that name appears in its input_names and is
    produced by some entry's lhs_name.  Ties are broken by original index, so a
    list that is already in producer-before-consumer order is returned unchanged
    (this keeps non-scan programs byte-identical).  Needed because _finalize_iter
    appends Scan entries last, after any downstream consumer of the scan output.
    """
    import heapq
    n = len(entries)
    produced: dict = {}
    for idx, (lhs, _, _, _) in enumerate(entries):
        if lhs is not None:
            produced[lhs] = idx  # last writer wins
    deps = []
    for (_, _, _, input_names) in entries:
        d = {produced[name] for name in input_names
             if name is not None and name in produced}
        deps.append(d)
    indeg = [len(d) for d in deps]
    consumers: list[list[int]] = [[] for _ in range(n)]
    for i, d in enumerate(deps):
        for j in d:
            if j != i:
                consumers[j].append(i)
    ready = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for c in consumers[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(ready, c)
    if len(order) != n:
        return entries  # cycle (unexpected) — leave as-is
    return [entries[i] for i in order]


def _live_entries(entries):
    """Return only entries reachable from the last output, preserving order.

    Unreachable entries — those whose lhs_name is never referenced by any
    downstream input_names — are silently dropped.  Entries with lhs_name=None
    (side-effect nodes) are always kept.
    """
    if not entries:
        return entries

    # Map each produced name to the index of the entry that produces it.
    name_to_idx = {}
    for i, (lhs, _, _, _) in enumerate(entries):
        if lhs is not None:
            name_to_idx[lhs] = i

    # BFS backward from the final entry.
    reachable = set()
    queue = [len(entries) - 1]
    while queue:
        idx = queue.pop()
        if idx in reachable:
            continue
        reachable.add(idx)
        _, _, _, input_names = entries[idx]
        for name in input_names:
            if name is not None and name in name_to_idx:
                queue.append(name_to_idx[name])

    # Entries with lhs_name=None are always live.
    for i, (lhs, _, _, _) in enumerate(entries):
        if lhs is None:
            reachable.add(i)

    # Return in original declaration order.
    return [e for i, e in enumerate(entries) if i in reachable]


def _external_names_from_value(value, exclude):
    """Return unique tensor names from a stripped RHS value, skipping excluded names.

    'value' is the output of _strip_iter_axis_from_value: an RHSExpression or
    SumExpr where state-proxy factors come first (canonical order) followed by
    external-tensor factors.  We collect the external names in order of first
    appearance, skipping any name in 'exclude'.

    'exclude' is a set of DynamicName objects (state tensor names and their proxies).
    """
    seen = set()
    result = []
    if hasattr(value, 'terms'):          # SumExpr: flatten all terms
        factors = [f for term in value.terms for f in term.factors]
    else:                                 # RHSExpression
        factors = value.factors
    for f in factors:
        if hasattr(f, 'name') and hasattr(f, 'indices'):   # IndexedTensor
            if f.name not in exclude and f.name not in seen:
                seen.add(f.name)
                result.append(f.name)
    return tuple(result)


def _drop_norm_invariant_terms(value, norm_uid):
    """Drop additive terms that are constant along the normalization axis.

    Applies the identity softmax(f + c) = softmax(f) when c's factors have
    no index with UID matching norm_uid.

    'value' is an RHSExpression or SumExpr.
    'norm_uid' is the UID of the NormAxis from the equation's LHS.

    Returns the value unchanged if it has only one term or if all terms
    depend on the norm axis.  Returns a single RHSExpression (not a SumExpr)
    if all but one term are dropped.
    """
    if not isinstance(value, SumExpr):
        return value

    kept = []
    for term in value.terms:
        # Check whether any IndexedTensor factor in this term is indexed by
        # the norm axis (i.e., has norm_uid among its index UIDs).
        depends_on_norm = any(
            isinstance(f, IndexedTensor) and any(ax.uid == norm_uid for ax in f.indices)
            for f in term.factors
        )
        if depends_on_norm:
            kept.append(term)

    # Nothing to drop — return original unchanged.
    if len(kept) == len(value.terms):
        return value

    # All terms dropped (shouldn't happen in practice) — return original.
    if len(kept) == 0:
        return value

    # Exactly one term remains — return it as a plain RHSExpression, but
    # preserve any nonlinearity operator from the SumExpr wrapper.
    if len(kept) == 1:
        term = kept[0]
        if value.operator is not None:
            return RHSExpression(term.factors, value.operator)
        return term

    return SumExpr(kept, value.operator)


# ---------------------------------------------------------------------------
# TL registry
# ---------------------------------------------------------------------------

class TL:
    """Registry for tensor logic equations.

    Access tensor names as attributes to build equations, then call
    to_equation() or to_morphism() to extract the result.

        tl = TL()
        i, j, k = axes('i j k')
        tl.Y[i,j] = tl.W[i,k] * tl.X[k,j]
        eq = tl.to_equation()

    Optionally declare tensors before use to attach kind and shape metadata:

        d, d_ff = real_axis('d', 512), real_axis('d_ff', 2048)
        tl.W_in.tensor(d_ff, d)

    Addition is supported on the RHS for composing elementwise sums:

        tl.Out[i] = tl.A[i] + tl.B[i]
        tl.Out[i] = relu(tl.H[i,k] * tl.W[k]) + tl.Bias[i]
    """

    def __init__(self) -> None:
        self._declarations: dict[str, TensorDeclaration] = {}
        self._ctx: fd.Context = fd.Context()
        self._name_to_axes: dict[fd.DynamicName, tuple[sc.RawAxis, ...]] = {}
        # Each entry: (lhs_name, morphism, output_axes, input_names)
        # input_names[i] is the tensor name for domain slot i (None for Iverson).
        # Scan/iteration entries use () for input_names (threading not needed there).
        self._entries: list[tuple[fd.DynamicName | None, object, tuple[sc.RawAxis, ...], tuple]] = []
        # Iterative tensor support: keyed by tensor name string
        self._pending_iter: dict[str, dict] = {}       # {'base': (...), 'recur': (...)}
        self._iteration_axes: dict[str, sc.RawAxis] = {}  # name → l axis
        # Names declared iterative explicitly via .iteration_axis()/.recur().  A
        # tensor in _iteration_axes but NOT here was registered only implicitly by
        # an `axis+int` LHS, which at finalize is reclassified to a scatter unless it
        # also carries a base case (design note papers/index_arithmetic.md §4, P3).
        self._explicit_iter: set[str] = set()
        # Per-output scatter options declared via TensorProxy.scatter(): the fill
        # for uncovered coordinates and the combine for overlapping writes (P3).
        self._scatter_opts: dict[str, dict] = {}       # name → {'fill', 'reduce'}
        self._iter_finalized: bool = False

    @property
    def _equations(self) -> list[TensorEquation]:
        """Backward-compat view: extract TensorEquation from each Broadcasted entry.

        Raises if any entry was produced by a SumExpr (+ expression), since those
        are represented as Composed morphisms, not single TensorEquations.
        """
        result = []
        for _, morph, _, _ in self._entries:
            if isinstance(morph, bc.Broadcasted) and isinstance(morph.operator, TensorEquation):
                result.append(morph.operator)
            else:
                raise ValueError(
                    "_equations is only available when all entries are TensorEquation-backed "
                    "(no + expressions). Use to_morphism() for programs that include addition."
                )
        return result

    def __getattr__(self, name: str) -> TensorProxy:
        if name.startswith('_'):
            raise AttributeError(name)
        return TensorProxy(name, self)

    def _register_declaration(self, name: str, decl: TensorDeclaration) -> None:
        self._declarations[name] = decl

    def _array_datatypes(self) -> dict[fd.DynamicName, bc.Datatype]:
        """Map predicate tensor names to bc.Bool() for bc_signature / to_morphism."""
        return {
            fd.DynamicName(name): bc.Bool()
            for name, decl in self._declarations.items()
            if decl.kind is TensorKind.PREDICATE
        }

    def _build_rhs_morphism(
        self,
        lhs_name: fd.DynamicName | None,
        lhs_indices: tuple[sc.RawAxis, ...],
        value: RHSExpression,
    ) -> tuple[bc.Broadcasted, tuple[sc.RawAxis, ...], tuple[fd.DynamicName | None, ...]]:
        """Build a Broadcasted morphism for one RHSExpression.

        Unifies any intermediate tensor axes with prior output axes via _ctx,
        then applies _ctx and calls bc_signature() to produce the morphism.
        Returns (morphism, canonical_output_axes, input_names) where
        input_names[i] is the tensor name for domain slot i (None for Iverson).
        """
        from data_structure.TensorExpr import IversonBinOp, IversonUnaryOp
        from data_structure.TensorLogic import _iverson_is_materializable
        eq = TensorEquation(
            lhs_name=lhs_name,
            lhs_indices=lhs_indices,
            rhs=tuple(
                TensorRef(f.name, f.indices) if isinstance(f, IndexedTensor) else f
                for f in value.factors
            ),
            operator=value.operator,
        )
        seen: set[fd.DynamicName | None] = set()
        for factor in eq.rhs:
            if not isinstance(factor, TensorRef):
                continue
            tensor_name = factor.name
            if tensor_name in self._name_to_axes and tensor_name not in seen:
                seen.add(tensor_name)
                prior_axes = self._name_to_axes[tensor_name]
                if len(prior_axes) == len(factor.axes):
                    for prior_ax, eq_ax in zip(prior_axes, factor.axes):
                        self._ctx.append_iter((prior_ax, eq_ax))
        applied_eq = self._ctx.apply(eq)
        linear = self._linear_factor(applied_eq)
        if linear is not None:
            return self._build_linear_morphism(applied_eq, linear, value.operator)
        br = applied_eq.bc_signature(array_datatypes=self._array_datatypes())
        domain_factors = [
            f for f in applied_eq.rhs
            if not (isinstance(f, (IversonBinOp, IversonUnaryOp))
                    and _iverson_is_materializable(f))
        ]
        input_names = tuple(
            f.name if isinstance(f, TensorRef) else None
            for f in domain_factors
        )
        return br, applied_eq.lhs_indices, input_names

    def _linear_factor(self, eq) -> tuple[int, TensorDeclaration] | None:
        """Return (rhs index, declaration) of a LINEAR-declared weight factor, if any."""
        for i, f in enumerate(eq.rhs):
            if not isinstance(f, TensorRef):
                continue
            key = f.name.body if isinstance(f.name.body, str) else None
            decl = self._declarations.get(key) if key else None
            if decl is not None and decl.kind is TensorKind.LINEAR:
                return (i, decl)
        return None

    def _build_linear_morphism(self, eq, linear, operator):
        """Compile `W * X` (W LINEAR-declared) to an ops.Linear box.

        The weight's declared in_axes/out_axes (each possibly multi-axis) give the
        input/output feature blocks; the weight becomes the layer's internal
        parameter (dropped from the caller inputs) and the activation X is the sole
        input.  Builds the Linear Broadcasted directly — feature axes concrete,
        batch axes TILED — then composes the nonlinearity.
        """
        i, decl = linear
        W = eq.rhs[i]
        acts = [f for j, f in enumerate(eq.rhs) if j != i and isinstance(f, TensorRef)]
        if len(acts) != 1:
            raise ValueError(
                f"a LINEAR-declared weight must multiply exactly one activation; "
                f"got {len(acts)} other tensor factors"
            )
        X = acts[0]
        wname = W.name.body if isinstance(W.name.body, str) else None
        linear_op = ops.Linear(
            name=fd.DynamicName(
                body='L',
                subscript=fd.DynamicName.from_str(wname),
                settings=fd.DynamicNameSettings(bold=True),
            ),
            bias=decl.bias,
        )
        # Infer feature blocks from the equation: weight axes shared with the
        # activation are contracted (in); weight axes shared with the lhs are
        # produced (out).  Both sets may be multi-axis.
        w_uids   = {ax.uid for ax in W.axes}
        x_uids   = {ax.uid for ax in X.axes}
        lhs_uids = {ax.uid for ax in eq.lhs_indices}
        in_uids  = w_uids & x_uids
        out_uids = w_uids & lhs_uids
        if in_uids & out_uids:
            raise ValueError(
                f"LINEAR weight {wname!r} has axes that appear in both the "
                f"activation and the lhs; cannot infer in/out role"
            )
        if in_uids | out_uids != w_uids:
            raise ValueError(
                f"LINEAR weight {wname!r} has axes that appear in neither the "
                f"activation nor the lhs"
            )
        # Build the Linear Broadcasted directly: feature axes concrete, batch
        # (activation axes that are not input features) TILED.  No identity
        # passthrough is composed in front, so no spurious einsum (Σ) box.
        degree = tuple(ax for ax in X.axes if ax.uid not in in_uids)
        pos = {ax.uid: i for i, ax in enumerate(degree)}
        in_shape  = tuple(bc.WeaveMode.TILED if ax.uid not in in_uids else ax
                          for ax in X.axes)
        out_shape = tuple(bc.WeaveMode.TILED if ax.uid not in out_uids else ax
                          for ax in eq.lhs_indices)
        morph = bc.Broadcasted(
            operator=linear_op,
            input_weaves=(bc.Weave(bc.Reals(), in_shape),),
            output_weaves=(bc.Weave(bc.Reals(), out_shape),),
            reindexings=(pc.Rearrangement(
                mapping=tuple(pos[ax.uid] for ax in X.axes if ax.uid in pos),
                _dom=degree,
            ),),
        )
        if operator is not None and not isinstance(operator, ops.Identity):
            if isinstance(operator, ops.SoftMax):
                morph = morph @ ops.SoftMax.template()
            elif isinstance(operator, ops.Normalize):
                morph = morph @ ops.Normalize.template()
            elif isinstance(operator, ops.Elementwise):
                morph = morph @ type(operator).template()
            else:
                raise NotImplementedError(
                    f"nonlinearity {operator!r} not supported after a Linear layer"
                )
        return morph, eq.lhs_indices, (X.name,)

    def _build_sum_morphism(
        self,
        lhs_name: fd.DynamicName | None,
        lhs_indices: tuple[sc.RawAxis, ...],
        value: SumExpr,
    ) -> tuple[pc.Composed, tuple[sc.RawAxis, ...], tuple[fd.DynamicName | None, ...]]:
        """Build a Composed morphism for a SumExpr.

        Compiles each additive term to a Broadcasted via _build_rhs_morphism,
        splits any nonlinearity so ConstructedTensorEquation sees only einsum
        steps, wraps them in a ProductOfMorphisms, then appends an AdditionOp
        Broadcasted that adds all term outputs together.
        Returns (composed_morphism, canonical_output_axes, input_names).
        """
        from data_structure.TensorLogic import _split_nonlinearity
        term_morphisms: list = []
        all_input_names: list[fd.DynamicName | None] = []
        degree: tuple[sc.RawAxis, ...] = lhs_indices
        for term in value.terms:
            br, degree, term_names = self._build_rhs_morphism(None, lhs_indices, term)
            all_input_names.extend(term_names)
            term_morphisms.append(
                _split_nonlinearity(br.operator, array_datatypes=self._array_datatypes())
            )

        n = len(term_morphisms)
        tiled_shape = tuple(bc.WeaveMode.TILED for _ in degree)
        add_weave = bc.Weave(bc.Reals(), tiled_shape)
        add_reidx = pc.Rearrangement(tuple(range(len(degree))), degree)
        add_br = bc.Broadcasted(
            operator=ops.AdditionOp(),
            input_weaves=tuple(add_weave for _ in range(n)),
            output_weaves=(add_weave,),
            reindexings=tuple(add_reidx for _ in range(n)),
        )
        prod = pc.ProductOfMorphisms(content=tuple(term_morphisms))
        result: pc.Composed | bc.Broadcasted = pc.Composed(content=(prod, add_br))
        op = value.operator
        if op is not None and not isinstance(op, ops.Identity):
            if isinstance(op, ops.SoftMax):
                result = result @ ops.SoftMax.template()
            elif isinstance(op, ops.Normalize):
                result = result @ ops.Normalize.template()
            elif isinstance(op, ops.Elementwise):
                result = result @ type(op).template()   # preserve the concrete op (e.g. ReLU)
            else:
                raise NotImplementedError(
                    f'No base morphism registered for SumExpr nonlinearity {op!r}'
                )
        return result, degree, tuple(all_input_names)

    def _check_const_bound(self, name: fd.DynamicName, pos: int, value: int) -> None:
        """Static bounds check for a constant index ``T[..., value, ...]``.

        The valid range is ``0 ≤ value < |axis|``; for the iteration axis of an
        iterative tensor the readable extent is the materialised history ``N+1``.
        Skipped silently when the axis size is unknown (e.g. an undeclared input);
        the runtime ``torch.select`` will catch any residual out-of-range access.
        """
        shape = self._name_to_axes.get(name)
        if shape is None:
            key = name.body if isinstance(name.body, str) else None
            decl = self._declarations.get(key) if key else None
            shape = decl.shape if decl is not None else None
        if shape is None or pos >= len(shape):
            return
        ax = shape[pos]
        key = name.body if isinstance(name.body, str) else None
        it = self._iteration_axes.get(key) if key else None
        if it is not None and ax.uid == it.uid and isinstance(it._size, nm.Integer):
            size = it._size._value + 1            # history L' = N+1
        elif isinstance(ax._size, nm.Integer):
            size = ax._size._value
        else:
            return
        if not (0 <= value < size):
            raise ValueError(
                f"constant index {value} out of range for axis {pos} of "
                f"'{key}' (size {size}) in a slice read"
            )

    def _build_reindex_factor(self, f):
        """Synthesise a `Reindex` (affine gather) entry for a factor with affine
        index slots, returning a plain read of the fresh fan-out intermediate.
        All slots become affine rows: a bare axis -> coeff 1; an int -> a constant;
        an affine expr -> its normal form (papers/index_arithmetic_p2_plan.md, 2a)."""
        from data_structure.TensorExpr import affine_normal_form
        fan_out: list[sc.RawAxis] = []

        def fan_idx(ax: sc.RawAxis) -> int:
            for k, a in enumerate(fan_out):
                if a.uid == ax.uid:
                    return k
            fan_out.append(ax)
            return len(fan_out) - 1

        rows: list = []
        for idx in f.indices:
            if isinstance(idx, sc.RawAxis):
                rows.append((0, ((fan_idx(idx), 1),)))
            elif isinstance(idx, int):
                rows.append((idx, ()))
            else:
                const, coeffs = affine_normal_form(idx)
                rows.append((const, tuple((fan_idx(ax), c) for ax, c in coeffs.items())))
        fan_axes = tuple(fan_out)
        self._check_gather_bounds(f.name, rows, fan_axes)
        n = getattr(self, '_slice_counter', 0)
        self._slice_counter = n + 1
        base = f.name.body if isinstance(f.name.body, str) else 'T'
        fresh = fd.DynamicName(f'{base}__g{n}')
        key = f.name.body if isinstance(f.name.body, str) else None
        decl = self._declarations.get(key) if key else None
        in_axes = decl.shape if decl is not None else ()
        reindex = Reindex(in_axes=in_axes, out_axes=fan_axes, rows=tuple(rows))
        self._entries.append((fresh, reindex, fan_axes, (f.name,)))
        self._name_to_axes[fresh] = fan_axes
        return IndexedTensor(fresh, fan_axes)

    def _check_gather_bounds(self, name, rows, fan_axes) -> None:
        """Range inference (P3): reject an affine gather whose image leaves a
        declared input axis.  Each input slot ``p`` reads ``const + Σ c·aₖ``; with
        the fan-out axis sizes the read interval ``[lo, hi]`` is known, so when the
        indexed input has a concrete declared size we require ``0 ≤ lo`` and
        ``hi < size``.  Skipped when the input or a fan-out size is unknown (the
        runtime advanced-index then catches any residual out-of-range read)."""
        key = name.body if isinstance(name.body, str) else None
        decl = self._declarations.get(key) if key else None
        if decl is None:
            return
        fan_sizes = [
            ax._size._value if isinstance(ax._size, nm.Integer) else None
            for ax in fan_axes
        ]
        for p, (const, terms) in enumerate(rows):
            if p >= len(decl.shape):
                continue
            ax = decl.shape[p]
            if not isinstance(ax._size, nm.Integer):
                continue
            size = ax._size._value
            if any(fan_sizes[k] is None for k, _ in terms):
                continue
            lo = const + sum(c * (0 if c >= 0 else fan_sizes[k] - 1) for k, c in terms)
            hi = const + sum(c * (fan_sizes[k] - 1 if c >= 0 else 0) for k, c in terms)
            if lo < 0 or hi >= size:
                raise ValueError(
                    f"affine gather reads positions [{lo}, {hi}] out of range for "
                    f"axis {p} of '{key}' (size {size})"
                )

    def _extract_const_slices(self, value, iter_axis: sc.RawAxis | None = None):
        """Rewrite RHS factors carrying literal-int index slots into slice entries.

        A constant index read ``T[..., c, ...]`` becomes a fresh intermediate
        ``T__sN`` produced by a `Slice` morphism (selecting coordinate ``c``),
        and the factor is rewritten to read ``T__sN`` over its free (non-constant)
        axes.  The slice entry is appended to ``self._entries`` and threads through
        the normal ThreadedComposed machinery (papers/index_arithmetic_plan.md, P1).
        Returns the value with const-indexed factors replaced.

        When ``iter_axis`` is supplied (scan-step pre-pass, P3), a factor whose
        affine slot references that axis is left untouched — the recurrence /
        look-back arithmetic stays on the Scan path (gate, design note §4); only
        gathers over non-iteration axes are rewritten.
        """
        from data_structure.TensorExpr import affine_normal_form

        def _affine_uses_iter(f) -> bool:
            if iter_axis is None:
                return False
            for idx in f.indices:
                if isinstance(idx, (IversonBinOp, IversonUnaryOp)):
                    try:
                        _, coeffs = affine_normal_form(idx)
                    except ValueError:
                        continue
                    if any(ax.uid == iter_axis.uid for ax in coeffs):
                        return True
            return False

        def rewrite_factor(f):
            if not isinstance(f, IndexedTensor):
                return f
            if _affine_uses_iter(f):
                return f                                   # recurrence/look-back: Scan path
            has_affine = any(isinstance(idx, (IversonBinOp, IversonUnaryOp))
                             for idx in f.indices)
            has_const = any(isinstance(idx, int) for idx in f.indices)
            if has_affine:
                return self._build_reindex_factor(f)       # P2 affine gather
            if not has_const:
                return f                                   # all plain axes
            # P1: pure constant index -> Slice (drop the constant axes)
            const_slots = [(p, idx) for p, idx in enumerate(f.indices)
                           if isinstance(idx, int)]
            for p, v in const_slots:
                self._check_const_bound(f.name, p, v)
            free = tuple(idx for idx in f.indices if not isinstance(idx, int))
            n = getattr(self, '_slice_counter', 0)
            self._slice_counter = n + 1
            base = f.name.body if isinstance(f.name.body, str) else 'T'
            fresh = fd.DynamicName(f'{base}__s{n}')
            slice_morph = Slice(
                positions=tuple(p for p, _ in const_slots),
                indices=tuple(v for _, v in const_slots),
            )
            self._entries.append((fresh, slice_morph, free, (f.name,)))
            self._name_to_axes[fresh] = free
            return IndexedTensor(fresh, free)

        if isinstance(value, RHSExpression):
            return RHSExpression(
                [rewrite_factor(f) for f in value.factors], value.operator)
        if isinstance(value, SumExpr):
            return SumExpr(
                [RHSExpression([rewrite_factor(f) for f in t.factors], t.operator)
                 for t in value.terms],
                value.operator,
            )
        return value

    def _register_entry(
        self,
        lhs_name: fd.DynamicName | None,
        lhs_indices: tuple[sc.RawAxis, ...],
        value: RHSExpression | SumExpr,
    ) -> None:
        """Build and store one morphism entry."""
        # P2 affine scatter: an LHS slot that is an affine expression (e.g. Y[2*i])
        # writes the body to affine output positions.  (Pure axis+int LHS is the
        # recurrence syntax, routed earlier in __setitem__.)
        if any(isinstance(ix, (IversonBinOp, IversonUnaryOp)) for ix in lhs_indices):
            self._register_scatter(lhs_name, lhs_indices, value)
            return

        # Propagate concrete sizes from declaration to LHS axes.
        name_str = lhs_name.body if lhs_name and isinstance(lhs_name.body, str) else None
        decl = self._declarations.get(name_str) if name_str else None
        if decl is not None:
            for decl_ax, lhs_ax in zip(decl.shape, lhs_indices):
                if type(decl_ax) is type(lhs_ax):
                    self._ctx.append_iter((decl_ax, lhs_ax))

        # Pre-pass: drop additive terms constant along the normalisation axis.
        # softmax(f + c) = softmax(f) when c has no factor indexed by NormAxis.
        norm_uid = None
        for ax in lhs_indices:
            if isinstance(ax, NormAxis):
                norm_uid = ax.uid
                break
        if norm_uid is not None:
            value = _drop_norm_invariant_terms(value, norm_uid)

        # Pre-pass: rewrite RHS factors with literal-int (constant) index slots
        # into slice entries, so the builders below see only plain-axis reads.
        value = self._extract_const_slices(value)

        if isinstance(value, RHSExpression):
            morph, out_axes, input_names = self._build_rhs_morphism(lhs_name, lhs_indices, value)
        else:
            morph, out_axes, input_names = self._build_sum_morphism(lhs_name, lhs_indices, value)

        self._entries.append((lhs_name, morph, out_axes, input_names))
        if lhs_name is not None:
            self._name_to_axes[lhs_name] = out_axes

    def _register_scatter(self, lhs_name, lhs_indices, value) -> None:
        """P2 affine scatter: build the body over the free (value) axes, then write
        it to affine output positions (`out[const+Σcₖaₖ] = value`, zero-filled).
        Injective + non-negative-coefficient only (P2); see index_arithmetic_p2_plan.md.
        """
        from data_structure.TensorExpr import affine_normal_form
        # Free axes = axes appearing across the LHS slots (value's axes), in order.
        free: list[sc.RawAxis] = []
        def fidx(ax):
            for k, a in enumerate(free):
                if a.uid == ax.uid:
                    return k
            free.append(ax)
            return len(free) - 1
        for ix in lhs_indices:
            if isinstance(ix, sc.RawAxis):
                fidx(ix)
            else:
                _, coeffs = affine_normal_form(ix)
                for ax in coeffs:
                    fidx(ax)
        free_axes = tuple(free)

        # Build the body value over the free axes as a normal intermediate entry
        # (this runs the RHS const/affine gather pre-pass too).
        n = getattr(self, '_slice_counter', 0)
        self._slice_counter = n + 1
        base = lhs_name.body if lhs_name and isinstance(lhs_name.body, str) else 'Y'
        val_name = fd.DynamicName(f'{base}__v{n}')
        self._register_entry(val_name, free_axes, value)
        value_axes = self._name_to_axes[val_name]

        def vidx(ax):
            for k, a in enumerate(value_axes):
                if a.uid == ax.uid:
                    return k
            raise ValueError("scatter LHS axis is not produced by the body")

        name_str = lhs_name.body if lhs_name and isinstance(lhs_name.body, str) else None
        decl = self._declarations.get(name_str) if name_str else None

        out_shape: list[int] = []
        rows: list = []
        for s, ix in enumerate(lhs_indices):
            const, coeffs = (0, {ix: 1}) if isinstance(ix, sc.RawAxis) else affine_normal_form(ix)
            if const < 0:
                raise ValueError("affine scatter requires a non-negative constant (P2)")
            terms = []
            maxpos = const
            for ax, c in coeffs.items():
                if c < 0:
                    raise ValueError("affine scatter requires non-negative coefficients (P2)")
                if not isinstance(ax._size, nm.Integer):
                    raise ValueError(f"affine scatter needs a concrete size for axis {ax}")
                terms.append((vidx(ax), c))
                maxpos += c * (ax._size._value - 1)
            # Range inference (P3): a declared output axis fixes the size (so the
            # tail beyond the image is filled, not truncated) and the image must
            # fit; otherwise infer the tight size maxpos+1 from the map.
            declared = (
                decl.shape[s]._size._value
                if decl is not None and s < len(decl.shape)
                and isinstance(decl.shape[s]._size, nm.Integer)
                else None
            )
            if declared is not None:
                if maxpos >= declared:
                    raise ValueError(
                        f"affine scatter for '{name_str}' writes position {maxpos} "
                        f"out of range for declared output axis {s} (size {declared})"
                    )
                out_shape.append(declared)
            else:
                out_shape.append(maxpos + 1)
            rows.append((const, tuple(terms)))

        # Coverage / conflict (P3): a non-injective affine map (overlapping writes)
        # is rejected unless the user declared a reduction; the fill covers the
        # coordinates the image misses.
        opts = self._scatter_opts.get(name_str, {}) if name_str else {}
        fill = float(opts.get('fill', 0.0))
        reduce = opts.get('reduce', None)
        in_sizes = tuple(
            ax._size._value for ax in value_axes
            if isinstance(ax._size, nm.Integer)
        )
        if reduce is None and len(in_sizes) == len(value_axes):
            if not _scatter_injective(in_sizes, tuple(rows)):
                raise ValueError(
                    f"scatter for '{name_str}' has overlapping writes (non-injective "
                    f"affine map); declare a combine via .scatter(reduce='sum')"
                )

        scat = Scatter(
            in_axes=value_axes, out_shape=tuple(out_shape), rows=tuple(rows),
            fill=fill, reduce=reduce,
        )
        self._entries.append((lhs_name, scat, value_axes, (val_name,)))
        if lhs_name is not None:
            self._name_to_axes[lhs_name] = value_axes

    # ------------------------------------------------------------------
    # Iteration support
    # ------------------------------------------------------------------

    def _build_rhs_morphism_with_ctx(
        self,
        lhs_name: fd.DynamicName | None,
        lhs_indices: tuple[sc.RawAxis, ...],
        value: RHSExpression,
        ctx: fd.Context,
    ) -> tuple[bc.Broadcasted, tuple[sc.RawAxis, ...]]:
        """Like _build_rhs_morphism but uses a supplied ctx without mutating self._ctx."""
        eq = TensorEquation(
            lhs_name=lhs_name,
            lhs_indices=lhs_indices,
            rhs=tuple(
                TensorRef(f.name, f.indices) if isinstance(f, IndexedTensor) else f
                for f in value.factors
            ),
            operator=value.operator,
        )
        applied_eq = ctx.apply(eq)
        # Honor .linear() weights inside a scan step/base: build an ops.Linear box
        # (the weight becomes an internal parameter, dropped from caller inputs),
        # mirroring the non-ctx _build_rhs_morphism path.
        linear = self._linear_factor(applied_eq)
        if linear is not None:
            morph, out_axes, _ = self._build_linear_morphism(applied_eq, linear, value.operator)
            return morph, out_axes
        br = applied_eq.bc_signature(array_datatypes=self._array_datatypes())
        return br, applied_eq.lhs_indices

    def _build_sum_morphism_with_ctx(
        self,
        lhs_name: fd.DynamicName | None,
        lhs_indices: tuple[sc.RawAxis, ...],
        value: SumExpr,
        ctx: fd.Context,
    ) -> tuple[pc.Composed, tuple[sc.RawAxis, ...]]:
        """Like _build_sum_morphism but uses a supplied ctx without mutating self._ctx."""
        term_morphisms: list[bc.Broadcasted] = []
        degree: tuple[sc.RawAxis, ...] = lhs_indices
        for term in value.terms:
            br, degree = self._build_rhs_morphism_with_ctx(None, lhs_indices, term, ctx)
            term_morphisms.append(br)
        n = len(term_morphisms)
        tiled_shape = tuple(bc.WeaveMode.TILED for _ in degree)
        add_weave = bc.Weave(bc.Reals(), tiled_shape)
        add_reidx = pc.Rearrangement(tuple(range(len(degree))), degree)
        add_br = bc.Broadcasted(
            operator=ops.AdditionOp(),
            input_weaves=tuple(add_weave for _ in range(n)),
            output_weaves=(add_weave,),
            reindexings=tuple(add_reidx for _ in range(n)),
        )
        prod = pc.ProductOfMorphisms(content=tuple(term_morphisms))
        composed = pc.Composed(content=(prod, add_br))
        return composed, degree

    def _build_step_morph(
        self,
        lhs_name: fd.DynamicName | None,
        lhs_indices: tuple[sc.RawAxis, ...],
        value: RHSExpression | SumExpr,
        ctx: fd.Context,
    ) -> object:
        from data_structure.TensorLogic import TensorEquation, _split_nonlinearity
        if isinstance(value, RHSExpression):
            morph, _ = self._build_rhs_morphism_with_ctx(lhs_name, lhs_indices, value, ctx)
        else:
            morph, _ = self._build_sum_morphism_with_ctx(lhs_name, lhs_indices, value, ctx)
        # Split any nonlinearity (e.g. relu) out of the einsum so the scan
        # step/base compiles: ConstructedTensorEquation only handles Identity
        # operators, with the activation composed as a separate Broadcasted.
        if isinstance(morph, bc.Broadcasted) and isinstance(morph.operator, TensorEquation):
            return _split_nonlinearity(morph.operator, array_datatypes=self._array_datatypes())
        return morph

    def _strip_iter_axis_from_value(
        self,
        value: RHSExpression | SumExpr,
        l: sc.RawAxis,
        state_proxies: dict[fd.DynamicName, fd.DynamicName],
    ) -> RHSExpression | SumExpr:
        """Strip l from all factor indices; rename state tensors to proxies.

        state_proxies maps each state tensor's DynamicName to its proxy name.
        State factors are placed first in canonical (sorted-by-name) order,
        regardless of their order of appearance in the RHS factors.
        """
        canonical_order = sorted(state_proxies, key=lambda dn: dn.body or '')

        def strip(f: IndexedTensor | IversonBinOp | IversonUnaryOp):
            if isinstance(f, IndexedTensor):
                stripped = tuple(ax for ax in f.indices if ax.uid != l.uid)
                name = state_proxies.get(f.name, f.name)
                return IndexedTensor(name, stripped)
            return f

        def reorder(factors: list) -> list:
            by_name = {
                f.name: f for f in factors
                if isinstance(f, IndexedTensor) and f.name in state_proxies
            }
            ordered_state = [by_name[n] for n in canonical_order if n in by_name]
            other = [
                f for f in factors
                if not (isinstance(f, IndexedTensor) and f.name in state_proxies)
            ]
            return [strip(f) for f in ordered_state] + [strip(f) for f in other]

        if isinstance(value, RHSExpression):
            return RHSExpression(reorder(value.factors), value.operator)
        return SumExpr([RHSExpression(reorder(t.factors), t.operator) for t in value.terms])

    def _recognize_affine(
        self,
        recur_value: RHSExpression | SumExpr,
        state_name_str: str,
        step_out: tuple[sc.RawAxis, ...],
        l: sc.RawAxis,
    ) -> ScanAffine | None:
        """Try to decompose the recurrence as H[l+1] = A_l·H[l] + b_l.

        Returns ScanAffine if the recurrence is affine in the state, else None.
        """
        state_name_dn = fd.DynamicName(state_name_str)
        step_ctx = self._ctx.without(l.uid)

        def has_state(term: RHSExpression) -> bool:
            return any(
                isinstance(f, IndexedTensor) and f.name == state_name_dn
                for f in term.factors
            )

        def strip_l(f: IndexedTensor | IversonBinOp | IversonUnaryOp):
            if isinstance(f, IndexedTensor):
                return IndexedTensor(
                    f.name, tuple(ax for ax in f.indices if ax.uid != l.uid)
                )
            return f

        terms = recur_value.terms if isinstance(recur_value, SumExpr) else [recur_value]
        state_terms = [t for t in terms if has_state(t)]
        bias_terms  = [t for t in terms if not has_state(t)]

        # Build ordered list of non-state factors across all terms (mirrors step module
        # input order produced by _strip_iter_axis_from_value).
        non_state_keys: list[tuple] = []
        for term in terms:
            for f in term.factors:
                if isinstance(f, IndexedTensor) and f.name != state_name_dn:
                    non_state_keys.append((f.name, tuple(ax.uid for ax in f.indices)))

        if len(state_terms) > 1:
            return None

        if len(state_terms) == 0:
            # Pure bias: A = identity, b = full RHS
            b_value: RHSExpression | SumExpr = (
                bias_terms[0] if len(bias_terms) == 1 else SumExpr(bias_terms)
            )
            b_morph = self._build_step_morph(None, step_out, b_value, step_ctx)
            b_positions = tuple(range(len(non_state_keys)))
            return ScanAffine(
                A_morphism=None, b_morphism=b_morph,
                state_in_axes=(), a_positions=(), b_positions=b_positions,
            )

        state_term = state_terms[0]
        if not isinstance(state_term.operator, ops.Identity):
            return None  # nonlinearity wraps state

        state_factors = [
            f for f in state_term.factors
            if isinstance(f, IndexedTensor) and f.name == state_name_dn
        ]
        if len(state_factors) != 1:
            return None  # state appears more than once (quadratic)

        state_factor = state_factors[0]
        # state_in_axes: indices of the state factor that are CONTRACTED (absent
        # from step_out).  Free axes shared with step_out are not matrix axes.
        step_out_uids = {ax.uid for ax in step_out}
        state_in_axes = tuple(
            ax for ax in state_factor.indices
            if ax.uid != l.uid and ax.uid not in step_out_uids
        )

        a_raw = [
            f for f in state_term.factors
            if not (isinstance(f, IndexedTensor) and f.name == state_name_dn)
        ]
        a_keys = [
            (f.name, tuple(ax.uid for ax in f.indices))
            for f in a_raw if isinstance(f, IndexedTensor)
        ]
        a_positions = tuple(non_state_keys.index(k) for k in a_keys)

        A_factors_stripped = [strip_l(f) for f in a_raw]
        A_lhs = step_out + state_in_axes
        A_morph: object | None
        if A_factors_stripped:
            A_morph, _ = self._build_rhs_morphism_with_ctx(
                None, A_lhs, RHSExpression(A_factors_stripped, ops.Identity()), step_ctx
            )
        else:
            A_morph = None  # pure identity (no coefficient)

        b_morph: object | None = None
        b_positions: tuple[int, ...] = ()
        if bias_terms:
            b_keys = [
                (f.name, tuple(ax.uid for ax in f.indices))
                for t in bias_terms
                for f in t.factors if isinstance(f, IndexedTensor)
            ]
            b_positions = tuple(non_state_keys.index(k) for k in b_keys)
            stripped_bias = [
                RHSExpression([strip_l(f) for f in t.factors], t.operator)
                for t in bias_terms
            ]
            b_value_expr: RHSExpression | SumExpr = (
                stripped_bias[0] if len(stripped_bias) == 1 else SumExpr(stripped_bias)
            )
            b_morph = self._build_step_morph(None, step_out, b_value_expr, step_ctx)

        return ScanAffine(
            A_morphism=A_morph, b_morphism=b_morph,
            state_in_axes=state_in_axes,
            a_positions=a_positions, b_positions=b_positions,
        )

    def _register_iter_recur(
        self,
        name_str: str,
        l: sc.RawAxis,
        non_iter_indices: tuple[sc.RawAxis, ...],
        iter_dim: int,
        value: RHSExpression | SumExpr,
        raw_lhs_indices: tuple = (),
    ) -> None:
        entry = self._pending_iter.setdefault(name_str, {})
        if 'recur' in entry:
            raise ValueError(
                f"Iterative tensor '{name_str}' already has a recurrence equation."
            )
        self._iteration_axes[name_str] = l
        entry['recur'] = (non_iter_indices, value)
        # Stash the raw `axis+int` LHS so finalize can reclassify this as a scatter
        # if the tensor turns out to carry no base case / iteration declaration.
        entry['recur_raw_lhs'] = (raw_lhs_indices, value)
        # Register full shape (with l in original position) for cross-equation unification
        full: list[sc.RawAxis] = list(non_iter_indices)
        full.insert(iter_dim, l)
        self._name_to_axes[fd.DynamicName(name_str)] = tuple(full)

    def _register_iter_base(
        self,
        name_str: str,
        non_int_indices: tuple[sc.RawAxis, ...],
        value: RHSExpression | SumExpr,
        base_literal: int,
    ) -> None:
        entry = self._pending_iter.setdefault(name_str, {})
        if 'base' in entry:
            raise ValueError(
                f"Iterative tensor '{name_str}' already has a base case equation."
            )
        entry['base'] = (non_int_indices, value, base_literal)

    def _is_pure_state_recurrence(
        self,
        recur_value: RHSExpression | SumExpr,
        state_name_dn: fd.DynamicName,
        l: sc.RawAxis,
    ) -> bool:
        """Return True iff no non-state factor is indexed by l (pure-state recurrence)."""
        factors = (
            recur_value.factors if isinstance(recur_value, RHSExpression)
            else [f for t in recur_value.terms for f in t.factors]
        )
        for f in factors:
            if isinstance(f, IndexedTensor) and f.name != state_name_dn:
                if any(ax.uid == l.uid for ax in f.indices):
                    return False
        return True

    def _check_no_lnext_on_rhs(
        self,
        recur_value: RHSExpression | SumExpr,
        l: sc.RawAxis,
        name_str: str,
    ) -> None:
        """Check 4.4: reject l+1 appearing in any RHS factor's indices."""
        factors = (
            recur_value.factors if isinstance(recur_value, RHSExpression)
            else [f for t in recur_value.terms for f in t.factors]
        )
        for f in factors:
            if not isinstance(f, IndexedTensor):
                continue
            for idx in f.indices:
                if (
                    isinstance(idx, IversonBinOp) and idx.op == '+'
                    and isinstance(idx.lhs, sc.RawAxis)
                    and idx.lhs.uid == l.uid
                    and isinstance(idx.rhs, int)
                ):
                    raise ValueError(
                        f"Causality violation in '{name_str}': "
                        f"'{f.name}' is indexed by l+{idx.rhs} on the RHS."
                    )

    def _finalize_iter_group(self, names: list[str], l: sc.RawAxis) -> None:
        """Build a coupled Scan for tensors sharing the same iteration axis l.

        Emits one Scan(n_states=k) entry with a synthetic lhs_name.  The Scan's
        step is a tuple of morphisms (one per state, in sorted(names) order); each
        step morphism takes (*all_state_proxies_in_canonical_order, *own_per_step).
        forward() returns tuple[Tensor, ...] with outputs in the same order.

        Known limitation: downstream TL equations within the same TL session that
        consume only a subset of the coupled outputs cannot be chained correctly,
        because ConstructedComposed feeds all outputs of this Scan to the next
        module.  Coupled scans should be the terminal computation in a TL session.
        """
        n = len(names)
        # Check 4.1: all axes must have a concrete size (same l, so one check).
        if not isinstance(l._size, nm.Integer):
            raise ValueError(
                f"Iteration axis for coupled group ({', '.join(names)}) has no "
                "concrete size; use real_axis('name', N) to supply a step count."
            )

        # Build per-tensor entries and validate.
        entries: list[dict] = []
        for name_str in names:
            entry = self._pending_iter.get(name_str, {})
            if 'recur' not in entry:
                raise ValueError(
                    f"Iterative tensor '{name_str}' has no recurrence equation."
                )
            if 'base' not in entry:
                raise ValueError(
                    f"Iterative tensor '{name_str}' has no base case equation."
                )
            _, _, base_literal = entry['base']
            if base_literal != 0:
                raise ValueError(
                    f"Base case for '{name_str}' is at l={base_literal}; "
                    "must be l=0."
                )
            self._check_no_lnext_on_rhs(entry['recur'][1], l, name_str)
            entries.append(entry)

        # Canonical ordering: sorted by name — enforced both in strip pass and in
        # the state tuple fed to each step module's forward().
        state_proxies = {
            fd.DynamicName(name_str): fd.DynamicName(name_str + '_state')
            for name_str in names  # names is already sorted(names) from caller
        }
        step_ctx = self._ctx.without(l.uid)

        proxy_list = list(state_proxies.values())  # G_state_dn, H_state_dn, ... in canonical order
        proxy_to_idx = {p: i for i, p in enumerate(proxy_list)}

        step_morphs: list[object] = []
        base_morphs: list[object] = []
        step_outs: list[tuple[sc.RawAxis, ...]] = []
        step_state_deps: list[tuple[int, ...]] = []

        for name_str, entry in zip(names, entries):
            step_out, recur_value = entry['recur']
            base_out, base_value, _ = entry['base']

            stripped = self._strip_iter_axis_from_value(
                recur_value, l, state_proxies
            )
            step_morphs.append(self._build_step_morph(None, step_out, stripped, step_ctx))
            base_morphs.append(self._build_step_morph(None, base_out, base_value, self._ctx))
            step_outs.append(step_out)

            # Track which states this morphism depends on, in morphism-domain order.
            # For SumExpr the domain is the concatenation of per-term domains;
            # for RHSExpression it is the factor order within the single term.
            seen_set: set[int] = set()
            dep_indices: list[int] = []
            terms = stripped.terms if isinstance(stripped, SumExpr) else [stripped]
            for term in terms:
                for f in term.factors:
                    if isinstance(f, IndexedTensor) and f.name in proxy_to_idx:
                        idx = proxy_to_idx[f.name]
                        if idx not in seen_set:
                            dep_indices.append(idx)
                            seen_set.add(idx)
            step_state_deps.append(tuple(dep_indices))

        combined_step_out: tuple[sc.RawAxis, ...] = sum(step_outs, ())
        group_lhs = fd.DynamicName('__coupled_' + '_'.join(names))
        scan = Scan(
            step=tuple(step_morphs),
            base=tuple(base_morphs),
            N=l._size,
            axis=l,
            affine=None,
            n_states=n,
            step_state_deps=tuple(step_state_deps),
        )
        self._entries.append((group_lhs, scan, combined_step_out + (l,), ()))

    def _value_reads_name(self, value, name: fd.DynamicName) -> bool:
        """True iff any factor in the RHS value reads tensor ``name`` (self-state)."""
        factors = (
            value.factors if isinstance(value, RHSExpression)
            else [f for t in value.terms for f in t.factors]
        )
        return any(
            isinstance(f, IndexedTensor) and f.name == name for f in factors
        )

    def _reclassify_offset_scatters(self) -> None:
        """Re-route ``axis+int`` LHS writes that are scatters, not recurrences.

        An ``axis+int`` LHS provisionally registered a recurrence in __setitem__.
        At finalize, a tensor with such a write but no base case, no ``recur()``
        morphism, and no explicit ``.iteration_axis()`` is an offset-scatter
        (``Y[i+1] = X[i]``), not a recurrence (design note §4, P3).  A body that
        reads the tensor itself is left as a (base-case-less) recurrence so the
        existing "no base case" error still fires.
        """
        for name_str in list(self._pending_iter.keys()):
            entry = self._pending_iter[name_str]
            if 'recur' not in entry:
                continue
            if 'base' in entry or 'recur_morphism' in entry:
                continue
            if name_str in self._explicit_iter:
                continue
            raw_lhs, value = entry['recur_raw_lhs']
            if self._value_reads_name(value, fd.DynamicName(name_str)):
                continue  # self-referential: a recurrence missing its base case
            del self._pending_iter[name_str]
            self._iteration_axes.pop(name_str, None)
            self._name_to_axes.pop(fd.DynamicName(name_str), None)
            self._register_scatter(fd.DynamicName(name_str), raw_lhs, value)

    def _finalize_iter(self) -> None:
        """Build Scan morphisms from _pending_iter and append to _entries."""
        if self._iter_finalized:
            return
        self._iter_finalized = True

        self._reclassify_offset_scatters()

        # Group tensors by iteration axis uid; coupled groups processed separately.
        axis_uid_to_names: dict[int, list[str]] = {}
        for name_str, l in self._iteration_axes.items():
            axis_uid_to_names.setdefault(id(l.uid), []).append(name_str)

        coupled_names: set[str] = set()
        for names_in_group in axis_uid_to_names.values():
            if len(names_in_group) > 1:
                canonical = sorted(names_in_group)
                l = self._iteration_axes[canonical[0]]
                self._finalize_iter_group(canonical, l)
                coupled_names.update(canonical)

        for name_str, entry in self._pending_iter.items():
            if name_str in coupled_names:
                continue  # handled by _finalize_iter_group above

            if 'recur' not in entry and 'recur_morphism' not in entry:
                raise ValueError(
                    f"Iterative tensor '{name_str}' has no recurrence equation."
                )
            if 'base' not in entry:
                raise ValueError(
                    f"Iterative tensor '{name_str}' has no base case equation."
                )

            l = self._iteration_axes[name_str]

            # Check 4.1: concrete size.
            if not isinstance(l._size, nm.Integer):
                raise ValueError(
                    f"Iteration axis for '{name_str}' has no concrete size; "
                    "use real_axis('name', N) to supply a step count."
                )

            lhs_name = fd.DynamicName(name_str)
            base_out, base_value, base_literal = entry['base']
            # P3: rewrite affine gathers over non-iteration axes in the base body
            # into top-level Reindex/Slice entries feeding the scan (gate on l).
            base_value = self._extract_const_slices(base_value, iter_axis=l)

            # Check 4.2: base case must be at l=0.
            if base_literal != 0:
                raise ValueError(
                    f"Base case for '{name_str}' is at l={base_literal}; "
                    "the iteration range starts at 0, so the base case must be l=0."
                )

            if 'recur_morphism' in entry:
                # Pre-built step morphism supplied via TensorProxy.recur().
                # Bypasses TL equation parsing; state shape is inferred from
                # the base case output axes.
                step_morph = entry['recur_morphism']
                step_out = base_out
                affine = None
                _input_names: tuple = ()
                _step_x_l_positions: tuple[int, ...] = ()
            else:
                step_out, recur_value = entry['recur']
                # Check 4.4: no l+1 on RHS.
                self._check_no_lnext_on_rhs(recur_value, l, name_str)
                # P3: rewrite affine gathers over non-iteration axes in the step
                # body into top-level Reindex/Slice entries feeding the scan.
                recur_value = self._extract_const_slices(recur_value, iter_axis=l)
                state_name_dn  = lhs_name
                state_proxy_dn = fd.DynamicName(name_str + '_state')
                step_ctx = self._ctx.without(l.uid)
                step_value = self._strip_iter_axis_from_value(
                    recur_value, l, {state_name_dn: state_proxy_dn}
                )
                step_morph = self._build_step_morph(None, step_out, step_value, step_ctx)
                affine = self._recognize_affine(recur_value, name_str, step_out, l)
                # Collect external input names for the live-pool routing table.
                # Exclude both the original state name and its proxy (both are
                # internal to the Scan loop).
                # Exclude the state name/proxy, and any .linear()-declared weight
                # (now an internal Linear parameter, not a caller input).
                _linear_names = {
                    fd.DynamicName(k) for k, d in self._declarations.items()
                    if d.kind is TensorKind.LINEAR
                }
                _exclude = {state_proxy_dn, state_name_dn} | _linear_names
                _base_external = _external_names_from_value(base_value, _exclude)
                _step_external = _external_names_from_value(step_value, _exclude)
                _input_names = _base_external + _step_external

                # Record l's position within each per-step input as the user wrote
                # it (from the un-stripped recurrence), so the runtime can move it
                # to last before slicing (l-last is the ConstructedScan contract).
                def _l_pos_in_recur(target_name):
                    factors = (
                        recur_value.factors if isinstance(recur_value, RHSExpression)
                        else [f for t in recur_value.terms for f in t.factors]
                    )
                    for f in factors:
                        if isinstance(f, IndexedTensor) and f.name == target_name:
                            for p, ax in enumerate(f.indices):
                                if not isinstance(ax, int) and getattr(ax, 'uid', None) == l.uid:
                                    return p
                    return -1
                _step_x_l_positions = tuple(
                    _l_pos_in_recur(nm_) for nm_ in _step_external)

            base_morph = self._build_step_morph(None, base_out, base_value, self._ctx)

            # Block is the correct morphism for pure-state recurrences (no per-step
            # external inputs), but requires base-case routing equivalent to Scan.
            # For now we always emit Scan and note Block as a future optimisation.
            scan = Scan(
                step=step_morph,
                base=base_morph,
                N=l._size,
                axis=l,
                affine=affine,
                step_x_l_positions=_step_x_l_positions,
            )
            self._entries.append((lhs_name, scan, step_out + (l,), _input_names))

    def _require_non_iterative(self, method: str) -> None:
        if self._pending_iter:
            raise ValueError(
                f"{method}() is not available for TL instances with iterative "
                "tensors — use to_morphism() instead."
            )

    def to_equation(self) -> TensorEquation:
        self._require_non_iterative('to_equation')
        eqs = self._equations
        if len(eqs) != 1:
            raise ValueError(f"expected exactly one equation, got {len(eqs)}")
        return eqs[0]

    def to_program(self) -> TensorProgram:
        self._require_non_iterative('to_program')
        return TensorProgram(equations=tuple(self._equations))

    def to_morphism(self):
        from data_structure.TensorLogic import TensorEquation, _split_nonlinearity

        def _compiled(morph):
            if isinstance(morph, bc.Broadcasted) and isinstance(morph.operator, TensorEquation):
                return _split_nonlinearity(morph.operator,
                                           array_datatypes=self._array_datatypes())
            return morph  # SumExpr Composed — nonlinearity already split

        self._finalize_iter()
        if not self._entries:
            raise ValueError("no equations registered")
        # Coupled Scans (n_states > 1) still produce input_names=() until Phase 4.
        # Uncoupled Scans now have proper input_names and go through ThreadedComposed.
        _has_unresolved_scan = any(
            not input_names and isinstance(morph, Scan)
            for _, morph, _, input_names in self._entries
        )
        if _has_unresolved_scan:
            morphisms = tuple(_compiled(morph) for _, morph, _, _ in self._entries)
            if len(morphisms) == 1:
                return morphisms[0]
            return pc.Composed(content=morphisms)
        entries = _live_entries(_topo_sort_entries(self._entries))
        internal_names = {lhs for lhs, _, _, _ in entries if lhs is not None}
        # Collect external tensor names in order of first appearance.
        external_order: list[fd.DynamicName] = []
        external_name_set: set[fd.DynamicName] = set()
        for _, _, _, input_names in entries:
            for name in input_names:
                if name is not None and name not in internal_names and name not in external_name_set:
                    external_order.append(name)
                    external_name_set.add(name)
        n_external = len(external_order)
        ext_idx = {name: i for i, name in enumerate(external_order)}
        produced_idx: dict[fd.DynamicName, int] = {}
        routing: list[tuple[int, ...]] = []
        for lhs_name, _, _, input_names in entries:
            route: list[int] = []
            for name in input_names:
                if name is None:
                    continue  # Iverson buffer — not a live-pool input
                if name in ext_idx:
                    route.append(ext_idx[name])
                else:
                    route.append(n_external + produced_idx[name])
            routing.append(tuple(route))
            if lhs_name is not None:
                produced_idx[lhs_name] = len(produced_idx)
        morphisms = tuple(_compiled(morph) for _, morph, _, _ in entries)
        return pc.ThreadedComposed(
            content=morphisms,
            routing=tuple(routing),
            n_external=n_external,
        )

    def bc_signature[B: bc.Datatype](
        self,
        signature: str = '',
        datatype: B = bc.Reals(),
        give_names: bool = True,
    ) -> bc.Broadcasted[B, sc.RawAxis] | pc.Composed:
        self._require_non_iterative('bc_signature')
        entries = self._entries
        if len(entries) != 1:
            raise ValueError(f"bc_signature() requires exactly one equation, got {len(entries)}")
        _, morph, _, _ = entries[0]
        from data_structure.TensorLogic import TensorEquation, _split_nonlinearity
        if not (isinstance(morph, bc.Broadcasted) and isinstance(morph.operator, TensorEquation)):
            return morph  # SumExpr Composed — terms already split by _build_sum_morphism
        eq = morph.operator
        return _split_nonlinearity(eq, datatype=datatype,
                                   array_datatypes=self._array_datatypes())


class TensorProxy:
    """Handle for a named tensor in a TL registry.

    __getitem__ returns an IndexedTensor for use on the RHS of an equation.
    __setitem__ captures a completed equation into the parent registry.
    tensor/predicate register a shape declaration.
    """

    def __init__(self, name: str, registry: TL) -> None:
        self._name = name
        self._registry = registry

    def _promote(self, indices: tuple[sc.RawAxis, ...]) -> tuple[sc.RawAxis, ...]:
        decl = self._registry._declarations.get(self._name)
        if decl is None:
            return indices
        if len(indices) != len(decl.shape):
            raise ValueError(
                f"tensor '{self._name}' declared with {len(decl.shape)} axes "
                f"but indexed with {len(indices)}"
            )
        if decl.kind is TensorKind.PREDICATE:
            return indices  # Bool datatype; axes no longer promoted to PredAxis
        return indices  # TENSOR — no promotion

    def __getitem__(self, indices: sc.RawAxis | tuple[sc.RawAxis, ...]) -> IndexedTensor:
        if not isinstance(indices, tuple):
            indices = (indices,)
        return IndexedTensor(fd.DynamicName(self._name), self._promote(indices))

    def __setitem__(
        self,
        indices: sc.RawAxis | tuple[sc.RawAxis, ...],
        value: IndexedTensor | RHSExpression | SumExpr,
    ) -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)
        if isinstance(value, IndexedTensor):
            value = RHSExpression([value], ops.Identity())

        # Detect recurrence LHS: one slot is IversonBinOp('+', RawAxis, int)
        # produced by `l + 1` via the monkey-patch in TensorExpr.py.
        iter_pos = next(
            (i for i, idx in enumerate(indices)
             if isinstance(idx, IversonBinOp) and idx.op == '+'
             and isinstance(idx.lhs, sc.RawAxis) and isinstance(idx.rhs, int)),
            None,
        )
        if iter_pos is not None:
            iter_ref = indices[iter_pos]
            l = iter_ref.lhs
            non_iter = tuple(idx for i, idx in enumerate(indices) if i != iter_pos)
            self._registry._register_iter_recur(
                self._name, l, non_iter, iter_pos, value, indices)
            return

        # Detect base-case LHS: one slot is a literal int (e.g. 0)
        int_pos = next(
            (i for i, idx in enumerate(indices) if isinstance(idx, int)),
            None,
        )
        if int_pos is not None:
            base_literal = indices[int_pos]
            non_int = tuple(idx for i, idx in enumerate(indices) if i != int_pos)
            self._registry._register_iter_base(self._name, non_int, value, base_literal)
            return

        # Normal (non-iterative) assignment
        decl = self._registry._declarations.get(self._name)
        if decl is not None and len(indices) != len(decl.shape):
            raise ValueError(
                f"tensor '{self._name}' declared with {len(decl.shape)} axes "
                f"but assigned with {len(indices)}"
            )
        self._registry._register_entry(fd.DynamicName(self._name), indices, value)

    def iteration_axis(self, l: sc.RawAxis) -> TensorProxy:
        """Declare l as the recurrence axis for this tensor.

        Registers l early so forward references in other equations (e.g.
        equations that read H[i, l] before H's own recurrence is defined)
        can find the axis.  Check 4.1 (concrete size) is enforced here.
        """
        if not isinstance(l._size, nm.Integer):
            raise ValueError(
                f"Iteration axis '{l}' has no concrete size; "
                "use real_axis('name', N)."
            )
        self._registry._iteration_axes[self._name] = l
        self._registry._explicit_iter.add(self._name)
        return self

    def recur(self, axis: sc.RawAxis, morphism: object) -> TensorProxy:
        """Register a pre-built morphism as this tensor's Scan step function.

        Use when the step function is too complex to express as a single TL
        equation — for example, a chained block built from multiple TL sessions.
        The morphism must accept the current state as its sole input and return
        the next state (no per-step external inputs; all weights are baked in).
        A base case must also be provided via tl.H[..., 0] = ...

        Example::

            l = real_axis('l', L)
            x, m = axes('x m')
            tl = TL()
            tl.H[x, m, 0] = tl.X[x, m]           # base: pass embeddings through
            tl.H.recur(l, transformer_layer())     # step: one full transformer layer
            return tl.to_morphism()
        """
        if not isinstance(axis._size, nm.Integer):
            raise ValueError(
                f"Iteration axis '{axis}' has no concrete size; "
                "use real_axis('name', N)."
            )
        self._registry._iteration_axes[self._name] = axis
        self._registry._explicit_iter.add(self._name)
        entry = self._registry._pending_iter.setdefault(self._name, {})
        entry['recur_morphism'] = morphism
        return self

    def scatter(self, *, fill: float = 0.0, reduce: str | None = None) -> TensorProxy:
        """Declare scatter coverage/conflict options for this (affine-LHS) output.

        ``fill`` is the value for output coordinates the affine map does not cover
        (zero default).  ``reduce`` declares how overlapping writes combine: the
        default ``None`` rejects a non-injective map at build time; ``'sum'``
        accumulates overlapping writes (design note papers/index_arithmetic.md §4.4).
        """
        if reduce not in (None, 'sum'):
            raise ValueError(
                f"scatter reduce must be None or 'sum', got {reduce!r}")
        self._registry._scatter_opts[self._name] = {'fill': fill, 'reduce': reduce}
        return self

    def tensor(self, *shape: sc.RawAxis) -> TensorProxy:
        """Declare this tensor with the given shape (default contraction semantics)."""
        self._registry._register_declaration(
            self._name,
            TensorDeclaration(kind=TensorKind.TENSOR, shape=shape),
        )
        return self

    def predicate(self, *shape: sc.RawAxis) -> TensorProxy:
        """Declare this tensor as Bool-typed (predicate); axes are not promoted."""
        self._registry._register_declaration(
            self._name,
            TensorDeclaration(kind=TensorKind.PREDICATE, shape=shape),
        )
        return self

    def linear(self, *shape: sc.RawAxis, bias: bool = False) -> TensorProxy:
        """Declare this tensor as a Linear/affine layer weight.

        List the weight's axes in the order they appear in the equations, e.g.

            tl.W_Q.linear(h, k, d)
            tl.Q[q, h, k] = tl.W_Q[h, k, d] * tl.X[q, d]

        The compiler infers which axes are contracted (shared with the activation)
        and which are produced (shared with the lhs) from the equation.  The
        contraction compiles to an ops.Linear operator (drawn as an 'L' box)
        instead of an einsum: the weight becomes the layer's internal parameter
        (dropped from caller inputs) and bias=True makes it affine (Wx + b).
        """
        self._registry._register_declaration(
            self._name,
            TensorDeclaration(
                kind=TensorKind.LINEAR,
                shape=shape,
                bias=bias,
            ),
        )
        return self


class IndexedTensor:
    """A tensor name subscripted by a tuple of axes.

    Combine with * to accumulate factors into an RHSExpression.
    Combine with + to form a SumExpr.
    """

    def __init__(self, name: fd.DynamicName, indices: tuple[sc.RawAxis, ...]) -> None:
        self.name = name
        self.indices = indices

    def __mul__(
        self,
        other: IndexedTensor | RHSExpression | IversonBinOp | IversonUnaryOp,
    ) -> RHSExpression:
        if isinstance(other, (IversonBinOp, IversonUnaryOp)):
            return RHSExpression([self, other], ops.Identity())
        if isinstance(other, IndexedTensor):
            return RHSExpression([self, other], ops.Identity())
        return RHSExpression([self, *other.factors], other.operator)

    def __rmul__(self, other: IndexedTensor) -> RHSExpression:
        return RHSExpression([other, self], ops.Identity())

    def __add__(self, other: IndexedTensor | RHSExpression) -> SumExpr:
        lhs = RHSExpression([self], ops.Identity())
        rhs = other if isinstance(other, RHSExpression) else RHSExpression([other], ops.Identity())
        return SumExpr([lhs, rhs])


class RHSExpression:
    """Accumulated factors on the RHS of an equation.

    Produced by combining IndexedTensors with *. Wrap in relu() or softmax()
    to attach a nonlinearity. Combine with + to form a SumExpr.
    """

    def __init__(
        self,
        factors: list[IndexedTensor | IversonBinOp | IversonUnaryOp],
        operator: bc.Operator,
    ) -> None:
        self.factors = factors
        self.operator = operator

    def __mul__(
        self,
        other: IndexedTensor | IversonBinOp | IversonUnaryOp,
    ) -> RHSExpression:
        return RHSExpression(self.factors + [other], self.operator)

    def __add__(self, other: IndexedTensor | RHSExpression) -> SumExpr:
        rhs = other if isinstance(other, RHSExpression) else RHSExpression([other], ops.Identity())
        return SumExpr([self, rhs])


class SumExpr:
    """An elementwise sum of RHSExpression terms.

    Produced by + between IndexedTensor, RHSExpression, or SumExpr objects.
    Each term is compiled to a Broadcasted morphism and the results are
    combined with AdditionOp. Supports chaining: A + B + C.
    An optional `operator` (set by relu/softmax/normalize wrappers) is applied
    after the addition.
    """

    def __init__(self, terms: list[RHSExpression], operator: bc.Operator | None = None) -> None:
        self.terms = terms
        self.operator = operator

    def __add__(self, other: IndexedTensor | RHSExpression) -> SumExpr:
        rhs = other if isinstance(other, RHSExpression) else RHSExpression([other], ops.Identity())
        return SumExpr(self.terms + [rhs])


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def axes(*names: str) -> tuple[sc.RawAxis, ...]:
    """Return a tuple of RawAxis objects, one per name.

    Accepts either variadic strings or a single space-separated string:
        i, j, k = axes('i j k')
        i, j, k = axes('i', 'j', 'k')
    LaTeX-style names like d_{ff} work as-is.
    """
    flat: list[str] = []
    for n in names:
        flat.extend(n.split())
    return tuple(sc.RawAxis.named(n) for n in flat)


def norm_axis(name: str, size: int | None = None) -> NormAxis:
    """Return a NormAxis — marks the normalisation dimension (e.g. softmax axis).

    When size is provided, the axis carries a concrete integer size, which is
    required for auto-materialising Iverson predicates that reference this axis.
    """
    base = NormAxis.named(name)
    if size is None:
        return base
    return NormAxis(uid=base.uid, _size=nm.Integer(size))


def nat_axis(name: str, size: int | None = None) -> NatAxis:
    """Return a NatAxis with an optional concrete integer size (ℕ dimension)."""
    base = sc.RawAxis.named(name)
    _size = nm.Integer(size) if size is not None else base._size
    return NatAxis(uid=base.uid, _size=_size)


def real_axis(name: str, size: int | None = None) -> sc.RawAxis:
    """Return a RawAxis with an optional concrete integer size (ℝ dimension)."""
    base = sc.RawAxis.named(name)
    if size is None:
        return base
    return sc.RawAxis(uid=base.uid, _size=nm.Integer(size))


# ---------------------------------------------------------------------------
# Operator wrappers
# ---------------------------------------------------------------------------

def relu(expr: IndexedTensor | RHSExpression | SumExpr) -> RHSExpression | SumExpr:
    """Wrap an expression with a ReLU nonlinearity."""
    if isinstance(expr, SumExpr):
        return SumExpr(expr.terms, ops.ReLU())
    if isinstance(expr, IndexedTensor):
        expr = RHSExpression([expr], ops.Identity())
    return RHSExpression(expr.factors, ops.ReLU())


def softmax(
    expr: IndexedTensor | RHSExpression | SumExpr,
    where: IversonExpr | None = None,
) -> RHSExpression | SumExpr:
    """Wrap a score expression with a SoftMax nonlinearity.

    where: optional Iverson predicate.  Positions where the predicate is False
           are masked to -inf before exponentiation (masked softmax), so they
           receive zero attention weight regardless of the raw score.
           The normalization axis must be marked with norm_axis() in the LHS.
    """
    op = ops.SoftMax(where_predicate=((where,) if where is not None else ()))
    if isinstance(expr, SumExpr):
        return SumExpr(expr.terms, op)
    if isinstance(expr, IndexedTensor):
        expr = RHSExpression([expr], ops.Identity())
    return RHSExpression(expr.factors, op)


def normalize(
    expr: IndexedTensor | RHSExpression | SumExpr,
    where: IversonExpr | None = None,
) -> RHSExpression | SumExpr:
    """Wrap an expression with a Normalize (sum-norm) nonlinearity.

    where: optional Iverson predicate.  Positions where the predicate is False
           are zeroed before the sum-normalization denominator is computed, so
           they receive zero weight and are zeroed in the output.
           The normalization axis must be marked with norm_axis() in the LHS.
    """
    op = ops.Normalize(where_predicate=((where,) if where is not None else ()))
    if isinstance(expr, SumExpr):
        return SumExpr(expr.terms, op)
    if isinstance(expr, IndexedTensor):
        expr = RHSExpression([expr], ops.Identity())
    return RHSExpression(expr.factors, op)
