import pytest
from data_structure.TensorDSL import (
    TL, TensorProxy, IndexedTensor, RHSExpression,
    NormAxis, NatAxis, TensorKind, TensorDeclaration,
    axes, norm_axis, nat_axis, real_axis, relu, softmax,
    ieq, imul, iabs,
)
from data_structure.TensorExpr import TensorRef, IversonBinOp, IversonUnaryOp
from data_structure.TensorLogic import TensorEquation, TensorProgram
import data_structure.BroadcastedCategory as bc
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd


# ---------------------------------------------------------------------------
# axes() / norm_axis()
# ---------------------------------------------------------------------------

def test_axes_variadic():
    i, j, k = axes('i', 'j', 'k')
    assert isinstance(i, sc.RawAxis)
    assert i.uid != j.uid != k.uid

def test_axes_single_string():
    i, j, k = axes('i j k')
    assert isinstance(k, sc.RawAxis)
    assert i.uid != k.uid

def test_axes_latex_name():
    (d_ff,) = axes('d_{ff}')
    assert isinstance(d_ff, sc.RawAxis)

def test_norm_axis_type():
    x = norm_axis('x')
    assert isinstance(x, NormAxis)
    assert isinstance(x, sc.RawAxis)

def test_norm_axis_accepts_optional_size():
    """norm_axis('x', 5) must produce a NormAxis with concrete Integer size."""
    x = norm_axis('x', 5)
    assert isinstance(x, NormAxis)
    assert x._size == nm.Integer(5)

def test_norm_axis_no_size_unchanged():
    """norm_axis('x') without size must still work as before (free size)."""
    x = norm_axis('x')
    assert isinstance(x, NormAxis)
    assert isinstance(x._size, nm.FreeNumeric)


# ---------------------------------------------------------------------------
# TL / TensorProxy
# ---------------------------------------------------------------------------

def test_getattr_returns_proxy():
    tl = TL()
    proxy = tl.W
    assert isinstance(proxy, TensorProxy)
    assert proxy._name == 'W'

def test_getattr_underscore_raises():
    tl = TL()
    with pytest.raises(AttributeError):
        _ = tl._secret

def test_proxy_getitem_single_axis():
    tl = TL()
    (i,) = axes('i')
    it = tl.X[i]
    assert isinstance(it, IndexedTensor)
    assert it.indices == (i,)

def test_proxy_getitem_tuple():
    tl = TL()
    i, j = axes('i j')
    it = tl.W[i, j]
    assert it.indices == (i, j)


# ---------------------------------------------------------------------------
# IndexedTensor * composition
# ---------------------------------------------------------------------------

def test_mul_two_indexed_tensors():
    tl = TL()
    i, j, k = axes('i j k')
    expr = tl.W[i, k] * tl.X[k, j]
    assert isinstance(expr, RHSExpression)
    assert len(expr.factors) == 2
    assert isinstance(expr.operator, ops.Identity)

def test_mul_three_factors():
    tl = TL()
    a, b, c, d = axes('a b c d')
    expr = tl.A[a, b] * tl.B[b, c] * tl.C[c, d]
    assert len(expr.factors) == 3

def test_rmul_order():
    tl = TL()
    i, j, k = axes('i j k')
    w = tl.W[i, k]
    x = tl.X[k, j]
    expr = w * x
    assert expr.factors[0].name == fd.DynamicName('W')
    assert expr.factors[1].name == fd.DynamicName('X')


# ---------------------------------------------------------------------------
# relu / softmax wrappers
# ---------------------------------------------------------------------------

def test_relu_on_expression():
    tl = TL()
    i, j, k = axes('i j k')
    expr = relu(tl.W[i, k] * tl.X[k, j])
    assert isinstance(expr.operator, ops.ReLU)
    assert len(expr.factors) == 2

def test_relu_on_single_tensor():
    tl = TL()
    (i,) = axes('i')
    expr = relu(tl.X[i])
    assert isinstance(expr.operator, ops.ReLU)
    assert len(expr.factors) == 1

def test_softmax_on_expression():
    tl = TL()
    q, h, k = axes('q h k')
    expr = softmax(tl.Query[q, h, k] * tl.Key[q, h, k])
    assert isinstance(expr.operator, ops.SoftMax)


# ---------------------------------------------------------------------------
# __setitem__ captures equations with TensorRef factors
# ---------------------------------------------------------------------------

def test_setitem_captures_equation():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    assert len(tl._equations) == 1
    eq = tl._equations[0]
    assert isinstance(eq, TensorEquation)
    assert eq.lhs_name == fd.DynamicName('Y')

def test_setitem_rhs_contains_tensor_refs():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    eq = tl._equations[0]
    assert all(isinstance(f, TensorRef) for f in eq.rhs)

def test_setitem_single_factor_coerced():
    tl = TL()
    i, j = axes('i j')
    tl.Y[i, j] = tl.X[i, j]
    eq = tl._equations[0]
    assert len(eq.rhs) == 1

def test_setitem_preserves_axis_uid_identity():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    eq = tl._equations[0]
    # k appears in both rhs TensorRefs; uid must match
    k_in_W = eq.rhs[0].axes[1]   # W's second index
    k_in_X = eq.rhs[1].axes[0]   # X's first index
    assert k_in_W.uid == k_in_X.uid

def test_multiple_equations_ordered():
    tl = TL()
    p, d, d_ff = axes('p d d_ff')
    tl.Hidden[p, d_ff] = relu(tl.W_in[d_ff, d] * tl.X[p, d])
    tl.Output[p, d] = tl.W_out[d, d_ff] * tl.Hidden[p, d_ff]
    assert len(tl._equations) == 2
    assert tl._equations[0].lhs_name == fd.DynamicName('Hidden')
    assert tl._equations[1].lhs_name == fd.DynamicName('Output')


# ---------------------------------------------------------------------------
# to_equation / to_program
# ---------------------------------------------------------------------------

def test_to_equation_single():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    eq = tl.to_equation()
    assert isinstance(eq, TensorEquation)

def test_to_equation_raises_on_multiple():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    tl.Z[i, j] = tl.Y[i, j] * tl.B[i, j]
    with pytest.raises(ValueError):
        tl.to_equation()

def test_to_program():
    tl = TL()
    p, d, d_ff = axes('p d d_ff')
    tl.Hidden[p, d_ff] = relu(tl.W_in[d_ff, d] * tl.X[p, d])
    tl.Output[p, d] = tl.W_out[d, d_ff] * tl.Hidden[p, d_ff]
    prog = tl.to_program()
    assert isinstance(prog, TensorProgram)
    assert len(prog.equations) == 2


# ---------------------------------------------------------------------------
# End-to-end: bc_signature and to_morphism agree with manual construction
# ---------------------------------------------------------------------------

def test_matmul_bc_signature():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    sig = tl.to_equation().bc_signature()
    # 2 inputs (W, X), 1 output
    assert len(sig.input_weaves) == 2
    assert len(sig.output_weaves) == 1

def test_ffn_to_morphism():
    tl = TL()
    p, d, d_ff = axes('p d d_ff')
    tl.Hidden[p, d_ff] = relu(tl.W_in[d_ff, d] * tl.X[p, d])
    tl.Output[p, d] = tl.W_out[d, d_ff] * tl.Hidden[p, d_ff]
    morph = tl.to_program().to_morphism()
    from data_structure.ProductCategory import ThreadedComposed
    assert isinstance(morph, ThreadedComposed)
    assert len(morph.content) == 2

def test_norm_axis_preserved_in_equation():
    tl = TL()
    q, h, k = axes('q h k')
    x = norm_axis('x')
    tl.Comp[h, q, x] = softmax(tl.Query[q, h, k] * tl.Key[x, h, k])
    eq = tl.to_equation()
    assert isinstance(eq.lhs_indices[2], NormAxis)
    assert isinstance(eq.operator, ops.SoftMax)


# ---------------------------------------------------------------------------
# nat_axis / real_axis helpers
# ---------------------------------------------------------------------------

def test_nat_axis_type():
    t = nat_axis('t')
    assert isinstance(t, NatAxis)
    assert isinstance(t, sc.RawAxis)

def test_nat_axis_concrete_size():
    t = nat_axis('t', 50000)
    assert isinstance(t._size, nm.Integer)

def test_nat_axis_free_size_when_no_size():
    t = nat_axis('t')
    assert isinstance(t._size, nm.FreeNumeric)

def test_real_axis_type():
    d = real_axis('d')
    assert isinstance(d, sc.RawAxis)
    assert not isinstance(d, NatAxis)

def test_real_axis_concrete_size():
    d = real_axis('d', 512)
    assert isinstance(d._size, nm.Integer)

def test_real_axis_free_size_when_no_size():
    d = real_axis('d')
    assert isinstance(d._size, nm.FreeNumeric)


# ---------------------------------------------------------------------------
# TensorDeclaration / TensorKind
# ---------------------------------------------------------------------------

def test_tensor_declaration_stored():
    tl = TL()
    d, d_ff = real_axis('d', 512), real_axis('d_ff', 2048)
    tl.W_in.tensor(d_ff, d)
    decl = tl._declarations['W_in']
    assert isinstance(decl, TensorDeclaration)
    assert decl.kind is TensorKind.TENSOR
    assert len(decl.shape) == 2

def test_predicate_declaration_stored():
    tl = TL()
    q, x = axes('q x')
    tl.Mask.predicate(q, x)
    decl = tl._declarations['Mask']
    assert decl.kind is TensorKind.PREDICATE

def test_selection_declaration_stored():
    tl = TL()
    t = nat_axis('t', 50000)
    d = real_axis('d', 512)
    tl.Emb.selection(t, d)
    decl = tl._declarations['Emb']
    assert decl.kind is TensorKind.SELECTION

def test_declaration_returns_proxy():
    tl = TL()
    (i,) = axes('i')
    result = tl.X.tensor(i)
    assert isinstance(result, TensorProxy)


# ---------------------------------------------------------------------------
# Axis promotion via __getitem__
# ---------------------------------------------------------------------------

def test_predicate_no_longer_promotes_axes():
    """PREDICATE kind no longer promotes axes to PredAxis; indices returned as-is."""
    tl = TL()
    q, x = axes('q x')
    tl.Mask.predicate(q, x)
    i, j = axes('i j')
    it = tl.Mask[i, j]
    # No PredAxis type — plain RawAxis
    assert all(type(ax) is sc.RawAxis for ax in it.indices)

def test_predicate_preserves_uid():
    tl = TL()
    q, x = axes('q x')
    tl.Mask.predicate(q, x)
    i, j = axes('i j')
    it = tl.Mask[i, j]
    assert it.indices[0].uid == i.uid
    assert it.indices[1].uid == j.uid

def test_tensor_no_promotion():
    tl = TL()
    d, d_ff = axes('d d_ff')
    tl.W.tensor(d_ff, d)
    i, j = axes('i j')
    it = tl.W[i, j]
    assert not isinstance(it.indices[0], NatAxis)
    assert not isinstance(it.indices[1], NatAxis)

def test_selection_promotes_nat_slots_only():
    tl = TL()
    t = nat_axis('t', 50000)
    d = real_axis('d', 512)
    tl.Emb.selection(t, d)
    i, j = axes('i j')
    it = tl.Emb[i, j]
    assert isinstance(it.indices[0], NatAxis)      # slot declared NatAxis → promoted
    assert not isinstance(it.indices[1], NatAxis)  # slot declared RawAxis → unchanged

def test_selection_type_only_no_size():
    tl = TL()
    tl.Emb.selection(nat_axis('t'), real_axis('d'))  # type only, no size
    i, j = axes('i j')
    it = tl.Emb[i, j]
    assert isinstance(it.indices[0], NatAxis)
    assert not isinstance(it.indices[1], NatAxis)

def test_no_declaration_unchanged():
    tl = TL()
    i, j = axes('i j')
    it = tl.X[i, j]
    assert it.indices == (i, j)


# ---------------------------------------------------------------------------
# Arity checking
# ---------------------------------------------------------------------------

def test_getitem_arity_mismatch_raises():
    tl = TL()
    d, d_ff = axes('d d_ff')
    tl.W.tensor(d_ff, d)
    (i,) = axes('i')
    with pytest.raises(ValueError, match="2 axes"):
        _ = tl.W[i]

def test_setitem_arity_mismatch_raises():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y.tensor(i, j)
    with pytest.raises(ValueError, match="2 axes"):
        tl.Y[i, j, k] = tl.X[i, j, k]


# ---------------------------------------------------------------------------
# End-to-end: declared matmul still produces valid bc_signature
# ---------------------------------------------------------------------------

def test_tensor_declared_matmul_bc_signature():
    tl = TL()
    i = real_axis('i', 64)
    j = real_axis('j', 64)
    k = real_axis('k', 64)
    tl.W.tensor(i, k)
    tl.X.tensor(k, j)
    tl.Y.tensor(i, j)
    i2, j2, k2 = axes('i j k')
    tl.Y[i2, j2] = tl.W[i2, k2] * tl.X[k2, j2]
    sig = tl.to_equation().bc_signature()
    assert len(sig.input_weaves) == 2
    assert len(sig.output_weaves) == 1


# ---------------------------------------------------------------------------
# bc_signature signature-string guard
# ---------------------------------------------------------------------------

def test_bc_signature_rejects_nonempty_signature_string():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    with pytest.raises(ValueError, match="signature"):
        tl.to_equation().bc_signature(signature="ij,jk->ij")

def test_bc_signature_empty_string_accepted():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    assert isinstance(tl.to_equation().bc_signature(signature=''), bc.Broadcasted)


# ---------------------------------------------------------------------------
# Bool semiring: predicate declaration flows to bc.Bool() weave datatype
# ---------------------------------------------------------------------------

def test_predicate_declaration_produces_bool_datatypes():
    """_array_datatypes() returns bc.Bool() for each PREDICATE-declared tensor."""
    tl = TL()
    q, x = axes('q x')
    tl.Mask.predicate(q, x)
    adt = tl._array_datatypes()
    assert adt[fd.DynamicName('Mask')] == bc.Bool()

def test_predicate_weave_has_bool_datatype():
    """bc_signature() with a PREDICATE tensor produces a Bool-typed input weave."""
    tl = TL()
    q, x = axes('q x')
    tl.Mask.predicate(q, x)
    tl.Out[q, x] = tl.Score[q, x] * tl.Mask[q, x]
    sig = tl.bc_signature()
    # rhs slot 0 = Score (Reals), slot 1 = Mask (Bool)
    assert sig.input_weaves[1].datatype == bc.Bool()

def test_non_predicate_weave_has_reals_datatype():
    """Non-predicate tensors default to Reals datatype in bc_signature."""
    tl = TL()
    q, x = axes('q x')
    tl.Mask.predicate(q, x)
    tl.Out[q, x] = tl.Score[q, x] * tl.Mask[q, x]
    sig = tl.bc_signature()
    assert sig.input_weaves[0].datatype == bc.Reals()


# ---------------------------------------------------------------------------
# Iverson expressions
# ---------------------------------------------------------------------------

def test_iverson_binop_from_mul():
    """Multiplying an IndexedTensor by an IversonBinOp gives an RHSExpression."""
    tl = TL()
    q, x = axes('q x')
    pred = q < x  # RawAxis comparison → IversonBinOp
    assert isinstance(pred, IversonBinOp)
    expr = tl.A[q, x] * pred
    assert isinstance(expr, RHSExpression)
    assert len(expr.factors) == 2
    assert isinstance(expr.factors[1], IversonBinOp)

def test_iverson_factor_in_rhs():
    """An IversonBinOp factor in an equation's rhs is stored as-is (not TensorRef)."""
    from data_structure.TensorExpr import TensorRef
    tl = TL()
    q, x = axes('q x')
    pred = q < x
    tl.Out[q, x] = tl.A[q, x] * pred
    eq = tl._equations[0]
    assert isinstance(eq.rhs[0], TensorRef)
    assert isinstance(eq.rhs[1], IversonBinOp)

def test_ieq_helper():
    """ieq(x, y) produces an IversonBinOp with op='=='."""
    q, x = axes('q x')
    pred = ieq(q, x)
    assert isinstance(pred, IversonBinOp)
    assert pred.op == '=='

def test_iabs_helper():
    """iabs(x) produces an IversonUnaryOp with op='abs'."""
    q, = axes('q')
    expr = iabs(q)
    assert isinstance(expr, IversonUnaryOp)
    assert expr.op == 'abs'

def test_compound_iverson_expression():
    """iabs(q - x) < threshold is a valid nested IversonBinOp."""
    from data_structure.TensorExpr import IversonConst
    from data_structure.Numeric import Integer
    q, x = axes('q x')
    diff = q - x          # IversonBinOp('-', q, x)
    abs_diff = iabs(diff) # IversonUnaryOp('abs', diff)
    pred = abs_diff < IversonConst(Integer(5))  # IversonBinOp('<', abs_diff, IversonConst(5))
    assert isinstance(pred, IversonBinOp)
    assert pred.op == '<'
    assert isinstance(pred.lhs, IversonUnaryOp)

def test_iverson_axes_extracted_correctly():
    """_factor_axes() returns the RawAxis leaves from an Iverson factor."""
    from data_structure.TensorExpr import _factor_axes
    q, x = axes('q x')
    pred = q < x
    leaf_axes = _factor_axes(pred)
    assert len(leaf_axes) == 2
    assert leaf_axes[0].uid == q.uid
    assert leaf_axes[1].uid == x.uid


def test_dead_equation_excluded_from_routing():
    """An equation whose output is never consumed must not appear in the
    compiled morphism.  ThreadedComposed.content length is the signal."""
    from data_structure.ProductCategory import ThreadedComposed
    q = real_axis('q', 3)
    m = real_axis('m', 4)
    k = real_axis('k', 4)
    tl = TL()
    # Dead: Unused is defined but never referenced downstream.
    tl.Unused[q, m] = tl.W_dead[m, k] * tl.X_dead[q, k]
    # Live: the actual output.
    tl.Out[q, m] = tl.W[m, k] * tl.X[q, k]
    morph = tl.to_morphism()
    assert isinstance(morph, ThreadedComposed)
    # Only one step in the chain — the dead equation must be gone.
    assert len(morph.content) == 1


def test_live_equation_retained():
    """An equation whose output feeds downstream must survive DCE."""
    from data_structure.ProductCategory import ThreadedComposed
    q = real_axis('q', 3)
    m = real_axis('m', 4)
    k = real_axis('k', 4)
    tl = TL()
    tl.Hidden[q, k] = relu(tl.W1[m, k] * tl.X[q, m])
    tl.Out[q, m]    = tl.W2[m, k] * tl.Hidden[q, k]
    morph = tl.to_morphism()
    assert isinstance(morph, ThreadedComposed)
    assert len(morph.content) == 2


# ---------------------------------------------------------------------------
# _external_names_from_value
# ---------------------------------------------------------------------------

def test_external_names_from_value_excludes_state():
    """_external_names_from_value must skip excluded (state proxy) names and
    return external tensor names in expression order."""
    from data_structure.TensorDSL import _external_names_from_value
    import data_structure.Term as fd
    i, k = axes('i k')
    tl = TL()
    # Build an expression: W[i,k] * H_state[i,k]
    expr = tl.W[i, k] * tl.H_state[i, k]   # RHSExpression with two factors
    h_proxy = fd.DynamicName('H_state')
    names = _external_names_from_value(expr, exclude={h_proxy})
    assert fd.DynamicName('W') in names
    assert h_proxy not in names
    # W should be first (expression order)
    assert names[0] == fd.DynamicName('W')


# ---------------------------------------------------------------------------
# Markov shift-invariance: softmax normalization simplification
# ---------------------------------------------------------------------------

def test_normalization_simplification_drops_constant_bias():
    """A bias that does not depend on the norm axis must be dropped before
    bc_signature() is called.

    tl.Comp[h, q, x] = softmax(tl.Q[q,h,k] * tl.K[x,h,k] + tl.bias[h])

    tl.bias[h] has free indices {h}; the norm axis is x (a NormAxis).
    Since x is not in {h}, the bias term is constant along x and must be
    removed.  The resulting Composed einsum step must have 2 input weaves
    (Q, K), not 3 (Q, K, bias).
    """
    from data_structure.ProductCategory import Composed
    q, h, k = axes('q h k')
    x = norm_axis('x')
    tl = TL()
    tl.Comp[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k] + tl.bias[h])
    sig = tl.bc_signature()
    assert isinstance(sig, Composed), (
        f"Expected Composed(einsum, softmax); got {type(sig).__name__}"
    )
    einsum = sig.content[0]
    assert len(einsum.input_weaves) == 2, (
        f"Expected 2 input weaves (Q, K); got {len(einsum.input_weaves)}"
    )


def test_normalization_simplification_keeps_axis_dependent_term():
    """A term that depends on the norm axis must NOT be dropped.

    tl.scale[x] has free index x (the NormAxis), so it must be kept after
    softmax simplification.  bc_signature() must produce a Composed morphism
    whose domain has 3 inputs (Q, K, scale).
    """
    from data_structure.ProductCategory import Composed
    q, h, k = axes('q h k')
    x = norm_axis('x')
    tl = TL()
    tl.Comp[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k] + tl.scale[x])
    sig = tl.bc_signature()
    assert isinstance(sig, Composed), (
        f"Expected Composed; got {type(sig).__name__}"
    )
    assert len(sig.dom()) == 3, (
        f"Expected 3 input weaves (Q, K, scale); got {len(sig.dom())}"
    )


def test_norm_axis_softmax_compiles_to_composed():
    """norm_axis() in the LHS of a softmax equation must compile to a Composed
    (einsum @ SoftMax template) rather than crashing with a type-equality error."""
    from data_structure.ProductCategory import Composed
    q, h, k = axes('q h k')
    x = norm_axis('x')
    tl = TL()
    tl.Out[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k])
    sig = tl.bc_signature()
    assert isinstance(sig, Composed), (
        f"Expected Composed(einsum, softmax); got {type(sig).__name__}"
    )


def test_softmax_where_stores_predicate():
    """softmax(expr, where=pred) must store the predicate in the operator."""
    q, x = axes('q x')
    tl = TL()
    pred = x <= q   # IversonBinOp('<=', x, q)
    expr = softmax(tl.A[q, x], where=pred)
    assert len(expr.operator.where_predicate) == 1
    assert isinstance(expr.operator.where_predicate[0], IversonBinOp)
    assert expr.operator.where_predicate[0].op == '<='


def test_softmax_no_where_has_empty_predicate():
    """softmax(expr) without where= must have empty where_predicate."""
    q, x = axes('q x')
    tl = TL()
    expr = softmax(tl.A[q, x])
    assert expr.operator.where_predicate == ()


# ---------------------------------------------------------------------------
# _compute_mask_alignment
# ---------------------------------------------------------------------------

def test_compute_mask_alignment_causal():
    """For (x <= q) with LHS [h, q, x]:
    - Iverson DFS order: [x(pos=0), q(pos=1)], shape (X, Q)
    - Score LHS order: [h, q, x]
    - q is at iverson pos 1, lhs pos 1
    - x is at iverson pos 0, lhs pos 2
    - h is not in iverson -> n_broadcast = 1
    - perm = (1, 0)  (put iverson dim 1 first, then dim 0)
    """
    from data_structure.TensorLogic import _compute_mask_alignment
    h = real_axis('h', 2)
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    pred = x <= q   # IversonBinOp('<=', x, q); DFS order: [x, q]
    perm, n_broadcast = _compute_mask_alignment(pred, (h, q, x))
    assert perm == (1, 0), f"Expected (1, 0), got {perm}"
    assert n_broadcast == 1, f"Expected 1, got {n_broadcast}"


def test_compute_mask_alignment_no_broadcast():
    """For (a == b) with LHS [a, b]: no broadcast dims, perm = (0, 1) (identity)."""
    from data_structure.TensorLogic import _compute_mask_alignment
    a = real_axis('a', 3)
    b = real_axis('b', 3)
    pred = ieq(a, b)  # IversonBinOp('==', a, b); DFS: [a, b]
    perm, n_broadcast = _compute_mask_alignment(pred, (a, b))
    assert perm == (0, 1)
    assert n_broadcast == 0


# ---------------------------------------------------------------------------
# MaskedSoftMax operator
# ---------------------------------------------------------------------------

def test_masked_softmax_operator_template_is_broadcasted():
    """MaskedSoftMax.template() must return a Broadcasted with 1-axis weaves."""
    import data_structure.Operators as ops
    import data_structure.BroadcastedCategory as bc
    q, x = axes('q x')
    pred = x <= q
    perm, n_broadcast = (1, 0), 1
    template = ops.MaskedSoftMax.template(
        iverson_factors=(pred,),
        mask_alignments=((perm, n_broadcast),),
    )
    assert isinstance(template, bc.Broadcasted)
    assert len(template.input_weaves) == 1


def test_split_nonlinearity_with_where_produces_masked_softmax():
    """softmax(QK, where=(x<=q)) must compile to Composed(score_einsum, MaskedSoftMax)."""
    import data_structure.Operators as ops
    from data_structure.ProductCategory import Composed
    q = real_axis('q', 4)
    h = real_axis('h', 2)
    k = real_axis('k', 3)
    x = norm_axis('x', 4)
    tl = TL()
    tl.S[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k], where=(x <= q))
    sig = tl.bc_signature()
    assert isinstance(sig, Composed), f"Expected Composed, got {type(sig).__name__}"
    import data_structure.BroadcastedCategory as bc
    last = sig.content[-1]
    assert isinstance(last, bc.Broadcasted)
    assert isinstance(last.operator, ops.MaskedSoftMax)
    assert len(last.operator.iverson_factors) == 1


def test_inot_helper():
    """inot(x) produces an IversonUnaryOp with op='~'."""
    from data_structure.TensorExpr import inot
    q = real_axis('q', 4)
    expr = inot(q)
    assert isinstance(expr, IversonUnaryOp)
    assert expr.op == '~'

def test_invert_on_binop():
    """~(q < x) produces IversonUnaryOp('~', IversonBinOp('<', q, x))."""
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    pred = q < x
    negated = ~pred
    assert isinstance(negated, IversonUnaryOp)
    assert negated.op == '~'
    assert isinstance(negated.operand, IversonBinOp)

def test_invert_on_unaryop():
    """~~(q < x) double-negates correctly (IversonUnaryOp wrapping IversonUnaryOp)."""
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    inner = ~(q < x)
    double = ~inner
    assert isinstance(double, IversonUnaryOp)
    assert double.op == '~'
    assert isinstance(double.operand, IversonUnaryOp)

def test_invert_on_rawaxis():
    """~q on a RawAxis produces IversonUnaryOp('~', q)."""
    q = real_axis('q', 4)
    negated = ~q
    assert isinstance(negated, IversonUnaryOp)
    assert negated.op == '~'
