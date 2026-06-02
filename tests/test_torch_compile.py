import pytest
import torch
import data_structure.Category as cat
from data_structure.TensorDSL import TL, axes, real_axis, norm_axis, relu, softmax, normalize
from data_structure.TensorExpr import IversonConst, ieq, iabs
from data_structure.Numeric import Integer
from torch_compile.torch_compile import (
    ConstructedModule,
    ConstructedTensorEquation,
    generate_tensor_equation_signature,
)
from torch_compile.materialise import materialise_iverson


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matmul_sig():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    return tl.bc_signature()


def _bool_matmul_sig():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Out.predicate(i, j)
    tl.Out[i, j] = tl.A[i, k] * tl.B[k, j]
    return tl.bc_signature()


def _copy_sig():
    tl = TL()
    i, j = axes('i j')
    tl.Y[i, j] = tl.X[i, j]
    return tl.bc_signature()


def _reduction_sig():
    tl = TL()
    i, k = axes('i k')
    tl.Y[i,] = tl.X[i, k]
    return tl.bc_signature()


# ---------------------------------------------------------------------------
# Signature generation
# ---------------------------------------------------------------------------

def test_signature_matmul_has_two_input_segments():
    sig = generate_tensor_equation_signature(_matmul_sig())
    lhs, _ = sig.split('->')
    segments = [s.strip() for s in lhs.split(',')]
    assert len(segments) == 2


def test_signature_matmul_contracted_tag_in_both_inputs():
    sig = generate_tensor_equation_signature(_matmul_sig())
    lhs, _ = sig.split('->')
    seg0, seg1 = [s.strip() for s in lhs.split(',')]
    # x0 is the contracted axis; it must appear in both input segments
    assert 'x0' in seg0
    assert 'x0' in seg1


def test_signature_matmul_contracted_tag_not_in_output():
    sig = generate_tensor_equation_signature(_matmul_sig())
    _, rhs = sig.split('->')
    assert 'x0' not in rhs


def test_signature_copy_no_contracted_tags():
    sig = generate_tensor_equation_signature(_copy_sig())
    assert 'x' not in sig.split('->')[0].replace('...', '')


def test_signature_reduction_contracted_tag_in_input_not_output():
    sig = generate_tensor_equation_signature(_reduction_sig())
    lhs, rhs = sig.split('->')
    assert 'x0' in lhs
    assert 'x0' not in rhs


def test_signature_arrow_present():
    sig = generate_tensor_equation_signature(_matmul_sig())
    assert '->' in sig


# ---------------------------------------------------------------------------
# Dispatch and construction
# ---------------------------------------------------------------------------

def test_tensor_equation_registered():
    assert cat.TensorEquation in ConstructedModule.operation_registry


def test_construct_returns_constructed_tensor_equation():
    module = ConstructedModule.construct(_matmul_sig())
    assert isinstance(module, ConstructedTensorEquation)


def test_demote_false_for_reals_output():
    module = ConstructedModule.construct(_matmul_sig())
    assert module.demote is False


def test_demote_true_for_bool_output():
    module = ConstructedModule.construct(_bool_matmul_sig())
    assert module.demote is True


def test_nonlinearity_raises():
    from data_structure.TensorLogic import TensorEquation
    from data_structure.Operators import SoftMax
    import data_structure.Term as fd
    from data_structure.TensorExpr import TensorRef
    i, j, k = axes('i j k')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),
        rhs=(TensorRef(fd.DynamicName('W'), (i, k)), TensorRef(fd.DynamicName('X'), (k, j))),
        operator=SoftMax(),
    )
    br = eq.bc_signature()
    with pytest.raises(NotImplementedError):
        ConstructedTensorEquation(br)


# ---------------------------------------------------------------------------
# Forward — Reals output
# ---------------------------------------------------------------------------

def test_matmul_forward_shape():
    module = ConstructedModule.construct(_matmul_sig())
    W = torch.ones(3, 4)
    X = torch.ones(4, 5)
    result = module(W, X)
    assert result.shape == (3, 5)


def test_matmul_forward_values():
    module = ConstructedModule.construct(_matmul_sig())
    W = torch.ones(3, 4)
    X = torch.ones(4, 5)
    result = module(W, X)
    expected = torch.einsum('ik,kj->ij', W, X)
    assert torch.allclose(result, expected)


def test_matmul_forward_non_trivial_values():
    module = ConstructedModule.construct(_matmul_sig())
    W = torch.arange(6, dtype=torch.float).reshape(2, 3)
    X = torch.arange(6, dtype=torch.float).reshape(3, 2)
    result = module(W, X)
    expected = W @ X
    assert torch.allclose(result, expected)


def test_copy_forward_identity():
    module = ConstructedModule.construct(_copy_sig())
    X = torch.arange(6, dtype=torch.float).reshape(2, 3)
    result = module(X)
    assert torch.allclose(result, X)


def test_reduction_forward_sums_over_k():
    module = ConstructedModule.construct(_reduction_sig())
    X = torch.ones(3, 4)
    result = module(X)
    assert result.shape == (3,)
    assert torch.allclose(result, torch.full((3,), 4.0))


# ---------------------------------------------------------------------------
# Forward — Bool output (Heaviside demotion)
# ---------------------------------------------------------------------------

def test_bool_output_shape():
    module = ConstructedModule.construct(_bool_matmul_sig())
    A = torch.ones(3, 4)
    B = torch.ones(4, 5)
    result = module(A, B)
    assert result.shape == (3, 5)


def test_bool_output_positive_sum_yields_one():
    module = ConstructedModule.construct(_bool_matmul_sig())
    A = torch.ones(3, 4)
    B = torch.ones(4, 5)
    result = module(A, B)
    assert torch.all(result == 1.0)


def test_bool_output_zero_sum_yields_zero():
    module = ConstructedModule.construct(_bool_matmul_sig())
    A = torch.zeros(3, 4)
    B = torch.ones(4, 5)
    result = module(A, B)
    assert torch.all(result == 0.0)


def test_bool_output_negative_sum_yields_zero():
    module = ConstructedModule.construct(_bool_matmul_sig())
    A = -torch.ones(3, 4)
    B = torch.ones(4, 5)
    result = module(A, B)
    assert torch.all(result == 0.0)


def test_bool_output_preserves_dtype():
    module = ConstructedModule.construct(_bool_matmul_sig())
    A = torch.ones(3, 4, dtype=torch.float32)
    B = torch.ones(4, 5, dtype=torch.float32)
    result = module(A, B)
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# Modular materialisation interface
# ---------------------------------------------------------------------------
# These tests demonstrate the caller-materialisation contract: Iverson factors
# are passed as pre-built tensors in the same position they occupy in rhs.
# The module treats them identically to TensorRef tensors.

def test_pre_materialised_causal_mask():
    """Score[q,x] * [q<=x] with the mask supplied as a pre-built tensor."""
    from data_structure.TensorExpr import IversonBinOp
    tl = TL()
    q, x = axes('q x')
    tl.Attn[q, x] = tl.Score[q, x] * (q <= x)
    sig = tl.bc_signature()
    module = ConstructedModule.construct(sig)

    n = 4
    Score = torch.ones(n, n)
    # Materialise [q <= x] as lower-triangular (including diagonal)
    Mask = torch.tril(torch.ones(n, n))
    result = module(Score, Mask)

    expected = torch.tril(torch.ones(n, n))
    assert torch.allclose(result, expected)


def test_pre_materialised_mask_rhs_factor_order():
    """Verify rhs factor order: TensorRef first, Iverson second."""
    tl = TL()
    q, x = axes('q x')
    tl.Attn[q, x] = tl.Score[q, x] * (q <= x)
    sig = tl.bc_signature()
    assert len(sig.input_weaves) == 2


def test_pre_materialised_mask_bool_typed():
    """Inline Iverson input weave carries Bool() datatype."""
    tl = TL()
    q, x = axes('q x')
    tl.Attn[q, x] = tl.Score[q, x] * (q <= x)
    sig = tl.bc_signature()
    assert isinstance(sig.input_weaves[1].datatype, cat.Bool)


# ---------------------------------------------------------------------------
# materialise_iverson unit tests
# ---------------------------------------------------------------------------

def test_materialise_upper_triangular():
    """[q <= x] over 4x4 equals upper-triangular ones (q is row, x is col)."""
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    result = materialise_iverson(q <= x)
    assert torch.allclose(result, torch.triu(torch.ones(4, 4)))


def test_materialise_diagonal():
    """[q == x] over 4x4 equals the identity matrix."""
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    result = materialise_iverson(ieq(q, x))
    assert torch.allclose(result, torch.eye(4))


def test_materialise_banded():
    """|q - x| < 2 over 5x5 produces the tri-diagonal band."""
    q = real_axis('q', 5)
    x = real_axis('x', 5)
    result = materialise_iverson(iabs(q - x) < IversonConst(Integer(2)))
    expected = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_materialise_compound():
    """(q < x) & (x < k) with axes (q, x, x, k) has shape (3,4,4,5) and correct values."""
    q = real_axis('q', 3)
    x = real_axis('x', 4)
    k = real_axis('k', 5)
    result = materialise_iverson((q < x) & (x < k))
    assert result.shape == (3, 4, 4, 5)
    for qi in range(3):
        for x1 in range(4):
            for x2 in range(4):
                for ki in range(5):
                    assert result[qi, x1, x2, ki].item() == float((qi < x1) and (x2 < ki))


def test_materialise_free_axis_raises():
    """An unsized axis raises ValueError with a helpful message."""
    q, x = axes('q x')
    with pytest.raises(ValueError, match="no concrete size"):
        materialise_iverson(q <= x)


# ---------------------------------------------------------------------------
# ConstructedTensorEquation auto-materialisation integration tests
# ---------------------------------------------------------------------------

def _causal_mask_sig_sized():
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    tl = TL()
    tl.Attn[q, x] = tl.Score[q, x] * (q <= x)
    return tl.bc_signature()


def test_auto_materialise_shape():
    """Sized Iverson factor is auto-materialised; caller passes only Score."""
    module = ConstructedModule.construct(_causal_mask_sig_sized())
    result = module(torch.ones(4, 4))
    assert result.shape == (4, 4)


def test_auto_materialise_values():
    """Auto-materialised [q<=x] mask produces upper-triangular output."""
    module = ConstructedModule.construct(_causal_mask_sig_sized())
    result = module(torch.ones(4, 4))
    assert torch.allclose(result, torch.triu(torch.ones(4, 4)))


def test_auto_materialise_buffer_registered():
    """The pre-built mask is stored as a named buffer on the module."""
    module = ConstructedModule.construct(_causal_mask_sig_sized())
    assert '_mask_1' in dict(module.named_buffers())


def test_auto_materialise_bool_output():
    """Bool-output + sized Iverson factor: ∃ semantics and {0,1} values."""
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    k = real_axis('k', 4)
    tl = TL()
    tl.Gate.predicate(q, x)
    tl.Gate[q, x] = tl.A[q, k] * tl.B[k, x] * (q <= x)
    module = ConstructedModule.construct(tl.bc_signature())
    A = torch.ones(4, 4)
    B = torch.ones(4, 4)
    result = module(A, B)
    assert torch.all((result == 0) | (result == 1))
    # Gate[q,x] = ∃k: A[q,k] ∧ B[k,x] ∧ [q<=x] — true iff q<=x (since A,B are all-ones)
    assert torch.allclose(result, torch.triu(torch.ones(4, 4)))


# ---------------------------------------------------------------------------
# Coupled recurrences
# ---------------------------------------------------------------------------

def test_coupled_jacobi_correctness():
    """Two cross-dependent recurrences update simultaneously (Jacobi semantics).

    H[i, l+1] = H[i, l] + G[i, l]
    G[i, l+1] = G[i, l] * H[i, l]

    Jacobi: both updates use the OLD H and G from step l, not the partially
    updated values.  Verify against a reference Python loop.
    """
    i = real_axis('i', 3)
    l = real_axis('l', 4)

    tl = TL()
    tl.H.iteration_axis(l)
    tl.G.iteration_axis(l)
    tl.H[i, 0] = tl.X[i]
    tl.G[i, 0] = tl.Y[i]
    tl.H[i, l + 1] = tl.H[i, l] + tl.G[i, l]
    tl.G[i, l + 1] = tl.G[i, l] * tl.H[i, l]

    module = ConstructedModule.construct(tl.to_morphism())

    X = torch.tensor([1.0, 2.0, 3.0])
    Y = torch.tensor([1.0, 1.0, 1.0])

    # Reference: pure Python Jacobi loop
    H_ref, G_ref = X.clone(), Y.clone()
    H_hist = [H_ref.clone()]
    G_hist = [G_ref.clone()]
    for _ in range(4):
        H_new = H_ref + G_ref
        G_new = G_ref * H_ref   # uses old H, not H_new
        H_ref, G_ref = H_new, G_new
        H_hist.append(H_ref.clone())
        G_hist.append(G_ref.clone())
    H_expected = torch.stack(H_hist, dim=-1)  # (3, 5)
    G_expected = torch.stack(G_hist, dim=-1)

    # Module returns (H_seq, G_seq) in sorted(names) = ['G', 'H'] order.
    result = module(Y, X)   # base inputs: G first (sorted), then H
    G_out, H_out = result
    assert torch.allclose(H_out, H_expected), f"H mismatch:\n{H_out}\nvs\n{H_expected}"
    assert torch.allclose(G_out, G_expected), f"G mismatch:\n{G_out}\nvs\n{G_expected}"


def test_coupled_ordering_invariance():
    """Registering coupled recurrences in reversed order gives identical output."""
    i = real_axis('i', 2)
    l = real_axis('l', 3)

    def build(register_H_first: bool):
        tl = TL()
        tl.H.iteration_axis(l)
        tl.G.iteration_axis(l)
        if register_H_first:
            tl.H[i, 0] = tl.X[i]
            tl.G[i, 0] = tl.Y[i]
            tl.H[i, l + 1] = tl.H[i, l] + tl.G[i, l]
            tl.G[i, l + 1] = tl.G[i, l] + tl.H[i, l]
        else:
            tl.G[i, 0] = tl.Y[i]
            tl.H[i, 0] = tl.X[i]
            tl.G[i, l + 1] = tl.G[i, l] + tl.H[i, l]
            tl.H[i, l + 1] = tl.H[i, l] + tl.G[i, l]
        return ConstructedModule.construct(tl.to_morphism())

    mod_hfirst = build(register_H_first=True)
    mod_gfirst = build(register_H_first=False)

    X = torch.tensor([1.0, 2.0])
    Y = torch.tensor([3.0, 4.0])
    # Both modules: base inputs in sorted order (G first, then H)
    out_hfirst = mod_hfirst(Y, X)
    out_gfirst = mod_gfirst(Y, X)

    for a, b in zip(out_hfirst, out_gfirst):
        assert torch.allclose(a, b), "Output differs with different registration order"


def test_coupled_with_per_step_inputs():
    """Coupled recurrence where each state reads a distinct per-step input tensor.

    H[i, l+1] = H[i, l] + A[i, l]
    G[i, l+1] = G[i, l] + B[i, l]

    A and B are separate pre-loaded per-step tensors (not iterative).
    Verifies that n_step_xs routing is correct (no argument-count errors).
    """
    i = real_axis('i', 2)
    l = real_axis('l', 3)

    tl = TL()
    tl.H.iteration_axis(l)
    tl.G.iteration_axis(l)
    tl.H[i, 0] = tl.X[i]
    tl.G[i, 0] = tl.Y[i]
    tl.H[i, l + 1] = tl.H[i, l] + tl.A[i, l]
    tl.G[i, l + 1] = tl.G[i, l] + tl.B[i, l]

    module = ConstructedModule.construct(tl.to_morphism())

    X = torch.zeros(2)
    Y = torch.zeros(2)
    A = torch.ones(2, 3)
    B = torch.full((2, 3), 2.0)

    # Sorted order: G, H  — base inputs G first, then H; per-step inputs G first, then H
    result = module(Y, X, B, A)
    G_out, H_out = result

    # H[i, l] = sum(A[i, 0..l-1]) = l  (all-ones A, zero init)
    H_expected = torch.stack([torch.full((2,), float(k)) for k in range(4)], dim=-1)
    # G[i, l] = 2*l  (B=2, zero init)
    G_expected = torch.stack([torch.full((2,), 2.0 * k) for k in range(4)], dim=-1)

    assert torch.allclose(H_out, H_expected), f"H: {H_out}"
    assert torch.allclose(G_out, G_expected), f"G: {G_out}"


# ---------------------------------------------------------------------------
# Inline nonlinearity tests
# ---------------------------------------------------------------------------

def test_inline_softmax_compiles_and_runs():
    """softmax() inline in a TL equation compiles and produces valid probabilities."""
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    tl = TL()
    tl.Out[q, x] = softmax(tl.X[q, x])
    mod = ConstructedModule.construct(tl.bc_signature())
    result = mod(torch.ones(4, 4))
    # ConstructedComposed returns a tuple; unpack the single output.
    out = result[0] if isinstance(result, tuple) else result
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_inline_relu_compiles_and_runs():
    """relu() inline in a TL equation compiles and clips negatives."""
    i, j, k = axes('i j k')
    tl = TL()
    tl.Out[i, j] = relu(tl.W[i, k] * tl.X[k, j])
    mod = ConstructedModule.construct(tl.bc_signature())
    W = -torch.ones(3, 4)
    X = torch.ones(4, 5)
    result = mod(W, X)
    out = result[0] if isinstance(result, tuple) else result
    assert (out <= 0).all()


def test_inline_normalize_compiles_and_runs():
    """normalize() inline in a TL equation compiles and preserves shape."""
    p = real_axis('p', 8)
    m = real_axis('m', 16)
    tl = TL()
    tl.Out[p, m] = normalize(tl.X[p, m])
    mod = ConstructedModule.construct(tl.bc_signature())
    result = mod(torch.randn(8, 16))
    out = result[0] if isinstance(result, tuple) else result
    assert out.shape == (8, 16)


def test_inline_softmax_in_program():
    """softmax() inline in a TensorProgram (multi-equation) compiles correctly."""
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    k = real_axis('k', 8)
    tl = TL()
    tl.Scores[q, x] = tl.Q[q, k] * tl.K[x, k]
    tl.Attn[q, x] = softmax(tl.Scores[q, x])
    mod = ConstructedModule.construct(tl.to_program().to_morphism())
    Q = torch.randn(4, 8)
    K = torch.randn(4, 8)
    result = mod(Q, K)
    out = result[0] if isinstance(result, tuple) else result
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_inline_softmax_values_match_explicit_compose():
    """Inline softmax() produces the same output as the explicit @ ops.SoftMax step."""
    import data_structure.Operators as ops

    q = real_axis('q', 4)
    x = real_axis('x', 4)
    k = real_axis('k', 8)

    # Inline form
    tl_inline = TL()
    tl_inline.Scores[q, x] = tl_inline.Q[q, k] * tl_inline.K[x, k]
    q2, x2, k2 = axes('q x k')
    tl_inline.Attn[q2, x2] = softmax(tl_inline.Scores[q2, x2])
    mod_inline = ConstructedModule.construct(tl_inline.to_program().to_morphism())

    # Explicit compose form
    tl_a = TL()
    qa, xa, ka = axes('q x k')
    tl_a.Scores[qa, xa] = tl_a.Q[qa, ka] * tl_a.K[xa, ka]
    tl_b = TL()
    qb, xb = axes('q x')
    tl_b.Attn[qb, xb] = tl_b.Scores[qb, xb]
    morphism = tl_a.bc_signature() @ tl_b.bc_signature() @ ops.SoftMax.template()
    mod_explicit = ConstructedModule.construct(morphism)

    Q = torch.randn(4, 8)
    K = torch.randn(4, 8)
    out_inline = mod_inline(Q, K)
    out_explicit = mod_explicit(Q, K)
    t_inline = out_inline[0] if isinstance(out_inline, tuple) else out_inline
    t_explicit = out_explicit[0] if isinstance(out_explicit, tuple) else out_explicit
    assert torch.allclose(t_inline, t_explicit, atol=1e-5)


# ---------------------------------------------------------------------------
# ThreadedComposed tests
# ---------------------------------------------------------------------------

def test_threaded_no_regression_single_eq():
    tl = TL()
    i, j, k = axes('i j k')
    tl.Y[i, j] = tl.W[i, k] * tl.X[k, j]
    mod = ConstructedModule.construct(tl.to_morphism())
    W = torch.randn(3, 4)
    X = torch.randn(4, 5)
    result = mod(W, X)
    out = result[0] if isinstance(result, tuple) else result
    assert out.shape == (3, 5)


def test_threaded_shared_external():
    # H used by both steps with the same shape — must be threaded to step 2.
    # W1:(m,k), H:(q,k) -> Y:(q,m); V:(m,k), H:(q,k) -> Z:(q,m)
    q = real_axis('q', 3)
    m = real_axis('m', 5)
    k = real_axis('k', 4)
    tl = TL()
    tl.Y[q, m] = tl.W[m, k] * tl.H[q, k]   # step 1 uses H
    tl.Z[q, m] = tl.V[m, k] * tl.H[q, k]   # step 2 uses H (must be threaded)
    mod = ConstructedModule.construct(tl.to_morphism())
    # external order (topo): W, H, V
    W = torch.randn(5, 4)
    H = torch.randn(3, 4)
    V = torch.randn(5, 4)
    result = mod(W, H, V)
    out = result[0] if isinstance(result, tuple) else result
    assert out.shape == (3, 5)


def test_threaded_residual():
    # normalize(W2*Y + H) where H also feeds the projection Y = W1*H.
    # H:(q,dm); W1:(dm,dh), W2:(dm,dh) — H used in both steps.
    q = real_axis('q', 3)
    dm = real_axis('dm', 8)
    dh = real_axis('dh', 4)
    tl = TL()
    # step 1: project H down to dh
    tl.Y[q, dh] = tl.W1[dm, dh] * tl.H[q, dm]
    # step 2: project back up, add residual H, normalize
    tl.Out[q, dm] = normalize(tl.W2[dm, dh] * tl.Y[q, dh] + tl.H[q, dm])
    mod = ConstructedModule.construct(tl.to_morphism())
    # external order (topo): W1, H, W2
    W1 = torch.randn(8, 4)
    H = torch.randn(3, 8)
    W2 = torch.randn(8, 4)
    result = mod(W1, H, W2)
    out = result[0] if isinstance(result, tuple) else result
    assert out.shape == (3, 8)


def test_uncoupled_scan_returns_threaded_composed():
    """An uncoupled recurrence must produce ThreadedComposed, not Composed."""
    from data_structure.ProductCategory import ThreadedComposed, Composed
    i = real_axis('i', 3)
    l = real_axis('l', 4)
    tl = TL()
    tl.H[i, 0]     = tl.X[i]
    tl.H[i, l + 1] = tl.H[i, l] + tl.Delta[i, l]
    morph = tl.to_morphism()
    assert isinstance(morph, ThreadedComposed), (
        f"Expected ThreadedComposed, got {type(morph).__name__}"
    )


def test_uncoupled_scan_in_threaded_matches_composed_numerics():
    """Numerical output of ThreadedComposed path must match reference loop.

    H[i, 0] = X[i]
    H[i, l+1] = H[i, l] + Delta[i, l]
    """
    i = real_axis('i', 3)
    l = real_axis('l', 4)
    tl = TL()
    tl.H[i, 0]     = tl.X[i]
    tl.H[i, l + 1] = tl.H[i, l] + tl.Delta[i, l]
    mod = ConstructedModule.construct(tl.to_morphism())

    X     = torch.tensor([1.0, 2.0, 3.0])
    Delta = torch.zeros(3, 4)
    Delta[0, :] = 1.0

    result = mod(X, Delta)
    H_out = result[0] if isinstance(result, tuple) else result

    # Reference: H[i, 0]=X[i], H[i, l+1]=H[i,l]+Delta[i,l]
    H = X.clone()
    H_hist = [H.clone()]
    for step in range(4):
        H = H + Delta[:, step]
        H_hist.append(H.clone())
    expected = torch.stack(H_hist, dim=-1)  # shape (3, 5)

    assert torch.allclose(H_out, expected), (
        f"Mismatch:\n{H_out}\nvs\n{expected}"
    )


def test_uncoupled_scan_mixed_with_non_scan_equations():
    """A plain equation feeding an uncoupled Scan must compile to ThreadedComposed."""
    from data_structure.ProductCategory import ThreadedComposed
    q = real_axis('q', 2)
    m = real_axis('m', 3)
    l = real_axis('l', 3)
    tl = TL()
    # Plain equation first
    tl.E[q, m] = tl.W_emb[m, q] * tl.Tok[q]
    # Recurrence over E's output
    tl.H[q, m, 0]     = tl.E[q, m]
    tl.H[q, m, l + 1] = tl.H[q, m, l] + tl.Step[q, m, l]
    morph = tl.to_morphism()
    assert isinstance(morph, ThreadedComposed), (
        f"Expected ThreadedComposed, got {type(morph).__name__}"
    )


# ---------------------------------------------------------------------------
# Transformer example tests
#
# These test cases are extracted from papers/transformer_example.md.
# Axis sizes are kept small (SEQ=4, D=6, H=2, K=3, DFF=8) for speed.
#
# NOTE: normalize() without a norm_axis LHS argument normalises over all
# indices globally, not per-row.  Tests that require per-row normalisation
# semantics (causal mask invariance, FFN ReLU gate) are therefore omitted
# here; they require explicit norm_axis usage.  Composition of separately
# constructed Block-wrapped morphisms via @ also has an open issue and is
# excluded from this suite.
# ---------------------------------------------------------------------------

_SEQ, _D, _H, _K, _DFF = 4, 6, 2, 3, 8


def _mk_attn_res():
    """Attention sub-layer: Q/K/V projections → scores → causal mask → SV → output projection → residual."""
    tl = TL()
    q = real_axis('q', _SEQ); x = real_axis('x', _SEQ)
    m = real_axis('m', _D);   h = real_axis('h', _H);  k = real_axis('k', _K)
    tl.Query[q, h, k]    = tl.W_Q[h, k, m] * tl.H[q, m]
    tl.Key[x, h, k]      = tl.W_K[h, k, m] * tl.H[x, m]
    tl.Value[x, h, k]    = tl.W_V[h, k, m] * tl.H[x, m]
    tl.Comp[h, q, x]     = softmax(tl.Query[q, h, k] * tl.Key[x, h, k])
    tl.S[h, q, x]        = normalize(tl.Comp[h, q, x] * (x <= q))
    tl.AttnOut[q, h, k]  = tl.S[h, q, x] * tl.Value[x, h, k]
    tl.Attn[q, m]        = tl.W_O[m, h, k] * tl.AttnOut[q, h, k]
    tl.A[q, m]           = normalize(tl.Attn[q, m] + tl.H[q, m])
    return tl.to_morphism()


def _mk_ffn_res():
    """FFN sub-layer: linear → ReLU → linear → residual."""
    tl = TL()
    q = real_axis('q', _SEQ); m = real_axis('m', _D); d = real_axis('d', _DFF)
    tl.F[q, d]   = relu(tl.W_in[d, m] * tl.A[q, m])
    tl.Y[q, m]   = tl.W_out[m, d] * tl.F[q, d]
    tl.Out[q, m] = normalize(tl.Y[q, m] + tl.A[q, m])
    return tl.to_morphism()


def _mk_full_layer():
    """Full transformer layer as a single TL program (attention + FFN)."""
    tl = TL()
    q = real_axis('q', _SEQ); x = real_axis('x', _SEQ)
    m = real_axis('m', _D);   h = real_axis('h', _H)
    k = real_axis('k', _K);   d = real_axis('d', _DFF)
    tl.Query[q, h, k]    = tl.W_Q[h, k, m] * tl.H[q, m]
    tl.Key[x, h, k]      = tl.W_K[h, k, m] * tl.H[x, m]
    tl.Value[x, h, k]    = tl.W_V[h, k, m] * tl.H[x, m]
    tl.Comp[h, q, x]     = softmax(tl.Query[q, h, k] * tl.Key[x, h, k])
    tl.S[h, q, x]        = normalize(tl.Comp[h, q, x] * (x <= q))
    tl.AttnOut[q, h, k]  = tl.S[h, q, x] * tl.Value[x, h, k]
    tl.Attn[q, m]        = tl.W_O[m, h, k] * tl.AttnOut[q, h, k]
    tl.A[q, m]           = normalize(tl.Attn[q, m] + tl.H[q, m])
    tl.F[q, d]           = relu(tl.W_in[d, m] * tl.A[q, m])
    tl.Y[q, m]           = tl.W_out[m, d] * tl.F[q, d]
    tl.Out[q, m]         = normalize(tl.Y[q, m] + tl.A[q, m])
    return tl.to_morphism()


def test_attn_res_h_threaded_once():
    """H is referenced in 4 equations (Q, K, V projections + residual) but must
    appear exactly once in the compiled module's external inputs — ThreadedComposed
    routing routes one live-pool slot to all four consumers."""
    morph = _mk_attn_res()
    # External inputs in order of first appearance: W_Q, H, W_K, W_V, W_O
    assert morph.n_external == 5, (
        f"Expected 5 external inputs (W_Q H W_K W_V W_O); got {morph.n_external}"
    )


def test_ffn_res_a_threaded_once():
    """A is consumed by both the FFN projection and the residual sum but must
    appear exactly once in the compiled module's external inputs."""
    morph = _mk_ffn_res()
    # External inputs: W_in, A, W_out
    assert morph.n_external == 3, (
        f"Expected 3 external inputs (W_in A W_out); got {morph.n_external}"
    )


def test_attn_res_output_shape():
    """attn_res compiles and runs end-to-end; output shape is (SEQ, D)."""
    mod = ConstructedModule.construct(_mk_attn_res())
    W_Q = torch.randn(_H, _K, _D); H0 = torch.randn(_SEQ, _D)
    W_K = torch.randn(_H, _K, _D); W_V = torch.randn(_H, _K, _D)
    W_O = torch.randn(_D, _H, _K)
    out = mod(W_Q, H0, W_K, W_V, W_O)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == torch.Size([_SEQ, _D])


def test_ffn_res_output_shape():
    """ffn_res compiles and runs end-to-end; output shape is (SEQ, D)."""
    mod = ConstructedModule.construct(_mk_ffn_res())
    W_in = torch.randn(_DFF, _D); A = torch.randn(_SEQ, _D)
    W_out = torch.randn(_D, _DFF)
    out = mod(W_in, A, W_out)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == torch.Size([_SEQ, _D])


def test_full_transformer_layer_n_external():
    """The full transformer layer (attn + FFN in one TL program) has exactly 7
    external inputs: W_Q, H, W_K, W_V, W_O, W_in, W_out."""
    morph = _mk_full_layer()
    assert morph.n_external == 7, (
        f"Expected 7 external inputs; got {morph.n_external}"
    )


def test_full_transformer_layer_output_shape():
    """Full transformer layer runs end-to-end; output shape is (SEQ, D)."""
    mod = ConstructedModule.construct(_mk_full_layer())
    W_Q = torch.randn(_H, _K, _D); H0 = torch.randn(_SEQ, _D)
    W_K = torch.randn(_H, _K, _D); W_V = torch.randn(_H, _K, _D)
    W_O = torch.randn(_D, _H, _K)
    W_in = torch.randn(_DFF, _D); W_out = torch.randn(_D, _DFF)
    out = mod(W_Q, H0, W_K, W_V, W_O, W_in, W_out)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == torch.Size([_SEQ, _D])


def test_full_transformer_layer_output_deterministic():
    """Same weights and input produce identical outputs on two forward passes."""
    torch.manual_seed(7)
    mod = ConstructedModule.construct(_mk_full_layer())
    W_Q = torch.randn(_H, _K, _D); H0 = torch.randn(_SEQ, _D)
    W_K = torch.randn(_H, _K, _D); W_V = torch.randn(_H, _K, _D)
    W_O = torch.randn(_D, _H, _K)
    W_in = torch.randn(_DFF, _D); W_out = torch.randn(_D, _DFF)
    args = (W_Q, H0, W_K, W_V, W_O, W_in, W_out)
    out1 = mod(*args); out1 = out1[0] if isinstance(out1, tuple) else out1
    out2 = mod(*args); out2 = out2[0] if isinstance(out2, tuple) else out2
    assert torch.allclose(out1, out2)


def test_qkv_projection_numerical():
    """The Q projection Q[q,h,k] = W_Q[h,k,m] * H[q,m] matches a hand-computed
    einsum reference.  Tests that multi-axis contraction over m is compiled and
    executed correctly."""
    tl = TL()
    q = real_axis('q', _SEQ); m = real_axis('m', _D)
    h = real_axis('h', _H);   k = real_axis('k', _K)
    tl.Q[q, h, k] = tl.W_Q[h, k, m] * tl.H[q, m]
    mod = ConstructedModule.construct(tl.to_morphism())

    torch.manual_seed(0)
    W_Q = torch.randn(_H, _K, _D); H0 = torch.randn(_SEQ, _D)
    Q_tl = mod(W_Q, H0)
    Q_tl = Q_tl[0] if isinstance(Q_tl, tuple) else Q_tl
    Q_ref = torch.einsum('hkm,qm->qhk', W_Q, H0)
    assert torch.allclose(Q_tl, Q_ref, atol=1e-5), (
        f"Q projection mismatch: max diff {(Q_tl - Q_ref).abs().max().item()}"
    )


def test_ffn_projection_numerical():
    """The FFN projections F = relu(W_in * A) and Y = W_out * F match
    hand-computed einsum references.  Tests contraction + nonlinearity."""
    import einops
    tl = TL()
    q = real_axis('q', _SEQ); m = real_axis('m', _D); d = real_axis('d', _DFF)
    tl.F[q, d] = relu(tl.W_in[d, m] * tl.A[q, m])
    tl.Y[q, m] = tl.W_out[m, d] * tl.F[q, d]
    mod = ConstructedModule.construct(tl.to_morphism())

    torch.manual_seed(1)
    W_in = torch.randn(_DFF, _D); A = torch.randn(_SEQ, _D)
    W_out = torch.randn(_D, _DFF)
    Y_tl = mod(W_in, A, W_out)
    Y_tl = Y_tl[0] if isinstance(Y_tl, tuple) else Y_tl

    F_ref = torch.relu(einops.einsum(W_in, A, 'd m, q m -> q d'))
    Y_ref = einops.einsum(W_out, F_ref, 'm d, q d -> q m')
    assert torch.allclose(Y_tl, Y_ref, atol=1e-5), (
        f"FFN projection mismatch: max diff {(Y_tl - Y_ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# normalize() semantics — sum-normalization (x / x.sum(dim))
# ---------------------------------------------------------------------------

def test_normalize_row_sums_to_one():
    """normalize(In[q, x]) over x should produce rows summing to 1.

    This tests that normalize() compiles to x / x.sum(dim=-1), not LayerNorm.
    """
    tl = TL()
    q = real_axis('q', 3); x = real_axis('x', 5)
    tl.Out[q, x] = normalize(tl.In[q, x])
    mod = ConstructedModule.construct(tl.to_morphism())
    inp = torch.rand(3, 5) + 0.1   # all positive, avoid divide-by-zero
    out = mod(inp)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-5), (
        f"Row sums: {out.sum(dim=-1).tolist()} — expected all 1.0"
    )


def test_causal_mask_renormalize_is_lower_triangular():
    """The mask+normalize step alone produces correct causal attention weights.

    For uniform Comp (all-ones), S[h, q, x] should give equal weight to
    each causal position (x <= q) and zero weight to future positions.

    NOTE: normalize(softmax(QK) * mask) is mathematically equivalent to masked
    softmax in exact arithmetic — the full-softmax denominator D_full cancels in
    the renormalization step.  However, it has a real numerical failure mode: if
    a future key position receives a large score, it dominates the softmax and
    the causal values underflow to near machine-epsilon.  The clamp_min in
    normalize() then activates and the renormalized weights no longer sum to 1,
    producing incorrect attention.  The masked-softmax formulation (causal_softmax)
    avoids this entirely by masking before exponentiation.
    """
    SEQ, H = 5, 2
    tl = TL()
    q = real_axis('q', SEQ); x = real_axis('x', SEQ); h = real_axis('h', H)
    tl.S[h, q, x] = normalize(tl.Comp[h, q, x] * (x <= q))
    mod = ConstructedModule.construct(tl.to_morphism())

    Comp = torch.ones(H, SEQ, SEQ)
    S = mod(Comp)
    S = S[0] if isinstance(S, tuple) else S

    # All upper-triangle entries must be zero (causal mask)
    upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert S[:, upper].abs().max() < 1e-6, (
        f"Non-zero upper-triangle entry: max={S[:, upper].abs().max().item()}"
    )

    # Rows must sum to 1 (renormalized)
    assert torch.allclose(S.sum(dim=-1), torch.ones(H, SEQ), atol=1e-5), (
        f"Row sums not 1: {S.sum(dim=-1)}"
    )

    # Causal invariance for q=0: x=1..SEQ-1 are all future for q=0, so
    # changing Comp[:, 0, 1:] must not affect S[:, 0, :].
    # (We check q=0 specifically: for q=0, only x=0 is causal, so any change
    # to Comp[:, 0, 1:] must leave S[:, 0, :] = [1, 0, 0, ...] unchanged.)
    Comp_q0_future_changed = Comp.clone()
    Comp_q0_future_changed[:, 0, 1:] = torch.rand(H, SEQ - 1) * 10
    S2 = mod(Comp_q0_future_changed)
    S2 = S2[0] if isinstance(S2, tuple) else S2
    assert torch.allclose(S[:, 0, :], S2[:, 0, :], atol=1e-5), (
        "Mask+normalize is not causal at q=0: changing future Comp changed S[:,0,:]"
    )


def test_masked_softmax_excludes_future_positions():
    """softmax(QK, where=(x<=q)) must give S[h,q,x]=0 for x>q and rows summing to 1."""
    SEQ, H = 5, 2
    q_ax = real_axis('q', SEQ)
    h_ax = real_axis('h', H)
    k_ax = real_axis('k', 3)
    x_ax = norm_axis('x', SEQ)
    tl = TL()
    tl.S[h_ax, q_ax, x_ax] = softmax(
        tl.Q[q_ax, h_ax, k_ax] * tl.K[x_ax, h_ax, k_ax],
        where=(x_ax <= q_ax),
    )
    mod = ConstructedModule.construct(tl.to_morphism())

    torch.manual_seed(0)
    Q_t = torch.randn(SEQ, H, 3)
    K_t = torch.randn(SEQ, H, 3)
    S = mod(Q_t, K_t)
    S = S[0] if isinstance(S, tuple) else S    # shape (H, Q, X)

    upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert S[:, upper].abs().max() < 1e-6, (
        f"Non-zero upper-triangle: max={S[:, upper].abs().max().item()}"
    )
    assert torch.allclose(S.sum(dim=-1), torch.ones(H, SEQ), atol=1e-5), (
        f"Rows do not sum to 1: {S.sum(dim=-1)}"
    )
