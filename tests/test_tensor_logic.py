import pytest
import data_structure.Term as fd
import data_structure.BroadcastedCategory as bc
from data_structure.BroadcastedCategory import WeaveMode
import data_structure.ProductCategory as pc
from data_structure.TensorLogic import TensorEquation, TensorProgram, topological_sort
from data_structure.TensorExpr import TensorRef
from data_structure.TensorDSL import NormAxis
from data_structure.ProductCategory import ThreadedComposed
from data_structure.StrideCategory import RawAxis, Axis
from data_structure.Operators import Identity, SoftMax


def test_norm_axis_is_rawaxis_subclass():
    ax = NormAxis()
    assert isinstance(ax, RawAxis)


def test_norm_axis_named_returns_norm_axis():
    ax = NormAxis.named('t')
    assert isinstance(ax, NormAxis)


def test_norm_axis_distinct_from_raw_axis():
    assert NormAxis is not RawAxis
    assert not isinstance(RawAxis(), NormAxis)


def _matmul_eq():
    """Y[i,j] = W[i,k] X[k,j]"""
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    k = RawAxis.named('k')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('W'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    return eq, i, j, k


def test_tensor_equation_construction():
    eq, i, j, k = _matmul_eq()
    assert eq.lhs_name == fd.DynamicName('Y')
    assert len(eq.lhs_indices) == 2
    assert len(eq.rhs) == 2


def test_retained_uids_contains_lhs_indices():
    eq, i, j, k = _matmul_eq()
    retained = eq.retained_uids()
    assert i.uid in retained
    assert j.uid in retained
    assert k.uid not in retained


def test_contracted_axes_are_rhs_only():
    eq, i, j, k = _matmul_eq()
    contracted = eq.contracted_axes()
    assert len(contracted) == 1
    assert contracted[0].uid == k.uid


def test_tensor_equation_with_norm_axis():
    b = RawAxis.named('b')
    p = RawAxis.named('p')
    d = RawAxis.named('d')
    t = NormAxis.named('t')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(b, p, t),
        rhs=(
            TensorRef(fd.DynamicName('W_O'), (t, d)),
            TensorRef(fd.DynamicName('Stream'), (b, p, d)),
        ),
        operator=SoftMax(),
    )
    retained = eq.retained_uids()
    assert t.uid in retained
    assert d.uid not in retained
    contracted = eq.contracted_axes()
    assert any(ax.uid == d.uid for ax in contracted)


def test_bc_signature_matrix_multiply_degree():
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert br.degree() == pc.ProdObject((i, j))


def test_bc_signature_matrix_multiply_input_count():
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert len(br.input_weaves) == 2
    assert len(br.output_weaves) == 1


def test_bc_signature_w_weave_shape():
    # W[i, k]: i is retained (TILED), k is contracted (concrete)
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert br.input_weaves[0]._shape == (bc.WeaveMode.TILED, k)


def test_bc_signature_x_weave_shape():
    # X[k, j]: k is contracted (concrete), j is retained (TILED)
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert br.input_weaves[1]._shape == (k, bc.WeaveMode.TILED)


def test_bc_signature_output_weave_all_tiled():
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert all(p is bc.WeaveMode.TILED for p in br.output_weaves[0]._shape)
    assert len(br.output_weaves[0]._shape) == 2  # i, j


def test_bc_signature_operator_is_equation():
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert br.operator is eq


def test_bc_signature_w_reindexing_cod():
    # W contributes degree axis i (pos 0) → cod = (i,)
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert br.reindexings[0].cod() == pc.ProdObject((i,))


def test_bc_signature_x_reindexing_cod():
    # X contributes degree axis j (pos 1) → cod = (j,)
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    assert br.reindexings[1].cod() == pc.ProdObject((j,))


def test_bc_signature_dom_reconstructs_input_shapes():
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    dom = br.dom()
    # W shape: (i, k); X shape: (k, j)
    assert dom[0] == bc.Array(bc.Reals(), (i, k))
    assert dom[1] == bc.Array(bc.Reals(), (k, j))


def test_bc_signature_cod_is_output_shape():
    eq, i, j, k = _matmul_eq()
    br = eq.bc_signature()
    cod = br.cod()
    assert cod[0] == bc.Array(bc.Reals(), (i, j))


def _chain_equations():
    """eq1: Hidden[i,j] = W1[i,k] X[k,j];  eq2: Y[i,m] = W2[i,j] Hidden[j,m]"""
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    k = RawAxis.named('k')
    eq1 = TensorEquation(
        lhs_name=fd.DynamicName('Hidden'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('W1'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    m = RawAxis.named('m')
    eq2 = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, m),
        rhs=(
            TensorRef(fd.DynamicName('W2'), (i, j)),
            TensorRef(fd.DynamicName('Hidden'), (j, m)),
        ),
        operator=Identity(),
    )
    return eq1, eq2


def test_topological_sort_already_ordered():
    eq1, eq2 = _chain_equations()
    result = topological_sort((eq1, eq2))
    assert result[0] is eq1
    assert result[1] is eq2


def test_topological_sort_reversed_input():
    eq1, eq2 = _chain_equations()
    result = topological_sort((eq2, eq1))
    assert result[0] is eq1
    assert result[1] is eq2


def test_tensor_program_single_equation():
    i = RawAxis.named('i')
    k = RawAxis.named('k')
    j = RawAxis.named('j')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('W'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    prog = TensorProgram(equations=(eq,))
    morphism = prog.to_morphism()
    assert isinstance(morphism, ThreadedComposed)
    assert len(morphism.content) == 1


def test_tensor_program_two_equation_chain():
    """Two equations in sequence; to_morphism() produces a Composed of length 2."""
    i = RawAxis.named('i')
    k = RawAxis.named('k')
    j = RawAxis.named('j')
    eq1 = TensorEquation(
        lhs_name=fd.DynamicName('Hidden'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('W1'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    # eq2 uses fresh axes that will be unified with eq1's lhs_indices
    i2 = RawAxis.named('i')
    j2 = RawAxis.named('j')
    m = RawAxis.named('m')
    eq2 = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i2, m),
        rhs=(
            TensorRef(fd.DynamicName('W2'), (i2, j2)),
            TensorRef(fd.DynamicName('Hidden'), (j2, m)),
        ),
        operator=Identity(),
    )
    prog = TensorProgram(equations=(eq1, eq2))
    morphism = prog.to_morphism()
    assert isinstance(morphism, ThreadedComposed)
    assert len(morphism.content) == 2


def test_tensor_program_cod_has_correct_rank():
    i = RawAxis.named('i')
    k = RawAxis.named('k')
    j = RawAxis.named('j')
    eq1 = TensorEquation(
        lhs_name=fd.DynamicName('Hidden'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('W1'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    i2 = RawAxis.named('i')
    j2 = RawAxis.named('j')
    m = RawAxis.named('m')
    eq2 = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i2, m),
        rhs=(
            TensorRef(fd.DynamicName('W2'), (i2, j2)),
            TensorRef(fd.DynamicName('Hidden'), (j2, m)),
        ),
        operator=Identity(),
    )
    prog = TensorProgram(equations=(eq1, eq2))
    morphism = prog.to_morphism()
    cod = morphism.cod()
    # Final output is Y[i2, m] — one Array with 2 axes
    assert len(cod) == 1
    assert len(cod[0]._shape) == 2


def test_topological_sort_independent_equations():
    i = RawAxis.named('i')
    k = RawAxis.named('k')
    eq_a = TensorEquation(
        lhs_name=fd.DynamicName('A'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('X'), (i, k)),),
        operator=Identity(),
    )
    eq_b = TensorEquation(
        lhs_name=fd.DynamicName('B'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('Y'), (i, k)),),
        operator=Identity(),
    )
    result = topological_sort((eq_a, eq_b))
    assert set(r.lhs_name for r in result) == {fd.DynamicName('A'), fd.DynamicName('B')}
    assert len(result) == 2


def _diag_eq():
    """Y[i] = X[i, i]  — diagonal extraction, repeated retained index."""
    i = RawAxis.named('i')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('X'), (i, i)),),
        operator=Identity(),
    )
    return eq, i


def test_diag_retained_uids():
    eq, i = _diag_eq()
    assert eq.retained_uids() == {i.uid}


def test_diag_no_contracted_axes():
    eq, i = _diag_eq()
    assert eq.contracted_axes() == ()


def test_diag_bc_signature_degree():
    eq, i = _diag_eq()
    br = eq.bc_signature()
    assert br.degree() == pc.ProdObject((i,))


def test_diag_bc_signature_dom_shape():
    # X[i, i]: input shape must be (i, i) — a square matrix
    eq, i = _diag_eq()
    br = eq.bc_signature()
    assert br.dom()[0] == bc.Array(bc.Reals(), (i, i))


def test_diag_bc_signature_cod_shape():
    # Y[i]: output shape is a 1-D vector
    eq, i = _diag_eq()
    br = eq.bc_signature()
    assert br.cod()[0] == bc.Array(bc.Reals(), (i,))


def test_diag_reindexing_mapping():
    # Both X positions map to degree position 0 — the diagonal constraint
    eq, i = _diag_eq()
    br = eq.bc_signature()
    assert br.reindexings[0].mapping == (0, 0)


def test_diag_reindexing_cod():
    # cod of the reindexing is (i, i) — both slots resolved to the same degree axis
    eq, i = _diag_eq()
    br = eq.bc_signature()
    assert br.reindexings[0].cod() == pc.ProdObject((i, i))


def test_topological_sort_raises_on_cycle():
    i = RawAxis.named('i')
    k = RawAxis.named('k')
    # A depends on B, B depends on A — cycle
    eq_a = TensorEquation(
        lhs_name=fd.DynamicName('A'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('B'), (i, k)),),
        operator=Identity(),
    )
    eq_b = TensorEquation(
        lhs_name=fd.DynamicName('B'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('A'), (i, k)),),
        operator=Identity(),
    )
    with pytest.raises(ValueError, match="cyclic"):
        topological_sort((eq_a, eq_b))


def test_exports_from_category():
    from data_structure.Category import NormAxis, TensorEquation, TensorProgram
    assert NormAxis is not None
    assert TensorEquation is not None
    assert TensorProgram is not None


# ---------------------------------------------------------------------------
# Self-join on a computed intermediate  (Gram matrix: Y[i,j] = H[i,k] H[j,k])
# ---------------------------------------------------------------------------

def _gram_program():
    """
    H[a, b] = W[a, k] X[k, b]      -- standard matmul, intermediate
    Y[i, j] = H[i, k] H[j, k]      -- Gram matrix, self-join on H

    All axis objects are fresh; k is shared between both H references in Y
    so the contraction is coupled.  i and j are independent — they are the
    two 'row' aliases that a self-join requires.
    """
    a = RawAxis.named('a')
    b = RawAxis.named('b')
    k_h = RawAxis.named('k')   # contracted inside H's definition

    h_eq = TensorEquation(
        lhs_name=fd.DynamicName('H'),
        lhs_indices=(a, b),
        rhs=(
            TensorRef(fd.DynamicName('W'), (a, k_h)),
            TensorRef(fd.DynamicName('X'), (k_h, b)),
        ),
        operator=Identity(),
    )

    i = RawAxis.named('i')
    j = RawAxis.named('j')
    k_y = RawAxis.named('k')   # contracted across both H references in Y

    y_eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('H'), (i, k_y)),
            TensorRef(fd.DynamicName('H'), (j, k_y)),
        ),
        operator=Identity(),
    )

    return TensorProgram(equations=(h_eq, y_eq)), i, j, k_y, a, b


def test_self_join_composed_length():
    prog, *_ = _gram_program()
    morphism = prog.to_morphism()
    assert isinstance(morphism, ThreadedComposed)
    assert len(morphism.content) == 2


def test_self_join_row_axes_are_distinct():
    """After to_morphism(), i and j must not have been merged into one UID."""
    prog, i, j, k_y, a, b = _gram_program()
    prog.to_morphism()
    # If the bug were present, ctx.apply would have unified i and j through
    # H's canonical row axis, making them the same UID.
    assert i.uid != j.uid


def test_self_join_y_morphism_has_two_h_input_weaves():
    """bc_signature() of Y must produce two input weaves, one per H reference."""
    prog, *_ = _gram_program()
    morphism = prog.to_morphism()
    y_br = morphism.content[1]   # second Broadcasted = Y equation
    assert len(y_br.input_weaves) == 2


def test_self_join_y_degree_has_two_axes():
    """Y[i, j]: degree must have exactly two axes (i and j)."""
    prog, *_ = _gram_program()
    morphism = prog.to_morphism()
    y_br = morphism.content[1]
    assert len(y_br.degree()) == 2


def test_self_join_contracted_axis_shared_between_h_references():
    """k appears in both H weaves as the same concrete target axis (same UID)."""
    prog, i, j, k_y, a, b = _gram_program()
    morphism = prog.to_morphism()
    y_br = morphism.content[1]
    # Each input weave has one TILED slot and one concrete axis (the contracted k).
    # The concrete axis in weave[0] and weave[1] must share a UID.
    def contracted_axis(weave):
        concretes = [s for s in weave._shape if s is not WeaveMode.TILED]
        assert len(concretes) == 1, "expected exactly one contracted axis per H weave"
        return concretes[0]

    k_in_first  = contracted_axis(y_br.input_weaves[0])
    k_in_second = contracted_axis(y_br.input_weaves[1])
    assert k_in_first.uid == k_in_second.uid


def test_self_join_h_morphism_cod_rank():
    """H's Broadcasted produces one output array with two axes."""
    prog, *_ = _gram_program()
    morphism = prog.to_morphism()
    h_br = morphism.content[0]
    cod = h_br.cod()
    assert len(cod) == 1
    assert len(cod[0]._shape) == 2


def test_self_join_topological_order():
    """H must be processed before Y even when equations are given reversed."""
    prog_fwd, *_ = _gram_program()
    h_eq, y_eq = prog_fwd.equations
    prog_rev = TensorProgram(equations=(y_eq, h_eq))
    # to_morphism() must not raise and must still produce length-2 Composed.
    morphism = prog_rev.to_morphism()
    assert len(morphism.content) == 2


# ── declarations parameter ───────────────────────────────────────────────────

def test_to_morphism_declarations_accepted():
    """Providing a correctly-ranked declaration for a known lhs_name raises no error."""
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    k = RawAxis.named('k')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('W'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    d_i = RawAxis.named('i')
    d_j = RawAxis.named('j')
    morphism = TensorProgram(equations=(eq,)).to_morphism(
        declarations={fd.DynamicName('Y'): (d_i, d_j)}
    )
    assert isinstance(morphism, ThreadedComposed)
    assert len(morphism.content) == 1


def test_to_morphism_unknown_declaration_is_ignored():
    """A declaration key that matches no lhs_name is silently ignored."""
    eq, *_ = _matmul_eq()
    morphism = TensorProgram(equations=(eq,)).to_morphism(
        declarations={fd.DynamicName('UNKNOWN'): (RawAxis.named('z'),)}
    )
    assert isinstance(morphism, ThreadedComposed)


# ── bug-exposing tests (expected to fail until bugs are fixed) ───────────────

def test_topological_sort_raises_on_duplicate_lhs_name():
    """Two equations sharing lhs_name should raise ValueError; currently silently drops one."""
    i = RawAxis.named('i')
    k = RawAxis.named('k')
    eq_a = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('X'), (i, k)),),
        operator=Identity(),
    )
    eq_b = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('Z'), (i, k)),),
        operator=Identity(),
    )
    with pytest.raises(ValueError, match="duplicate"):
        topological_sort((eq_a, eq_b))


def test_to_morphism_declaration_rank_mismatch_raises():
    """A declaration whose length != len(lhs_indices) should raise ValueError; currently truncates silently."""
    import pytest
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    k = RawAxis.named('k')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),   # rank 2
        rhs=(
            TensorRef(fd.DynamicName('W'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    with pytest.raises(ValueError, match="rank"):
        TensorProgram(equations=(eq,)).to_morphism(
            declarations={fd.DynamicName('Y'): (RawAxis.named('i'),)}  # rank 1
        )
