# Claude Opus 5 (1M context), effort high. Rewritten by Claude Fable 5.1, effort 80,
# on 2026-09-20, for the rules following the released code.
'''Check the quantization package: the datatype, the registry, the pass and the
quantised text-only DeepSeek-V4.1-Flash.

    python quantization/validate_quantization.py

Each check states one claim and prints one line. The script exits non-zero on the
first claim not holding, and `validations/run_validations.py` finds it by its name.

The claims about the quantised text-only model are the claims made by
`notebooks/sota/DeepSeekV41Flash.ipynb`, because that notebook keeps
the prose and the figures and its package holds its assertions, which is the rule for
a SOTA notebook with a package of its own.
'''
from __future__ import annotations

# Run as `python quantization/validate_quantization.py` from the repository root.
# Python prepends the script's own directory, where `data_structure` is this
# package's folder and would shadow the top-level one - drop it.
import os, sys
sys.path = [p for p in sys.path
            if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import advanced_axis_dynamics.data_structure.Operators as aops
import agent_display as ad
import construction_helpers as ch  # noqa: F401 - the @ overload aligns the axes
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import deepseek.data_structure as dst
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import para.data_structure.ParaWrap as para_wrap
import quantization.algebra.strip_quantisations as strip_quantisations
import quantization.data_structure.Quantization as Quantization
import quantization.processing.quantise_model as quantise_model
import quantization.registries.operator_quantisations as operator_quantisations
import term_utilities.term_utilities as tutil

import notebooks.display.axis_sizes as axis_sizes
import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
import notebooks.sota.DeepSeekV41Flash.operator_explanations as operator_explanations
import notebooks.sota.DeepSeekV41Flash.quantised_text_only_model as quantised_text_only_model
import notebooks.sota.DeepSeekV41Flash.text_only_model as text_only_model

BF16 = Quantization.BF16
FP32 = Quantization.FP32
E4M3 = Quantization.E4M3
E2M1 = Quantization.E2M1
INT64 = Quantization.INT64
INT32 = Quantization.INT32
MXFP8 = Quantization.MXFP8
MXFP4 = Quantization.MXFP4
FP8_WEIGHT = quantised_text_only_model.FP8_WEIGHT

POLICY = quantise_model.QuantizationPolicy(
    inputs=BF16, activations=BF16, scalars=FP32, rounded_operands=E4M3,
    weights={'V': FP32}, weights_by_default=E4M3,
    boxes={'K': quantise_model.BoxPolicy(
        contractions=quantise_model.ContractionQuantisation.ACCUMULATED)},
    block_results={'T': (BF16,)})


def _ask(morphism: cat.Broadcasted, *operands: Quantization.Quantified | None,
         enclosing_box: str | None = None,
         ) -> operator_quantisations.OperatorQuantisation:
    return operator_quantisations.rule_for(morphism.operator)(
        operator_quantisations.QuantisationQuestion(
            morphism=morphism, policy=POLICY, operand_quantisations=operands,
            enclosing_box=enclosing_box))


def _quantisation(datatype: cat.Datatype) -> Quantization.Quantified | None:
    return Quantization.quantisation_of(datatype)


# ==========================================================================
# The datatype.
# ==========================================================================
def check_the_quantisation_of_an_unquantised_index() -> None:
    '''`quantisation_of` finds no quantisation on an index that carries none, and
    `holds_real_numbers` is false for an index.'''
    assert Quantization.quantisation_of(cat.Natural(nm.Integer(8))) is None
    assert not Quantization.holds_real_numbers(cat.Natural(nm.Integer(8)))


def check_the_quantisation_of_a_complex_datatype() -> None:
    '''A complex number is quantised by quantising the reals wrapped by it, and an
    index keeps its datatype.'''
    complex_reals = dst.Complex(cat.Reals())
    quantised = Quantization.with_quantisation(complex_reals, FP32)
    assert quantised == dst.Complex(FP32)
    assert Quantization.quantisation_of(quantised) == FP32
    index = cat.Natural(nm.Integer(8))
    assert Quantization.with_quantisation(index, FP32) == index


def check_a_block_scaled_quantisation() -> None:
    '''A block-scaled quantisation differs from its element format, is named by the
    Microscaling specification where that names it and by its scale otherwise, is
    described with its packing and its scale, and is what a cast into it writes.'''
    assert MXFP8 != E4M3 and MXFP8.form is E4M3.form
    assert Quantization.format_name(MXFP8) == 'MXFP8'
    assert Quantization.format_name(MXFP4) == 'MXFP4'
    assert Quantization.format_name(FP8_WEIGHT) == 'E4M3 with UE8M0 per 32x32'
    assert Quantization.describe_quantisation(MXFP8) == (
        'MXFP8: E4M3, 8 bits, packed 4 to a word, with one UE8M0 scale per 32 '
        'channels')
    assert Quantization.describe_quantisation(BF16) == 'BF16, 16 bits, packed 2 to a word'
    assert Quantization.is_cast(Quantization.TypeConvert(source=BF16, target=MXFP8))
    assert Quantization.with_quantisation(cat.Reals(), MXFP8) == MXFP8
    assert BF16.vector == nm.Integer(2) and E2M1.vector == nm.Integer(8)


def check_the_quantisation_of_an_index() -> None:
    '''An index takes an integer quantisation and keeps its own bound, the
    quantisation read back off it compares equal whatever the bound, a quantisation
    of the reals leaves an index alone, and an index holds no real number.'''
    index = cat.Natural(nm.Integer(8))
    quantised = Quantization.with_quantisation(index, INT64)
    assert isinstance(quantised, Quantization.Quantified)
    assert quantised.wraps == index
    assert _quantisation(quantised) == INT64
    assert _quantisation(Quantization.with_quantisation(
        cat.Natural(nm.Integer(9)), INT64)) == INT64
    assert Quantization.with_quantisation(index, FP32) == index
    assert Quantization.holds_natural_numbers(quantised)
    assert not Quantization.holds_real_numbers(quantised)


def check_a_cast_changes_the_quantisation() -> None:
    '''`is_cast` reports whether a conversion changes the quantisation, which a
    conversion between two formats does and a conversion from one format into itself
    does not.'''
    rounding = Quantization.TypeConvert(source=BF16, target=E4M3)
    assert Quantization.is_cast(rounding)
    assert not Quantization.is_cast(
        Quantization.TypeConvert(source=E4M3, target=E4M3))


# ==========================================================================
# The registry.
# ==========================================================================
def check_the_rule_of_a_projection() -> None:
    '''A projection whose weight has fewer bits than the activations rounds a BF16
    operand to E4M3, first brings an FP32 operand to BF16, and returns BF16. A
    projection whose weight is FP32 reads and returns FP32.'''
    rounded = ops.Linear.template(1, 1, name='W')
    assert _ask(rounded, BF16).operands == (E4M3,)
    assert _ask(rounded, BF16).results == (BF16,)
    assert _ask(rounded, FP32).operands == (BF16,)
    wide = ops.Linear.template(1, 1, name='V')
    assert _ask(wide, BF16).operands == (FP32,)
    assert _ask(wide, BF16).results == (FP32,)


def check_the_rule_of_an_embedding() -> None:
    '''A lookup returns its row at the quantisation of the table, and a projection
    whose rounded quantisation the row already carries requires no cast of it. A
    table of integers returns the integer quantisation of the policy.'''
    axis = cat.RawAxis.named('d')
    table = ops.Embedding.template(cat.Natural(nm.Integer(8)), (axis,), name='W')
    assert _ask(table, None).results == (E4M3,)
    rounded = ops.Linear.template(1, 1, name='W')
    assert _ask(rounded, E4M3).operands == (E4M3,)
    integers = ops.Embedding.template(
        cat.Natural(nm.Integer(8)), (axis,), name='M', datatype=cat.Natural(nm.Integer(4)))
    assert _ask(integers, None).results == (INT64,)


def check_the_rule_of_a_contraction() -> None:
    '''An einsum reads its operands at the quantisation with the most bits among
    them and returns it. Inside a box named by the policy as a fused kernel, a
    contraction reads its operands at the quantisation with the fewest bits and
    returns the scalar quantisation, and a broadcast product still promotes.'''
    contraction = ops.Einops.template('q d, x d -> q x')
    eager = _ask(contraction, BF16, FP32)
    assert eager.operands == (FP32, FP32) and eager.results == (FP32,)
    fused = _ask(contraction, BF16, FP32, enclosing_box='K')
    assert fused.operands == (BF16, BF16) and fused.results == (FP32,)
    product = ops.Einops.template('q d, q d -> q d')
    assert _ask(product, BF16, FP32, enclosing_box='K').results == (FP32,)
    assert _ask(product, BF16, BF16, enclosing_box='K').results == (BF16,)


def check_the_rule_of_the_scalar_operators() -> None:
    '''A softmax reads and returns the scalar quantisation, a normalisation with a
    gain reads and returns the activation quantisation, and one without a gain the
    scalar quantisation.'''
    softmax = _ask(ops.SoftMax.template(), E4M3)
    assert softmax.operands == (FP32,) and softmax.results == (FP32,)
    with_gain = _ask(ops.Normalize.template(), FP32)
    assert with_gain.operands == (BF16,) and with_gain.results == (BF16,)
    without = _ask(ops.Normalize.template(gain=False, bias=False), BF16)
    assert without.operands == (FP32,) and without.results == (FP32,)


def check_every_operator_of_the_model_has_a_rule() -> None:
    '''The registry declares a rule for every operator class held by the text-only
    model, so no operation of it falls back on the scalar quantisation.'''
    missing = quantise_model.leaves_without_a_rule(
        text_only_model.v41_flash_text_only)
    assert not missing, f'{len(missing)} operator classes with no rule: {missing}'


# ==========================================================================
# The pass on a small expression.
# ==========================================================================
def a_projection_and_a_softmax() -> cat.Morphism:
    return ops.Linear.template(1, 1, name='W') @ ops.SoftMax.template()


def check_the_pass_on_a_projection_and_a_softmax() -> None:
    '''A projection followed by a softmax rounds its BF16 operand to E4M3, returns
    BF16, reads it into FP32 for the softmax, and returns the activation
    quantisation, with one cast at each of the three sites.'''
    quantised = quantise_model.quantise_model(a_projection_and_a_softmax(), POLICY)
    model = quantised.morphism
    assert not quantise_model.unquantised_weaves(model)
    assert quantise_model.cast_counts(model) == {
        ('BF16', 'E4M3'): 1, ('BF16', 'FP32'): 1, ('FP32', 'BF16'): 1}
    projection, = (node for node in tutil.type_search(cat.Broadcasted, model)
                   if isinstance(node.operator, ops.Linear))
    assert _quantisation(projection.dom()[0].datatype) == E4M3
    assert _quantisation(projection.cod()[0].datatype) == BF16
    softmax, = (node for node in tutil.type_search(cat.Broadcasted, model)
                if isinstance(node.operator, ops.SoftMax))
    assert _quantisation(softmax.cod()[0].datatype) == FP32
    assert _quantisation(model.cod()[0].datatype) == BF16
    assert quantised.weight_quantisations == {'W': E4M3}


def check_the_quantised_expression_round_trips() -> None:
    '''The quantised expression converts to a hypergraph and back with the same
    domain, codomain and operations.'''
    model = quantise_model.quantise_model(
        a_projection_and_a_softmax(), POLICY).morphism
    returned = h2m.hypergraph_to_morphism(hg.Multigraph.from_morphism(model))
    assert returned.dom() == model.dom()
    assert returned.cod() == model.cod()
    assert (len(quantise_model.graph_leaves(returned))
            == len(quantise_model.graph_leaves(model)))


def check_a_titled_block_returns_the_quantisation_named_by_the_policy() -> None:
    '''A block whose title is named by the policy has a cast on its result inside
    the block, so the wire leaving it carries the named quantisation, and the model's
    own result then needs no cast.'''
    titled = cat.Block.template(ops.SoftMax.template(), title='T')
    model = quantise_model.quantise_model(titled, POLICY).morphism
    assert quantise_model.cast_counts(model) == {
        ('BF16', 'FP32'): 1, ('FP32', 'BF16'): 1}
    block, = tutil.type_search(cat.Block, model)
    inside = tuple(node for node in tutil.type_search(cat.Broadcasted, block.body)
                   if isinstance(node.operator, Quantization.TypeConvert))
    assert len(inside) == 2, f'{len(inside)} casts inside the titled block'
    assert _quantisation(model.cod()[0].datatype) == BF16


# ==========================================================================
# The quantised text-only model.
# ==========================================================================
MODEL = quantised_text_only_model.v41_flash_text_only_quantised


def _body(name: str) -> cat.Morphism:
    part = quantised_text_only_model.part_named(name)
    box = part.body if isinstance(part, para_wrap.ParaWrap) else part
    return box.operator.block.body


def _nodes(morphism: cat.Morphism, kind: type[cat.Operator]
           ) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in quantise_model.operations_of(morphism)
                 if isinstance(node.operator, kind))


def check_every_wire_of_the_model_carries_a_quantisation() -> None:
    '''No wire of the quantised text-only model holds a real number without a
    quantisation.'''
    unwritten = quantised_text_only_model.unquantised_weaves()
    assert not unwritten, f'{len(unwritten)} wires of the model carry no quantisation'


def check_every_conversion_of_the_model_rounds() -> None:
    '''Every conversion of the quantised text-only model reads one format into
    another, the three round trips of the caches included.'''
    assert quantise_model.conversions_between_two_quantisations(MODEL)
    assert (len(quantise_model.casts_of(MODEL))
            == len(quantise_model.conversions_of(MODEL)))


def check_the_casts_of_the_model() -> None:
    '''The quantised text-only model holds 190 cast operations, in the eight pairs of
    formats tabulated by the notebook, counted once per written operation.'''
    counts = quantised_text_only_model.cast_counts()
    assert counts == {
        ('BF16', 'FP32'): 73,
        ('BF16', 'MXFP8'): 25,
        ('E2M1', 'FP32'): 2,
        ('E4M3', 'FP32'): 2,
        ('FP32', 'BF16'): 79,
        ('FP32', 'E2M1'): 2,
        ('FP32', 'E4M3'): 2,
        ('INT64', 'INT32'): 5,
    }, counts
    assert sum(counts.values()) == 190


def check_an_activation_is_rounded_only_in_front_of_a_rounded_projection() -> None:
    '''Every cast into MXFP8 inserted by the pass stands in front of a projection
    whose weight has fewer bits than an activation, every such projection reads
    every operand in MXFP8 and returns BF16, and every other projection reads and
    returns the quantisation of its weight.'''
    policy = quantised_text_only_model.RELEASED_POLICY
    quantisations = quantised_text_only_model.WEIGHT_QUANTISATIONS
    for projection in _nodes(MODEL, ops.Linear):
        name = projection.operator.name.to_bodies()
        weight = quantisations[name]
        operands = {_quantisation(weave.datatype) for weave in projection.input_weaves}
        result = _quantisation(projection.output_weaves[0].datatype)
        if operator_quantisations.has_fewer_bits_than_the_activations(weight, policy):
            assert operands <= {MXFP8}, (name, operands)
            assert result == BF16, (name, result)
        else:
            assert operands <= {weight}, (name, operands)
            assert result == weight, (name, result)
    for cast in quantise_model.casts_of(MODEL):
        if _quantisation(cast.operator.target) == MXFP8 \
                and cast.operator.name is not None \
                and cast.operator.name.to_bodies() == quantise_model.CAST_NAME:
            assert _quantisation(cast.operator.source) == BF16, cast


def check_the_rows_of_the_ngram_table_reach_the_projections_with_no_cast() -> None:
    '''The n-gram table of Engram is MXFP8, the lookup returns its row in MXFP8, and
    the key and value projections read that row with no cast between, because
    rounding a row that already carries its scale changes no value.'''
    body = _body('Eng{1}')
    tables = tuple(node for node in _nodes(body, ops.Embedding)
                   if Quantization.holds_real_numbers(node.output_weaves[0].datatype))
    assert len(tables) == 1, len(tables)
    assert _quantisation(tables[0].output_weaves[0].datatype) == MXFP8
    projections = tuple(node for node in _nodes(body, ops.Linear)
                        if node.operator.name.to_bodies() in ('W^{K}{1}', 'W^{V}{1}'))
    assert len(projections) == 2, len(projections)
    for projection in projections:
        assert _quantisation(projection.input_weaves[0].datatype) == MXFP8
    assert not any(_quantisation(cast.operator.target) == MXFP8
                   for cast in _nodes(body, Quantization.TypeConvert))


def check_the_row_of_a_cast_into_mxfp8() -> None:
    '''Every cast into MXFP8 of the model takes the row of the rounding kernel, and
    a cast inside a cache round trip keeps the row of its stored form.'''
    into_mxfp8 = tuple(cast for cast in quantise_model.casts_of(MODEL)
                       if _quantisation(cast.operator.target) == MXFP8)
    assert len(into_mxfp8) == 25, len(into_mxfp8)
    for cast in into_mxfp8:
        row = quantised_text_only_model.explain_quantised_cast(cast)
        assert row is not None and row.title == r'\text{Cast to MXFP8}', row
    into_e2m1 = tuple(cast for cast in quantise_model.casts_of(MODEL)
                      if _quantisation(cast.operator.target) == E2M1)
    assert into_e2m1, 'no cast into E2M1'
    for cast in into_e2m1:
        row = quantised_text_only_model.explain_quantised_cast(cast)
        assert row is not None and row.title == r'\text{Cast to E2M1}', row


def check_the_boxes_named_by_the_policy() -> None:
    '''The router and the coefficients return FP32, the indexer scores hold no cast,
    the candidate pool holds one cast and it writes the positions it returns in
    INT32, and the attention kernel reads BF16 into FP32 accumulators, casts the
    probabilities to BF16 and reads its positions in INT32.'''
    for name in ('Gate', 'Coef'):
        box = quantised_text_only_model.part_named(name)
        assert all(_quantisation(weave.datatype) == FP32
                   for weave in box.output_weaves), name
    assert not _nodes(_body('Sco'), Quantization.TypeConvert)
    assert quantise_model.cast_counts(_body('Pool')) == {('INT64', 'INT32'): 1}
    pool = quantised_text_only_model.part_named('Pool')
    assert _quantisation(pool.output_weaves[0].datatype) == INT32
    positions = tuple(
        weave for gather in quantised_text_only_model.boxes_named('Gth')
        for weave in gather.input_weaves
        if Quantization.holds_natural_numbers(weave.datatype))
    assert positions, 'no gather reads positions'
    assert all(_quantisation(weave.datatype) == INT32 for weave in positions)
    core = _body('Core')
    contractions = tuple(
        node for node in _nodes(core, ops.Einops)
        if operator_quantisations.has_a_contraction(node.operator))
    assert len(contractions) == 3, len(contractions)
    for node in contractions:
        operands = {_quantisation(weave.datatype) for weave in node.input_weaves}
        if len(node.input_weaves) == 2:
            assert operands == {BF16}, node
        else:
            assert operands == {FP32}, node
        assert _quantisation(node.output_weaves[0].datatype) == FP32, node
    assert quantise_model.cast_counts(core) == {('FP32', 'BF16'): 2}


def check_the_residual_stream_is_held_in_bf16() -> None:
    '''Every coefficient box reads the residual in BF16, and the collapse vector
    handed by a sublayer to the next stays FP32.'''
    for box in quantised_text_only_model.boxes_named('Coef'):
        assert _quantisation(box.input_weaves[0].datatype) == BF16
        assert _quantisation(box.output_weaves[0].datatype) == FP32
    assert _quantisation(MODEL.cod()[0].datatype) == FP32


def check_a_concatenation_reads_one_quantisation() -> None:
    '''The inverse rotary box rounds the rotated half to BF16 before concatenating it
    with the unrotated half, so both parts of the concatenation carry BF16.'''
    for concatenation in _nodes(_body('Rot^{-1}'), aops.ConcatenateAxes):
        assert all(_quantisation(weave.datatype) == BF16
                   for weave in concatenation.input_weaves), concatenation


def check_the_weights_of_the_model() -> None:
    '''The three projections of a routed expert are held in E2M1, the token
    embedding, the first output projection and the two BF16 projections of the
    indexer in BF16, eleven weights read through `.float()` or declared FP32 by the
    released code in FP32, every other weight in E4M3, and the token map, which holds
    integers, is not in the table.'''
    quantisations = quantised_text_only_model.WEIGHT_QUANTISATIONS
    assert len(quantisations) == 33, (
        f'{len(quantisations)} weights rather than thirty-three')
    assert quantisations['E\\mathrm{cmp}'] == INT64
    for name in quantised_text_only_model.ROUTED_EXPERT_PROJECTIONS:
        assert quantisations[name] == MXFP4, name
    for name in ('E', 'W^{Oa}', 'k^{I}', 'w^{I}'):
        assert quantisations[name] == BF16, name
    for name in ('H0', 'H1', 'H2', 'W^{C}', 'W^{Z}', 'W^{R}', '\\mathrm{bias}',
                 '\\mathrm{sink}', '\\mathrm{w}{1}', '\\mathrm{w}{14}', 'L'):
        assert quantisations[name] == FP32, name
    for name in ('W^{Qa}', 'W^{Qb}', 'W^{KV}', 'W^{Ob}', 'q^{I}', 'W^{Gs}',
                 'W^{Us}', 'W^{Ds}', 'W^{K}{1}', 'W^{V}{14}'):
        assert quantisations[name] == FP8_WEIGHT, name
    for name in ('E\\mathrm{ng}{1}', 'E\\mathrm{ng}{14}'):
        assert quantisations[name] == MXFP8, name


def check_the_indices_of_the_model() -> None:
    '''Every index of the quantised text-only model carries INT64, the released
    `int64`, except the positions returned by the candidate pool, read by the gather
    and held by the pool and selection slots, which carry INT32. The five casts
    between integers stand where the released code writes `.int()`: on the result of
    the candidate pool, and on the positions picked by each indexer before the gather
    reads them and before they are dropped onto the selection slot.'''
    integer_quantisations = {
        _quantisation(weave.datatype)
        for node in quantise_model.operations_of(MODEL)
        for weave in (*node.input_weaves, *node.output_weaves)
        if Quantization.holds_natural_numbers(weave.datatype)}
    assert integer_quantisations == {INT64, INT32}, integer_quantisations
    for selection in _nodes(MODEL, dst.TopK):
        positions = tuple(weave for weave in selection.output_weaves
                          if Quantization.holds_natural_numbers(weave.datatype))
        assert all(_quantisation(weave.datatype) == INT64 for weave in positions)
    between_integers = tuple(
        cast for cast in quantise_model.casts_of(MODEL)
        if Quantization.holds_natural_numbers(cast.operator.target))
    assert len(between_integers) == 5, len(between_integers)
    assert all(_quantisation(cast.operator.source) == INT64
               and _quantisation(cast.operator.target) == INT32
               for cast in between_integers)
    for name in ('Full', 'Rex'):
        body = _body(name)
        gathers = tuple(node for node in quantise_model.operations_of(body)
                        if isinstance(node.operator, ops.BlockOperator)
                        and node.operator.name is not None
                        and node.operator.name.to_bodies() == 'Gth')
        assert gathers, name
        for gather in gathers:
            assert _quantisation(gather.input_weaves[0].datatype) == INT32, name


def check_the_cast_between_naturals_opens_a_box() -> None:
    '''The one `ops.Cast` of each Engram hash reads a natural under INT64 and writes a
    natural under INT64, and its inspection row names both bounds and the format, as
    the row of the unquantised model names the bounds. The row is found as well once
    a figure has written the assigned sizes into the names of the bounds, which is
    how every HTML page presents the model.'''
    settings = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        axis_sizes=axis_sizes.AxisSizes.SUBSCRIPT,
        assigned_sizes=quantised_text_only_model.released_assigned_sizes())
    for term in (MODEL, notebook_diagrams.present_each_side(MODEL, settings)):
        casts = tuple(node for node in quantise_model.operations_of(term)
                      if isinstance(node.operator, ops.Cast))
        assert len(casts) == len(omitted_mechanisms.ENGRAM_LAYERS), len(casts)
        for cast in casts:
            row = operator_explanations.explain_any_cast(cast)
            assert row is not None, cast.output_weaves[0].datatype
            assert 'Nat' in row.formula
            assert row.description.endswith('held in INT64.'), row.description[-60:]


def check_the_boxes_of_the_model_keep_their_bodies() -> None:
    '''No tag of the quantised text-only model names two bodies, so a figure
    recycling blocks draws the body of each tag.'''
    bodies: dict[object, set[cat.Morphism]] = {}
    for block in tutil.type_search(cat.Block, MODEL):
        bodies.setdefault(block.block_tag.uid, set()).add(block.body)
    clashing = {tag for tag, found in bodies.items() if len(found) > 1}
    assert not clashing, f'{len(clashing)} tags name two bodies'


# ==========================================================================
# The quantisations taken back off a model.
# ==========================================================================
STRIPPED = quantised_text_only_model.v41_flash_text_only_without_quantisations

UNQUANTISED = text_only_model.v41_flash_text_only


def _weaves_holding_a_number(morphism: cat.Morphism) -> tuple[cat.Weave, ...]:
    return tuple(weave for weave in tutil.type_search(cat.Weave, morphism)
                 if Quantization.holds_a_number(weave.datatype))


def _box_body(morphism: cat.Morphism, name: str) -> cat.Morphism:
    part = quantised_text_only_model.part_named(name, morphism)
    box = part.body if isinstance(part, para_wrap.ParaWrap) else part
    return box.operator.block.body


def check_stripping_the_model_leaves_no_quantisation() -> None:
    '''The functor takes every quantisation off the quantised text-only model. No
    datatype of the result carries a `Quantified`, every weave of it holding a number
    is reported as unquantised, and it holds no conversion at all, so it holds no
    cast either.'''
    assert not tuple(tutil.type_search(Quantization.Quantified, STRIPPED))
    weaves = _weaves_holding_a_number(STRIPPED)
    assert len(quantise_model.unquantised_weaves(STRIPPED)) == len(weaves), (
        f'{len(weaves)} weaves hold a number and '
        f'{len(quantise_model.unquantised_weaves(STRIPPED))} carry no quantisation')
    assert not quantise_model.conversions_of(STRIPPED)
    assert not quantise_model.casts_of(STRIPPED)
    assert len(quantise_model.conversions_of(MODEL)) == 190


def check_the_stripped_model_reads_as_the_unquantised_model() -> None:
    '''The stripped model and `text_only_model.v41_flash_text_only` have the same
    listing once both are recycled. The two differ before recycling in the order of
    two independent operations at the head of the model, the initial collapse vector
    and the embedding, which the conversion through a hypergraph settles for both.'''
    assert (ad.listing(h2m.recycle(STRIPPED))
            == ad.listing(h2m.recycle(UNQUANTISED)))
    assert len(quantise_model.unquantised_weaves(STRIPPED)) == len(
        quantise_model.unquantised_weaves(UNQUANTISED))


def check_the_bodies_of_the_stripped_model_read_as_the_unquantised_bodies() -> None:
    '''Every box of the stripped model has the listing its box in
    `text_only_model.v41_flash_text_only` has, except the three cache round trips.
    The listing of the model itself states a box and not the body of one, so the
    bodies are compared box by box.'''
    for name in ('Full', 'MoE', 'Eng{1}', 'Sco', 'Core', 'Gate', 'Pool'):
        stripped = ad.listing(h2m.recycle(_box_body(STRIPPED, name)))
        unquantised = ad.listing(h2m.recycle(_box_body(UNQUANTISED, name)))
        assert stripped == unquantised, name


def check_the_stripped_cache_round_trip_holds_the_scaling_and_no_rounding() -> None:
    '''The FP4 round trip of the unquantised text-only model holds four conversions,
    written by `quantised_caches` rather than by the pass, and the stripped round trip
    holds none of them. A conversion between two quantisations of one channel is what
    the functor removes, whoever wrote it, so the round trip is left as the scaling it
    wraps: the group view, the largest magnitude, the scale, the division, the clamp
    and the multiplication back.'''
    written = _nodes(_box_body(UNQUANTISED, 'FP4e'), Quantization.TypeConvert)
    assert len(written) == 4, len(written)
    assert not _nodes(_box_body(STRIPPED, 'FP4e'), Quantization.TypeConvert)


def check_stripping_a_model_with_no_quantisation_returns_it() -> None:
    '''A model carrying no quantisation is returned as the object it was given,
    which is what preserves the sharing of a term.'''
    model = a_projection_and_a_softmax()
    assert strip_quantisations.strip_quantisations(model) is model


def check_stripping_is_idempotent() -> None:
    '''Stripping a stripped model returns the object it was given, because the model
    it returns carries no quantisation.'''
    assert strip_quantisations.strip_quantisations(STRIPPED) is STRIPPED
    model = a_projection_and_a_softmax()
    quantised = quantise_model.quantise_model(model, POLICY).morphism
    stripped = strip_quantisations.strip_quantisations(quantised)
    assert strip_quantisations.strip_quantisations(stripped) is stripped
    assert ad.listing(stripped) == ad.listing(model)


def check_a_conversion_that_is_not_between_two_quantisations_is_kept() -> None:
    '''The functor removes a conversion whose two sides carry a quantisation of one
    value. A conversion into a datatype carrying no quantisation, and a conversion
    from a quantised index into a real number, each fail one of the two tests and
    stay.'''
    assert strip_quantisations.converts_between_two_quantisations(
        Quantization.TypeConvert(source=BF16, target=MXFP8))
    assert not strip_quantisations.converts_between_two_quantisations(
        Quantization.TypeConvert(source=BF16, target=cat.Reals()))
    quantised_index = Quantization.with_quantisation(
        cat.Natural(nm.Integer(8)), INT32)
    assert not strip_quantisations.converts_between_two_quantisations(
        Quantization.TypeConvert(source=quantised_index, target=BF16))
    assert strip_quantisations.without_quantisations(dst.Complex(MXFP8)) == (
        dst.Complex(cat.Reals()))
    bounded = cat.Natural(nm.Integer(8))
    assert strip_quantisations.without_quantisations(
        Quantization.with_quantisation(bounded, INT32)) == bounded


CHECKS = (
    check_the_quantisation_of_an_unquantised_index,
    check_the_quantisation_of_a_complex_datatype,
    check_a_block_scaled_quantisation,
    check_the_quantisation_of_an_index,
    check_a_cast_changes_the_quantisation,
    check_the_rule_of_a_projection,
    check_the_rule_of_an_embedding,
    check_the_rule_of_a_contraction,
    check_the_rule_of_the_scalar_operators,
    check_every_operator_of_the_model_has_a_rule,
    check_the_pass_on_a_projection_and_a_softmax,
    check_the_quantised_expression_round_trips,
    check_a_titled_block_returns_the_quantisation_named_by_the_policy,
    check_every_wire_of_the_model_carries_a_quantisation,
    check_every_conversion_of_the_model_rounds,
    check_the_casts_of_the_model,
    check_an_activation_is_rounded_only_in_front_of_a_rounded_projection,
    check_the_rows_of_the_ngram_table_reach_the_projections_with_no_cast,
    check_the_row_of_a_cast_into_mxfp8,
    check_the_boxes_named_by_the_policy,
    check_the_residual_stream_is_held_in_bf16,
    check_a_concatenation_reads_one_quantisation,
    check_the_weights_of_the_model,
    check_the_indices_of_the_model,
    check_the_cast_between_naturals_opens_a_box,
    check_the_boxes_of_the_model_keep_their_bodies,
    check_stripping_the_model_leaves_no_quantisation,
    check_the_stripped_model_reads_as_the_unquantised_model,
    check_the_bodies_of_the_stripped_model_read_as_the_unquantised_bodies,
    check_the_stripped_cache_round_trip_holds_the_scaling_and_no_rounding,
    check_stripping_a_model_with_no_quantisation_returns_it,
    check_stripping_is_idempotent,
    check_a_conversion_that_is_not_between_two_quantisations_is_kept,
)


def run() -> int:
    '''Every check, one line each, and the number failed.'''
    failures = 0
    for check in CHECKS:
        try:
            check()
        except Exception as reason:
            failures += 1
            print(f'FAILED {check.__name__}: {type(reason).__name__}: {reason}')
        else:
            print(f'ok     {check.__name__}')
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} quantization checks passed')
    return failures


if __name__ == '__main__':
    sys.exit(1 if run() else 0)
