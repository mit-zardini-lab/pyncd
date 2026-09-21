'''Check every claim about the mechanisms the model leaves out.

Written by Claude Fable 5.1, reasoning effort 80.

    python notebooks/sota/DeepSeekV41Flash/validate_omitted_mechanisms.py

`omitted_mechanisms.py` writes each mechanism the model leaves out, and the claims
about them live here, one `check_` function per mechanism, in the way
`validate_deepseek_v41_flash.py` holds the claims of the model's own notebook. Every
check is structural: the axes, the degrees, the datatypes, the affine forms and the
places the generic operators stand are compared, and no listing is diffed.
'''
from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable

# The repository root, for a run by path rather than through
# `validations/run_validations.py`, which puts it on PYTHONPATH itself.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa: E402
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.StrideCategory as sc  # noqa: E402
import para.data_structure.inject as inject  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402

from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.candidate_pool as candidate_pool  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.clamped_mixture_of_experts as clamped_mixture_of_experts  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.dspark_draft_chain as dspark_draft_chain  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.gumbel_max_sampler as gumbel_max_sampler  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.mhc_with_epsilons as mhc_with_epsilons  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.pinned_candidate_pool as pinned_candidate_pool  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.quantised_caches as quantised_caches  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.released_constants as released_constants  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.rotary_embedding as rotary_embedding  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.rotated_indexer as rotated_indexer  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.scaled_attention_core as scaled_attention_core  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.vision_pathway as vision_pathway  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.whole_model as whole_model  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.write_at_token_positions as write_at_token_positions  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (  # noqa: E402
    axes, node_with_operator)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (  # noqa: E402
    a, b, c, d, i, m, n, x)
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (  # noqa: E402
    G, Hp, K, L, U, W, Wp, H, t, z, zbar)

LOOKBACK_SIZES = {L.local_size(): 4, a.local_size(): 2, b.local_size(): 8}
PAIR_INDICES = cat.Array(cat.Natural(t.local_size()), (t,))
TOKEN_POSITIONS = cat.Array(cat.Natural(x.local_size()), (x,))
IMAGINARY_UNIT = nm.Constant(nm.ConstantSymbol.IMAGINARY_UNIT)


def generic_names(term: cat.Morphism) -> list[str]:
    '''The body of the name of every `GenericOperator` in `term`, sorted.'''
    return sorted(operator.name.body
                  for operator in tutil.type_search(ops.GenericOperator, term))


def engram_module_of_the_first_layer() -> cat.Morphism:
    return omitted_mechanisms.engram_module(omitted_mechanisms.ENGRAM_LAYERS[0])


def check_the_generic_operators() -> None:
    '''Each mechanism holds a generic operator at the operations the notebook names and
    at no other, and the mechanisms the notebook calls expressible hold none.'''
    expected = {
        rotary_embedding.rotation_defined: [],
        rotary_embedding.yarn_ramp_defined: ['r'],
        rotary_embedding.yarn_frequencies_defined: ["\\theta'"],
        scaled_attention_core.scale_scores: [],
        rotated_indexer.scale_head_weights: [],
        pinned_candidate_pool.select_candidate_blocks_of_every_token: [],
        engram_module_of_the_first_layer: [],
        dspark_draft_chain.draft_five_tokens: ['\\mathrm{Exp}(1)', '\\mathrm{draft}'],
        gumbel_max_sampler.sample_from_probabilities: ['\\mathrm{Exp}(1)'],
        dspark_draft_chain.head_over_draft_positions: [],
        dspark_draft_chain.confidence_head: [],
        vision_pathway.image_pathway_with_delimiters: ['\\mathrm{ViT}'],
        vision_pathway.write_cell_features: [],
        quantised_caches.entry_round_trip: [],
        quantised_caches.indexer_round_trip: ['\\lceil x \\rceil'],
        quantised_caches.window_round_trip: ['\\lceil x \\rceil'],
        omitted_mechanisms.compress_step: ['\\mathrm{step}'],
        clamped_mixture_of_experts.shared_expert: [],
        clamped_mixture_of_experts.mix_one_token: [],
        clamped_mixture_of_experts.temper_scores: [],
        clamped_mixture_of_experts.bias_of_modality: [],
        mhc_with_epsilons.sinkhorn: [],
        mhc_with_epsilons.predict_one_token: [],
    }
    for build, names in expected.items():
        assert generic_names(build()) == names, (build.__name__, generic_names(build()))


def stride_morphism_named(name: str, term: cat.Morphism) -> sc.StrideMorphism:
    found, = {row for row in tutil.type_search(sc.StrideMorphism, term)
              if row.name is not None and row.name.to_bodies() == name}
    return found


def table_of(rotation: cat.Broadcasted) -> cat.Broadcasted:
    '''The one `dst.Rotary` of a rotation box, which holds the factor of every
    position and pair.'''
    table, = (node for node in tutil.type_search(cat.Broadcasted, rotation)
              if isinstance(node.operator, dst.Rotary))
    return table


def check_the_rotary_embedding() -> None:
    '''A rotation site cuts the vector of every position into the channels the rotation
    leaves alone and the channels it rotates, reads the rotated channels as complex
    numbers, multiplies each pair by the factor its table holds for the position and
    the pair, writes the pairs back as reals and joins the two runs onto the channel
    axis the vector arrived on. The site of the token latents is one box over the
    latent, with no generic operator. A layer holding compressed entries turns with
    the YaRN table at the rotary base, the two sliding-window layers with the plain
    table at the window base, an entry is turned at the position of the first token
    of its group, the attention output is turned back by the conjugate of the table,
    and an indexer array holds the same rotated channels inside a key of width d.'''
    rotate_latents = rotary_embedding.ROTATE_TOKEN_LATENTS[rotary_embedding.RotaryKind.YARN]
    assert axes(rotate_latents) == ([['x', 'c']], [['x', 'c']])
    assert isinstance(rotate_latents.operator, ops.BlockOperator)
    assert generic_names(rotate_latents) == []
    cut = node_with_operator(aops.DeconcatenateAxes, rotate_latents)
    assert cut.operator.parts() == (zbar, z)
    assert cut.operator.concatenated_axis() is c
    assert z.local_size() == nm.Integer(2) * t.local_size()
    assert nm.is_zero(nm.collect_like_terms(
        zbar.local_size() + z.local_size() - c.local_size()))
    pairing = node_with_operator(dst.PairsAsComplex, rotate_latents)
    assert pairing.output_weaves[0].datatype == omitted_mechanisms.COMPLEX
    assert tuple(pairing.output_weaves[0].target().shape()) == (t,)
    turn = node_with_operator(ops.Einops, rotate_latents)
    assert [weave.datatype for weave in turn.input_weaves] == [
        omitted_mechanisms.COMPLEX, omitted_mechanisms.COMPLEX]
    assert tuple(turn.degree()) == (x, t)
    assert tuple(node_with_operator(dst.Decomplex, rotate_latents).cod()[0].shape()) == (
        x, z)
    assert node_with_operator(aops.ConcatenateAxes, rotate_latents).operator.parts() == (
        zbar, z)

    yarn = table_of(rotate_latents).operator
    assert isinstance(yarn, dst.YarnRotary)
    assert yarn.base == released_constants.ROTARY_BASE
    assert yarn.factor == released_constants.YARN_FACTOR
    assert (yarn.ramp_start, yarn.ramp_end) == (
        released_constants.YARN_RAMP_START, released_constants.YARN_RAMP_END)
    assert yarn.position_stride == nm.Integer(1)
    plain = table_of(
        rotary_embedding.ROTATE_TOKEN_LATENTS[rotary_embedding.RotaryKind.PLAIN]).operator
    assert not isinstance(plain, dst.YarnRotary)
    assert plain.base == released_constants.WINDOW_ROTARY_BASE
    assert table_of(rotary_embedding.ROTATE_ENCODER_ENTRIES).operator.position_stride == (
        a.local_size())
    turned_back = rotary_embedding.ROTATE_ATTENTION_OUTPUT_BACK[
        rotary_embedding.RotaryKind.YARN]
    conjugate, = (node for node in tutil.type_search(cat.Broadcasted, turned_back)
                  if isinstance(node.operator, ops.Arithmetic)
                  and list(tutil.type_search(nm.Conjugate, node.operator.formula)))
    assert conjugate.output_weaves[0].datatype == omitted_mechanisms.COMPLEX
    for indexer_rotation in (rotary_embedding.ROTATE_INDEXER_QUERIES,
                             rotary_embedding.ROTATE_ENCODER_INDEXER_KEYS,
                             rotary_embedding.ROTATE_DECODER_INDEXER_KEYS):
        indexer_cut = node_with_operator(aops.DeconcatenateAxes, indexer_rotation)
        assert indexer_cut.operator.parts() == (rotary_embedding.dbar, z)
        assert indexer_cut.operator.concatenated_axis() is d


def reads_of_one_position(term: cat.Morphism) -> list[sc.StrideMorphism]:
    '''The reindexings of `term` with an empty domain and one row, each of which reads
    one position of one axis.'''
    return [row for row in tutil.type_search(sc.StrideMorphism, term)
            if row._dom == () and len(row._cod_stride_shift) == 1]


def nodes_with_operator[O: cat.Operator](
    kind: type[O],
    term: cat.Morphism,
) -> list[cat.Broadcasted]:
    return [node for node in tutil.type_search(cat.Broadcasted, term)
            if isinstance(node.operator, kind)]


def check_the_deconcatenation_defined() -> None:
    '''The deconcatenation is defined as a copy and one view per part, and the
    deconcatenation it defines is the one the rotation of the token latents applies.'''
    cut = rotary_embedding.deconcatenation_defined()
    assert isinstance(cut.left_hand_side.operator, aops.DeconcatenateAxes)
    assert cut.left_hand_side.operator.parts() == (zbar, z)
    assert cut.left_hand_side.operator == node_with_operator(
        aops.DeconcatenateAxes,
        rotary_embedding.ROTATE_TOKEN_LATENTS[rotary_embedding.RotaryKind.YARN]).operator
    assert len(nodes_with_operator(ops.View, cut.right_hand_side)) == 2
    assert not generic_names(cut.right_hand_side)


def check_the_ramp_over_every_pair() -> None:
    '''The ramp is defined over every pair. The index of every pair is an
    `ops.Arrange` over the pairs, and the ramp is one clamp to the unit interval
    applied to it, which reads a `Natural`, returns a real and holds no index. The
    definition reads no position of the pair axis.'''
    definition = rotary_embedding.yarn_ramp_defined()
    assert tuple(definition.left_hand_side.dom()) == tuple(
        definition.right_hand_side.dom())
    assert tuple(definition.left_hand_side.cod()) == tuple(
        definition.right_hand_side.cod()) == (rotary_embedding.VALUE_OF_EVERY_PAIR,)
    assert generic_names(definition.left_hand_side) == ['r']
    assert not generic_names(definition.right_hand_side)
    assert not reads_of_one_position(definition.left_hand_side)
    assert not reads_of_one_position(definition.right_hand_side)

    positions = node_with_operator(ops.Arrange, definition.right_hand_side)
    assert tuple(positions.cod()) == (PAIR_INDICES,)
    ramp = node_with_operator(ops.Arithmetic, definition.right_hand_side)
    assert ramp.input_weaves[0].datatype == cat.Natural(t.local_size())
    assert ramp.output_weaves[0].datatype == cat.Reals()
    assert tuple(ramp.degree()) == (t,)
    assert nm.contains_free_input(ramp.operator.formula)
    clamp, = tutil.type_search(nm.Clamp, ramp.operator)
    assert clamp.clamps_to_unit_interval()
    assert not list(tutil.type_search(nm.RectifiedLinear, ramp))
    assert {released_constants.YARN_RAMP_START, released_constants.YARN_RAMP_END} == set(
        tutil.type_search(nm.FreeNumeric, ramp.operator))
    assert ramp in nodes_with_operator(
        ops.Arithmetic, rotary_embedding.rotation_defined().right_hand_side)


def check_the_frequencies_over_every_pair() -> None:
    '''The YaRN frequencies are defined over every pair. The power of the base and the
    ramp are each one formula applied to the index of the pair, so each reads a
    `Natural`, returns a real and holds no index. One `ops.Arrange` over the pairs
    returns the indices, and a copy feeds both formulas. The factor that interpolates
    along the ramp is a third formula, applied to the ramp, and the frequencies are
    the product of the power and that factor.'''
    definition = rotary_embedding.yarn_frequencies_defined()
    assert tuple(definition.left_hand_side.dom()) == tuple(
        definition.right_hand_side.dom())
    assert tuple(definition.left_hand_side.cod()) == tuple(
        definition.right_hand_side.cod()) == (rotary_embedding.VALUE_OF_EVERY_PAIR,)
    assert not reads_of_one_position(definition.left_hand_side)
    assert not reads_of_one_position(definition.right_hand_side)
    assert generic_names(definition.left_hand_side) == ["\\theta'"]
    assert not generic_names(definition.right_hand_side)

    positions, = nodes_with_operator(ops.Arrange, definition.right_hand_side)
    assert tuple(positions.cod()) == (PAIR_INDICES,)
    copies = [rearrangement for rearrangement in tutil.type_search(
                  cat.Rearrangement, definition.right_hand_side)
              if tuple(rearrangement.mapping) == (0, 0)
              and tuple(rearrangement.dom()) == (PAIR_INDICES,)]
    assert len(copies) == 1

    formulas = nodes_with_operator(ops.Arithmetic, definition.right_hand_side)
    read_an_index = [node for node in formulas
                     if node.input_weaves[0].datatype == cat.Natural(t.local_size())]
    read_a_real = [node for node in formulas
                   if node.input_weaves[0].datatype == cat.Reals()]
    assert len(read_an_index) == 2 and len(read_a_real) == 1
    interpolation, = read_a_real
    power, = (node for node in read_an_index
              if released_constants.ROTARY_BASE in set(
                  tutil.type_search(nm.FreeNumeric, node.operator)))
    ramp, = (node for node in read_an_index if node is not power)
    assert ramp == node_with_operator(
        ops.Arithmetic, rotary_embedding.yarn_ramp_defined().right_hand_side)
    for formula_holder in formulas:
        assert formula_holder.output_weaves[0].datatype == cat.Reals()
        assert nm.contains_free_input(formula_holder.operator.formula)
    assert released_constants.YARN_FACTOR in set(
        tutil.type_search(nm.FreeNumeric, interpolation.operator))


def check_the_rotation_over_every_position() -> None:
    '''The table of turns is defined over every token and pair at once, as the
    standard expansion of the operator writes it. The position of every token is an
    `ops.Arrange` over the token axis with the datatype of an index of that axis, the
    angle is the product of the positions and the frequencies, whose operands carry
    two datatypes, and the exponential reads a real and returns a complex number. The
    definition reads no position of the token axis and holds no generic operator.'''
    definition = rotary_embedding.rotation_defined()
    assert tuple(definition.left_hand_side.dom()) == tuple(
        definition.right_hand_side.dom()) == ()
    assert tuple(definition.left_hand_side.cod()) == tuple(
        definition.right_hand_side.cod()) == (
            cat.Array(omitted_mechanisms.COMPLEX, (x, t)),)
    assert isinstance(definition.left_hand_side.operator, dst.YarnRotary)
    assert not reads_of_one_position(definition.left_hand_side)
    assert not reads_of_one_position(definition.right_hand_side)
    assert not generic_names(definition.right_hand_side)

    positions, = (node for node in nodes_with_operator(
                      ops.Arrange, definition.right_hand_side)
                  if tuple(node.cod()) == (TOKEN_POSITIONS,))
    assert positions.has_empty_domain()
    assert positions.operator.name.to_latex() == 'i_x'
    assert positions.output_weaves[0].datatype == cat.Natural(x.local_size())
    assert tuple(positions.degree()) == ()

    angles, = (node for node in nodes_with_operator(
                   ops.Einops, definition.right_hand_side)
               if tuple(node.degree()) == (x, t))
    assert [weave.datatype for weave in angles.input_weaves] == [
        cat.Natural(x.local_size()), cat.Reals()]
    assert angles.output_weaves[0].datatype == cat.Reals()
    assert angles.operator.signature == ((), ())

    factors, = (node for node in nodes_with_operator(
                    ops.Arithmetic, definition.right_hand_side)
                if node.output_weaves[0].datatype == omitted_mechanisms.COMPLEX)
    assert factors.input_weaves[0].datatype == cat.Reals()
    assert tuple(factors.degree()) == (x, t)
    assert IMAGINARY_UNIT in set(tutil.type_search(nm.Constant, factors))
    assert not list(tutil.type_search(nm.FreeNumeric, factors.operator))


def check_the_score_scales() -> None:
    '''Each scale is one formula applied to every value on a wire and has no axis of
    its own. The attention score is multiplied by the reciprocal square root of the
    latent width, inside the core that exponentiates the scores, and the weight of an
    indexer head by the reciprocal square root of the key width times the head count,
    inside the scoring box that sums the rectified scores under it. Each map carries
    the name its module gives a role to.'''
    score_scale = scaled_attention_core.scale_scores()
    head_weight_scale = rotated_indexer.scale_head_weights()
    for scale in (score_scale, head_weight_scale):
        assert isinstance(scale.operator, ops.Arithmetic)
        assert axes(scale) == ([[]], [[]])
        assert nm.contains_free_input(scale.operator.formula)
    assert score_scale.operator.formula == nm.x * nm.Power.template(
        c.local_size(), nm.Integer(-1) / nm.Integer(2))
    assert head_weight_scale.operator.formula == nm.x * nm.Power.template(
        d.local_size() * i.local_size(), nm.Integer(-1) / nm.Integer(2))
    assert score_scale.operator.name.to_bodies() == (
        scaled_attention_core.SCORE_SCALE_NAME)
    assert head_weight_scale.operator.name.to_bodies() == (
        rotated_indexer.HEAD_WEIGHT_SCALE_NAME)
    assert set(scaled_attention_core.ARITHMETIC_ROLES) == {
        scaled_attention_core.SCORE_SCALE_NAME}
    assert set(rotated_indexer.ARITHMETIC_ROLES) == {
        rotated_indexer.HEAD_WEIGHT_SCALE_NAME}
    applied = {
        scaled_attention_core.SCALED_CORE_ON_CONCATENATED_SLOTS.candidate: score_scale,
        scaled_attention_core.SCALED_CORE_ON_THE_WINDOW.candidate: score_scale,
        rotated_indexer.INDEXER_e: head_weight_scale,
        rotated_indexer.INDEXER_d: head_weight_scale,
    }
    for term, scale in applied.items():
        assert any(node.operator == scale.operator
                   for node in nodes_with_operator(ops.Arithmetic, term))


def check_the_pinned_block() -> None:
    '''The pool with the pin has the pool's own domain and codomain and is one box
    computed once per token, confirmed against the same selection written out over
    every token. The pin is positive infinity written at block 0 by a covariant view
    of the row that names position 0 and added to the block scores, with no generic
    operator.'''
    pinned = pinned_candidate_pool.PINNED_CANDIDATE_POOL
    assert axes(pinned) == axes(candidate_pool.CANDIDATE_POOL)
    assert isinstance(pinned.operator, ops.BlockOperator)
    assert tuple(pinned.degree()) == (x,)
    assert pinned_candidate_pool.PINNED_POOL_CONFIRMATION.is_confirmed()
    assert generic_names(pinned) == []
    pin = node_with_operator(ops.ConstantOp, pinned)
    assert pin.operator.value == nm.Constant(nm.ConstantSymbol.INFINITY)
    written, = (node for node in tutil.type_search(cat.Broadcasted, pinned)
                if isinstance(node.operator, aops.CovariantView)
                and node.operator.name.to_bodies() == pinned_candidate_pool.PIN)
    row, = written.operator.reindexing._cod_stride_shift
    assert row[0] is candidate_pool.P_reach and row[1] == () and row[2] == nm.Integer(0)
    assert written.operator.reindexing._dom == ()
    for kind in (ops.Maximum, ops.View, ops.AdditionOp):
        assert len(list(node for node in tutil.type_search(cat.Broadcasted, pinned)
                        if isinstance(node.operator, kind))) == 1


def engram_array_names(module: cat.Morphism) -> set[str]:
    '''The names of the learned arrays and the fixed arrays of an Engram module, which
    are the names that belong to one layer alone.'''
    return {node.operator.name.to_bodies()
            for node in (nodes_with_operator(ops.Linear, module)
                         + nodes_with_operator(ops.FixedArray, module))}


def check_engram() -> None:
    '''The module of an Engram layer holds no generic operator. The token map is an
    `ops.Embedding` that reads a token identifier and returns a compressed identifier,
    broadcast over the tokens. The lookback is a guarded affine read of the tokens
    ending at each position, and the lookup is an `Embedding` on the rows of the
    layer's table. The stream and the key pass through one normalisation, which is one
    term because it is the same operation on both, and it carries the released epsilon,
    no gain and no bias. The dot product is one `Einops` over the stream, the weight
    and the key, and the gate is a formula over the standard numerics. Every array that
    belongs to one layer carries the number of that layer, so the two modules share
    none.'''
    layer = omitted_mechanisms.ENGRAM_LAYERS[0]
    engram = omitted_mechanisms.engram_module(layer)
    assert axes(engram) == ([['x', 'n', 'm'], ['x']], [['x', 'n', 'm']])
    assert engram.dom()[1] == whole_model.v41_flash.dom()[0]
    assert generic_names(engram) == []
    lookback = omitted_mechanisms.L_reach
    assert isinstance(lookback, AffineGuards.AffineSparseAxis)
    assert lookback.guides == (x,)
    for query in range(6):
        assert mark_sparse_domains.live_positions(
            lookback, (query,), LOOKBACK_SIZES) == list(range(min(4, query + 1)))

    token_map = node_with_operator(
        ops.Embedding, omitted_mechanisms.map_tokens_to_compressed_identifiers())
    assert token_map.input_weaves[0].datatype == whole_model.v41_flash.dom()[0].datatype
    assert token_map.output_weaves[0].datatype == omitted_mechanisms.COMPRESSED_IDS
    assert tuple(token_map.degree()) == (x,)
    lookup = node_with_operator(ops.Embedding, omitted_mechanisms.look_up_rows(layer))
    assert lookup.input_weaves[0].datatype == (
        omitted_mechanisms.table_row_of_layer(layer))
    assert tuple(lookup.degree()) == (x, G, K)

    normalisation, = nodes_with_operator(ops.Normalize, engram)
    assert not normalisation.operator.gain and not normalisation.operator.bias
    assert normalisation.operator.epsilon == released_constants.NORM_EPSILON
    assert tuple(normalisation.degree()) == (x, n)
    product = node_with_operator(
        ops.Einops, omitted_mechanisms.weighted_dot_product(layer))
    assert len(product.input_weaves) == 3
    assert tuple(product.degree()) == (x, n)

    formula = omitted_mechanisms.signed_root_gate().operator.formula
    for numeric in (nm.Sigmoid, nm.Sign, nm.SquareRoot, nm.LargerOf, nm.AbsoluteValue):
        assert list(tutil.type_search(numeric, formula))
    assert nm.expand_every_expandable(formula) != formula
    assert not list(tutil.type_search(nm.Expandable, nm.expand_every_expandable(formula)))
    assert released_constants.ENGRAM_GATE_FLOOR in set(
        tutil.type_search(nm.FreeNumeric, formula))

    first, second = omitted_mechanisms.ENGRAM_LAYERS
    assert set(omitted_mechanisms.TABLE_ROWS_OF_LAYER) == {first, second}
    assert omitted_mechanisms.table_row_count(first).uid._name.to_bodies() == 'T1'
    assert omitted_mechanisms.table_row_count(first) != (
        omitted_mechanisms.table_row_count(second))
    for name in engram_array_names(engram):
        assert name.endswith(f'{{{layer}}}'), name
    assert not engram_array_names(engram) & engram_array_names(
        omitted_mechanisms.engram_module(second))
    try:
        omitted_mechanisms.engram_module(2)
    except omitted_mechanisms.LayerHoldsNoEngram:
        pass
    else:
        raise AssertionError('layer 2 gives an Engram module')


def check_the_written_out_hash() -> None:
    '''The hash of a layer holds no generic operator. A multiplier is below `2^63`
    divided by the identifier bound, so the products are 64-bit naturals with no cast
    and the exclusive or reads them as they are. The view named prefix gives the order
    `i_G` the newest `i_G + 2` slots. The primes are one fixed array over the `|G||K|`
    pairs, the offsets are the guarded sum of the primes before each pair, and both
    are read over the orders and the heads. The remainder carries the datatype of
    the primes, and the one cast narrows the row numbers to the rows of the layer's
    table.'''
    layer = omitted_mechanisms.ENGRAM_LAYERS[0]
    row = omitted_mechanisms.table_row_of_layer(layer)
    hashing = omitted_mechanisms.address_table_rows(layer)
    assert axes(hashing) == ([['x']], [['x', 'G', 'K']])
    hash_box = omitted_mechanisms.hash_ngrams(layer)
    assert isinstance(hash_box.operator, ops.BlockOperator)
    assert hash_box.operator.name.to_bodies() == f"{omitted_mechanisms.HASH_BOX}{{{layer}}}"
    assert hash_box.operator.block.block_tag.aesthetics.title == text.HASH_TITLE
    assert generic_names(hashing) == []
    assert ops.bit_width(omitted_mechanisms.INT64) == nm.Integer(63)
    assert omitted_mechanisms.PRODUCT == omitted_mechanisms.INT64
    assert omitted_mechanisms.MULTIPLIER.max_value == (
        omitted_mechanisms.INT64.max_value / omitted_mechanisms.COMPRESSED_IDS.max_value)
    cast_to_row, = nodes_with_operator(ops.Cast, hashing)
    assert cast_to_row.input_weaves[0].datatype == (
        omitted_mechanisms.remainder_plus_offset(layer))
    assert cast_to_row.output_weaves[0].datatype == row
    exclusive_or = node_with_operator(ops.BitwiseXor, hashing)
    assert exclusive_or.input_weaves[0].datatype == omitted_mechanisms.INT64
    before = omitted_mechanisms.GK_before
    assert isinstance(before, AffineGuards.AffineSparseAxis)
    assert before.guides == (omitted_mechanisms.GK,)
    pair_sizes = {G.local_size(): 3, K.local_size(): 8}
    for pair in (0, 1, 2, 23):
        assert mark_sparse_domains.live_positions(before, (pair,), pair_sizes) == list(
            range(24 - pair, 24))
    offsets, = (node for node in nodes_with_operator(ops.Einops, hashing)
                if node.input_weaves[0].datatype == omitted_mechanisms.PRIME)
    assert offsets.output_weaves[0].datatype == row
    pairs = [node for node in nodes_with_operator(ops.View, hashing)
             if node.operator.name is not None
             and node.operator.name.to_bodies() == 'pair']
    assert sorted(node.output_weaves[0].datatype.max_value.to_latex()
                  for node in pairs) == ['T_1', '\\bar{p}']
    prefix = omitted_mechanisms.L_prefix
    assert isinstance(prefix, AffineGuards.AffineSparseAxis)
    assert prefix.guides == (G,)
    sizes = {L.local_size(): 4, G.local_size(): 3}
    for order in range(3):
        assert mark_sparse_domains.live_positions(prefix, (order,), sizes) == list(
            range(2 - order, 4))
    remainder = node_with_operator(ops.Modulo, hashing)
    assert remainder.output_weaves[0].datatype == omitted_mechanisms.PRIME
    fixed = [node.operator.name.to_bodies()
             for node in nodes_with_operator(ops.FixedArray, hashing)]
    assert sorted(fixed) == ['\\alpha{1}', 'p{1}']


def generic_operator_named(name: str, term: cat.Morphism) -> cat.Broadcasted:
    found, = (node for node in nodes_with_operator(ops.GenericOperator, term)
              if node.operator.name.to_bodies() == name)
    return found


def check_dspark() -> None:
    '''The drafter reads the backbone's probabilities and the three stream means and
    hands out five drafts and five confidences. The draft trunk and the exponential
    draw of the sampler are its two generic operators. The three projections of the
    backbone state, the head, the Markov head and the confidence head are standard,
    the five steps read five draft positions, and a draft is a position on the
    vocabulary axis, which is the datatype the model embeds.'''
    drafter = dspark_draft_chain.draft_five_tokens()
    means = [['x', 'm']] * dspark_draft_chain.TARGET_LAYERS
    results = [[]] * (2 * dspark_draft_chain.DRAFT_STEPS)
    assert axes(drafter) == ([['x', 'v'], *means], results)
    assert generic_names(drafter) == sorted(
        (dspark_draft_chain.DRAFT_NAME, gumbel_max_sampler.DRAW_NAME))
    assert (drafter.cod()[0].datatype == gumbel_max_sampler.SAMPLED_TOKEN.datatype
            == cat.Natural(omitted_mechanisms.VOCABULARY.local_size()))
    assert drafter.dom()[0].shape()[1] is omitted_mechanisms.VOCABULARY

    trunk = generic_operator_named(dspark_draft_chain.DRAFT_NAME, drafter)
    assert [tuple(weave.target().shape()) for weave in trunk.input_weaves] == [(m,), ()]
    assert tuple(trunk.output_weaves[0].target().shape()) == (omitted_mechanisms.S, m)
    assert tuple(trunk.degree()) == ()

    assert axes(dspark_draft_chain.project_stream_means()) == (means, [['m']])
    assert axes(dspark_draft_chain.head_over_draft_positions()) == (
        [['S', 'm']], [['S', 'v']])
    assert axes(dspark_draft_chain.markov_head()) == ([[]], [['v'], ['Z']])
    assert axes(dspark_draft_chain.confidence_head()) == ([['m'], ['Z']], [[]])
    assert not generic_names(dspark_draft_chain.head_over_draft_positions())
    assert not generic_names(dspark_draft_chain.confidence_head())
    assert sorted(linear.name.to_bodies() for linear
                  in tutil.type_search(ops.Linear, dspark_draft_chain.confidence_head())
                  ) == ['W^{\\mathrm{conf}}Z', 'W^{\\mathrm{conf}}m']
    assert {view.name.to_bodies() for view
            in tutil.type_search(sc.StrideMorphism, drafter)
            if view.name is not None} == {
        dspark_draft_chain.LAST_POSITION_NAME, gumbel_max_sampler.ONLY_SLOT_NAME,
        *(dspark_draft_chain.table_key(
            dspark_draft_chain.draft_position_name(position))
          for position in range(dspark_draft_chain.DRAFT_STEPS))}

    sampler = gumbel_max_sampler.sample_from_probabilities()
    assert axes(sampler) == ([['v']], [[]])
    assert generic_names(sampler) == [gumbel_max_sampler.DRAW_NAME]
    draw = generic_operator_named(gumbel_max_sampler.DRAW_NAME, sampler)
    assert draw.has_empty_domain()
    assert tuple(draw.output_weaves[0].target().shape()) == (
        omitted_mechanisms.VOCABULARY,)
    selection = node_with_operator(dst.TopK, sampler)
    assert selection.operator.k == nm.Integer(1)
    assert selection.operator.form is dst.SelectionForm.ONLY_SELECTION
    assert tuple(selection.input_weaves[0].target().shape()) == (
        omitted_mechanisms.VOCABULARY,)
    reciprocal = node_with_operator(ops.Arithmetic, sampler)
    assert reciprocal.operator.formula == nm.Integer(1) / nm.x
    assert gumbel_max_sampler.SAMPLING_TEMPERATURE in set(
        tutil.type_search(nm.FreeNumeric, dspark_draft_chain.draft_step()))


def check_the_vision_pathway() -> None:
    '''The pathway reads the residual, the token position of every cell, the patches of
    one image and the position and the kind of every delimiter. The vision encoder is
    its one generic operator and consumes the whole patch grid. The pixel unshuffle is
    the product of two group views and one identity, so a cell is handed out as its two
    offsets and its features. The projector is two `Linear`s with a bias and a pointwise
    map. The write of a cell at its token is two `inject.Inject` and a product, and the
    kind of a delimiter selects one of the three learned vectors.'''
    pathway = vision_pathway.image_pathway_with_delimiters()
    assert axes(pathway) == (
        [['x', 'm'], ["H'", "W'"], ['H', 'W', 'F'], ['\\Delta'], ['\\Delta']],
        [['x', 'm']])
    assert generic_names(pathway) == [vision_pathway.VISION_ENCODER_NAME]
    encoder = generic_operator_named(vision_pathway.VISION_ENCODER_NAME, pathway)
    assert tuple(encoder.input_weaves[0].target().shape()) == (
        H, W, omitted_mechanisms.F)
    assert tuple(encoder.output_weaves[0].target().shape()) == (
        H, W, omitted_mechanisms.M)
    assert tuple(encoder.degree()) == ()

    assert vision_pathway.CELL_ROWS._dom == (Hp, U)
    assert vision_pathway.CELL_COLUMNS._dom == (Wp, omitted_mechanisms.V)
    for group_view, grid_axis, offset_axis in (
            (vision_pathway.CELL_ROWS, H, U),
            (vision_pathway.CELL_COLUMNS, W, omitted_mechanisms.V)):
        axis, strides, shift = group_view._cod_stride_shift[0]
        assert axis is grid_axis
        assert strides == (offset_axis.local_size(), nm.Integer(1))
        assert shift == nm.Integer(0)
    assert H.local_size() == U.local_size() * Hp.local_size()
    assert tuple(vision_pathway.CELLS_OF_THE_PATCH_GRID.cod()) == (
        H, W, omitted_mechanisms.M)

    projector = vision_pathway.project_cells()
    assert axes(projector) == ([['H', 'W', 'M']], [["H'", "W'", 'm']])
    assert not generic_names(projector)
    assert all(linear.operator.bias
               for linear in nodes_with_operator(ops.Linear, projector))
    assert node_with_operator(ops.Arithmetic, projector).operator.name.to_bodies() == (
        vision_pathway.GELU_NAME)

    write = write_at_token_positions.write_at_positions(
        omitted_mechanisms.IMAGE_POSITIONS, omitted_mechanisms.CELL_FEATURES)
    assert axes(write) == (
        [['x', 'm'], ["H'", "W'"], ["H'", "W'", 'm']], [['x', 'm']])
    assert not generic_names(write)
    injections = nodes_with_operator(inject.Inject, write)
    assert len(injections) == 2
    for injection in injections:
        assert tuple(injection.input_weaves[0].target().shape()) == (Hp, Wp)
        assert injection.input_weaves[0].datatype == (
            omitted_mechanisms.IMAGE_POSITIONS.datatype)
        assert tuple(injection.output_weaves[0].target().shape()) == (x,)
    assert sorted(tuple(injection.degree()) for injection in injections) == [(), (m,)]
    assert node_with_operator(ops.ConstantOp, write).operator.value == nm.Integer(1)

    delimiters = node_with_operator(ops.Linear, vision_pathway.delimiter_vectors())
    assert ops.selects_weights(delimiters)
    assert delimiters.input_weaves[0].datatype == vision_pathway.DELIMITER_KIND
    assert vision_pathway.DELIMITERS.local_size() == Hp.local_size() + nm.Integer(2)


def converts_over(term: cat.Morphism, degree: tuple[cat.Axis, ...]) -> set[
        tuple[cat.Datatype, cat.Datatype]]:
    '''The source and the target of every `Quantization.TypeConvert` of `term` that is
    broadcast over `degree`.'''
    return {(node.operator.source, node.operator.target)
            for node in nodes_with_operator(Quantization.TypeConvert, term)
            if tuple(node.degree()) == degree}


def check_the_fp4_cache() -> None:
    '''Each of the three round trips reads a latent and returns its own shape, cuts
    the channels into the groups that share one scale with the view named qgrp and
    merges them back with the covariant view of the same reindexing, and converts every
    grouped channel into the form it stores and back to the reals. The entry trip
    stores four bits and converts the scale of a group into the eight-bit form, which
    the standard operators state without a generic operator. The indexer trip stores
    four bits and the window trip eight, and each rounds the scale of a group up to a
    power of two, where the ceiling is the only generic operator.'''
    reals = cat.Reals()
    assert quantised_caches.FP4 is omitted_mechanisms.FP4
    assert (quantised_caches.FP4.wraps, quantised_caches.FP4.size,
            quantised_caches.FP4.form) == (reals, nm.Integer(4), Quantization.Encoding.E2M1)
    assert (quantised_caches.FP8.wraps, quantised_caches.FP8.size,
            quantised_caches.FP8.form) == (reals, nm.Integer(8), Quantization.Encoding.E4M3)
    assert quantised_caches.ENTRY_GROUPS.group_channels is not (
        quantised_caches.WINDOW_GROUPS.group_channels)
    assert quantised_caches.INDEXER_GROUPS.group_channels is (
        quantised_caches.WINDOW_GROUPS.group_channels)

    trips = (
        (quantised_caches.ENTRY_ROUND_TRIP, quantised_caches.ENTRY_GROUPS,
         quantised_caches.FP4, []),
        (quantised_caches.INDEXER_ROUND_TRIP, quantised_caches.INDEXER_GROUPS,
         quantised_caches.FP4, [quantised_caches.CEILING]),
        (quantised_caches.WINDOW_ROUND_TRIP, quantised_caches.WINDOW_GROUPS,
         quantised_caches.FP8, [quantised_caches.CEILING]))
    for trip, groups, stored, generics in trips:
        latent = cat.Array(reals, (groups.channels,))
        assert tuple(trip.dom()) == tuple(trip.cod()) == (latent,)
        assert isinstance(trip.operator, ops.BlockOperator)
        assert generic_names(trip) == generics

        split = stride_morphism_named(quantised_caches.SCALE_GROUP_SPLIT, trip)
        assert split._dom == groups.grouped_axes()
        (axis, strides, shift), = split._cod_stride_shift
        assert axis is groups.channels
        assert strides == (groups.group_channels.local_size(), nm.Integer(1))
        assert shift == nm.Integer(0)
        cut = node_with_operator(ops.View, trip)
        assert cut.reindexings == (split,)
        merged = node_with_operator(aops.CovariantView, trip)
        assert merged.operator.reindexing == split
        assert tuple(merged.input_weaves[0].target().shape()) == groups.grouped_axes()
        assert tuple(merged.output_weaves[0].target().shape()) == (groups.channels,)

        assert converts_over(trip, groups.grouped_axes()) == {
            (reals, stored), (stored, reals)}

    entry_scale = converts_over(quantised_caches.ENTRY_ROUND_TRIP, (omitted_mechanisms.E,))
    assert entry_scale == {(reals, quantised_caches.FP8), (quantised_caches.FP8, reals)}
    for trip, groups, _, _ in trips[1:]:
        assert converts_over(trip, (groups.groups,)) == set()


def check_the_swiglu_clamps() -> None:
    '''An expert limits its gate branch from above and its up branch on both sides
    before it multiplies the two, so both limits stand inside `clamped_swiglu` and the
    expert that holds it reads and returns one token's hidden state. The limit from
    above is a formula over the rectified linear function, and the limit on both sides
    is an `nm.Clamp` between minus the limit and the limit, which expands to
    indicators.'''
    expert = clamped_mixture_of_experts.shared_expert()
    assert axes(expert) == ([['m']], [['m']])
    assert generic_names(expert) == []
    above = clamped_mixture_of_experts.clamp_from_above()
    both = clamped_mixture_of_experts.clamp_on_both_sides()
    assert isinstance(above.operator, ops.Arithmetic)
    assert isinstance(both.operator, ops.Arithmetic)
    formulas = {node.operator.formula
                for node in tutil.type_search(cat.Broadcasted, expert)
                if isinstance(node.operator, ops.Arithmetic)}
    assert above.operator.formula in formulas
    assert both.operator.formula in formulas
    assert len(list(tutil.type_search(nm.RectifiedLinear, above))) == 1
    clamp, = tutil.type_search(nm.Clamp, both)
    assert (clamp.lower, clamp.upper) == (
        -released_constants.SWIGLU_LIMIT, released_constants.SWIGLU_LIMIT)
    assert nm.is_zero(nm.collect_like_terms(
        nm.expand_every_expandable(clamp)
        - (-released_constants.SWIGLU_LIMIT
           + (nm.x + released_constants.SWIGLU_LIMIT)
           * nm.IsPositive(nm.x + released_constants.SWIGLU_LIMIT)
           - (nm.x - released_constants.SWIGLU_LIMIT)
           * nm.IsPositive(nm.x - released_constants.SWIGLU_LIMIT))))


def check_the_router_extras() -> None:
    '''The gate box reads one token's hidden state and its modality and hands out the
    six gates. The temperature is a pointwise map over the router's logits, the
    correction bias is a `Linear` whose index input selects one of its two weight
    vectors, and the gates are divided by their sum plus the router's epsilon by one
    `ops.L1Norm` carrying that epsilon.'''
    gate, ke = clamped_mixture_of_experts.expert_gate()
    assert axes(gate) == ([['m'], []], [['k/e']])
    assert generic_names(gate) == []
    tempered = clamped_mixture_of_experts.temper_scores()
    assert isinstance(tempered.operator, ops.Arithmetic)
    assert released_constants.ROUTER_TEMPERATURE in set(
        tutil.type_search(nm.FreeNumeric, tempered.operator))
    bias = clamped_mixture_of_experts.bias_of_modality()
    assert axes(bias) == ([[]], [['e']])
    assert ops.selects_weights(bias)
    assert bias.input_weaves[0].datatype == released_constants.MODALITY
    normalisation = node_with_operator(
        ops.L1Norm, clamped_mixture_of_experts.normalise_gates(ke))
    assert normalisation.operator.epsilon == released_constants.ROUTER_EPSILON
    assert tuple(normalisation.input_weaves[0].target().shape()) == (ke,)


def check_the_compressor_state() -> None:
    step = omitted_mechanisms.compress_step()
    assert axes(step) == ([['c'], ['c'], ['a', 'c'], ['a', 'c']],
                          [['a', 'c'], ['a', 'c'], ['c']])


def check_the_epsilons() -> None:
    '''The Sinkhorn normalisation runs in the released order: one first round, whose
    division of the rows is exact and whose division of the columns carries the epsilon
    of the hyper-connections, and nineteen later rounds that carry that epsilon in both
    divisions. Every division is one `ops.L1Norm`. The four streams are normalised with
    the epsilon of the released configuration under the root and with neither a gain
    nor a bias, and the epsilon is added by one formula, which stands after the sigmoid
    of the collapse vector and after the row softmax of the combine matrix.'''
    sinkhorn = mhc_with_epsilons.sinkhorn()
    assert axes(sinkhorn) == ([['n', 'N']], [['n', 'N']])
    assert generic_names(sinkhorn) == []
    rounds = {block.block_tag.repetition: block
              for block in tutil.type_search(cat.Block, sinkhorn)}
    assert set(rounds) == {nm.Integer(1),
                           nm.Integer(mhc_with_epsilons.LATER_SINKHORN_ROUNDS)}
    divisions = {node.operator.epsilon
                 for node in tutil.type_search(cat.Broadcasted, sinkhorn)
                 if isinstance(node.operator, ops.L1Norm)}
    assert divisions == {ops.NO_EPSILON, released_constants.MHC_EPSILON}
    exact = [node for node in tutil.type_search(cat.Broadcasted, rounds[nm.Integer(1)])
             if isinstance(node.operator, ops.L1Norm)
             and node.operator.epsilon == ops.NO_EPSILON]
    assert len(exact) == 1

    streams = mhc_with_epsilons.normalise_streams_without_gain()
    assert streams.operator.epsilon == released_constants.NORM_EPSILON
    assert not streams.operator.gain and not streams.operator.bias

    coefficients = mhc_with_epsilons.predict_one_token()
    assert axes(coefficients) == ([['n', 'm']], [['n'], ['n'], ['n', 'N']])
    assert generic_names(coefficients) == []
    added = {node.operator.formula
             for node in tutil.type_search(cat.Broadcasted, coefficients)
             if isinstance(node.operator, ops.Arithmetic)}
    assert mhc_with_epsilons.add_mhc_epsilon().operator.formula in added
    assert released_constants.MHC_EPSILON in set(
        tutil.type_search(nm.FreeNumeric, coefficients))


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_generic_operators,
    check_the_rotary_embedding,
    check_the_deconcatenation_defined,
    check_the_ramp_over_every_pair,
    check_the_frequencies_over_every_pair,
    check_the_rotation_over_every_position,
    check_the_score_scales,
    check_the_pinned_block,
    check_engram,
    check_the_written_out_hash,
    check_dspark,
    check_the_vision_pathway,
    check_the_fp4_cache,
    check_the_swiglu_clamps,
    check_the_router_extras,
    check_the_compressor_state,
    check_the_epsilons,
)


def run_checks() -> int:
    '''Every check, one line each, and the number that failed.'''
    failures = 0
    for check in CHECKS:
        try:
            check()
        except Exception as reason:
            failures += 1
            print(f'FAILED  {check.__name__}: {type(reason).__name__}: {reason}')
        else:
            print(f'ok      {check.__name__}')
    return failures


if __name__ == '__main__':
    failed = run_checks()
    print(f'{len(CHECKS) - failed} of {len(CHECKS)} omitted-mechanism checks passed')
    sys.exit(1 if failed else 0)
