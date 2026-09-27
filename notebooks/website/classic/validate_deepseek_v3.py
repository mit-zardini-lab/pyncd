# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Check every claim `notebooks/website/classic/DeepSeekV3.ipynb` makes about the
released DeepSeek-V3.

    python notebooks/website/classic/validate_deepseek_v3.py

The notebook holds the prose and the figures and no check. The claims live here, one
`check_` function per claim, in the order the notebook makes them: the sizes, the
attention and its causal read, the rotary embedding, the gate, the mixture of experts,
the layers, the quantisations, the cached pass, and the interactive page. The script
prints one line per check and the time the run took, and exits non-zero on a failure.

The models are built in `notebooks/classic/released_deepseek_v3.py`,
`quantised_deepseek_v3.py`, `cached_deepseek_v3.py` and `deepseek_v3_page_variants.py`.
`check_the_grouped_choice_is_the_released_choice` runs the gate of the released code
and the gate of the expression on random scores with numpy, because the two choose the
experts through different arrays.
'''
from __future__ import annotations

import dataclasses
import pathlib
import sys
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy  # noqa: E402

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation  # noqa: E402,E501
import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import agent_display  # noqa: E402
import caching.data_structure.Caching as Caching  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import graphs.processing.Hypergraph2Morphism as h2m  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402

import notebooks.classic.cached_deepseek_v3 as cached_deepseek_v3  # noqa: E402
import notebooks.classic.deepseek_v3 as deepseek_v3  # noqa: E402
import notebooks.classic.deepseek_v3_page_variants as deepseek_v3_page_variants  # noqa
import notebooks.classic.quantised_deepseek_v3 as quantised_deepseek_v3  # noqa: E402
import notebooks.classic.released_deepseek_v3 as released_deepseek_v3  # noqa: E402
import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.sota_figures as figures  # noqa: E402
from notebooks.classic.released_deepseek_v3_wording import (  # noqa: E402
    TEXT as released_text)
from notebooks.display.explain_operators import shows_whole_formula  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axes  # noqa: E402

MODEL = released_deepseek_v3.MODEL
DECODE = released_deepseek_v3.decode_form()
CACHED = cached_deepseek_v3.CACHED_PASS
SIZES = released_deepseek_v3.released_assigned_sizes()
BF16 = Quantization.BF16
FP32 = Quantization.FP32
INT64 = Quantization.INT64

NOTEBOOK_DIAGRAMS = released_deepseek_v3.with_explanation_tables(
    notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.INLINE,
        dark_mode=notebook_diagrams.ColorMode.LIGHT,
        axis_sizes=figures.AxisSizes.SUBSCRIPT,
        assigned_sizes=SIZES,
        advanced_display=notebook_diagrams.AdvancedDisplay.LEGEND,
        title='DeepSeek-V3'))
'''The settings of the setup cell of the notebook.'''
QUANTISED_DIAGRAMS = quantised_deepseek_v3.with_quantised_explanation_tables(
    dataclasses.replace(NOTEBOOK_DIAGRAMS, clean_quantisation_labels=False))
PAGE = deepseek_v3_page_variants.page_settings(QUANTISED_DIAGRAMS)
VARIANTS = deepseek_v3_page_variants.page_variants(PAGE)
VARIANTS_BY_IDENTIFIER = {variant.identifier: variant for variant in VARIANTS}

RELEASED_BIAS_RANGE = (2.0271353721618652, 8.049522399902344)
'''The smallest and the largest correction bias of the 58 layers of the mixture of
experts, read from the shards of `deepseek-ai/DeepSeek-V3` at `e815299` on
2026-09-27.'''


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def operations(term: object) -> tuple[cat.Broadcasted, ...]:
    return tuple(tutil.type_search(cat.Broadcasted, term))


def named(name: str, term: object) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in operations(term)
                 if node.operator.name is not None
                 and node.operator.name.to_bodies() == name)


def one_named(name: str, term: object) -> cat.Broadcasted:
    found = set(named(name, term))
    require(len(found) == 1, f'{len(found)} operations are named {name}')
    return found.pop()


def body_of(name: str, term: object) -> cat.Morphism:
    return released_deepseek_v3.box_named(name, term).operator.block


def shape_names(array: cat.Array) -> list[str]:
    return [axis.uid._name.to_bodies() if axis.uid._name is not None else '?'
            for axis in array.shape()]


def quantisation_of(array: cat.Array) -> Quantization.Quantified | None:
    return Quantization.quantisation_of(array.datatype)


def recycled_box_listings(term: object) -> dict[str, set[str]]:
    '''The listing of the recycled body of every box of `term`, by the name of the box,
    so that two bodies holding the same operations in another order compare equal.'''
    found: dict[str, set[str]] = {}
    for node in operations(term):
        if isinstance(node.operator, ops.BlockOperator):
            found.setdefault(node.operator.name.to_bodies(), set()).add(
                agent_display.listing(h2m.recycle(node.operator.block)))
    return found


def presented_terms() -> dict[str, tuple[object, notebook_diagrams.DiagramSettings]]:
    '''The term and the settings of every variant of the page, a derived variant taking
    the Python statement of its functor.'''
    return {
        variant.identifier: (
            notebook_diagrams.term_of_page_variant(variant, VARIANTS_BY_IDENTIFIER),
            notebook_diagrams.settings_of_page_variant(
                variant, VARIANTS_BY_IDENTIFIER, PAGE))
        for variant in VARIANTS}


# ==========================================================================
# The sizes and the ends of the model.
# ==========================================================================
def check_the_sizes_are_those_of_config_671b() -> None:
    expected = {'v': 129280, 'm': 7168, 'h': 128, '\\ellq': 1536, '\\ell': 512,
                'dn': 128, 'dr': 64, 'dv': 128, 'u': 18432, 'n': 256, 'g': 8, 'j': 32,
                'p': 4, 'b': 2, 'c': 128, 'k': 8, 'f': 2048, 'S': 1, 'D': 3, 'L': 58}
    require(SIZES == expected, f'the model binds {SIZES}')
    require(SIZES['g'] * SIZES['j'] == SIZES['n']
            and SIZES['p'] * SIZES['j'] == SIZES['c'],
            'the groups hold every expert, and the candidates every expert of a kept '
            'group')
    require(SIZES['D'] + SIZES['L'] == 61, 'the dense and mixture layers make 61')


def check_the_model_reads_identifiers_and_returns_scores() -> None:
    require(axes(MODEL) == ([['x']], [['x', 'v']]),
            f'the model reads and returns {axes(MODEL)}')


# ==========================================================================
# Multi-head latent attention and its causal read.
# ==========================================================================
def check_the_query_and_key_carry_d_channels_and_the_value_dv() -> None:
    require(axes(released_deepseek_v3.query_generation_through_a_low_rank())
            == ([['x', 'm']], [['x', 'h', 'd']]),
            'the queries of every head carry d channels')
    require(axes(deepseek_v3.key_value_generation())
            == ([['x', 'm']], [['x', 'h', 'd'], ['x', 'h', 'dv']]),
            'the keys carry d channels and the values d_v')
    require(deepseek_v3.d.local_size()
            == deepseek_v3.dn.local_size() + deepseek_v3.dr.local_size(),
            'd is d_n followed by d_r')


def check_the_queries_pass_through_a_low_rank() -> None:
    attention = released_deepseek_v3.multi_head_latent_attention()
    down = one_named('W^{DQ}', attention)
    require(shape_names(down.cod()[0]) == ['x', '\\ellq'],
            'W^DQ projects the hidden state to the query latent')
    norms = [node for node in operations(attention)
             if isinstance(node.operator, ops.Normalize)
             and shape_names(node.dom()[0]) == ['x', '\\ellq']]
    require(len(norms) == 1, 'an RMSNorm normalises the query latent')
    for name, channels in (('W^{UQ}', 'dn'), ('W^{QR}', 'dr')):
        node = one_named(name, attention)
        require(shape_names(node.dom()[0]) == ['x', '\\ellq']
                and shape_names(node.cod()[0]) == ['x', 'h', channels],
                f'{name} projects the query latent to {channels} channels per head')


def check_the_keys_and_values_come_from_one_latent() -> None:
    attention = released_deepseek_v3.multi_head_latent_attention()
    require(shape_names(one_named('W^{DKV}', attention).cod()[0]) == ['x', '\\ell'],
            'W^DKV projects to a latent of l values per token')
    require(shape_names(one_named('W^{KR}', attention).cod()[0]) == ['x', 'dr'],
            'the turned key channels are projected once per token, with no head axis')
    for name, channels in (('W^{UK}', 'dn'), ('W^{UV}', 'dv')):
        node = one_named(name, attention)
        require(shape_names(node.dom()[0]) == ['x', '\\ell']
                and shape_names(node.cod()[0]) == ['x', 'h', channels],
                f'{name} projects the latent to {channels} channels per head')


def check_each_part_of_a_joined_weight_is_a_map_of_its_own() -> None:
    linear_names = {node.operator.name.to_bodies() for node in operations(MODEL)
                    if isinstance(node.operator, ops.Linear)}
    parts = {'W^{UQ}', 'W^{QR}', 'W^{DKV}', 'W^{KR}', 'W^{UK}', 'W^{UV}'}
    require(parts <= linear_names, f'the model lacks {sorted(parts - linear_names)}')


def check_the_mask_is_a_read_with_no_mask_operator() -> None:
    view = deepseek_v3.read_back_from_every_position(
        deepseek_v3.x, deepseek_v3.w, (deepseek_v3.h, deepseek_v3.d),
        deepseek_v3.MASK_NAME)
    slots = view.cod()[0].shape()[1]
    require(isinstance(slots, AffineGuards.AffineSparseAxis),
            'the view named Mask marks its slot axis')
    live = [mark_sparse_domains.live_positions(
        slots, (token,), {deepseek_v3.x.local_size(): 4}) for token in range(4)]
    require(live == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]],
            f'in a sequence of four tokens the live slots are {live}')
    require(not any(isinstance(node.operator, ops.WeightedTriangularLower)
                    for node in operations(MODEL)), 'the model holds no mask operator')


def check_the_decode_form_reads_the_hidden_state_at_the_mask() -> None:
    attention = body_of(deepseek_v3.ATTENTION_BOX, DECODE)
    masks = [leaf.wraps for leaf in quantise_model.graph_leaves(attention)
             if isinstance(leaf.wraps, cat.Broadcasted)
             and leaf.wraps.operator.name is not None
             and leaf.wraps.operator.name.to_bodies() == deepseek_v3.MASK_NAME]
    require(len(masks) == 1, f'{len(masks)} masks in the attention of the decode form')
    mask, = masks
    require(tuple(mask.dom()) == (deepseek_v3.STATE,),
            'the one mask reads the hidden state at the copy')
    for name in ('W^{DKV}', 'W^{KR}'):
        read = shape_names(one_named(name, attention).dom()[0])
        require(read == ['x', 'w|x', 'm'],
                f'{name} runs over the tokens and the slots, reading {read}')


# ==========================================================================
# The rotary embedding and YaRN.
# ==========================================================================
def check_one_box_turns_keys_once_per_token_and_queries_once_per_head() -> None:
    turned = sorted(shape_names(node.dom()[0]) for node in operations(
        released_deepseek_v3.multi_head_latent_attention())
        if node.operator == deepseek_v3.ROTARY_EMBEDDING.operator)
    require(turned == [['x', 'dr'], ['x', 'h', 'dr']],
            f'the box RoPE turns {turned}')


def check_the_ramp_and_the_scale_of_the_scores() -> None:
    require(released_deepseek_v3.yarn_ramp() == (10, 23), 'the ramp runs from 10 to 23')
    require(round(released_deepseek_v3.attention_factor(), 4) == 1.3689, 'mu = 1.3689')
    require(round(released_deepseek_v3.score_scale(), 5) == 0.13523,
            'the scores are scaled by 0.13523')
    tables = [node.operator for node in operations(MODEL)
              if isinstance(node.operator, dst.YarnRotary)]
    require(bool(tables) and all(
        table.factor == deepseek_v3.YARN_FACTOR
        and table.ramp_start == deepseek_v3.YARN_RAMP_START
        and table.ramp_end == deepseek_v3.YARN_RAMP_END for table in tables),
        'every rotary table is the YaRN table with the factor kappa')


# ==========================================================================
# The gate.
# ==========================================================================
def check_the_gate_holds_the_released_steps() -> None:
    gate = released_deepseek_v3.GATE_BODY
    kinds = sorted(type(node.operator).__name__ for node in operations(gate))
    require(kinds == sorted(['Linear', 'Arithmetic', 'Linear', 'AdditionOp', 'View',
                             'TopK', 'Einops', 'TopK', 'MergedPositions',
                             'CovariantView', 'IndexSelect', 'TopK', 'Arithmetic',
                             'Select', 'L1Norm', 'Arithmetic']),
            f'the gate holds {kinds}')
    selections = {node.operator.form: node for node in operations(gate)
                  if isinstance(node.operator, dst.TopK)}
    best = selections[dst.SelectionForm.ONLY_WEIGHTS]
    kept = selections[dst.SelectionForm.ONLY_SELECTION]
    chosen = selections[dst.SelectionForm.WEIGHTS]
    require(shape_names(best.dom()[0]) == ['g', 'j'] and shape_names(best.cod()[0])
            == ['g', 'b'], 'the best biased scores of every group are values alone')
    require(shape_names(kept.dom()[0]) == ['g']
            and Quantization.holds_natural_numbers(kept.cod()[0].datatype),
            'the kept groups are positions alone')
    require(dst.selects_over_positions(chosen)
            and shape_names(chosen.dom()[0]) == ['c']
            and isinstance(chosen.cod()[0].shape()[0], dst.SparseAxis),
            'the chosen experts are selected among the candidates onto k/n')
    group, = [node for node in named(released_deepseek_v3.GROUP_VIEW_NAME, gate)
              if isinstance(node.operator, ops.View)]
    (row_axis, strides, shift), = released_deepseek_v3.GROUP_SPLIT._cod_stride_shift
    require(group.operator.name.to_bodies() == released_deepseek_v3.GROUP_VIEW_NAME
            and strides == (released_deepseek_v3.j.local_size(), nm.Integer(1))
            and shift == nm.Integer(0), 'expert i_j of group i_g is |j| i_g + i_j')


def released_choice(biased: numpy.ndarray) -> set[int]:
    '''The experts `Gate.forward` of the released code chooses from one token's biased
    scores: the groups outside the four best set to minus infinity, then the top
    eight.'''
    grouped = biased.reshape(SIZES['g'], SIZES['j'])
    group_scores = numpy.sort(grouped, axis=-1)[:, -SIZES['b']:].sum(axis=-1)
    kept = numpy.argsort(-group_scores)[:SIZES['p']]
    dropped = numpy.ones(SIZES['g'], dtype=bool)
    dropped[kept] = False
    masked = numpy.where(dropped[:, None], -numpy.inf, grouped).reshape(-1)
    return set(numpy.argsort(-masked)[:SIZES['k']].tolist())


def expression_choice(biased: numpy.ndarray) -> set[int]:
    '''The experts the gate of the expression chooses: the positions of the experts of
    the kept groups laid out along the candidate axis, and the top eight of the biased
    scores read there, reported as positions of the experts.'''
    width = SIZES['j']
    grouped = biased.reshape(SIZES['g'], width)
    best = numpy.sort(grouped, axis=-1)[:, -SIZES['b']:].sum(axis=-1)
    kept = numpy.argsort(-best)[:SIZES['p']]
    candidates = (width * kept[:, None] + numpy.arange(width)[None, :]).reshape(-1)
    return set(candidates[numpy.argsort(-biased[candidates])[:SIZES['k']]].tolist())


def check_the_grouped_choice_is_the_released_choice() -> None:
    generator = numpy.random.default_rng(965)
    low, high = RELEASED_BIAS_RANGE
    for _ in range(2000):
        scores = 1 / (1 + numpy.exp(-generator.normal(size=SIZES['n'])))
        biased = scores + generator.uniform(low, high, size=SIZES['n'])
        released = released_choice(biased)
        require(released == expression_choice(biased),
                'the candidates of the kept groups give the released eight experts')
        require(all(biased[expert] > 0 for expert in released),
                'every chosen biased score is positive, so the indicator is one')


def check_the_gate_box_is_its_body_once_per_token() -> None:
    require(released_deepseek_v3.GATE_CONFIRMATION.named_difference is None,
            f'{released_deepseek_v3.GATE_CONFIRMATION.named_difference}')
    require(released_deepseek_v3.GATE.degree() == cat.ProdObject((deepseek_v3.x,)),
            'the gate is computed once per token')


# ==========================================================================
# The mixture of experts and the layers.
# ==========================================================================
def check_the_routed_experts_read_the_chosen_slots() -> None:
    routed = released_deepseek_v3.routed_experts()
    chosen = released_deepseek_v3.chosen
    for name in ('W{1}', 'W{3}'):
        node, = set(named(name, routed))
        require(tuple(node.cod()[0].shape())[1] == chosen,
                f'{name} returns the chosen slots')
    diagonal, = set(named(deepseek_v3.DIAGONAL_NAME, routed))
    require(tuple(diagonal.cod()[0].shape())[1] == chosen,
            'the view named Diagonal keeps the output of the chosen expert')
    combine = [node for node in operations(deepseek_v3.mixture_from_gate(
        released_deepseek_v3.GATE, routed, released_deepseek_v3.shared_expert(),
        chosen))
        if isinstance(node.operator, ops.Einops) and len(node.input_weaves) == 2
        and any(axis == chosen for array in node.dom() for axis in array.shape())
        and all(axis != chosen for axis in node.cod()[0].shape())]
    require(len(combine) == 1, 'the gates and the experts meet in one contraction')


def check_the_shared_expert_is_one_swiglu_of_width_s() -> None:
    require(deepseek_v3.s.local_size()
            == deepseek_v3.SHARED_COUNT * deepseek_v3.f.local_size()
            and SIZES['S'] == 1, 'the shared expert has the width of one routed expert')


def check_three_dense_layers_and_fifty_eight_mixture_layers() -> None:
    for layers, count in ((released_deepseek_v3.dense_layers(), 'D'),
                          (released_deepseek_v3.mixture_layers(), 'L')):
        require(layers.dom() == layers.cod(), 'a layer returns the array it reads')
        require(layers.block_tag.repetition.uid._name.to_bodies() == count,
                f'the layer repeats {count} times')


# ==========================================================================
# The quantisations.
# ==========================================================================
def check_every_projection_weight_is_fp8_in_blocks_of_128() -> None:
    fp8 = quantised_deepseek_v3.FP8_BLOCK_WEIGHT
    require(fp8.form is Quantization.Encoding.E4M3 and fp8.scale is not None
            and fp8.scale.form is Quantization.Encoding.FP32
            and fp8.scale.channels == nm.Integer(128)
            and fp8.scale.rows == nm.Integer(128),
            'the weight format is E4M3 with an FP32 scale per 128 by 128 block')
    held = dict(quantised_deepseek_v3.WEIGHT_QUANTISATIONS)
    other = {'E': BF16, 'W^{g}': BF16, deepseek_v3.HEAD_NAME: BF16,
             released_deepseek_v3.ROUTER_BIAS_NAME: FP32}
    require(all(held[name] == quantisation for name, quantisation in other.items()),
            'the embedding, the router weight and the head are BF16, the bias FP32')
    projections = {name: quantisation for name, quantisation in held.items()
                   if name not in other}
    require(len(projections) == 11 and all(q == fp8 for q in projections.values()),
            f'the projections are held at {set(projections.values())}')


def check_no_activation_is_rounded_to_fp8() -> None:
    for name, term in (('decode', quantised_deepseek_v3.decode_quantised),
                       ('cached', quantised_deepseek_v3.cached_quantised)):
        eight_bit = [weave for weave in tutil.type_search(cat.Weave, term)
                     if (quantisation := Quantization.quantisation_of(weave.datatype))
                     is not None and quantisation.size == nm.Integer(8)]
        require(not eight_bit, f'the {name} pass holds {len(eight_bit)} 8-bit wires')
        linears = [node for node in operations(term)
                   if isinstance(node.operator, ops.Linear) and node.input_weaves]
        require(all(quantisation_of(array) == BF16
                    for node in linears for array in node.dom()),
                f'every projection of the {name} pass reads BF16')


def check_every_wire_carries_a_quantisation_and_every_operator_a_rule() -> None:
    for name, term in (('decode', quantised_deepseek_v3.decode_quantised),
                       ('cached', quantised_deepseek_v3.cached_quantised)):
        require(not quantised_deepseek_v3.unquantised_weaves(term),
                f'every wire of the {name} pass carries a quantisation')
        require(not quantised_deepseek_v3.operators_without_a_rule(term),
                f'every operator of the {name} pass has a rule')


def check_the_gate_chooses_in_fp32_and_weights_in_bf16() -> None:
    gate = body_of(released_deepseek_v3.GATE_BOX,
                   quantised_deepseek_v3.decode_quantised)
    kinds = {type(node.operator): node for node in operations(gate)}
    require(quantisation_of(kinds[ops.AdditionOp].cod()[0]) == FP32,
            'the biased copy of the scores is FP32')
    require(quantisation_of(kinds[dst.IndexSelect].cod()[0]) == FP32,
            'the candidates are chosen among FP32 scores')
    require(quantisation_of(kinds[dst.Select].cod()[0]) == BF16
            and quantisation_of(gate.cod()[0]) == BF16,
            'the gates are read from the BF16 scores and returned in BF16')


def check_the_rotary_embedding_computes_in_fp32() -> None:
    body = body_of(deepseek_v3.ROTARY_EMBEDDING.operator.name.to_bodies(),
                   quantised_deepseek_v3.decode_quantised)
    decomplex, = [node for node in operations(body)
                  if isinstance(node.operator, dst.Decomplex)]
    require(quantisation_of(decomplex.cod()[0]) == FP32
            and quantisation_of(body.cod()[0]) == BF16
            and quantisation_of(body.dom()[0]) == BF16,
            'the pairs are turned in FP32 between BF16 on either side')


def check_the_softmax_returns_bf16() -> None:
    for term in (quantised_deepseek_v3.decode_quantised,
                 quantised_deepseek_v3.cached_quantised):
        attention = body_of(deepseek_v3.ATTENTION_BOX, term)
        softmax, = {node for node in operations(attention)
                    if isinstance(node.operator, ops.SoftMax)}
        require(quantisation_of(softmax.cod()[0]) == BF16,
                'the softmax returns BF16')


def check_every_conversion_is_between_bf16_and_fp32() -> None:
    for term in (quantised_deepseek_v3.decode_quantised,
                 quantised_deepseek_v3.cached_quantised):
        pairs = set(quantised_deepseek_v3.cast_counts(term))
        require(pairs == {('BF16', 'FP32'), ('FP32', 'BF16')},
                f'the conversions read and write {pairs}')


def check_the_identifiers_and_the_positions_are_int64() -> None:
    term = quantised_deepseek_v3.decode_quantised
    require(quantisation_of(term.dom()[0]) == INT64, 'the token identifiers are INT64')
    positions = [array for node in operations(term)
                 if isinstance(node.operator, dst.TopK)
                 for array in node.cod()
                 if Quantization.holds_natural_numbers(array.datatype)]
    require(bool(positions)
            and all(quantisation_of(array) == INT64 for array in positions),
            'the positions picked by a top-k are INT64')


def check_stripping_returns_each_pass_in_the_reals() -> None:
    for name, stripped, original in (
            ('decode', quantised_deepseek_v3.decode_without_quantisations, DECODE),
            ('cached', quantised_deepseek_v3.cached_without_quantisations,
             CACHED.expression)):
        require(agent_display.listing(stripped) == agent_display.listing(original),
                f'the stripped {name} pass has the listing of the pass in the reals')
        require(recycled_box_listings(stripped) == recycled_box_listings(original),
                f'every box body of the stripped {name} pass is the body in the reals')


# ==========================================================================
# The cached pass.
# ==========================================================================
def check_the_attention_has_twenty_six_placements() -> None:
    require(len(cached_deepseek_v3.ATTENTION_PLACEMENTS) == 26,
            f'{len(cached_deepseek_v3.ATTENTION_PLACEMENTS)} placements')


def check_the_narrowest_placement_caches_the_latent_and_the_turned_key() -> None:
    widths = cached_deepseek_v3.cache_widths_of_the_attention()
    require(widths == {'c_{RMSNorm}': 512, 'c_{RoPE}': 64}, f'the caches hold {widths}')
    over_the_cache = cached_deepseek_v3.NARROWEST.computed_over_the_cache
    names = sorted(operator.name.to_bodies() for operator in over_the_cache
                   if isinstance(operator, ops.Linear))
    require(names == ['W^{UK}', 'W^{UV}'],
            f'the linear maps over the cache are {names}')
    require(any(isinstance(operator, aops.ConcatenateAxes)
                for operator in over_the_cache)
            and any(isinstance(operator, ops.View) for operator in over_the_cache),
            'the join of the key and the repeat of the turned key run over the cache')


def check_every_layer_caches_576_values_per_token() -> None:
    boxes = cached_deepseek_v3.attention_boxes(CACHED.expression)
    require(len(boxes) == 1, 'every layer runs one attention box')
    require(cached_deepseek_v3.values_cached_per_token() == 61 * 576 == 35136,
            'the 61 layers cache 35136 values per token')


def check_the_absorbed_order_takes_fewer_operations() -> None:
    written = cached_deepseek_v3.attention_operations(
        cached_deepseek_v3.WRITTEN_ORDER_PASS)
    absorbed = cached_deepseek_v3.attention_operations(CACHED)
    require(f'{written:.2e}' == '1.10e+12' and f'{absorbed:.2e}' == '9.50e+09',
            f'one layer takes {written:.3e} and {absorbed:.3e} operations')


def check_the_pass_is_the_released_absorb_mode() -> None:
    body = cached_deepseek_v3.cached_attention().operator.block
    leaves = quantise_model.graph_leaves(body)
    made_by = {wire: leaf for leaf in leaves for wire in leaf.cod}
    readers = {}
    for leaf in leaves:
        for wire in leaf.dom:
            readers.setdefault(wire, []).append(leaf)

    def leaf_named(name: str):
        found, = [leaf for leaf in leaves if isinstance(leaf.wraps, cat.Broadcasted)
                  and leaf.wraps.operator.name is not None
                  and leaf.wraps.operator.name.to_bodies() == name]
        return found

    key, value, queries = (leaf_named(name) for name in ('W^{UK}', 'W^{UV}', 'W^{UQ}'))
    require(not key.dom and not value.dom,
            'W^UK and W^UV are weights contracted in place of maps over the cache')
    caches = [leaf for leaf in leaves if isinstance(leaf.wraps, cat.Broadcasted)
              and isinstance(leaf.wraps.operator, Caching.Caching)]
    require(sorted(shape_names(leaf.wraps.dom()[0])[-1] for leaf in caches)
            == ['\\ell', 'dr'], 'the caches hold the latent and the turned key')
    require(not any(isinstance(leaf.wraps, cat.Broadcasted)
                    and isinstance(leaf.wraps.operator, ops.Linear) and leaf.dom
                    and any(isinstance(axis, AxisConcatenation.ConcatenatedAxis)
                            for array in leaf.wraps.dom() for axis in array.shape())
                    for leaf in leaves),
            'no linear map reads an array over the cached tokens')
    key_reader, = readers[key.cod[0]]
    require(queries.cod[0] in key_reader.dom, 'the queries meet W^UK before the scores')
    value_reader, = readers[value.cod[0]]
    weighted_sum = made_by[next(
        wire for wire in value_reader.dom if wire != value.cod[0])]
    softmax, = [leaf for leaf in leaves if isinstance(leaf.wraps, cat.Broadcasted)
                and isinstance(leaf.wraps.operator, ops.SoftMax)]
    require(softmax.cod[0] in weighted_sum.dom,
            'W^UV is applied after the weighted sum over the slots')


def check_the_pass_reads_the_new_tokens_alone() -> None:
    expression = CACHED.expression
    require(shape_names(expression.dom()[0]) == ['xnew']
            and shape_names(expression.cod()[0]) == ['xnew', 'v'],
            'the pass reads and returns the new tokens')
    require(not any(axis == deepseek_v3.x
                    for node in operations(expression)
                    for array in (*node.dom(), *node.cod()) for axis in array.shape()),
            'no operation of the pass carries the token axis of the model')


def check_the_caches_are_bf16_at_the_released_quantisations() -> None:
    caches = [node for node in operations(quantised_deepseek_v3.cached_quantised)
              if isinstance(node.operator, Caching.Caching)]
    require(len(caches) == 2 and all(
        quantisation_of(node.dom()[0]) == quantisation_of(node.cod()[0]) == BF16
        for node in caches), 'both caches hold BF16')


def check_the_rotary_table_is_read_at_the_new_tokens() -> None:
    reads = [node for node in operations(CACHED.expression)
             if isinstance(node.operator, ops.View)
             and any(isinstance(array.datatype, dst.Complex) for array in node.dom())]
    require(bool(reads) and all(
        shape_names(node.cod()[0])[0] == 'xnew' for node in reads),
        'the table is read at the positions of the new tokens')


# ==========================================================================
# The interactive page.
# ==========================================================================
def check_the_page_holds_four_consistent_variants() -> None:
    notebook_diagrams.check_page_variants(
        VARIANTS, deepseek_v3_page_variants.INITIAL_VARIANT)
    require([variant.identifier for variant in VARIANTS]
            == ['decode-quantised', 'decode-unquantised', 'cached-quantised',
                'cached-unquantised'], 'the page holds the four forms')
    derived = [variant for variant in VARIANTS if variant.term is None]
    require(all(variant.functor is notebook_diagrams.PageFunctor.DEQUANTISE
                for variant in derived) and len(derived) == 2,
            'both unquantised forms are derived by the dequantisation functor')


def check_the_forms_in_the_reals_carry_the_inspection_text_of_the_reals() -> None:
    held_in = released_text.WEIGHT_QUANTISATION_SENTENCE.split('{')[0]
    for variant in VARIANTS:
        if variant.term is not None:
            continue
        settings = notebook_diagrams.settings_of_page_variant(
            variant, VARIANTS_BY_IDENTIFIER, PAGE)
        source = notebook_diagrams.settings_of_page_variant(
            VARIANTS_BY_IDENTIFIER[variant.derived_from], VARIANTS_BY_IDENTIFIER, PAGE)
        require(settings.operator_roles == released_deepseek_v3.OPERATOR_ROLES
                and settings.operator_explanations
                == released_deepseek_v3.OPERATOR_EXPLANATIONS,
                f'{variant.identifier} carries the roles and the explanations of the '
                'model in the reals')
        require(notebook_diagrams.changes_the_inspection_text(settings, source),
                f'{variant.identifier} carries inspection text of its own')
        require(not any(held_in in role.role
                        for role in settings.operator_roles.values()),
                f'no role of {variant.identifier} names the quantisation of a weight')
        others = {field.name for field in dataclasses.fields(settings)} - {
            'operator_roles', 'operator_explanations'}
        require(all(getattr(settings, name) == getattr(source, name)
                    for name in others),
                f'{variant.identifier} is drawn under the settings of its source')


def check_the_functor_leaves_no_quantisation_on_the_presented_page() -> None:
    for variant in VARIANTS:
        if variant.term is None:
            continue
        settings = notebook_diagrams.settings_of_page_variant(
            variant, VARIANTS_BY_IDENTIFIER, PAGE)
        presented = notebook_diagrams.present_each_side(variant.term, settings)
        packaged, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
        derived = notebook_diagrams.apply_page_functor(
            notebook_diagrams.PageFunctor.DEQUANTISE, packaged)
        require(not any(
            True for _ in tutil.type_search(Quantization.Quantified, derived)),
                f'the functor leaves a quantisation on {variant.identifier}')
        require(not any(isinstance(node.operator, Quantization.TypeConvert)
                        for node in operations(derived)),
                f'the functor leaves a conversion on {variant.identifier}')


def check_every_variant_draws_a_legend_and_inspection_boxes() -> None:
    for identifier, (_, settings) in presented_terms().items():
        require(settings.advanced_display
                is notebook_diagrams.AdvancedDisplay.INTERACTIVE,
                f'{identifier} draws no inspection boxes')


def check_every_legend_row_carries_a_code_name() -> None:
    legends = notebook_diagrams.page_variant_legends(VARIANTS, PAGE)
    for identifier, (term, _) in presented_terms().items():
        axes_by_uid = {axis.uid._id: axis for axis in tutil.type_search(cat.Axis, term)}
        rows = legends[identifier]
        require(bool(rows), f'{identifier} has no legend')
        for row in rows:
            require(bool(row['codeName']),
                    f'{row["text"]} of {identifier} has no code name')
            sizes = [axes_by_uid[uid].local_size() for uid in row['uids']
                     if uid in axes_by_uid]
            one_symbol = any(isinstance(size, nm.FreeNumeric)
                             and size.uid._name is not None
                             for size in sizes)
            require(not one_symbol or bool(row['sizeCodeName']),
                    f'the size of {row["text"]} of {identifier} has no code name')


def check_every_wire_of_the_page_carries_a_quantisation() -> None:
    for identifier in ('decode-quantised', 'cached-quantised'):
        term, settings = presented_terms()[identifier]
        presented = notebook_diagrams.present_each_side(term, settings)
        require(not quantise_model.unquantised_weaves(presented),
                f'a wire of {identifier} carries no quantisation')


def check_every_named_view_opens_a_box() -> None:
    for identifier, (term, settings) in presented_terms().items():
        presented = notebook_diagrams.present_each_side(term, settings)
        explained = explain_reindexings.present(
            presented, settings.reindexing_explanations)
        unexplained = explain_reindexings.names_of_unexplained_views(explained)
        require(not unexplained, f'no box opens over {unexplained} of {identifier}')


def check_every_weight_and_hidden_formula_has_a_role() -> None:
    names = {node.operator.name.to_bodies()
             for term in (DECODE, CACHED.expression) for node in operations(term)
             if isinstance(node.operator, ops.Linear)}
    missing = sorted(names - released_deepseek_v3.OPERATOR_ROLES.keys())
    require(not missing, f'the weights {missing} have no role')
    hidden = {node.operator.name.to_bodies() for node in operations(DECODE)
              if isinstance(node.operator, ops.Arithmetic)
              and not shows_whole_formula(node.operator)}
    unexplained = sorted(hidden - released_deepseek_v3.ARITHMETIC_ROLES.keys())
    require(not unexplained, f'the maps {unexplained} have no role')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_sizes_are_those_of_config_671b,
    check_the_model_reads_identifiers_and_returns_scores,
    check_the_query_and_key_carry_d_channels_and_the_value_dv,
    check_the_queries_pass_through_a_low_rank,
    check_the_keys_and_values_come_from_one_latent,
    check_each_part_of_a_joined_weight_is_a_map_of_its_own,
    check_the_mask_is_a_read_with_no_mask_operator,
    check_the_decode_form_reads_the_hidden_state_at_the_mask,
    check_one_box_turns_keys_once_per_token_and_queries_once_per_head,
    check_the_ramp_and_the_scale_of_the_scores,
    check_the_gate_holds_the_released_steps,
    check_the_grouped_choice_is_the_released_choice,
    check_the_gate_box_is_its_body_once_per_token,
    check_the_routed_experts_read_the_chosen_slots,
    check_the_shared_expert_is_one_swiglu_of_width_s,
    check_three_dense_layers_and_fifty_eight_mixture_layers,
    check_every_projection_weight_is_fp8_in_blocks_of_128,
    check_no_activation_is_rounded_to_fp8,
    check_every_wire_carries_a_quantisation_and_every_operator_a_rule,
    check_the_gate_chooses_in_fp32_and_weights_in_bf16,
    check_the_rotary_embedding_computes_in_fp32,
    check_the_softmax_returns_bf16,
    check_every_conversion_is_between_bf16_and_fp32,
    check_the_identifiers_and_the_positions_are_int64,
    check_stripping_returns_each_pass_in_the_reals,
    check_the_attention_has_twenty_six_placements,
    check_the_narrowest_placement_caches_the_latent_and_the_turned_key,
    check_every_layer_caches_576_values_per_token,
    check_the_absorbed_order_takes_fewer_operations,
    check_the_pass_is_the_released_absorb_mode,
    check_the_pass_reads_the_new_tokens_alone,
    check_the_caches_are_bf16_at_the_released_quantisations,
    check_the_rotary_table_is_read_at_the_new_tokens,
    check_the_page_holds_four_consistent_variants,
    check_the_forms_in_the_reals_carry_the_inspection_text_of_the_reals,
    check_the_functor_leaves_no_quantisation_on_the_presented_page,
    check_every_variant_draws_a_legend_and_inspection_boxes,
    check_every_legend_row_carries_a_code_name,
    check_every_wire_of_the_page_carries_a_quantisation,
    check_every_named_view_opens_a_box,
    check_every_weight_and_hidden_formula_has_a_role,
)


def report_each_check() -> int:
    '''Every check, with one line printed for each and a line of totals, returning how
    many of them failed.'''
    started = time.perf_counter()
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'ok      {check.__name__}')
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures += 1
            print(f'FAILED  {check.__name__}: {type(error).__name__}: {error}')
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} DeepSeek-V3 checks passed in '
          f'{time.perf_counter() - started:.0f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
