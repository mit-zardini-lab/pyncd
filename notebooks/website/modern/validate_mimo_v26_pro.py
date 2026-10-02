# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim `notebooks/website/modern/MiMoV26Pro.ipynb` makes about
MiMo-V2.6-Pro.

    python notebooks/website/modern/validate_mimo_v26_pro.py

The notebook is public facing and holds the prose and the figures alone. The claims live
here, one `check_` function per claim, in the order the notebook makes them: the axes
and the ends of the model, the layer plan, the rotary embedding, the reads of the
earlier tokens and the CausalSlide, the attention, the feed-forward maps and the output
head, the tables of the inspection boxes, the pass over new tokens and its caches, and
the page of two variants. The script prints one line per check and the time the run
took, and it exits non-zero on a failure.

`check_the_rotation_matches_rotate_half_by_value` turns random vectors the way
`apply_rotary_pos_emb` of the reference turns them and the way the rotary box turns
them, with the permutation of the box read from the rows of its covariant view, and
compares the pairs and the scores. `check_the_released_biases_keep_every_gate` reads
`router_correction_bias_ranges.json` beside the model, which records the correction
biases of the checkpoint per layer.
'''
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys
import time
from collections.abc import Callable

import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import caching.algebra.cache_contents as cache_contents  # noqa: E402
import caching.algebra.derive_cached_pass as derive_cached_pass  # noqa: E402
import caching.data_structure.Caching as Caching  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402

import notebooks.display.explain_cached_reads as explain_cached_reads  # noqa: E402
import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.sota_figures as figures  # noqa: E402
import notebooks.sota.MiMoV26Pro.assemble_page_variants as assemble_page_variants  # noqa: E402
import notebooks.sota.MiMoV26Pro.attention_modes as attention_modes  # noqa: E402
import notebooks.sota.MiMoV26Pro.declared_axes as declared_axes  # noqa: E402
import notebooks.sota.MiMoV26Pro.derive_cached_mimo_v26_pro as derive_cached_mimo_v26_pro  # noqa: E402,E501
import notebooks.sota.MiMoV26Pro.feed_forward as feed_forward  # noqa: E402
import notebooks.sota.MiMoV26Pro.grouped_query_attention as grouped_query_attention  # noqa: E402,E501
import notebooks.sota.MiMoV26Pro.layer_stack as layer_stack  # noqa: E402
import notebooks.sota.MiMoV26Pro.mixture_of_experts as mixture_of_experts  # noqa: E402
import notebooks.sota.MiMoV26Pro.operator_explanations as operator_explanations  # noqa: E402
import notebooks.sota.MiMoV26Pro.released_constants as released_constants  # noqa: E402
import notebooks.sota.MiMoV26Pro.rotary_embedding as rotary_embedding  # noqa: E402
import notebooks.sota.MiMoV26Pro.slide_causal_reads as slide_causal_reads  # noqa: E402
import notebooks.sota.MiMoV26Pro.whole_model as whole_model  # noqa: E402
from notebooks.display.explain_operators import shows_whole_formula  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axes  # noqa: E402
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text  # noqa: E402,E501

MODEL: cat.Morphism = whole_model.mimo
DECODE: cat.Morphism = slide_causal_reads.mimo_slid
CACHED: cat.Morphism = derive_cached_mimo_v26_pro.cached_mimo
ASSIGNED_SIZES: dict[str, int] = whole_model.released_assigned_sizes()
CACHED_SIZES: dict[str, int] = derive_cached_mimo_v26_pro.cached_assigned_sizes()
PAGE = assemble_page_variants.page_settings(figures.DiagramMode.HTML)
VARIANTS = assemble_page_variants.page_variants(PAGE)
BIAS_RANGES = (pathlib.Path(__file__).resolve().parents[2]
               / 'sota/MiMoV26Pro/router_correction_bias_ranges.json')

RELEASED_AXIS_WIDTHS: dict[str, int] = {
    'm': 6144, 'h': 8, 'g': 16, 't': 32, 'n': 128, 'u': 128, 'w': 128, 'e': 384,
    'k': 8, 'f': 2048, 'd': 16384, 'v': 152576}
LAYERS = 70
FULL_ATTENTION_LAYERS = (0, 7, 15, 23, 31, 39, 47, 55, 62, 69)
'''The layers `hybrid_layer_pattern` marks 0, read from `config.json` lines 47 to 118
on 2026-09-28.'''
DENSE_LAYERS = (0,)
'''The layers `moe_layer_freq` marks 0, read from `config.json` lines 126 to 197.'''
KEYS_AND_VALUES_PER_TOKEN = 1536 + 1024
HIDDEN_STATE_PER_TOKEN = 6144
VALUES_OF_THE_FULL_LAYERS_PER_TOKEN = 25600
VALUES_OF_THE_WINDOW_LAYERS = 19507200
ROTATION_TRIALS = 64


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def size_of(axis: str) -> int:
    return ASSIGNED_SIZES[axis]


def width_of(axis: cat.Axis) -> int:
    '''The released size of `axis`, whose size may be a sum or a product of the sizes
    the configuration assigns.'''
    size = axis.local_size()
    return nm.evaluate_integer(size, {
        symbol: ASSIGNED_SIZES[symbol.uid._name.to_bodies()]
        for symbol in nm.free_symbols(size)})


def shape_of(array: cat.Array) -> tuple[cat.Axis, ...]:
    return tuple(array.shape())


def boxes_named(short_name: str, term: object) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in tutil.type_search(cat.Broadcasted, term)
                 if whole_model.is_box_named(node, short_name))


def body_of(name: str, term: cat.Morphism) -> cat.Morphism:
    return whole_model.part_named(name, term).operator.block


def linear_maps_named(name: str, term: object) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in tutil.type_search(cat.Broadcasted, term)
                 if isinstance(node.operator, ops.Linear)
                 and node.operator.name.to_bodies() == name)


def bodies_of(name: str) -> str:
    '''The text a table keys a name by, from its LaTeX.'''
    return fd.DynamicName.from_str(name).to_bodies()


def views_named(name: str, operations: tuple[cat.Broadcasted, ...]
                ) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in operations
                 if isinstance(node.operator, ops.View) and node.operator.name is not None
                 and node.operator.name.to_bodies() == name)


def operations_outside_boxes(term: object) -> tuple[cat.Broadcasted, ...]:
    '''The operations of `term` that no box of it holds, with each box itself.'''
    match term:
        case cat.Broadcasted():
            return (term,)
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return tuple(node for part in parts for node in operations_outside_boxes(part))
        case cat.Block(body=body):
            return operations_outside_boxes(body)
    return ()


def operators_other_than_views(term: cat.Morphism) -> collections.Counter:
    return collections.Counter(
        (type(node.operator).__name__,
         None if getattr(node.operator, 'name', None) is None
         else node.operator.name.to_bodies())
        for node in quantise_model.operations_of(term)
        if not isinstance(node.operator, ops.View))


def sublayer_kinds(term: object) -> list[str]:
    '''The kind of every sublayer `term` runs, in order, with every repeated block
    written out as many times as it repeats.'''
    kinds = {attention_modes.FULL_BOX: 'full', attention_modes.WINDOW_BOX: 'window',
             feed_forward.DENSE_BOX: 'dense', mixture_of_experts.MIXTURE_BOX: 'mixture'}
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return [kind for part in parts for kind in sublayer_kinds(part)]
        case cat.Block(body=body, block_tag=block_tag):
            return sublayer_kinds(body) * block_tag.repetition._value
        case cat.Broadcasted(operator=ops.BlockOperator(name=name)) if name is not None:
            kind = kinds.get(name.to_bodies())
            return [kind] if kind is not None else []
    return []


def caches_of(term: cat.Morphism) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in tutil.type_search(cat.Broadcasted, term)
                 if isinstance(node.operator, Caching.Caching))


def as_sent(term: cat.Morphism, settings: notebook_diagrams.DiagramSettings
            ) -> cat.Morphism:
    presented = notebook_diagrams.present_each_side(term, settings)
    sent, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
    return sent


def variant(identifier: str) -> notebook_diagrams.PageVariant:
    return next(entry for entry in VARIANTS if entry.identifier == identifier)


# ==========================================================================
# Reading the figures, and the model.
# ==========================================================================
def check_the_model_reads_identifiers_and_returns_logits() -> None:
    '''The model reads the identifier of each token and returns one logit per token and
    vocabulary entry.'''
    require(axes(MODEL) == ([['x']], [['x', 'v']]),
            f'the model reads and returns {axes(MODEL)}')
    tokens, = MODEL.dom()
    require(isinstance(tokens.datatype, cat.Natural),
            f'the model reads {tokens.datatype}')


def check_the_axes_have_the_released_sizes() -> None:
    '''The axes of the table take the sizes of the configuration, the turned channels
    and the head are 64 and 192 wide, the halves are two, and the distance axis is as
    long as the prompt.'''
    differing = {name: ASSIGNED_SIZES.get(name) for name, width
                 in RELEASED_AXIS_WIDTHS.items() if ASSIGNED_SIZES.get(name) != width}
    require(not differing, f'the configuration sizes {differing}')
    widths = tuple(map(width_of, (declared_axes.p, declared_axes.a, declared_axes.c)))
    require(widths == (64, 192, 2), f'p, a and c are {widths} wide')
    require(declared_axes.r.local_size() == declared_axes.x.local_size(),
            'the distance axis is not as long as the prompt')


def check_the_constants_are_named_symbols() -> None:
    '''Each constant of the table is a named symbol of the model.'''
    symbols = set(tutil.type_search(nm.FreeNumeric, MODEL))
    missing = [constant.symbol.to_latex() for constant in
               released_constants.RELEASED_CONSTANTS if constant.symbol not in symbols]
    require(not missing, f'the model holds no {missing}')


def check_two_axes_hold_a_value_at_some_positions() -> None:
    '''The distance axis `r|x` and the window axis `w|x` each carry the affine form
    `i_x - i_s >= 0` over the query axis.'''
    for axis in (grouped_query_attention.distances, grouped_query_attention.window_slots):
        require(isinstance(axis, AffineGuards.AffineSparseAxis)
                and tuple(axis.guides) == (declared_axes.x,)
                and axis.stride == nm.Integer(-1)
                and axis.guide_strides == (nm.Integer(1),),
                f'{axis.uid._name.to_bodies()} carries no form i_x - i_s >= 0')


# ==========================================================================
# The layer plan.
# ==========================================================================
def check_the_stack_holds_seventy_layers() -> None:
    '''The layer stack holds 70 layers, 10 of full attention and 60 of a sliding
    window.'''
    stack = layer_stack.layer_stack
    count = layer_stack.layer_count(stack)
    full = layer_stack.runs_of_blocks_titled(text.FULL_TITLE, stack)
    window = layer_stack.runs_of_blocks_titled(text.WINDOW_TITLE, stack)
    require((count, full, window) == (LAYERS, 10, 60),
            f'the stack holds {count} layers, {full} full and {window} sliding window')


def check_the_layer_plan_is_the_released_plan() -> None:
    '''Layers 0, 7, 15, 23, 31, 39, 47, 55, 62 and 69 attend fully and the rest through
    the window, and layer 0 alone holds the dense MLP.'''
    kinds = sublayer_kinds(layer_stack.layer_stack)
    plan = tuple(zip(kinds[0::2], kinds[1::2]))
    released = tuple(
        ('full' if layer in FULL_ATTENTION_LAYERS else 'window',
         'dense' if layer in DENSE_LAYERS else 'mixture') for layer in range(LAYERS))
    differing = [layer for layer, (drawn, wanted) in enumerate(zip(plan, released))
                 if drawn != wanted]
    require(len(plan) == LAYERS and not differing,
            f'the plan holds {len(plan)} layers and differs at layers {differing}')


def check_every_repeated_block_returns_its_domain() -> None:
    '''Every block of the plan returns the hidden state it reads.'''
    blocks = (layer_stack.first_layer, layer_stack.first_group, layer_stack.long_groups,
              layer_stack.last_groups)
    require(all(block.dom() == block.cod() for block in blocks),
            f'the blocks read and return {[axes(block) for block in blocks]}')


# ==========================================================================
# The rotary embedding.
# ==========================================================================
def check_the_turned_rows_are_written_as_two_halves() -> None:
    '''The projections of the turned channels of a query head and of a key head write
    their rows over the halves `c` and the 32 pairs `t`.'''
    require(size_of('t') == 32 and width_of(declared_axes.c) == 2,
            f"t is {size_of('t')}")
    produced = {name: {shape_of(node.cod()[0])[-2:]
                       for node in linear_maps_named(name, MODEL)}
                for name in ('W^{Qr}', 'W^{Kr}')}
    require(all(shapes == {(declared_axes.c, declared_axes.t)}
                for shapes in produced.values()),
            f'the turned projections produce {produced}')


def check_the_pairs_view_writes_each_half_into_a_pair() -> None:
    '''The covariant view named Pairs writes entry `(i_c, i_t)` at channel
    `2 i_t + i_c`, and each rotary box holds it once before the reading of pairs as
    complex numbers.'''
    (axis, strides, shift), = rotary_embedding.PAIRS._cod_stride_shift
    require(axis == declared_axes.p and tuple(rotary_embedding.PAIRS._dom)
            == (declared_axes.c, declared_axes.t)
            and strides == (nm.Integer(1), nm.Integer(2)) and shift == nm.Integer(0),
            f'Pairs writes at strides {strides} and shift {shift}')
    for box in (rotary_embedding.ROTATE_FULL_ATTENTION_CHANNELS,
                rotary_embedding.ROTATE_SLIDING_WINDOW_CHANNELS):
        kinds = [type(node.operator).__name__
                 for node in tutil.type_search(cat.Broadcasted, box.operator.block)]
        require(kinds.count('CovariantView') == 1 and kinds.count('PairsAsComplex') == 1
                and kinds.count('Decomplex') == 1,
                f'a rotary box holds {kinds}')


def reference_rotation(halves: numpy.ndarray, angles: numpy.ndarray) -> numpy.ndarray:
    '''`apply_rotary_pos_emb` of the reference on one vector of 64 channels:
    `q cos + rotate_half(q) sin`, with the 32 angles repeated over the two halves.'''
    repeated = numpy.concatenate((angles, angles))
    first, second = numpy.split(halves, 2)
    rotated_half = numpy.concatenate((-second, first))
    return halves * numpy.cos(repeated) + rotated_half * numpy.sin(repeated)


def box_rotation(halves: numpy.ndarray, angles: numpy.ndarray) -> numpy.ndarray:
    '''The rotary box on the same vector: the rows of the projection read as `(c, t)`,
    written by the row of the view named Pairs, read as complex pairs, turned and
    written back in the order of the pairs.'''
    pairs = width_of(declared_axes.t)
    (_, (half_stride, pair_stride), shift), = rotary_embedding.PAIRS._cod_stride_shift
    ordered = numpy.zeros(2 * pairs)
    for half in range(2):
        for pair in range(pairs):
            channel = (half_stride._value * half + pair_stride._value * pair
                       + shift._value)
            ordered[channel] = halves[half * pairs + pair]
    turned = (ordered[0::2] + 1j * ordered[1::2]) * numpy.exp(1j * angles)
    written = numpy.zeros(2 * pairs)
    written[0::2], written[1::2] = turned.real, turned.imag
    return written


def check_the_rotation_matches_rotate_half_by_value() -> None:
    '''The box turns the pair made of channel `i` of the first half and channel `i` of
    the second half, as `rotate_half` does, and a query and a key turned by the box
    score as they score in the reference.'''
    generator = numpy.random.default_rng(20260928)
    pairs = width_of(declared_axes.t)
    worst_pair = worst_score = 0.0
    for _ in range(ROTATION_TRIALS):
        query, key = generator.standard_normal((2, 2 * pairs))
        query_angles, key_angles = generator.uniform(0, 100, (2, pairs))
        reference = reference_rotation(query, query_angles)
        box = box_rotation(query, query_angles)
        worst_pair = max(worst_pair, float(numpy.abs(
            numpy.concatenate((box[0::2], box[1::2])) - reference).max()))
        worst_score = max(worst_score, abs(
            float(reference @ reference_rotation(key, key_angles))
            - float(box @ box_rotation(key, key_angles))))
    require(worst_pair < 1e-12 and worst_score < 1e-10,
            f'the pairs differ by {worst_pair} and the scores by {worst_score}')


def check_the_two_modes_read_two_tables() -> None:
    '''The full attention layers turn at the base `\\beta` and the sliding window layers
    at the base `\\beta_w`, each for its queries and its keys.'''
    def bases(box: cat.Broadcasted) -> set[nm.Numeric]:
        return {node.operator.base for node in tutil.type_search(cat.Broadcasted, box)
                if isinstance(node.operator, dst.Rotary)}
    full = attention_modes.full_attention()
    window = attention_modes.sliding_window_attention()
    require(len(boxes_named(rotary_embedding.ROTATION_BOX, full)) == 2
            and len(boxes_named(rotary_embedding.WINDOW_ROTATION_BOX, window)) == 2,
            'a mode does not turn its queries and its keys with its own box')
    require(bases(rotary_embedding.ROTATE_FULL_ATTENTION_CHANNELS)
            == {released_constants.ROTARY_BASE}
            and bases(rotary_embedding.ROTATE_SLIDING_WINDOW_CHANNELS)
            == {released_constants.WINDOW_ROTARY_BASE},
            'the tables are built at other bases')


# ==========================================================================
# The reads of the earlier tokens and the CausalSlide.
# ==========================================================================
def check_the_views_read_back_from_each_query() -> None:
    '''The views named Back and Window read token `i_x - i_s` for slot `i_s` of query
    `i_x`, the first over as many slots as tokens and the second over 128.'''
    for read, slots in ((grouped_query_attention.READ_BACK, declared_axes.r),
                        (grouped_query_attention.READ_WINDOW, declared_axes.w)):
        (axis, strides, shift), = read._cod_stride_shift
        require(axis == declared_axes.x and strides == (nm.Integer(1), nm.Integer(-1))
                and shift == nm.Integer(0)
                and read._dom[1].local_size() == slots.local_size(),
                f'{read.name.to_bodies()} reads at {strides} and {shift}')
    require(size_of('w') == 128, f"the window holds {size_of('w')} slots")


def check_the_slide_reads_the_hidden_state_back_once() -> None:
    '''In the CausalSlide each mode reads the normalised hidden state back once, at the
    copy that feeds the queries, the keys and the values, and the key and value
    projections run over the queries and the slots while the query projections run over
    the queries.'''
    for mode, name, slots in (
            (attention_modes.FULL_BOX, grouped_query_attention.BACK_VIEW_NAME,
             grouped_query_attention.distances),
            (attention_modes.WINDOW_BOX, grouped_query_attention.WINDOW_VIEW_NAME,
             grouped_query_attention.window_slots)):
        body = body_of(mode, DECODE)
        backs = views_named(bodies_of(name), operations_outside_boxes(body))
        require(len(backs) == 1 and shape_of(backs[0].dom()[0])
                == (declared_axes.x, declared_axes.m),
                f'the {mode} body reads back {len(backs)} times')
        for weight in ('W^{Kr}', 'W^{Kn}', 'W^{V}'):
            read = {shape_of(node.dom()[0])[:2]
                    for node in linear_maps_named(weight, body)}
            require(read == {(declared_axes.x, slots)}, f'{weight} reads {read}')
        for weight in ('W^{Qr}', 'W^{Qn}'):
            read = {shape_of(node.dom()[0]) for node in linear_maps_named(weight, body)}
            require(read == {(declared_axes.x, declared_axes.m)}, f'{weight} reads {read}')


def check_the_slide_moves_the_reads_past_operations_broadcast_over_the_tokens() -> None:
    '''The CausalSlide holds the operators of the model and no other, and a mode reads
    back once where the model as built reads back twice, once for the keys and once for
    the values.'''
    require(set(operators_other_than_views(DECODE))
            == set(operators_other_than_views(MODEL)),
            'the slide adds or removes an operator other than a view')
    for mode, name in ((attention_modes.FULL_BOX, grouped_query_attention.BACK_VIEW_NAME),
                       (attention_modes.WINDOW_BOX,
                        grouped_query_attention.WINDOW_VIEW_NAME)):
        counts = tuple(len(views_named(
            bodies_of(name), operations_outside_boxes(body_of(mode, term))))
            for term in (MODEL, DECODE))
        require(counts == (2, 1), f'the {mode} body reads back {counts} times')


# ==========================================================================
# The attention.
# ==========================================================================
def check_the_sizes_of_the_attention() -> None:
    '''128 query heads in 8 groups of 16, 8 key heads and 8 value heads, a query head
    and a key head of 64 turned and 128 other channels, a value head of 128.'''
    require((size_of('h'), size_of('g'), size_of('n'), size_of('u'))
            == (8, 16, 128, 128),
            f"h, g, n and u are {size_of('h'), size_of('g'), size_of('n'), size_of('u')}")
    require(width_of(declared_axes.h) * width_of(declared_axes.g) == 128,
            'the query heads are not 128')


def check_the_cores_are_computed_once_per_query() -> None:
    '''Both cores are one box computed once per query, confirmed by broadcasting the
    box back over the queries, and every query head of a group reads the key and the
    value of its group.'''
    require(grouped_query_attention.CORE.is_confirmed()
            and grouped_query_attention.SINK_CORE.is_confirmed(),
            'a core does not expand back to its written-out form')
    require(axes(grouped_query_attention.CORE.candidate)
            == ([['x', 'h', 'g', 'a'], ['x', 'r|x', 'h', 'a'], ['x', 'r|x', 'h', 'u']],
                [['x', 'h', 'g', 'u']]),
            f'the full core reads {axes(grouped_query_attention.CORE.candidate)}')
    require(axes(grouped_query_attention.SINK_CORE.candidate)
            == ([['h', 'g'], ['x', 'h', 'g', 'a'], ['x', 'w|x', 'h', 'a'],
                 ['x', 'w|x', 'h', 'u']], [['x', 'h', 'g', 'u']]),
            f'the sink core reads {axes(grouped_query_attention.SINK_CORE.candidate)}')


def check_the_scores_are_divided_by_the_root_of_the_head_width() -> None:
    '''Both cores divide every score by the square root of `|a|`, 192.'''
    for core in (grouped_query_attention.CORE, grouped_query_attention.SINK_CORE):
        scales = [node.operator.formula for node in
                  tutil.type_search(cat.Broadcasted, core.candidate.operator.block)
                  if isinstance(node.operator, ops.Arithmetic)
                  and node.operator.name.to_bodies() == bodies_of(
                      grouped_query_attention.SCORE_SCALE_NAME)]
        require(scales == [nm.x / nm.SquareRoot(declared_axes.a.local_size())],
                f'a core scales by {scales}')


def check_the_sink_joins_the_softmax_of_the_window_layers_alone() -> None:
    '''A sliding window layer holds the logit of every query head, exponentiated once
    outside its core, and its core adds it to the sum of the exponentials, takes one
    reciprocal and weights the values by the quotients. A full attention layer holds no
    logit and takes a softmax.'''
    window = attention_modes.sliding_window_attention()
    full = attention_modes.full_attention()
    sinks = linear_maps_named(grouped_query_attention.SINK_NAME, window)
    require(len(sinks) == 1 and shape_of(sinks[0].cod()[0])
            == (declared_axes.h, declared_axes.g) and not sinks[0].dom(),
            f'the window layer holds {len(sinks)} logits')
    require(not linear_maps_named(grouped_query_attention.SINK_NAME, full),
            'a full attention layer holds a logit')
    body = grouped_query_attention.SINK_CORE.candidate.operator.block
    kinds = collections.Counter(type(node.operator).__name__
                                for node in tutil.type_search(cat.Broadcasted, body))
    require(kinds['AdditionOp'] == 1 and kinds['SoftMax'] == 0,
            f'the sink core holds {dict(kinds)}')
    full_kinds = collections.Counter(
        type(node.operator).__name__ for node in
        tutil.type_search(cat.Broadcasted, grouped_query_attention.CORE.candidate))
    require(full_kinds['SoftMax'] == 1, f'the full core holds {dict(full_kinds)}')


def check_the_values_are_scaled_before_they_are_read() -> None:
    '''The value scale `\\lambda` multiplies the output of `W^{V}` before the view that
    reads the earlier tokens.'''
    for mode in (attention_modes.full_attention(), attention_modes.sliding_window_attention()):
        block = whole_model.part_titled(text.VALUE_TITLE, mode)
        formulas = [node.operator.formula for node in tutil.type_search(cat.Broadcasted, block)
                    if isinstance(node.operator, ops.Arithmetic)]
        require(formulas == [nm.x * released_constants.VALUE_SCALE]
                and shape_of(block.cod()[0]) == shape_of(declared_axes.VALUES),
                f'the values are scaled by {formulas}')


# ==========================================================================
# The feed-forward maps and the output head.
# ==========================================================================
def check_one_dense_layer_and_sixty_nine_mixtures() -> None:
    '''The dense MLP runs in 1 layer and the mixture of experts in 69, and both are one
    box computed once per token.'''
    stack = layer_stack.layer_stack
    dense = layer_stack.runs_of_blocks_titled(text.DENSE_MLP_TITLE, stack)
    mixture = layer_stack.runs_of_blocks_titled(text.MIXTURE_TITLE, stack)
    require((dense, mixture) == (1, 69),
            f'the dense MLP runs {dense} times and the mixture {mixture} times')
    require(feed_forward.DENSE_MLP_CONFIRMATION.is_confirmed()
            and mixture_of_experts.MIXTURE_CONFIRMATION.is_confirmed(),
            'a box does not expand back to its body over the tokens')


def check_the_router_keeps_eight_of_three_hundred_and_eighty_four() -> None:
    '''The router scores 384 experts and keeps 8, each expert is 2048 wide, the dense
    MLP is 16384 wide, and the mixture holds no shared expert.'''
    require((size_of('e'), size_of('k'), size_of('f'), size_of('d'))
            == (384, 8, 2048, 16384),
            f"e, k, f and d are {size_of('e'), size_of('k'), size_of('f'), size_of('d')}")
    gates = boxes_named(mixture_of_experts.GATE_BOX, mixture_of_experts.MIXTURE)
    require(len(gates) == 1, f'the mixture holds {len(gates)} router boxes')
    projections = {node.operator.name.to_bodies() for node in
                   tutil.type_search(cat.Broadcasted, mixture_of_experts.MIXTURE)
                   if isinstance(node.operator, ops.Linear)}
    require(projections == {'W^{R}', 'W^{G}', 'W^{U}', 'W^{D}',
                            bodies_of(mixture_of_experts.ROUTER_BIAS_NAME)},
            f'the mixture holds the weights {sorted(projections)}')


def check_the_released_biases_keep_every_gate() -> None:
    '''In every mixture at least 344 of the 384 correction biases are positive, so at
    least 8 biased scores are positive for every token, and every expert the top-8
    keeps has a positive biased score. The biases lie between -0.349 and 0.810.'''
    layers = json.loads(BIAS_RANGES.read_text(encoding='utf-8'))['layers']
    fewest = min(layer['positive'] for layer in layers)
    smallest = min(layer['smallest'] for layer in layers)
    largest = max(layer['largest'] for layer in layers)
    require(len(layers) == 69 and fewest == 344 and fewest >= size_of('k'),
            f'{len(layers)} layers, at least {fewest} positive biases')
    require(math.isclose(smallest, -0.349012, abs_tol=1e-6)
            and math.isclose(largest, 0.809529, abs_tol=1e-6),
            f'the biases lie between {smallest} and {largest}')


def check_the_output_head_writes_every_vocabulary_entry() -> None:
    '''The output head maps the hidden state of 6144 channels onto 152576 entries with a
    weight of its own.'''
    require(size_of('m') == 6144 and size_of('v') == 152576,
            f"m is {size_of('m')} and v is {size_of('v')}")
    require(len(linear_maps_named('W^{L}', MODEL)) == 1,
            'the output head has no weight of its own')


# ==========================================================================
# The tables the inspection boxes are filled from.
# ==========================================================================
def check_every_weight_has_a_role() -> None:
    '''Every weight of the model has a row saying what it is for.'''
    names = {node.operator.name.to_bodies()
             for node in tutil.type_search(cat.Broadcasted, MODEL)
             if isinstance(node.operator, ops.Linear)}
    missing = sorted(names - operator_explanations.OPERATOR_ROLES.keys())
    require(not missing, f'the weights {missing} have no role')


def check_every_named_view_and_hidden_formula_is_explained() -> None:
    '''Every named view opens a box, and every elementwise map whose name hides part of
    its formula has a role.'''
    nodes = tuple(tutil.type_search(cat.Broadcasted, MODEL))
    views = {node.operator.name.to_bodies() for node in nodes
             if isinstance(node.operator, ops.View) and node.operator.name is not None}
    hidden = {node.operator.name.to_bodies() for node in nodes
              if isinstance(node.operator, ops.Arithmetic)
              and not shows_whole_formula(node.operator)}
    unexplained_views = sorted(
        views - operator_explanations.REINDEXING_EXPLANATIONS.keys())
    unexplained_maps = sorted(hidden - operator_explanations.ARITHMETIC_ROLES.keys())
    require(not unexplained_views and not unexplained_maps,
            f'the views {unexplained_views} and the maps {unexplained_maps} have no row')
    unboxed = explain_reindexings.names_of_unexplained_views(explain_reindexings.present(
        MODEL, operator_explanations.REINDEXING_EXPLANATIONS))
    require(not unboxed, f'no box opens over the views {unboxed}')


# ==========================================================================
# The pass over new tokens and its caches.
# ==========================================================================
def check_the_pass_reads_and_returns_the_new_tokens() -> None:
    '''The pass reads the identifiers of the new tokens and returns their logits.'''
    require(axes(CACHED) == ([['xnew']], [['xnew', 'v']]),
            f'the pass reads and returns {axes(CACHED)}')


def check_no_operator_is_computed_over_the_cache() -> None:
    '''The derivation computes no operator over the cache, so every cache holds the
    operand of a read of the earlier tokens: the joined keys and the scaled values.'''
    require(not derive_cached_mimo_v26_pro.DERIVED.computed_over_the_cache,
            'an operator is computed over the cache')
    followed = collections.Counter(
        type(operator).__name__ if operator is not None else None
        for operator in derive_cached_mimo_v26_pro.DERIVED.operators_before_caches)
    require(set(followed) == {'ConcatenateAxes', 'Arithmetic'},
            f'the caches follow {dict(followed)}')


def check_the_caches_of_each_attention_mode() -> None:
    '''Every layer caches its keys, 1536 values per token, and its values, 1024. A full
    attention layer caches every earlier token, and a sliding window layer the last
    `|w| - 1` of them.'''
    modes = {mode.mode: mode for mode in derive_cached_mimo_v26_pro.caches_by_mode()}
    full, window = modes[attention_modes.FULL_BOX], modes[attention_modes.WINDOW_BOX]
    require(full.layers == 10 and window.layers == 60,
            f'{full.layers} full and {window.layers} sliding window layers cache')
    for mode in (full, window):
        require(sorted(values for _, values, _ in mode.caches) == [1024, 1536],
                f'the {mode.mode} layers cache {mode.caches}')
    kept, = derive_cached_mimo_v26_pro.DERIVED.kept_tokens
    require(kept.parts[0].local_size() == nm.collect_like_terms(
                declared_axes.w.local_size() + nm.Integer(-1)),
            f'a sliding window layer keeps {kept.parts[0].local_size().to_latex()}')
    kept_by_mode = ({kept for _, _, kept in full.caches},
                    {kept for _, _, kept in window.caches})
    require(kept_by_mode == ({None}, {size_of('w') - 1}),
            f'the full and the sliding window caches keep {kept_by_mode} earlier tokens')


def check_the_totals_of_the_caches() -> None:
    '''The 10 full attention layers hold 25,600 values for every token, and the 60
    sliding window layers hold 2,560 values for each of at most 127 tokens,
    19,507,200 values in all.'''
    modes = {mode.mode: mode for mode in derive_cached_mimo_v26_pro.caches_by_mode()}
    full, window = modes[attention_modes.FULL_BOX], modes[attention_modes.WINDOW_BOX]
    kept = size_of('w') - 1
    require(full.values_per_token() == VALUES_OF_THE_FULL_LAYERS_PER_TOKEN
            and window.values_per_token_of_a_layer() == KEYS_AND_VALUES_PER_TOKEN
            and window.values_per_token() * kept == VALUES_OF_THE_WINDOW_LAYERS,
            f'{full.values_per_token()} and {window.values_per_token() * kept}')


def check_the_slide_would_cache_the_hidden_state() -> None:
    '''Derived from the CausalSlide, every layer caches its normalised hidden state,
    6144 values per token.'''
    derived = derive_cached_pass.derive_cached_pass(
        DECODE, declared_axes.x, derive_cached_mimo_v26_pro.OLD_TOKENS,
        derive_cached_mimo_v26_pro.NEW_TOKENS, frozenset())
    sizes = derive_cached_mimo_v26_pro.cached_assigned_sizes(derived.expression)
    widths = {cache_contents.entries_per_token(cache, sizes)
              for cache in caches_of(derived.expression)}
    require(widths == {HIDDEN_STATE_PER_TOKEN}, f'the slid pass caches {widths}')


# ==========================================================================
# The page.
# ==========================================================================
def check_the_page_variants_name_one_another() -> None:
    '''The page holds two variants in the groups Decode and Cached, each in the reals,
    and opens on the decode variant.'''
    notebook_diagrams.check_page_variants(VARIANTS, assemble_page_variants.INITIAL_VARIANT)
    require([(entry.identifier, entry.group.title, entry.title, entry.term is not None)
             for entry in VARIANTS]
            == [('decode-unquantised', 'Decode', 'Unquantised', True),
                ('cached-unquantised', 'Cached', 'Unquantised', True)],
            'the variants are named otherwise')
    require(variant('decode-unquantised').term is DECODE
            and variant('cached-unquantised').term is CACHED,
            'a variant draws another term')
    cached_rows = set(variant('cached-unquantised').settings.reindexing_explanations)
    require(set(explain_cached_reads.CACHED_READ_EXPLANATIONS) <= cached_rows,
            'the cached variant loses the rows of the reads')


def check_every_legend_row_carries_a_code_name() -> None:
    '''Every variant draws its legend, and every row of every legend carries the name
    of its axis in generated code.'''
    legends = notebook_diagrams.page_variant_legends(VARIANTS, PAGE)
    for identifier, rows in legends.items():
        missing = [row['text'] for row in rows if not row['codeName']]
        require(rows and not missing, f'{identifier}: the rows {missing} have no code name')


def check_every_view_of_every_variant_opens_a_box() -> None:
    '''Every named view of every variant, the reads of the token axis among them, opens
    an inspection box on the page.'''
    for entry in VARIANTS:
        unexplained = explain_reindexings.names_of_unexplained_views(
            as_sent(entry.term, entry.settings))
        require(not unexplained, f'{entry.identifier}: no box opens over {unexplained}')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_model_reads_identifiers_and_returns_logits,
    check_the_axes_have_the_released_sizes,
    check_the_constants_are_named_symbols,
    check_two_axes_hold_a_value_at_some_positions,
    check_the_stack_holds_seventy_layers,
    check_the_layer_plan_is_the_released_plan,
    check_every_repeated_block_returns_its_domain,
    check_the_turned_rows_are_written_as_two_halves,
    check_the_pairs_view_writes_each_half_into_a_pair,
    check_the_rotation_matches_rotate_half_by_value,
    check_the_two_modes_read_two_tables,
    check_the_views_read_back_from_each_query,
    check_the_slide_reads_the_hidden_state_back_once,
    check_the_slide_moves_the_reads_past_operations_broadcast_over_the_tokens,
    check_the_sizes_of_the_attention,
    check_the_cores_are_computed_once_per_query,
    check_the_scores_are_divided_by_the_root_of_the_head_width,
    check_the_sink_joins_the_softmax_of_the_window_layers_alone,
    check_the_values_are_scaled_before_they_are_read,
    check_one_dense_layer_and_sixty_nine_mixtures,
    check_the_router_keeps_eight_of_three_hundred_and_eighty_four,
    check_the_released_biases_keep_every_gate,
    check_the_output_head_writes_every_vocabulary_entry,
    check_every_weight_has_a_role,
    check_every_named_view_and_hidden_formula_is_explained,
    check_the_pass_reads_and_returns_the_new_tokens,
    check_no_operator_is_computed_over_the_cache,
    check_the_caches_of_each_attention_mode,
    check_the_totals_of_the_caches,
    check_the_slide_would_cache_the_hidden_state,
    check_the_page_variants_name_one_another,
    check_every_legend_row_carries_a_code_name,
    check_every_view_of_every_variant_opens_a_box,
)


def report_each_check() -> int:
    '''Every check, with one line printed for each and a line of totals, returning how
    many of them failed.'''
    started = time.perf_counter()
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'ok      {check.__name__}', flush=True)
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures += 1
            print(f'FAILED  {check.__name__}: {type(error).__name__}: {error}', flush=True)
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} checks of the MiMo-V2.6-Pro website '
          f'notebook passed in {time.perf_counter() - started:.0f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
