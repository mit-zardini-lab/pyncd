# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim `notebooks/website/classic/Mixtral8x7B.ipynb` makes about
Mixtral-8x7B.

    python notebooks/website/classic/validate_mixtral_8x7b.py

The notebook is public facing and holds the prose and the figures alone. The claims
live here, one `check_` function per claim, in the order the notebook makes them: the
sizes of the released model, the grouped-query attention and its rotary embedding, the
causal mask and its CausalSlide, the sparse mixture of experts, the whole model, the
quantisations of the reference implementation, the cached pass and its caches, and the
page with its four variants. The script prints one line per check and the time the run
took, and it exits non-zero on a failure.

The page derives each unquantised variant in the browser by removing every
quantisation from the quantised variant.
`check_removing_the_quantisations_returns_the_decode_form` and
`check_removing_the_quantisations_returns_the_cached_pass` apply the Python statement of
that functor and compare the listing of its result with the listing of the unquantised
form, each recycled through a hypergraph, because the quantisation pass gives the body
of a box a block tag of its own.
'''
from __future__ import annotations

import pathlib
import sys
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa: E402
import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards  # noqa: E402
import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa: E402,E501
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import agent_display as ad  # noqa: E402
import caching.algebra.cache_contents as cache_contents  # noqa: E402
import caching.algebra.derive_cached_pass as derive_cached_pass  # noqa: E402
import caching.data_structure.Caching as Caching  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import graphs.processing.Hypergraph2Morphism as h2m  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import term_utilities.generate_config as generate_config  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402

import notebooks.classic.cached_mixtral_8x7b as cached_mixtral_8x7b  # noqa: E402
import notebooks.classic.mixtral_8x7b as mixtral_8x7b  # noqa: E402
import notebooks.classic.mixtral_8x7b_page_variants as mixtral_8x7b_page_variants  # noqa: E402,E501
import notebooks.classic.quantised_mixtral_8x7b as quantised_mixtral_8x7b  # noqa: E402
import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.sota_figures as figures  # noqa: E402
from notebooks.classic.mixtral_8x7b_wording import TEXT as text  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axes  # noqa: E402

MODEL = quantised_mixtral_8x7b.MIXTRAL
DECODE = quantised_mixtral_8x7b.MIXTRAL_IN_THE_CAUSAL_SLIDE
QUANTISED_DECODE = quantised_mixtral_8x7b.mixtral_quantised
CACHED_PASS = cached_mixtral_8x7b.CACHED_PASS
CACHED = cached_mixtral_8x7b.mixtral_cached
QUANTISED_CACHED = cached_mixtral_8x7b.mixtral_cached_quantised
ATTENTION = mixtral_8x7b.attention()
SLID_ATTENTION = slide_causal_reads_backwards.slide_causal_reads_backwards(ATTENTION)
MIXTURE = mixtral_8x7b.sparse_mixture_of_experts()

BF16 = quantised_mixtral_8x7b.BF16
FP32 = quantised_mixtral_8x7b.FP32
INT64 = quantised_mixtral_8x7b.INT64
QUERY_HEADS = 32
PROMPT_LENGTH_IN_THE_EXAMPLE = 4
CONTEXT_LENGTH = 32768
GIBIBYTE = 2 ** 30
STATE_CACHING_SLOWDOWN = 7
KIBIBYTE = 2 ** 10

DECODE_CAST_COUNTS: dict[tuple[str, str], int] = {
    ('BF16', 'FP32'): 4, ('FP32', 'BF16'): 5}
'''The casts written in the quantised decode form. Each of the two boxes of the rotary
embedding reads its pairs into FP32 and rounds the turned channels to BF16, the router
reads the kept scores into FP32 and rounds the weights to BF16, the attention kernel
rounds the probabilities and its output to BF16, and the logits are read into FP32. The
box turning the keys holds a body of its own in the CausalSlide, because it reads the
table of turns at the position of each slot.'''

CACHED_CAST_COUNTS: dict[tuple[str, str], int] = {
    ('BF16', 'FP32'): 3, ('FP32', 'BF16'): 4}
'''The casts written in the quantised cached pass, where the queries and the keys are
turned by one body.'''

WEIGHT_NAMES: tuple[str, ...] = (
    'E', 'W^{Q}', 'W^{K}', 'W^{V}', 'W^{O}', 'W^{g}',
    *(mixtral_8x7b.table_key(name) for name in (
        mixtral_8x7b.FIRST_EXPERT_NAME, mixtral_8x7b.SECOND_EXPERT_NAME,
        mixtral_8x7b.THIRD_EXPERT_NAME)),
    mixtral_8x7b.OUTPUT_PROJECTION_NAME)

DIAGRAMS = figures.DiagramSettings(
    mode=figures.DiagramMode.INLINE,
    dark_mode=notebook_diagrams.ColorMode.LIGHT,
    axis_sizes=figures.AxisSizes.SUBSCRIPT,
    assigned_sizes=cached_mixtral_8x7b.RELEASED_SIZES,
    advanced_display=figures.AdvancedDisplay.LEGEND,
    operator_explanations=mixtral_8x7b.OPERATOR_EXPLANATIONS,
    operator_roles=mixtral_8x7b.OPERATOR_ROLES,
    operator_references=mixtral_8x7b.OPERATOR_REFERENCES,
    reindexing_explanations=mixtral_8x7b.REINDEXING_EXPLANATIONS,
    title='Mixtral-8x7B')
'''The settings declared at the top of the setup cell of the notebook.'''

PAGE = mixtral_8x7b_page_variants.page_settings(DIAGRAMS)
VARIANTS = mixtral_8x7b_page_variants.page_variants(
    QUANTISED_DECODE, QUANTISED_CACHED, PAGE)


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook not met by the expression.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def quantisation_of(datatype: cat.Datatype) -> Quantization.Quantified | None:
    return Quantization.quantisation_of(datatype)


def operators_of(term: cat.Morphism) -> list[cat.Operator]:
    return [node.operator for node in tutil.type_search(cat.Broadcasted, term)]


def operations(term: cat.Morphism,
               kind: type[cat.Operator]) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in quantise_model.operations_of(term)
                 if isinstance(node.operator, kind))


def block_titled(term: cat.Morphism, title: str) -> cat.Block:
    '''The one block of `term` carrying `title`.'''
    found = [block for block in tutil.type_search(cat.Block, term)
             if block.block_tag.aesthetics is not None
             and block.block_tag.aesthetics.title == title]
    require(len(found) == 1, f'{len(found)} blocks titled {title}')
    return found[0]


def boxes_named(term: cat.Morphism, name: str) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in operations(term, ops.BlockOperator)
                 if node.operator.name is not None
                 and node.operator.name.to_bodies() == name)


def views_named(term: cat.Morphism, name: str) -> tuple[cat.Broadcasted, ...]:
    '''Every view of `term` named `name`, once for every time it is written, the
    bodies of its boxes included.'''
    return tuple(node for node in operations(term, ops.View)
                 if node.operator.name is not None
                 and node.operator.name.to_bodies() == name)


def views_named_outside_boxes(term: cat.Morphism,
                              name: str) -> tuple[cat.Broadcasted, ...]:
    '''Every view of `term` named `name` that no box holds.'''
    return tuple(leaf.wraps for leaf in quantise_model.graph_leaves(term)
                 if isinstance(leaf.wraps, cat.Broadcasted)
                 and isinstance(leaf.wraps.operator, ops.View)
                 and leaf.wraps.operator.name is not None
                 and leaf.wraps.operator.name.to_bodies() == name)


def cast_pairs(term: cat.Morphism) -> list[tuple[str, str]]:
    return [(Quantization.format_name(quantisation_of(cast.operator.source)),
             Quantization.format_name(quantisation_of(cast.operator.target)))
            for cast in quantise_model.casts_of(term)]


def ends_of(morphism: cat.Morphism) -> set[Quantization.Quantified | None]:
    return {quantisation_of(array.datatype)
            for array in (*morphism.dom(), *morphism.cod())}


def results_of(node: cat.Broadcasted) -> Quantization.Quantified | None:
    return quantisation_of(node.output_weaves[0].datatype)


def operands_of(node: cat.Broadcasted) -> set[Quantization.Quantified | None]:
    return {quantisation_of(weave.datatype) for weave in node.input_weaves}


def listing(term: cat.Morphism) -> str:
    return ad.listing(h2m.recycle(term))


def carries(term: cat.Morphism, axis: cat.Axis) -> bool:
    return any(candidate == axis
               for operation in tutil.type_search(cat.Broadcasted, term)
               for array in (*operation.dom(), *operation.cod())
               for candidate in array.shape())


def as_sent_to_the_page(term: cat.Morphism,
                        settings: notebook_diagrams.DiagramSettings) -> cat.Morphism:
    '''`term` as `show_diagram` sends it under `settings`, after every presentation
    pass and the wrapping of the explained operators.'''
    presented = notebook_diagrams.present_each_side(term, settings)
    sent, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
    return sent


# ==========================================================================
# The sizes of the released model.
# ==========================================================================
def check_the_sizes_are_those_of_the_released_configuration() -> None:
    '''The configuration binds every size symbol of the model to the size of the
    released weights, 32 query heads are 8 key-value heads of 4 query heads, and the 64
    rotary pairs are half of the 128 channels of a head.'''
    config = generate_config.NumericConfig.template(MODEL)
    config.assign_values(**cached_mixtral_8x7b.RELEASED_SIZES)
    sizes = config.assigned_integers_by_name()
    require(sizes == cached_mixtral_8x7b.RELEASED_SIZES,
            f'the configuration binds {sizes}')
    key_value_heads = sizes["h'"]
    require(key_value_heads * sizes['g'] == QUERY_HEADS,
            f'{key_value_heads} key-value heads of {sizes["g"]} query heads')
    require(sizes['m'] // QUERY_HEADS == sizes['d'], 'the head width is not m / 32')
    require(mixtral_8x7b.t.local_size()
            == mixtral_8x7b.d.local_size() * mixtral_8x7b.HALF,
            'the rotary pairs are not half of the channels of a head')


# ==========================================================================
# Grouped-query attention with a rotary embedding.
# ==========================================================================
def check_the_query_projection_produces_groups_of_heads() -> None:
    '''The query projection produces the queries of every token as h' groups of g
    heads, and no view groups the heads.'''
    projections = {node.operator.name.to_bodies(): node
                   for node in tutil.type_search(cat.Broadcasted, ATTENTION)
                   if isinstance(node.operator, ops.Linear)}
    require(axes(projections['W^{Q}']) == ([['x', 'm']], [['x', "h'", 'g', 'd']]),
            f"W^Q maps {axes(projections['W^{Q}'])}")
    names = {operator.name.to_bodies() for operator in operators_of(ATTENTION)
             if isinstance(operator, ops.View)}
    require(names == {mixtral_8x7b.MASK_NAME},
            f'the views of the attention are {names}')


def check_the_keys_and_the_values_carry_the_key_value_heads() -> None:
    '''The keys and the values carry the key-value heads and no query head, so every
    query head of a group reads the same key-value head.'''
    require(axes(mixtral_8x7b.read_earlier_tokens())[1] == [['x', 'w|x', "h'", 'd']],
            f'the keys are read as {axes(mixtral_8x7b.read_earlier_tokens())[1]}')


def check_one_box_turns_the_queries_and_the_keys() -> None:
    '''The queries and the keys are turned by one box, RoPE, written over the vector of
    one head of one token and computed once per head, whose table turns by the base
    beta.'''
    rotations = [node for node in tutil.type_search(cat.Broadcasted, ATTENTION)
                 if node.operator == mixtral_8x7b.ROTARY_EMBEDDING.operator]
    require(len(rotations) == 2, f'{len(rotations)} rotations')
    require(axes(mixtral_8x7b.ROTARY_EMBEDDING) == ([['x', 'd']], [['x', 'd']]),
            f'the box maps {axes(mixtral_8x7b.ROTARY_EMBEDDING)}')
    tables = [operator for operator in operators_of(ATTENTION)
              if isinstance(operator, dst.Rotary)]
    require(tables and all(table.base == mixtral_8x7b.ROTARY_BASE for table in tables),
            'a table turns by another base')


# ==========================================================================
# The causal mask and the CausalSlide.
# ==========================================================================
def check_the_mask_marks_its_slot_axis() -> None:
    '''The view named Mask reads token i_x - i_w at slot i_w, marks its slot axis as an
    affine sparse axis w|x as long as the sequence, and in a prompt of four tokens token
    i reads the slots 0 to i. The model holds no mask operator.'''
    slots = mixtral_8x7b.read_earlier_tokens().cod()[0].shape()[1]
    require(isinstance(slots, AffineGuards.AffineSparseAxis),
            'the slots are not guarded')
    require(mixtral_8x7b.w.local_size() == mixtral_8x7b.x.local_size(),
            'the slot axis is not as long as the sequence')
    live = [mark_sparse_domains.live_positions(
                slots, (token,),
                {mixtral_8x7b.x.local_size(): PROMPT_LENGTH_IN_THE_EXAMPLE})
            for token in range(PROMPT_LENGTH_IN_THE_EXAMPLE)]
    require(live == [list(range(token + 1))
                     for token in range(PROMPT_LENGTH_IN_THE_EXAMPLE)],
            f'the live slots are {live}')
    require(not any(isinstance(operator, ops.WeightedTriangularLower)
                    for operator in operators_of(MODEL)),
            'the model holds a mask operator')


def check_the_causal_slide_reads_the_state_once() -> None:
    '''In the CausalSlide one view named Mask reads the normalised state at the copy
    that feeds the queries, the key and the value projections run over the tokens and
    the slots, and the query projection reads the state of each token.'''
    masks = views_named_outside_boxes(SLID_ATTENTION, mixtral_8x7b.MASK_NAME)
    require(len(masks) == 1 and tuple(masks[0].dom()) == (mixtral_8x7b.STATE,),
            f'{len(masks)} masks, reading {[tuple(mask.dom()) for mask in masks]}')
    projections = {node.operator.name.to_bodies(): node
                   for node in tutil.type_search(cat.Broadcasted, SLID_ATTENTION)
                   if isinstance(node.operator, ops.Linear)}
    for name in ('W^{K}', 'W^{V}'):
        require(axes(projections[name])[0] == [['x', 'w|x', 'm']],
                f'{name} reads {axes(projections[name])[0]}')
    require(axes(projections['W^{Q}'])[0] == [['x', 'm']],
            f"W^Q reads {axes(projections['W^{Q}'])[0]}")


def check_the_keys_are_turned_at_the_position_of_their_slot() -> None:
    '''In the CausalSlide the box turning the keys reads the table of turns through the
    view named Mask, so the key at slot i_w of token i_x is turned by the angle of
    position i_x - i_w. The two rotary boxes hold two bodies under two block tags, so a
    figure draws both.'''
    key_boxes = [box for box in boxes_named(SLID_ATTENTION, mixtral_8x7b.ROTARY_BOX)
                 if axes(box)[0] == [['x', 'w|x', "h'", 'd']]]
    require(len(key_boxes) == 1, f'{len(key_boxes)} boxes turn the keys')
    body = key_boxes[0].operator.block.body
    tables = [node for node in tutil.type_search(cat.Broadcasted, body)
              if isinstance(node.operator, dst.Rotary)]
    reads = views_named(body, mixtral_8x7b.MASK_NAME)
    require(len(tables) == 1 and len(reads) == 1
            and tuple(reads[0].dom()) == tuple(tables[0].cod()),
            'the key box does not read its table through the mask')
    tags = {box.operator.block.block_tag.uid
            for box in boxes_named(SLID_ATTENTION, mixtral_8x7b.ROTARY_BOX)}
    require(len(tags) == 2, f'the two rotary boxes hold {len(tags)} block tags')


# ==========================================================================
# The sparse mixture of experts.
# ==========================================================================
def check_the_router_keeps_two_experts_on_a_sparse_axis() -> None:
    '''The top-k keeps |k| of the n experts on a sparse axis over the experts, and the
    mixture holds one selection and one softmax, taken over the selected experts.'''
    selection = mixtral_8x7b.selection()
    chosen = selection.cod()[0].shape()[-1]
    require(isinstance(chosen, dst.SparseAxis)
            and selection.operator.k == mixtral_8x7b.selected_count,
            'the selection is not a top-k onto a sparse axis')
    operators = operators_of(MIXTURE)
    require(sum(isinstance(operator, dst.TopK) for operator in operators) == 1
            and sum(isinstance(operator, ops.SoftMax) for operator in operators) == 1,
            'the mixture holds another count of selections or softmaxes')
    softmax, = [node for node in tutil.type_search(cat.Broadcasted, MIXTURE)
                if isinstance(node.operator, ops.SoftMax)]
    require(axes(softmax)[0] == [['x', 'k/n']], f'the softmax reads {axes(softmax)[0]}')


def check_the_expert_projections_read_the_chosen_experts() -> None:
    '''Each expert projection produces the expert axis, which composition aligns with
    the sparse axis k/n, and the view named Diagonal keeps the output of the expert
    each slot chose.'''
    names = {mixtral_8x7b.table_key(name) for name in (
        mixtral_8x7b.FIRST_EXPERT_NAME, mixtral_8x7b.THIRD_EXPERT_NAME)}
    for node in tutil.type_search(cat.Broadcasted, MIXTURE):
        if (isinstance(node.operator, ops.Linear)
                and node.operator.name.to_bodies() in names):
            require(axes(node)[1] == [['x', 'k/n', 'f']],
                    f'{node.operator.name.to_bodies()} returns {axes(node)[1]}')
    diagonal, = views_named(MIXTURE, mixtral_8x7b.DIAGONAL_NAME)
    require(axes(diagonal) == ([['x', 'k/n', 'k/n', 'm']], [['x', 'k/n', 'm']]),
            f'the diagonal maps {axes(diagonal)}')


# ==========================================================================
# The decoder layer and the whole model.
# ==========================================================================
def check_the_layer_repeats_and_the_model_reads_tokens() -> None:
    '''The decoder layer returns the array it reads and repeats N times, and the model
    reads one token identifier per position and returns one distribution over the
    vocabulary per position.'''
    layer = mixtral_8x7b.decoder_layer()
    require(layer.dom() == layer.cod()
            and layer.block_tag.repetition == mixtral_8x7b.LAYER_COUNT,
            'the layer is not a loop over the hidden state')
    require(cached_mixtral_8x7b.RELEASED_SIZES['N'] == 32, 'N is not 32')
    require(axes(MODEL) == ([['x']], [['x', 'v']]) and axes(DECODE) == axes(MODEL),
            f'the model maps {axes(MODEL)}')


# ==========================================================================
# The quantisations of the reference implementation.
# ==========================================================================
def check_every_weight_is_bf16() -> None:
    '''Every weight of the model is read in BF16, a format that packs two values to a
    word and is labelled 2BF16.'''
    held = dict(quantised_mixtral_8x7b.WEIGHT_QUANTISATIONS)
    require(held == {name: BF16 for name in WEIGHT_NAMES}, f'the weights are {held}')
    require(BF16.vector == nm.Integer(2),
            f'BF16 packs {BF16.vector.to_latex()} values to a word, and its label '
            'reads otherwise')


def check_every_wire_carries_a_quantisation() -> None:
    '''Every wire of the quantised decode form and of the quantised cached pass carries
    a quantisation, and every operator of the model has a rule.'''
    for label, term in (('decode', QUANTISED_DECODE), ('cached', QUANTISED_CACHED)):
        unwritten = quantised_mixtral_8x7b.unquantised_weaves(term)
        require(not unwritten, f'{len(unwritten)} wires of the {label} form carry none')
    require(not quantised_mixtral_8x7b.operators_without_a_rule(),
            f'{quantised_mixtral_8x7b.operators_without_a_rule()} have no rule')


def check_the_model_reads_int64_and_returns_fp32() -> None:
    '''The model reads the token identifiers in INT64 and returns FP32, and every
    residual connection and every RMSNorm reads and returns BF16.'''
    tokens, = QUANTISED_DECODE.dom()
    result, = QUANTISED_DECODE.cod()
    require(quantisation_of(tokens.datatype) == INT64
            and quantisation_of(result.datatype) == FP32,
            f'the model reads {tokens.datatype} and returns {result.datatype}')
    for block in tutil.type_search(cat.Block, QUANTISED_DECODE):
        aesthetics = block.block_tag.aesthetics
        if aesthetics is not None and aesthetics.title in (
                text.ATTENTION_RESIDUAL_TITLE, text.MIXTURE_RESIDUAL_TITLE):
            require(ends_of(block) == {BF16},
                    f'a residual reads and returns {ends_of(block)}')
    for norm in operations(QUANTISED_DECODE, ops.Normalize):
        require(ends_of(norm) == {BF16},
                f'an RMSNorm reads and returns {ends_of(norm)}')


def check_the_rotary_embedding_turns_in_fp32() -> None:
    '''Each box of the rotary embedding reads and returns BF16. Inside it the table of
    turns is FP32, the pairs are read into FP32 before the product, and the turned
    channels are rounded to BF16.'''
    boxes = boxes_named(QUANTISED_DECODE, mixtral_8x7b.ROTARY_BOX)
    require(len(boxes) == 2, f'{len(boxes)} rotary boxes')
    for box in boxes:
        require(ends_of(box) == {BF16},
                f'a rotary box reads and returns {ends_of(box)}')
        body = box.operator.block.body
        require(cast_pairs(body) == [('BF16', 'FP32'), ('FP32', 'BF16')],
                f'a rotary body casts {cast_pairs(body)}')
        table, = operations(body, dst.Rotary)
        product, = operations(body, ops.Einops)
        require(results_of(table) == FP32 and operands_of(product) == {FP32}
                and results_of(product) == FP32, 'the turn is not FP32')


def check_the_attention_core_is_flash_attention() -> None:
    '''The core reads BF16 queries, keys and values and returns BF16. Its scores, their
    scaling and their softmax are FP32, the probabilities are rounded to BF16 before the
    product with the values, and that product accumulates in FP32.'''
    core = block_titled(QUANTISED_DECODE, text.CORE_TITLE)
    require(ends_of(core) == {BF16}, f'the core reads and returns {ends_of(core)}')
    require(cast_pairs(core.body) == [('FP32', 'BF16'), ('FP32', 'BF16')],
            f'the core casts {cast_pairs(core.body)}')
    scores, weighted = operations(core.body, ops.Einops)
    require(operands_of(scores) == {BF16} and results_of(scores) == FP32,
            'the scores are not an FP32 product of BF16 operands')
    require(operands_of(weighted) == {BF16} and results_of(weighted) == FP32,
            'the weighted sum is not an FP32 product of BF16 operands')
    scaling, = operations(core.body, ops.Arithmetic)
    softmax, = operations(core.body, ops.SoftMax)
    require(results_of(scaling) == FP32 and results_of(softmax) == FP32,
            'the scaling or the softmax is not FP32')


def check_the_router_takes_its_softmax_in_fp32() -> None:
    '''The router keeps the top two of the BF16 scores, reads them into FP32 for the
    softmax, and rounds the weights to BF16.'''
    router = block_titled(QUANTISED_DECODE, text.ROUTER_TITLE)
    selection, = operations(router.body, dst.TopK)
    softmax, = operations(router.body, ops.SoftMax)
    require(operands_of(selection) == {BF16}, 'the top-2 does not read BF16')
    require(results_of(softmax) == FP32, 'the softmax is not FP32')
    require(cast_pairs(router.body) == [('BF16', 'FP32'), ('FP32', 'BF16')],
            f'the router casts {cast_pairs(router.body)}')
    require({quantisation_of(array.datatype) for array in router.cod()} == {BF16},
            'the router does not return BF16')


def check_the_experts_compute_in_bf16() -> None:
    '''The SiLU and the product of the two branches of every expert return BF16, no
    expert casts, and the weighted sum of the two chosen experts is BF16.'''
    experts = block_titled(QUANTISED_DECODE, text.EXPERTS_TITLE)
    require(not cast_pairs(experts.body),
            f'the experts cast {cast_pairs(experts.body)}')
    for node in (*operations(experts.body, ops.Arithmetic),
                 *operations(experts.body, ops.Einops)):
        require(results_of(node) == BF16, f'{node.operator} returns {results_of(node)}')
    mixture = block_titled(QUANTISED_DECODE, text.MIXTURE_TITLE)
    combine = [node for node in operations(mixture.body, ops.Einops)
               if axes(node)[1] == [['x', 'm']]]
    require(len(combine) == 1 and results_of(combine[0]) == BF16
            and operands_of(combine[0]) == {BF16}, 'the weighted sum is not BF16')


def check_the_logits_are_read_into_fp32() -> None:
    '''The output projection returns BF16 scores, which are read into FP32 before the
    softmax.'''
    output = block_titled(QUANTISED_DECODE, text.OUTPUT_TITLE)
    require(cast_pairs(output.body) == [('BF16', 'FP32')],
            f'the output casts {cast_pairs(output.body)}')
    softmax, = operations(output.body, ops.SoftMax)
    require(operands_of(softmax) == {FP32}, 'the softmax does not read FP32')


def check_the_casts_of_the_decode_form() -> None:
    '''The quantised decode form writes nine casts, four upcasts and five roundings.'''
    counts = dict(quantised_mixtral_8x7b.cast_counts())
    require(counts == DECODE_CAST_COUNTS, f'the casts are {counts}')
    require(quantise_model.conversions_between_two_quantisations(QUANTISED_DECODE),
            'a conversion reads or writes no quantisation')


def check_removing_the_quantisations_returns_the_decode_form() -> None:
    '''Removing every quantisation from the quantised decode form leaves no
    quantisation and no conversion and returns the decode form, and the Python
    statement of the page's functor does the same.'''
    stripped = quantised_mixtral_8x7b.mixtral_without_quantisations
    require(not tuple(tutil.type_search(Quantization.Quantified, stripped)),
            'the stripped form carries a quantisation')
    require(not quantise_model.conversions_of(stripped),
            'the stripped form holds a cast')
    require(listing(stripped) == listing(DECODE), 'the stripped form differs')
    functor = notebook_diagrams.apply_page_functor(
        notebook_diagrams.PageFunctor.DEQUANTISE, QUANTISED_DECODE)
    require(listing(functor) == listing(DECODE), 'the page functor differs')


# ==========================================================================
# The cached pass.
# ==========================================================================
def check_the_pass_reads_the_new_tokens_alone() -> None:
    '''The pass reads the token identifiers of the new tokens and returns their
    distributions, and no operation of the pass carries the token axis of the model.'''
    new_tokens = cached_mixtral_8x7b.NEW_TOKENS
    require(axes(CACHED) == ([['xnew']], [['xnew', 'v']]),
            f'the pass maps {axes(CACHED)}')
    require(tuple(CACHED.dom()[0].shape()) == (new_tokens,),
            'the pass reads another axis')
    require(not carries(CACHED, mixtral_8x7b.x), 'an operation carries x')


def check_the_caches_hold_the_turned_keys_and_the_values() -> None:
    '''The pass caches the keys after the rotary embedding and the values, one array of
    each over every cached token and every key-value head, and holds no other cache.'''
    caches = CACHED_PASS.caches()
    names = sorted(cache_contents.cache_name(cache) for cache in caches)
    require(names == ['c_{RoPE}', 'c_{W^{V}}'], f'the caches are {names}')
    cached_tokens = CACHED_PASS.cached_tokens
    for cache in caches:
        require(tuple(cache.cod()[0].shape()) == (cached_tokens, mixtral_8x7b.kv_heads,
                                                  mixtral_8x7b.d),
                f'a cache holds {axes(cache)[1]}')
    require(not CACHED_PASS.kept_tokens, 'a cache keeps only some earlier tokens')
    before = {operator.name.to_bodies()
              for operator in CACHED_PASS.operators_before_caches}
    require(before == {mixtral_8x7b.ROTARY_BOX, 'W^{V}'}, f'the caches follow {before}')


def check_the_mask_reads_the_cache_back_from_the_new_token() -> None:
    '''Slot i_w of new token i_new reads position |x_old| + i_new - i_w of the cache.'''
    masks = views_named_outside_boxes(CACHED, mixtral_8x7b.MASK_NAME)
    require(len(masks) == 2, f'{len(masks)} masks')
    for view in masks:
        (row_axis, strides, shift), *_ = move_reads_backwards.as_stride_morphism(
            view.reindexings[0])._cod_stride_shift
        require(row_axis == CACHED_PASS.cached_tokens
                and shift == cached_mixtral_8x7b.EARLIER_TOKENS.local_size()
                and [stride.to_latex() for stride in strides[:2]] == ['1', '-1'],
                f'the mask reads {row_axis} at '
                f'{[stride.to_latex() for stride in strides]} + {shift}')


def check_the_placements_of_one_layer() -> None:
    '''One layer has six placements. The reference's caches stand on the operands of
    the causal reads, keep 2,048 values per token, and make the cheapest pass at one new
    token after 32,767 earlier ones. Caching the keys before the rotation keeps as many
    values and recomputes the rotation of every cached key. The state cached once keeps
    4,096 values, and every placement caching the state takes several times as long.'''
    rows = cached_mixtral_8x7b.placement_rows()
    require(len(rows) == 6, f'{len(rows)} placements')
    by_caches = {row.caches: row for row in rows}
    reference = by_caches[('c_{RoPE}', 'c_{W^{V}}')]
    before_rotation = by_caches[('c_{W^{K}}', 'c_{W^{V}}')]
    state_once = by_caches[('c_{0}',)]
    require(reference.computed_over_the_cache == 0
            and reference.values_per_token == 2048,
            f'the reference placement is {reference}')
    require(before_rotation.values_per_token == 2048
            and before_rotation.microseconds > reference.microseconds,
            f'caching before the rotation is {before_rotation}')
    require(min(rows, key=lambda row: row.microseconds) == reference,
            'another placement is cheaper than the reference')
    require(state_once.values_per_token == 4096,
            f'the state cached once is {state_once}')
    require(all(row.microseconds > STATE_CACHING_SLOWDOWN * reference.microseconds
                for row in rows if 'c_{0}' in row.caches),
            'a placement caching the state is less than several times as slow')
    reference_pass = cached_mixtral_8x7b.placement_on_the_operands_of_the_causal_reads(
        cached_mixtral_8x7b.LAYER_PLACEMENTS)
    require(cached_mixtral_8x7b.cache_names(reference_pass) == reference.caches,
            'the placement on the operands of the reads is another')
    machine = cached_mixtral_8x7b.MACHINE
    require(machine.matrix_operations_per_second == 989e12
            and machine.memory_bytes_per_second == 3.35e12
            and machine.bytes_per_element == 2,
            f'the machine is {machine}')
    require(cached_mixtral_8x7b.DECODE_SIZES['xold'] == 32767
            and cached_mixtral_8x7b.DECODE_SIZES['xnew'] == 1,
            'the pass reads other sizes')


def check_the_causal_slide_caches_the_normalised_state() -> None:
    '''The pass derived from the CausalSlide caches the normalised state once in each
    layer, 4,096 values per token, as the last placement of the table does.'''
    slid_pass = derive_cached_pass.derive_cached_pass(
        DECODE, mixtral_8x7b.x, cached_mixtral_8x7b.EARLIER_TOKENS,
        cached_mixtral_8x7b.NEW_TOKENS)
    last = cached_mixtral_8x7b.placement_rows()[-1]
    require(cached_mixtral_8x7b.cache_names(slid_pass) == last.caches == ('c_{0}',),
            f'the CausalSlide caches {cached_mixtral_8x7b.cache_names(slid_pass)}')
    slid_values = cached_mixtral_8x7b.values_per_token(slid_pass)
    require(slid_values == last.values_per_token == 4096,
            f'the CausalSlide caches {slid_values} values')


def check_the_cache_of_one_token() -> None:
    '''The caches hold 128 KiB for one token over the 32 layers, and 4 GiB for the
    32,768 tokens of the context.'''
    per_token = cached_mixtral_8x7b.cached_bytes_per_token()
    require(per_token == 128 * KIBIBYTE, f'{per_token} bytes per token')
    require(per_token * CONTEXT_LENGTH == 4 * GIBIBYTE, 'the context is not 4 GiB')


def check_the_quantised_caches_hold_bf16() -> None:
    '''Every cache of the quantised pass saves and loads BF16, the pass reads INT64 and
    returns FP32, and it writes seven casts.'''
    caches = [node for node in tutil.type_search(cat.Broadcasted, QUANTISED_CACHED)
              if isinstance(node.operator, Caching.Caching)]
    require(len(caches) == 2 and all(ends_of(cache) == {BF16} for cache in caches),
            f'the caches hold {[ends_of(cache) for cache in caches]}')
    require(ends_of(QUANTISED_CACHED) == {INT64, FP32},
            f'the pass ends in {ends_of(QUANTISED_CACHED)}')
    counts = dict(quantised_mixtral_8x7b.cast_counts(QUANTISED_CACHED))
    require(counts == CACHED_CAST_COUNTS, f'the casts are {counts}')


def check_removing_the_quantisations_returns_the_cached_pass() -> None:
    '''Removing every quantisation from the quantised cached pass returns the cached
    pass derived from the unquantised model, and the Python statement of the page's
    functor does the same.'''
    stripped = cached_mixtral_8x7b.mixtral_cached_without_quantisations
    require(not tuple(tutil.type_search(Quantization.Quantified, stripped)),
            'the stripped pass carries a quantisation')
    require(listing(stripped) == listing(CACHED), 'the stripped pass differs')
    functor = notebook_diagrams.apply_page_functor(
        notebook_diagrams.PageFunctor.DEQUANTISE, QUANTISED_CACHED)
    require(listing(functor) == listing(CACHED), 'the page functor differs')


# ==========================================================================
# The interactive page.
# ==========================================================================
def check_the_page_holds_four_variants() -> None:
    '''The page holds a quantised and an unquantised variant of the decode form and of
    the cached pass, opens on the quantised decode form, and derives each unquantised
    variant from its quantised variant by removing the quantisations. The inspection
    box over a weight states its format in a quantised variant alone.'''
    notebook_diagrams.check_page_variants(
        VARIANTS, mixtral_8x7b_page_variants.INITIAL_VARIANT)
    require([(variant.identifier, variant.group.identifier, variant.title)
             for variant in VARIANTS]
            == [('decode-quantised', 'decode', 'Quantised'),
                ('decode-unquantised', 'decode', 'Unquantised'),
                ('cached-quantised', 'cached', 'Quantised'),
                ('cached-unquantised', 'cached', 'Unquantised')],
            'the variants differ')
    for variant in VARIANTS:
        if variant.title == 'Unquantised':
            require(variant.term is None
                    and variant.functor is notebook_diagrams.PageFunctor.DEQUANTISE
                    and variant.derived_from == variant.identifier.replace(
                        'unquantised', 'quantised'),
                    f'{variant.identifier} is not derived from its quantised variant')
    require(mixtral_8x7b_page_variants.INITIAL_VARIANT == 'decode-quantised',
            'the page opens on another variant')
    for variant in VARIANTS:
        roles = variant.settings.operator_roles
        states_a_quantisation = [
            name for name, role in roles.items()
            if name in quantised_mixtral_8x7b.WEIGHT_QUANTISATIONS
            and text.WEIGHT_QUANTISATION_SENTENCE.format(quantisation='BF16')
            in role.role]
        if variant.term is None:
            require(not states_a_quantisation,
                    f'{variant.identifier} states the format of {states_a_quantisation}')
        else:
            require(len(states_a_quantisation)
                    == len(quantised_mixtral_8x7b.WEIGHT_QUANTISATIONS),
                    f'{variant.identifier} states the format of {states_a_quantisation}')


def check_every_legend_row_carries_a_code_name() -> None:
    '''Every variant draws a legend, every row of every legend carries the code name of
    its axis, and every axis whose size is one named symbol carries the code name of
    that symbol.'''
    by_identifier = {variant.identifier: variant for variant in VARIANTS}
    legends = notebook_diagrams.page_variant_legends(VARIANTS, PAGE)
    for variant in VARIANTS:
        settings = notebook_diagrams.settings_of_page_variant(
            variant, by_identifier, PAGE)
        require(settings.advanced_display is figures.AdvancedDisplay.INTERACTIVE,
                f'{variant.identifier} opens no inspection box')
        rows = legends[variant.identifier]
        require(rows, f'{variant.identifier} draws no legend')
        missing = [row['text'] for row in rows if not row['codeName']]
        require(not missing,
                f'the rows {missing} of {variant.identifier} have no code name')
        term = notebook_diagrams.term_of_page_variant(variant, by_identifier)
        for axis in tutil.type_search(cat.Axis, as_sent_to_the_page(term, settings)):
            size = axis.local_size()
            if (axis.uid._name is not None and isinstance(size, nm.FreeNumeric)
                    and size.uid._name is not None):
                require(size.uid._name.code_form is not None,
                        f'the size of {axis.uid._name.to_bodies()} has no code name')


def check_every_wire_and_view_of_the_page_opens_as_drawn() -> None:
    '''Every wire of a quantised variant as it is sent to the page carries its
    quantisation, and every named view of every variant opens an inspection box, the
    view named New reading the table of turns at the positions of the new tokens
    included.'''
    by_identifier = {variant.identifier: variant for variant in VARIANTS}
    for variant in VARIANTS:
        settings = notebook_diagrams.settings_of_page_variant(
            variant, by_identifier, PAGE)
        sent = as_sent_to_the_page(
            notebook_diagrams.term_of_page_variant(variant, by_identifier), settings)
        if variant.term is not None:
            unwritten = quantise_model.unquantised_weaves(sent)
            require(not unwritten,
                    f'{len(unwritten)} wires of {variant.identifier} carry no '
                    'quantisation')
        unexplained = explain_reindexings.names_of_unexplained_views(sent)
        require(not unexplained,
                f'the views {unexplained} of {variant.identifier} open no box')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_sizes_are_those_of_the_released_configuration,
    check_the_query_projection_produces_groups_of_heads,
    check_the_keys_and_the_values_carry_the_key_value_heads,
    check_one_box_turns_the_queries_and_the_keys,
    check_the_mask_marks_its_slot_axis,
    check_the_causal_slide_reads_the_state_once,
    check_the_keys_are_turned_at_the_position_of_their_slot,
    check_the_router_keeps_two_experts_on_a_sparse_axis,
    check_the_expert_projections_read_the_chosen_experts,
    check_the_layer_repeats_and_the_model_reads_tokens,
    check_every_weight_is_bf16,
    check_every_wire_carries_a_quantisation,
    check_the_model_reads_int64_and_returns_fp32,
    check_the_rotary_embedding_turns_in_fp32,
    check_the_attention_core_is_flash_attention,
    check_the_router_takes_its_softmax_in_fp32,
    check_the_experts_compute_in_bf16,
    check_the_logits_are_read_into_fp32,
    check_the_casts_of_the_decode_form,
    check_removing_the_quantisations_returns_the_decode_form,
    check_the_pass_reads_the_new_tokens_alone,
    check_the_caches_hold_the_turned_keys_and_the_values,
    check_the_mask_reads_the_cache_back_from_the_new_token,
    check_the_placements_of_one_layer,
    check_the_causal_slide_caches_the_normalised_state,
    check_the_cache_of_one_token,
    check_the_quantised_caches_hold_bf16,
    check_removing_the_quantisations_returns_the_cached_pass,
    check_the_page_holds_four_variants,
    check_every_legend_row_carries_a_code_name,
    check_every_wire_and_view_of_the_page_opens_as_drawn,
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
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} Mixtral-8x7B checks passed in '
          f'{time.perf_counter() - started:.0f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
