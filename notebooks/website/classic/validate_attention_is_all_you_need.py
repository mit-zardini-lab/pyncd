# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim `notebooks/website/classic/AttentionIsAllYouNeed.ipynb`
makes about the transformer of *Attention Is All You Need*.

    python notebooks/website/classic/validate_attention_is_all_you_need.py

The notebook is public facing and holds the prose and the figures alone. The claims live
here, one `check_` function per claim, in the order the notebook makes them: the sizes
of the base model, the embedding and the positional encoding, the attention of the
encoder, the mask of the decoder and its CausalSlide, the cross-attention, the
feed-forward layer and the layer normalisation, the two stacks, the whole model, the
quantisations of tensor2tensor, the cached pass of the decoder and what it does with
the encoder, the quantisations of the cached pass, the unquantised variants of the
page, and the differences from the hand-drawn diagram. The script prints one line per
check and the time the run took, and it exits non-zero on a failure.

The page derives each unquantised variant in the browser by removing every quantisation
from the quantised variant. `check_removing_the_quantisations_returns_the_decode_form`
and `check_removing_the_quantisations_returns_the_cached_form` apply the Python
statement of that functor and compare its result with the unquantised form. The two
agree once each is recycled through a hypergraph and the identity of every block tag is
forgotten, because the quantisation pass gives the body of a box a tag of its own.
'''
from __future__ import annotations

import cmath
import pathlib
import sys
import time
from collections.abc import Callable
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa: E402
import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards  # noqa: E402
import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa: E402,E501
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import algebra.broadcasted_recycle as broadcasted_recycle  # noqa: E402
import caching.algebra.cache_contents as cache_contents  # noqa: E402
import caching.data_structure.Caching as Caching  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import term_utilities.generate_config as generate_config  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
from para.data_structure.ParaBlockOperator import (  # noqa: E402
    slots_dropped, slots_grabbed)

import notebooks.classic.attention_is_all_you_need as attention_is_all_you_need  # noqa: E402
import notebooks.classic.attention_is_all_you_need_page_variants as attention_is_all_you_need_page_variants  # noqa: E402,E501
import notebooks.classic.cached_attention_is_all_you_need as cached_attention_is_all_you_need  # noqa: E402,E501
import notebooks.classic.quantised_attention_is_all_you_need as quantised_attention_is_all_you_need  # noqa: E402,E501
import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.sota_figures as figures  # noqa: E402
from notebooks.classic.attention_is_all_you_need_wording import TEXT as text  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axes  # noqa: E402

MODEL = attention_is_all_you_need.transformer()
CONFIG = generate_config.NumericConfig.template(MODEL)
CONFIG.assign_values(m=512, h=8, k=64, f=2048, N=6)
SIZES: dict[str, int] = CONFIG.assigned_integers_by_name()

DECODE = slide_causal_reads_backwards.slide_causal_reads_backwards(MODEL)
REFERENCE_PASS = cached_attention_is_all_you_need.reference_placement()
CACHED = REFERENCE_PASS.expression
QUANTISED_DECODE = quantised_attention_is_all_you_need.quantise_as_released(DECODE)
QUANTISED_CACHED = quantised_attention_is_all_you_need.quantise_as_released(CACHED)

SOURCE_POSITIONS = attention_is_all_you_need.x
TARGET_POSITIONS = attention_is_all_you_need.y
HEADS = attention_is_all_you_need.h
HEAD_WIDTH = attention_is_all_you_need.k
KEYS_OF_THE_SOURCE = (SOURCE_POSITIONS, HEADS, HEAD_WIDTH)
ENCODER_CORE = attention_is_all_you_need.scaled_dot_product_attention(
    SOURCE_POSITIONS, KEYS_OF_THE_SOURCE, (SOURCE_POSITIONS,))

MASKED_SUBLAYER = attention_is_all_you_need.add_and_norm(
    attention_is_all_you_need.decoder_masked_self_attention(), TARGET_POSITIONS)
SLID_MASKED_SUBLAYER = slide_causal_reads_backwards.slide_causal_reads_backwards(
    MASKED_SUBLAYER)

FP32 = quantised_attention_is_all_you_need.FP32
INT32 = quantised_attention_is_all_you_need.INT32
TARGET_LENGTH_IN_THE_EXAMPLE = 5

NOTEBOOK_DIAGRAMS = figures.DiagramSettings(
    mode=figures.DiagramMode.INLINE,
    dark_mode=notebook_diagrams.ColorMode.LIGHT,
    axis_sizes=figures.AxisSizes.SUBSCRIPT,
    assigned_sizes=SIZES,
    advanced_display=figures.AdvancedDisplay.LEGEND,
    operator_explanations=attention_is_all_you_need.OPERATOR_EXPLANATIONS,
    operator_roles=attention_is_all_you_need.OPERATOR_ROLES,
    operator_references=attention_is_all_you_need.OPERATOR_REFERENCES,
    reindexing_explanations=attention_is_all_you_need.REINDEXING_EXPLANATIONS,
    title='Attention Is All You Need')
'''The settings declared by the setup cell of the notebook.'''

PAGE = attention_is_all_you_need_page_variants.page_settings(NOTEBOOK_DIAGRAMS)
VARIANTS = attention_is_all_you_need_page_variants.page_variants(
    QUANTISED_DECODE.morphism, QUANTISED_CACHED.morphism, PAGE)
VARIANT_IDENTIFIERS = ('decode-quantised', 'decode-unquantised', 'cached-quantised',
                       'cached-unquantised')


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def operations_of(term: object) -> list[cat.Broadcasted]:
    '''Every distinct operation of `term`, inside the body of every box.'''
    return list(tutil.type_search(cat.Broadcasted, term))


def written_operations(term: cat.Morphism) -> list[cat.Broadcasted]:
    '''Every operation written in `term`, once for every place it is written, so
    two equal views written in two places count twice.'''
    return list(quantise_model.operations_of(term))


def written_classes(term: cat.Morphism) -> list[str]:
    return sorted(type(operation.operator).__name__
                  for operation in written_operations(term))


def top_level_classes(term: cat.Morphism) -> list[str]:
    '''The operator classes of `term` outside the body of every box.'''
    return sorted(type(leaf.wraps.operator).__name__
                  for leaf in quantise_model.graph_leaves(term)
                  if isinstance(leaf.wraps, cat.Broadcasted))


def operators_of(term: object) -> list[cat.Operator]:
    return [operation.operator for operation in operations_of(term)]


def operator_classes_of(term: object) -> list[type[cat.Operator]]:
    return [type(operator) for operator in operators_of(term)]


def name_of(operator: cat.Operator) -> str | None:
    return None if operator.name is None else operator.name.to_text()


def views_named(name: str, term: cat.Morphism) -> list[cat.Broadcasted]:
    return [operation for operation in written_operations(term)
            if isinstance(operation.operator, ops.View)
            and name_of(operation.operator) == name]


def linear_maps_named(name: str, term: object) -> list[cat.Broadcasted]:
    return [operation for operation in operations_of(term)
            if isinstance(operation.operator, ops.Linear)
            and name_of(operation.operator) == name]


def size_of(axis: cat.Axis) -> int:
    return cache_contents.size_of(axis.local_size(), SIZES)


def carries(term: object, axis: cat.Axis) -> bool:
    return any(candidate == axis
               for operation in operations_of(term)
               for array in (*operation.dom(), *operation.cod())
               for candidate in array.shape())


def with_block_tags_forgotten[T](term: T) -> T:
    '''`term` with the identity of every block tag replaced by one fixed identity, each
    shared part visited once.'''
    rewritten: dict[int, tuple[object, object]] = {}

    def forget(part: Any) -> Any:
        found = rewritten.get(id(part))
        if found is not None:
            return found[1]
        if isinstance(part, cat.BlockTag):
            result = part.reconstruct(uid=part.uid.reconstruct(_id=0))
        elif isinstance(part, (fd.Term, tuple)):
            result = fd.deep_reconstruct(part, forget)
        else:
            result = part
        rewritten[id(part)] = (part, result)
        return result

    return forget(term)


def same_up_to_block_tags(one: cat.Morphism, other: cat.Morphism) -> bool:
    '''Whether `one` and `other` are one expression once each is recycled at every
    level, the body of every box included, and the identity of every block tag is
    forgotten.'''
    recycled = tuple(map(broadcasted_recycle.recycle_broadcasted, (one, other)))
    return (with_block_tags_forgotten(recycled[0])
            == with_block_tags_forgotten(recycled[1]))


def quantisations_on_the_wires(morphism: cat.Morphism) -> set[str]:
    return {Quantization.format_name(Quantization.quantisation_of(weave.datatype))
            for weave in tutil.type_search(cat.Weave, morphism)
            if Quantization.holds_a_number(weave.datatype)}


def token_row(view: cat.Broadcasted, cached_axis: cat.Axis
              ) -> tuple[fd.Prod[nm.Numeric], nm.Numeric]:
    '''The strides and the shift of the row of `view` that reads `cached_axis`.'''
    read = move_reads_backwards.as_stride_morphism(view.reindexings[0])
    rows = [(strides, shift) for axis, strides, shift in read._cod_stride_shift
            if axis == cached_axis]
    require(len(rows) == 1,
            f'{view.operator.name} reads the cached axis {len(rows)} times')
    return rows[0]


# ==========================================================================
# The sizes of the base model.
# ==========================================================================
def check_the_sizes_of_the_base_model() -> None:
    '''m = 512, h = 8, k = 64, f = 2048, N = 6, t = 256 and |w| = |y|.'''
    require(SIZES == {'m': 512, 'h': 8, 'k': 64, 'f': 2048, 'N': 6},
            f'the configuration binds {SIZES}')
    frequency_pairs = size_of(attention_is_all_you_need.t)
    require(frequency_pairs == 256, f'|t| is {frequency_pairs}')
    require(attention_is_all_you_need.w.local_size() == TARGET_POSITIONS.local_size(),
            'the slot axis w has another size than the target')


# ==========================================================================
# The embedding and the positional encoding.
# ==========================================================================
def check_the_embedding_is_scaled_and_added_to_the_positional_encoding() -> None:
    '''The embedding is multiplied by the root of m, the positional encoding is added,
    and dropout is applied to the sum.'''
    embedding = attention_is_all_you_need.embed(
        SOURCE_POSITIONS, text.INPUT_EMBEDDING_TITLE,
        text.INPUT_EMBEDDING_DESCRIPTION)
    classes = top_level_classes(embedding)
    require(classes == ['AdditionOp', 'Arithmetic', 'BlockOperator', 'Dropout',
                        'Embedding'], f'the embedding block holds {classes}')
    scales = [leaf.wraps.operator for leaf in quantise_model.graph_leaves(embedding)
              if isinstance(leaf.wraps.operator, ops.Arithmetic)]
    root_of_m = nm.SquareRoot(attention_is_all_you_need.m.local_size())
    require(len(scales) == 1
            and name_of(scales[0]) == attention_is_all_you_need.EMBEDDING_SCALE_NAME
            and scales[0].formula == nm.x * root_of_m,
            f'the embedding is scaled by {[name_of(scale) for scale in scales]}')


def check_the_positional_encoding_reads_nothing_and_returns_x_by_m() -> None:
    positional = attention_is_all_you_need.positional_encoding(SOURCE_POSITIONS)
    require(len(positional.dom()) == 0 and axes(positional)[1] == [['x', 'm']],
            f'the positional encoding reads and returns {axes(positional)}')


def check_the_positional_encoding_is_the_table_of_turns_written_as_channels() -> None:
    '''The box PE holds the rotary table, the product of the imaginary unit with the
    conjugate, and the writing of each complex number as two channels.'''
    positional = attention_is_all_you_need.positional_encoding(SOURCE_POSITIONS)
    require(isinstance(positional.operator, ops.BlockOperator)
            and positional.operator.name.to_bodies()
            == attention_is_all_you_need.POSITIONAL_ENCODING_BOX,
            'the positional encoding is not the box PE')
    classes = operator_classes_of(positional.operator.block)
    require(classes == [dst.Rotary, ops.Arithmetic, dst.Decomplex],
            f'the body of PE holds {[kind.__name__ for kind in classes]}')


def check_the_imaginary_unit_times_the_conjugate_puts_the_sine_first() -> None:
    for angle in (0.3, 0.7, 2.1):
        turned = 1j * cmath.exp(1j * angle).conjugate()
        require(cmath.isclose(turned, complex(cmath.sin(angle), cmath.cos(angle))),
                f'at the angle {angle} the product is {turned}')


# ==========================================================================
# Scaled dot-product attention and the self-attention of the encoder.
# ==========================================================================
def check_w_o_contracts_the_heads_and_the_head_width() -> None:
    projection = attention_is_all_you_need.output_projection(SOURCE_POSITIONS)
    require(axes(projection) == ([['x', 'h', 'k']], [['x', 'm']]),
            f'W^O reads and returns {axes(projection)}')


def check_attention_reads_queries_keys_and_values_at_x() -> None:
    require(axes(ENCODER_CORE) == ([['x', 'h', 'k']] * 3, [['x', 'h', 'k']]),
            f'scaled dot-product attention reads and returns {axes(ENCODER_CORE)}')


def check_the_encoder_scores_carry_the_source_axis_at_two_positions() -> None:
    scores, = [operation for operation in operations_of(ENCODER_CORE)
               if isinstance(operation.operator, ops.Einops)
               and axes(operation)[1] == [['h', 'x', 'x']]]
    shape = scores.cod()[0].shape()
    require(axes(scores)[1] == [['h', 'x', 'x']] and shape[1] is shape[2],
            f'the scores are over {axes(scores)[1]}')


# ==========================================================================
# The mask of the decoder as a read of the earlier positions.
# ==========================================================================
def check_the_view_named_mask_marks_its_slot_axis() -> None:
    '''The view is named Mask, set upright, and returns the slot axis as w|y.'''
    view = attention_is_all_you_need.read_earlier_targets()
    require(name_of(view.operator) == attention_is_all_you_need.MASK_NAME
            == '\\mathrm{Mask}',
            f'the view is named {name_of(view.operator)}')
    slots = view.cod()[0].shape()[1]
    require(isinstance(slots, AffineGuards.AffineSparseAxis),
            f'the slot axis is a {type(slots).__name__}')


def check_query_i_reads_slots_zero_to_i() -> None:
    slots = attention_is_all_you_need.read_earlier_targets().cod()[0].shape()[1]
    length = attention_is_all_you_need.y.local_size()
    live = [mark_sparse_domains.live_positions(
                slots, (query,), {length: TARGET_LENGTH_IN_THE_EXAMPLE})
            for query in range(TARGET_LENGTH_IN_THE_EXAMPLE)]
    require(live == [list(range(query + 1))
                     for query in range(TARGET_LENGTH_IN_THE_EXAMPLE)],
            f'the queries read the slots {live}')


def check_the_model_holds_no_mask_operator() -> None:
    require(not any(isinstance(operator, ops.WeightedTriangularLower)
                    for operator in operators_of(MODEL)),
            'the model holds a triangular mask operator')


def check_the_slid_sublayer_reads_its_input_through_one_mask() -> None:
    '''In the figure the view named Mask reads the input of the sublayer, and W^K and
    W^V run at every target position and every slot.'''
    masks = views_named(attention_is_all_you_need.MASK_NAME, SLID_MASKED_SUBLAYER)
    require(len(masks) == 1
            and tuple(masks[0].dom()) == (attention_is_all_you_need.TARGET_STATE,),
            f'{len(masks)} masks read {[axes(mask)[0] for mask in masks]}')
    for name in ('W^{K}', 'W^{V}'):
        projection, = linear_maps_named(name, SLID_MASKED_SUBLAYER)
        require(axes(projection)[0] == [['y', 'w|y', 'm']],
                f'{name} reads {axes(projection)[0]}')


def check_the_slide_moves_the_mask_and_changes_no_other_operator() -> None:
    '''The slid sublayer holds the operators of the sublayer as built, with one view in
    place of two.'''
    before = written_classes(MASKED_SUBLAYER)
    after = written_classes(SLID_MASKED_SUBLAYER)
    require(before.count('View') == 2 and after.count('View') == 1,
            f'the sublayer writes {before.count("View")} views and its slide '
            f'{after.count("View")}')
    before.remove('View')
    require(before == after, f'the sublayer holds {before} and its slide {after}')


# ==========================================================================
# Cross-attention reads the encoded input from the tape.
# ==========================================================================
def check_the_encoded_input_is_the_one_slot() -> None:
    require(slots_dropped(MODEL) == slots_grabbed(MODEL)
            == {attention_is_all_you_need.ENCODED_INPUT},
            f'the model writes {slots_dropped(MODEL)} and reads {slots_grabbed(MODEL)}')


def check_the_cross_attention_reads_and_returns_the_target_state() -> None:
    cross_attention = attention_is_all_you_need.decoder_cross_attention()
    require(axes(cross_attention) == ([['y', 'm']], [['y', 'm']]),
            f'the cross-attention reads and returns {axes(cross_attention)}')


# ==========================================================================
# The feed-forward layer and the layer normalisation.
# ==========================================================================
def check_the_feed_forward_layer_maps_m_to_f_and_back_with_biases() -> None:
    linears = [operator
               for operator in operators_of(
                   attention_is_all_you_need.feed_forward(SOURCE_POSITIONS))
               if isinstance(operator, ops.Linear)]
    require([name_of(operator) for operator in linears] == ['W{1}', 'W{2}']
            and all(operator.bias for operator in linears),
            f'the feed-forward layer holds {list(map(name_of, linears))}')
    require(size_of(attention_is_all_you_need.f) == 2048,
            f'|f| is {size_of(attention_is_all_you_need.f)}')


def check_every_sublayer_stands_inside_an_add_and_norm_block() -> None:
    blocks = [block for block in tutil.type_search(cat.Block, MODEL)
              if block.block_tag.aesthetics.title == text.ADD_NORM_TITLE]
    require(len(blocks) == 5,
            f'one encoder layer and one decoder layer hold {len(blocks)} Add & Norm '
            'blocks for their five sublayers')


def check_every_layer_normalisation_has_a_gain_and_a_bias() -> None:
    norms = [operator for operator in operators_of(MODEL)
             if isinstance(operator, ops.LayerNorm)]
    require(norms and all(operator.gain and operator.bias for operator in norms),
            f'{len(norms)} layer normalisations, not all with a gain and a bias')


# ==========================================================================
# The encoder stack and the decoder stack.
# ==========================================================================
def check_each_stack_returns_the_array_it_reads() -> None:
    for name, stack in (('encoder', attention_is_all_you_need.encoder_stack()),
                        ('decoder', attention_is_all_you_need.decoder_stack())):
        require(stack.dom() == stack.cod(), f'the {name} stack returns another array')


def check_each_stack_repeats_n_times() -> None:
    for name, stack in (('encoder', attention_is_all_you_need.encoder_stack()),
                        ('decoder', attention_is_all_you_need.decoder_stack())):
        require(stack.block_tag.repetition == attention_is_all_you_need.LAYER_COUNT,
                f'the {name} stack repeats {stack.block_tag.repetition}')


# ==========================================================================
# The whole model.
# ==========================================================================
def check_the_model_reads_two_sentences_from_one_vocabulary() -> None:
    source_tokens, target_tokens = MODEL.dom()
    require(isinstance(source_tokens.datatype, cat.Natural)
            and source_tokens.datatype == target_tokens.datatype,
            f'the model reads {source_tokens.datatype} and {target_tokens.datatype}')


def check_the_model_returns_one_distribution_per_target_position() -> None:
    require(axes(MODEL) == ([['x'], ['y']], [['y', 'v']]),
            f'the model reads and returns {axes(MODEL)}')
    output = attention_is_all_you_need.output_probabilities()
    require(isinstance(operators_of(output)[-1],
                       ops.SoftMax), 'the output block does not end in a softmax')


def check_the_source_and_the_target_have_lengths_of_their_own() -> None:
    source_tokens, target_tokens = MODEL.dom()
    require(source_tokens.shape()[0] != target_tokens.shape()[0],
            'the source and the target share one axis')


def check_the_decode_form_reads_the_input_of_the_masked_attention_through_the_mask(
) -> None:
    masks = views_named(attention_is_all_you_need.MASK_NAME, DECODE)
    require(len(masks) == 1
            and tuple(masks[0].dom()) == (attention_is_all_you_need.TARGET_STATE,),
            f'the decode form holds {len(masks)} masks')
    require(axes(DECODE) == axes(MODEL), f'the decode form reads {axes(DECODE)}')


# ==========================================================================
# The quantisations of tensor2tensor.
# ==========================================================================
def check_every_wire_of_the_quantised_model_carries_fp32_or_int32() -> None:
    morphism = QUANTISED_DECODE.morphism
    require(not quantise_model.unquantised_weaves(morphism),
            f'{len(quantise_model.unquantised_weaves(morphism))} wires carry no '
            'quantisation')
    found = quantisations_on_the_wires(morphism)
    require(found == {'FP32', 'INT32'}, f'the wires carry {sorted(found)}')


def check_every_real_wire_is_fp32_and_every_identifier_int32() -> None:
    for weave in tutil.type_search(cat.Weave, QUANTISED_DECODE.morphism):
        if Quantization.holds_real_numbers(weave.datatype):
            require(Quantization.quantisation_of(weave.datatype) == FP32,
                    f'a real wire carries {weave.datatype}')
        if Quantization.holds_natural_numbers(weave.datatype):
            require(Quantization.quantisation_of(weave.datatype) == INT32,
                    f'an index wire carries {weave.datatype}')
    source, target = QUANTISED_DECODE.morphism.dom()
    require(Quantization.quantisation_of(source.datatype) == INT32
            and Quantization.quantisation_of(target.datatype) == INT32,
            'the token identifiers are not INT32')


def check_the_quantised_model_holds_no_cast() -> None:
    conversions = quantise_model.conversions_of(QUANTISED_DECODE.morphism)
    require(not conversions,
            f'the quantised model holds {len(conversions)} conversions')


def check_every_weight_is_read_in_fp32() -> None:
    weights = dict(QUANTISED_DECODE.weight_quantisations)
    require(sorted(weights) == sorted(
                ['E', 'W^{Q}', 'W^{K}', 'W^{V}', 'W^{O}', 'W{1}', 'W{2}', 'E^{\\top}'])
            and set(weights.values()) == {FP32},
            f'the weights are {weights}')


def check_every_operator_has_a_quantisation_rule() -> None:
    for name, morphism in (('model', DECODE), ('cached pass', CACHED)):
        missing = quantise_model.leaves_without_a_rule(morphism)
        require(not missing, f'the {name} holds operators with no rule: {missing}')


# ==========================================================================
# The cached pass of the decoder.
# ==========================================================================
def check_the_pass_reads_and_returns_the_new_target_positions() -> None:
    require(axes(CACHED) == ([['ynew']], [['ynew', 'v']]),
            f'the pass reads and returns {axes(CACHED)}')


def check_no_operation_of_the_pass_carries_the_target_axis() -> None:
    require(not carries(CACHED, attention_is_all_you_need.y),
            'an operation of the pass carries the target axis y')


def check_the_mask_reads_the_cache_at_the_earlier_positions() -> None:
    '''Slot i_w of new position i_new reads position |y_old| + i_new - i_w of the
    cached axis.'''
    cached_axis = REFERENCE_PASS.cached_tokens
    masks = views_named(attention_is_all_you_need.MASK_NAME, CACHED)
    require(len(masks) == 2, f'the pass holds {len(masks)} masks')
    for mask in masks:
        strides, shift = token_row(mask, cached_axis)
        require(shift == cached_attention_is_all_you_need.EARLIER_TARGETS.local_size()
                and [stride.to_latex() for stride in strides[:2]] == ['1', '-1'],
                f'the mask reads the cache with strides '
                f'{[stride.to_latex() for stride in strides]} and shift '
                f'{shift.to_latex()}')


def check_the_positional_encoding_is_read_at_the_new_positions() -> None:
    '''The table is computed over y_old + y_new and read at |y_old| + i_new.'''
    cached_axis = REFERENCE_PASS.cached_tokens
    boxes = [operation for operation in operations_of(CACHED)
             if isinstance(operation.operator, ops.BlockOperator)]
    require(len(boxes) == 1, f'the pass holds {len(boxes)} boxes')
    body = boxes[0].operator.block
    tables = [operation for operation in operations_of(body)
              if isinstance(operation.operator, dst.Rotary)]
    require(len(tables) == 1 and tables[0].cod()[0].shape()[0] == cached_axis,
            'the table is not computed over y_old + y_new')
    reads = [operation for operation in operations_of(body)
             if isinstance(operation.operator, ops.View)]
    require(len(reads) == 1, f'the body of PE holds {len(reads)} views')
    strides, shift = token_row(reads[0], cached_axis)
    require(shift == cached_attention_is_all_you_need.EARLIER_TARGETS.local_size()
            and [stride.to_latex() for stride in strides] == ['1', '0'],
            f'the table is read with strides {[s.to_latex() for s in strides]} and '
            f'shift {shift.to_latex()}')


def check_the_decoder_has_four_placements_in_each_layer() -> None:
    '''The keys and the values, the input for the keys beside the values, the input
    for the values beside the keys, and the input once for both.'''
    placements = cached_attention_is_all_you_need.placements_over_new_targets(
        attention_is_all_you_need.decode())
    found = sorted(
        (tuple(sorted(name_of(operator)
                      for operator in placement.computed_over_the_cache)),
         tuple(sorted(cache_contents.entries_per_token(cache, SIZES)
                      for cache in placement.caches())))
        for placement in placements)
    require(found == [((), (512, 512)), (('W^{K}',), (512, 512)),
                      (('W^{K}', 'W^{V}'), (512,)), (('W^{V}',), (512, 512))],
            f'the placements are {found}')
    state_caches = [cache for placement in placements for cache in placement.caches()
                    if placement.computed_over_the_cache
                    and cache.dom()[0].shape()[-1] == attention_is_all_you_need.m]
    require(len(state_caches) == 3,
            f'{len(state_caches)} caches of the input of the sublayer')


def check_the_reference_placement_caches_the_keys_and_the_values() -> None:
    '''The caches follow W^K and W^V directly, hold every head over y_old + y_new, and
    compute no operator over the cache.'''
    require(REFERENCE_PASS.computed_over_the_cache == frozenset(),
            'the reference placement computes operators over the cache')
    require([name_of(operator) for operator in REFERENCE_PASS.operators_before_caches]
            == ['W^{K}', 'W^{V}'],
            f'the caches follow {REFERENCE_PASS.operators_before_caches}')
    cached_axis = REFERENCE_PASS.cached_tokens
    for cache in REFERENCE_PASS.caches():
        require(tuple(cache.cod()[0].shape())
                == (cached_axis, HEADS, HEAD_WIDTH),
                f'a cache holds {axes(cache)[1]}')
    require(cached_axis.parts == (cached_attention_is_all_you_need.EARLIER_TARGETS,
                                  cached_attention_is_all_you_need.NEW_TARGETS),
            'the cached axis is not y_old followed by y_new')


def check_the_caches_hold_6144_values_per_target_position() -> None:
    per_cache = [cache_contents.entries_per_token(cache, SIZES)
                 for cache in REFERENCE_PASS.caches()]
    stack, = [block for block in tutil.type_search(cat.Block, CACHED)
              if block.block_tag.repetition == attention_is_all_you_need.LAYER_COUNT]
    inside = [operation for operation in operations_of(stack)
              if isinstance(operation.operator, Caching.Caching)]
    total = sum(per_cache) * SIZES['N']
    require(per_cache == [512, 512] and len(inside) == 2 and total == 6144,
            f'the caches hold {per_cache} values per position in each of '
            f'{SIZES["N"]} layers, {total} in all')


# ==========================================================================
# The encoder and the cross-attention in the cached pass.
# ==========================================================================
def check_the_whole_model_pass_keeps_every_operation_over_the_source() -> None:
    '''Derived from the whole model, the pass holds every operation over the source
    positions as the model writes it, and the encoder still writes A.'''
    derived = cached_attention_is_all_you_need.derive_pass_over_new_targets(MODEL)
    whole = derived.expression
    target_axes = (TARGET_POSITIONS, cached_attention_is_all_you_need.NEW_TARGETS,
                   derived.cached_tokens)

    def over_the_source(term: object) -> set[cat.Broadcasted]:
        '''Every operation of `term` over the source positions and no target
        position.'''
        return {operation for operation in operations_of(term)
                if any(axis == attention_is_all_you_need.x
                       for array in (*operation.dom(), *operation.cod())
                       for axis in array.shape())
                and not any(axis in target_axes
                            for array in (*operation.dom(), *operation.cod())
                            for axis in array.shape())}

    require(over_the_source(whole) == over_the_source(MODEL),
            f'{len(over_the_source(MODEL) - over_the_source(whole))} operations over '
            'the source changed')
    require(slots_dropped(whole) == {attention_is_all_you_need.ENCODED_INPUT},
            f'the pass writes {slots_dropped(whole)}')
    require(axes(whole) == ([['x'], ['ynew']], [['ynew', 'v']]),
            f'the pass reads and returns {axes(whole)}')


def check_the_cross_attention_projects_a_in_every_step() -> None:
    '''The pass of the decoder reads A from the tape and projects it into keys and
    values over the whole source.'''
    require(slots_grabbed(CACHED) == {attention_is_all_you_need.ENCODED_INPUT},
            f'the pass reads {slots_grabbed(CACHED)}')
    grabs = [grab for grab in tutil.type_search(Para.Grab, CACHED)]
    require(len(grabs) == 1 and grabs[0].size == attention_is_all_you_need.SOURCE_STATE,
            f'the pass holds {len(grabs)} grabs')
    for name in ('W^{K}', 'W^{V}'):
        over_the_source = [projection for projection in linear_maps_named(name, CACHED)
                           if axes(projection)[0] == [['x', 'm']]]
        require(len(over_the_source) == 1,
                f'{len(over_the_source)} projections {name} of the encoded input')


# ==========================================================================
# The quantisations of the cached pass.
# ==========================================================================
def check_the_caches_of_the_quantised_pass_hold_fp32() -> None:
    caches = [operation for operation in operations_of(QUANTISED_CACHED.morphism)
              if isinstance(operation.operator, Caching.Caching)]
    require(len(caches) == 2 and all(
                Quantization.quantisation_of(array.datatype) == FP32
                for cache in caches for array in (*cache.dom(), *cache.cod())),
            f'the caches hold {[cache.cod()[0].datatype for cache in caches]}')


def check_the_quantised_pass_holds_no_cast() -> None:
    conversions = quantise_model.conversions_of(QUANTISED_CACHED.morphism)
    require(not conversions, f'the quantised pass holds {len(conversions)} conversions')
    found = quantisations_on_the_wires(QUANTISED_CACHED.morphism)
    require(found == {'FP32', 'INT32'}, f'the wires of the pass carry {sorted(found)}')


# ==========================================================================
# The unquantised variants of the page.
# ==========================================================================
def check_removing_the_quantisations_returns_the_decode_form() -> None:
    stripped = notebook_diagrams.apply_page_functor(
        notebook_diagrams.PageFunctor.DEQUANTISE, QUANTISED_DECODE.morphism)
    require(same_up_to_block_tags(stripped, DECODE),
            'the quantised model with its quantisations removed is not the model')


def check_removing_the_quantisations_returns_the_cached_form() -> None:
    stripped = notebook_diagrams.apply_page_functor(
        notebook_diagrams.PageFunctor.DEQUANTISE, QUANTISED_CACHED.morphism)
    require(same_up_to_block_tags(stripped, CACHED),
            'the quantised pass with its quantisations removed is not the pass')


def check_the_page_holds_the_four_variants() -> None:
    '''Decode and Cached, each quantised and unquantised, with each unquantised variant
    derived from its quantised variant by the functor removing every quantisation, and
    the page opening on the quantised decode form.'''
    require(tuple(variant.identifier for variant in VARIANTS) == VARIANT_IDENTIFIERS,
            f'the page holds {[variant.identifier for variant in VARIANTS]}')
    notebook_diagrams.check_page_variants(
        VARIANTS, attention_is_all_you_need_page_variants.INITIAL_VARIANT)
    derived = {variant.identifier: variant.derived_from for variant in VARIANTS
               if variant.functor is notebook_diagrams.PageFunctor.DEQUANTISE}
    require(derived == {'decode-unquantised': 'decode-quantised',
                        'cached-unquantised': 'cached-quantised'},
            f'the derived variants are {derived}')


def check_every_variant_draws_a_legend() -> None:
    by_identifier = {variant.identifier: variant for variant in VARIANTS}
    for variant in VARIANTS:
        settings = notebook_diagrams.settings_of_page_variant(
            variant, by_identifier, PAGE)
        require(settings.advanced_display in (figures.AdvancedDisplay.LEGEND,
                                              figures.AdvancedDisplay.INTERACTIVE),
                f'{variant.identifier} is drawn under {settings.advanced_display}')


def check_every_legend_row_of_every_variant_carries_a_code_name() -> None:
    '''Every axis in the legend of the four variants carries a code name, and the size
    of an axis that is one named symbol carries one too.'''
    legends = notebook_diagrams.page_variant_legends(VARIANTS, PAGE)
    axes_by_uid = {axis.uid._id: axis
                   for term in (QUANTISED_DECODE.morphism, QUANTISED_CACHED.morphism)
                   for axis in tutil.type_search(cat.Axis, term)}
    require(tuple(legends) == VARIANT_IDENTIFIERS and all(legends.values()),
            f'the legends are {[(name, len(rows)) for name, rows in legends.items()]}')
    for identifier, rows in legends.items():
        for row in rows:
            require(row['codeName'] is not None,
                    f'the axis {row["text"]} of {identifier} carries no code name')
            require(all(uid in axes_by_uid for uid in row['uids']),
                    f'the axis {row["text"]} of {identifier} is not an axis of '
                    'the terms')
            sizes = {axes_by_uid[uid].local_size() for uid in row['uids']}
            if any(isinstance(size, nm.FreeNumeric) and size.uid._name is not None
                   for size in sizes):
                require(row['sizeCodeName'] is not None,
                        f'the size of {row["text"]} of {identifier} carries no '
                        'code name')


def check_every_named_view_of_every_variant_opens_a_box() -> None:
    '''The view named Mask, and in the cached pass the read of the table of the
    positional encoding at the new positions, open an inspection box on the page.'''
    by_identifier = {variant.identifier: variant for variant in VARIANTS}
    for variant in VARIANTS:
        settings = notebook_diagrams.settings_of_page_variant(
            variant, by_identifier, PAGE)
        presented = notebook_diagrams.present_each_side(
            notebook_diagrams.term_of_page_variant(variant, by_identifier), settings)
        presented, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
        unexplained = explain_reindexings.names_of_unexplained_views(presented)
        require(not unexplained,
                f'the views {unexplained} of {variant.identifier} open no box')


def check_the_unquantised_variants_carry_the_tables_of_the_reals() -> None:
    '''Each unquantised variant is drawn under the tables of the model in the reals,
    and no role shown by an inspection box names a quantisation, so the box over a
    weight of an unquantised variant says nothing of FP32.'''
    by_identifier = {variant.identifier: variant for variant in VARIANTS}
    for variant in VARIANTS:
        settings = notebook_diagrams.settings_of_page_variant(
            variant, by_identifier, PAGE)
        roles = settings.operator_roles or {}
        mentions = [name for name, role in roles.items()
                    if any(format_name in role.role
                           for format_name in ('FP32', 'INT32', 'float32', 'int32'))]
        require(not mentions,
                f'the roles {mentions} of {variant.identifier} name a quantisation')
        if variant.functor is notebook_diagrams.PageFunctor.DEQUANTISE:
            require(settings.operator_roles
                    == attention_is_all_you_need.OPERATOR_ROLES
                    and settings.operator_explanations
                    == attention_is_all_you_need.OPERATOR_EXPLANATIONS,
                    f'{variant.identifier} is drawn under other tables')


def check_a_cache_opens_its_expansion_on_the_page() -> None:
    '''In the Cached form a cache opens a box holding its expansion, beside the layer
    normalisation, the softmax and the linear maps.'''
    by_identifier = {variant.identifier: variant for variant in VARIANTS}
    variant = by_identifier['cached-quantised']
    settings = notebook_diagrams.settings_of_page_variant(variant, by_identifier, PAGE)
    presented = notebook_diagrams.present_each_side(variant.term, settings)
    _, _, auxiliary = notebook_diagrams.package_auxiliary(presented, settings)
    expansions = auxiliary.get('expansions', {})
    expanded = {expansion['operator'] for expansion in (
        expansions.values() if isinstance(expansions, dict) else expansions)}
    require({'Caching', 'LayerNorm', 'Linear', 'SoftMax'} <= expanded,
            f'the boxes of the cached pass write out {sorted(expanded)}')


# ==========================================================================
# The differences from the hand-drawn diagram.
# ==========================================================================
def check_the_output_projection_carries_no_bias() -> None:
    projection, = linear_maps_named(
        attention_is_all_you_need.OUTPUT_PROJECTION_NAME, MODEL)
    require(not projection.operator.bias, 'E^T carries a bias')


def check_the_projections_carry_the_names_of_the_paper() -> None:
    names = {name_of(operator) for operator in operators_of(MODEL)
             if isinstance(operator, ops.Linear)}
    require(names == {'W^{Q}', 'W^{K}', 'W^{V}', 'W^{O}', 'W{1}', 'W{2}',
                      attention_is_all_you_need.OUTPUT_PROJECTION_NAME},
            f'the linear maps are named {sorted(names)}')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_sizes_of_the_base_model,
    check_the_embedding_is_scaled_and_added_to_the_positional_encoding,
    check_the_positional_encoding_reads_nothing_and_returns_x_by_m,
    check_the_positional_encoding_is_the_table_of_turns_written_as_channels,
    check_the_imaginary_unit_times_the_conjugate_puts_the_sine_first,
    check_w_o_contracts_the_heads_and_the_head_width,
    check_attention_reads_queries_keys_and_values_at_x,
    check_the_encoder_scores_carry_the_source_axis_at_two_positions,
    check_the_view_named_mask_marks_its_slot_axis,
    check_query_i_reads_slots_zero_to_i,
    check_the_model_holds_no_mask_operator,
    check_the_slid_sublayer_reads_its_input_through_one_mask,
    check_the_slide_moves_the_mask_and_changes_no_other_operator,
    check_the_encoded_input_is_the_one_slot,
    check_the_cross_attention_reads_and_returns_the_target_state,
    check_the_feed_forward_layer_maps_m_to_f_and_back_with_biases,
    check_every_sublayer_stands_inside_an_add_and_norm_block,
    check_every_layer_normalisation_has_a_gain_and_a_bias,
    check_each_stack_returns_the_array_it_reads,
    check_each_stack_repeats_n_times,
    check_the_model_reads_two_sentences_from_one_vocabulary,
    check_the_model_returns_one_distribution_per_target_position,
    check_the_source_and_the_target_have_lengths_of_their_own,
    check_the_decode_form_reads_the_input_of_the_masked_attention_through_the_mask,
    check_every_wire_of_the_quantised_model_carries_fp32_or_int32,
    check_every_real_wire_is_fp32_and_every_identifier_int32,
    check_the_quantised_model_holds_no_cast,
    check_every_weight_is_read_in_fp32,
    check_every_operator_has_a_quantisation_rule,
    check_the_pass_reads_and_returns_the_new_target_positions,
    check_no_operation_of_the_pass_carries_the_target_axis,
    check_the_mask_reads_the_cache_at_the_earlier_positions,
    check_the_positional_encoding_is_read_at_the_new_positions,
    check_the_decoder_has_four_placements_in_each_layer,
    check_the_reference_placement_caches_the_keys_and_the_values,
    check_the_caches_hold_6144_values_per_target_position,
    check_the_whole_model_pass_keeps_every_operation_over_the_source,
    check_the_cross_attention_projects_a_in_every_step,
    check_the_caches_of_the_quantised_pass_hold_fp32,
    check_the_quantised_pass_holds_no_cast,
    check_removing_the_quantisations_returns_the_decode_form,
    check_removing_the_quantisations_returns_the_cached_form,
    check_the_page_holds_the_four_variants,
    check_every_variant_draws_a_legend,
    check_every_legend_row_of_every_variant_carries_a_code_name,
    check_every_named_view_of_every_variant_opens_a_box,
    check_the_unquantised_variants_carry_the_tables_of_the_reals,
    check_a_cache_opens_its_expansion_on_the_page,
    check_the_output_projection_carries_no_bias,
    check_the_projections_carry_the_names_of_the_paper,
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
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} Attention Is All You Need checks '
          f'passed in {time.perf_counter() - started:.0f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
