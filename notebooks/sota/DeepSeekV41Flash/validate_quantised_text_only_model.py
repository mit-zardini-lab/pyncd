'''Check every claim `notebooks/sota/DeepSeekV41Flash.ipynb` makes about the quantised
text-only DeepSeek-V4.1-Flash.

Written by Claude Opus 5 (1M context), reasoning effort high.

    python notebooks/sota/DeepSeekV41Flash/validate_quantised_text_only_model.py

The notebook holds the prose and the figures, and the claims live here, one `check_`
function per claim, in the shape of
`notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash_integrated.py`. The script
prints one line per check and the time the run took, and it exits non-zero on a failure.
`check_every_claim_of_the_notebook` runs the same checks from the cell the notebook
places after its setup cell, so a reader executing the notebook sees one line confirming
each claim where the notebook asserted it before 2026-09-20.

The checks stand in the order the notebook makes the claims. The first five are the
claims of the setup cell, which are the ends of the model, the forty layers, the seven
tape slots and the counts of casts and of quantised weights. Then comes one check per
figure, stating the released size of every axis the figure is drawn at and the number of
sites the mechanism of the figure is composed at. The last thirteen are the quantisation
claims: every wire carrying a quantisation, every conversion changing one, the casts
counted by the pair of formats each reads and writes, and the model with every
quantisation taken back off it.

Six of the claims are checked by `quantization/validate_quantization.py` as well, and
the listing comparison of the stripped model calls the two functions that check calls.
Each of the six is stated again here, because this file is the record of what the
notebook claims and a reader of the notebook looks for a claim under the wording the
notebook gives it.
'''
from __future__ import annotations

import pathlib
import sys
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import agent_display as ad  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import graphs.processing.Hypergraph2Morphism as h2m  # noqa: E402
import quantization.algebra.strip_quantisations as strip_quantisations  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
from para.data_structure.ParaBlockOperator import (  # noqa: E402
    slots_dropped, slots_grabbed)

import notebooks.sota.DeepSeekV41Flash.divided_layer_stack as divided_layer_stack  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.integrated_explanations as integrated_explanations  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.quantised_text_only_model as quantised_text_only_model  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.text_only_model as text_only_model  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axes  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

UNQUANTISED: cat.Morphism = text_only_model.v41_flash_text_only
MODEL: cat.Morphism = quantised_text_only_model.v41_flash_text_only_quantised
STRIPPED: cat.Morphism = (
    quantised_text_only_model.v41_flash_text_only_without_quantisations)
HEAD: cat.Morphism = quantised_text_only_model.part_before_the_layers()
TAIL: cat.Morphism = quantised_text_only_model.part_after_the_layers()
ONE_LAYER: cat.Block = quantised_text_only_model.part_titled(text.WINDOW_LAYER_TITLE)
ASSIGNED_SIZES: dict[str, int] = quantised_text_only_model.released_assigned_sizes()

RELEASED_CAST_COUNTS: dict[tuple[str, str], int] = {
    ('BF16', 'FP32'): 73,
    ('BF16', 'MXFP8'): 25,
    ('E2M1', 'FP32'): 2,
    ('E4M3', 'FP32'): 2,
    ('FP32', 'BF16'): 79,
    ('FP32', 'E2M1'): 2,
    ('FP32', 'E4M3'): 2,
    ('INT64', 'INT32'): 5,
}

CASTS_OF_THE_MODEL = 190
QUANTISED_WEIGHTS = 33
TAPE_SLOTS = 7
LAYERS = 40


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def count_of_boxes_named(name: str) -> int:
    '''How many boxes of the quantised model carry `name`.'''
    return len(quantised_text_only_model.boxes_named(name))


def size_of(axis: str) -> int:
    return ASSIGNED_SIZES[axis]


# ==========================================================================
# The setup cell.
# ==========================================================================
def check_quantising_the_model_changes_no_axis_of_it() -> None:
    '''Quantising a model changes no axis of it.'''
    require(axes(UNQUANTISED) == axes(MODEL),
            f'the unquantised model reads and returns {axes(UNQUANTISED)} and the '
            f'quantised model {axes(MODEL)}')


def check_the_model_reads_identifiers_and_returns_probabilities() -> None:
    '''The model reads the identifier of each token of the prompt and returns one
    probability per token and vocabulary entry.'''
    require(axes(MODEL) == ([['x']], [['x', 'v']]),
            f'the model reads and returns {axes(MODEL)}')


def check_the_stack_holds_forty_layers() -> None:
    '''The text-only stack holds forty layers.'''
    count = divided_layer_stack.layer_count(text_only_model.text_only_stack)
    require(count == LAYERS, f'the stack holds {count} layers')


def check_every_tape_slot_written_in_the_model_is_read_in_it() -> None:
    '''Every slot of the tape written somewhere in the model is read somewhere in it,
    and the model writes seven of them.'''
    dropped = slots_dropped(UNQUANTISED)
    grabbed = slots_grabbed(UNQUANTISED)
    require(dropped == grabbed,
            f'the model writes {len(dropped)} slots and reads {len(grabbed)}')
    require(len(dropped) == TAPE_SLOTS, f'the model writes {len(dropped)} slots')


def check_the_casts_and_the_weights_of_the_model() -> None:
    '''The quantised model holds 190 casts, and the policy gives a quantisation to each
    of the 33 weights of the released model.'''
    casts = len(quantise_model.casts_of(MODEL))
    weights = len(quantised_text_only_model.WEIGHT_QUANTISATIONS)
    require(casts == CASTS_OF_THE_MODEL, f'the model holds {casts} casts')
    require(weights == QUANTISED_WEIGHTS, f'the policy quantises {weights} weights')


# ==========================================================================
# The embedding, the four residual streams and one layer.
# ==========================================================================
def check_the_model_reads_the_token_identifiers_and_nothing_else() -> None:
    '''The model reads the token identifiers and nothing else.'''
    require(axes(HEAD)[0] == [['x']], f'the head of the model reads {axes(HEAD)[0]}')


def check_the_head_returns_the_collapse_vector_and_the_streams() -> None:
    '''The head of the model returns the collapse vector and the four streams.'''
    require(axes(HEAD)[1] == [['x', 'n'], ['x', 'n', 'm']],
            f'the head of the model returns {axes(HEAD)[1]}')


def check_the_sizes_of_the_streams_and_the_hidden_state() -> None:
    '''The released model carries four residual streams, each of 5120 channels.'''
    require(size_of('n') == 4 and size_of('m') == 5120,
            f"n is {size_of('n')} and m is {size_of('m')}")


def check_the_coefficient_prediction_is_written_once() -> None:
    '''The coefficient prediction is written once and composed at every sublayer.'''
    require(count_of_boxes_named('Coef') == 1,
            f"the model holds {count_of_boxes_named('Coef')} coefficient boxes")


def check_a_layer_returns_the_two_arrays_it_reads() -> None:
    '''A layer reads the collapse vector and the four streams and returns the same
    two.'''
    require(axes(ONE_LAYER)[0] == axes(ONE_LAYER)[1],
            f'a layer reads {axes(ONE_LAYER)[0]} and returns {axes(ONE_LAYER)[1]}')


# ==========================================================================
# The sliding window, the rotation and the compressor.
# ==========================================================================
def check_the_nine_sites_composing_the_attention_core() -> None:
    '''Nine sites compose the attention core, one per attention mode and per half.'''
    require(count_of_boxes_named('Core') == 9,
            f"the model composes the core at {count_of_boxes_named('Core')} sites")


def check_the_sizes_of_the_window_and_the_attention() -> None:
    '''The released model attends over a window of 128 tokens with 64 query heads, at a
    shared key-value latent of 512 channels and a query low rank of 1280 channels.'''
    require(size_of('w') == 128 and size_of('h') == 64,
            f"w is {size_of('w')} and h is {size_of('h')}")
    require(size_of('c') == 512 and size_of('q') == 1280,
            f"c is {size_of('c')} and q is {size_of('q')}")


def check_the_sites_that_turn_an_array_and_turn_one_back() -> None:
    '''Nine sites turn an array, and two more turn an attention output back.'''
    require(count_of_boxes_named('Rot') == 9,
            f"the model turns an array at {count_of_boxes_named('Rot')} sites")
    require(count_of_boxes_named('Rot^{-1}') == 2,
            f"the model turns an output back at "
            f"{count_of_boxes_named('Rot^{-1}')} sites")


def check_the_turned_channels_are_read_as_complex_numbers() -> None:
    '''The 64 turned channels are read as 32 complex numbers.'''
    require(size_of('t') == 32, f"t is {size_of('t')}")


def check_the_three_sites_that_compress() -> None:
    '''Three sites compress: the encoder Full layer at two sites and the decoder at
    one.'''
    require(count_of_boxes_named('Comp') == 3,
            f"the model compresses at {count_of_boxes_named('Comp')} sites")


def check_the_encoder_folds_two_tokens_into_one_entry() -> None:
    '''The encoder folds two tokens into one entry.'''
    require(size_of('a') == 2, f"a is {size_of('a')}")


# ==========================================================================
# The quantised caches.
# ==========================================================================
def check_the_ceiling_is_the_only_operator_without_a_rule() -> None:
    '''The ceiling that rounds a scale up to a power of two is the only operator of the
    text-only model with no rule for its value.'''
    generic = sorted(integrated_explanations.TEXT_ONLY_GENERIC_OPERATOR_EXPLANATIONS)
    require(generic == ['\\lceil x \\rceil'],
            f'the text-only model holds the generic operators {generic}')


def check_a_compressed_entry_is_thirty_two_groups() -> None:
    '''A compressed entry is 32 groups of 16 channels, one FP4 scale to a group.'''
    require(size_of('E') == 32 and size_of('y') == 16,
            f"E is {size_of('E')} and y is {size_of('y')}")


# ==========================================================================
# The indexer, the candidate pool and the three attention modes.
# ==========================================================================
def check_the_encoder_and_the_decoder_each_hold_one_indexer() -> None:
    '''The encoder and the decoder each hold one indexer.'''
    require(count_of_boxes_named('Idx') == 2,
            f"the model holds {count_of_boxes_named('Idx')} indexers")


def check_the_eight_sites_that_gather_compressed_entries() -> None:
    '''Eight sites gather the compressed entries at the distances held by a
    selection.'''
    require(count_of_boxes_named('Gth') == 8,
            f"the model gathers at {count_of_boxes_named('Gth')} sites")


def check_the_sizes_of_the_indexer_and_the_selection() -> None:
    '''The released indexer runs 32 heads of 128 channels each, and a selection holds
    512 slots.'''
    require(size_of('i') == 32 and size_of('d') == 128,
            f"i is {size_of('i')} and d is {size_of('d')}")
    require(size_of('s') == 512, f"s is {size_of('s')}")


def check_the_candidate_pool_is_computed_once() -> None:
    '''The candidate pool is computed once, by the first layer of the decoder.'''
    require(count_of_boxes_named('Pool') == 1,
            f"the model holds {count_of_boxes_named('Pool')} candidate pools")


def check_the_pool_covers_eight_distances_per_kept_block() -> None:
    '''The pool covers eight distances for each of the 2048 kept blocks.'''
    require(size_of('u') == 8 and size_of('p') == 2048,
            f"u is {size_of('u')} and p is {size_of('p')}")
    require(size_of('C') == size_of('u') * size_of('p'),
            f"C is {size_of('C')} and the blocks hold "
            f"{size_of('u') * size_of('p')} distances")


def check_the_three_sites_of_the_full_mode() -> None:
    '''The encoder writes its Full layer twice, inside the repeated groups and in the
    third group written alone, and the decoder writes its own.'''
    require(count_of_boxes_named('Full') == 3,
            f"the model holds {count_of_boxes_named('Full')} Full layers")


def check_the_one_reindex_mode_and_the_four_reuse_modes() -> None:
    '''The model writes Reindex mode once and Reuse mode four times.'''
    require(count_of_boxes_named('Rex') == 1,
            f"the model holds {count_of_boxes_named('Rex')} Reindex layers")
    require(count_of_boxes_named('Reu') == 4,
            f"the model holds {count_of_boxes_named('Reu')} Reuse layers")


# ==========================================================================
# The mixture of experts, Engram, the stack and the output.
# ==========================================================================
def check_the_router_is_one_box_inside_the_mixture() -> None:
    '''The router is one box inside the mixture.'''
    require(count_of_boxes_named('Gate') == 1,
            f"the model holds {count_of_boxes_named('Gate')} routers")


def check_the_sizes_of_the_mixture_of_experts() -> None:
    '''The released mixture holds 384 routed experts, six of them chosen for a token,
    each with a hidden state of 2304 channels.'''
    require(size_of('e') == 384 and size_of('k') == 6,
            f"e is {size_of('e')} and k is {size_of('k')}")
    require(size_of('f') == 2304, f"f is {size_of('f')}")


def check_the_two_engram_modules_of_the_model() -> None:
    '''The model holds one Engram module before layer 1 and one before layer 14.'''
    require(count_of_boxes_named('Eng{1}') == 1,
            f"the model holds {count_of_boxes_named('Eng{1}')} Engram modules of "
            f"layer 1")
    require(count_of_boxes_named('Eng{14}') == 1,
            f"the model holds {count_of_boxes_named('Eng{14}')} Engram modules of "
            f"layer 14")


def check_the_ngram_orders_and_the_hash_heads() -> None:
    '''Three n-gram orders and eight hash heads give 24 ranges of rows.'''
    require(size_of('G') == 3 and size_of('K') == 8,
            f"G is {size_of('G')} and K is {size_of('K')}")


def check_the_sizes_of_the_engram_lookup() -> None:
    '''One n-gram hash of the released model reads four identifiers, and one Engram row
    holds 256 channels.'''
    require(size_of('L') == 4 and size_of('D') == 256,
            f"L is {size_of('L')} and D is {size_of('D')}")


def check_the_wires_crossing_the_layer_stack() -> None:
    '''The residual and the collapse vector are the only wires crossing the stack.'''
    crossing = axes(text_only_model.text_only_stack)[0]
    require(crossing == [['x', 'n'], ['x', 'n', 'm']], f'the stack reads {crossing}')


def check_the_model_returns_one_probability_per_entry() -> None:
    '''The model returns one probability per token and vocabulary entry.'''
    require(axes(TAIL)[1] == [['x', 'v']],
            f'the tail of the model returns {axes(TAIL)[1]}')


def check_the_size_of_the_vocabulary() -> None:
    '''The released vocabulary holds 129280 entries.'''
    require(size_of('v') == 129280, f"v is {size_of('v')}")


# ==========================================================================
# The quantisation of every wire and the 190 casts.
# ==========================================================================
def check_every_wire_of_the_model_carries_a_quantisation() -> None:
    '''Every wire of the quantised model holding a number carries a quantisation.'''
    weaves = quantise_model.unquantised_weaves(MODEL)
    require(not weaves, f'{len(weaves)} wires of the model carry no quantisation')


def check_every_conversion_reads_one_quantisation_into_another() -> None:
    '''Every conversion of the quantised model reads one quantisation into another.'''
    require(quantise_model.conversions_between_two_quantisations(MODEL),
            'a conversion of the model reads a datatype carrying no quantisation')


def check_every_conversion_changes_the_quantisation_of_its_operand() -> None:
    '''Every conversion of the quantised model changes the quantisation of its operand.
    A conversion changing a quantisation is a cast, and a figure draws a cast as no
    glyph at all, with the format written by it coloured on the wire.'''
    casts = len(quantise_model.casts_of(MODEL))
    conversions = len(quantise_model.conversions_of(MODEL))
    require(casts == conversions,
            f'{conversions} conversions of the model hold {casts} casts')


def check_the_casts_counted_by_the_formats_they_read_and_write() -> None:
    '''The quantised model holds 73 casts reading BF16 into FP32, 79 rounding FP32 into
    BF16, 25 rounding BF16 into MXFP8, two rounding into E2M1 and two into E4M3 with
    four reading a rounded value back into FP32, and five converting a position from
    INT64 to INT32. The eight counts sum to 190.'''
    counts = dict(quantised_text_only_model.cast_counts())
    require(counts == RELEASED_CAST_COUNTS, f'the model casts {counts}')
    require(sum(counts.values()) == CASTS_OF_THE_MODEL,
            f'the counts sum to {sum(counts.values())}')


def check_every_operator_class_of_the_model_has_a_rule() -> None:
    '''The registry gives a rule to every operator class of the model.'''
    without = quantise_model.leaves_without_a_rule(UNQUANTISED)
    require(not without, f'the registry gives no rule to {without}')


def check_a_block_scaled_format_differs_from_its_elements() -> None:
    '''A block-scaled format differs from the format of its elements.'''
    require(Quantization.MXFP8 != Quantization.E4M3,
            'MXFP8 equals the E4M3 of its elements')


def check_a_word_holds_two_bf16_values_and_one_fp32_value() -> None:
    '''A word holds two BF16 values and one FP32 value.'''
    require(Quantization.BF16.vector == 2 * Quantization.FP32.vector,
            f'a word holds {Quantization.BF16.vector} BF16 values and '
            f'{Quantization.FP32.vector} FP32 values')


# ==========================================================================
# The model with every quantisation taken back off it.
# ==========================================================================
def check_no_datatype_of_the_stripped_model_is_quantised() -> None:
    '''No datatype of the stripped model carries a quantisation.'''
    require(not strip_quantisations.holds_a_quantisation(STRIPPED),
            'a datatype of the stripped model carries a quantisation')


def check_the_stripped_model_holds_no_conversion() -> None:
    '''The stripped model holds no conversion at all, so it holds no cast.'''
    conversions = quantise_model.conversions_of(STRIPPED)
    require(not conversions,
            f'the stripped model holds {len(conversions)} conversions')


def check_stripping_an_unquantised_model_returns_it() -> None:
    '''Stripping a model that carries no quantisation returns the object it was
    given.'''
    require(strip_quantisations.strip_quantisations(STRIPPED) is STRIPPED,
            'stripping the stripped model returns a new object')


def check_the_stripped_model_carries_the_same_wires() -> None:
    '''The stripped model and the model given to the pass carry the same wires.'''
    stripped = len(quantise_model.unquantised_weaves(STRIPPED))
    unquantised = len(quantise_model.unquantised_weaves(UNQUANTISED))
    require(stripped == unquantised,
            f'the stripped model carries {stripped} wires and the model given to the '
            f'pass carries {unquantised}')


def check_the_stripped_model_reads_as_the_unquantised_model() -> None:
    '''The stripped model and `text_only_model.v41_flash_text_only` have the same
    `agent_display` listing once both are recycled, so the difference between the
    figure of the quantised model and the figure of the stripped model is the 190 casts
    and the format on each label. The check of the same name in
    `quantization/validate_quantization.py` compares the same two listings through the
    same two functions.'''
    require(ad.listing(h2m.recycle(STRIPPED)) == ad.listing(h2m.recycle(UNQUANTISED)),
            'the stripped model and the unquantised model have different listings')


def check_the_conversions_before_and_after_stripping() -> None:
    '''The quantised model holds 190 conversions and the stripped model holds none.'''
    before = len(quantise_model.conversions_of(MODEL))
    after = len(quantise_model.conversions_of(STRIPPED))
    require(before == CASTS_OF_THE_MODEL,
            f'the quantised model holds {before} conversions')
    require(after == 0, f'the stripped model holds {after} conversions')


CHECKS: tuple[Callable[[], None], ...] = (
    check_quantising_the_model_changes_no_axis_of_it,
    check_the_model_reads_identifiers_and_returns_probabilities,
    check_the_stack_holds_forty_layers,
    check_every_tape_slot_written_in_the_model_is_read_in_it,
    check_the_casts_and_the_weights_of_the_model,
    check_the_model_reads_the_token_identifiers_and_nothing_else,
    check_the_head_returns_the_collapse_vector_and_the_streams,
    check_the_sizes_of_the_streams_and_the_hidden_state,
    check_the_coefficient_prediction_is_written_once,
    check_a_layer_returns_the_two_arrays_it_reads,
    check_the_nine_sites_composing_the_attention_core,
    check_the_sizes_of_the_window_and_the_attention,
    check_the_sites_that_turn_an_array_and_turn_one_back,
    check_the_turned_channels_are_read_as_complex_numbers,
    check_the_three_sites_that_compress,
    check_the_encoder_folds_two_tokens_into_one_entry,
    check_the_ceiling_is_the_only_operator_without_a_rule,
    check_a_compressed_entry_is_thirty_two_groups,
    check_the_encoder_and_the_decoder_each_hold_one_indexer,
    check_the_eight_sites_that_gather_compressed_entries,
    check_the_sizes_of_the_indexer_and_the_selection,
    check_the_candidate_pool_is_computed_once,
    check_the_pool_covers_eight_distances_per_kept_block,
    check_the_three_sites_of_the_full_mode,
    check_the_one_reindex_mode_and_the_four_reuse_modes,
    check_the_router_is_one_box_inside_the_mixture,
    check_the_sizes_of_the_mixture_of_experts,
    check_the_two_engram_modules_of_the_model,
    check_the_ngram_orders_and_the_hash_heads,
    check_the_sizes_of_the_engram_lookup,
    check_the_wires_crossing_the_layer_stack,
    check_the_model_returns_one_probability_per_entry,
    check_the_size_of_the_vocabulary,
    check_every_wire_of_the_model_carries_a_quantisation,
    check_every_conversion_reads_one_quantisation_into_another,
    check_every_conversion_changes_the_quantisation_of_its_operand,
    check_the_casts_counted_by_the_formats_they_read_and_write,
    check_every_operator_class_of_the_model_has_a_rule,
    check_a_block_scaled_format_differs_from_its_elements,
    check_a_word_holds_two_bf16_values_and_one_fp32_value,
    check_no_datatype_of_the_stripped_model_is_quantised,
    check_the_stripped_model_holds_no_conversion,
    check_stripping_an_unquantised_model_returns_it,
    check_the_stripped_model_carries_the_same_wires,
    check_the_stripped_model_reads_as_the_unquantised_model,
    check_the_conversions_before_and_after_stripping,
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
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} quantised text-only model '
          f'checks passed in {time.perf_counter() - started:.0f} s')
    return failures


def check_every_claim_of_the_notebook() -> None:
    '''Every check, run from `notebooks/sota/DeepSeekV41Flash.ipynb`, raising when one
    of them fails so that the notebook fails with it.'''
    failures = report_each_check()
    if failures:
        raise ClaimDoesNotHold(
            f'{failures} of {len(CHECKS)} claims of the notebook do not hold')


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
