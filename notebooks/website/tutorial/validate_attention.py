# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Check every claim `notebooks/website/tutorial/Attention.ipynb` makes.

    python notebooks/website/tutorial/validate_attention.py

The notebook holds the prose and the figures, and the claims live here, one `check_`
function per claim, in the order the notebook makes them: the shapes and the operations
of scaled dot-product attention, the softmax written out with its division moved past
the weighted sum, the training step, and the page. The script prints one line per check
and exits non-zero on a failure. The comparisons with `torch.autograd` are made by
`check_gradients_in_torch.py`.
'''
from __future__ import annotations

import math
import pathlib
import sys
from collections.abc import Callable, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import para.algebra.tangent as tangent  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402

import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.website.tutorial.check_gradients_in_torch as torch_check  # noqa
import notebooks.website.tutorial.derive_training_step as derive_training_step  # noqa
import notebooks.website.tutorial.express_attention as express_attention  # noqa
import notebooks.website.tutorial.tutorial_pages as tutorial_pages  # noqa: E402
from notebooks.website.tutorial.tutorial_checks import (  # noqa: E402
    block_titles, check_every_legend_row_carries_code_names, degree_names, drops_of,
    ends, grabbed_slot, grabs_of, is_contraction_reading, names, nodes_of, producer_of,
    slots_by_shape, slots_the_exponentials_are_rebuilt_from,
    report_each_check, require, roots_of, saved_shapes, slot_name, targets_read)
from notebooks.website.tutorial.tutorial_wording import TEXT as text  # noqa

ATTENTION = express_attention.scaled_dot_product_attention()
TRAINING_STEP = derive_training_step.derive_training_step(ATTENTION)
PAGE = notebook_diagrams.DiagramSettings(
    mode=notebook_diagrams.DiagramMode.HTML,
    advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
    operator_explanations=express_attention.OPERATOR_EXPLANATIONS,
    operator_references=express_attention.OPERATOR_REFERENCES,
    operator_roles=express_attention.SINGLE_HEAD_ROLES,
    reindexing_explanations=express_attention.REINDEXING_EXPLANATIONS)
SIZES = {'q': 3, 'x': 5, 'd': 4, 'v': 6}
SEED = 0


def attention_in_torch(inputs: Sequence[torch.Tensor],
                       weights: torch_check.Tensors) -> torch.Tensor:
    queries, keys, values = inputs
    scores = queries @ keys.T / math.sqrt(queries.shape[-1])
    return torch.softmax(scores, dim=-1) @ values


def seeds_of(term: object) -> list[cat.Broadcasted]:
    return nodes_of(cat.Operator, term)


def check_attention_reads_queries_keys_and_values() -> None:
    require(ends(ATTENTION) == ([['q', 'd'], ['x', 'd'], ['x', 'v']], [['q', 'v']]),
            f'the attention reads and returns {ends(ATTENTION)}')


def check_the_operations_are_the_dot_product_the_scale_the_softmax_and_the_sum() -> None:
    kinds = [type(node.operator) for node in seeds_of(ATTENTION)]
    require(kinds == [ops.Einops, ops.Arithmetic, ops.SoftMax, ops.Einops],
            f'the operations are {[kind.__name__ for kind in kinds]}')


def check_the_scale_multiplies_by_the_inverse_root_of_the_key_width() -> None:
    scale, = nodes_of(ops.Arithmetic, ATTENTION)
    expected = nm.x * express_attention.key_width.local_size() ** express_attention.MINUS_HALF
    require(scale.operator.formula == expected,
            f'the scale computes {scale.operator.formula.to_latex()}')


def check_the_dot_product_sums_over_d_and_is_broadcast_over_q_and_x() -> None:
    dot_product = seeds_of(ATTENTION)[0]
    require(targets_read(dot_product) == [['d'], ['d']]
            and degree_names(dot_product) == ['q', 'x'],
            f'the dot product reads {targets_read(dot_product)} over '
            f'{degree_names(dot_product)}')


def check_the_softmax_normalises_the_scores_of_each_query() -> None:
    softmax, = nodes_of(ops.SoftMax, ATTENTION)
    require(targets_read(softmax) == [['x']] and degree_names(softmax) == ['q'],
            f'the softmax reads {targets_read(softmax)} over {degree_names(softmax)}')


def check_the_weighted_sum_sums_over_the_keys() -> None:
    weighted_sum = seeds_of(ATTENTION)[-1]
    require(targets_read(weighted_sum) == [['x'], ['x']]
            and degree_names(weighted_sum) == ['q', 'v'],
            f'the weighted sum reads {targets_read(weighted_sum)} over '
            f'{degree_names(weighted_sum)}')


def check_one_block_groups_the_operations_under_its_title() -> None:
    require(block_titles(ATTENTION) == [text.CORE_TITLE]
            and isinstance(ATTENTION, cat.Block)
            and ATTENTION.block_tag.aesthetics.fill_color == express_attention.CORE_COLOUR,
            f'the blocks are titled {block_titles(ATTENTION)}')


def check_the_softmax_is_written_out() -> None:
    prepared = TRAINING_STEP.prepared
    formulas = {node.operator.formula for node in nodes_of(ops.Arithmetic, prepared)}
    require(not nodes_of(ops.SoftMax, prepared)
            and {nm.E ** nm.x, nm.x ** nm.Integer(-1)} <= formulas
            and ends(prepared) == ends(ATTENTION),
            f'the rewritten attention computes {sorted(f.to_latex() for f in formulas)}')


def check_the_reciprocal_scales_the_result_of_each_query() -> None:
    prepared = TRAINING_STEP.prepared
    roots = roots_of(prepared)
    reciprocal, = (root for root in roots
                   if isinstance(root.wraps, cat.Broadcasted)
                   and isinstance(root.wraps.operator, ops.Arithmetic)
                   and root.wraps.operator.formula == nm.x ** nm.Integer(-1))
    readers = [root for root in roots if reciprocal.cod[0] in root.dom]
    require(len(readers) == 1 and isinstance(readers[0].wraps.operator, ops.Einops)
            and [names(tuple(array.shape())) for array in readers[0].wraps.dom()]
            == [['q'], ['q', 'v']],
            'the reciprocal of the row sum scales an array other than the result of '
            'each query')


def check_the_rewritten_attention_computes_the_same_result() -> None:
    generator = torch.Generator().manual_seed(SEED)
    written = torch_check.compile_pass(ATTENTION, SIZES)
    rewritten = torch_check.compile_pass(TRAINING_STEP.prepared, SIZES)
    operands = [torch.randn(*shape, generator=generator, dtype=torch_check.DTYPE)
                for shape in written.own_operand_shapes()]
    require(torch.allclose(torch_check.run(written.module, operands)[0],
                           torch_check.run(rewritten.module, operands)[0],
                           atol=torch_check.TOLERANCE),
            'the rewritten attention returns another result')


def check_the_backward_pass_reverses_the_ends_of_the_attention() -> None:
    collapsed = TRAINING_STEP.recomputed
    require(ends(collapsed.forward) == ends(ATTENTION)
            and tuple(collapsed.backward.dom()) == tuple(tangent.obj(ATTENTION.cod()))
            and tuple(collapsed.backward.cod()) == tuple(tangent.obj(ATTENTION.dom())),
            'the passes do not read and return the arrays of the attention')


def check_the_backward_pass_grabs_the_slots_the_forward_pass_drops() -> None:
    collapsed = TRAINING_STEP.recomputed
    dropped = {slot_name(drop.tape) for drop in drops_of(collapsed.forward)}
    grabbed = {slot_name(grab.tape) for grab in grabs_of(collapsed.backward)}
    require(dropped == grabbed, f'the forward pass drops {sorted(dropped)} and the '
                                f'backward pass grabs {sorted(grabbed)}')


def check_the_collapsed_tape_holds_the_exponentials_of_the_scores() -> None:
    saved = saved_shapes(TRAINING_STEP.collapsed)
    require(['q', 'x'] in saved, f'the collapsed forward pass saves {saved}')


def check_the_backward_pass_rebuilds_the_exponentials_from_the_queries_and_the_keys(
) -> None:
    slots = slots_by_shape(TRAINING_STEP.recomputed)
    rebuilt = slots_the_exponentials_are_rebuilt_from(TRAINING_STEP.recomputed)
    require(rebuilt == [slots[('q', 'd')] + slots[('x', 'd')]],
            f'the backward pass rebuilds exponentials from the slots {rebuilt}, and the '
            f'forward pass saves {slots}')


def check_the_forward_pass_saves_five_arrays() -> None:
    saved = saved_shapes(TRAINING_STEP.recomputed)
    require(saved == sorted([['q', 'd'], ['x', 'd'], ['x', 'v'], ['q'], ['q', 'v']]),
            f'the forward pass saves {saved}')


def check_the_backward_pass_computes_the_row_statistic_from_the_output() -> None:
    collapsed = TRAINING_STEP.recomputed
    output_slot, = (slot_name(drop.tape) for drop in drops_of(collapsed.forward)
                    if names(tuple(drop.size.shape())) == ['q', 'v'])
    roots = roots_of(collapsed.backward)
    statistics = [
        root for root in roots if is_contraction_reading(root, [['v'], ['v']])
        and len(root.dom) == 2
        and {(grabbed_slot(wire, roots), producer_of(wire, roots) is None)
             for wire in root.dom} == {(None, True), (output_slot, False)}]
    require(len(statistics) == 1,
            f'{len(statistics)} contractions sum the gradient of the output against '
            'the saved output')


def check_the_backward_pass_scales_the_rebuilt_scores_and_their_gradient() -> None:
    forward_scale, = nodes_of(ops.Arithmetic, ATTENTION)
    backward_roots = roots_of(TRAINING_STEP.recomputed.backward)
    scales = [root for root in backward_roots
              if isinstance(root.wraps, cat.Broadcasted)
              and root.wraps.operator == forward_scale.operator]
    rebuilt_scores = [
        root for root in scales
        if is_contraction_reading(producer_of(root.dom[0], backward_roots),
                                  [['d'], ['d']])]
    require(len(scales) == 2 and len(rebuilt_scores) == 1,
            f'the backward pass holds {len(scales)} scales, of which '
            f'{len(rebuilt_scores)} scale a dot product over d')


def check_the_training_step_keeps_the_block() -> None:
    collapsed = TRAINING_STEP.recomputed
    require(block_titles(collapsed.forward) == [text.CORE_TITLE]
            and block_titles(collapsed.backward) == [f'R[{text.CORE_TITLE}]'],
            f'the passes hold the blocks {block_titles(collapsed.forward)} and '
            f'{block_titles(collapsed.backward)}')


def check_the_gradients_agree_with_autograd() -> None:
    comparison = torch_check.compare_with_autograd(
        TRAINING_STEP.recomputed, attention_in_torch, SIZES,
        torch.Generator().manual_seed(SEED))
    require(comparison.all_agree(), f'the comparison with autograd reads {comparison}')


def page_variants() -> list[notebook_diagrams.PageVariant]:
    return tutorial_pages.attention_page(ATTENTION, TRAINING_STEP, PAGE)


def check_the_page_switches_between_the_forward_pass_and_the_training_step() -> None:
    variants = page_variants()
    notebook_diagrams.check_page_variants(variants, tutorial_pages.FORWARD)
    require([variant.identifier for variant in variants]
            == [tutorial_pages.FORWARD, tutorial_pages.TRAINING]
            and variants[0].term is ATTENTION,
            f'the page holds {[variant.identifier for variant in variants]}')


def check_every_legend_row_of_the_page_names_its_axis_in_code() -> None:
    check_every_legend_row_carries_code_names(page_variants(), PAGE)


CHECKS: tuple[Callable[[], None], ...] = (
    check_attention_reads_queries_keys_and_values,
    check_the_operations_are_the_dot_product_the_scale_the_softmax_and_the_sum,
    check_the_scale_multiplies_by_the_inverse_root_of_the_key_width,
    check_the_dot_product_sums_over_d_and_is_broadcast_over_q_and_x,
    check_the_softmax_normalises_the_scores_of_each_query,
    check_the_weighted_sum_sums_over_the_keys,
    check_one_block_groups_the_operations_under_its_title,
    check_the_softmax_is_written_out,
    check_the_reciprocal_scales_the_result_of_each_query,
    check_the_rewritten_attention_computes_the_same_result,
    check_the_backward_pass_reverses_the_ends_of_the_attention,
    check_the_backward_pass_grabs_the_slots_the_forward_pass_drops,
    check_the_collapsed_tape_holds_the_exponentials_of_the_scores,
    check_the_backward_pass_rebuilds_the_exponentials_from_the_queries_and_the_keys,
    check_the_forward_pass_saves_five_arrays,
    check_the_backward_pass_computes_the_row_statistic_from_the_output,
    check_the_backward_pass_scales_the_rebuilt_scores_and_their_gradient,
    check_the_training_step_keeps_the_block,
    check_the_gradients_agree_with_autograd,
    check_the_page_switches_between_the_forward_pass_and_the_training_step,
    check_every_legend_row_of_the_page_names_its_axis_in_code,
)


def main() -> int:
    return 1 if report_each_check(CHECKS, 'Expressing Attention') else 0


if __name__ == '__main__':
    sys.exit(main())
