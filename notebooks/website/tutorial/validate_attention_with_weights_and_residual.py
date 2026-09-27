# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Check every claim `AttentionWithWeightsAndResidual.ipynb` makes.

    python notebooks/website/tutorial/validate_attention_with_weights_and_residual.py

The notebook holds the prose and the figures, and the claims live here, one `check_`
function per claim, in the order the notebook makes them: the projections and the
residual connection, the causal mask and the CausalSlide, the weights read from the
tape, the training step, and the page. The script prints one line per check and exits
non-zero on a failure. The comparisons in PyTorch are made by
`check_gradients_in_torch.py`.
'''
from __future__ import annotations

import math
import pathlib
import sys
from collections.abc import Callable, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa
import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards  # noqa
import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import para.algebra.tangent as tangent  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import para.data_structure.transpose as transpose  # noqa: E402

import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.website.tutorial.check_gradients_in_torch as torch_check  # noqa
import notebooks.website.tutorial.derive_training_step as derive_training_step  # noqa
import notebooks.website.tutorial.express_attention as express_attention  # noqa
import notebooks.website.tutorial.tutorial_pages as tutorial_pages  # noqa: E402
from notebooks.website.tutorial.tutorial_checks import (  # noqa: E402
    block_titles, check_every_legend_row_carries_code_names, drops_of, ends, grabs_of,
    linear_named, names, nodes_of, producer_of, report_each_check, require, roots_of,
    saved_shapes, slot_name, slot_read_through_views,
    slots_the_exponentials_are_rebuilt_from, targets_read)
from notebooks.website.tutorial.tutorial_wording import TEXT as text  # noqa

MODEL = express_attention.attention_with_weights_and_residual()
DISPLAYED = slide_causal_reads_backwards.slide_causal_reads_backwards(MODEL)
TRAINING_STEP = derive_training_step.derive_training_step(MODEL)
PAGE = notebook_diagrams.DiagramSettings(
    mode=notebook_diagrams.DiagramMode.HTML,
    advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
    operator_explanations=express_attention.OPERATOR_EXPLANATIONS,
    operator_references=express_attention.OPERATOR_REFERENCES,
    operator_roles=express_attention.SINGLE_HEAD_ROLES,
    reindexing_explanations=express_attention.REINDEXING_EXPLANATIONS)
WEIGHTS = ('W^{Q}', 'W^{K}', 'W^{V}', 'W^{O}')
SIZES = {'x': 5, 'm': 6, 'd': 4}
SEED = 0


def residual_attention_in_torch(inputs: Sequence[torch.Tensor],
                                weights: torch_check.Tensors) -> torch.Tensor:
    '''Causal self-attention with a residual connection, written directly in PyTorch.
    A weight holds the input channels first, as `grab_parameters` writes it.'''
    state, = inputs
    queries = state @ weights['W^{Q}']
    keys = state @ weights['W^{K}']
    values = state @ weights['W^{V}']
    scores = queries @ keys.T / math.sqrt(queries.shape[-1])
    causal = torch.ones(scores.shape, dtype=torch.bool).tril()
    weights_of_softmax = torch.softmax(scores.masked_fill(~causal, -math.inf), dim=-1)
    return state + (weights_of_softmax @ values) @ weights['W^{O}']


def masks_of(term: object) -> list[cat.Broadcasted]:
    return [node for node in nodes_of(ops.View, term)
            if node.operator.name is not None
            and node.operator.name.to_bodies() == express_attention.MASK_VIEW_NAME]


def mask_slots() -> AffineGuards.AffineSparseAxis:
    mask, = masks_of(DISPLAYED)
    return tuple(mask.cod()[0].shape())[1]


def check_the_model_reads_and_returns_the_state() -> None:
    require(ends(MODEL) == ([['x', 'm']], [['x', 'm']]) and ends(DISPLAYED) == ends(MODEL),
            f'the model reads and returns {ends(MODEL)}')


def check_three_projections_read_the_state_and_w_o_projects_back() -> None:
    projections = {name: (targets_read(linear_named(name, MODEL)),
                          names(tuple(linear_named(name, MODEL).output_weaves[0]
                                      .target().shape())))
                   for name in WEIGHTS}
    require(projections == {'W^{Q}': ([['m']], ['d']), 'W^{K}': ([['m']], ['d']),
                            'W^{V}': ([['m']], ['d']), 'W^{O}': ([['d']], ['m'])},
            f'the projections read and write {projections}')


def check_the_residual_connection_adds_the_attention_to_the_state() -> None:
    addition, = nodes_of(ops.AdditionOp, MODEL)
    require(isinstance(MODEL, cat.Block)
            and MODEL.block_tag.aesthetics.title == text.RESIDUAL_TITLE
            and names_of(addition.dom()) == [['x', 'm'], ['x', 'm']],
            'the residual connection does not add two arrays over x and m')


def names_of(arrays: Sequence[cat.Array]) -> list[list[str]]:
    return [names(tuple(array.shape())) for array in arrays]


def check_the_mask_holds_token_i_x_minus_i_w_at_slot_i_w() -> None:
    mask, = masks_of(DISPLAYED)
    read = move_reads_backwards.as_stride_morphism(mask.reindexings[0])
    rows = [(axis.uid._name.to_bodies(), tuple(strides), shift)
            for axis, strides, shift in read._cod_stride_shift]
    require(rows[0] == ('x', (nm.Integer(1), nm.Integer(-1), nm.Integer(0)),
                        nm.Integer(0)),
            f'the mask reads {rows}')


def check_the_slot_axis_has_as_many_positions_as_there_are_tokens() -> None:
    require(mask_slots().local_size() == express_attention.tokens.local_size(),
            'the slot axis is not as long as the sequence')


def check_the_mask_marks_the_slots_holding_a_token() -> None:
    slots = mask_slots()
    live = [mark_sparse_domains.live_positions(
                slots, (token,), {express_attention.tokens.local_size(): 4})
            for token in range(4)]
    require(isinstance(slots, AffineGuards.AffineSparseAxis)
            and names([slots]) == ['w|x']
            and live == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]],
            f'in a sequence of four tokens the live slots are {live}')


def check_no_operator_masks_the_scores() -> None:
    kinds = {type(node.operator) for node in nodes_of(cat.Operator, MODEL)}
    require(kinds == {ops.Linear, ops.View, ops.Einops, ops.Arithmetic, ops.SoftMax,
                      ops.AdditionOp},
            f'the model holds the operators {sorted(kind.__name__ for kind in kinds)}')


def check_the_model_is_built_with_the_mask_after_the_projections() -> None:
    roots = roots_of(MODEL)
    read = sorted(
        (linear.wraps.operator.name.to_bodies(), names_of(root.wraps.dom()))
        for root in roots if root.wraps in masks_of(MODEL)
        for linear in (producer_of(root.dom[0], roots),))
    require(read == [('W^{K}', [['x', 'd']]), ('W^{V}', [['x', 'd']])],
            f'the masks of the model as built read {read}')


def check_the_causal_slide_reads_the_state_once() -> None:
    masks = masks_of(DISPLAYED)
    runs = [names_of(linear_named(name, DISPLAYED).dom()) for name in ('W^{K}', 'W^{V}')]
    require(len(masks) == 1 and names_of(masks[0].dom()) == [['x', 'm']]
            and runs == [[['x', 'w|x', 'm']], [['x', 'w|x', 'm']]],
            f'the CausalSlide holds {len(masks)} masks, and W^K and W^V read {runs}')


def check_the_two_forms_compute_the_same_result() -> None:
    agreements = [torch_check.forward_agrees_with(
                      form, residual_attention_in_torch, SIZES,
                      torch.Generator().manual_seed(SEED))
                  for form in (MODEL, DISPLAYED)]
    require(agreements == [True, True],
            f'the model as built and its CausalSlide agree with causal attention as '
            f'{agreements}')


def check_every_matrix_is_grabbed_from_a_slot_named_after_it() -> None:
    parametrised = TRAINING_STEP.parametrised
    grabs = {slot_name(grab.tape): names(tuple(grab.size.shape()))
             for grab in grabs_of(parametrised)}
    require(grabs == {'W^{Q}': ['m', 'd'], 'W^{K}': ['m', 'd'], 'W^{V}': ['m', 'd'],
                      'W^{O}': ['d', 'm']}
            and all(isinstance(grab.tape, Para.OuterTapeSlot)
                    for grab in grabs_of(parametrised))
            and ends(parametrised) == ends(MODEL),
            f'the model grabs {grabs}')


def check_the_passes_read_and_return_the_arrays_of_the_model() -> None:
    collapsed = TRAINING_STEP.recomputed
    require(ends(collapsed.forward) == ends(MODEL)
            and tuple(collapsed.backward.dom()) == tuple(tangent.obj(MODEL.cod()))
            and tuple(collapsed.backward.cod()) == tuple(tangent.obj(MODEL.dom())),
            'the passes do not read and return the arrays of the model')


def check_the_backward_pass_writes_a_gradient_for_every_matrix() -> None:
    collapsed = TRAINING_STEP.recomputed
    written = {slot_name(drop.tape) for drop in drops_of(collapsed.backward)}
    grabbed = {slot_name(grab.tape) for grab in grabs_of(collapsed.backward)}
    require(written == {torch_check.gradient_slot_name(name) for name in WEIGHTS}
            and set(WEIGHTS) <= grabbed,
            f'the backward pass writes {sorted(written)} and grabs {sorted(grabbed)}')


def check_the_training_step_is_derived_from_the_model_as_built() -> None:
    runs = [names_of(linear_named(name, TRAINING_STEP.recomputed.forward).dom())
            for name in ('W^{K}', 'W^{V}')]
    require(runs == [[['m', 'd'], ['x', 'm']]] * 2,
            f'W^K and W^V read {runs} in the forward pass of the training step')


def check_a_training_step_of_the_causal_slide_saves_keys_and_values_per_slot() -> None:
    saved = saved_shapes(derive_training_step.derive_training_step(DISPLAYED).recomputed)
    require(saved.count(['x', 'w|x', 'd']) == 2,
            f'the training step of the CausalSlide saves {saved}')


def check_the_softmax_is_written_out_in_the_training_step() -> None:
    collapsed = TRAINING_STEP.recomputed
    require(not nodes_of(ops.SoftMax, collapsed.forward)
            and not nodes_of(ops.SoftMax, collapsed.backward),
            'a softmax stands in the training step')


def check_the_backward_pass_rebuilds_the_exponentials_from_the_queries_and_the_keys(
) -> None:
    forward_roots = roots_of(TRAINING_STEP.recomputed.forward)
    slot_of_projection = {
        name: slot_name(drop.wraps.tape) for drop in forward_roots
        if isinstance(drop.wraps, Para.Drop)
        for name in (linear_name_producing(drop.dom[0], forward_roots),)
        if name is not None}
    rebuilt = slots_the_exponentials_are_rebuilt_from(TRAINING_STEP.recomputed)
    require(rebuilt == [[slot_of_projection['W^{Q}'], slot_of_projection['W^{K}']]],
            f'the backward pass rebuilds exponentials from the slots {rebuilt}, and the '
            f'projections are saved to {slot_of_projection}')


def linear_name_producing(wire: object, roots: tuple) -> str | None:
    producer = producer_of(wire, roots)
    if producer is None or operation_of(producer) is not ops.Linear:
        return None
    return producer.wraps.operator.name.to_bodies()


def check_the_forward_pass_saves_six_arrays() -> None:
    saved = saved_shapes(TRAINING_STEP.recomputed)
    require(saved == sorted([['x', 'm'], ['x', 'd'], ['x', 'd'], ['x', 'd'], ['x'],
                             ['x', 'd']]),
            f'the forward pass saves {saved}')


def check_the_backward_pass_reads_the_keys_and_the_values_through_the_mask() -> None:
    roots = backward_roots()
    masked_reads = [root for root in roots if root.wraps in masks_of(
        TRAINING_STEP.recomputed.backward)]
    read_slots = sorted(slot_read_through_views(root.dom[0], roots)
                        for root in masked_reads)
    forward_roots = roots_of(TRAINING_STEP.recomputed.forward)
    projected = sorted(slot_name(drop.wraps.tape) for drop in forward_roots
                       if isinstance(drop.wraps, Para.Drop)
                       and linear_name_producing(drop.dom[0], forward_roots)
                       in ('W^{K}', 'W^{V}'))
    require(read_slots == projected and len(projected) == 2,
            f'the backward pass reads {read_slots} through the mask, and the keys and '
            f'the values are saved to {projected}')


def backward_roots() -> tuple:
    return roots_of(TRAINING_STEP.recomputed.backward)


def operation_of(root: object) -> type | None:
    wraps = getattr(root, 'wraps', None)
    return type(wraps.operator) if isinstance(wraps, cat.Broadcasted) else None


def check_the_gradient_of_the_output_reaches_the_state_directly() -> None:
    roots = backward_roots()
    last_additions = [
        root for root in roots if operation_of(root) is ops.AdditionOp
        and any(producer_of(wire, roots) is None for wire in root.dom)
        and not any(root.cod[0] in other.dom for other in roots)]
    require(len(last_additions) == 1,
            f'{len(last_additions)} additions add the incoming gradient to the result')


def check_the_mask_reverses_into_its_transpose_for_the_keys_and_the_values() -> None:
    transposes = [root.wraps for root in backward_roots()
                  if operation_of(root) is transpose.ReindexTranspose]
    require(len(transposes) == 2
            and all(names_of(node.dom()) == [['x', 'w|x', 'd']]
                    and names_of(node.cod()) == [['x', 'd']]
                    and node.operator.name.to_bodies() == express_attention.MASK_VIEW_NAME
                    for node in transposes),
            f'the backward pass holds {len(transposes)} transposes of the mask')


def check_the_gradients_through_the_three_projections_are_added_once() -> None:
    roots = backward_roots()
    joins = [root for root in roots if operation_of(root) is ops.AdditionOp
             and len(root.dom) == 3
             and {operation_of(producer_of(wire, roots)) for wire in root.dom}
             == {transpose.Transpose}]
    require(len(joins) == 1 and names_of(joins[0].wraps.dom()) == [['x', 'm']] * 3,
            f'{len(joins)} additions join the gradients through W^Q, W^K and W^V')


def check_the_weight_gradients_sum_over_the_tokens() -> None:
    roots = backward_roots()
    consumed = {}
    reads_the_state = {}
    for root in roots:
        if isinstance(root.wraps, Para.Drop):
            producer = producer_of(root.dom[0], roots)
            consumed[slot_name(root.wraps.tape)] = targets_read(producer.wraps)
            reads_the_state[slot_name(root.wraps.tape)] = any(
                grabbed_shape == ['x', 'm']
                for grabbed_shape in (
                    names(tuple(producer_of(wire, roots).wraps.size.shape()))
                    for wire in producer.dom
                    if isinstance(getattr(producer_of(wire, roots), 'wraps', None),
                                  Para.Grab)))
    by_tokens = [['x'], ['x']]
    require(consumed == {'dW^{K}': by_tokens, 'dW^{V}': by_tokens, 'dW^{Q}': by_tokens,
                         'dW^{O}': by_tokens}
            and reads_the_state == {'dW^{K}': True, 'dW^{V}': True, 'dW^{Q}': True,
                                    'dW^{O}': False},
            f'the weight gradients sum over {consumed} and read the saved state as '
            f'{reads_the_state}')


def check_the_training_step_keeps_every_block() -> None:
    collapsed = TRAINING_STEP.recomputed
    written = block_titles(MODEL)
    require(len(written) == 3 and block_titles(collapsed.forward) == written
            and block_titles(collapsed.backward) == [f'R[{title}]' for title in written],
            f'the passes hold the blocks {block_titles(collapsed.forward)} and '
            f'{block_titles(collapsed.backward)}')


def check_the_gradients_agree_with_autograd() -> None:
    comparison = torch_check.compare_with_autograd(
        TRAINING_STEP.recomputed, residual_attention_in_torch, SIZES,
        torch.Generator().manual_seed(SEED))
    require(comparison.all_agree() and set(comparison.weight_gradients_agree) == set(WEIGHTS),
            f'the comparison with autograd reads {comparison}')


def page_variants() -> list[notebook_diagrams.PageVariant]:
    return tutorial_pages.attention_with_weights_and_residual_page(
        DISPLAYED, TRAINING_STEP, PAGE)


def check_the_page_switches_between_the_forward_pass_and_the_training_step() -> None:
    variants = page_variants()
    notebook_diagrams.check_page_variants(variants, tutorial_pages.FORWARD)
    require([variant.identifier for variant in variants]
            == [tutorial_pages.FORWARD, tutorial_pages.TRAINING]
            and variants[0].term is DISPLAYED,
            f'the page holds {[variant.identifier for variant in variants]}')


def check_every_legend_row_of_the_page_names_its_axis_in_code() -> None:
    check_every_legend_row_carries_code_names(page_variants(), PAGE)


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_model_reads_and_returns_the_state,
    check_three_projections_read_the_state_and_w_o_projects_back,
    check_the_residual_connection_adds_the_attention_to_the_state,
    check_the_mask_holds_token_i_x_minus_i_w_at_slot_i_w,
    check_the_slot_axis_has_as_many_positions_as_there_are_tokens,
    check_the_mask_marks_the_slots_holding_a_token,
    check_no_operator_masks_the_scores,
    check_the_model_is_built_with_the_mask_after_the_projections,
    check_the_causal_slide_reads_the_state_once,
    check_the_two_forms_compute_the_same_result,
    check_every_matrix_is_grabbed_from_a_slot_named_after_it,
    check_the_passes_read_and_return_the_arrays_of_the_model,
    check_the_backward_pass_writes_a_gradient_for_every_matrix,
    check_the_training_step_is_derived_from_the_model_as_built,
    check_a_training_step_of_the_causal_slide_saves_keys_and_values_per_slot,
    check_the_softmax_is_written_out_in_the_training_step,
    check_the_backward_pass_rebuilds_the_exponentials_from_the_queries_and_the_keys,
    check_the_forward_pass_saves_six_arrays,
    check_the_backward_pass_reads_the_keys_and_the_values_through_the_mask,
    check_the_gradient_of_the_output_reaches_the_state_directly,
    check_the_mask_reverses_into_its_transpose_for_the_keys_and_the_values,
    check_the_gradients_through_the_three_projections_are_added_once,
    check_the_weight_gradients_sum_over_the_tokens,
    check_the_training_step_keeps_every_block,
    check_the_gradients_agree_with_autograd,
    check_the_page_switches_between_the_forward_pass_and_the_training_step,
    check_every_legend_row_of_the_page_names_its_axis_in_code,
)


def main() -> int:
    return 1 if report_each_check(
        CHECKS, 'Expressing Attention with Weights and a Residual Connection') else 0


if __name__ == '__main__':
    sys.exit(main())
