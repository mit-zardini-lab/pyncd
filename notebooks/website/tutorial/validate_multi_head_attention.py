# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Check every claim `notebooks/website/tutorial/MultiHeadAttention.ipynb` makes.

    python notebooks/website/tutorial/validate_multi_head_attention.py

The notebook holds the prose and the figures, and the claims live here, one `check_`
function per claim, in the order the notebook makes them: the projections into heads,
the sum over the heads, the attention computed once per head, the CausalSlide, and the
page. The script prints one line per check and exits non-zero on a failure.
'''
from __future__ import annotations

import math
import pathlib
import sys
from collections.abc import Callable, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa
import data_structure.Category as cat  # noqa: E402
import data_structure.Operators as ops  # noqa: E402

import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.website.tutorial.check_gradients_in_torch as torch_check  # noqa
import notebooks.website.tutorial.express_attention as express_attention  # noqa
import notebooks.website.tutorial.tutorial_pages as tutorial_pages  # noqa: E402
from notebooks.website.tutorial.tutorial_checks import (  # noqa: E402
    block_titles, check_every_legend_row_carries_code_names, degree_names, ends,
    linear_named, names, nodes_of, report_each_check, require, targets_read)
from notebooks.website.tutorial.tutorial_wording import TEXT as text  # noqa

MODEL = express_attention.multi_head_attention()
DISPLAYED = slide_causal_reads_backwards.slide_causal_reads_backwards(MODEL)
PAGE = notebook_diagrams.DiagramSettings(
    mode=notebook_diagrams.DiagramMode.HTML,
    advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
    operator_explanations=express_attention.OPERATOR_EXPLANATIONS,
    operator_references=express_attention.OPERATOR_REFERENCES,
    operator_roles=express_attention.MULTI_HEAD_ROLES,
    reindexing_explanations=express_attention.REINDEXING_EXPLANATIONS)
SIZES = {'x': 5, 'm': 6, 'h': 3, 'd': 4}
SEED = 0


def sum_over_heads_in_torch(inputs: Sequence[torch.Tensor],
                            weights: torch_check.Tensors) -> torch.Tensor:
    '''The sum over the heads of the causal attention of each head, multiplied by the
    rows of `W^O` that belong to the head.'''
    state, = inputs
    result = torch.zeros(state.shape, dtype=state.dtype)
    for head in range(weights['W^{Q}'].shape[1]):
        queries = state @ weights['W^{Q}'][:, head]
        keys = state @ weights['W^{K}'][:, head]
        values = state @ weights['W^{V}'][:, head]
        scores = queries @ keys.T / math.sqrt(queries.shape[-1])
        causal = torch.ones(scores.shape, dtype=torch.bool).tril()
        attended = torch.softmax(scores.masked_fill(~causal, -math.inf), dim=-1) @ values
        result = result + attended @ weights['W^{O}'][head]
    return result


def core_operations() -> list:
    '''The dot product, the scale, the softmax and the weighted sum, in that order.'''
    return [node for node in nodes_of(cat.Operator, DISPLAYED)
            if isinstance(node.operator, (ops.Einops, ops.Arithmetic, ops.SoftMax))]


def check_the_attention_reads_and_returns_the_state() -> None:
    require(ends(DISPLAYED) == ([['x', 'm']], [['x', 'm']]),
            f'the attention reads and returns {ends(DISPLAYED)}')


def check_the_projections_produce_heads_and_channels() -> None:
    projections = {name: (targets_read(linear_named(name, MODEL)),
                          names(tuple(linear_named(name, MODEL).output_weaves[0]
                                      .target().shape())))
                   for name in ('W^{Q}', 'W^{K}', 'W^{V}', 'W^{O}')}
    require(projections == {'W^{Q}': ([['m']], ['h', 'd']),
                            'W^{K}': ([['m']], ['h', 'd']),
                            'W^{V}': ([['m']], ['h', 'd']),
                            'W^{O}': ([['h', 'd']], ['m'])},
            f'the projections read and write {projections}')


def check_the_attention_is_the_sum_over_the_heads() -> None:
    require(torch_check.forward_agrees_with(
                DISPLAYED, sum_over_heads_in_torch, SIZES,
                torch.Generator().manual_seed(SEED)),
            'the attention differs from the sum over the heads of the attention of '
            'each head')


def check_every_operation_of_the_attention_is_broadcast_over_the_heads() -> None:
    degrees = [degree_names(node) for node in core_operations()]
    require(len(degrees) == 4 and all('h' in degree for degree in degrees),
            f'the operations of the attention are broadcast over {degrees}')


def check_the_softmax_normalises_each_head_and_token_separately() -> None:
    softmax, = nodes_of(ops.SoftMax, DISPLAYED)
    require(targets_read(softmax) == [['w|x']] and degree_names(softmax) == ['h', 'x'],
            f'the softmax reads {targets_read(softmax)} over {degree_names(softmax)}')


def check_the_causal_slide_reads_the_state_once() -> None:
    masks = nodes_of(ops.View, DISPLAYED)
    runs = [[names(tuple(array.shape()))
             for array in linear_named(name, DISPLAYED).dom()]
            for name in ('W^{K}', 'W^{V}')]
    require(len(masks) == 1
            and masks[0].operator.name.to_bodies() == express_attention.MASK_VIEW_NAME
            and [names(tuple(array.shape())) for array in masks[0].dom()] == [['x', 'm']]
            and runs == [[['x', 'w|x', 'm']], [['x', 'w|x', 'm']]],
            f'the CausalSlide holds {len(masks)} views, and W^K and W^V read {runs}')


def check_the_attention_holds_the_block_of_scaled_dot_product_attention() -> None:
    require(block_titles(DISPLAYED) == [text.MULTI_HEAD_TITLE, text.CORE_TITLE],
            f'the blocks are {block_titles(DISPLAYED)}')


def page_variants() -> list[notebook_diagrams.PageVariant]:
    return tutorial_pages.multi_head_attention_page(DISPLAYED, PAGE)


def check_the_page_carries_the_forward_pass() -> None:
    variants = page_variants()
    notebook_diagrams.check_page_variants(variants, tutorial_pages.FORWARD)
    require([variant.identifier for variant in variants] == [tutorial_pages.FORWARD]
            and variants[0].term is DISPLAYED,
            f'the page holds {[variant.identifier for variant in variants]}')


def check_every_legend_row_of_the_page_names_its_axis_in_code() -> None:
    check_every_legend_row_carries_code_names(page_variants(), PAGE)


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_attention_reads_and_returns_the_state,
    check_the_projections_produce_heads_and_channels,
    check_the_attention_is_the_sum_over_the_heads,
    check_every_operation_of_the_attention_is_broadcast_over_the_heads,
    check_the_softmax_normalises_each_head_and_token_separately,
    check_the_causal_slide_reads_the_state_once,
    check_the_attention_holds_the_block_of_scaled_dot_product_attention,
    check_the_page_carries_the_forward_pass,
    check_every_legend_row_of_the_page_names_its_axis_in_code,
)


def main() -> int:
    return 1 if report_each_check(CHECKS, 'Expressing Multi-Head Attention') else 0


if __name__ == '__main__':
    sys.exit(main())
