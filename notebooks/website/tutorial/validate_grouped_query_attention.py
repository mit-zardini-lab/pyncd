# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Check every claim `notebooks/website/tutorial/GroupedQueryAttention.ipynb`
makes.

    python notebooks/website/tutorial/validate_grouped_query_attention.py

The notebook holds the prose and the figures, and the claims live here, one `check_`
function per claim, in the order the notebook makes them: the grouped projections, the
key head and the value head shared by a group, the sum over the groups and their heads,
the pairing of the released code of Mixtral-8x7B, the number of keys and values, the
CausalSlide, and the page. The script prints one line per check and exits non-zero on a
failure.
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
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402

import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.website.tutorial.check_gradients_in_torch as torch_check  # noqa
import notebooks.website.tutorial.express_attention as express_attention  # noqa
import notebooks.website.tutorial.tutorial_pages as tutorial_pages  # noqa: E402
from notebooks.website.tutorial.tutorial_checks import (  # noqa: E402
    block_titles, check_every_legend_row_carries_code_names, ends, linear_named, names,
    nodes_of, report_each_check, require, targets_read)
from notebooks.website.tutorial.tutorial_wording import TEXT as text  # noqa

MODEL = express_attention.grouped_query_attention()
DISPLAYED = slide_causal_reads_backwards.slide_causal_reads_backwards(MODEL)
PAGE = notebook_diagrams.DiagramSettings(
    mode=notebook_diagrams.DiagramMode.HTML,
    advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
    operator_explanations=express_attention.OPERATOR_EXPLANATIONS,
    operator_references=express_attention.OPERATOR_REFERENCES,
    operator_roles=express_attention.GROUPED_QUERY_ROLES,
    reindexing_explanations=express_attention.REINDEXING_EXPLANATIONS)
SIZES = {'x': 5, 'm': 6, "h'": 2, 'g': 3, 'd': 4}
SEED = 0


def shapes_of(arrays: Sequence[cat.Array]) -> list[list[str]]:
    return [names(tuple(array.shape())) for array in arrays]


def contractions() -> list[cat.Broadcasted]:
    '''The dot product and the weighted sum, in that order.'''
    return nodes_of(ops.Einops, DISPLAYED)


def sum_over_groups_and_heads_in_torch(inputs: Sequence[torch.Tensor],
                                       weights: torch_check.Tensors) -> torch.Tensor:
    '''The sum over every group and every head of the group of the causal attention of
    that head, with the key head and the value head of its group.'''
    state, = inputs
    groups, heads_per_group = weights['W^{Q}'].shape[1:3]
    result = torch.zeros(state.shape, dtype=state.dtype)
    for group in range(groups):
        keys = state @ weights['W^{K}'][:, group]
        values = state @ weights['W^{V}'][:, group]
        for head in range(heads_per_group):
            queries = state @ weights['W^{Q}'][:, group, head]
            scores = queries @ keys.T / math.sqrt(queries.shape[-1])
            causal = torch.ones(scores.shape, dtype=torch.bool).tril()
            attended = (torch.softmax(scores.masked_fill(~causal, -math.inf), dim=-1)
                        @ values)
            result = result + attended @ weights['W^{O}'][group, head]
    return result


def repeated_keys_in_torch(inputs: Sequence[torch.Tensor],
                           weights: torch_check.Tensors) -> torch.Tensor:
    '''The attention as the released code of Mixtral-8x7B computes it: the queries
    projected onto `|h'| |g|` heads, and every key head and value head repeated `|g|`
    times with `repeat_interleave`, so that query head `i` reads key head
    `floor(i / |g|)`.'''
    state, = inputs
    groups, heads_per_group, width = weights['W^{Q}'].shape[1:4]
    tokens = state.shape[0]
    queries = torch.einsum('xm,mhgd->xhgd', state, weights['W^{Q}']).reshape(
        tokens, groups * heads_per_group, width)
    keys = torch.einsum('xm,mhd->xhd', state, weights['W^{K}']).repeat_interleave(
        heads_per_group, dim=1)
    values = torch.einsum('xm,mhd->xhd', state, weights['W^{V}']).repeat_interleave(
        heads_per_group, dim=1)
    scores = torch.einsum('xid,yid->ixy', queries, keys) / math.sqrt(width)
    causal = torch.ones(scores.shape[1:], dtype=torch.bool).tril()
    attended = torch.einsum(
        'ixy,yid->xid', torch.softmax(scores.masked_fill(~causal, -math.inf), dim=-1),
        values)
    return torch.einsum('xhgd,hgdm->xm',
                        attended.reshape(tokens, groups, heads_per_group, width),
                        weights['W^{O}'])


def check_the_attention_reads_and_returns_the_state() -> None:
    require(ends(DISPLAYED) == ([['x', 'm']], [['x', 'm']]),
            f'the attention reads and returns {ends(DISPLAYED)}')


def check_the_queries_carry_groups_of_heads_and_the_keys_carry_groups() -> None:
    projections = {name: (targets_read(linear_named(name, MODEL)),
                          names(tuple(linear_named(name, MODEL).output_weaves[0]
                                      .target().shape())))
                   for name in ('W^{Q}', 'W^{K}', 'W^{V}', 'W^{O}')}
    require(projections == {'W^{Q}': ([['m']], ["h'", 'g', 'd']),
                            'W^{K}': ([['m']], ["h'", 'd']),
                            'W^{V}': ([['m']], ["h'", 'd']),
                            'W^{O}': ([["h'", 'g', 'd']], ['m'])},
            f'the projections read and write {projections}')


def check_every_query_head_of_a_group_scores_against_the_key_head_of_its_group() -> None:
    dot_product, _ = contractions()
    require(shapes_of(dot_product.dom()) == [['x', "h'", 'g', 'd'], ['x', 'w|x', "h'", 'd']]
            and shapes_of(dot_product.cod()) == [["h'", 'g', 'x', 'w|x']]
            and targets_read(dot_product) == [['d'], ['d']],
            f'the dot product reads {shapes_of(dot_product.dom())}')


def check_the_weighted_sum_reads_the_value_head_of_the_group() -> None:
    _, weighted_sum = contractions()
    require(shapes_of(weighted_sum.dom()) == [["h'", 'g', 'x', 'w|x'],
                                              ['x', 'w|x', "h'", 'd']]
            and shapes_of(weighted_sum.cod()) == [['x', "h'", 'g', 'd']],
            f'the weighted sum reads {shapes_of(weighted_sum.dom())}')


def check_the_attention_is_the_sum_over_the_groups_and_their_heads() -> None:
    require(torch_check.forward_agrees_with(
                DISPLAYED, sum_over_groups_and_heads_in_torch, SIZES,
                torch.Generator().manual_seed(SEED)),
            'the attention differs from the sum over the groups and their heads')


def check_the_released_pairing_of_query_heads_with_key_heads_is_the_grouping() -> None:
    require(torch_check.forward_agrees_with(
                DISPLAYED, repeated_keys_in_torch, SIZES,
                torch.Generator().manual_seed(SEED)),
            'the attention differs from the attention with repeated keys and values')


def check_no_view_regroups_the_heads_and_nothing_is_repeated() -> None:
    views = [node.operator.name.to_bodies() for node in nodes_of(ops.View, MODEL)]
    require(views == [express_attention.MASK_VIEW_NAME],
            f'the views of the attention are named {views}')


def check_the_keys_and_the_values_are_g_times_fewer_than_the_queries() -> None:
    def channels(name: str) -> int:
        target = linear_named(name, MODEL).output_weaves[0].target()
        return math.prod(
            nm.evaluate_integer(axis.local_size(), {
                symbol: SIZES[symbol.uid._name.to_bodies()]
                for symbol in nm.free_symbols(axis.local_size())})
            for axis in target.shape())
    require(channels('W^{Q}') == SIZES['g'] * channels('W^{K}')
            and channels('W^{K}') == channels('W^{V}'),
            f'the queries hold {channels("W^{Q}")} channels and the keys '
            f'{channels("W^{K}")}')


def check_the_causal_slide_reads_the_state_once() -> None:
    masks = nodes_of(ops.View, DISPLAYED)
    runs = [shapes_of(linear_named(name, DISPLAYED).dom()) for name in ('W^{K}', 'W^{V}')]
    require(len(masks) == 1 and shapes_of(masks[0].dom()) == [['x', 'm']]
            and runs == [[['x', 'w|x', 'm']], [['x', 'w|x', 'm']]],
            f'the CausalSlide holds {len(masks)} views, and W^K and W^V read {runs}')


def check_the_attention_holds_the_block_of_scaled_dot_product_attention() -> None:
    require(block_titles(DISPLAYED) == [text.GROUPED_QUERY_TITLE, text.CORE_TITLE],
            f'the blocks are {block_titles(DISPLAYED)}')


def page_variants() -> list[notebook_diagrams.PageVariant]:
    return tutorial_pages.grouped_query_attention_page(DISPLAYED, PAGE)


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
    check_the_queries_carry_groups_of_heads_and_the_keys_carry_groups,
    check_every_query_head_of_a_group_scores_against_the_key_head_of_its_group,
    check_the_weighted_sum_reads_the_value_head_of_the_group,
    check_the_attention_is_the_sum_over_the_groups_and_their_heads,
    check_the_released_pairing_of_query_heads_with_key_heads_is_the_grouping,
    check_no_view_regroups_the_heads_and_nothing_is_repeated,
    check_the_keys_and_the_values_are_g_times_fewer_than_the_queries,
    check_the_causal_slide_reads_the_state_once,
    check_the_attention_holds_the_block_of_scaled_dot_product_attention,
    check_the_page_carries_the_forward_pass,
    check_every_legend_row_of_the_page_names_its_axis_in_code,
)


def main() -> int:
    return 1 if report_each_check(CHECKS, 'Expressing Grouped-Query Attention') else 0


if __name__ == '__main__':
    sys.exit(main())
