# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim made about GLM-5.3 as `notebooks/sota/GLM53/` builds it.

    python notebooks/sota/GLM53/validate_glm53.py

The claims live here, one `check_` function per claim, in the shape of
`notebooks/sota/DeepSeekV41Flash/validate_quantised_text_only_model.py`, and
`notebooks/website/modern/validate_glm53.py` runs every one of them for
`notebooks/website/modern/GLM53.ipynb`. The script prints one line per check and the
time the run took, and it exits non-zero on a failure.
`check_every_claim_of_the_notebook` runs the same checks and raises when one of them
fails, for a notebook cell that runs them.

The checks stand in this order: the ends of the model and
its layer plan, the rotary embedding, the indexer and its selection, the attention, the
feed-forward maps, the output head, and the tables that fill the inspection boxes.
`check_the_layer_plan_is_the_released_plan` compares the attention mode and the
feed-forward map of every one of the 78 layers with the lists derived by the reference,
and the two released lists were compared with the `indexer_types` and the
`mlp_layer_types` of the checkpoint on 2026-09-23.
'''
from __future__ import annotations

import pathlib
import sys
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import algebra.reindexing_absorption as reindexing_absorption  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import para.data_structure.ParaWrap as para_wrap  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
from notebooks.display.explain_operators import shows_whole_formula  # noqa: E402
import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
from para.data_structure.ParaBlockOperator import (  # noqa: E402
    slots_dropped, slots_grabbed)

import notebooks.sota.GLM53.attention_modes as attention_modes  # noqa: E402
import notebooks.sota.GLM53.declared_axes as declared_axes  # noqa: E402
import notebooks.sota.GLM53.feed_forward as feed_forward  # noqa: E402
import notebooks.sota.GLM53.layer_stack as layer_stack  # noqa: E402
import notebooks.sota.GLM53.lightning_indexer as lightning_indexer  # noqa: E402
import notebooks.sota.GLM53.mixture_of_experts as mixture_of_experts  # noqa: E402
import notebooks.sota.GLM53.multi_latent_attention as multi_latent_attention  # noqa
import notebooks.sota.GLM53.operator_explanations as operator_explanations  # noqa: E402
import notebooks.sota.GLM53.rotary_embedding as rotary_embedding  # noqa: E402
import notebooks.sota.GLM53.whole_model as whole_model  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (  # noqa: E402
    axes, axis_name)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text  # noqa

MODEL: cat.Morphism = whole_model.glm53
ASSIGNED_SIZES: dict[str, int] = whole_model.released_assigned_sizes()

LAYERS = 78
FULL_LAYERS = 21
INDEX_TOPK_FREQUENCY = 4
INDEX_SKIP_TOPK_OFFSET = 3
FIRST_K_DENSE_REPLACE = 3


def released_indexer_type(layer: int) -> str:
    '''The attention mode `GlmMoeDsaConfig.__post_init__` gives layer `layer` from
    `index_topk_freq` and `index_skip_topk_offset`.'''
    shifted = max(layer - INDEX_SKIP_TOPK_OFFSET + 1, 0)
    return 'full' if shifted % INDEX_TOPK_FREQUENCY == 0 else 'shared'


def released_mlp_layer_type(layer: int) -> str:
    return 'dense' if layer < FIRST_K_DENSE_REPLACE else 'sparse'


RELEASED_PLAN: tuple[tuple[str, str], ...] = tuple(
    (released_indexer_type(layer), released_mlp_layer_type(layer))
    for layer in range(LAYERS))

SUBLAYER_KINDS: dict[str, str] = {
    attention_modes.FULL_BOX: 'full',
    attention_modes.SHARED_BOX: 'shared',
    feed_forward.DENSE_BOX: 'dense',
    mixture_of_experts.MIXTURE_BOX: 'sparse',
}


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


def boxes_named(short_name: str, term: cat.Morphism) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in tutil.type_search(cat.Broadcasted, term)
                 if whole_model.is_box_named(node, short_name))


def box_name_of(term: object) -> str | None:
    '''The short name of the box `term` is, or stands inside a wrap of.'''
    match term:
        case para_wrap.ParaWrap(body=body):
            return box_name_of(body)
        case cat.Broadcasted(operator=ops.BlockOperator(name=name)) if name is not None:
            return name.to_bodies()
    return None


def sublayer_sequence(term: object) -> list[str]:
    '''The kind of every sublayer `term` runs, in order, with every repeated block
    written out as many times as it repeats.'''
    match term:
        case cat.Composed(content=parts):
            return [kind for part in parts for kind in sublayer_sequence(part)]
        case cat.Block(body=body, block_tag=block_tag):
            return sublayer_sequence(body) * block_tag.repetition._value
        case cat.ProductOfMorphisms(content=parts):
            return [kind for part in parts for kind in sublayer_sequence(part)]
    name = box_name_of(term)
    return [SUBLAYER_KINDS[name]] if name in SUBLAYER_KINDS else []


def layer_plan(term: object) -> tuple[tuple[str, str], ...]:
    '''The attention mode and the feed-forward map of every layer of `term`.'''
    kinds = sublayer_sequence(term)
    return tuple(zip(kinds[0::2], kinds[1::2]))


# ==========================================================================
# The ends of the model and its layer plan.
# ==========================================================================
def check_the_model_reads_identifiers_and_returns_logits() -> None:
    '''The model reads the identifier of each token and returns one logit per token and
    vocabulary entry.'''
    require(axes(MODEL) == ([['x']], [['x', 'v']]),
            f'the model reads and returns {axes(MODEL)}')
    tokens, = MODEL.dom()
    require(isinstance(tokens.datatype, cat.Natural),
            f'the model reads {tokens.datatype}')


def check_the_stack_holds_seventy_eight_layers() -> None:
    '''The layer stack holds 78 layers, and 21 of them run their own indexer.'''
    count = layer_stack.layer_count(layer_stack.layer_stack)
    require(count == LAYERS, f'the stack holds {count} layers')
    full = layer_stack.runs_of_blocks_titled(text.FULL_TITLE, layer_stack.layer_stack)
    require(full == FULL_LAYERS, f'{full} layers run their own indexer')


def check_the_layer_plan_is_the_released_plan() -> None:
    '''Layer `i` runs its own indexer when `max(i - 2, 0)` is a multiple of 4, and the
    first three layers hold the dense MLP.'''
    plan = layer_plan(layer_stack.layer_stack)
    differing = [layer for layer, (drawn, released) in
                 enumerate(zip(plan, RELEASED_PLAN)) if drawn != released]
    require(len(plan) == LAYERS and not differing,
            f'the plan holds {len(plan)} layers and differs from the release at '
            f'layers {differing}')


def check_every_repeated_block_returns_its_domain() -> None:
    '''Every block of the plan returns the hidden state it reads, so the residual is the
    only wire between the layers.'''
    blocks = (layer_stack.layers_before_the_first_group, layer_stack.first_group,
              layer_stack.repeated_groups)
    require(all(block.dom() == block.cod() for block in blocks),
            f'the blocks read {[axes(block)[0] for block in blocks]} and return '
            f'{[axes(block)[1] for block in blocks]}')


def check_the_selection_is_the_one_tape_slot() -> None:
    '''The selection is the one tape slot of the model, and it is read wherever it is
    written.'''
    dropped = slots_dropped(MODEL)
    grabbed = slots_grabbed(MODEL)
    require(dropped == grabbed and len(dropped) == 1,
            f'the model writes {len(dropped)} slots and reads {len(grabbed)}')


# ==========================================================================
# The rotary embedding.
# ==========================================================================
def check_the_turned_channels_are_thirty_two_pairs() -> None:
    '''The attention turns 64 channels of every query head and key as 32 pairs, and the
    indexer turns the first 64 of its 128 channels.'''
    require(size_of('t') == 32 and size_of('d') == 128,
            f"t is {size_of('t')} and d is {size_of('d')}")
    cut, = (node for node in tutil.type_search(
                cat.Broadcasted,
                rotary_embedding.ROTATE_INDEXER_CHANNELS.operator.block)
            if isinstance(node.operator, aops.DeconcatenateAxes))
    turned, left_alone = cut.operator.parts()
    require(turned == declared_axes.p and left_alone == declared_axes.dbar,
            f'the indexer cuts its channels into {axis_name(turned)} and '
            f'{axis_name(left_alone)}')


def check_a_full_layer_turns_four_arrays_and_a_shared_layer_two() -> None:
    '''A Full layer turns the queries, the keys, the indexer queries and the indexer
    keys, and a Shared layer turns the queries and the keys.'''
    full = len(boxes_named(rotary_embedding.ROTATION_BOX,
                           attention_modes.full_attention()))
    shared = len(boxes_named(rotary_embedding.ROTATION_BOX,
                             attention_modes.shared_attention(
                                 attention_modes.SLOT_SELECTION)))
    require(full == 4 and shared == 2,
            f'a Full layer turns {full} arrays and a Shared layer {shared}')


# ==========================================================================
# The indexer and its selection.
# ==========================================================================
def check_the_indexer_reads_the_low_rank_and_the_hidden_state() -> None:
    '''The indexer reads the query low rank and the hidden state and returns one score
    per query and token at or before it.'''
    require(axes(lightning_indexer.INDEXER)
            == ([['x', 'q'], ['x', 'm']], [['x', 'r|x']]),
            f'the indexer reads and returns {axes(lightning_indexer.INDEXER)}')


def check_the_sizes_of_the_indexer() -> None:
    '''The indexer runs 32 heads of 128 channels and keeps 2048 tokens.'''
    require(size_of('i') == 32 and size_of('d') == 128 and size_of('s') == 2048,
            f"i is {size_of('i')}, d is {size_of('d')} and s is {size_of('s')}")


def check_the_selection_hands_out_positions_alone() -> None:
    '''The top-2048 hands out the distances of the kept tokens and no score.'''
    select = lightning_indexer.SELECT
    require(select.operator.form is dst.SelectionForm.ONLY_SELECTION,
            f'the selection is in the {select.operator.form.name} form')
    require(isinstance(select.cod()[0].datatype, cat.Natural),
            f'the selection hands out {select.cod()[0].datatype}')


def check_the_boxes_computed_once_per_index_expand_back() -> None:
    '''The indexer scores, the attention core, the dense MLP and the mixture are each
    one box computed once per index, and each expands back to its written-out form.'''
    confirmations = {
        'Sco': lightning_indexer.SCORING.is_confirmed(),
        'Core': multi_latent_attention.CORE.is_confirmed(),
        'MLP': feed_forward.DENSE_MLP_CONFIRMATION.is_confirmed(),
        'MoE': mixture_of_experts.MIXTURE_CONFIRMATION.is_confirmed(),
    }
    require(all(confirmations.values()), f'confirmed: {confirmations}')


def check_a_full_layer_runs_one_indexer_and_a_shared_layer_none() -> None:
    '''A Full layer runs one indexer and one top-2048, and a Shared layer runs
    neither.'''
    def selections(term: cat.Morphism) -> int:
        return sum(isinstance(node.operator, dst.TopK)
                   for node in tutil.type_search(cat.Broadcasted, term))
    full = attention_modes.full_attention()
    shared = attention_modes.shared_attention(attention_modes.SLOT_SELECTION)
    require(len(boxes_named(lightning_indexer.INDEXER_BOX, full)) == 1
            and selections(full) == 1,
            f'a Full layer runs {selections(full)} selections')
    require(not boxes_named(lightning_indexer.INDEXER_BOX, shared)
            and selections(shared) == 0,
            f'a Shared layer runs {selections(shared)} selections')


# ==========================================================================
# The attention.
# ==========================================================================
def check_the_sizes_of_the_attention() -> None:
    '''The attention runs 64 heads. A query head and a key head hold 192 unturned and
    64 turned channels, a value head holds 256, the query low rank 2048 and the latent
    512.'''
    require(size_of('h') == 64 and size_of('n') == 192 and size_of('u') == 256,
            f"h is {size_of('h')}, n is {size_of('n')} and u is {size_of('u')}")
    require(size_of('q') == 2048 and size_of('c') == 512,
            f"q is {size_of('q')} and c is {size_of('c')}")
    widths = tuple(map(
        width_of, (declared_axes.p, declared_axes.a, declared_axes.dbar)))
    require(widths == (64, 256, 64),
            f'p, a and the unturned indexer channels are {widths} wide')


def check_the_join_of_the_key_reads_one_turned_key_per_token() -> None:
    '''The keys and the values hold no view, so no array is copied to every head. The
    join of the key is the repeat of the turned key over the heads absorbed into a join
    of two arrays that both carry the heads. Its turned operand is therefore read
    through a reindexing that deletes the head.'''
    keys_and_values = whole_model.part_titled(text.KEYS_AND_VALUES_TITLE, MODEL)
    operations = tuple(tutil.type_search(cat.Broadcasted, keys_and_values))
    views = [node for node in operations if isinstance(node.operator, ops.View)]
    require(not views, f'the keys and the values hold {len(views)} views')
    x, h, n, p, a = (declared_axes.x, declared_axes.h, declared_axes.n, declared_axes.p,
                     declared_axes.a)
    repeat_over_heads = ops.View.template(reindexing=cat.Rearrangement((0, 2), (x, h, p)))
    join_of_heads = aops.ConcatenateAxes.template(((x, h, n), (x, h, p)), concatenated=a)
    joins = [node for node in operations if isinstance(node.operator, aops.ConcatenateAxes)]
    require(joins == [reindexing_absorption.absorb(repeat_over_heads, join_of_heads, 1)],
            f'the keys and the values hold the joins {joins}')
    require(axes(joins[0]) == ([['x', 'h', 'n'], ['x', 'p']], [['x', 'h', 'a']]),
            f'the join of the key reads and returns {axes(joins[0])}')


def check_the_core_reads_the_selected_keys_and_values() -> None:
    '''The core is computed once per query and head, and reads the query, the keys at
    the selected tokens and the values at the selected tokens.'''
    candidate = multi_latent_attention.CORE.candidate
    require(axes(candidate) == ([['x', 'h', 'a'], ['x', 'h', 's|x', 'a'],
                                  ['x', 'h', 's|x', 'u']], [['x', 'h', 'u']]),
            f'the core reads and returns {axes(candidate)}')
    require(len(candidate.degree()) == 2,
            f'the core is broadcast over {len(candidate.degree())} axes')


def check_both_modes_read_and_return_the_hidden_state() -> None:
    '''Every attention mode reads and returns the hidden state alone, with the
    selection on the tape.'''
    modes = (attention_modes.FULL, attention_modes.FULL_PUBLISHING,
             attention_modes.FULL_PUBLISHING_IN_GROUP, attention_modes.SHARED,
             attention_modes.SHARED_IN_GROUP)
    require(all(axes(mode) == ([['x', 'm']], [['x', 'm']]) for mode in modes),
            f'the modes read and return {[axes(mode) for mode in modes]}')


# ==========================================================================
# The feed-forward maps.
# ==========================================================================
def check_three_dense_layers_and_seventy_five_mixtures() -> None:
    '''The dense MLP runs in 3 layers and the mixture of experts in 75.'''
    stack = layer_stack.layer_stack
    dense = layer_stack.runs_of_blocks_titled(text.DENSE_MLP_TITLE, stack)
    mixture = layer_stack.runs_of_blocks_titled(text.MIXTURE_TITLE, stack)
    require(dense == 3 and mixture == 75,
            f'the dense MLP runs {dense} times and the mixture {mixture} times')


def check_the_router_keeps_eight_of_two_hundred_and_fifty_six() -> None:
    '''The router scores 256 experts and keeps 8, and each expert is 2048 wide, as the
    shared expert is. The dense MLP is 12288 wide.'''
    require(size_of('e') == 256 and size_of('k') == 8,
            f"e is {size_of('e')} and k is {size_of('k')}")
    require(size_of('f') == 2048 and size_of('g') == 12288,
            f"f is {size_of('f')} and g is {size_of('g')}")
    gates = boxes_named(mixture_of_experts.GATE_BOX, mixture_of_experts.MIXTURE)
    require(len(gates) == 1, f'the mixture holds {len(gates)} router boxes')


# ==========================================================================
# The output head.
# ==========================================================================
def check_the_output_head_writes_every_vocabulary_entry() -> None:
    '''The output head maps the hidden state of 6144 channels onto 154880 entries.'''
    require(size_of('m') == 6144 and size_of('v') == 154880,
            f"m is {size_of('m')} and v is {size_of('v')}")


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
    '''Every named view has an explanation that opens a box over it, and every
    elementwise map whose name hides part of its formula has a role. The repeat and the
    diagonal are rearrangements, which carry no name of their own, so the check presents
    the model as the display does and asks for a block over every named view.'''
    nodes = tuple(tutil.type_search(cat.Broadcasted, MODEL))
    views = {node.operator.name.to_bodies() for node in nodes
             if isinstance(node.operator, ops.View) and node.operator.name is not None}
    hidden = {node.operator.name.to_bodies() for node in nodes
              if isinstance(node.operator, ops.Arithmetic)
              and not isinstance(node.operator, ops.View)
              and not shows_whole_formula(node.operator)}
    unexplained_views = sorted(
        views - operator_explanations.REINDEXING_EXPLANATIONS.keys())
    unexplained_maps = sorted(hidden - operator_explanations.ARITHMETIC_ROLES.keys())
    require(not unexplained_views and not unexplained_maps,
            f'the views {unexplained_views} and the maps {unexplained_maps} have '
            f'no row')
    unboxed_views = explain_reindexings.names_of_unexplained_views(
        explain_reindexings.present(
            MODEL, operator_explanations.REINDEXING_EXPLANATIONS))
    require(not unboxed_views, f'no box opens over the views {unboxed_views}')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_model_reads_identifiers_and_returns_logits,
    check_the_stack_holds_seventy_eight_layers,
    check_the_layer_plan_is_the_released_plan,
    check_every_repeated_block_returns_its_domain,
    check_the_selection_is_the_one_tape_slot,
    check_the_turned_channels_are_thirty_two_pairs,
    check_a_full_layer_turns_four_arrays_and_a_shared_layer_two,
    check_the_indexer_reads_the_low_rank_and_the_hidden_state,
    check_the_sizes_of_the_indexer,
    check_the_selection_hands_out_positions_alone,
    check_the_boxes_computed_once_per_index_expand_back,
    check_a_full_layer_runs_one_indexer_and_a_shared_layer_none,
    check_the_sizes_of_the_attention,
    check_the_join_of_the_key_reads_one_turned_key_per_token,
    check_the_core_reads_the_selected_keys_and_values,
    check_both_modes_read_and_return_the_hidden_state,
    check_three_dense_layers_and_seventy_five_mixtures,
    check_the_router_keeps_eight_of_two_hundred_and_fifty_six,
    check_the_output_head_writes_every_vocabulary_entry,
    check_every_weight_has_a_role,
    check_every_named_view_and_hidden_formula_is_explained,
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
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} GLM-5.3 checks passed in '
          f'{time.perf_counter() - started:.0f} s')
    return failures


def check_every_claim_of_the_notebook() -> None:
    '''Every check, raising when one of them fails, so that a notebook cell running
    the checks fails with it.'''
    failures = report_each_check()
    if failures:
        raise ClaimDoesNotHold(
            f'{failures} of {len(CHECKS)} claims of the notebook do not hold')


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
