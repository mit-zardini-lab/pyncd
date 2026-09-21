# Claude Opus 5 (1M context), high effort.
'''Checking the discovery of a broadcast against a hand-written attention core.

The core below is the one `notebooks/sota/DeepSeekV41Flash/attention_core.py`
writes, cut down to one window branch and one selected branch and to the operations
the head axis touches. Its queries and its learned sink carry the head axis, its
window latents and its selected latents do not, and its result carries it, so the
reindexings the discovery has to find are `((0,), (0,), (), ())`.

    python algebra/validate_discovering_broadcasts.py

Both directions are checked. The core is discovered as one block broadcast over
the head axis, and a candidate that states that the queries are read once for
every head is refused with the domain position it disagrees on.

`obsidian/02-categories/Discovering Broadcasts.md` holds the mathematics.
'''
from __future__ import annotations
import sys

import construction_helpers as ch  # noqa: F401 - operator overloads
import construction_helpers.lift as lift
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops

import algebra.discovering_broadcasts as discovering_broadcasts
import algebra.reindexing_absorption as reindexing_absorption

R = cat.Reals()
HEAD = cat.RawAxis.named('h')
TOKEN = cat.RawAxis.named('x')
CHANNEL = cat.RawAxis.named('c')
WINDOW = cat.RawAxis.named('w')
SELECTED = cat.RawAxis.named('s')


def hold(array: cat.Array) -> cat.Rearrangement:
    return cat.ProdObject((array,)).identity()


def route(mapping: tuple[int, ...],
          arrays: tuple[cat.Array, ...]) -> cat.Rearrangement:
    return cat.Rearrangement(mapping, arrays)


def over(axes: tuple[cat.RawAxis, ...],
         morphism: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    return lift.morphism_object_lift(morphism, cat.ProdObject(axes))


def reciprocal() -> cat.BroadcastedCategory:
    return ops.Arithmetic.template(nm.Integer(1) / nm.x, name='z^{-1}')


def add_sink(head: tuple[cat.RawAxis, ...]) -> cat.Broadcasted:
    '''The learned sink logit added onto the denominator, read at the head alone.

    Its weave holds one tiled slot per axis of the degree it reads, so the sink
    is a scalar per head where the denominator is a scalar per head and token.
    '''
    tiled = (cat.WeaveMode.TILED,)
    return cat.Broadcasted(
        operator=ops.AdditionOp(),
        input_weaves=(cat.Weave(R, tiled * (1 + len(head))),
                      cat.Weave(R, tiled * len(head))),
        output_weaves=(cat.Weave(R, tiled * (1 + len(head))),),
        reindexings=(
            cat.Rearrangement(tuple(range(1 + len(head))), (*head, TOKEN)),
            cat.Rearrangement(tuple(range(len(head))), (*head, TOKEN))))


def attention_core(head: tuple[cat.RawAxis, ...]) -> cat.Block:
    '''One softmax over the window slots, the selected slots and the sink.

    `head` is `(HEAD,)` for the core the notebook draws and `()` for the body
    the discovery has to find, so the two forms are written once and the axis is
    the only difference between them.
    '''
    lead = 'h ' if head else ''
    sink = cat.Array(R, head)
    queries = cat.Array(R, (*head, TOKEN, CHANNEL))
    window = cat.Array(R, (TOKEN, WINDOW, CHANNEL))
    selected = cat.Array(R, (TOKEN, SELECTED, CHANNEL))
    window_scores = cat.Array(R, (*head, TOKEN, WINDOW))
    selected_scores = cat.Array(R, (*head, TOKEN, SELECTED))
    denominator = cat.Array(R, (*head, TOKEN))
    return cat.Block.template(
        route((0, 1, 2, 1, 3, 2, 3), (sink, queries, window, selected))
        @ (hold(sink)
           * ops.Einops.template(f'{lead}x c, x w c -> {lead}x w')
           * ops.Einops.template(f'{lead}x c, x s c -> {lead}x s')
           * hold(window) * hold(selected))
        @ route((0, 1, 1, 2, 2, 3, 4),
                (sink, window_scores, selected_scores, window, selected))
        @ (hold(sink)
           * ops.Einops.template(f'{lead}x w -> {lead}x')
           * hold(window_scores)
           * ops.Einops.template(f'{lead}x s -> {lead}x')
           * hold(selected_scores)
           * hold(window) * hold(selected))
        @ route((0, 1, 3, 2, 4, 5, 6),
                (sink, denominator, window_scores, denominator,
                 selected_scores, window, selected))
        @ (hold(sink) * over((*head, TOKEN), ops.AdditionOp.template())
           * hold(window_scores) * hold(selected_scores)
           * hold(window) * hold(selected))
        @ route((1, 0, 2, 3, 4, 5),
                (sink, denominator, window_scores, selected_scores,
                 window, selected))
        @ (add_sink(head) * hold(window_scores) * hold(selected_scores)
           * hold(window) * hold(selected))
        @ (over((*head, TOKEN), reciprocal())
           * hold(window_scores) * hold(selected_scores)
           * hold(window) * hold(selected))
        @ route((0, 1, 3, 2, 4),
                (denominator, window_scores, selected_scores, window, selected))
        @ (hold(denominator)
           * ops.Einops.template(f'{lead}x w, x w c -> {lead}x c')
           * ops.Einops.template(f'{lead}x s, x s c -> {lead}x c'))
        @ (hold(denominator) * over((*head, TOKEN, CHANNEL),
                                    ops.AdditionOp.template()))
        @ ops.Einops.template(f'{lead}x, {lead}x c -> {lead}x c'),
        title='Core', fill_color='#C5BEDF')


def cases() -> dict[str, bool]:
    core = attention_core((HEAD,))
    found = discovering_broadcasts.discover_broadcast_over_axes(
        core, (HEAD,), 'Core')
    expansion = discovering_broadcasts.expand_broadcast_of_block(found.candidate)
    repeats = [operation
               for operation in discovering_broadcasts.operations_of(expansion)
               if reindexing_absorption.is_node(operation)]
    dropped_head = discovering_broadcasts.confirm_broadcast_expansion(
        core,
        discovering_broadcasts.broadcast_block_over_axes(
            found.body, (HEAD,), ((0,), (), (), ()), 'Core'))
    no_degree = discovering_broadcasts.confirm_broadcast_expansion(
        core,
        discovering_broadcasts.broadcast_block_over_axes(
            found.body, (), ((), (), (), ()), 'Core'))
    trailing_head = over((TOKEN, HEAD), ops.AdditionOp.template())
    try:
        discovering_broadcasts.remove_leading_degree_axes(
            trailing_head, (HEAD,))
        trailing_head_refused = 'nothing was raised'
    except discovering_broadcasts.DegreeAxesNotRemovable as reason:
        trailing_head_refused = str(reason)
    return {
        'the head axis is discovered as the core\'s degree': found.is_confirmed(),
        'the queries and the sink read the head and the latents do not':
            tuple(reindexing.mapping
                  for reindexing in found.candidate.reindexings)
            == ((0,), (0,), (), ()),
        'the discovered body reads the sink at the empty target':
            tuple(found.body.dom()[0].shape()) == (),
        'the discovered body carries the head nowhere':
            tuple(found.candidate.operator.block.dom()) == tuple(
                attention_core(()).dom()),
        'the candidate has the core\'s own domain and codomain':
            tuple(found.candidate.dom()) == tuple(core.dom())
            and tuple(found.candidate.cod()) == tuple(core.cod()),
        'the expansion repeats each latent along the head':
            len(repeats) == 2,
        'a candidate that drops the head from the queries is refused':
            not dropped_head.is_confirmed()
            and dropped_head.named_difference == (
                'domain position 1 is [h, x, c] in the original and [x, c] in '
                'the expansion'),
        'a candidate with an empty degree states no broadcast':
            not no_degree.is_confirmed()
            and 'empty degree' in no_degree.named_difference,
        'an operation carrying the head at a later degree position is refused':
            trailing_head_refused == (
                'AdditionOp([x, h] [x, h]) -> [x, h] at degree [x, h] carries h '
                'at a later degree position, and a deletion takes a leading '
                'degree alone'),
    }


if __name__ == '__main__':
    results = cases()
    for name, passed in results.items():
        print(f'  {"ok  " if passed else "FAIL"}  {name}')
    sys.exit(0 if all(results.values()) else 1)
