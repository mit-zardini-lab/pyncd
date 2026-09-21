'''The sliding window and the attention core of DeepSeek-V4.1-Flash.

Written by Claude Opus 5, effort high.

One softmax covers the window slots, the selected slots and the learned sink.

The window view reads the token `j_w` positions back from query `i_x` at slot `j_w`, so
its row is `x' = x - w`, at stride 1 on the query and stride -1 on the slot. Slot 0 is
the current token and no slot reads a later one, so the window is causal by the sign of
the slot's stride. A query earlier than position `|w| - 1` reads before the first token
at its last slots, and `mark_sparse_domains.guarded_view` marks the slot axis `w|x` from
that row, live where `i_x - j_w >= 0`, with its empty positions at the end.

`attend_over_all_heads_on_slots` is the core over one kind of slot, written for every
head: one score contraction, one exponential, one denominator sum and one contraction
against the latents. The model reads the window slots and the selected slots under one
softmax by laying the two kinds of latent end to end along one slot axis with
`aops.ConcatenateAxes` and handing the result to that core, which is the reference's own
spelling. `ConcatenateAxes` holds one reindexing per part, read covariantly, their images
filling the slot axis. That axis is an `AxisConcatenation.ConcatenatedAxis`: it references
the two slot axes, its size is the sum of theirs, and it is labelled by them. The core is
written over the raw axes `w` and `s`, and a mode composes the guarded `w|x` of the window
view and the guarded `s|x` of the selection into it, so in a mode the axis reads
`w|x + s|x`.

The concatenated axis is dense. The window slots and the selected slots each carry an
affine form saying which of their positions hold a value for a query, and the live
positions of the concatenation are the union of two runs under two forms, which no single
affine form states. `concatenation_expansion.expand_concatenations` rewrites every
consumer of a concatenated wire into the combination of the consumers of the parts, so
each part and its own form are restored and nothing downstream needs the union. The
concatenation feeds a box here, and `expand_concatenations_through_blocks` writes that
box out over the head axis before rewriting its consumers. What it returns is
`attend_over_all_heads_with_entries`, the two-branch form, and the notebook asserts that
by comparing the two listings.

`discovering_broadcasts.discover_broadcast_over_axes` reads the head axis off the core's
own operations, deletes the head from the body, returns the body as one box, and confirms
the box by broadcasting it back over the head axis and comparing. The reindexings are
`((0,), (0,), ())`: the sink logit and the query are read at the head and the
concatenated latents are read whole by every head, which is what multi-query attention
over one shared key-value latent means. The concatenation stands outside the box, because
the two kinds of latent are shared by every head and `ops.BlockOperator.expand` lifts a
whole body over the degree, so a box holding the concatenation would concatenate the
latents once per head.

The sink logit is an operand of the core rather than a weight inside it, so the mode that
encloses the box supplies it and a layer's own sink is drawn in the layer's figure.
'''
from __future__ import annotations

from collections.abc import Callable

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    exponential, reciprocal, weights)
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, Q, WKV, c, h, m, s, w, x
from notebooks.sota.DeepSeekV41Flash.reference_links import kernel_lines, model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

CORE_COLOUR = '#C5BEDF'
SINK_LOGIT = cat.Array(R, (h,))
SLOT_SCORES = cat.Array(R, (h, x, w))
SELECTED_SCORES = cat.Array(R, (h, x, s))
DENOMINATOR = cat.Array(R, (h, x))
SELECTED_KV = cat.Array(R, (x, s, c))


def window_view() -> cat.Broadcasted:
    '''The |w| tokens at and before the current one: the stride morphism x' = x - w on
    the positions, times the identity on the channels.'''
    win_map = sc.StrideMorphism(
        _dom=(x, w),
        _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
        name=fd.DynamicName('win'))
    return mark_sparse_domains.guarded_view(
        reindexing=(win_map, cat.ProdObject((c,)).identity()), name='win')


def project_window_latents() -> cat.BroadcastedCategory:
    '''One normalised latent per token, from this layer's own hidden state, before the
    window view reads it per query.'''
    return ((x >> ops.Linear.template((m,), (c,), 'W^{KV}'))
            @ over((x,), ops.Normalize.template((c,))))


def window_kv() -> cat.BroadcastedCategory:
    '''This layer's own sliding-window latents, from its own hidden state.'''
    return project_window_latents() @ window_view()


def score_slots(letter: str) -> cat.Broadcasted:
    '''Every query head against the latent at every slot. `letter` is the einops token
    of the slot axis, which a concatenated axis needs because it has no name of its own
    and is labelled by its parts.'''
    return ops.Einops.template(f'h x c, x {letter} c -> h x {letter}')


def sum_over_slots(letter: str) -> cat.Broadcasted:
    '''The denominator of one kind of slot, the sum of its exponentials over its
    slots.'''
    return ops.Einops.template(f'h x {letter} -> h x')


def weight_slots(letter: str) -> cat.Broadcasted:
    '''The numerator of one kind of slot, its exponentials against its own latents.'''
    return ops.Einops.template(f'h x {letter}, x {letter} c -> h x c')


def exponentiated_scores[A: cat.Axis](
    slot_axis: A, letter: str) -> cat.BroadcastedCategory:
    '''One kind of slot's scores, exponentiated.'''
    return score_slots(letter) @ over((h, x, slot_axis), exponential())


def concatenate_slot_latents() -> cat.Broadcasted:
    '''The window latents and the selected latents laid end to end along one slot axis,
    with the queries and the channels as the degree.'''
    return aops.ConcatenateAxes.template(((x, w, c), (x, s, c)))


def sink_logits() -> cat.Broadcasted:
    '''The learned sink logit, one number per head with no value attached. A head whose
    scores are all small puts most of its weight here and takes almost nothing from its
    slots.'''
    return weights('\\mathrm{sink}', (h,))


def add_sink() -> cat.BroadcastedCategory:
    '''The exponentiated sink logit added to the denominator of every query. The logit is
    one number per head, so the addition's first reindexing deletes the query degree.'''
    T = cat.WeaveMode.TILED
    return ((over((h,), exponential()) * hold(DENOMINATOR))
            @ cat.Broadcasted(
                operator=ops.AdditionOp(),
                input_weaves=(cat.Weave(R, (T,)), cat.Weave(R, (T, T))),
                output_weaves=(cat.Weave(R, (T, T)),),
                reindexings=(route((0,), (h, x)), route((0, 1), (h, x)))))


def attend_over_all_heads_on_slots[A: cat.Axis](
    slot_axis: A,
    letter: str,
    exponentiate_scores_of: Callable[[A, str], cat.BroadcastedCategory] = (
        exponentiated_scores),
) -> cat.BroadcastedCategory:
    '''The softmax over one kind of slot and the sink, written for every head at once.

    The query scores against the latent at every slot, the scores are exponentiated, one
    copy is summed into the denominator and the other contracts against the latents, the
    exponentiated sink logit joins the denominator, and the reciprocal of the denominator
    scales the numerator. `exponentiate_scores_of` builds, for a slot axis and its
    einops letter, the expression from the query and the latents to the exponentiated
    scores. A model that scales its scores before the exponential passes its own.
    '''
    scores = cat.Array(R, (h, x, slot_axis))
    latents = cat.Array(R, (x, slot_axis, c))
    return (route((0, 1, 2, 2), (SINK_LOGIT, Q, latents))
            @ (hold(SINK_LOGIT) * exponentiate_scores_of(slot_axis, letter)
               * hold(latents))
            @ route((0, 1, 1, 2), (SINK_LOGIT, scores, latents))
            @ (hold(SINK_LOGIT) * sum_over_slots(letter) * hold(scores)
               * hold(latents))
            @ (add_sink() * hold(scores) * hold(latents))
            @ (over((h, x), reciprocal()) * hold(scores) * hold(latents))
            @ (hold(DENOMINATOR) * weight_slots(letter))
            @ ops.Einops.template('h x, h x c -> h x c'))


def attend_over_all_heads_with_entries() -> cat.Block:
    '''The same core with the two kinds of slot written out as two branches, which is
    what the expansion of the concatenation gives: each branch exponentiates its own
    scores, sums its own denominator and contracts against its own latents, and the two
    denominators and the two numerators are added.'''
    return cat.Block.template(
        route((0, 1, 2, 1, 3, 2, 3), (SINK_LOGIT, Q, WKV, SELECTED_KV))
        @ (hold(SINK_LOGIT) * exponentiated_scores(w, 'w')
           * exponentiated_scores(s, 's') * hold(WKV) * hold(SELECTED_KV))
        @ route((0, 1, 1, 2, 2, 3, 4),
                (SINK_LOGIT, SLOT_SCORES, SELECTED_SCORES, WKV, SELECTED_KV))
        @ (hold(SINK_LOGIT) * sum_over_slots('w') * hold(SLOT_SCORES)
           * sum_over_slots('s') * hold(SELECTED_SCORES)
           * hold(WKV) * hold(SELECTED_KV))
        @ route((0, 1, 3, 2, 4, 5, 6),
                (SINK_LOGIT, DENOMINATOR, SLOT_SCORES, DENOMINATOR, SELECTED_SCORES,
                 WKV, SELECTED_KV))
        @ (hold(SINK_LOGIT) * over((h, x), ops.AdditionOp.template())
           * hold(SLOT_SCORES) * hold(SELECTED_SCORES)
           * hold(WKV) * hold(SELECTED_KV))
        @ (add_sink() * hold(SLOT_SCORES) * hold(SELECTED_SCORES)
           * hold(WKV) * hold(SELECTED_KV))
        @ (over((h, x), reciprocal()) * hold(SLOT_SCORES) * hold(SELECTED_SCORES)
           * hold(WKV) * hold(SELECTED_KV))
        @ route((0, 1, 3, 2, 4),
                (DENOMINATOR, SLOT_SCORES, SELECTED_SCORES, WKV, SELECTED_KV))
        @ (hold(DENOMINATOR) * weight_slots('w') * weight_slots('s'))
        @ (hold(DENOMINATOR) * over((h, x, c), ops.AdditionOp.template()))
        @ ops.Einops.template('h x, h x c -> h x c'),
        title=text.CORE_TITLE, description=text.TWO_BRANCH_DESCRIPTION, fill_color=CORE_COLOUR,
        references=CORE_REFERENCES)


CORE_REFERENCES = (model_lines(639), model_lines(780), kernel_lines(392),
                   kernel_lines(362, 364))
SLOT_CONCATENATION = concatenate_slot_latents()
CONCATENATED_SLOTS = SLOT_CONCATENATION.cod()[0].shape()[1]
CONCATENATED_CORE = attend_over_all_heads_on_slots(CONCATENATED_SLOTS, 't')
WINDOW_CORE = attend_over_all_heads_on_slots(w, 'w')

CORE_ON_CONCATENATED_SLOTS = discovering_broadcasts.discover_broadcast_over_axes(
    cat.Block.template(CONCATENATED_CORE, title=text.CORE_TITLE,
                       description=text.CORE_DESCRIPTION, fill_color=CORE_COLOUR,
                       references=CORE_REFERENCES),
    (h,), 'Core')
CORE_ON_THE_WINDOW = discovering_broadcasts.discover_broadcast_over_axes(
    cat.Block.template(WINDOW_CORE, title=text.CORE_TITLE,
                       description=text.WINDOW_CORE_DESCRIPTION, fill_color=CORE_COLOUR,
                       references=CORE_REFERENCES),
    (h,), 'Core')


def attend_over_all_heads_with_concatenated_slots() -> cat.BroadcastedCategory:
    '''The concatenation of the two kinds of latent followed by the per-head box, with
    the sink logit as an operand rather than as a weight, which is the form the two
    spellings are compared in.

    `concatenation_expansion.expand_concatenations_through_blocks` writes the box out
    over the head axis and then rewrites every consumer of the concatenated axis, and
    what it returns is `attend_over_all_heads_with_entries` under
    `discovering_broadcasts.normalise_for_comparison`. The normalisation is needed
    because writing the box out reads each shared operand through a repeat, one per
    part, which it absorbs into the operations that read it.
    '''
    return ((hold(SINK_LOGIT) * hold(Q) * SLOT_CONCATENATION)
            @ CORE_ON_CONCATENATED_SLOTS.candidate)


def attend_over_window_and_entries() -> cat.BroadcastedCategory:
    '''The same core with this layer's own sink logit supplied as the first operand,
    which is what a mode composes.'''
    return ((sink_logits() * hold(Q) * SLOT_CONCATENATION)
            @ CORE_ON_CONCATENATED_SLOTS.candidate)


def attend_over_window() -> cat.BroadcastedCategory:
    '''The window-only core as one box computed once per head, with the sink logit
    supplied as its first operand. There is one kind of slot, so there is nothing to
    concatenate.'''
    return (sink_logits() * hold(Q) * hold(WKV)) @ CORE_ON_THE_WINDOW.candidate
