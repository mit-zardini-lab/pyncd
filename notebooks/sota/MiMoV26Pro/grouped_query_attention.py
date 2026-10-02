# Claude Opus 5.5 (1M context), effort 40.
'''The grouped-query attention of MiMo-V2.6-Pro, which every layer runs over a read of
the earlier tokens.

The attention is `MiMoV2Attention` of the reference. It projects the hidden state of
every token onto 128 query heads, 8 key heads and 8 value heads with one weight,
`qkv_proj`, and cuts the result with `torch.split`. A linear map followed by a cut of
its results is one linear map per part, per `obsidian/06-practice/Representing
Models.md`, and the reference cuts a query head and a key head a second time, into the
64 channels the rotary embedding turns and the 128 it passes through. The expression
therefore holds one weight per part:

    W^{Qr}  onto the 64 turned channels of every query head
    W^{Qn}  onto the 128 channels of every query head that are not turned
    W^{Kr}  onto the 64 turned channels of every key head
    W^{Kn}  onto the 128 channels of every key head that are not turned
    W^{V}   onto the 128 channels of every value head

The reference holds the 64 turned channels of a head as two halves, the real parts of
the 32 pairs and then their imaginary parts. `W^{Qr}` and `W^{Kr}` write the halves into
their rows as the axis `c`, per the ruling on groupings, and the rotary box of
`rotary_embedding` reads them. The turned channels come first in a head, and
`aops.ConcatenateAxes` joins the two parts onto `a`, which is the `torch.cat` of the
reference. The reference multiplies every
value by `attention_value_scale` before the rotary embedding and before its cache, so
the scale stands on the values here, after `W^{V}`.

A query reads the earlier tokens through one of two views. A full attention layer reads
every token at or before the query, through the view named Back, and a sliding window
layer reads the 128 tokens ending at the query, through the view named Window. Slot
`i_w` of query `i_x` reads token `i_x - i_w`, so slot 0 is the query's own token, and a
slot counting back past the first token reads the universal unit, which
`mark_sparse_domains.mark_sparse_domain` marks as the axis `w|x`. The reference states
the window as the mask of `create_sliding_window_causal_mask` in `transformers`, which
keeps key `k` for query `q` where `q - 128 < k <= q`, 128 keys with the query's own.
Both views read the keys of the 8 key heads and the values of the 8 value heads, and
every query head of a group reads the key-value head of its group.

The core of a full attention layer is a softmax over the distances. The core of a
sliding window layer adds a learned logit per query head, `attention_sink_bias`, to the
softmax over the window slots: the reference concatenates the logit to the scores of
every query, takes the softmax over the slots and the logit, and drops the column of the
logit. The expression writes that softmax out, as `notebooks/sota/DeepSeekV41Flash/`
writes the sink of its attention core: the exponential of every score, their sum, the
exponential of the logit added to the sum, and each exponential divided by the total.
Each core is one box computed once per query, confirmed by
`discovering_broadcasts.discover_broadcast_over_axes`. The exponential of the logit is
the same for every query, so the mode computes it once and hands it to the box. Written
inside the box, it would be computed once per query when the box is expanded, and the
confirmation refuses that.

    queries(box)            STATE[x, m] -> QUERIES[x, h, g, a]
    keys(box)               STATE[x, m] -> KEYS[x, h, a]
    values()                STATE[x, m] -> VALUES[x, h, u]
    READ_BACK, READ_WINDOW  the two views, over the keys and the values of every head
    CORE, SINK_CORE         the two cores, computed once per query
    output_projection()     R[x, h, g, u] -> STATE[x, m]
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    exponential, reciprocal, weights)
from notebooks.sota.MiMoV26Pro.declared_axes import (
    QUERIES, R, EXPONENTIATED_SINKS, STATE, a, c, g, h, m, n, p, r, t, u, w, x)
from notebooks.sota.MiMoV26Pro.reference_links import (
    SLIDING_WINDOW_MASK_REFERENCES, modeling_lines)
from notebooks.sota.MiMoV26Pro.released_constants import VALUE_SCALE
from notebooks.sota.MiMoV26Pro.rotary_embedding import rotate_every_head
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text

QUERY_COLOUR = '#E8DFF0'
KEY_COLOUR = '#DDE8D6'
VALUE_COLOUR = '#F5E6D3'
CORE_COLOUR = '#C5BEDF'
CORE_BOX = 'Core'
SINK_CORE_BOX = 'CoreS'
BACK_VIEW_NAME = '\\mathrm{Back}'
WINDOW_VIEW_NAME = '\\mathrm{Window}'
SCORE_SCALE_NAME = 'x / \\sqrt{\\lvert a \\rvert}'
VALUE_SCALE_NAME = '\\lambda x'
SINK_NAME = '\\mathrm{sink}'

QUERY_REFERENCES = (modeling_lines(278, 283), modeling_lines(370, 378),
                    modeling_lines(305, 310))
KEY_REFERENCES = (modeling_lines(278, 283), modeling_lines(370, 379),
                  modeling_lines(305, 310))
VALUE_REFERENCES = (modeling_lines(278, 283), modeling_lines(370, 380),
                    modeling_lines(302, 303))
CORE_REFERENCES = (modeling_lines(63, 68), modeling_lines(71, 102),
                   modeling_lines(261))
SINK_REFERENCES = (modeling_lines(89, 97), modeling_lines(268, 275),
                   modeling_lines(326, 330), modeling_lines(338, 339))
OUTPUT_REFERENCES = (modeling_lines(288), modeling_lines(354, 355))


def queries(rotation_box: cat.Broadcasted) -> cat.Block:
    '''`STATE[x, m] -> QUERIES[x, h, g, a]`: the turned channels of every query head,
    turned at the position of the token once per head, and the channels passed
    through, joined with the turned channels first.'''
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ (((x >> ops.Linear.template((m,), (h, g, c, t), 'W^{Qr}'))
            @ rotate_every_head(rotation_box, (h, g)))
           * (x >> ops.Linear.template((m,), (h, g, n), 'W^{Qn}')))
        @ aops.ConcatenateAxes.template(((x, h, g, p), (x, h, g, n)), concatenated=a),
        title=text.QUERY_TITLE, fill_color=QUERY_COLOUR,
        description=text.QUERY_DESCRIPTION, references=QUERY_REFERENCES)


def keys(rotation_box: cat.Broadcasted) -> cat.Block:
    '''`STATE[x, m] -> KEYS[x, h, a]`: the key of every key-value head, built as a query
    head is built.'''
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ (((x >> ops.Linear.template((m,), (h, c, t), 'W^{Kr}'))
            @ rotate_every_head(rotation_box, (h,)))
           * (x >> ops.Linear.template((m,), (h, n), 'W^{Kn}')))
        @ aops.ConcatenateAxes.template(((x, h, p), (x, h, n)), concatenated=a),
        title=text.KEY_TITLE, fill_color=KEY_COLOUR,
        description=text.KEY_DESCRIPTION, references=KEY_REFERENCES)


def scale_values() -> cat.Broadcasted:
    '''Every value multiplied by the value scale.'''
    return ops.Arithmetic.template(nm.x * VALUE_SCALE, name=VALUE_SCALE_NAME)


def values() -> cat.Block:
    '''`STATE[x, m] -> VALUES[x, h, u]`: the value of every key-value head, scaled.'''
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (h, u), 'W^{V}'))
        @ over((x, h, u), scale_values()),
        title=text.VALUE_TITLE, fill_color=VALUE_COLOUR,
        description=text.VALUE_DESCRIPTION, references=VALUE_REFERENCES)


def read_back_view(slots: cat.Axis, name: str) -> sc.StrideMorphism:
    '''The read of token `i_x - i_s` at slot `i_s` of each query `i_x`.'''
    return sc.StrideMorphism(
        _dom=(x, slots),
        _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
        name=fd.DynamicName(name))


READ_BACK = mark_sparse_domains.mark_sparse_domain(read_back_view(r, BACK_VIEW_NAME))
'''The read of every token at or before each query, with the distance axis marked `r|x`,
live where `i_x - i_r >= 0`.'''

READ_WINDOW = mark_sparse_domains.mark_sparse_domain(
    read_back_view(w, WINDOW_VIEW_NAME))
'''The read of the `|w|` tokens ending at each query, with the slot axis marked `w|x`,
live where `i_x - i_w >= 0`.'''

distances = READ_BACK._dom[1]
window_slots = READ_WINDOW._dom[1]


def read_through[A: cat.Axis](read: sc.StrideMorphism, name: str,
                              rest: tuple[A, ...]) -> cat.Broadcasted:
    '''An array over the tokens read through `read`, times the identity on `rest`.'''
    return ops.View.template(
        reindexing=(read, cat.ProdObject(rest).identity()), name=name)


def attend_over_every_query(slots: cat.Axis) -> cat.Block:
    '''The softmax of every query head over `slots`, written out over every query: the
    score of the query against the key of its group at every slot, the scale
    `1 / \\sqrt{|a|}`, the softmax over the slots and the sum of the values under it.'''
    selected_keys = (x, slots, h, a)
    selected_values = (x, slots, h, u)
    scores = (x, h, g, slots)
    return cat.Block.template(
        (hold(QUERIES) * hold(cat.Array(R, selected_keys))
         * hold(cat.Array(R, selected_values)))
        @ (einops_simplification.einsum(((x, h, g, a), selected_keys), scores, R)
           * hold(cat.Array(R, selected_values)))
        @ (over(scores, scale_scores()) * hold(cat.Array(R, selected_values)))
        @ (over((x, h, g), ops.SoftMax.template())
           * hold(cat.Array(R, selected_values)))
        @ einops_simplification.einsum((scores, selected_values), (x, h, g, u), R),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, references=CORE_REFERENCES)


def scale_scores() -> cat.Broadcasted:
    return ops.Arithmetic.template(
        nm.x / nm.SquareRoot(a.local_size()), name=SCORE_SCALE_NAME)


def add_exponentiated_sink(total: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    '''The exponential of the logit of every query head added to the sum of the
    exponentials of its scores. The logit is one number per query head, so the
    addition reads it at the heads alone.'''
    tiled = cat.WeaveMode.TILED
    return cat.Broadcasted(
        operator=ops.AdditionOp(),
        input_weaves=(cat.Weave(R, (tiled, tiled)),
                      cat.Weave(R, (tiled, tiled, tiled))),
        output_weaves=(cat.Weave(R, (tiled, tiled, tiled)),),
        reindexings=(route((1, 2), total), route((0, 1, 2), total)))


def attend_with_a_sink_over_every_query(slots: cat.Axis) -> cat.Block:
    '''The softmax of every query head over `slots` and the logit of its head, written
    out over every query. The first operand is the exponential of the logit of every
    query head, which is the same for every query and is computed once, outside the
    box. The exponentials of the scores are summed, the exponential of the logit is
    added to the sum, every exponential of a score is divided by the total, and the
    quotients weight the values. The quotient of the logit is not computed, which is
    the column the reference drops.'''
    selected_keys = (x, slots, h, a)
    selected_values = (x, slots, h, u)
    scores = (x, h, g, slots)
    total = (x, h, g)
    values_held = hold(cat.Array(R, selected_values))
    exponentials = cat.Array(R, scores)
    return cat.Block.template(
        (hold(EXPONENTIATED_SINKS) * hold(QUERIES) * hold(cat.Array(R, selected_keys))
         * values_held)
        @ (hold(EXPONENTIATED_SINKS)
           * (einops_simplification.einsum(((x, h, g, a), selected_keys), scores, R)
              @ over(scores, scale_scores()) @ over(scores, exponential()))
           * values_held)
        @ route((0, 1, 1, 2),
                (EXPONENTIATED_SINKS, exponentials, cat.Array(R, selected_values)))
        @ (hold(EXPONENTIATED_SINKS)
           * einops_simplification.einsum((scores,), total, R)
           * hold(exponentials) * values_held)
        @ (add_exponentiated_sink(total) * hold(exponentials) * values_held)
        @ (over(total, reciprocal()) * hold(exponentials) * values_held)
        @ (einops_simplification.einsum((total, scores), scores, R) * values_held)
        @ einops_simplification.einsum((scores, selected_values), (x, h, g, u), R),
        title=text.SINK_CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.SINK_CORE_DESCRIPTION,
        references=(*CORE_REFERENCES, *SINK_REFERENCES))


CORE = discovering_broadcasts.discover_broadcast_over_axes(
    attend_over_every_query(distances), (x,), CORE_BOX)
SINK_CORE = discovering_broadcasts.discover_broadcast_over_axes(
    attend_with_a_sink_over_every_query(window_slots), (x,), SINK_CORE_BOX)


def output_projection() -> cat.BroadcastedCategory:
    '''`R[x, h, g, u] -> STATE[x, m]`: the values of the 128 query heads mapped back
    onto the hidden width.'''
    return x >> ops.Linear.template((h, g, u), (m,), 'W^{O}')


def exponentiated_sink_logits() -> cat.BroadcastedCategory:
    '''The exponential of the learned logit of every query head,
    `attention_sink_bias`.'''
    return weights(SINK_NAME, (h, g)) @ over((h, g), exponential())


READ_REFERENCES = (modeling_lines(84, 87), modeling_lines(1642, 1657),
                   *SLIDING_WINDOW_MASK_REFERENCES)
'''The mask added to the scores in the eager attention, the two masks built by the
model, and the lines of `transformers` that decide what the window reads.'''
