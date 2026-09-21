'''The attention core of DeepSeek-V4.1-Flash with the score scale written as an
operator.

Written by Claude Fable 5.1, reasoning effort 80.

The released kernel multiplies every score of a query against a latent by
`|c|^{-1/2}` before the softmax, and adds the exponentiated sink logit to the divisor
with no scale. `notebooks.sota.DeepSeekV41Flash.attention_core` folds the scale into
the query weight. Here the scale is the elementwise map `scale_scores`, between the
score contraction and the exponential, and the sink logit reaches its own exponential
as it does in that module. `rotated_indexer.scale_head_weights` is the other scale the
released model applies and the base model folds into a weight.

`attention_core.attend_over_all_heads_on_slots` writes the core for every head and
takes the exponentiation of the scores as an argument, so the two cores here are that
function given `scaled_exponentiated_scores`. Each is boxed once per head by
`discovering_broadcasts.discover_broadcast_over_axes`, which confirms the box by
expanding it over the head axis and comparing, as the cores of the base model are
confirmed.

    SCALED_CORE_ON_CONCATENATED_SLOTS   the window slots and the selected slots
    SCALED_CORE_ON_THE_WINDOW           the window slots alone, for layers 0 and 1

`attend_over_window_and_entries` and `attend_over_window` supply the layer's own sink
logit and the concatenation of the two kinds of latent, with the domain and the
codomain of the functions of the same name in `attention_core`.

`ARITHMETIC_ROLES` and `ARITHMETIC_REFERENCES` are the rows the explanation tables of
the integrated model take for the scale.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd

import notebooks.sota.DeepSeekV41Flash.attention_core as attention_core
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over
from notebooks.sota.DeepSeekV41Flash.custom_operations import exponential
from notebooks.sota.DeepSeekV41Flash.declared_axes import Q, WKV, c, h, w, x
from notebooks.sota.DeepSeekV41Flash.reference_links import kernel_lines, model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

CORE_BOX = 'Core'
SCORE_SCALE_NAME = '\\lvert c \\rvert^{-1/2} x'
SCALED_CORE_REFERENCES = (*attention_core.CORE_REFERENCES, model_lines(651),
                          kernel_lines(365, 367), kernel_lines(382, 383))


def scale_scores() -> cat.Broadcasted:
    '''The factor `|c|^{-1/2}` the released kernel multiplies every attention score
    by before the softmax.'''
    return ops.Arithmetic.template(
        nm.x * nm.Power.template(c.local_size(), nm.Integer(-1) / nm.Integer(2)),
        name=SCORE_SCALE_NAME)


def scaled_exponentiated_scores[A: cat.Axis](
    slot_axis: A, letter: str) -> cat.BroadcastedCategory:
    '''One kind of slot's scores, multiplied by `|c|^{-1/2}` and exponentiated.'''
    return (attention_core.score_slots(letter)
            @ over((h, x, slot_axis), scale_scores())
            @ over((h, x, slot_axis), exponential()))


def scaled_core_over_every_head[A: cat.Axis](
    slot_axis: A, letter: str, description: str) -> cat.Block:
    return cat.Block.template(
        attention_core.attend_over_all_heads_on_slots(
            slot_axis, letter, scaled_exponentiated_scores),
        title=text.CORE_TITLE, description=description,
        fill_color=attention_core.CORE_COLOUR, references=SCALED_CORE_REFERENCES)


SCALED_CORE_ON_CONCATENATED_SLOTS = discovering_broadcasts.discover_broadcast_over_axes(
    scaled_core_over_every_head(
        attention_core.CONCATENATED_SLOTS, 't', text.SCALED_CORE_DESCRIPTION),
    (h,), CORE_BOX)
SCALED_CORE_ON_THE_WINDOW = discovering_broadcasts.discover_broadcast_over_axes(
    scaled_core_over_every_head(w, 'w', text.SCALED_WINDOW_CORE_DESCRIPTION),
    (h,), CORE_BOX)


def attend_over_window_and_entries() -> cat.BroadcastedCategory:
    '''The scaled core over the window slots and the selected slots, with this layer's
    own sink logit supplied as the first operand.'''
    return ((attention_core.sink_logits() * hold(Q)
             * attention_core.SLOT_CONCATENATION)
            @ SCALED_CORE_ON_CONCATENATED_SLOTS.candidate)


def attend_over_window() -> cat.BroadcastedCategory:
    '''The scaled core over the window slots alone, with this layer's own sink logit
    supplied as the first operand.'''
    return ((attention_core.sink_logits() * hold(Q) * hold(WKV))
            @ SCALED_CORE_ON_THE_WINDOW.candidate)


ARITHMETIC_ROLES: dict[str, str] = {
    SCORE_SCALE_NAME: (
        text.SCORE_SCALE_ROLE),
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    SCORE_SCALE_NAME: (model_lines(651), kernel_lines(365, 367)),
}
