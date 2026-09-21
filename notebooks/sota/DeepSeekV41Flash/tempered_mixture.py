# Claude Opus 5 (1M context), effort high.
'''The mixture of DeepSeek-V4.1-Flash computed once per token, fed the modality of
every token from the tape.

`notebooks.sota.DeepSeekV41Flash.clamped_mixture_of_experts` states the mixture of one
token, with the SwiGLU clamps, the router's temperature, the correction bias chosen by
the token's modality and the epsilon of the gate normalisation. This module broadcasts
that body over the tokens and wires the modalities in.

    MIXTURE_BODY, MIXTURE   the mixture of one token, and the box that computes it once
                            per token from the hidden states and the modalities
    MIXTURE_SUBLAYER_BODY   the mixture fed its modalities from the tape, which has the
                            hidden state alone as its domain and codomain

A sublayer body of `mhc_with_epsilons.mhc_sublayer` reads the hidden state alone, so the
modality of every token reaches the mixture through `SLOT_MODALITY`. The mixture is an
`ops.BlockOperator` and the grab stands beside it.
`para.algebra.para_sparse_expansion.expand_sparse_onto_tape` returns a
`para.data_structure.ParaWrap` as it stands, so the router's selection would stay
compressed inside a `para.data_structure.ParaBlockOperator`, and the pass does enter an
`ops.BlockOperator`. For a figure,
`notebooks.display.tape_presentation.wrap_inside_boxes` rebuilds a plain box that a grab
feeds with the tape at its port and as a seed inside its body.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import para.data_structure.Para as Para

from notebooks.sota.DeepSeekV41Flash.clamped_mixture_of_experts import (
    MODALITY_OF_EVERY_TOKEN, mix_one_token)
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over
from notebooks.sota.DeepSeekV41Flash.declared_axes import state, x
from notebooks.sota.DeepSeekV41Flash.mixture_of_experts import MIXTURE_BOX
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import (
    SLOT_MODALITY)


MIXTURE_BODY = mix_one_token()


def mix_over_all_tokens() -> cat.BroadcastedCategory:
    '''The same mixture written out over every token, which the box is confirmed
    against. It lifts the body the box holds, because `cat.Block.template` mints a
    fresh tag per call and the comparison reads the tags of the expert blocks.'''
    return over((x,), MIXTURE_BODY)


MIXTURE = discovering_broadcasts.broadcast_block_over_axes(
    MIXTURE_BODY, (x,), ((0,), (0,)), MIXTURE_BOX)
MIXTURE_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    mix_over_all_tokens(), MIXTURE)

MIXTURE_SUBLAYER_BODY = (
    (hold(state) * Para.Grab(tape=SLOT_MODALITY, size=MODALITY_OF_EVERY_TOKEN))
    @ MIXTURE)
