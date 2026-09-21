# Claude Opus 5, effort high.
'''DSpark placed in the integrated model: the tap that writes a layer's stream mean onto
the tape, the grab that reads the three means back, and the box the model ends in.

Written by Claude Fable 5.1, reasoning effort 80. The mechanism was moved to
`notebooks.sota.DeepSeekV41Flash.dspark_draft_chain` and
`notebooks.sota.DeepSeekV41Flash.gumbel_max_sampler` by Claude Opus 5, effort high, on
2026-09-19, and this module holds the tape wiring and the placement alone.

    tap_stream_mean, drop_stream_mean
                            the mean over the four streams of the residual entering a
                            layer, dropped onto `SLOT_STREAM_MEAN` at the member a loop
                            counter names
    grab_stream_means       the three members read back after the forty layers
    dspark_block, DSPARK    the block from the backbone's probabilities to the five
                            drafts and the five confidences, and its box
    probabilities_with_drafts
                            the box beside a copy of the probabilities it reads

`divided_layer_stack` places `tap_stream_mean` at the head of the body of the last three
Reuse layers, which are layers 37, 38 and 39, and hands it the counter of that repeated
block. `grab_stream_means` reads the members 0, 1 and 2, so the counter has to run from
zero.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import para.data_structure.Para as Para

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    hold, para_boxed, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import COLLAPSE, X, state
from notebooks.sota.DeepSeekV41Flash.dspark_draft_chain import (
    DSPARK_COLOUR, PART_COLOUR, PROBABILITIES, TARGET_LAYERS, draft_five_tokens,
    mean_over_streams, product_of)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import (
    SLOT_STREAM_MEAN)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

DSPARK_BOX = 'DSpark'


def drop_stream_mean(counter: nm.Numeric) -> cat.Block:
    '''The mean over the streams of the residual, written to the member of
    `SLOT_STREAM_MEAN` that `counter` names.'''
    return cat.Block.template(
        mean_over_streams()
        @ Para.LoopDrop(tape=SLOT_STREAM_MEAN, size=state, index=counter),
        title=text.TAP_TITLE, fill_color=PART_COLOUR,
        formula=('\\mathrm{mean}[i_{x}, i_{m}] = \\frac{1}{\\lvert n \\rvert} '
                 '\\sum_{i_{n} \\in n} X[i_{x}, i_{n}, i_{m}]'),
        description=text.DROP_STREAM_MEAN_DESCRIPTION,
        references=(model_lines(1264, 1266), inference_config_lines(9)))


def tap_stream_mean(counter: nm.Numeric) -> cat.BroadcastedCategory:
    '''The collapse vector and the residual passed on unchanged, with the mean over the
    streams of the residual dropped at the member `counter` names. `counter` is the
    counter of the repeated block the tap stands in, and it runs from zero.'''
    return (route((0, 1, 1), (COLLAPSE, X))
            @ (hold(COLLAPSE) * hold(X) * drop_stream_mean(counter)))


def grab_stream_means() -> cat.Morphism:
    '''The three members the taps wrote, in the order of the layers that wrote them.'''
    return product_of(
        Para.LoopGrab(tape=SLOT_STREAM_MEAN, size=state, index=nm.Integer(member))
        for member in range(TARGET_LAYERS))


def dspark_block() -> cat.Block:
    '''From the backbone's probabilities to the five drafts and the five confidences.
    The three stream means arrive from the tape.'''
    return cat.Block.template(
        (hold(PROBABILITIES) * grab_stream_means()) @ draft_five_tokens(),
        title=text.DSPARK_TITLE, fill_color=DSPARK_COLOUR,
        formula=('g = \\mathrm{draft}(\\mathrm{main}, \\mathrm{token}_{0}), \\quad '
                 '\\mathrm{token}_{0} = \\mathrm{sample}(p[\\lvert x \\rvert - 1])'),
        description=text.DSPARK_BLOCK_DESCRIPTION,
        references=(model_lines(129, 130), model_lines(1269, 1270),
                    model_lines(1274, 1282), model_lines(1128, 1156),
                    inference_config_lines(6, 12)))


DSPARK = para_boxed(dspark_block(), DSPARK_BOX)


def probabilities_with_drafts() -> cat.Morphism:
    '''The probabilities passed on beside the five drafts and the five confidences
    DSpark reads off them, which is what the whole model ends in.'''
    return route((0, 0), (PROBABILITIES,)) @ (hold(PROBABILITIES) * DSPARK)
