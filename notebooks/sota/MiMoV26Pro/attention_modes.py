# Claude Opus 5.5 (1M context), effort 40.
'''The two attention modes of MiMo-V2.6-Pro, full attention and sliding window
attention.

`hybrid_layer_pattern` in the configuration gives every layer one of two modes, 0 for
full attention and 1 for a sliding window. `MiMoV2DecoderLayer` builds the attention of
a layer with `is_swa` set from the pattern, and the two modes differ in four ways:

    full attention      every token at or before the query, the table of turns at
                        the base beta, and no sink
    sliding window      the 128 tokens ending at the query, the table of turns at the
                        base beta_w, and the learned logit of every query head

The configuration declares the same numbers of heads and the same head widths for the
two modes. Each mode reads the normalised hidden state and returns the output of the
attention on the hidden width, and each is one box.

    FULL        full attention, the box `Full`
    WINDOW      sliding window attention, the box `SWA`
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat

from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, route
from notebooks.sota.MiMoV26Pro.declared_axes import QUERIES, R, STATE, a, h, u, x
from notebooks.sota.MiMoV26Pro.grouped_query_attention import (
    BACK_VIEW_NAME, CORE, READ_BACK, READ_REFERENCES, READ_WINDOW, SINK_CORE,
    WINDOW_VIEW_NAME, exponentiated_sink_logits, keys, output_projection, queries,
    read_through, values, window_slots)
from notebooks.sota.MiMoV26Pro.reference_links import (
    checkpoint_config_lines, modeling_lines)
from notebooks.sota.MiMoV26Pro.rotary_embedding import (
    ROTATE_FULL_ATTENTION_CHANNELS, ROTATE_SLIDING_WINDOW_CHANNELS)
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text

FULL_BOX = 'Full'
WINDOW_BOX = 'SWA'
FULL_COLOUR = '#C5BEDF'
WINDOW_COLOUR = '#B8D8CE'

MODE_REFERENCES = (modeling_lines(220, 391), modeling_lines(400, 404),
                   checkpoint_config_lines(2, 3), checkpoint_config_lines(47, 118),
                   checkpoint_config_lines(368, 374))


def full_attention() -> cat.Block:
    '''Every query head attends to every token at or before its query.'''
    return cat.Block.template(
        route((0, 0, 0), (STATE,))
        @ (queries(ROTATE_FULL_ATTENTION_CHANNELS)
           * (keys(ROTATE_FULL_ATTENTION_CHANNELS)
              @ read_through(READ_BACK, BACK_VIEW_NAME, (h, a)))
           * (values() @ read_through(READ_BACK, BACK_VIEW_NAME, (h, u))))
        @ CORE.candidate
        @ output_projection(),
        title=text.FULL_TITLE, fill_color=FULL_COLOUR,
        description=text.FULL_ATTENTION_DESCRIPTION,
        references=(*MODE_REFERENCES, *READ_REFERENCES))


def sliding_window_attention() -> cat.Block:
    '''Every query head attends to the 128 tokens ending at its query and to the learned
    logit of its head.'''
    window_keys = cat.Array(R, (x, window_slots, h, a))
    window_values = cat.Array(R, (x, window_slots, h, u))
    return cat.Block.template(
        route((0, 0, 0), (STATE,))
        @ (queries(ROTATE_SLIDING_WINDOW_CHANNELS)
           * (keys(ROTATE_SLIDING_WINDOW_CHANNELS)
              @ read_through(READ_WINDOW, WINDOW_VIEW_NAME, (h, a)))
           * (values() @ read_through(READ_WINDOW, WINDOW_VIEW_NAME, (h, u))))
        @ (exponentiated_sink_logits() * hold(QUERIES) * hold(window_keys)
           * hold(window_values))
        @ SINK_CORE.candidate
        @ output_projection(),
        title=text.WINDOW_TITLE, fill_color=WINDOW_COLOUR,
        description=text.WINDOW_ATTENTION_DESCRIPTION,
        references=(*MODE_REFERENCES, *READ_REFERENCES))


FULL = boxed(full_attention(), FULL_BOX)
WINDOW = boxed(sliding_window_attention(), WINDOW_BOX)
