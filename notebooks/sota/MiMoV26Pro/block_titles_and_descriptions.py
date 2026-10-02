# Claude Opus 5.5 (1M context), effort 40.
'''Every block title, block description and inspection-box sentence of MiMo-V2.6-Pro.

A title heads a block in a figure and a description fills its inspection box. Both are
prose a reader sees. The wording is held in `block_titles_and_descriptions.json` beside
this module, one entry per name, and `BlockTitlesAndDescriptions` names every entry as
a field, so that a module reads `text.CORE_TITLE` with the name checked and typed, and
the file is edited, copied and translated with no Python in the way. The modules under
`notebooks/sota/MiMoV26Pro/` import `TEXT` as `text`.

The file is read by `utilities/wording_json.py`, which states its form. A sentence
shared by several descriptions is one entry, and each description holding it names it
as `$NAME`. A formula stays beside the algebra that builds it, because a formula is
written from the axes of its morphism. The entries are grouped by mechanism in the
order the model is built, and the fields below follow the same order.
`notebooks/sota/GLM53/block_titles_and_descriptions.py` is the pattern.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).parent / 'block_titles_and_descriptions.json'


@dataclass(frozen=True)
class BlockTitlesAndDescriptions:
    ROTARY_TITLE: str
    WINDOW_ROTARY_TITLE: str
    PAIRS_OF_THE_TURNED_CHANNELS: str
    TURN_BY_THE_TOKEN: str
    SCORE_BY_DISTANCE: str
    TABLE_IN_THE_BOX: str
    HALVES_OF_THE_REFERENCE: str
    FULL_ROTATION_DESCRIPTION: str
    WINDOW_ROTATION_DESCRIPTION: str
    PAIRS_MERGE_DESCRIPTION: str
    CHANNEL_JOIN_DESCRIPTION: str
    PAIRS_AS_COMPLEX_DESCRIPTION: str
    DECOMPLEX_DESCRIPTION: str
    ROTARY_TABLE_ROLE: str

    QUERY_TITLE: str
    KEY_TITLE: str
    VALUE_TITLE: str
    ONE_WEIGHT_PER_PART: str
    QUERY_DESCRIPTION: str
    KEY_DESCRIPTION: str
    VALUE_DESCRIPTION: str

    CORE_TITLE: str
    SINK_CORE_TITLE: str
    SCORE_STEPS_SENTENCE: str
    GROUPS_OF_HEADS_SENTENCE: str
    CORE_DESCRIPTION: str
    SINK_CORE_DESCRIPTION: str

    FULL_TITLE: str
    WINDOW_TITLE: str
    FULL_ATTENTION_DESCRIPTION: str
    WINDOW_ATTENTION_DESCRIPTION: str

    DENSE_MLP_TITLE: str
    SWIGLU_STEPS_SENTENCE: str
    DENSE_MLP_DESCRIPTION: str
    GATE_TITLE: str
    EXPERTS_TITLE: str
    MIXTURE_TITLE: str
    GATE_DESCRIPTION: str
    ROUTED_EXPERTS_DESCRIPTION: str
    MIXTURE_DESCRIPTION: str

    RESIDUAL_TITLE: str
    RESIDUAL_DESCRIPTION: str
    FIRST_LAYER_TITLE: str
    WINDOW_LAYERS_TITLE: str
    FIRST_GROUP_TITLE: str
    LONG_GROUPS_TITLE: str
    LAST_GROUPS_TITLE: str
    FIRST_LAYER_DESCRIPTION: str
    WINDOW_LAYERS_DESCRIPTION: str
    FIRST_GROUP_DESCRIPTION: str
    LONG_GROUPS_DESCRIPTION: str
    LAST_GROUPS_DESCRIPTION: str

    EMBEDDING_TITLE: str
    OUTPUT_TITLE: str
    EMBEDDING_BLOCK_DESCRIPTION: str
    OUTPUT_DESCRIPTION: str

    TOP_K_EXPERTS_DESCRIPTION: str
    SELECT_DESCRIPTION: str
    EMBEDDING_DESCRIPTION: str
    BACK_VIEW_DESCRIPTION: str
    WINDOW_VIEW_DESCRIPTION: str
    DIAGONAL_VIEW_DESCRIPTION: str

    QUERY_TURNED_PROJECTION_ROLE: str
    QUERY_UNTURNED_PROJECTION_ROLE: str
    KEY_TURNED_PROJECTION_ROLE: str
    KEY_UNTURNED_PROJECTION_ROLE: str
    VALUE_PROJECTION_ROLE: str
    OUTPUT_PROJECTION_ROLE: str
    SINK_ROLE: str
    ROUTER_ROLE: str
    CORRECTION_BIAS_ROLE: str
    EXPERT_GATE_PROJECTION_ROLE: str
    EXPERT_UP_PROJECTION_ROLE: str
    EXPERT_DOWN_PROJECTION_ROLE: str
    DENSE_GATE_PROJECTION_ROLE: str
    DENSE_UP_PROJECTION_ROLE: str
    DENSE_DOWN_PROJECTION_ROLE: str
    OUTPUT_HEAD_ROLE: str

    ATTENTION_SCALE_ROLE: str
    VALUE_SCALE_ROLE: str
    ROUTER_SIGMOID_ROLE: str
    SILU_ROLE: str
    INDICATOR_ROLE: str
    RECIPROCAL_ROLE: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> BlockTitlesAndDescriptions:
        return wording_json.load_dataclass(cls, path)


TEXT = BlockTitlesAndDescriptions.load()
