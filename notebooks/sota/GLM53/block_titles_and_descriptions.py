# Claude Opus 5.5 (1M context), effort 40.
'''Every block title, block description and inspection-box sentence of GLM-5.3.

A title heads a block in a figure and a description fills its inspection box. Both are
prose a reader sees. The wording is held in `block_titles_and_descriptions.json` beside
this module, one entry per name, and `BlockTitlesAndDescriptions` names every entry as
a field, so that a module reads `text.CORE_TITLE` with the name checked and typed, and
the file is edited, copied and translated with no Python in the way. The modules under
`notebooks/sota/GLM53/` import `TEXT` as `text`.

The file is read by `utilities/wording_json.py`, which states its form. A sentence
shared by several descriptions is one entry, and each description holding it names it
as `$NAME`. A formula stays beside the algebra that builds it, because a formula is
written from the axes of its morphism. The entries are grouped by mechanism in the
order the model is built, and the fields below follow the same order.
`notebooks/sota/DeepSeekV41Flash/block_titles_and_descriptions.py` is the pattern.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class BlockTitlesAndDescriptions:
    ROTARY_TITLE: str
    INDEXER_ROTARY_TITLE: str
    PAIRS_OF_THE_TURNED_CHANNELS: str
    TURN_BY_THE_TOKEN: str
    SCORE_BY_DISTANCE: str
    TABLE_IN_THE_BOX: str
    HALVES_OF_THE_REFERENCE: str
    QUERY_KEY_ROTATION_DESCRIPTION: str
    INDEXER_ROTATION_DESCRIPTION: str
    CHANNEL_JOIN_DESCRIPTION: str
    CHANNEL_CUT_DESCRIPTION: str
    PAIRS_AS_COMPLEX_DESCRIPTION: str
    DECOMPLEX_DESCRIPTION: str
    ROTARY_TABLE_ROLE: str

    SCORE_TITLE: str
    INDEXER_TITLE: str
    SCORE_EVERY_QUERY_DESCRIPTION: str
    INDEXER_DESCRIPTION: str

    QUERY_TITLE: str
    KEYS_AND_VALUES_TITLE: str
    QUERY_STEPS_SENTENCE: str
    QUERY_PATH_DESCRIPTION: str
    QUERY_WITHOUT_LOW_RANK_DESCRIPTION: str
    KEYS_AND_VALUES_DESCRIPTION: str

    GATHER_TITLE: str
    GATHER_STEPS_SENTENCE: str
    GATHER_KEYS_DESCRIPTION: str
    GATHER_VALUES_DESCRIPTION: str
    CORE_TITLE: str
    CORE_DESCRIPTION: str

    FULL_TITLE: str
    SHARED_TITLE: str
    FULL_STEPS_SENTENCE: str
    FULL_ATTENTION_DESCRIPTION: str
    FULL_ATTENTION_PUBLISHING_DESCRIPTION: str
    SHARED_ATTENTION_DESCRIPTION: str

    DENSE_MLP_TITLE: str
    SWIGLU_STEPS_SENTENCE: str
    DENSE_MLP_DESCRIPTION: str
    GATE_TITLE: str
    EXPERTS_TITLE: str
    SHARED_EXPERT_TITLE: str
    MIXTURE_TITLE: str
    GATE_DESCRIPTION: str
    ROUTED_EXPERTS_DESCRIPTION: str
    SHARED_EXPERT_DESCRIPTION: str
    MIXTURE_DESCRIPTION: str

    RESIDUAL_TITLE: str
    RESIDUAL_DESCRIPTION: str
    DENSE_LAYER_TITLE: str
    SHARED_LAYER_TITLE: str
    FIRST_GROUP_TITLE: str
    GROUP_TITLE: str
    DENSE_LAYERS_DESCRIPTION: str
    SHARED_LAYERS_DESCRIPTION: str
    FIRST_GROUP_DESCRIPTION: str
    REPEATED_GROUPS_DESCRIPTION: str

    EMBEDDING_TITLE: str
    OUTPUT_TITLE: str
    EMBEDDING_BLOCK_DESCRIPTION: str
    OUTPUT_DESCRIPTION: str

    TOP_K_TOKENS_DESCRIPTION: str
    TOP_K_EXPERTS_DESCRIPTION: str
    SELECT_DESCRIPTION: str
    INDEX_SELECT_DESCRIPTION: str
    EMBEDDING_DESCRIPTION: str
    BACK_VIEW_DESCRIPTION: str
    DIAGONAL_VIEW_DESCRIPTION: str

    QUERY_DOWN_PROJECTION_ROLE: str
    QUERY_UNTURNED_PROJECTION_ROLE: str
    QUERY_TURNED_PROJECTION_ROLE: str
    LATENT_PROJECTION_ROLE: str
    KEY_TURNED_PROJECTION_ROLE: str
    KEY_EXPANSION_ROLE: str
    VALUE_EXPANSION_ROLE: str
    OUTPUT_PROJECTION_ROLE: str
    INDEXER_QUERY_PROJECTION_ROLE: str
    INDEXER_KEY_PROJECTION_ROLE: str
    INDEXER_HEAD_WEIGHTS_ROLE: str
    ROUTER_ROLE: str
    CORRECTION_BIAS_ROLE: str
    EXPERT_GATE_PROJECTION_ROLE: str
    EXPERT_UP_PROJECTION_ROLE: str
    EXPERT_DOWN_PROJECTION_ROLE: str
    SHARED_GATE_PROJECTION_ROLE: str
    SHARED_UP_PROJECTION_ROLE: str
    SHARED_DOWN_PROJECTION_ROLE: str
    DENSE_GATE_PROJECTION_ROLE: str
    DENSE_UP_PROJECTION_ROLE: str
    DENSE_DOWN_PROJECTION_ROLE: str
    OUTPUT_HEAD_ROLE: str

    ATTENTION_SCALE_ROLE: str
    INDEXER_SCORE_SCALE_ROLE: str
    INDEXER_HEAD_SCALE_ROLE: str
    ROUTE_SCALE_ROLE: str
    ROUTER_SIGMOID_ROLE: str
    SILU_ROLE: str
    RECTIFIER_ROLE: str
    INDICATOR_ROLE: str

    ROUNDING_KERNEL_DESCRIPTION: str
    WEIGHT_QUANTISATION_SENTENCE: str
    FILE_QUANTISATION_SENTENCE: str
    EMBEDDING_TABLE_ROLE: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> BlockTitlesAndDescriptions:
        return wording_json.load_dataclass(cls, path)


TEXT = BlockTitlesAndDescriptions.load()
