# Claude Opus 5.5 (1M context), effort 40.
'''Every block title, block description and inspection-box sentence of Kimi K3.

A title heads a block in a figure and a description fills its inspection box. Both are
prose a reader sees. The wording is held in `block_titles_and_descriptions.json` beside
this module, one entry per name, and `BlockTitlesAndDescriptions` names every entry as
a field, so that a module reads `text.CORE_TITLE` with the name checked and typed. The
modules under `notebooks/sota/KimiK3/` import `TEXT` as `text`.

The file is read by `utilities/wording_json.py`, which states its form.
`notebooks/sota/GLM53/block_titles_and_descriptions.py` is the pattern.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class BlockTitlesAndDescriptions:
    SITU_STEPS_SENTENCE: str
    SITU_BOUND_SENTENCE: str
    DENSE_MLP_TITLE: str
    DENSE_MLP_DESCRIPTION: str
    GATE_TITLE: str
    GATE_DESCRIPTION: str
    EXPERTS_TITLE: str
    ROUTED_EXPERTS_DESCRIPTION: str
    LATENT_TITLE: str
    LATENT_DESCRIPTION: str
    SHARED_EXPERT_TITLE: str
    SHARED_EXPERT_DESCRIPTION: str
    MIXTURE_TITLE: str
    MIXTURE_DESCRIPTION: str
    QUERY_TITLE: str
    QUERY_PATH_DESCRIPTION: str
    KEYS_AND_VALUES_TITLE: str
    KEYS_AND_VALUES_DESCRIPTION: str
    CORE_TITLE: str
    CORE_DESCRIPTION: str
    OUTPUT_GATE_TITLE: str
    OUTPUT_GATE_DESCRIPTION: str
    MLA_TITLE: str
    MLA_DESCRIPTION: str
    CONVOLUTION_TITLE: str
    CONVOLUTION_DESCRIPTION: str
    L2_NORM_TITLE: str
    L2_NORM_DESCRIPTION: str
    DELTA_QKV_TITLE: str
    DELTA_QKV_DESCRIPTION: str
    DECAY_TITLE: str
    DECAY_DESCRIPTION: str
    STRENGTH_TITLE: str
    STRENGTH_DESCRIPTION: str
    SCAN_STEP_TITLE: str
    SCAN_STEP_DESCRIPTION: str
    SCAN_TITLE: str
    SCAN_DESCRIPTION: str
    DELTA_OUTPUT_TITLE: str
    DELTA_OUTPUT_DESCRIPTION: str
    KDA_TITLE: str
    KDA_DESCRIPTION: str
    MIX_TITLE: str
    MIX_DESCRIPTION: str
    RESIDUAL_TITLE: str
    RESIDUAL_DESCRIPTION: str
    FIRST_LAYER_TITLE: str
    FIRST_LAYER_DESCRIPTION: str
    DELTA_LAYER_TITLE: str
    DELTA_LAYERS_DESCRIPTION: str
    LATENT_LAYER_TITLE: str
    LATENT_LAYER_DESCRIPTION: str
    GROUP_TITLE: str
    GROUP_DESCRIPTION: str
    FIRST_BLOCK_TITLE: str
    FIRST_BLOCK_DESCRIPTION: str
    REPEATED_BLOCKS_TITLE: str
    REPEATED_BLOCKS_DESCRIPTION: str
    LAST_BLOCK_TITLE: str
    LAST_BLOCK_DESCRIPTION: str
    EMBEDDING_TITLE: str
    EMBEDDING_BLOCK_DESCRIPTION: str
    OUTPUT_TITLE: str
    OUTPUT_DESCRIPTION: str
    TOP_K_EXPERTS_DESCRIPTION: str
    SELECT_DESCRIPTION: str
    EMBEDDING_DESCRIPTION: str
    KEY_JOIN_DESCRIPTION: str
    WRITE_AT_ENTRY_DESCRIPTION: str
    WRITE_AT_TOKEN_DESCRIPTION: str
    BACK_VIEW_DESCRIPTION: str
    REPEAT_VIEW_DESCRIPTION: str
    DIAGONAL_VIEW_DESCRIPTION: str
    TAPS_VIEW_DESCRIPTION: str
    DEPTHWISE_VIEW_DESCRIPTION: str
    EMBEDDING_ENTRY_VIEW_DESCRIPTION: str
    TOKEN_VIEW_DESCRIPTION: str
    QUERY_DOWN_PROJECTION_ROLE: str
    QUERY_UP_PROJECTION_ROLE: str
    LATENT_PROJECTION_ROLE: str
    SHARED_KEY_PROJECTION_ROLE: str
    KEY_EXPANSION_ROLE: str
    VALUE_EXPANSION_ROLE: str
    LATENT_OUTPUT_GATE_ROLE: str
    LATENT_OUTPUT_PROJECTION_ROLE: str
    DELTA_QUERY_PROJECTION_ROLE: str
    DELTA_KEY_PROJECTION_ROLE: str
    DELTA_VALUE_PROJECTION_ROLE: str
    QUERY_TAPS_ROLE: str
    KEY_TAPS_ROLE: str
    VALUE_TAPS_ROLE: str
    DECAY_DOWN_PROJECTION_ROLE: str
    DECAY_UP_PROJECTION_ROLE: str
    DECAY_BIAS_ROLE: str
    DECAY_RATE_ROLE: str
    STRENGTH_PROJECTION_ROLE: str
    DELTA_OUTPUT_GATE_ROLE: str
    DELTA_OUTPUT_PROJECTION_ROLE: str
    ROUTER_ROLE: str
    CORRECTION_BIAS_ROLE: str
    LATENT_DOWN_PROJECTION_ROLE: str
    LATENT_UP_PROJECTION_ROLE: str
    EXPERT_GATE_PROJECTION_ROLE: str
    EXPERT_UP_PROJECTION_ROLE: str
    EXPERT_DOWN_PROJECTION_ROLE: str
    SHARED_GATE_PROJECTION_ROLE: str
    SHARED_UP_PROJECTION_ROLE: str
    SHARED_DOWN_PROJECTION_ROLE: str
    DENSE_GATE_PROJECTION_ROLE: str
    DENSE_UP_PROJECTION_ROLE: str
    DENSE_DOWN_PROJECTION_ROLE: str
    ATTENTION_MIX_WEIGHT_ROLE: str
    FEED_FORWARD_MIX_WEIGHT_ROLE: str
    OUTPUT_MIX_WEIGHT_ROLE: str
    OUTPUT_HEAD_ROLE: str
    EMBEDDING_TABLE_ROLE: str
    ATTENTION_SCALE_ROLE: str
    QUERY_SCALE_ROLE: str
    SIGMOID_ROLE: str
    SILU_ROLE: str
    INDICATOR_ROLE: str
    SITU_GATE_ROLE: str
    SITU_UP_ROLE: str
    DECAY_ROLE: str
    EXPONENTIAL_ROLE: str
    INVERSE_LENGTH_ROLE: str
    NEGATE_ROLE: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> BlockTitlesAndDescriptions:
        return wording_json.load_dataclass(cls, path)


TEXT = BlockTitlesAndDescriptions.load()
