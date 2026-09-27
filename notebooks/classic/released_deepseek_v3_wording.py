# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The block titles, block descriptions and inspection-box sentences of the released
DeepSeek-V3 that differ from those of the hand-drawn diagram.

`released_deepseek_v3.py` builds the model at the sizes of `config_671B.json`, and the
sentences that name a size are written here at those sizes. The sentences that name no
size, such as the description of the rotary embedding or of the causal mask, are read
from `deepseek_v3_wording.json`, so each is written once. `utilities/wording_json.py`
states the form of the file.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class ReleasedDeepSeekV3Wording:
    MODEL_TITLE: str
    MODEL_DESCRIPTION: str
    EMBEDDING_TITLE: str
    EMBEDDING_DESCRIPTION: str
    DENSE_LAYER_TITLE: str
    DENSE_LAYER_DESCRIPTION: str
    MOE_LAYER_TITLE: str
    MOE_LAYER_DESCRIPTION: str
    ATTENTION_TITLE: str
    ATTENTION_DESCRIPTION: str
    QUERY_GENERATION_TITLE: str
    QUERY_GENERATION_DESCRIPTION: str
    MLP_TITLE: str
    MLP_DESCRIPTION: str
    MOE_TITLE: str
    MOE_DESCRIPTION: str
    GATE_TITLE: str
    GATE_DESCRIPTION: str
    ROUTED_EXPERTS_TITLE: str
    ROUTED_EXPERTS_DESCRIPTION: str
    SHARED_EXPERTS_TITLE: str
    SHARED_EXPERTS_DESCRIPTION: str
    OUTPUT_TITLE: str
    OUTPUT_DESCRIPTION: str
    TOP_TWO_DESCRIPTION: str
    KEPT_GROUPS_DESCRIPTION: str
    CHOSEN_EXPERTS_DESCRIPTION: str
    INDEX_SELECT_DESCRIPTION: str
    SELECT_DESCRIPTION: str
    MERGED_POSITIONS_DESCRIPTION: str
    CANDIDATE_VIEW_DESCRIPTION: str
    SCORE_SCALE_ROLE: str
    INDICATOR_ROLE: str
    ROUTE_SCALE_ROLE: str
    QUERY_DOWN_ROLE: str
    QUERY_UP_ROLE: str
    ROTARY_QUERY_ROLE: str
    CORRECTION_BIAS_ROLE: str
    EMBEDDING_TABLE_ROLE: str
    CACHED_MODEL_DESCRIPTION: str
    CACHED_ATTENTION_DESCRIPTION: str
    WEIGHT_QUANTISATION_SENTENCE: str
    DEQUANTISED_WEIGHT_SENTENCE: str
    GROUP_VIEW_DESCRIPTION: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> ReleasedDeepSeekV3Wording:
        return wording_json.load_dataclass(cls, path)


TEXT = ReleasedDeepSeekV3Wording.load()
