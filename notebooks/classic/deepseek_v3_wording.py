# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Every block title, block description and inspection-box sentence of DeepSeek-V3.

The wording is held in `deepseek_v3_wording.json` beside this module, one entry per
name, and `DeepSeekV3Wording` names every entry as a field, so a module reads
`text.NAME` with the name checked. `utilities/wording_json.py` states the form of the
file. The descriptions carry the annotations of the hand-drawn diagram at
https://zardini.mit.edu/diagrams/, rewritten as statements of what each part computes. A
formula stays in the module that builds the model, beside the algebra it is written
from.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class DeepSeekV3Wording:
    MODEL_TITLE: str
    MODEL_DESCRIPTION: str
    EMBEDDING_TITLE: str
    EMBEDDING_DESCRIPTION: str
    DENSE_LAYER_TITLE: str
    DENSE_LAYER_DESCRIPTION: str
    MOE_LAYER_TITLE: str
    MOE_LAYER_DESCRIPTION: str
    RESIDUAL_TITLE: str
    RESIDUAL_DESCRIPTION: str
    ATTENTION_TITLE: str
    ATTENTION_DESCRIPTION: str
    QUERY_GENERATION_TITLE: str
    QUERY_GENERATION_DESCRIPTION: str
    KV_GENERATION_TITLE: str
    KV_GENERATION_DESCRIPTION: str
    ROTARY_TITLE: str
    ROTARY_DESCRIPTION: str
    CORE_TITLE: str
    CORE_DESCRIPTION: str
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
    EMBEDDING_OPERATOR_DESCRIPTION: str
    PAIRS_AS_COMPLEX_DESCRIPTION: str
    DECOMPLEX_DESCRIPTION: str
    TOP_K_DESCRIPTION: str
    CONCATENATE_DESCRIPTION: str
    SILU_ROLE: str
    SIGMOID_ROLE: str
    SCORE_SCALE_ROLE: str
    QUERY_ROLE: str
    ROTARY_QUERY_ROLE: str
    LATENT_ROLE: str
    ROTARY_KEY_ROLE: str
    KEY_ROLE: str
    VALUE_ROLE: str
    OUTPUT_ROLE: str
    GATE_ROLE: str
    FIRST_PROJECTION_ROLE: str
    THIRD_PROJECTION_ROLE: str
    SECOND_PROJECTION_ROLE: str
    HEAD_ROLE: str
    RMS_NORM_ROLE: str
    YARN_TABLE_ROLE: str
    MASK_VIEW_DESCRIPTION: str
    DIAGONAL_VIEW_DESCRIPTION: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> DeepSeekV3Wording:
        return wording_json.load_dataclass(cls, path)


TEXT = DeepSeekV3Wording.load()
