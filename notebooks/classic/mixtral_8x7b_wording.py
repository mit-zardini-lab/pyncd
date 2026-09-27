# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Every block title, block description and inspection-box sentence of Mixtral-8x7B.

The wording is held in `mixtral_8x7b_wording.json` beside this module, one entry per
name, and `MixtralWording` names every entry as a field, so a module reads `text.NAME`
with the name checked. `utilities/wording_json.py` states the form of the file. The
descriptions carry the annotations of the hand-drawn diagram at
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
class MixtralWording:
    MODEL_TITLE: str
    MODEL_DESCRIPTION: str
    EMBEDDING_TITLE: str
    EMBEDDING_DESCRIPTION: str
    LAYER_TITLE: str
    LAYER_DESCRIPTION: str
    ATTENTION_RESIDUAL_TITLE: str
    MIXTURE_RESIDUAL_TITLE: str
    RESIDUAL_DESCRIPTION: str
    ATTENTION_TITLE: str
    ATTENTION_DESCRIPTION: str
    CORE_TITLE: str
    CORE_DESCRIPTION: str
    ROTARY_TITLE: str
    ROTARY_DESCRIPTION: str
    MIXTURE_TITLE: str
    MIXTURE_DESCRIPTION: str
    ROUTER_TITLE: str
    ROUTER_DESCRIPTION: str
    EXPERTS_TITLE: str
    EXPERTS_DESCRIPTION: str
    OUTPUT_TITLE: str
    OUTPUT_DESCRIPTION: str
    EMBEDDING_OPERATOR_DESCRIPTION: str
    PAIRS_AS_COMPLEX_DESCRIPTION: str
    DECOMPLEX_DESCRIPTION: str
    TOP_K_DESCRIPTION: str
    SILU_ROLE: str
    SCORE_SCALE_ROLE: str
    QUERY_ROLE: str
    KEY_ROLE: str
    VALUE_ROLE: str
    OUTPUT_ROLE: str
    GATE_ROLE: str
    FIRST_EXPERT_ROLE: str
    THIRD_EXPERT_ROLE: str
    SECOND_EXPERT_ROLE: str
    HEAD_ROLE: str
    RMS_NORM_ROLE: str
    ROTARY_TABLE_ROLE: str
    MASK_VIEW_DESCRIPTION: str
    DIAGONAL_VIEW_DESCRIPTION: str
    WEIGHT_QUANTISATION_SENTENCE: str
    EMBEDDING_TABLE_ROLE: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> MixtralWording:
        return wording_json.load_dataclass(cls, path)


TEXT = MixtralWording.load()
