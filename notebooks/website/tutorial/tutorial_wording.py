# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Every block title, block description and inspection-box sentence of the four tutorial
pages on attention.

The wording is held in `tutorial_wording.json` beside this module, one entry per
name, and `TutorialWording` names every entry as a field, so a module reads `text.NAME`
with the name checked. `utilities/wording_json.py` states the form of the file. A formula
stays in `express_attention.py`, beside the algebra it is written from.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class TutorialWording:
    CORE_TITLE: str
    CORE_DESCRIPTION: str
    SELF_ATTENTION_TITLE: str
    SELF_ATTENTION_DESCRIPTION: str
    RESIDUAL_TITLE: str
    RESIDUAL_DESCRIPTION: str
    MULTI_HEAD_TITLE: str
    MULTI_HEAD_DESCRIPTION: str
    GROUPED_QUERY_TITLE: str
    GROUPED_QUERY_DESCRIPTION: str
    SCORE_SCALE_ROLE: str
    MASK_VIEW_DESCRIPTION: str
    QUERY_ROLE: str
    KEY_ROLE: str
    VALUE_ROLE: str
    OUTPUT_ROLE: str
    HEAD_QUERY_ROLE: str
    HEAD_KEY_ROLE: str
    HEAD_VALUE_ROLE: str
    HEAD_OUTPUT_ROLE: str
    GROUPED_QUERY_ROLE: str
    GROUPED_KEY_ROLE: str
    GROUPED_VALUE_ROLE: str
    GROUPED_OUTPUT_ROLE: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> TutorialWording:
        return wording_json.load_dataclass(cls, path)


TEXT = TutorialWording.load()
