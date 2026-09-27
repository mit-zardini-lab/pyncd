# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Every block title, block description and inspection-box sentence of the transformer
of *Attention Is All You Need*.

The wording is held in `attention_is_all_you_need_wording.json` beside this module, one
entry per name, and `TransformerWording` names every entry as a field, so a module reads
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
class TransformerWording:
    TRANSFORMER_TITLE: str
    TRANSFORMER_DESCRIPTION: str
    INPUT_EMBEDDING_TITLE: str
    INPUT_EMBEDDING_DESCRIPTION: str
    OUTPUT_EMBEDDING_TITLE: str
    OUTPUT_EMBEDDING_DESCRIPTION: str
    EMBEDDING_SENTENCE: str
    SUM_DROPOUT_SENTENCE: str
    POSITIONAL_ENCODING_TITLE: str
    POSITIONAL_ENCODING_DESCRIPTION: str
    SELF_ATTENTION_TITLE: str
    SELF_ATTENTION_DESCRIPTION: str
    MASKED_SELF_ATTENTION_TITLE: str
    MASKED_SELF_ATTENTION_DESCRIPTION: str
    CROSS_ATTENTION_TITLE: str
    CROSS_ATTENTION_DESCRIPTION: str
    HEADS_SENTENCE: str
    MASK_SENTENCE: str
    CORE_TITLE: str
    CORE_DESCRIPTION: str
    FEED_FORWARD_TITLE: str
    FEED_FORWARD_DESCRIPTION: str
    ADD_NORM_TITLE: str
    ADD_NORM_DESCRIPTION: str
    ENCODER_TITLE: str
    ENCODER_DESCRIPTION: str
    DECODER_TITLE: str
    DECODER_DESCRIPTION: str
    OUTPUT_TITLE: str
    OUTPUT_DESCRIPTION: str
    DROPOUT_DESCRIPTION: str
    EMBEDDING_OPERATOR_DESCRIPTION: str
    DECOMPLEX_DESCRIPTION: str
    QUERY_ROLE: str
    KEY_ROLE: str
    VALUE_ROLE: str
    OUTPUT_ROLE: str
    FIRST_FEED_FORWARD_ROLE: str
    SECOND_FEED_FORWARD_ROLE: str
    OUTPUT_PROJECTION_ROLE: str
    LAYER_NORM_ROLE: str
    POSITIONAL_TABLE_ROLE: str
    EMBEDDING_SCALE_ROLE: str
    SCORE_SCALE_ROLE: str
    RELU_ROLE: str
    SINE_FIRST_ROLE: str
    MASK_VIEW_DESCRIPTION: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> TransformerWording:
        return wording_json.load_dataclass(cls, path)


TEXT = TransformerWording.load()
