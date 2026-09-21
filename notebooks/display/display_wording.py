# Claude Fable 5.1, effort 80.
'''The prose the display passes write into an inspection box.

The sentences of a cast between two quantisations and the sentence over an
elementwise map are held in `display_wording.json` beside this module, one entry per
name, and `DisplayWording` names every entry as a field, so that
`notebooks/display/cast_presentation.py` and `notebooks/display/explain_operators.py`
read `text.NAME` with the name checked and typed, and the file is edited, copied and
translated with no Python in the way. `utilities/wording_json.py` states the form of
the file. A sentence composed with a value, such as the number of bits of a format, is
an entry with a field in braces, filled at the call with `.format`. A page that
switches between wordings resolves a second file against this one beside the file of
the model, per `websocket_transfer/localise_descriptions.py`.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class DisplayWording:
    ELEMENTWISE_MAP_SENTENCE: str
    CAST_EVERY_VALUE_ROUNDED_SENTENCE: str
    CAST_BITS_AGAINST_PHRASE: str
    CAST_FEWER_BITS_SENTENCE: str
    CAST_MORE_BITS_SENTENCE: str
    CAST_SAME_BITS_SENTENCE: str
    SCALE_GROUP_OF_CONSECUTIVE_VALUES_PHRASE: str
    SCALE_GROUP_OF_A_BLOCK_PHRASE: str
    SCALE_SENTENCE: str
    CAST_READS_SENTENCE: str
    CAST_WIRE_LABEL_SENTENCE: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> DisplayWording:
        return wording_json.load_dataclass(cls, path)


TEXT = DisplayWording.load()
