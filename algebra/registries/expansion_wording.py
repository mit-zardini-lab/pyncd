# Claude Fable 5.1, effort 80.
'''The prose an inspection box shows for an operator written out by a registry.

The descriptions of the standard expansions, the sentences of a linear map and the
sentences of a rotary table are held in `expansion_wording.json` beside this module,
one entry per name, and `ExpansionWording` names every entry as a field, so that
`algebra/operator_expansion.py`, `para/processing/write_linear_formula.py`,
`deepseek/registries/standard_expansions.py` and
`advanced_axis_dynamics/registries/standard_expansions.py` read `text.NAME` with the
name checked and typed, and the file is edited, copied and translated with no Python
in the way. `utilities/wording_json.py` states the form of the file. A sentence
composed with a value, such as the letter of a position axis, is an entry with a field
in braces, filled at the call with `.format`. A page that switches between wordings
resolves a second file against this one beside the file of the model, per
`websocket_transfer/localise_descriptions.py`.
'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class ExpansionWording:
    L1_NORM_EXPANSION_DESCRIPTION: str
    L2_NORM_EXPANSION_DESCRIPTION: str
    SOFTMAX_EXPANSION_DESCRIPTION: str
    NORMALIZE_EXPANSION_DESCRIPTION: str
    DECONCATENATION_EXPANSION_DESCRIPTION: str
    LEARNED_ARRAY_DESCRIPTION: str
    LINEAR_MAP_SENTENCE: str
    LINEAR_INDEX_OPERAND_SENTENCE: str
    LINEAR_BIAS_SENTENCE: str
    LINEAR_ARRAYS_FROM_THE_TAPE_SENTENCE: str
    ROTARY_TURN_SENTENCE: str
    ROTARY_TABLE_UNIT_STRIDE_SENTENCE: str
    ROTARY_TABLE_STRIDE_SENTENCE: str
    ROTARY_FASTEST_PAIR_SENTENCE: str
    ROTARY_FASTEST_PAIR_BEFORE_YARN_SENTENCE: str
    YARN_SENTENCE: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> ExpansionWording:
        return wording_json.load_dataclass(cls, path)


TEXT = ExpansionWording.load()
