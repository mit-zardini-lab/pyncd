# Claude Opus 5.5 (1M context), effort 40.
'''The prose an inspection box shows over a `Caching`, held in `cache_wording.json`
beside this module in the form `utilities/wording_json.py` states, as
`algebra/registries/expansion_wording.py` holds the prose of the other expansions.'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class CacheWording:
    CACHING_EXPANSION_DESCRIPTION: str
    CACHING_EXPANSION_FORMULA: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> CacheWording:
        return wording_json.load_dataclass(cls, path)


TEXT = CacheWording.load()
