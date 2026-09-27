# Claude Opus 5.5 (1M context), effort 40.
'''The block descriptions of the cached GLM-5.3, held in `wording.json` beside this
module in the form `utilities/wording_json.py` states. A block the cache leaves
unchanged keeps its title and its description from
`notebooks/sota/GLM53/block_titles_and_descriptions.json`, so the two figures of one
part carry one title.'''
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import utilities.wording_json as wording_json

WORDING_FILE = pathlib.Path(__file__).with_suffix('.json')


@dataclass(frozen=True)
class CachedGLM53Wording:
    POSITION_OF_THIS_PASS_SENTENCE: str
    QUERY_KEY_ROTATION_AT_THIS_PASS_DESCRIPTION: str
    INDEXER_ROTATION_AT_THIS_PASS_DESCRIPTION: str
    SCORE_AGAINST_THE_CACHE_DESCRIPTION: str
    CACHED_INDEXER_DESCRIPTION: str
    QUERY_PATH_AT_THIS_PASS_DESCRIPTION: str
    QUERY_WITHOUT_LOW_RANK_AT_THIS_PASS_DESCRIPTION: str
    CACHED_KEYS_AND_VALUES_DESCRIPTION: str
    GATHER_CACHED_KEYS_DESCRIPTION: str
    GATHER_CACHED_VALUES_DESCRIPTION: str
    CACHED_FULL_DESCRIPTION: str
    CACHED_FULL_PUBLISHING_DESCRIPTION: str
    CACHED_SHARED_DESCRIPTION: str

    @classmethod
    def load(cls, path: pathlib.Path = WORDING_FILE) -> CachedGLM53Wording:
        return wording_json.load_dataclass(cls, path)


TEXT = CachedGLM53Wording.load()
