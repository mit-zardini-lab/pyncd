# Claude Opus 5.5 (1M context), effort 40.
'''The pinned links into the cache of the reference implementation of GLM-5.3.

The model is read from `modeling_glm_moe_dsa.py` at the commit of
`notebooks/sota/GLM53/reference_links.py`. The cache it writes to is the
`DynamicCache` of `src/transformers/cache_utils.py` at the same commit,
`7cd73d9df0c1` of `huggingface/transformers`, read on 2026-09-25. Every layer of
GLM-5.3 declares the layer type `indexed_attention`, and the cache gives each such layer
a `DynamicIndexedLayer`, which holds the states handed to `update` and, apart from them,
the indexer keys handed to `update_indexer`.
'''
from __future__ import annotations

import data_structure.Category as cat
from term_utilities.code_references import pinned_link

from notebooks.sota.GLM53.reference_links import (
    TRANSFORMERS_URL, library_lines, line_label, modeling_lines)

CACHE_UTILS_PATH = 'src/transformers/cache_utils.py'
CACHE_READ_ON = '2026-09-25'


def cache_utils_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`cache_utils.py` at the pinned `transformers` commit.'''
    return pinned_link(TRANSFORMERS_URL, CACHE_UTILS_PATH, line, end_line,
                       label=line_label('cache_utils.py', line, end_line))


DYNAMIC_LAYER_UPDATE = cache_utils_lines(129, 148)
INDEXED_LAYER = cache_utils_lines(334, 366)
LAYER_TYPE_OF_EVERY_LAYER = cache_utils_lines(1256)
LATENT_CACHED_BEFORE_EXPANSION = modeling_lines(407, 413)
INDEXER_KEY_CACHED = modeling_lines(236, 239)
POSITIONS_AFTER_THE_CACHE = modeling_lines(702, 705)

CACHE_UPDATE_REFERENCES = (INDEXER_KEY_CACHED, INDEXED_LAYER)
LATENT_CACHE_REFERENCES = (LATENT_CACHED_BEFORE_EXPANSION, DYNAMIC_LAYER_UPDATE)
POSITION_REFERENCES = (POSITIONS_AFTER_THE_CACHE,)


CACHE_FORMAT_READ_ON = '2026-09-27'
DYNAMIC_CACHE_OF_THE_MODEL = modeling_lines(699, 700)
DYNAMIC_CACHE_OF_GENERATE = library_lines('generation/utils.py', 2142, 2144)
NO_CACHE_IMPLEMENTATION_BY_DEFAULT = library_lines(
    'generation/configuration_utils.py', 411)
CACHE_HOLDS_THE_DTYPE_HANDED_TO_IT = (cache_utils_lines(123, 127),
                                      cache_utils_lines(143, 148))
INDEXER_CACHE_HOLDS_THE_DTYPE_HANDED_TO_IT = (cache_utils_lines(348, 351),
                                              cache_utils_lines(353, 366))
EXPANSION_OF_EVERY_CACHED_TOKEN = (modeling_lines(362, 379), modeling_lines(413))
'''The lines, read on 2026-09-27 at the same commit, stating the class and the format
of the cache. The model creates a `DynamicCache` where none is passed, and `generate`
builds the same class unless another cache implementation is requested, which none is
by default. Each layer of the cache takes the dtype of the first states handed to it
and concatenates later states in it, for the latent and the turned key and for the
indexer key alike. The attention expands every cached latent on every pass.'''
