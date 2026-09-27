# Claude Opus 5.5 (1M context), effort 40.
'''The axes a cache adds to GLM-5.3, beside the axes of `notebooks/sota/GLM53/`.

A pass reads the tokens `x` appended since the pass before, which is the whole prompt
in the first pass and one token in a pass of plain decoding. The tokens cached by the
earlier passes are the axis `P`, and every token the cache holds after this pass is
the axis `P + x`, an `AxisConcatenation.ConcatenatedAxis` whose positions are those of
`P` followed by those of `x`. `CACHED_TOKENS` is that axis, and every array loaded
from a cache stands on it.

The distance axis `r` of the uncached model has one position per token of `x`,
because a query can count back over every token before it. In the cached model a query
counts back over every cached token, so `CACHED_DISTANCES` has `|P| + |x|` positions.
Distance `i_r` of query `i_x` reads position `|P| + i_x - i_r` of `P + x`, because
query `i_x` stands at position `|P| + i_x` of the sequence. The read marks the distance
axis as `r|x`, live where `|P| + i_x - i_r >= 0`.

With `|P| = 0` every axis here has the size of the axis it replaces, and the cached
model computes what the uncached model computes.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Term as fd

import caching.data_structure.Caching as Caching
from notebooks.sota.GLM53.declared_axes import R, c, d, m, p, x

P = cat.RawAxis.named('P', code_form='past_tokens')
CACHED_TOKENS = Caching.cached_token_axis(P, x)
CACHED_DISTANCES = fd.DynamicName('r', code_form='cached_distances').capture(
    cat.RawAxis(_size=CACHED_TOKENS.local_size()))

LATENT_OF_THIS_PASS = cat.Array(R, (x, c))
TURNED_KEY_OF_THIS_PASS = cat.Array(R, (x, p))
INDEXER_KEY_OF_THIS_PASS = cat.Array(R, (x, d))
CACHED_LATENT = cat.Array(R, (CACHED_TOKENS, c))
CACHED_TURNED_KEY = cat.Array(R, (CACHED_TOKENS, p))
CACHED_INDEXER_KEY = cat.Array(R, (CACHED_TOKENS, d))
STATE_OF_THIS_PASS = cat.Array(R, (x, m))

LATENT_CACHE = '\\mathrm{lat}'
TURNED_KEY_CACHE = '\\mathrm{rot}'
INDEXER_KEY_CACHE = '\\mathrm{idx}'
