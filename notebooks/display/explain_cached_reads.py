# Claude Opus 5.5 (1M context), effort 40.
'''The inspection boxes over the two reads of the token axis that a derived cached pass
writes as views.

`caching.algebra.derive_cached_pass` carries two reads of the token axis through a
model. The read of the new tokens, named `New`, reads position `|x_old| + i_new` for
the new token `i_new`. The read of a cached axis, named `Cached`, reads the kept earlier
tokens followed by the new ones. A read the crawl cannot carry further is written as a
view carrying the name of the read. A table of positions, such as the table of turns of
a rotary embedding, reads no token and is computed over every cached token, and a view
named `New` reads it at the new tokens.

`CACHED_READ_EXPLANATIONS` gives each name a row of the table `explain_reindexings`
reads, keyed by the bodies of the name as every table of a model is.
`with_cached_read_explanations` adds the rows to the table of a figure's settings, so
the page of every derived pass opens a box over both views.
'''
from __future__ import annotations

import dataclasses

import caching.algebra.derive_cached_pass as derive_cached_pass

import notebooks.display.notebook_diagrams as notebook_diagrams
from notebooks.display.explain_reindexings import ReindexingExplanation

NEW_TOKENS_READ_DESCRIPTION = (
    'Reads the array at the positions of the new tokens of this pass. The new tokens '
    'follow the tokens of the earlier passes, so new token i_new reads position '
    '|x_old| + i_new of the sequence. A table of positions, such as the table of turns '
    'of a rotary embedding, is computed over every token of the sequence and read this '
    'way, which adds the number of cached tokens to the position of every new token.')
CACHED_TOKENS_READ_DESCRIPTION = (
    'Reads the array at the positions a cache holds: the earlier tokens it keeps, '
    'followed by the new tokens of this pass. Where the cache keeps the last |K| '
    'earlier tokens, position i of the cached axis is position |x_old| - |K| + i of the '
    'sequence.')

CACHED_READ_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    derive_cached_pass.NEW_TOKENS_READ_NAME.to_bodies(): ReindexingExplanation(
        description=NEW_TOKENS_READ_DESCRIPTION),
    derive_cached_pass.CACHED_TOKENS_READ_NAME.to_bodies(): ReindexingExplanation(
        description=CACHED_TOKENS_READ_DESCRIPTION),
}


def with_cached_read_explanations(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` with the rows of `CACHED_READ_EXPLANATIONS` added to its table of
    reindexing explanations.'''
    return dataclasses.replace(settings, reindexing_explanations={
        **(settings.reindexing_explanations or {}), **CACHED_READ_EXPLANATIONS})
