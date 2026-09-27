# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The four forms of the released DeepSeek-V3 between which its interactive page
switches.

The page holds the pass over every token of the sequence, under Decode, and the pass
over the new tokens, under Cached, each at the quantisations of the released code. The
form of each pass in the reals is derived in the browser from the quantised form by the
dequantisation functor, which `quantization.algebra.strip_quantisations` states in
Python. `notebooks/website/classic/DeepSeekV3.ipynb` writes the page, and
`notebooks/website/classic/validate_deepseek_v3.py` checks it.
'''
from __future__ import annotations

import dataclasses

import notebooks.classic.quantised_deepseek_v3 as quantised_deepseek_v3
import notebooks.classic.released_deepseek_v3 as released_deepseek_v3
import notebooks.display.explain_cached_reads as explain_cached_reads
import notebooks.display.notebook_diagrams as notebook_diagrams

SLUG = 'DeepSeekV3'
INITIAL_VARIANT = 'decode-quantised'
PAGE_WIDTH = 960

DECODE_GROUP = notebook_diagrams.PageVariantGroup('decode', 'Decode')
CACHED_GROUP = notebook_diagrams.PageVariantGroup('cached', 'Cached')


def page_settings(
    quantised_diagrams: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''The settings of the page, from the settings the notebook draws its quantised
    figures under: one HTML file with an inspection box over every block and operator,
    drawn with the bodies of the boxes left to the inspection boxes.'''
    return dataclasses.replace(
        quantised_diagrams, mode=notebook_diagrams.DiagramMode.HTML,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
        display_mode=notebook_diagrams.DisplayMode.FAST,
        sub_blocks=notebook_diagrams.SubBlocks.NO_BODIES, width=PAGE_WIDTH)


def in_the_reals(
    source: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''The settings of a variant derived in the reals from a quantised variant drawn
    under `source`: the roles of the weights without the sentence naming the
    quantisation each weight is held in, and the explanations of the operators without
    the row of a cast, so the page carries the inspection text of the variant beside
    that of its source.'''
    return dataclasses.replace(
        source, operator_roles=released_deepseek_v3.OPERATOR_ROLES,
        operator_explanations=released_deepseek_v3.OPERATOR_EXPLANATIONS)


def page_variants(
    page: notebook_diagrams.DiagramSettings,
) -> tuple[notebook_diagrams.PageVariant, ...]:
    '''The four forms of the page. The cached pass carries the explanations of the reads
    of the token axis written by its derivation, and each form in the reals carries the
    inspection text of the model in the reals.'''
    cached = explain_cached_reads.with_cached_read_explanations(page)
    return (
        notebook_diagrams.PageVariant(
            identifier='decode-quantised', group=DECODE_GROUP, title='Quantised',
            detail=('Every token of the sequence, at the quantisations of the FP8 '
                    'checkpoint'),
            term=quantised_deepseek_v3.decode_quantised),
        notebook_diagrams.PageVariant(
            identifier='decode-unquantised', group=DECODE_GROUP, title='Unquantised',
            detail='Every token of the sequence, in the reals',
            derived_from='decode-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE,
            settings=in_the_reals(page)),
        notebook_diagrams.PageVariant(
            identifier='cached-quantised', group=CACHED_GROUP, title='Quantised',
            detail='The new tokens, with the latent and the turned key cached in BF16',
            term=quantised_deepseek_v3.cached_quantised,
            settings=cached),
        notebook_diagrams.PageVariant(
            identifier='cached-unquantised', group=CACHED_GROUP, title='Unquantised',
            detail='The new tokens, with the latent and the turned key cached, in the '
                   'reals',
            derived_from='cached-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE,
            settings=in_the_reals(cached)),
    )
