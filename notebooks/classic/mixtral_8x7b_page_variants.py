# Claude Opus 5.5 (1M context), effort 40.
'''The four variants of the interactive page of Mixtral-8x7B.

The page draws two forms of the model. Decode is the whole model with no cache, in the
CausalSlide form, written by `quantised_mixtral_8x7b`. Cached is one pass over the new
tokens, reading the keys and the values of the earlier tokens from caches, derived by
`cached_mixtral_8x7b`. Each form is drawn at the quantisations of the reference
implementation and in the real numbers. The browser derives each unquantised variant
from its quantised variant by applying the functor
`notebook_diagrams.PageFunctor.DEQUANTISE`, which removes every quantisation and every
cast between two quantisations of one value, so the file carries two terms.

The quantised forms label every wire with its quantisation, so they are wider than the
unquantised forms, and each variant carries a wrap width of its own. Each width stands
in the middle of the window of widths at which every block of the variant is drawn in
one row, found on 2026-09-27 by writing the page at every 50 pixels from 1350 to 2050
and counting the regions of every block fill of every variant. The decoder layer is
drawn over two rows at every one of the four widths, the second row starting at the
residual connection around the mixture of experts.

`notebooks/website/classic/Mixtral8x7B.ipynb` writes the page with
`page_variants`, and its validator reads the legend of every variant through the same
function.
'''
from __future__ import annotations

import dataclasses

import data_structure.Category as cat

import notebooks.classic.quantised_mixtral_8x7b as quantised_mixtral_8x7b
import notebooks.display.explain_cached_reads as explain_cached_reads
import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.display.sota_figures as figures

SLUG = 'Mixtral8x7B'
INITIAL_VARIANT = 'decode-quantised'
DECODE_QUANTISED_WIDTH = 1775
'''The middle of the window from 1700 to 1850.'''
DECODE_UNQUANTISED_WIDTH = 1475
'''The middle of the window from 1400 to 1550.'''
CACHED_QUANTISED_WIDTH = 1900
'''The middle of the window from 1800 to 2000.'''
CACHED_UNQUANTISED_WIDTH = 1625
'''The middle of the window from 1550 to 1700.'''
DECODE_GROUP = notebook_diagrams.PageVariantGroup('decode', 'Decode')
CACHED_GROUP = notebook_diagrams.PageVariantGroup('cached', 'Cached')


def page_settings(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` written to one HTML file at `DECODE_QUANTISED_WIDTH`, with an
    inspection box over every block and every operator, the boxes drawn without
    their bodies, every wire labelled with its quantisation, and the inspection box
    over every weight stating the quantisation of the weight.'''
    return dataclasses.replace(
        settings, mode=figures.DiagramMode.HTML,
        advanced_display=figures.AdvancedDisplay.INTERACTIVE,
        display_mode=figures.DisplayMode.FAST, sub_blocks=figures.SubBlocks.NO_BODIES,
        clean_quantisation_labels=False, width=DECODE_QUANTISED_WIDTH,
        operator_roles=quantised_mixtral_8x7b.quantised_operator_roles())


def page_variants(
    quantised_decode: cat.Morphism, quantised_cached: cat.Morphism,
    settings: notebook_diagrams.DiagramSettings,
) -> tuple[notebook_diagrams.PageVariant, ...]:
    '''The quantised decode form and the quantised cached pass, each beside its
    unquantised variant derived in the browser. Every variant is drawn under
    `settings`, the settings of the page, at its own width. The cached variants also
    explain the two reads of the token axis written as views by the derivation of the
    pass, and the unquantised variants give each weight a role stating no
    quantisation.'''
    cached = explain_cached_reads.with_cached_read_explanations(settings)
    unquantised_roles = quantised_mixtral_8x7b.unquantised_operator_roles()
    return (
        notebook_diagrams.PageVariant(
            identifier='decode-quantised', group=DECODE_GROUP, title='Quantised',
            detail='The whole model with BF16 weights and activations and FP32 '
                   'arithmetic where the reference implementation upcasts.',
            term=quantised_decode,
            settings=dataclasses.replace(settings, width=DECODE_QUANTISED_WIDTH)),
        notebook_diagrams.PageVariant(
            identifier='decode-unquantised', group=DECODE_GROUP, title='Unquantised',
            detail='The whole model in the real numbers.',
            settings=dataclasses.replace(
                settings, width=DECODE_UNQUANTISED_WIDTH,
                operator_roles=unquantised_roles),
            derived_from='decode-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE),
        notebook_diagrams.PageVariant(
            identifier='cached-quantised', group=CACHED_GROUP, title='Quantised',
            detail='One pass over the new tokens, with the keys and the values of the '
                   'earlier tokens cached in BF16.',
            term=quantised_cached,
            settings=dataclasses.replace(cached, width=CACHED_QUANTISED_WIDTH)),
        notebook_diagrams.PageVariant(
            identifier='cached-unquantised', group=CACHED_GROUP, title='Unquantised',
            detail='One pass over the new tokens in the real numbers.',
            settings=dataclasses.replace(
                cached, width=CACHED_UNQUANTISED_WIDTH,
                operator_roles=unquantised_roles),
            derived_from='cached-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE),
    )
