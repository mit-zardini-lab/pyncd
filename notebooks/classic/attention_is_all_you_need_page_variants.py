# Claude Opus 5.5 (1M context), effort 40.
'''The four variants of the interactive page of the transformer of *Attention Is All
You Need*.

The page draws two forms of the model. Decode is the whole model with no cache, in the
CausalSlide form, and Cached is one step of the decoder over the new target positions,
derived by `cached_attention_is_all_you_need`. Each form is drawn with the quantisations
of tensor2tensor, written by `quantised_attention_is_all_you_need`, and in the real
numbers. The browser derives each unquantised variant from its quantised variant by
applying the functor `notebook_diagrams.PageFunctor.DEQUANTISE`, which removes every
quantisation, so the file carries two terms.

`notebooks/website/classic/AttentionIsAllYouNeed.ipynb` writes the page with
`page_variants`, and its validator reads the legend of every variant through the same
function.
'''
from __future__ import annotations

import dataclasses

import data_structure.Category as cat

import notebooks.display.explain_cached_reads as explain_cached_reads
import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.display.sota_figures as figures

SLUG = 'AttentionIsAllYouNeed'
INITIAL_VARIANT = 'decode-quantised'
DECODE_WIDTH = 1800
UNQUANTISED_DECODE_WIDTH = 1700
CACHED_WIDTH = 2100
UNQUANTISED_CACHED_WIDTH = 2000
'''The target widths of the rows of the four variants, which are planned to keep every
block whole. The labels of the quantisations widen a figure, so each unquantised variant
takes a width of its own. The widths were chosen on 2026-10-02. In both Decode variants
the encoder layer stands whole on one row, the output embedding and the masked
self-attention take the next row, and the cross-attention, the feed-forward layer and
the output probabilities take the last. The input embedding stands beside the encoder
layer in the reals and takes a row of its own in the quantised variant. Both Cached
variants take two rows, the second starting at the cross-attention.'''
DECODE_GROUP = notebook_diagrams.PageVariantGroup('decode', 'Decode')
CACHED_GROUP = notebook_diagrams.PageVariantGroup('cached', 'Cached')


def page_settings(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` written to one HTML file at `DECODE_WIDTH`, with an inspection box
    over every block and every operator, the boxes drawn without their bodies, and
    every wire labelled with its quantisation.'''
    return dataclasses.replace(
        settings, mode=figures.DiagramMode.HTML,
        advanced_display=figures.AdvancedDisplay.INTERACTIVE,
        display_mode=figures.DisplayMode.FAST, sub_blocks=figures.SubBlocks.NO_BODIES,
        clean_quantisation_labels=False, width=DECODE_WIDTH)


def page_variants(
    quantised_decode: cat.Morphism, quantised_cached: cat.Morphism,
    settings: notebook_diagrams.DiagramSettings,
) -> tuple[notebook_diagrams.PageVariant, ...]:
    '''The quantised decode form and the quantised cached pass, each beside its
    unquantised variant derived in the browser. The decode variants are drawn under
    `settings`, the settings of the page, and the cached variants under `settings` with
    an inspection box over the reads of the new and the cached target positions, each
    variant at its own width.'''
    cached_settings = explain_cached_reads.with_cached_read_explanations(settings)
    return (
        notebook_diagrams.PageVariant(
            identifier='decode-quantised', group=DECODE_GROUP, title='Quantised',
            detail='The whole model, with FP32 on every real value and INT32 on the '
                   'token identifiers, as tensor2tensor runs it.',
            term=quantised_decode),
        notebook_diagrams.PageVariant(
            identifier='decode-unquantised', group=DECODE_GROUP, title='Unquantised',
            detail='The whole model in the real numbers.',
            settings=dataclasses.replace(settings, width=UNQUANTISED_DECODE_WIDTH),
            derived_from='decode-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE),
        notebook_diagrams.PageVariant(
            identifier='cached-quantised', group=CACHED_GROUP, title='Quantised',
            detail='One step of the decoder over the new target positions, with the '
                   'keys and the values cached in FP32.',
            term=quantised_cached,
            settings=dataclasses.replace(cached_settings, width=CACHED_WIDTH)),
        notebook_diagrams.PageVariant(
            identifier='cached-unquantised', group=CACHED_GROUP, title='Unquantised',
            detail='One step of the decoder in the real numbers.',
            settings=dataclasses.replace(
                cached_settings, width=UNQUANTISED_CACHED_WIDTH),
            derived_from='cached-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE),
    )
