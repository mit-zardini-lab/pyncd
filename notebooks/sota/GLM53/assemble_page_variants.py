# Claude Opus 5.5 (1M context), effort 40.
'''The four forms of GLM-5.3 that one interactive page switches between, and the
settings the page is written under.

The page has two groups. Decode is the model reading every token of the prompt, drawn
in the CausalSlide by `slide_causal_reads`. Cached is the pass over the new tokens,
derived by `notebooks/caching/CachedGLM53/derive_cached_glm53.py`. Each group holds the
form quantised as `transformers` runs the FP8 checkpoint and the form in the reals. The
page carries the two quantised forms and derives each form in the reals in the browser
with the functor `notebook_diagrams.PageFunctor.DEQUANTISE`, whose Python statement is
`quantization.algebra.strip_quantisations.strip_quantisations`.
`notebooks/website/modern/validate_glm53.py` checks that the functor applied to
each quantised form gives the form in the reals.

`notebooks/website/modern/GLM53.ipynb` writes the page with `page_settings` and
`page_variants`.
'''
from __future__ import annotations

import dataclasses

import notebooks.caching.CachedGLM53.derive_cached_glm53 as derive_cached_glm53
import notebooks.display.explain_cached_reads as explain_cached_reads
import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.display.sota_figures as figures
import notebooks.sota.GLM53.operator_explanations as operator_explanations
import notebooks.sota.GLM53.quantised_whole_model as quantised_whole_model
import notebooks.sota.GLM53.slide_causal_reads as slide_causal_reads
import notebooks.sota.GLM53.whole_model as whole_model

PAGE_SLUG = 'GLM53'
PAGE_TITLE = 'GLM-5.3'
INITIAL_VARIANT = 'decode-quantised'
QUANTISED_WIDTH = 1500
UNQUANTISED_WIDTH = 1325
'''The target widths of the rows, which are planned to keep every block whole. One is
for the forms carrying a quantisation on every wire, and one is for the forms in the
reals, whose labels are shorter. In all four forms the embedding and the layers with a
dense MLP take the first row, each IndexShare group takes a row of its own, and the
output logits take the last row. The widths were chosen on 2026-10-02.'''

DECODE_GROUP = notebook_diagrams.PageVariantGroup('decode', 'Decode')
CACHED_GROUP = notebook_diagrams.PageVariantGroup('cached', 'Cached')


def page_settings(mode: notebook_diagrams.DiagramMode) -> notebook_diagrams.DiagramSettings:
    '''The settings of the page: the legend and an inspection box over every block,
    operator, weight and cast, the body of every box left out of the figure and shown
    in its inspection box, and the quantisation kept on every wire.'''
    return quantised_whole_model.with_quantised_explanation_tables(
        figures.DiagramSettings(
            mode=mode,
            display_mode=figures.DisplayMode.FAST,
            tape=figures.TapePresentation.ABSORBED,
            axis_sizes=figures.AxisSizes.SUBSCRIPT,
            axis_label_font_size=0.8,
            block_recycling=figures.BlockRecycling.RECYCLED,
            advanced_display=figures.AdvancedDisplay.INTERACTIVE,
            expanded_parameters=figures.ExpandedParameters.WEIGHT_ARRAYS,
            clean_quantisation_labels=False,
            sub_blocks=figures.SubBlocks.NO_BODIES,
            title=PAGE_TITLE))


def in_the_reals(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` at the width of the forms in the reals, with the inspection tables of
    the model in the reals, whose roles name no quantisation.'''
    return operator_explanations.with_explanation_tables(
        dataclasses.replace(settings, width=UNQUANTISED_WIDTH))


def page_variants(
    settings: notebook_diagrams.DiagramSettings,
) -> tuple[notebook_diagrams.PageVariant, ...]:
    '''The four variants of the page, each quantised variant drawn under `settings`
    with its own sizes and width, and each variant in the reals derived from the
    quantised variant of its group, under `in_the_reals` of the settings of that
    variant.'''
    decode_settings = dataclasses.replace(
        settings, width=QUANTISED_WIDTH,
        assigned_sizes=whole_model.released_assigned_sizes(
            slide_causal_reads.glm53_quantised_slid))
    cached_settings = explain_cached_reads.with_cached_read_explanations(
        dataclasses.replace(
            settings, width=QUANTISED_WIDTH,
            assigned_sizes=derive_cached_glm53.cached_assigned_sizes(
                derive_cached_glm53.cached_glm53_quantised)))
    return (
        notebook_diagrams.PageVariant(
            identifier='decode-quantised', group=DECODE_GROUP, title='Quantised',
            detail='Every token of the prompt, as transformers runs the FP8 checkpoint',
            term=slide_causal_reads.glm53_quantised_slid, settings=decode_settings),
        notebook_diagrams.PageVariant(
            identifier='decode-unquantised', group=DECODE_GROUP, title='Unquantised',
            detail='Every token of the prompt, in the reals',
            settings=in_the_reals(decode_settings),
            derived_from='decode-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE),
        notebook_diagrams.PageVariant(
            identifier='cached-quantised', group=CACHED_GROUP, title='Quantised',
            detail='The new tokens, with the latent and the keys cached in BF16',
            term=derive_cached_glm53.cached_glm53_quantised, settings=cached_settings),
        notebook_diagrams.PageVariant(
            identifier='cached-unquantised', group=CACHED_GROUP, title='Unquantised',
            detail='The new tokens, with the latent and the keys cached, in the reals',
            settings=explain_cached_reads.with_cached_read_explanations(
                in_the_reals(cached_settings)),
            derived_from='cached-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE),
    )
