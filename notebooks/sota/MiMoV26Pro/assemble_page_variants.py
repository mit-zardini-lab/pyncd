# Claude Opus 5.5 (1M context), effort 40.
'''The two forms of MiMo-V2.6-Pro that one interactive page switches between, and the
settings the page is written under.

The page has two groups. Decode is the model reading every token of the prompt, drawn
in the CausalSlide by `slide_causal_reads`. Cached is the pass over the new tokens,
derived by `derive_cached_mimo_v26_pro`. Each group holds the form in the reals alone,
and the variant is named Unquantised, as the variants in the reals of the other modern
pages are, so a quantised variant can join each group under the same address. The
`transformers` 5.3.0 named by the configuration of the checkpoint does not read the
MXFP4 store of the routed experts, and `notebooks/website/modern/MiMoV26Pro.ipynb`
states why no quantised form is drawn.

`notebooks/website/modern/MiMoV26Pro.ipynb` writes the page with `page_settings` and
`page_variants`.
'''
from __future__ import annotations

import dataclasses

import notebooks.display.explain_cached_reads as explain_cached_reads
import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.display.sota_figures as figures
import notebooks.sota.MiMoV26Pro.derive_cached_mimo_v26_pro as derive_cached_mimo_v26_pro  # noqa: E501
import notebooks.sota.MiMoV26Pro.operator_explanations as operator_explanations
import notebooks.sota.MiMoV26Pro.slide_causal_reads as slide_causal_reads
import notebooks.sota.MiMoV26Pro.whole_model as whole_model

PAGE_SLUG = 'MiMoV26Pro'
PAGE_TITLE = 'MiMo-V2.6-Pro'
INITIAL_VARIANT = 'decode-unquantised'
PAGE_WIDTH = 1800
'''The target width of the rows of both forms, which are planned to keep every block
whole. The embedding and layer 0 take the first row, each of the three groups of layers
takes a row of its own, and the output logits stand beside the last group. The width was
chosen on 2026-10-02.'''

DECODE_GROUP = notebook_diagrams.PageVariantGroup('decode', 'Decode')
CACHED_GROUP = notebook_diagrams.PageVariantGroup('cached', 'Cached')


def page_settings(
    mode: notebook_diagrams.DiagramMode,
) -> notebook_diagrams.DiagramSettings:
    '''The settings of the page: the legend and an inspection box over every block,
    operator, weight and view, and the body of every box left out of the figure and
    shown in its inspection box.'''
    return operator_explanations.with_explanation_tables(
        figures.DiagramSettings(
            mode=mode,
            display_mode=figures.DisplayMode.FAST,
            tape=figures.TapePresentation.ABSORBED,
            axis_sizes=figures.AxisSizes.SUBSCRIPT,
            axis_label_font_size=0.8,
            block_recycling=figures.BlockRecycling.RECYCLED,
            advanced_display=figures.AdvancedDisplay.INTERACTIVE,
            expanded_parameters=figures.ExpandedParameters.WEIGHT_ARRAYS,
            sub_blocks=figures.SubBlocks.NO_BODIES,
            title=PAGE_TITLE))


def page_variants(
    settings: notebook_diagrams.DiagramSettings,
) -> tuple[notebook_diagrams.PageVariant, ...]:
    '''The two variants of the page, each drawn under `settings` with its own sizes,
    and the cached one with the rows of the reads of the token axis.'''
    decode_settings = dataclasses.replace(
        settings, width=PAGE_WIDTH,
        assigned_sizes=whole_model.released_assigned_sizes(
            slide_causal_reads.mimo_slid))
    cached_settings = explain_cached_reads.with_cached_read_explanations(
        dataclasses.replace(
            settings, width=PAGE_WIDTH,
            assigned_sizes=derive_cached_mimo_v26_pro.cached_assigned_sizes()))
    return (
        notebook_diagrams.PageVariant(
            identifier='decode-unquantised', group=DECODE_GROUP, title='Unquantised',
            detail='Every token of the prompt, in the reals',
            term=slide_causal_reads.mimo_slid, settings=decode_settings),
        notebook_diagrams.PageVariant(
            identifier='cached-unquantised', group=CACHED_GROUP, title='Unquantised',
            detail='The new tokens, with the keys and the values cached, in the reals',
            term=derive_cached_mimo_v26_pro.cached_mimo, settings=cached_settings),
    )
