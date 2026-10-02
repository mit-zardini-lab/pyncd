# Claude Opus 5.5 (1M context), effort 40.
'''The forms of Kimi K3 that one interactive page carries, and the settings the page is
written under.

The page carries the model reading every token of the prompt, drawn in the CausalSlide
by `slide_causal_reads`, in the reals. `caching.algebra.derive_cached_pass` has no rule
for a loop over the tokens, which the delta rule is, so the page carries no pass over new
tokens, and `obsidian/06-practice/Open Gaps.md` records the rule.

`notebooks/website/modern/KimiK3.ipynb` writes the page with `page_settings` and
`page_variants`.
'''
from __future__ import annotations

import dataclasses

import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.display.sota_figures as figures
import notebooks.sota.KimiK3.operator_explanations as operator_explanations
import notebooks.sota.KimiK3.slide_causal_reads as slide_causal_reads
import notebooks.sota.KimiK3.whole_model as whole_model

PAGE_SLUG = 'KimiK3'
PAGE_TITLE = 'Kimi K3'
INITIAL_VARIANT = 'decode-unquantised'
DECODE_WIDTH = 1500
'''The target width of the rows, which are planned to keep every block whole. Each row
holds one layer of two sublayers. The embedding stands beside layer 0, and the output
logits stand beside the last layer. The width was chosen on 2026-10-02.'''

DECODE_GROUP = notebook_diagrams.PageVariantGroup('decode', 'Decode')


def page_settings(mode: notebook_diagrams.DiagramMode) -> notebook_diagrams.DiagramSettings:
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
    '''The variants of the page, each drawn under `settings` with its own sizes and
    width.'''
    decode_settings = dataclasses.replace(
        settings, width=DECODE_WIDTH,
        assigned_sizes=whole_model.released_assigned_sizes(slide_causal_reads.kimi_k3_slid))
    return (
        notebook_diagrams.PageVariant(
            identifier='decode-unquantised', group=DECODE_GROUP, title='Unquantised',
            detail='Every token of the prompt, in the reals',
            term=slide_causal_reads.kimi_k3_slid, settings=decode_settings),
    )
