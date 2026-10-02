# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The variants of the four tutorial pages on attention, and the width each is drawn at.

A notebook passes the list one of these functions returns to
`notebook_diagrams.show_page_variants`, and its validator passes the same list to
`notebook_diagrams.page_variant_legends`, so the page and its checks read one list. Every
variant is drawn under the settings of the page with its own width, because a variant's
settings replace the settings of the page whole. The two pages whose model has a training
step switch between the forward pass and the training step, and the other two carry the
forward pass alone, with no selector.
'''
from __future__ import annotations

import dataclasses

import data_structure.Category as cat

import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.website.tutorial.derive_training_step as derive_training_step

PASS = notebook_diagrams.PageVariantGroup('pass', 'Pass')
FORWARD = 'forward'
TRAINING = 'training'

ATTENTION_SLUG = 'Attention'
ATTENTION_WITH_WEIGHTS_AND_RESIDUAL_SLUG = 'AttentionWithWeightsAndResidual'
MULTI_HEAD_ATTENTION_SLUG = 'MultiHeadAttention'
GROUPED_QUERY_ATTENTION_SLUG = 'GroupedQueryAttention'

TRAINING_DETAIL = 'The forward pass saving to the tape, over the backward pass.'
CAUSAL_SLIDE_DETAIL = 'The model in the CausalSlide form.'


def forward_variant(term: cat.Morphism, detail: str,
                    page: notebook_diagrams.DiagramSettings,
                    width: int) -> notebook_diagrams.PageVariant:
    return notebook_diagrams.PageVariant(
        FORWARD, PASS, 'Forward', detail, term=term,
        settings=dataclasses.replace(page, width=width))


def training_variant(step: derive_training_step.TrainingStep,
                     page: notebook_diagrams.DiagramSettings,
                     width: int) -> notebook_diagrams.PageVariant:
    return notebook_diagrams.PageVariant(
        TRAINING, PASS, 'Training', TRAINING_DETAIL, term=step.forward_over_backward(),
        settings=dataclasses.replace(page, width=width))


def attention_page(
    attention: cat.Morphism, step: derive_training_step.TrainingStep,
    page: notebook_diagrams.DiagramSettings,
) -> list[notebook_diagrams.PageVariant]:
    return [forward_variant(attention, 'The attention as it is written.', page, 900),
            training_variant(step, page, 1400)]


def attention_with_weights_and_residual_page(
    displayed: cat.Morphism, step: derive_training_step.TrainingStep,
    page: notebook_diagrams.DiagramSettings,
) -> list[notebook_diagrams.PageVariant]:
    return [forward_variant(displayed, CAUSAL_SLIDE_DETAIL, page, 1400),
            training_variant(step, page, 900)]


def multi_head_attention_page(
    displayed: cat.Morphism, page: notebook_diagrams.DiagramSettings,
) -> list[notebook_diagrams.PageVariant]:
    return [forward_variant(displayed, CAUSAL_SLIDE_DETAIL, page, 1400)]


def grouped_query_attention_page(
    displayed: cat.Morphism, page: notebook_diagrams.DiagramSettings,
) -> list[notebook_diagrams.PageVariant]:
    return [forward_variant(displayed, CAUSAL_SLIDE_DETAIL, page, 1500)]
