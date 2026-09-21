# Claude Opus 5, effort high.
'''The Gumbel-max sampler of DeepSeek-V4.1-Flash, which draws one token from the
probabilities of one position.

Written by Claude Fable 5.1, reasoning effort 80, in
`notebooks.sota.DeepSeekV41Flash.dspark_drafter`. Split into this module by
Claude Opus 5, effort high, on 2026-09-19, when the user asked for every mechanism to be
stated in its newest form.

The released `sample` divides the logits by a temperature, turns them into
probabilities, divides each probability by an independent draw from the exponential
distribution of rate 1, and returns the position of the largest ratio. Dividing by an
exponential draw and taking the largest is the Gumbel-max method, and the token it
returns is drawn with the probabilities themselves.

    temper_logits           the logits divided by the sampling temperature and turned
                            into probabilities
    exponential_draws       one independent draw for every entry of the vocabulary,
                            which is the one `ops.GenericOperator` of this module
    keep_largest, read_only_slot
                            the position of the largest ratio, as a top-1 selection
                            that hands out the position alone, read as one number
    sample_from_probabilities, SAMPLER
                            the block from one distribution to one token, and its box

A sampled token is a position on the vocabulary axis, so its datatype is
`cat.Natural(VOCABULARY.local_size())`, which is the datatype `dst.TopK` hands out. The
token identifiers the model reads carry `cat.Natural` of the vocabulary symbol, which is
the same number under another name.

`SAMPLER_RELEASED_CONSTANTS` gives the released value of the sampling temperature, in
the form of `released_constants.RELEASED_CONSTANTS`. `REINDEXING_EXPLANATIONS` says what
the read of the one selected slot is for, by the text of its name, and
`GENERIC_OPERATOR_EXPLANATIONS` says what the draw is, in the form
`operator_explanations` holds its own. The explanation tables of the omissions notebook
and of the integrated model both take those rows from here.
'''
from __future__ import annotations

import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst

from notebooks.display.explain_operators import OperatorExplanation
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over
from notebooks.sota.DeepSeekV41Flash.declared_axes import R
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (
    VOCABULARY, generic_operator)
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.released_constants import ReleasedConstant
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

SAMPLING_TEMPERATURE = nm.FreeNumeric.named('\\vartheta')
ONLY_SLOT = fd.DynamicName('1').capture(cat.RawAxis(_size=nm.Integer(1)))

DISTRIBUTION = cat.Array(R, (VOCABULARY,))
SAMPLED_TOKEN = cat.Array(cat.Natural(VOCABULARY.local_size()), ())

DRAW_NAME = '\\mathrm{Exp}(1)'
GENERIC_OPERATOR_NAMES = frozenset({DRAW_NAME})
ONLY_SLOT_NAME = 'first'

SAMPLER_BOX = 'Smp'
SAMPLER_COLOUR = '#E6E0F3'

SAMPLER_RELEASED_CONSTANTS: tuple[ReleasedConstant, ...] = (
    ReleasedConstant(
        symbol=SAMPLING_TEMPERATURE, released_value='1',
        meaning='the temperature the sampler divides the logits by',
        reference=model_lines(53)),
)


def temper_logits() -> cat.BroadcastedCategory:
    '''The logits divided by the sampling temperature and turned into probabilities.'''
    return (over((VOCABULARY,), ops.Arithmetic.template(nm.x / SAMPLING_TEMPERATURE))
            @ ops.SoftMax.template())


def exponential_draws() -> cat.Broadcasted:
    '''One independent draw from the exponential distribution of rate 1 for every entry
    of the vocabulary. A draw is no function of an operand, so no standard operator
    states it.'''
    return generic_operator(DRAW_NAME, (), (DISTRIBUTION,))


def keep_largest() -> cat.Broadcasted:
    '''The position of the largest entry along the vocabulary, as a top-1 selection that
    hands out the position alone. The count kept is the literal one, because no
    configuration sets it.'''
    return dst.TopK.template(
        k=nm.Integer(1), axis=VOCABULARY, selected_axis=ONLY_SLOT,
        form=dst.SelectionForm.ONLY_SELECTION)


def read_only_slot() -> cat.Broadcasted:
    '''The one slot of a top-1 selection read as a single number.'''
    return ops.View.template(
        base=SAMPLED_TOKEN.datatype,
        reindexing=(sc.StrideMorphism(
            _dom=(), _cod_stride_shift=((ONLY_SLOT, (), nm.Integer(0)),),
            name=fd.DynamicName(ONLY_SLOT_NAME)),),
        name=ONLY_SLOT_NAME)


def sample_from_probabilities() -> cat.Block:
    '''One token drawn with the probabilities of one position: each probability divided
    by its own draw, and the position of the largest ratio.'''
    return cat.Block.template(
        (hold(DISTRIBUTION)
         * (exponential_draws()
            @ over((VOCABULARY,), ops.Arithmetic.template(nm.Integer(1) / nm.x))))
        @ einops_simplification.einsum(((VOCABULARY,), (VOCABULARY,)), (VOCABULARY,))
        @ keep_largest()
        @ read_only_slot(),
        title=text.SAMPLER_TITLE, fill_color=SAMPLER_COLOUR,
        formula=('\\mathrm{token} = \\arg\\max_{i_{\\overline{v}} \\in \\overline{v}} '
                 '\\frac{p[i_{\\overline{v}}]}{\\epsilon[i_{\\overline{v}}]}, \\quad '
                 '\\epsilon[i_{\\overline{v}}] \\sim \\mathrm{Exp}(1)'),
        description=text.SAMPLE_FROM_PROBABILITIES_DESCRIPTION,
        references=(model_lines(1285, 1292),))


SAMPLER = boxed(sample_from_probabilities(), SAMPLER_BOX)

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    ONLY_SLOT_NAME: ReindexingExplanation(
        description=text.ONLY_SLOT_DESCRIPTION,
        references=(model_lines(1292),)),
}

DRAW_EXPLANATION = OperatorExplanation(
    title=r'\text{Exponential Draw}',
    formula=(r'\epsilon[i_{\overline{v}}] \sim \mathrm{Exp}(1) '
             r'\text{ independently for every } i_{\overline{v}} \in \overline{v}'),
    description=text.EXPONENTIAL_DRAW_DESCRIPTION,
    references=(model_lines(1292), model_lines(1285, 1292), model_lines(1270),
                model_lines(1153)))

GENERIC_OPERATOR_EXPLANATIONS: dict[str, OperatorExplanation] = {
    DRAW_NAME: DRAW_EXPLANATION,
}
