# Claude Opus 5, effort high.
'''DSpark, the drafter of DeepSeek-V4.1-Flash, from the backbone's probabilities and the
three stream means to the five draft tokens and their confidences.

Written by Claude Fable 5.1, reasoning effort 80, in
`notebooks.sota.DeepSeekV41Flash.dspark_drafter`. Moved here by Claude Opus 5,
effort high, on 2026-09-19, when the user asked for every mechanism to be stated in
its newest form. The tap that writes the three means onto the tape, and the
placement of DSpark at the end of the model, stay in that module.

The backbone produces one token per pass. DSpark is a second model of three layers that
guesses the five tokens after it, so that the next pass of the backbone can check the
five guesses together. A guessed token is called a draft, and the token the backbone
produced is called the accepted token. The released code drafts at every decode step,
from the one token the step produced. This module states the same computation at the
last position of the sequence the expression reads.

    mean_over_streams       the mean over the four streams of the residual entering a
                            layer, which the tap of a target layer writes to the tape
    read_last_position      the read of the last position of an array over the tokens
    project_stream_means    the three means at the last position projected onto one
                            hidden state, and `main_norm`
    draft_trunk             the three draft blocks as one `ops.GenericOperator`
    head_over_draft_positions
                            the draft norm and the backbone's output head over the five
                            positions
    markov_head, confidence_head, draft_step
                            one draft step and the two heads it holds, each a box
    draft_chain             the five steps written out
    accept_token            the token the backbone produced, sampled from the
                            probabilities of the last position
    draft_five_tokens       the whole drafter, from the probabilities and the three
                            means to the five drafts and the five confidences

The released `main_proj` reads the three means laid end to end. A linear map of three
arrays laid end to end is the sum of a linear map of each, so the projection is three
`ops.Linear` and two additions, and no array stacks the three means.

The five draft steps are written out. A repeated block was built in the two forms the
package offers, and the passes the model goes through accept neither. With the drafted
token carried as a loop variable by `Para.StreamGrab` and `Para.StreamDrop`, every
result of the loop is on the tape, and
`graphs.processing.Hypergraph2Morphism.recycle` removes a loop none of whose results a
wire reads. With the drafted token carried on a wire the loop survives recycling. The
draft and the confidence of each step then have to leave through a `Para.LoopDrop`
inside the loop, and `para.data_structure.ParaBlockOperator.expose_tape_as_ports`
refuses a seed inside a nested scope, so the block could not be boxed. The written-out
chain holds one box five times, and the box is drawn once. The read of a draft position
stands outside the box, because the position is the one thing that differs between the
steps.

The sampler that every step ends in is `gumbel_max_sampler`.

`OPERATOR_ROLES` and `REINDEXING_EXPLANATIONS` say what each weight and each named
reindexing of this module is for, by the text of its name, and
`GENERIC_OPERATOR_EXPLANATIONS` and `EMBEDDING_EXPLANATIONS` say what the draft trunk
and the Markov embedding are, in the form `operator_explanations` holds its own. The
explanation tables of the omissions notebook and of the integrated model both take those
rows from here. `RELEASED_SIZES` gives the released sizes of the two axes DSpark adds.
'''
from __future__ import annotations

import functools
import operator
from collections.abc import Iterable

import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.display.explain_operators import OperatorExplanation
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, over, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, m, n, x
from notebooks.sota.DeepSeekV41Flash.gumbel_max_sampler import (
    SAMPLED_TOKEN, SAMPLER, temper_logits)
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (
    DRAFT_STATE, DRAFT_STATES, LOGITS, MARKOV_EMBEDDING, S, VOCABULARY, Z,
    generic_operator)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    declared_at, inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

TARGET_LAYERS = 3
FIRST_TARGET_LAYER = 37
DRAFT_STEPS = 5
RELEASED_SIZES: dict[str, int] = {'S': 5, 'Z': 256}

STREAM_MEAN = cat.Array(R, (x, m))
PROBABILITIES = cat.Array(R, (x, VOCABULARY))
DRAFT_LOGITS = cat.Array(R, (S, VOCABULARY))
CONFIDENCE = cat.Array(R, ())
READS_OF_ONE_POSITION = (LOGITS, DRAFT_STATE)
RESULTS_OF_ONE_STEP = (SAMPLED_TOKEN, CONFIDENCE)

DRAFT_NAME = '\\mathrm{draft}'
GENERIC_OPERATOR_NAMES = frozenset({DRAFT_NAME})
LAST_POSITION_NAME = 'last'
MARKOV_EMBEDDING_NAME = '\\mathrm{mk}'
MARKOV_PROJECTION_NAME = 'W^{M}'
CONFIDENCE_OF_STATE_NAME = 'W^{\\mathrm{conf}}_m'
CONFIDENCE_OF_EMBEDDING_NAME = 'W^{\\mathrm{conf}}_Z'

MARKOV_BOX = 'Mkv'
CONFIDENCE_BOX = 'Conf'
STEP_BOX = 'Step'

DSPARK_COLOUR = '#FCEFDC'
PART_COLOUR = '#F8E3C5'
HEAD_COLOUR = '#DBDFEF'
STEP_COLOUR = '#F4D9B4'


def product_of(parts: Iterable[cat.Morphism]) -> cat.Morphism:
    '''The parallel product of `parts`, in their order.'''
    return functools.reduce(operator.mul, parts)


def composition_of(parts: Iterable[cat.Morphism]) -> cat.Morphism:
    '''The sequential composition of `parts`, the first applied first.'''
    return functools.reduce(operator.matmul, parts)


def mean_over_streams() -> cat.BroadcastedCategory:
    '''The mean over the four streams of every token's residual: the sum over the
    stream axis divided by the number of streams.'''
    return (einops_simplification.einsum(((x, n, m),), (x, m))
            @ over((x, m), ops.Arithmetic.template(nm.x / n.local_size())))


def read_last_position(trailing_axes: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    '''The entry at the last token of a real array whose first axis is the token axis.
    The row has no stride and the shift |x| - 1, so it reads one position.'''
    last = sc.StrideMorphism(
        _dom=(),
        _cod_stride_shift=((x, (), x.local_size() - nm.Integer(1)),),
        name=fd.DynamicName(LAST_POSITION_NAME))
    return ops.View.template(
        base=R, reindexing=(last, cat.ProdObject(tuple(trailing_axes)).identity()),
        name=LAST_POSITION_NAME)


def main_projection_name(member: int) -> str:
    return f'W^{{\\mathrm{{main}}}}_{member}'


def project_stream_mean(member: int) -> cat.BroadcastedCategory:
    '''One mean read at the last position and projected by its own third of the
    released `main_proj`.'''
    return (read_last_position((m,))
            @ ops.Linear.template((m,), (m,), main_projection_name(member)))


def project_stream_means() -> cat.Block:
    '''The three means at the last position summed after three projections and
    normalised, which is the released `main_norm(main_proj(main_hidden))`.'''
    add = over((m,), ops.AdditionOp.template())
    return cat.Block.template(
        product_of(project_stream_mean(member) for member in range(TARGET_LAYERS))
        @ (add * hold(DRAFT_STATE))
        @ add
        @ ops.Normalize.template((m,)),
        title=text.BACKBONE_STATE_TITLE, fill_color=PART_COLOUR,
        formula=('\\mathrm{main}[i_{m}] = \\mathrm{RMSNorm}\\Big( \\sum_{k = 0}^{2} '
                 '\\sum_{i_{m\'} \\in m} W^{\\mathrm{main}}_{k}[i_{m\'}, i_{m}]\\, '
                 '\\mathrm{mean}_{k}[\\lvert x \\rvert - 1, i_{m\'}] \\Big)[i_{m}]'),
        description=text.PROJECT_STREAM_MEANS_DESCRIPTION,
        references=(model_lines(1113, 1114), model_lines(1130), model_lines(1271)))


def draft_trunk() -> cat.Broadcasted:
    '''The three draft blocks, from the projected backbone state and the accepted token
    to one hidden state per draft position. The blocks hold a window cache of their
    own, which is state across decode steps and no operand.'''
    return generic_operator(DRAFT_NAME, (DRAFT_STATE, SAMPLED_TOKEN), (DRAFT_STATES,))


def head_over_draft_positions() -> cat.Block:
    '''The logits of the five draft positions. The `Linear` carries the name and the
    axes of the backbone's output head, because the released code shares that weight.'''
    return cat.Block.template(
        over((S,), ops.Normalize.template((m,))
             @ ops.Linear.template((m,), (VOCABULARY,))),
        title=text.HEAD_TITLE, fill_color=HEAD_COLOUR,
        formula=('\\lambda[i_{S}, i_{\\overline{v}}] = \\sum_{i_{m} \\in m} '
                 'W[i_{m}, i_{\\overline{v}}]\\, \\mathrm{RMSNorm}(g[i_{S}])[i_{m}]'),
        description=text.HEAD_OVER_DRAFT_POSITIONS_DESCRIPTION,
        references=(model_lines(1116), model_lines(1144, 1145),
                    model_lines(1212, 1213)))


def markov_head() -> cat.Block:
    '''The bias one token puts on the logits of the next position, and the embedding
    the bias is read from, which the confidence head reads as well.'''
    return cat.Block.template(
        ops.Embedding.template(SAMPLED_TOKEN.datatype, (Z,), name=MARKOV_EMBEDDING_NAME)
        @ route((0, 0), (MARKOV_EMBEDDING,))
        @ (ops.Linear.template((Z,), (VOCABULARY,), MARKOV_PROJECTION_NAME)
           * hold(MARKOV_EMBEDDING)),
        title=text.MARKOV_TITLE, fill_color=PART_COLOUR,
        formula=('\\mathrm{bias}[i_{\\overline{v}}] = \\sum_{i_{Z} \\in Z} '
                 'W^{M}[i_{Z}, i_{\\overline{v}}]\\, E_{\\mathrm{mk}}[\\mathrm{token}, '
                 'i_{Z}]'),
        description=text.MARKOV_HEAD_DESCRIPTION,
        references=(model_lines(1077, 1086), inference_config_lines(10)))


MARKOV_HEAD = boxed(markov_head(), MARKOV_BOX)


def confidence_head() -> cat.Block:
    '''One number for a draft position, from the position's hidden state and the Markov
    embedding of the token that entered the step. The released weight reads the two
    laid end to end, which is the sum of a `Linear` of each.'''
    return cat.Block.template(
        (ops.Linear.template((m,), (), CONFIDENCE_OF_STATE_NAME)
         * ops.Linear.template((Z,), (), CONFIDENCE_OF_EMBEDDING_NAME))
        @ ops.AdditionOp.template(),
        title=text.CONFIDENCE_TITLE, fill_color=PART_COLOUR,
        formula=('\\mathrm{conf} = \\sum_{i_{m} \\in m} '
                 'W^{\\mathrm{conf}}_{m}[i_{m}]\\, g[i_{m}] + \\sum_{i_{Z} \\in Z} '
                 'W^{\\mathrm{conf}}_{Z}[i_{Z}]\\, E[i_{Z}]'),
        description=text.CONFIDENCE_HEAD_DESCRIPTION,
        references=(model_lines(1089, 1097), model_lines(1154, 1155)))


CONFIDENCE_HEAD = boxed(confidence_head(), CONFIDENCE_BOX)


def draft_step() -> cat.Block:
    '''One draft position: its logits biased by the Markov head of the token before
    it, the next token sampled from them, and the confidence of the position.'''
    return cat.Block.template(
        (hold(LOGITS) * hold(DRAFT_STATE) * MARKOV_HEAD)
        @ route((0, 2, 1, 3), (LOGITS, DRAFT_STATE, LOGITS, MARKOV_EMBEDDING))
        @ ((over((VOCABULARY,), ops.AdditionOp.template()) @ temper_logits() @ SAMPLER)
           * CONFIDENCE_HEAD),
        title=text.STEP_TITLE, fill_color=STEP_COLOUR,
        formula=('\\mathrm{token}_{k+1} = \\mathrm{sample}\\big( \\mathrm{softmax}( '
                 '(\\lambda[k] + \\mathrm{bias}(\\mathrm{token}_{k})) / \\vartheta ) '
                 '\\big)'),
        description=text.DRAFT_STEP_DESCRIPTION,
        references=(model_lines(1149, 1153), model_lines(1288, 1291)))


DRAFT_STEP = boxed(draft_step(), STEP_BOX)


def draft_position_name(position: int) -> str:
    return f'S_{position}'


def read_draft_position(
    position: int, trailing_axes: tuple[cat.Axis, ...],
) -> cat.Broadcasted:
    '''The entry at one draft position of a real array whose first axis is the draft
    axis.'''
    row = sc.StrideMorphism(
        _dom=(), _cod_stride_shift=((S, (), nm.Integer(position)),),
        name=fd.DynamicName.from_str(draft_position_name(position)))
    return ops.View.template(
        base=R, reindexing=(row, cat.ProdObject(tuple(trailing_axes)).identity()),
        name=draft_position_name(position))


def read_every_draft_position() -> cat.Morphism:
    '''The logits and the hidden state of each draft position on wires of their own,
    with the accepted token after those of position 0, where the first step reads
    it.'''
    sources = (0, 1, 2) + (0, 1) * (DRAFT_STEPS - 1)
    reads = tuple(
        read_draft_position(position, (VOCABULARY,))
        * read_draft_position(position, (m,))
        for position in range(DRAFT_STEPS))
    return (route(sources, (DRAFT_LOGITS, DRAFT_STATES, SAMPLED_TOKEN))
            @ product_of((reads[0], hold(SAMPLED_TOKEN), *reads[1:])))


def run_draft_step(position: int) -> cat.Morphism:
    '''The step of `position` applied between the results of the steps before it and
    the reads of the steps after it, and its draft copied to the step after it.'''
    finished = RESULTS_OF_ONE_STEP * position
    pending = READS_OF_ONE_POSITION * (DRAFT_STEPS - 1 - position)
    applied = product_of((*map(hold, finished), DRAFT_STEP, *map(hold, pending)))
    if not pending:
        return applied
    results = (*finished, *RESULTS_OF_ONE_STEP, *pending)
    drafted = len(finished)
    after_next_reads = drafted + len(RESULTS_OF_ONE_STEP) + len(READS_OF_ONE_POSITION)
    return applied @ route(
        (*range(after_next_reads), drafted, *range(after_next_reads, len(results))),
        results)


def gather_drafts_and_confidences() -> cat.Rearrangement:
    '''The five drafts ahead of the five confidences, each in the order of the steps.'''
    wires = len(RESULTS_OF_ONE_STEP) * DRAFT_STEPS
    return route(
        (*range(0, wires, len(RESULTS_OF_ONE_STEP)),
         *range(1, wires, len(RESULTS_OF_ONE_STEP))),
        RESULTS_OF_ONE_STEP * DRAFT_STEPS)


def draft_chain() -> cat.Block:
    '''The five draft steps in order, each reading the draft of the step before.'''
    return cat.Block.template(
        composition_of((
            read_every_draft_position(),
            *(run_draft_step(position) for position in range(DRAFT_STEPS)),
            gather_drafts_and_confidences())),
        title=text.CHAIN_TITLE, fill_color=DSPARK_COLOUR,
        formula=('(\\mathrm{token}_{k+1}, \\mathrm{conf}_{k}) = \\mathrm{Step}( '
                 '\\lambda[k], g[k], \\mathrm{token}_{k}), \\quad k = 0, \\dots, 4'),
        description=text.DRAFT_CHAIN_DESCRIPTION,
        references=(model_lines(1146, 1156), inference_config_lines(7)))


def accept_token() -> cat.BroadcastedCategory:
    '''The token the backbone produced: the probabilities of the last position, and one
    token sampled from them.'''
    return read_last_position((VOCABULARY,)) @ SAMPLER


def draft_five_tokens() -> cat.BroadcastedCategory:
    '''From the backbone's probabilities and the three stream means of the target
    layers to the five drafts and the five confidences.'''
    return ((accept_token() * project_stream_means())
            @ route((1, 0, 0), (SAMPLED_TOKEN, DRAFT_STATE))
            @ (draft_trunk() * hold(SAMPLED_TOKEN))
            @ route((0, 0, 1), (DRAFT_STATES, SAMPLED_TOKEN))
            @ (head_over_draft_positions() * hold(DRAFT_STATES) * hold(SAMPLED_TOKEN))
            @ draft_chain())


def main_projection_role(member: int) -> OperatorRole:
    return declared_at(
        text.MAIN_PROJECTION_ROLE.format(
            layer=FIRST_TARGET_LAYER + member, member=member), 1113)


def draft_position_explanation(position: int) -> ReindexingExplanation:
    return ReindexingExplanation(
        description=text.DRAFT_POSITION_DESCRIPTION.format(position=position),
        references=(model_lines(1150, 1153),))


def table_key(name: str) -> str:
    '''The text of a name, which is what the explanation tables are keyed by.'''
    return fd.DynamicName.from_str(name).to_bodies()


OPERATOR_ROLES: dict[str, OperatorRole] = {
    **{table_key(main_projection_name(member)): main_projection_role(member)
       for member in range(TARGET_LAYERS)},
    table_key(MARKOV_PROJECTION_NAME): declared_at(
        text.MARKOV_PROJECTION_ROLE, 1081),
    table_key(CONFIDENCE_OF_STATE_NAME): declared_at(
        text.CONFIDENCE_STATE_HALF_ROLE, 1093),
    table_key(CONFIDENCE_OF_EMBEDDING_NAME): declared_at(
        text.CONFIDENCE_MARKOV_HALF_ROLE,
        1093),
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    LAST_POSITION_NAME: ReindexingExplanation(
        description=text.LAST_POSITION_DESCRIPTION,
        references=(model_lines(1010, 1011),)),
    **{table_key(draft_position_name(position)): draft_position_explanation(position)
       for position in range(DRAFT_STEPS)},
}

DRAFT_TRUNK_EXPLANATION = OperatorExplanation(
    title=r'\text{Draft Trunk}',
    formula=(
        r'\mathrm{draft}(\mathrm{main}, t)[i_{S}, i_{m}] = \sum_{i_{n} \in n} '
        r'A[i_{S}, i_{n}]\, \big( (B_{3} \circ B_{2} \circ B_{1})(d;\ \mathrm{main}) '
        r'\big)[i_{S}, i_{n}, i_{m}], \qquad d[0, i_{n}, i_{m}] = E[t, i_{m}], \quad '
        r'd[i_{S}, i_{n}, i_{m}] = E[t_{\mathrm{noise}}, i_{m}] '
        r'\text{ for } i_{S} \geq 1'),
    description=text.DRAFT_TRUNK_DESCRIPTION,
    references=(
        model_lines(1032, 1074), model_lines(1020, 1029), model_lines(1122, 1126),
        model_lines(1131, 1134), model_lines(1275, 1279), model_lines(1144),
        model_lines(142, 149), inference_config_lines(6, 12)))

GENERIC_OPERATOR_EXPLANATIONS: dict[str, OperatorExplanation] = {
    DRAFT_NAME: DRAFT_TRUNK_EXPLANATION,
}

MARKOV_EMBEDDING_EXPLANATION = OperatorExplanation(
    title=r'\text{Markov Embedding}',
    formula=r'E_{\mathrm{mk}}(t)[i_{Z}] = W^{\mathrm{mk}}[t, i_{Z}]',
    description=text.MARKOV_EMBEDDING_DESCRIPTION,
    references=(model_lines(1080), model_lines(1083, 1086),
                inference_config_lines(10)))

EMBEDDING_EXPLANATIONS: dict[str, OperatorExplanation] = {
    f'E{table_key(MARKOV_EMBEDDING_NAME)}': MARKOV_EMBEDDING_EXPLANATION,
}
