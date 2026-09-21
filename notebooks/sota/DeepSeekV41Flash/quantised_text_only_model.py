# Claude Opus 5 (1M context), effort high. Rewritten by Claude Fable 5.1, effort 80,
# on 2026-09-20, to the quantisations declared by the released code, line by line.
'''The text-only DeepSeek-V4.1-Flash with a quantisation on every wire and a cast
wherever a value changes quantisation.

`text_only_model.v41_flash_text_only` is written in the reals, with the three round
trips of the quantised caches as the only places a quantisation appears. A
quantisation is the number format a value is held in, together with the size of that
format in bits. `RELEASED_POLICY` states the quantisations of the released model, read
from `inference/model.py`, `inference/kernel.py`, `inference/convert.py` and
`inference/generate.py` at the commit pinned by `reference_links.py`, and
`quantization.processing.quantise_model` writes them onto every wire of the
expression, putting a `TypeConvert` named `cast` wherever an operation requires a
quantisation not carried by the value. Every wire of the result is labelled with its
quantisation, and every change of quantisation is an operation in the figure.

The released code holds BF16 on every tensor passed from one module or kernel to the
next, computes in FP32 inside a module upcasting on entry and inside every kernel, and
rounds an activation in one place, in front of a projection whose weight is FP8 or
FP4. That rounding is to MXFP8, which is E4M3 with one UE8M0 scale per 32 channels:
the rounding kernel takes the largest magnitude of each group of 32 channels, rounds
that magnitude over 448 up to a power of two, divides the group by it and rounds each
channel to E4M3, and the GEMM multiplies its FP32 accumulator by the scale of the
operand and the scale of the weight. A cast to bare E4M3 would saturate above 448 and
flush below 2^-9, so the wire in front of such a projection carries MXFP8, and the
inspection box over the cast states the scale. A weight held in FP8 is E4M3 with one
UE8M0 scale per 32 by 32 block, and a routed expert is MXFP4, which is E2M1 with one
UE8M0 scale per 32 channels. The n-gram table of Engram is MXFP8, and a row read from
it reaches the key and value projections with no cast, because rounding a row that
already carries the scale changes no value. Four boxes depart from the pattern and
`BOX_POLICIES` names them: the attention kernel keeps its accumulators in FP32 and
casts the probabilities to BF16 for the second matrix multiply, the indexer scores in
BF16 from end to end, and the router and the mixing coefficients hand FP32 on. The
residual stream is written back in BF16 at the end of every sublayer, which is a block
of the expression rather than a box, so `BLOCK_RESULTS` names the title of the
sublayer.

The three caches are the other place the released code rounds, and the rounding is
written out by `quantised_caches` rather than by a cast: the kernel is called in place,
so it computes the scale of each group, rounds the group through E2M1 or E4M3, and
multiplies the group by its scale again before the cache is written. The cache holds
BF16 numbers carrying the rounding, and no scale is kept beside them, because the
value stored is already the element multiplied by its scale.

`with_quantised_explanation_tables` adds the row of a cast to the tables of the
text-only model. `quantised_caches.explain_cast` already explains a cast into or out of
a stored form of a round trip, naming the numbers that form holds and the released
lines of the rounding kernel. Every other cast takes
`notebooks.display.cast_presentation.cast_explanation`, which names the quantisation
the cast reads and the one it writes and says what the pair of formats does to a
value. A cast is drawn thin, so the box opens from the coloured format on the wire.
`quantised_operator_roles` writes the quantisation of each weight into the role shown
by an inspection box over it, so a figure states the quantisation of a weight, which
has no wire.

`notebooks/sota/DeepSeekV41Flash.ipynb` draws the model and its parts and asserts
its claims.
'''
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import para.data_structure.ParaWrap as para_wrap
import quantization.algebra.strip_quantisations as strip_quantisations
import quantization.data_structure.Quantization as Quantization
import quantization.processing.quantise_model as quantise_model
import term_utilities.term_utilities as tutil

import notebooks.display.cast_presentation as cast_presentation
import notebooks.display.notebook_diagrams as notebook_diagrams
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
import notebooks.sota.DeepSeekV41Flash.operator_explanations as operator_explanations
import notebooks.sota.DeepSeekV41Flash.quantised_caches as quantised_caches
import notebooks.sota.DeepSeekV41Flash.integrated_explanations as integrated_explanations
import notebooks.sota.DeepSeekV41Flash.text_only_model as text_only_model
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    convert_lines, engram_lines, generate_lines, kernel_lines, model_lines,
    top_level_config_lines)
from websocket_transfer.auxiliary_information import OperatorRole

BF16 = Quantization.BF16
FP32 = Quantization.FP32
E4M3 = Quantization.E4M3
E2M1 = Quantization.E2M1
UE8M0 = Quantization.UE8M0
INT64 = Quantization.INT64
INT32 = Quantization.INT32
MXFP8 = Quantization.MXFP8
MXFP4 = Quantization.MXFP4
FP8_WEIGHT = Quantization.block_scaled(E4M3, UE8M0, channels=32, rows=32)
'''A weight held in FP8 by the released code: E4M3 elements with one UE8M0 scale per
32 by 32 block, declared at `inference/model.py` lines 219 to 224 and read by the GEMM
at `inference/kernel.py` lines 256 to 269.'''

ROUTED_EXPERT_PROJECTIONS = ('W^{G}', 'W^{U}', 'W^{D}')
MIXING_PROJECTIONS = ('H0', 'H1', 'H2')
COMPRESSOR_PROJECTIONS = ('W^{C}', 'W^{Z}')
STREAM_WEIGHTS = ('\\mathrm{w}{1}', '\\mathrm{w}{14}')
INDEXER_BF16_PROJECTIONS = ('k^{I}', 'w^{I}')
NGRAM_TABLES = tuple(
    f'E\\mathrm{{ng}}{{{layer}}}' for layer in omitted_mechanisms.ENGRAM_LAYERS)

RELEASED_WEIGHTS: dict[str, Quantization.Quantified] = {
    **{name: MXFP4 for name in ROUTED_EXPERT_PROJECTIONS},
    **{name: FP32 for name in MIXING_PROJECTIONS},
    **{name: FP32 for name in COMPRESSOR_PROJECTIONS},
    **{name: FP32 for name in STREAM_WEIGHTS},
    **{name: BF16 for name in INDEXER_BF16_PROJECTIONS},
    **{name: MXFP8 for name in NGRAM_TABLES},
    'E': BF16,
    'W^{Oa}': BF16,
    'W^{R}': FP32,
    '\\mathrm{bias}': FP32,
    '\\mathrm{sink}': FP32,
    'L': FP32,
    'E\\mathrm{cmp}': INT64,
}
'''The read quantisation of every weight not declared at the default of the released
code, which is `FP8_WEIGHT`. The n-gram tables hold one UE8M0 scale per 32 channels of
a row, at `inference/model.py` lines 307 to 310, so a table is MXFP8 rather than the
weight form. The token map of Engram is a table of integers, built from Python
integers at `inference/engram.py` line 155, which `torch.tensor` holds in `int64`.'''

FILE_QUANTISATIONS: dict[str, Quantization.Quantified] = {
    'W^{R}': BF16,
    '\\mathrm{w}{1}': BF16,
    '\\mathrm{w}{14}': BF16,
    'L': BF16,
    'W^{Oa}': FP8_WEIGHT,
    'W^{C}': BF16,
}
'''The quantisation of a weight in the checkpoint where that differs from its read
quantisation. The weight of the router and the stream weights of Engram are BF16
parameters read through `.float()`, the output head is loaded into FP32 so the logits
come out in FP32, the first output projection is FP8 with scales in the file and is
written out in BF16 by `convert.py`, and the projection of the compressor is BF16 in the
file and promoted to FP32 where the compressor pools.'''

ATTENTION_KERNEL_BOX = 'Core'
INDEXER_SCORES_BOX = 'Sco'
CANDIDATE_POOL_BOX = 'Pool'
GATHER_BOX = 'Gth'
ROUTER_BOX = 'Gate'
COEFFICIENTS_BOX = 'Coef'

BOX_POLICIES: dict[str, quantise_model.BoxPolicy] = {
    ATTENTION_KERNEL_BOX: quantise_model.BoxPolicy(
        contractions=quantise_model.ContractionQuantisation.ACCUMULATED),
    INDEXER_SCORES_BOX: quantise_model.BoxPolicy(
        arithmetic=quantise_model.ArithmeticQuantisation.CARRIED),
    CANDIDATE_POOL_BOX: quantise_model.BoxPolicy(
        arithmetic=quantise_model.ArithmeticQuantisation.CARRIED,
        integer_results=INT32),
    GATHER_BOX: quantise_model.BoxPolicy(integer_operands=INT32),
    ROUTER_BOX: quantise_model.BoxPolicy(results=FP32),
    COEFFICIENTS_BOX: quantise_model.BoxPolicy(results=FP32),
}
'''The sites at which the released code departs from its pattern. The attention
kernel reads BF16 and keeps its two accumulators in FP32. The scoring of the indexer
and the candidate pool are eager code over BF16 tensors, never upcasting, and the
candidate pool returns its positions in `int32`. The gather of the selected entries
stands for the read the attention kernel makes at the positions given to it, and
that kernel declares its positions as `int32`, which the indexer converts them to
before handing them over. The router and the mixing kernel return FP32, multiplied in
FP32 by the experts and the stream mixes.'''

BLOCK_RESULTS: dict[str, tuple[Quantization.Quantified | None, ...]] = {
    text.MHC_TITLE: (None, BF16)}
'''A sublayer returns the collapse vector read by the next sublayer and the residual
stream. The released `Block.forward` returns the residual written back in BF16 and the
collapse vector as computed by the mixing kernel, in FP32.'''

SLOT_QUANTISATIONS: dict[str, Quantization.Quantified] = {
    '\\mathrm{pool}': INT32,
    '\\mathrm{sel}b': INT32,
    '\\mathrm{sel}B': INT32,
}
'''The slots of the tape holding positions in `int32`: the candidate pool returned
by `_candidate_pool` and the positions selected by the indexer, which it converts
before returning them and which the reuse layers read from the cache.'''

RELEASED_POLICY = quantise_model.QuantizationPolicy(
    inputs=BF16,
    activations=BF16,
    scalars=FP32,
    rounded_operands=MXFP8,
    results=None,
    weights=RELEASED_WEIGHTS,
    weights_by_default=FP8_WEIGHT,
    boxes=BOX_POLICIES,
    block_results=BLOCK_RESULTS,
    slots=SLOT_QUANTISATIONS)

QUANTISED = quantise_model.quantise_model(
    text_only_model.v41_flash_text_only, RELEASED_POLICY)

v41_flash_text_only_quantised = QUANTISED.morphism

WEIGHT_QUANTISATIONS: Mapping[str, Quantization.Quantified] = (
    QUANTISED.weight_quantisations)

v41_flash_text_only_without_quantisations = strip_quantisations.strip_quantisations(
    v41_flash_text_only_quantised)
'''The quantised model with every quantisation taken back off it, which is the
arithmetic of the text-only model with no format on any wire and no conversion
anywhere. `quantization.algebra.strip_quantisations` states the functor. The three
cache round trips lose the conversions written into them by `quantised_caches` as
well, because each of those reads one quantisation of a channel into another, so the
result is the model of `text_only_model` with its three round trips reduced to the
scaling they wrap.'''


def released_assigned_sizes() -> dict[str, int]:
    '''The released size of every symbol of the quantised model, as given by
    `text_only_model.released_assigned_sizes` for the source model.'''
    return text_only_model.released_assigned_sizes(v41_flash_text_only_quantised)


def part_named(name: str,
               model: cat.Morphism = v41_flash_text_only_quantised) -> cat.Morphism:
    '''The first box of `model` whose operator carries `name`, drawn as returned by
    the pass.

    A box whose block holds tape seeds stands inside a `ParaWrap` naming the slot
    carried by each of its taped ports, and a figure draws the wrap, so a wrap
    is looked for first.

    A model holding one mechanism at two quantisations holds two boxes of that name,
    because the pass quantises each body at the quantisations carried by its own
    operands. The first in structural order is the one shown by a figure, and the
    caption says its site.
    '''
    for wrap in tutil.type_search(para_wrap.ParaWrap, model):
        if _is_named(wrap.body, name):
            return wrap
    for node in boxes_named(name, model):
        return node
    raise NameIsNotABoxOfTheModel(f'{name} names no box of the model')


def _is_named(morphism: cat.Morphism, name: str) -> bool:
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, ops.BlockOperator)
            and morphism.operator.name is not None
            and morphism.operator.name.to_bodies() == name)


class NameIsNotABoxOfTheModel(ValueError):
    '''A name asked of the quantised model and carried by no box of it.'''


def boxes_named(name: str, model: cat.Morphism = v41_flash_text_only_quantised
                ) -> tuple[cat.Broadcasted, ...]:
    '''Every box of `model` whose operator carries `name`.'''
    return tuple(
        node for node in tutil.type_search(cat.Broadcasted, model)
        if _is_named(node, name))


def part_titled(title: str, model: cat.Morphism = v41_flash_text_only_quantised
                ) -> cat.Block:
    '''The first block of `model` carrying `title`, which is how a part the model
    composes without boxing it is asked for. `part_named` asks for a box instead.'''
    for block in tutil.type_search(cat.Block, model):
        if (block.block_tag.aesthetics is not None
                and block.block_tag.aesthetics.title == title):
            return block
    raise TitleIsNotABlockOfTheModel(f'{title} titles no block of the model')


class TitleIsNotABlockOfTheModel(ValueError):
    '''A title asked of the quantised model and carried by no block of it.'''


LAYER_STACK_TITLES: frozenset[str] = frozenset(
    factor.block_tag.aesthetics.title
    for factor in text_only_model.text_only_stack.content
    if isinstance(factor, cat.Block) and factor.block_tag.aesthetics is not None)
'''The title of every block the layer stack composes at its top level, read off
`text_only_model.text_only_stack` so that the two cannot disagree.'''


def _layer_stack_positions(model: cat.Composed) -> tuple[int, int]:
    titled = [
        position for position, factor in enumerate(model.content)
        if isinstance(factor, cat.Block) and factor.block_tag.aesthetics is not None
        and factor.block_tag.aesthetics.title in LAYER_STACK_TITLES]
    if not titled:
        raise NoLayerStandsInTheModel(
            f'no factor of the model is titled by one of {sorted(LAYER_STACK_TITLES)}')
    return titled[0], titled[-1]


class NoLayerStandsInTheModel(ValueError):
    '''A model asked for its layer stack and holding no layer.'''


def part_before_the_layers(
    model: cat.Composed = v41_flash_text_only_quantised,
) -> cat.Morphism:
    '''The head of `model`: everything the model computes before its first layer,
    which is the embedding, the drop of the token identifiers, the expansion into the
    four residual streams and the collapse vector the first sublayer starts from.'''
    first, _ = _layer_stack_positions(model)
    return cat.Composed.from_iter(model.content[:first])


def part_after_the_layers(
    model: cat.Composed = v41_flash_text_only_quantised,
) -> cat.Morphism:
    '''The tail of `model`: everything the model computes after its last layer, which
    is the collapse of the four streams, the normalisation, the output head and the
    softmax over the vocabulary.'''
    _, last = _layer_stack_positions(model)
    return cat.Composed.from_iter(model.content[last + 1:])


# ==========================================================================
# The policy as a table.
# ==========================================================================
POLICY_TABLE_HEADER: tuple[str, str] = (
    '| quantisation | what it applies to | released lines |',
    '|---|---|---|')


def _link(reference: cat.CodeReference) -> str:
    return (reference.label if reference.url is None
            else f'[{reference.label}]({reference.url})')


def _lines(*references: cat.CodeReference) -> str:
    return ', '.join(map(_link, references))



POLICY_ROWS: tuple[tuple[Quantization.Quantified, str, str], ...] = (
    (BF16,
     'every tensor returned by a module or a kernel: the result of a matrix multiply, '
     'the result of an RMSNorm, the residual stream written back at the end of a '
     'sublayer, and the value returned by every box unless a row below names another',
     _lines(kernel_lines(299), kernel_lines(583), model_lines(293), model_lines(960),
            model_lines(966), model_lines(904), model_lines(365), model_lines(485),
            generate_lines(118))),
    (FP32,
     'the arithmetic inside a module upcasting on entry and inside every kernel: the '
     'router, the SwiGLU of an expert, the mixing coefficients, the gate of Engram, '
     'the pooling of a compressor, the logits, the attention kernel and the rounding '
     'kernels of the caches',
     _lines(model_lines(811), model_lines(843, 848), model_lines(952, 954),
            model_lines(355, 365), model_lines(464), model_lines(1012),
            kernel_lines(51), kernel_lines(135), kernel_lines(339, 385))),
    (MXFP8,
     'the operand of a projection whose weight is FP8 or FP4, which is the one place '
     'the released code rounds an activation: the rounding kernel writes the E4M3 '
     'elements and one UE8M0 scale per 32 channels, and the GEMM multiplies its FP32 '
     'accumulator by that scale and the scale of the weight',
     _lines(model_lines(181, 206), kernel_lines(41, 96), kernel_lines(256, 269),
            kernel_lines(544, 553))),
    (MXFP8,
     'the n-gram tables of Engram, whose rows reach the key and value projections '
     'with no cast, because the released lookup multiplies a row by its scales into '
     'BF16 and the projection rounds it back with the same groups of 32, which '
     'changes no value',
     _lines(model_lines(307, 310), model_lines(316, 321), model_lines(353))),
    (FP32,
     'the values returned by the router and the mixing coefficients, multiplied in '
     'FP32 by the expert weights and the stream mixes, and the probabilities returned '
     'by the model',
     _lines(model_lines(811, 827), model_lines(850), kernel_lines(414, 419),
            model_lines(959), model_lines(965), model_lines(1291))),
    (BF16,
     'the queries and the cached latents read by the attention kernel, and the '
     'probabilities cast down by it for the product with the values, while its two '
     'accumulators stay in FP32',
     _lines(kernel_lines(329, 330), kernel_lines(339), kernel_lines(377),
            kernel_lines(385))),
    (BF16,
     'the indexer scores from end to end: the einsum of the queries against the keys, '
     'the rectification, and the sum under the head weights. The keys and queries '
     'were rounded to four bits and written back as BF16, and the key cache is a '
     'BF16 buffer, so the reference implementation scores in BF16 and takes neither '
     'the smaller cache nor the faster scoring that the four-bit form allows',
     _lines(model_lines(521, 525), model_lines(546), model_lines(552),
            model_lines(555, 557))),
    (MXFP4,
     'the gate, up and down projections of a routed expert, with one UE8M0 scale per '
     '32 channels along the contracted axis',
     _lines(model_lines(872, 880), top_level_config_lines(106), model_lines(219, 224))),
    (FP8_WEIGHT,
     'every weight declared by the released code at its default: the query, key and '
     'second output projections of the attention, the indexer queries, the shared '
     'expert, and the key and value projections of Engram, with one UE8M0 scale per '
     'block of 32 rows by 32 channels',
     _lines(model_lines(26), model_lines(1191), model_lines(640, 650),
            model_lines(514), model_lines(887), model_lines(345),
            model_lines(225, 232))),
    (BF16,
     'the token embedding, the first output projection, written out of its FP8 file '
     'form by the conversion, and the indexer key and head-weight projections',
     _lines(model_lines(166), convert_lines(157, 173), model_lines(787),
            model_lines(518), model_lines(515))),
    (FP32,
     'the mixing projections and the compressor projections, the router weight and '
     'the stream weights of Engram, which are BF16 in the file and read through '
     '`.float()`, the correction bias, the sink logit, and the output head',
     _lines(model_lines(940, 946), model_lines(446, 448), model_lines(805, 811),
            model_lines(347, 356), model_lines(806), model_lines(639),
            model_lines(1006))),
    (BF16,
     'the compressed entries, the indexer keys and the window latents in the caches, '
     'which hold the value returned by a rounding kernel called in place: each group '
     'of channels is divided by its scale, rounded through E2M1 or E4M3 and '
     'multiplied by its scale again, so the cache holds BF16 numbers carrying the '
     'rounding and keeps no scale beside them',
     _lines(model_lines(705, 707), model_lines(546), model_lines(758, 761),
            kernel_lines(81, 86), kernel_lines(167, 172),
            *quantised_caches.CAST_REFERENCES[Quantization.Encoding.E2M1][1:])),
    (INT64,
     'every index: the token identifiers on arrival, the token map and the n-gram '
     'hash of Engram, the positions picked by a top-k selection, and the routing of '
     'the experts',
     _lines(generate_lines(51, 53), engram_lines(64, 83), engram_lines(155, 157),
            model_lines(579), model_lines(822))),
    (INT32,
     'the positions returned by the candidate pool, and the positions read by the '
     'attention kernel, which the indexer converts before handing them over',
     _lines(model_lines(425, 426), model_lines(580), model_lines(1029),
            kernel_lines(333))),
)


def policy_table() -> str:
    '''The markdown table of the released policy, one row per quantisation and site,
    for a notebook cell.'''
    rows = tuple(
        f'| {Quantization.describe_quantisation(quantisation)} | {applies_to} | {line} |'
        for quantisation, applies_to, line in POLICY_ROWS)
    return '\n'.join((*POLICY_TABLE_HEADER, *rows))


WEIGHT_TABLE_HEADER: tuple[str, str] = (
    '| weight | read quantisation | quantisation in the file |', '|---|---|---|')


def weight_table() -> str:
    '''The markdown table of the quantisation of every weight held by the quantised
    model. A weight has no wire, so the figure cannot label it and the table and the
    inspection box over the operator say the quantisation instead. The third column
    names the quantisation held in the file where that differs.'''
    rows = tuple(
        f'| ${name}$ | {Quantization.describe_quantisation(quantisation)} | {_file_quantisation(name)} |'
        for name, quantisation in sorted(WEIGHT_QUANTISATIONS.items()))
    return '\n'.join((*WEIGHT_TABLE_HEADER, *rows))


def _file_quantisation(name: str) -> str:
    return ('the same' if name not in FILE_QUANTISATIONS
            else Quantization.describe_quantisation(FILE_QUANTISATIONS[name]))


# ==========================================================================
# What an inspection box shows over a cast and over a weight.
# ==========================================================================
ROUNDING_KERNEL_FORMULA = (
    r'\begin{gathered}'
    r'a[i_{g}] = \max_{i_{y} \in y} \lvert x[32\, i_{g} + i_{y}] \rvert \\'
    r's[i_{g}] = 2^{\lceil \log_{2} (\max(a[i_{g}],\, 10^{-4}) / 448) \rceil} \\'
    r'\hat{x}[32\, i_{g} + i_{y}] = \mathrm{E4M3}\big(\ulcorner x[32\, i_{g} + i_{y}] '
    r'/ s[i_{g}] \lrcorner_{-448}^{448}\big)'
    r'\end{gathered}')
'''The rounding kernel in front of a projection whose weight is FP8 or FP4, in index
notation: `g` is the groups of 32 channels of a row, `y` the channels of one group,
`a` the largest magnitude of a group, `s` its scale and `\\hat{x}` the E4M3 element
written beside `s`.'''

ROUNDING_KERNEL_DESCRIPTION = (
    text.ROUNDING_KERNEL_DESCRIPTION)


def explain_block_scaled_cast(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row of an inspection box over a cast into MXFP8, which is the rounding
    kernel of the released code, with its formula and its released lines. `None` for
    any other operator.'''
    operator = target.operator
    if not isinstance(operator, Quantization.TypeConvert):
        return None
    written = Quantization.quantisation_of(operator.target)
    read = Quantization.quantisation_of(operator.source)
    if written != MXFP8 or read is None or read.scale is not None:
        return None
    return OperatorExplanation(
        title=r'\text{Cast to MXFP8}',
        formula=ROUNDING_KERNEL_FORMULA,
        description=ROUNDING_KERNEL_DESCRIPTION,
        references=(kernel_lines(41, 96), model_lines(181, 206),
                    kernel_lines(256, 269), kernel_lines(544, 553)))


def explain_quantised_cast(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row of an inspection box over a cast. A cast into or out of a stored form
    of a cache round trip takes the row written for it by `quantised_caches`, which
    names the numbers held by that form and the released lines of the rounding
    kernel. A cast into MXFP8 takes `explain_block_scaled_cast`, which states the
    scale of each group and the kernel computing it. Every other cast takes the row
    `cast_presentation.cast_explanation` writes from the operator, which names the
    quantisation the cast reads and the one it writes.'''
    return (operator_explanations.explain_any_cast(target)
            or explain_block_scaled_cast(target)
            or cast_presentation.cast_explanation(target))


QUANTISED_OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    **integrated_explanations.TEXT_ONLY_OPERATOR_EXPLANATIONS,
    Quantization.TypeConvert: explain_quantised_cast,
}


def _role_with_its_quantisation(
    name: str, role: OperatorRole, quantisation: Quantization.Quantified,
) -> OperatorRole:
    held = ('' if name not in FILE_QUANTISATIONS
            else ' ' + text.FILE_QUANTISATION_SENTENCE.format(
                quantisation=Quantization.describe_quantisation(
                    FILE_QUANTISATIONS[name])))
    read = text.WEIGHT_QUANTISATION_SENTENCE.format(
        quantisation=Quantization.describe_quantisation(quantisation))
    return dataclasses.replace(role, role=f'{role.role} {read}{held}')


def quantised_operator_roles() -> dict[str, OperatorRole]:
    '''The roles of the text-only model, with the quantisation of each weight written
    into the sentence shown by the inspection box over that weight.

    A weight has no wire, so nothing in the figure states its quantisation. The role
    is the one place a reader already looks for the meaning of a weight, which is
    where the quantisation belongs.
    '''
    roles = dict(integrated_explanations.OPERATOR_ROLES)
    return {
        name: (_role_with_its_quantisation(name, role, WEIGHT_QUANTISATIONS[name])
               if name in WEIGHT_QUANTISATIONS else role)
        for name, role in roles.items()}


def with_quantised_explanation_tables(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` with the tables of the quantised text-only model: the tables of the
    text-only model, the row of a cast, and the roles carrying the quantisation of
    each weight.'''
    return dataclasses.replace(
        integrated_explanations.with_text_only_explanation_tables(settings),
        operator_explanations=QUANTISED_OPERATOR_EXPLANATIONS,
        operator_roles=quantised_operator_roles())


# ==========================================================================
# What the model holds once it is quantised.
# ==========================================================================
def cast_counts() -> Mapping[tuple[str, str], int]:
    '''How many casts of the quantised model read each pair of formats, counted as
    written operations.'''
    return quantise_model.cast_counts(v41_flash_text_only_quantised)


def unquantised_weaves() -> fd.Prod[cat.Weave]:
    '''Every wire of the quantised model holding a real number and carrying no
    quantisation, which is none.'''
    return quantise_model.unquantised_weaves(v41_flash_text_only_quantised)


def operators_without_a_rule() -> fd.Prod[type[cat.Operator]]:
    '''Every operator class of the model given no rule by the registry, which is
    none.'''
    return quantise_model.leaves_without_a_rule(text_only_model.v41_flash_text_only)
