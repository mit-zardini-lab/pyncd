# Claude Opus 5.5 (1M context), effort 40.
'''GLM-5.3 with a quantisation on every wire and a cast wherever a value changes
quantisation, as `transformers` runs the FP8 checkpoint.

`whole_model.glm53` is written in the reals. A quantisation is the number format a
value is held in, together with the size of that format in bits. Z.ai publishes the
model twice: `zai-org/GLM-5.3`, whose `config.json` declares the fine-grained FP8
quantisation, and `zai-org/GLM-5.3-BF16`, which holds every weight in BF16. The
quantised model here is the FP8 checkpoint as the reference implementation runs it.
`RELEASED_POLICY` states its quantisations, read from `modeling_glm_moe_dsa.py`, from
the FP8 integration of `transformers` at the same commit, from the two kernels that
integration loads, and from the headers of the checkpoint's shards, all pinned in
`reference_links`. `quantization.processing.quantise_model` writes them onto every
wire of the expression, putting a `TypeConvert` named `cast` wherever an operation
requires a quantisation not carried by the value.

The checkpoint holds its projections in E4M3 with one FP32 scale per block of 128 rows
by 128 channels, `FP8_BLOCK_WEIGHT`. The quantizer replaces every `nn.Linear` and the
routed experts with `FP8Linear` and `FP8Experts`, except the modules the configuration
lists as unconverted and the indexer's head weights, which the model keeps out of FP8.
The embedding, the output head and the indexer's head weights stay BF16, and the router
weight is BF16 in the file and read into FP32 by the router. The model is loaded in the
BF16 given by the configuration, so every tensor passed from one module to the next is
BF16. An FP8 projection rounds its BF16 operand to E4M3 with one FP32 scale per 128
channels of a token, `FP8_ROUNDED_OPERAND`, multiplies in E4M3 with an FP32
accumulator, and writes BF16.

Eight boxes depart from the default rules, and `BOX_POLICIES` names them. The rotary
table is cast to BF16 and the turn is computed in BF16. The indexer scores in FP32 and
hands its scores to the top-2048 in FP32. The attention core is the memory-efficient
kernel of `scaled_dot_product_attention` in PyTorch 2.10.0 built for CUDA 12.8, which
reads BF16 and keeps its scores in FP32. The kernel rounds the exponentials of the
scores to BF16 for the product with the values and divides by their FP32 sum
afterwards. The expression writes the softmax as
one operation, so its cast to BF16 stands after the division. The dense MLP and the
experts apply the SiLU and the product of the two branches in BF16. The experts cast
the gates to BF16 before the product with their outputs, and `torch.sum` adds the eight
products in an FP32 accumulator and returns BF16. The router returns FP32. The selection shared through the tape is held
in `int32`, which is how the indexer returns it and how a Shared layer receives it.

`with_quantised_explanation_tables` adds the row of a cast and the quantisation of each
weight to the tables of `operator_explanations`, so every cast and every weight opens
an inspection box. `notebooks/website/modern/GLM53.ipynb` draws the model and writes
it to its page, and `validate_quantised_glm53.py` checks the claims made about it.
'''
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import data_structure.Category as cat
import data_structure.Term as fd
import quantization.algebra.strip_quantisations as strip_quantisations
import quantization.data_structure.Quantization as Quantization
import quantization.processing.quantise_model as quantise_model
from websocket_transfer.auxiliary_information import OperatorRole

import notebooks.display.cast_presentation as cast_presentation
import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.sota.GLM53.feed_forward as feed_forward
import notebooks.sota.GLM53.lightning_indexer as lightning_indexer
import notebooks.sota.GLM53.mixture_of_experts as mixture_of_experts
import notebooks.sota.GLM53.multi_latent_attention as multi_latent_attention
import notebooks.sota.GLM53.operator_explanations as operator_explanations
import notebooks.sota.GLM53.rotary_embedding as rotary_embedding
import notebooks.sota.GLM53.whole_model as whole_model
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text
from notebooks.sota.GLM53.declared_axes import SLOT_SELECTION
from notebooks.sota.GLM53.reference_links import (
    checkpoint_config_lines, checkpoint_file, declared_at, deep_gemm_lines,
    finegrained_fp8_lines, library_lines, modeling_lines, pytorch_lines,
    PYTORCH_ENVIRONMENT_REFERENCES)

BF16 = Quantization.BF16
FP32 = Quantization.FP32
E4M3 = Quantization.E4M3
INT64 = Quantization.INT64
INT32 = Quantization.INT32

WEIGHT_BLOCK = 128
FP8_BLOCK_WEIGHT = Quantization.block_scaled(
    E4M3, FP32, channels=WEIGHT_BLOCK, rows=WEIGHT_BLOCK)
'''A projection of the FP8 checkpoint: E4M3 elements with one FP32 inverse scale per
block of 128 rows by 128 channels, declared by `quantization_config` at `config.json`
lines 224 to 231, allocated by `FP8Linear` at `integrations/finegrained_fp8.py` lines
305 to 317, and stored as `F8_E4M3` beside an `F32` `weight_scale_inv` in every shard.
A block may be cut short where the rows of a weight are not a multiple of 128, as the
576 rows of `kv_a_proj_with_mqa` are. The expression gives each part of `q_b_proj` and
`kv_b_proj` a weight of its own, and a block of 128 rows of the joined weight can hold
rows of two parts, such as the last 64 unturned rows and the 64 turned rows of one
query head.'''

FP8_ROUNDED_OPERAND = Quantization.block_scaled(E4M3, FP32, channels=WEIGHT_BLOCK)
'''The operand of an FP8 projection: E4M3 elements with one FP32 scale per 128
consecutive channels of a token. DeepGEMM writes it through `per_token_cast_to_fp8`
with `gran_k=128` and no power-of-two rounding of the scale, at `integrations/deepgemm.py`
lines 444 and 607, and the Triton kernel writes the same format inline, at
`finegrained-fp8 utils.py` lines 433 to 442.'''

UNCONVERTED_BF16_WEIGHTS = ('E', 'W^{L}', 'w^{I}')
RELEASED_WEIGHTS: dict[str, Quantization.Quantified] = {
    **{name: BF16 for name in UNCONVERTED_BF16_WEIGHTS},
    'W^{R}': FP32,
    mixture_of_experts.ROUTER_BIAS_NAME: FP32,
}
'''The read quantisation of every weight not held at `FP8_BLOCK_WEIGHT`. The embedding
and the output head are listed as unconverted by the configuration. The indexer's head
weights are kept out of FP8 by `_keep_in_fp32_modules`, which upcasts them to FP32 only
for an FP16 load, so a BF16 load reads them in BF16. The router weight is read into
FP32 by the router, and the correction bias is an FP32 buffer.'''

FILE_QUANTISATIONS: dict[str, Quantization.Quantified] = {'W^{R}': BF16}
'''The quantisation of a weight in the checkpoint where that differs from its read
quantisation. The router weight is BF16 in the shards and read through
`.type(torch.float32)`.'''

ROTATION_BOX = rotary_embedding.ROTATION_BOX
INDEXER_BOX = lightning_indexer.INDEXER_BOX
INDEXER_SCORES_BOX = lightning_indexer.SCORE_BOX
ATTENTION_CORE_BOX = multi_latent_attention.CORE_BOX
GATHER_BOX = multi_latent_attention.GATHER_BOX
DENSE_MLP_BOX = feed_forward.DENSE_BOX
MIXTURE_BOX = mixture_of_experts.MIXTURE_BOX
ROUTER_BOX = mixture_of_experts.GATE_BOX

BOX_POLICIES: dict[str, quantise_model.BoxPolicy] = {
    ROTATION_BOX: quantise_model.BoxPolicy(
        arithmetic=quantise_model.ArithmeticQuantisation.CARRIED),
    INDEXER_SCORES_BOX: quantise_model.BoxPolicy(
        results=FP32, contractions=quantise_model.ContractionQuantisation.SCALAR),
    INDEXER_BOX: quantise_model.BoxPolicy(results=FP32),
    ATTENTION_CORE_BOX: quantise_model.BoxPolicy(
        contractions=quantise_model.ContractionQuantisation.ACCUMULATED),
    GATHER_BOX: quantise_model.BoxPolicy(integer_operands=INT64),
    DENSE_MLP_BOX: quantise_model.BoxPolicy(
        arithmetic=quantise_model.ArithmeticQuantisation.CARRIED),
    MIXTURE_BOX: quantise_model.BoxPolicy(
        arithmetic=quantise_model.ArithmeticQuantisation.CARRIED,
        contractions=quantise_model.ContractionQuantisation.FEWEST_BITS),
    ROUTER_BOX: quantise_model.BoxPolicy(results=FP32),
}
'''The boxes departing from the default rules, each for the lines `POLICY_ROWS`
cites. The rotary table is cast to the BF16 of the hidden state handed to the rotary
embedding and the turn is BF16 arithmetic. The indexer reads its queries and keys into FP32 before scoring and
returns FP32 scores, and so does the box around it. The attention core is a fused
kernel. The gather at the selected tokens reads the positions through `.long()`. The
dense MLP and the mixture apply their elementwise maps in BF16, and the mixture casts
the FP32 gates to the BF16 of the experts before the sum. The router returns FP32.'''

SLOT_QUANTISATIONS: dict[str, Quantization.Quantified] = {
    SLOT_SELECTION.uid._name.to_bodies(): INT32}
'''The selection is held in `int32` on the tape, because the indexer converts the
positions picked by `topk` to `int32` before returning them and the layer hands that
value on to the Shared layers after it.'''

RELEASED_POLICY = quantise_model.QuantizationPolicy(
    inputs=BF16,
    activations=BF16,
    scalars=FP32,
    rounded_operands=FP8_ROUNDED_OPERAND,
    results=BF16,
    weights=RELEASED_WEIGHTS,
    weights_by_default=FP8_BLOCK_WEIGHT,
    boxes=BOX_POLICIES,
    integers=INT64,
    slots=SLOT_QUANTISATIONS)

QUANTISED = quantise_model.quantise_model(whole_model.glm53, RELEASED_POLICY)

glm53_quantised = QUANTISED.morphism

WEIGHT_QUANTISATIONS: Mapping[str, Quantization.Quantified] = (
    QUANTISED.weight_quantisations)

glm53_without_quantisations = strip_quantisations.strip_quantisations(glm53_quantised)
'''The quantised model with every quantisation taken back off it, which is the
arithmetic of `whole_model.glm53` with no format on any wire and no conversion
anywhere.'''


def released_assigned_sizes() -> dict[str, int]:
    '''The released size of every symbol of the quantised model.'''
    return whole_model.released_assigned_sizes(glm53_quantised)


# ==========================================================================
# The policy as a table.
# ==========================================================================
POLICY_TABLE_HEADER: tuple[str, str] = (
    '| quantisation | what it applies to | lines |', '|---|---|---|')


def _link(reference: cat.CodeReference) -> str:
    return (reference.label if reference.url is None
            else f'[{reference.label}]({reference.url})')


def _lines(*references: cat.CodeReference) -> str:
    return ', '.join(map(_link, references))


FP8_LINEAR = 'integrations/finegrained_fp8.py'
MEMORY_EFFICIENT_ATTENTION = (
    'aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h')

ATTENTION_KERNEL_REFERENCES: tuple[cat.CodeReference, ...] = (
    *PYTORCH_ENVIRONMENT_REFERENCES,
    pytorch_lines('aten/src/ATen/Context.h', 458, 463),
    pytorch_lines('aten/src/ATen/native/transformers/cuda/sdp_utils.cpp', 781),
    pytorch_lines('aten/src/ATen/native/transformers/sdp_utils_cpp.h', 259, 267),
    pytorch_lines('aten/src/ATen/native/transformers/cuda/attention.cu', 1615, 1617),
    pytorch_lines(
        'aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassF.h',
        26, 30),
    pytorch_lines(MEMORY_EFFICIENT_ATTENTION, 1225, 1259),
    pytorch_lines(MEMORY_EFFICIENT_ATTENTION, 1285, 1293),
    pytorch_lines(MEMORY_EFFICIENT_ATTENTION, 421, 426),
    pytorch_lines(MEMORY_EFFICIENT_ATTENTION, 925, 926),
    pytorch_lines(MEMORY_EFFICIENT_ATTENTION, 1038, 1110))
'''The lines of PyTorch 2.10.0 deciding the kernel run by
`scaled_dot_product_attention` and its quantisations. The wheel for CUDA 12.8 installs
a cuDNN too old for PyTorch to put cuDNN attention first. PyTorch then tries the flash
kernel first and uses it only when no mask is passed, and tries the memory-efficient
kernel second. Of its BF16 kernels only the one reading keys block by block from global
memory accepts value heads of 256 channels. That kernel takes the exponential of each
FP32 score less the largest score among the keys read so far, against which it rescales
its earlier output, stores the exponential in the element type of the inputs for the
product with the values, and divides the FP32 result by the FP32 sum of the
exponentials before writing the output.'''

BF16_SUM_REFERENCE = pytorch_lines('aten/src/ATen/native/cuda/ReduceSumProdKernel.cu',
                                   212, 225)
'''The lines of PyTorch giving `torch.sum` over a BF16 tensor an FP32 accumulator.'''

POLICY_ROWS: tuple[tuple[Quantization.Quantified, str, str], ...] = (
    (FP8_BLOCK_WEIGHT,
     'every projection of the attention, the indexer queries and keys, the dense MLP, '
     'the shared expert and the routed experts, which the quantizer replaces with '
     '`FP8Linear` or `FP8Experts` because the configuration does not list them as '
     'unconverted. The expression splits `q_b_proj`, `kv_a_proj_with_mqa` and '
     '`kv_b_proj` into one weight per part. A block of 128 rows is a block of the '
     'joined weight of the reference, so a block of `q_b_proj` or `kv_b_proj` can hold '
     'rows of two parts',
     _lines(checkpoint_config_lines(224, 232), library_lines(FP8_LINEAR, 834, 916),
            library_lines(FP8_LINEAR, 285, 317), library_lines(FP8_LINEAR, 609, 693),
            library_lines('utils/quantization_config.py', 1710, 1731),
            checkpoint_file('model-00040-of-00141.safetensors'))),
    (FP8_ROUNDED_OPERAND,
     'the operand of an FP8 projection, rounded per token before a product in E4M3 '
     'with an FP32 accumulator. DeepGEMM rounds the operand of an `FP8Linear` on a '
     'Hopper GPU where the `kernels` package loads it and each process holds one GPU. '
     'DeepGEMM ships builds for PyTorch 2.10 and later alone, and the release pinned '
     'here is PyTorch 2.10.0. The Triton kernel rounds the operand otherwise, including '
     'a model spread over several GPUs by one process, and it rounds every operand of '
     'the routed experts, because `grouped_mm` calls no other kernel',
     _lines(library_lines(FP8_LINEAR, 208, 282),
            library_lines(FP8_LINEAR, 788, 815),
            library_lines(FP8_LINEAR, 562, 588),
            library_lines('integrations/deepgemm.py', 581, 622),
            library_lines('integrations/deepgemm.py', 419, 444),
            deep_gemm_lines('utils/math.py', 25, 37),
            finegrained_fp8_lines('matmul.py', 96, 165),
            finegrained_fp8_lines('utils.py', 433, 442))),
    (BF16,
     'every tensor passed from one module to the next, because the configuration '
     'declares `bfloat16` and the model is loaded at the dtype of its configuration: '
     'the embedding, the result of every projection and every RMS normalisation, the '
     'residual, the turned channels, the keys and values, the output of the attention '
     'and of every feed-forward map, and the logits',
     _lines(checkpoint_config_lines(7), library_lines('modeling_utils.py', 4096, 4097),
            library_lines('modeling_utils.py', 831, 835),
            library_lines(FP8_LINEAR, 194, 202),
            library_lines('integrations/deepgemm.py', 608),
            modeling_lines(57, 62), modeling_lines(621, 626), modeling_lines(800, 802))),
    (FP32,
     'the normalisation inside an RMS normalisation, which the module then rounds to '
     'BF16 and multiplies by its BF16 gain in BF16, the router from its linear map to '
     'the normalised gates, the indexer scores from the product of the queries and the '
     'keys to the top-2048, the attention scores and the exponentials of their softmax '
     'inside the attention kernel, and the accumulator of every FP8 product',
     _lines(modeling_lines(57, 62), modeling_lines(494, 519), modeling_lines(239, 244),
            library_lines('integrations/sdpa_attention.py', 158),
            pytorch_lines(MEMORY_EFFICIENT_ATTENTION, 1285, 1293),
            finegrained_fp8_lines('matmul.py', 140, 150))),
    (BF16,
     'the table of turns, computed in FP32 and cast to the BF16 of the hidden state '
     'handed to the rotary embedding, and the turn of every pair, which multiplies BF16 '
     'by BF16',
     _lines(modeling_lines(116, 122), modeling_lines(719, 720),
            modeling_lines(153, 160))),
    (BF16,
     'the SiLU and the product of the gate and up branches of the dense MLP, of the '
     'shared expert and of every routed expert, and the eight gates, which the experts '
     'cast to the BF16 of their outputs before the product with them, with the default '
     '`grouped_mm` implementation of the experts. The eight products are summed by '
     '`torch.sum` over a BF16 tensor, which accumulates in FP32 and returns BF16, so '
     'the sum is rounded once, as the eager `FP8Experts.forward` rounds its FP32 buffer '
     'once',
     _lines(modeling_lines(476, 477), library_lines(FP8_LINEAR, 699, 710),
            library_lines(FP8_LINEAR, 591), library_lines(FP8_LINEAR, 604, 606),
            BF16_SUM_REFERENCE, library_lines(FP8_LINEAR, 712, 752),
            library_lines('modeling_utils.py', 1879, 1880))),
    (BF16,
     'the queries, keys and values read by the attention core. SDPA is the default '
     'attention of the reference, and the reference always passes it the mask of the '
     'selection, so PyTorch 2.10.0 in its wheel for CUDA 12.8 runs the memory-efficient '
     'kernel. The kernel keeps the scores in FP32, rounds the exponential of each score '
     'less the largest score among the keys read so far to BF16 before the product with '
     'the values, and divides the FP32 result by the FP32 sum of the exponentials before '
     'writing BF16. The expression writes the softmax as one operation, so its cast to '
     'BF16 stands after the division',
     _lines(library_lines('modeling_utils.py', 1841, 1842), modeling_lines(430, 458),
            library_lines('integrations/sdpa_attention.py', 158, 170),
            *ATTENTION_KERNEL_REFERENCES)),
    (BF16,
     'the token embedding, the output head and the indexer head weights, which stay out '
     'of FP8. The configuration lists the first two as unconverted, and the model names '
     'the third in its `_keep_in_fp32_modules`, which upcasts only for an FP16 load',
     _lines(checkpoint_config_lines(331), checkpoint_config_lines(691),
            modeling_lines(649), library_lines('quantizers/quantizer_finegrained_fp8.py',
                                               126, 128),
            library_lines('modeling_utils.py', 3757, 3771),
            checkpoint_file('model-00001-of-00141.safetensors'))),
    (FP32,
     'the router weight, BF16 in the file and read through `.type(torch.float32)`, and '
     'the correction bias, an FP32 buffer held in FP32 in the file',
     _lines(modeling_lines(492), modeling_lines(496), modeling_lines(647),
            checkpoint_config_lines(283), checkpoint_config_lines(293),
            checkpoint_file('model-00040-of-00141.safetensors'))),
    (INT64,
     'the token identifiers, the positions picked by a top-k, and the positions read '
     'by the gather at the selected tokens. The reference reads the positions through '
     '`.long()` where it scatters them into the mask of the selection, which the gather '
     'stands for. The gather of a Full layer reads the INT64 positions of the top-2048, '
     'where the reference converts them to `int32` and back, which changes no value',
     _lines(modeling_lines(252, 253), modeling_lines(430, 441), modeling_lines(513))),
    (INT32,
     'the selection on the tape, converted by the indexer before it is returned and '
     'handed from a Full layer to the Shared layers after it',
     _lines(modeling_lines(253), modeling_lines(462), modeling_lines(627),
            modeling_lines(722, 733))),
)


def policy_table() -> str:
    '''The markdown table of the policy, one row per quantisation and site, for a
    notebook cell.'''
    rows = tuple(
        f'| {Quantization.describe_quantisation(quantisation)} | {applies_to} | {line} |'
        for quantisation, applies_to, line in POLICY_ROWS)
    return '\n'.join((*POLICY_TABLE_HEADER, *rows))


WEIGHT_TABLE_HEADER: tuple[str, str] = (
    '| weight | read quantisation | quantisation in the file |', '|---|---|---|')


def _file_quantisation(name: str) -> str:
    return ('the same' if name not in FILE_QUANTISATIONS
            else Quantization.describe_quantisation(FILE_QUANTISATIONS[name]))


def weight_table() -> str:
    '''The markdown table of the quantisation of every weight of the quantised model.
    A weight has no wire, so the figure cannot label it, and the table and the
    inspection box over the weight say the quantisation instead.'''
    rows = tuple(
        f'| ${name}$ | {Quantization.describe_quantisation(quantisation)} | '
        f'{_file_quantisation(name)} |'
        for name, quantisation in sorted(WEIGHT_QUANTISATIONS.items()))
    return '\n'.join((*WEIGHT_TABLE_HEADER, *rows))


def cast_counts() -> Mapping[tuple[str, str], int]:
    '''How many casts of the quantised model read each pair of formats, counted as
    written operations.'''
    return quantise_model.cast_counts(glm53_quantised)


CAST_TABLE_HEADER: tuple[str, str] = (
    '| read | written | casts written |', '|---|---|---|')


def cast_table() -> str:
    '''The markdown table of `cast_counts`, for a notebook cell.'''
    rows = tuple(f'| {read} | {written} | {count} |'
                 for (read, written), count in cast_counts().items())
    return '\n'.join((*CAST_TABLE_HEADER, *rows))


# ==========================================================================
# What an inspection box shows over a cast and over a weight.
# ==========================================================================
ROUNDING_KERNEL_FORMULA = (
    r'\begin{gathered}'
    r'a[i_{g}] = \max_{i_{y} \in y} \lvert x[128\, i_{g} + i_{y}] \rvert \\'
    r's[i_{g}] = \max(a[i_{g}],\, 10^{-4}) / 448 \\'
    r'\hat{x}[128\, i_{g} + i_{y}] = \mathrm{E4M3}\big(x[128\, i_{g} + i_{y}] '
    r'/ s[i_{g}]\big)'
    r'\end{gathered}')
'''The rounding in front of an FP8 projection, in index notation: `g` is the groups of
128 channels of a token, `y` the channels of one group, `a` the largest magnitude of a
group, `s` its FP32 scale and `\\hat{x}` the E4M3 element written beside `s`.'''

ROUNDING_KERNEL_REFERENCES: tuple[cat.CodeReference, ...] = (
    deep_gemm_lines('utils/math.py', 25, 37),
    library_lines('integrations/deepgemm.py', 581, 622),
    finegrained_fp8_lines('utils.py', 433, 442),
    finegrained_fp8_lines('matmul.py', 96, 165),
    library_lines(FP8_LINEAR, 208, 282),
    library_lines(FP8_LINEAR, 562, 588),
    library_lines(FP8_LINEAR, 788, 815))


def explain_rounding_cast(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row of an inspection box over a cast into `FP8_ROUNDED_OPERAND`, which is
    the rounding done by the FP8 projection, with its formula and its lines. `None`
    for any other operator.'''
    operator = target.operator
    if not isinstance(operator, Quantization.TypeConvert):
        return None
    if Quantization.quantisation_of(operator.target) != FP8_ROUNDED_OPERAND:
        return None
    return OperatorExplanation(
        title=r'\text{Cast to E4M3 with an FP32 Scale per 128 Channels}',
        formula=ROUNDING_KERNEL_FORMULA,
        description=text.ROUNDING_KERNEL_DESCRIPTION,
        references=ROUNDING_KERNEL_REFERENCES)


def explain_quantised_cast(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row of an inspection box over a cast: `explain_rounding_cast` for the
    rounding in front of an FP8 projection, and the row written by
    `cast_presentation.cast_explanation` for every other cast.'''
    return (explain_rounding_cast(target)
            or cast_presentation.cast_explanation(target))


QUANTISED_OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    **operator_explanations.OPERATOR_EXPLANATIONS,
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


EMBEDDING_TABLE_ROLES: dict[str, OperatorRole] = {
    operator_explanations.table_key('E'): declared_at(text.EMBEDDING_TABLE_ROLE, 669)}
'''The role of the embedding table, which `operator_explanations` explains through the
box over the lookup and gives no role, so that the quantisation of the table has a
sentence to be written into.'''


def quantised_operator_roles() -> dict[str, OperatorRole]:
    '''The roles of `operator_explanations` and of the embedding table, with the
    quantisation of each weight written into the sentence shown by the inspection box
    over that weight.'''
    roles = {**operator_explanations.OPERATOR_ROLES, **EMBEDDING_TABLE_ROLES}
    return {
        name: (_role_with_its_quantisation(name, role, WEIGHT_QUANTISATIONS[name])
               if name in WEIGHT_QUANTISATIONS else role)
        for name, role in roles.items()}


def with_quantised_explanation_tables(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` with the tables of `operator_explanations`, the row of a cast, and
    the roles carrying the quantisation of each weight.'''
    return dataclasses.replace(
        operator_explanations.with_explanation_tables(settings),
        operator_explanations=QUANTISED_OPERATOR_EXPLANATIONS,
        operator_roles=quantised_operator_roles())


# ==========================================================================
# What the model holds once it is quantised.
# ==========================================================================
def unquantised_weaves(model: cat.Morphism = glm53_quantised) -> fd.Prod[cat.Weave]:
    '''Every wire of `model` holding a number and carrying no quantisation, which is
    none.'''
    return quantise_model.unquantised_weaves(model)


def operators_without_a_rule() -> fd.Prod[type[cat.Operator]]:
    '''Every operator class of the model given no rule by the registry, which is
    none.'''
    return quantise_model.leaves_without_a_rule(whole_model.glm53)
