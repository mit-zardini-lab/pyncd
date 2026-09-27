# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The released DeepSeek-V3 with a quantisation on every wire and a cast wherever a
value changes quantisation, as its inference code runs the FP8 checkpoint.

A quantisation is the number format a value is held in, together with the size of that
format in bits. `released_deepseek_v3.py` writes the model in the reals.
`RELEASED_POLICY` states the quantisations of the released inference code, read from
`inference/model.py`, `inference/kernel.py`, `inference/generate.py`, the configuration
`config_671B.json`, the headers of the checkpoint's shards and PyTorch 2.4.1, all pinned
in `reference_links.py` and read on 2026-09-27. `quantization.processing.quantise_model`
writes them onto every wire of the expression and puts a `TypeConvert` named `cast`
wherever an operation requires a quantisation the value does not carry.

The configuration sets `dtype` to `fp8`, so every `Linear` of the model holds its weight
in E4M3 with one FP32 scale per block of 128 rows by 128 channels, `FP8_BLOCK_WEIGHT`,
which is the format GLM-5.3 declares under the same name. The embedding, the router
weight and the head are created in the default dtype and hold BF16, and the correction
bias is created in FP32. `generate.py` sets the default dtype to BF16 and never assigns
`gemm_impl`, whose default is `bf16`, so every FP8 projection dequantises its weight to
BF16 with `weight_dequant` and multiplies with `F.linear` in BF16. The activation is
never rounded to FP8, which `RELEASED_POLICY` states by naming BF16 as the quantisation
of the operand of a projection whose weight has fewer bits than an activation.

Every tensor passed between modules is BF16. The attention, the MLP, the mixture and the
gate are eager code over BF16 tensors: the score scale, the sigmoid, the SiLU, the
products, the normalisation of the gates and the route scale return BF16, and the
softmax computes in FP32 and returns BF16 through `.type_as(x)`. `BOX_POLICIES` names
those four boxes as computing at the carried quantisation. The rotary embedding reads
its pairs into FP32, multiplies them by the complex64 table and writes BF16 back, which
is the default of a box. The RMSNorm of PyTorch 2.4.1 is a chain of BF16 tensor
operations. The correction bias promotes the copy of the scores it is added to, so the
selection of the groups and the experts runs in FP32, and the gates are read from the
unbiased BF16 scores. The token identifiers and the positions a top-k picks are INT64.

The caches of the pass of `cached_deepseek_v3.py` are buffers created in the default
dtype, so the pass holds its caches in BF16, which `caching.data_structure.Caching`
carries as the quantisation of the values written to it.

`with_quantised_explanation_tables` writes the quantisation of each weight into the
sentence an inspection box shows over it, because a weight has no wire to label.
`notebooks/website/classic/DeepSeekV3.ipynb` draws both passes, and
`notebooks/website/classic/validate_deepseek_v3.py` checks the claims the
notebook makes about them.
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

import notebooks.classic.cached_deepseek_v3 as cached_deepseek_v3
import notebooks.classic.deepseek_v3 as deepseek_v3
import notebooks.classic.released_deepseek_v3 as released_deepseek_v3
import notebooks.display.cast_presentation as cast_presentation
import notebooks.display.notebook_diagrams as notebook_diagrams
from notebooks.classic.reference_links import (
    deepseek_checkpoint_file, deepseek_inference_lines, deepseek_lines,
    deepseek_repository_lines, pytorch_lines, released_config_lines)
from notebooks.classic.released_deepseek_v3_wording import TEXT as text
from notebooks.display.explain_operators import ExplanationOfOperator

BF16 = Quantization.BF16
FP32 = Quantization.FP32
E4M3 = Quantization.E4M3
INT64 = Quantization.INT64

WEIGHT_BLOCK = 128
FP8_BLOCK_WEIGHT = Quantization.block_scaled(
    E4M3, FP32, channels=WEIGHT_BLOCK, rows=WEIGHT_BLOCK)
'''A projection of the FP8 checkpoint: E4M3 elements with one FP32 scale per block of
128 rows by 128 channels. `Linear` allocates the weight in `Linear.dtype`, which is
`torch.float8_e4m3fn` under `dtype = "fp8"`, and the scale in FP32 at `model.py` lines
183 to 187 and 760. `README_WEIGHTS.md` lines 60 to 92 state the format, and every
shard holds each projection as `F8_E4M3` beside an `F32` `weight_scale_inv`. A block is
cut short where the rows are not a multiple of 128, as the 576 rows of
`kv_a_proj_with_mqa` are. The blocks of `kv_b_proj` fall on the boundary between the
key rows and the value rows of every head, and the blocks of `q_b_proj` can hold rows of
`W^{UQ}` and `W^{QR}` together.'''

UNROUNDED_OPERAND = BF16
'''The operand of an FP8 projection. The default `gemm_impl = "bf16"` dequantises the
weight to BF16 and calls `F.linear`, so the operand stays BF16, where the other path
would round it to E4M3 with one FP32 scale per 128 channels through `act_quant`.'''

RELEASED_WEIGHTS: dict[str, Quantization.Quantified] = {
    'E': BF16,
    'W^{g}': BF16,
    deepseek_v3.HEAD_NAME: BF16,
    released_deepseek_v3.ROUTER_BIAS_NAME: FP32,
}
'''The quantisation of every weight not held at `FP8_BLOCK_WEIGHT`. The embedding and
the router weight are parameters created in the default dtype, the head is created
with the default dtype passed explicitly, and the correction bias is created in
`torch.float32`.'''

BOX_POLICIES: dict[str, quantise_model.BoxPolicy] = {
    name: quantise_model.BoxPolicy(
        arithmetic=quantise_model.ArithmeticQuantisation.CARRIED)
    for name in (deepseek_v3.ATTENTION_BOX, deepseek_v3.MLP_BOX, deepseek_v3.MOE_BOX,
                 released_deepseek_v3.GATE_BOX)}
'''The four boxes whose elementwise arithmetic returns the quantisation of its operand,
because the released modules compute on BF16 tensors without upcasting them.'''

RELEASED_POLICY = quantise_model.QuantizationPolicy(
    inputs=BF16,
    activations=BF16,
    scalars=FP32,
    rounded_operands=UNROUNDED_OPERAND,
    results=BF16,
    weights=RELEASED_WEIGHTS,
    weights_by_default=FP8_BLOCK_WEIGHT,
    boxes=BOX_POLICIES,
    integers=INT64)

QUANTISED_DECODE = quantise_model.quantise_model(
    released_deepseek_v3.decode_form(), RELEASED_POLICY)
QUANTISED_CACHED = quantise_model.quantise_model(
    cached_deepseek_v3.CACHED_PASS.expression, RELEASED_POLICY)

decode_quantised = QUANTISED_DECODE.morphism
cached_quantised = QUANTISED_CACHED.morphism

WEIGHT_QUANTISATIONS: Mapping[str, Quantization.Quantified] = {
    **QUANTISED_DECODE.weight_quantisations, **QUANTISED_CACHED.weight_quantisations}

decode_without_quantisations = strip_quantisations.strip_quantisations(decode_quantised)
cached_without_quantisations = strip_quantisations.strip_quantisations(cached_quantised)
'''The two quantised passes with every quantisation taken back off them, which a page
derives in the browser with the dequantisation functor.'''


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


FIRST_SHARD = 'model-00001-of-000163.safetensors'
LAST_SHARD = 'model-00160-of-000163.safetensors'
RMS_NORM_OF_PYTORCH = pytorch_lines('aten/src/ATen/native/layer_norm.cpp', 266, 309)

POLICY_ROWS: tuple[tuple[Quantization.Quantified, str, str], ...] = (
    (FP8_BLOCK_WEIGHT,
     'every projection of the attention, the dense MLP, the routed experts and the '
     'shared expert. `config_671B.json` sets `dtype` to `fp8`, so `Linear.dtype` is '
     '`torch.float8_e4m3fn`, and every `Linear` allocates an FP32 scale per block of '
     '128 by 128. The checkpoint holds each of them as `F8_E4M3` beside an `F32` '
     '`weight_scale_inv`',
     _lines(released_config_lines(21), deepseek_lines(760), deepseek_lines(183, 187),
            deepseek_repository_lines('README_WEIGHTS.md', 60, 92),
            deepseek_checkpoint_file('config.json', 37, 45),
            deepseek_checkpoint_file(FIRST_SHARD))),
    (BF16,
     'the operand of every FP8 projection, which is not rounded. The default '
     '`gemm_impl` is `bf16`, `generate.py` never assigns it, and under it `linear` '
     'dequantises the weight to the default dtype with `weight_dequant` and calls '
     '`F.linear` in BF16. The other path, `act_quant` and `fp8_gemm`, is not run',
     _lines(deepseek_lines(16), deepseek_lines(153, 163),
            deepseek_inference_lines('kernel.py', 89, 110),
            deepseek_inference_lines('generate.py', 100, 119))),
    (BF16,
     'every tensor passed from one module to the next, because `generate.py` sets the '
     'default dtype to BF16 before it builds the model: the embedding, the result of '
     'every projection, every RMSNorm, the residual, the turned channels, the output '
     'of the attention, of the MLP and of the mixture, the gates and the logits',
     _lines(deepseek_inference_lines('generate.py', 109, 116), deepseek_lines(124),
            deepseek_lines(393), deepseek_lines(490), deepseek_lines(598),
            deepseek_lines(682, 693), deepseek_lines(733, 734), deepseek_lines(769))),
    (BF16,
     'the RMSNorm, which PyTorch 2.4.1 writes as a chain of tensor operations on the '
     'BF16 input and the BF16 weight, each rounding to BF16',
     _lines(deepseek_lines(294), RMS_NORM_OF_PYTORCH,
            deepseek_repository_lines('inference/requirements.txt', 1))),
    (BF16,
     'the arithmetic of the attention, the MLP, the experts and the gate: the score '
     'scale, the SiLU and the product of the two branches, the sigmoid of the router, '
     'the division of the gates by their sum and the route scale. The softmax computes '
     'in FP32 and returns BF16 through `.type_as(x)`',
     _lines(deepseek_lines(479), deepseek_lines(486, 490), deepseek_lines(532),
            deepseek_lines(580), deepseek_lines(594, 598), deepseek_lines(633))),
    (FP32,
     'the rotary embedding, which reads the pairs into FP32 and multiplies them by the '
     'complex64 table before writing BF16 back',
     _lines(deepseek_lines(366, 374), deepseek_lines(389, 393))),
    (FP32,
     'the correction bias, created in FP32 and held as `F32` in the checkpoint, and '
     'the copy of the scores it is added to, so the choice of the groups and the experts '
     'runs in FP32',
     _lines(deepseek_lines(564), deepseek_lines(582, 593),
            deepseek_checkpoint_file(FIRST_SHARD))),
    (BF16,
     'the token embedding, the router weight and the head, created in the default '
     'dtype and held as BF16 in the checkpoint',
     _lines(deepseek_lines(105), deepseek_lines(563), deepseek_lines(769),
            deepseek_checkpoint_file(FIRST_SHARD),
            deepseek_checkpoint_file(LAST_SHARD))),
    (INT64,
     'the token identifiers, held in `torch.long`, and the positions picked by each '
     'top-k',
     _lines(deepseek_inference_lines('generate.py', 54, 56), deepseek_lines(590),
            deepseek_lines(593))),
    (BF16,
     'the two caches of the pass, the normalised latent and the turned key, which are '
     'buffers created in the default dtype',
     _lines(deepseek_lines(443, 444), deepseek_lines(484, 485))),
)


def policy_table() -> str:
    '''The markdown table of the policy, one row per quantisation and site.'''
    rows = tuple(
        f'| {Quantization.describe_quantisation(quantisation)} | {applies_to} | {line} |'
        for quantisation, applies_to, line in POLICY_ROWS)
    return '\n'.join((*POLICY_TABLE_HEADER, *rows))


WEIGHT_TABLE_HEADER: tuple[str, str] = (
    '| weight | held in memory and in the checkpoint | read by the product as |',
    '|---|---|---|')


def read_by_the_product(quantisation: Quantization.Quantified) -> str:
    '''The quantisation a product reads a weight held at `quantisation` in, which is
    BF16 for an FP8 weight, dequantised before every product.'''
    if quantisation == FP8_BLOCK_WEIGHT:
        return Quantization.describe_quantisation(UNROUNDED_OPERAND)
    return 'the same'


def weight_table() -> str:
    '''The markdown table of the quantisation of every weight of the two passes. A
    weight has no wire, so the figure cannot label it.'''
    rows = tuple(
        f'| ${name}$ | {Quantization.describe_quantisation(quantisation)} | '
        f'{read_by_the_product(quantisation)} |'
        for name, quantisation in sorted(WEIGHT_QUANTISATIONS.items()))
    return '\n'.join((*WEIGHT_TABLE_HEADER, *rows))


def cast_counts(morphism: cat.Morphism) -> Mapping[tuple[str, str], int]:
    '''How many casts of `morphism` read each pair of formats, counted as written
    operations.'''
    return quantise_model.cast_counts(morphism)


CAST_TABLE_HEADER: tuple[str, str] = (
    '| read | written | casts written in the decode pass | casts written in the cached '
    'pass |', '|---|---|---|---|')


def cast_table() -> str:
    '''The markdown table of the casts of both passes.'''
    decode = cast_counts(decode_quantised)
    cached = cast_counts(cached_quantised)
    rows = tuple(f'| {read} | {written} | {decode.get((read, written), 0)} | '
                 f'{cached.get((read, written), 0)} |'
                 for read, written in sorted({*decode, *cached}))
    return '\n'.join((*CAST_TABLE_HEADER, *rows))


# ==========================================================================
# What an inspection box shows over a weight.
# ==========================================================================
QUANTISED_OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    **released_deepseek_v3.OPERATOR_EXPLANATIONS,
    Quantization.TypeConvert: cast_presentation.cast_explanation,
}


def role_with_its_quantisation(
    role: OperatorRole, quantisation: Quantization.Quantified,
) -> OperatorRole:
    held = text.WEIGHT_QUANTISATION_SENTENCE.format(
        quantisation=Quantization.describe_quantisation(quantisation))
    read = ('' if quantisation != FP8_BLOCK_WEIGHT
            else ' ' + text.DEQUANTISED_WEIGHT_SENTENCE)
    return dataclasses.replace(role, role=f'{role.role} {held}{read}')


def quantised_operator_roles() -> dict[str, OperatorRole]:
    '''The roles of the released model, with the quantisation of each weight written
    into the sentence an inspection box shows over it.'''
    return {
        name: (role_with_its_quantisation(role, WEIGHT_QUANTISATIONS[name])
               if name in WEIGHT_QUANTISATIONS else role)
        for name, role in released_deepseek_v3.OPERATOR_ROLES.items()}


def with_quantised_explanation_tables(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` with the tables of the released model, the row of a cast, and the
    roles carrying the quantisation of each weight.'''
    return dataclasses.replace(
        released_deepseek_v3.with_explanation_tables(settings),
        operator_explanations=QUANTISED_OPERATOR_EXPLANATIONS,
        operator_roles=quantised_operator_roles())


# ==========================================================================
# What the passes hold once they are quantised.
# ==========================================================================
def unquantised_weaves(morphism: cat.Morphism) -> fd.Prod[cat.Weave]:
    '''Every wire of `morphism` holding a number and carrying no quantisation.'''
    return quantise_model.unquantised_weaves(morphism)


def operators_without_a_rule(morphism: cat.Morphism) -> fd.Prod[type[cat.Operator]]:
    '''Every operator class of `morphism` given no rule by the registry.'''
    return quantise_model.leaves_without_a_rule(morphism)
