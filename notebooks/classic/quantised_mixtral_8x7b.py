# Claude Opus 5.5 (1M context), effort 40.
'''Mixtral-8x7B with a quantisation on every wire and a cast wherever a value changes
quantisation, as `mistral-inference` runs the released weights.

`mixtral_8x7b.mixtral` is written in the reals. A quantisation is the number format a
value is held in, together with the size of that format in bits. The released
checkpoint holds every weight in BF16, and the reference implementation loads it
without converting it, so the model runs in BF16. `RELEASED_POLICY` states the
quantisations of the reference, read from `mistral-inference` at the commit pinned in
`reference_links`, from the headers of the checkpoint's shards, and from the attention
kernel that `xformers` calls, all pinned below and read on 2026-09-27.
`quantization.processing.quantise_model` writes them onto every wire of the expression,
putting a `TypeConvert` named `cast` wherever an operation requires a quantisation not
carried by the value.

Four places compute in another quantisation than BF16. The RMSNorm divides by the
root mean square in FP32 and rounds back to BF16 before the gain, and the rule of a
normalisation states the module as one operation reading and returning BF16. The
rotary embedding reads the queries and the keys into FP32, multiplies them by a table
of FP32 complex numbers and rounds the result to BF16. The attention kernel is
FlashAttention-2, which
keeps the scores, their softmax and its accumulator in FP32 and rounds the
probabilities to BF16 before the product with the values, and `ATTENTION_CORE_POLICY`
states it for the block of the core. The router takes the softmax of the two kept
scores in FP32 and rounds the weights to BF16, which `BLOCK_RESULTS` states. The logits
are returned in FP32, and the sampler takes their softmax in FP32. Everything else is
eager BF16: the SiLU and the product of the two branches of an expert return BF16, which
`EXPERTS_POLICY` states, and the weighted sum of the two experts is accumulated into a
BF16 tensor.

The expression reads the pairs of channels of the rotary embedding as complex numbers
and then casts them to FP32, where the reference casts the channels and then pairs
them. Pairing moves no value, so the two orders hold the same numbers.

The quantised model is written on the CausalSlide of the model, which is the form a
figure draws, and `mixtral_without_quantisations` is the quantised model with every
quantisation taken back off it. `notebooks/website/classic/Mixtral8x7B.ipynb`
draws both and `validate_mixtral_8x7b.py` beside it checks every claim made
about them.
'''
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards
import data_structure.Category as cat
import data_structure.Term as fd
import quantization.algebra.strip_quantisations as strip_quantisations
import quantization.data_structure.Quantization as Quantization
import quantization.processing.quantise_model as quantise_model
from term_utilities.code_references import pinned_link
from websocket_transfer.auxiliary_information import OperatorRole

import notebooks.classic.mixtral_8x7b as mixtral_8x7b
from notebooks.classic.mixtral_8x7b_wording import TEXT as text
from notebooks.classic.reference_links import (
    MISTRAL_INFERENCE_BASE, MIXTRAL_WEIGHTS_BASE, mistral_lines, mixtral_config_lines)

BF16 = Quantization.BF16
FP32 = Quantization.FP32
INT64 = Quantization.INT64

QUANTISATION_READ_ON = '2026-09-27'

XFORMERS_COMMIT = '03b91d7d9ff295ae68a320e2e733dd6c2ef8f342'
XFORMERS_URL = (
    f'https://github.com/facebookresearch/xformers/blob/{XFORMERS_COMMIT}/')
'''`xformers` at the commit of the tag `v0.0.35`, the latest release on 2026-09-27.
`mistral-inference` requires `xformers>=0.0.24`, so an install made that day takes
this release.'''

PYTORCH_COMMIT = 'ba56102387ef21a3b04b357e5b183d48f0afefc7'
PYTORCH_URL = f'https://github.com/pytorch/pytorch/blob/{PYTORCH_COMMIT}/'
'''PyTorch at the commit of the tag `v2.8.0`. PyTorch builds its FlashAttention-2
kernel from the `third_party/flash-attention` submodule, pinned below.'''

FLASH_ATTENTION_COMMIT = '979702c87a8713a8e0a5e9fee122b90d2ef13be5'
FLASH_ATTENTION_URL = (
    f'https://github.com/Dao-AILab/flash-attention/blob/{FLASH_ATTENTION_COMMIT}/')
'''`Dao-AILab/flash-attention` at the commit PyTorch `v2.8.0` holds as its
`third_party/flash-attention` submodule.'''


def xformers_lines(path: str, line: int,
                   end_line: int | None = None) -> cat.CodeReference:
    '''A file under `xformers/ops/fmha/` at the pinned `xformers` commit.'''
    return pinned_link(XFORMERS_URL, f'xformers/ops/fmha/{path}', line, end_line,
                       label=line_label(f'xformers {path}', line, end_line))


def pytorch_lines(path: str, line: int,
                  end_line: int | None = None) -> cat.CodeReference:
    '''A file of PyTorch at the pinned commit.'''
    return pinned_link(PYTORCH_URL, path, line, end_line,
                       label=line_label(f'pytorch {path.rsplit("/", 1)[-1]}', line,
                                        end_line))


def flash_attention_lines(path: str, line: int,
                          end_line: int | None = None) -> cat.CodeReference:
    '''A file under `csrc/flash_attn/src/` of FlashAttention at the pinned commit.'''
    return pinned_link(FLASH_ATTENTION_URL, f'csrc/flash_attn/src/{path}', line,
                       end_line, label=line_label(f'flash-attention {path}', line,
                                                  end_line))


def checkpoint_file(path: str) -> cat.CodeReference:
    '''A file of the released checkpoint at its pinned commit, such as a shard whose
    header gives the datatype of each tensor held in the shard.'''
    return pinned_link(MIXTRAL_WEIGHTS_BASE, path)


def line_label(file: str, line: int, end_line: int | None) -> str:
    lines = (f'L{line}' if end_line is None or end_line == line
             else f'L{line}-L{end_line}')
    return f'{file} {lines}'


FIRST_SHARD = 'model-00001-of-00019.safetensors'
LAST_SHARD = 'model-00019-of-00019.safetensors'

# ==========================================================================
# The policy.
# ==========================================================================
ATTENTION_CORE_POLICY = quantise_model.BoxPolicy(
    contractions=quantise_model.ContractionQuantisation.ACCUMULATED)
'''The block of the attention core is one call of FlashAttention-2. Its products read
BF16 and accumulate in FP32, and its softmax is FP32 arithmetic, so the probabilities
are rounded to BF16 before the product with the values.'''

EXPERTS_POLICY = quantise_model.BoxPolicy(
    arithmetic=quantise_model.ArithmeticQuantisation.CARRIED)
'''The block of the experts is eager code that never upcasts, so the SiLU and the
product of the two branches return BF16.'''

TITLED_BLOCK_POLICIES: dict[str, quantise_model.BoxPolicy] = {
    text.CORE_TITLE: ATTENTION_CORE_POLICY,
    text.EXPERTS_TITLE: EXPERTS_POLICY,
}

BLOCK_RESULTS: dict[str, fd.Prod[Quantization.Quantified | None]] = {
    text.CORE_TITLE: (BF16,),
    text.ROUTER_TITLE: (BF16,),
}
'''The attention kernel writes its output in BF16, and the router rounds the FP32
softmax of the two kept scores to BF16 with `.to(inputs.dtype)`.'''

RELEASED_POLICY = quantise_model.QuantizationPolicy(
    inputs=BF16,
    activations=BF16,
    scalars=FP32,
    results=FP32,
    weights_by_default=BF16,
    boxes=TITLED_BLOCK_POLICIES,
    block_results=BLOCK_RESULTS,
    integers=INT64)

MIXTRAL = mixtral_8x7b.mixtral()

MIXTRAL_IN_THE_CAUSAL_SLIDE = slide_causal_reads_backwards.slide_causal_reads_backwards(
    MIXTRAL)
'''The model with its two causal reads moved back to the copy that feeds the queries,
which is the form drawn by a figure of a model.'''

QUANTISED = quantise_model.quantise_model(MIXTRAL_IN_THE_CAUSAL_SLIDE, RELEASED_POLICY)

mixtral_quantised = QUANTISED.morphism

WEIGHT_QUANTISATIONS: Mapping[str, Quantization.Quantified] = (
    QUANTISED.weight_quantisations)

mixtral_without_quantisations = strip_quantisations.strip_quantisations(
    mixtral_quantised)
'''The quantised model with every quantisation taken back off it, which is the
arithmetic of the model with no format on any wire and no conversion anywhere.'''


# ==========================================================================
# The policy as a table.
# ==========================================================================
POLICY_TABLE_HEADER: tuple[str, str] = (
    '| quantisation | what it applies to | lines |', '|---|---|---|')

FLASH_KERNEL = 'flash_fwd_kernel.h'

ATTENTION_KERNEL_REFERENCES: tuple[cat.CodeReference, ...] = (
    mistral_lines('transformer_layers.py', 88),
    xformers_lines('dispatch.py', 86, 130),
    xformers_lines('flash3.py', 116, 124),
    pinned_link(MISTRAL_INFERENCE_BASE, 'pyproject.toml', 25, 32),
    xformers_lines('flash.py', 50, 82),
    xformers_lines('flash.py', 558, 590),
    pytorch_lines('aten/src/ATen/CMakeLists.txt', 172, 173),
    flash_attention_lines('kernel_traits.h', 26),
    flash_attention_lines(FLASH_KERNEL, 303),
    flash_attention_lines(FLASH_KERNEL, 343, 347),
    flash_attention_lines(FLASH_KERNEL, 367),
    flash_attention_lines(FLASH_KERNEL, 433, 436))
'''The lines deciding the attention kernel and its quantisations. The reference calls
`memory_efficient_attention`, whose dispatch tries FlashAttention-3, then
FlashAttention-2, then its CUTLASS kernel. FlashAttention-3 needs the `flash_attn_3`
package and the `flash_attn` package is optional, and neither is a dependency of
`mistral-inference`, so `xformers` calls the FlashAttention-2 kernel PyTorch is built
with. The kernel keeps the scores in an FP32 accumulator, takes the softmax in FP32,
converts the probabilities to the element type of the inputs for the product with the
values, accumulates that product in FP32, divides by the FP32 sum of the exponentials,
and converts the result to the element type.'''

POLICY_ROWS: tuple[tuple[Quantization.Quantified, str, tuple[cat.CodeReference, ...]],
                   ...] = (
    (BF16,
     'every weight: the embedding, the four projections of the attention, the router, '
     'the three projections of every expert, the output projection and the gain of '
     'every RMSNorm. The checkpoint declares `bfloat16`, every tensor of its shards is '
     '`BF16`, and the reference loads the file with `dtype=None`, which keeps the '
     'datatype of the file',
     (mixtral_config_lines(25), checkpoint_file(FIRST_SHARD),
      checkpoint_file(LAST_SHARD),
      mistral_lines('transformer.py', 298, 338), mistral_lines('main.py', 124))),
    (BF16,
     'every tensor passed from one module to the next: the embedding, the result of '
     'every projection and every RMSNorm, the residual stream, the turned queries and '
     'keys, the values, the output of the attention, of every expert and of the '
     'mixture, and the logits before they are returned',
     (mistral_lines('transformer.py', 193),
      mistral_lines('transformer_layers.py', 66, 70),
      mistral_lines('transformer_layers.py', 165, 168),
      mistral_lines('transformer.py', 235))),
    (FP32,
     'the normalisation inside every RMSNorm, rounded to BF16 by the module and '
     'multiplied by its BF16 gain in BF16. The expression writes the RMSNorm as one '
     'operation reading and returning BF16',
     (mistral_lines('transformer_layers.py', 115, 120),)),
    (FP32,
     'the table of the rotary embedding, held as `complex64`, and the product of each '
     'pair of channels with it. The reference reads the queries and the keys into FP32 '
     'before pairing their channels and rounds the turned channels to BF16',
     (mistral_lines('rope.py', 6, 10), mistral_lines('rope.py', 18, 23),
      mistral_lines('transformer.py', 108, 120))),
    (FP32,
     'the scores of the attention, their softmax and the accumulator of the product '
     'with the values inside FlashAttention-2, which reads BF16 queries, keys and '
     'values and writes BF16. The probabilities are rounded to BF16 before the product '
     'with the '
     'values. The expression writes the softmax as one operation, so the kernel\'s '
     'division by the sum of the exponentials, which follows that product, stands '
     'before the rounding',
     ATTENTION_KERNEL_REFERENCES),
    (FP32,
     'the softmax of the two scores the router keeps, taken with `dtype=torch.float` '
     'and rounded to BF16 with `.to(inputs.dtype)`. The top-2 selection reads the BF16 '
     'scores of the gate projection',
     (mistral_lines('moe.py', 25, 27),)),
    (BF16,
     'the SiLU and the product of the two branches of every expert, returned in BF16 '
     'by eager PyTorch, and the weighted sum of the two chosen experts, accumulated '
     'into a '
     'BF16 tensor by the reference',
     (mistral_lines('transformer_layers.py', 105, 106),
      mistral_lines('moe.py', 28, 31))),
    (FP32,
     'the logits the forward pass returns, converted with `.float()` because '
     '`softmax_fp32` is set by default, and the softmax the sampler takes of them',
     (mistral_lines('transformer.py', 39), mistral_lines('transformer.py', 239, 240),
      mistral_lines('generate.py', 151, 153))),
    (INT64,
     'the token identifiers, handed to the model as `torch.long` by the reference',
     (mistral_lines('generate.py', 95, 96), mistral_lines('generate.py', 139))),
)


def _link(reference: cat.CodeReference) -> str:
    return (reference.label if reference.url is None
            else f'[{reference.label}]({reference.url})')


def policy_table() -> str:
    '''The markdown table of the policy, one row per quantisation and site.'''
    rows = tuple(
        f'| {Quantization.format_name(quantisation)} | {applies_to} | '
        f'{", ".join(map(_link, references))} |'
        for quantisation, applies_to, references in POLICY_ROWS)
    return '\n'.join((*POLICY_TABLE_HEADER, *rows))


def cast_counts(model: cat.Morphism = mixtral_quantised
                ) -> Mapping[tuple[str, str], int]:
    '''How many casts of `model` read each pair of formats, counted as written
    operations.'''
    return quantise_model.cast_counts(model)


CAST_TABLE_HEADER: tuple[str, str] = (
    '| read | written | casts written |', '|---|---|---|')


def cast_table(model: cat.Morphism = mixtral_quantised) -> str:
    rows = tuple(f'| {read} | {written} | {count} |'
                 for (read, written), count in cast_counts(model).items())
    return '\n'.join((*CAST_TABLE_HEADER, *rows))


# ==========================================================================
# What an inspection box shows over a weight.
# ==========================================================================
EMBEDDING_TABLE_ROLES: dict[str, OperatorRole] = {
    mixtral_8x7b.table_key('E'): OperatorRole(
        role=text.EMBEDDING_TABLE_ROLE,
        references=(mistral_lines('transformer.py', 57),))}
'''The role of the embedding table, so that the quantisation of the table has a
sentence to be written into.'''


def _role_with_its_quantisation(role: OperatorRole,
                                quantisation: Quantization.Quantified) -> OperatorRole:
    sentence = text.WEIGHT_QUANTISATION_SENTENCE.format(
        quantisation=Quantization.format_name(quantisation))
    return dataclasses.replace(role, role=f'{role.role} {sentence}')


def unquantised_operator_roles() -> dict[str, OperatorRole]:
    '''The roles of the model and of the embedding table, stating no quantisation, for
    a figure of the model in the real numbers.'''
    return {**mixtral_8x7b.OPERATOR_ROLES, **EMBEDDING_TABLE_ROLES}


def quantised_operator_roles() -> dict[str, OperatorRole]:
    '''The roles of the model and of the embedding table, with the quantisation of
    each weight written into the sentence shown by the inspection box over that
    weight.'''
    return {
        name: (_role_with_its_quantisation(role, WEIGHT_QUANTISATIONS[name])
               if name in WEIGHT_QUANTISATIONS else role)
        for name, role in unquantised_operator_roles().items()}


# ==========================================================================
# What the model holds once it is quantised.
# ==========================================================================
def unquantised_weaves(model: cat.Morphism = mixtral_quantised) -> fd.Prod[cat.Weave]:
    '''Every wire of `model` holding a number and carrying no quantisation, which is
    none.'''
    return quantise_model.unquantised_weaves(model)


def operators_without_a_rule() -> fd.Prod[type[cat.Operator]]:
    '''Every operator class of the model given no rule by the registry, which is
    none.'''
    return quantise_model.leaves_without_a_rule(MIXTRAL)
