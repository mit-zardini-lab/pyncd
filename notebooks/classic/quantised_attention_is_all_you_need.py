# Claude Opus 5.5 (1M context), effort 40.
'''The transformer of *Attention Is All You Need* with a quantisation on every wire, as
tensor2tensor runs it.

A quantisation is the number format a value is held in, together with the size of that
format in bits. tensor2tensor, the code linked by the paper, holds every activation
and every weight of the base model in FP32. `basic_params1` sets `activation_dtype` and
`weight_dtype` to `float32`, `transformer_base_v1` starts from those parameters and
changes neither, and the model converts a value to another format only where
`activation_dtype` names a 16-bit format or `weight_dtype` names `bfloat16`. The input
pipeline converts the token identifiers from `int64` to `int32` before the model reads
them, and the beam search used by the paper to decode hands the decoder the `int32`
identifiers returned by `tf.nn.top_k`.

`RELEASED_POLICY` states those quantisations: FP32 on every real value and every weight,
and INT32 on every index. `quantise_as_released` writes them onto every wire of a model
through `quantization.processing.quantise_model`. No operation of the model requires a
value in another quantisation, so the pass writes no cast. A cache of the cached pass
holds the FP32 keys and values appended to it, which is the format of the tensors
concatenated onto its cache by tensor2tensor.

`notebooks/website/classic/AttentionIsAllYouNeed.ipynb` cites the lines of
tensor2tensor, and `obsidian/04-quantization/Quantization.md` states the pass.
'''
from __future__ import annotations

import data_structure.Category as cat
import quantization.data_structure.Quantization as Quantization
import quantization.processing.quantise_model as quantise_model

FP32 = Quantization.FP32
INT32 = Quantization.INT32

RELEASED_POLICY = quantise_model.QuantizationPolicy(
    inputs=FP32,
    activations=FP32,
    scalars=FP32,
    rounded_operands=FP32,
    results=FP32,
    weights_by_default=FP32,
    integers=INT32)
'''Every real value and every weight in FP32, and every token identifier in INT32. No
weight has fewer bits than an activation, so no operand is rounded in front of a
projection and `rounded_operands` names the activation quantisation.'''


def quantise_as_released(model: cat.Morphism) -> quantise_model.QuantisedModel:
    return quantise_model.quantise_model(model, RELEASED_POLICY)
