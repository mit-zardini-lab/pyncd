# Claude Opus 5 (1M context), effort high.
'''Engram as the released DeepSeek-V4.1-Flash computes it, one boxed module for each of
the two layers that hold one.

Engram adds a looked-up value to the four residual streams before layer 1 and before
layer 14. The run of the last two, three or four token identifiers that ends at a token
is hashed into rows of a learned table, the rows are read and projected into one key for
each stream and one shared value, and each stream's gate decides how much of the value
the stream receives. `engram_of_layer` returns the module of one layer as a box from the
residual to the residual, which reads the token identifiers and the modality from the
tape slots that the input of the model writes.

`notebooks.sota.DeepSeekV41Flash.omitted_mechanisms` states the token map, the written-out
hash, the table, the two projections and the gate, and this module imports every one of
them. It adds the two operations that read the modality of a token, which a model that
reads text alone has no use for:

    keep_text_tokens    one at a text token and zero at an image token, as a real
    gate_every_stream   the gate of a stream times that number, which sets the gate of
                        an image token to zero

The released code also writes a pad identifier into a lookback slot that holds an image
token and into every slot behind it. The hash here reads the identifiers alone and
leaves that padding out.

The module holds no `ops.GenericOperator`. The hash is written out over `ops.FixedArray`,
`ops.Einops` retyped by `ops.with_datatypes`, `ops.BitwiseXor`, `ops.Modulo` and
`ops.Cast`, which `data_structure/Operators.py` has held since 2026-09-18.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as ParaBlockOperator

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    hold, over, para_boxed, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, X, m, n, x
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (
    ENGRAM_COLOUR, GATE_COLOUR, KEY, TOKEN_IDS, VALUE, address_table_rows,
    map_tokens_to_compressed_identifiers, named_for_layer, normalised_dot_product,
    read_key_and_value, require_engram_layer, signed_root_gate)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import (
    MODALITY, SLOT_IDS, SLOT_MODALITY)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

ENGRAM_BOX = 'Eng'

MODALITY_OF_EVERY_TOKEN = cat.Array(MODALITY, (x,))


def keep_text_tokens() -> cat.BroadcastedCategory:
    '''One at a text token and zero at an image token, as a real. The modality is zero
    for text and one for an image, and the formula reads a natural number and returns
    a real.'''
    return over((x,), ops.Arithmetic.template(
        nm.Integer(1) - nm.x, base=cat.Array(MODALITY, ()), output_datatype=R))


def gate_every_stream(layer: int) -> cat.Block:
    '''From the residual, the keys and the modality to one gate per token and stream.
    The gate of an image token is zero, so an image token receives nothing from
    Engram.'''
    return cat.Block.template(
        ((normalised_dot_product(layer) @ over((x, n), signed_root_gate()))
         * keep_text_tokens())
        @ ops.Einops.template('x n, x -> x n'),
        title=text.ENGRAM_GATE_TITLE, fill_color=GATE_COLOUR,
        formula=(
            'y[i_{x}, i_{n}] = \\frac{\\sum_{i_{m} \\in m} X[i_{x}, i_{n}, i_{m}]\\, '
            'w[i_{n}, i_{m}]\\, k[i_{x}, i_{n}, i_{m}]}'
            '{\\mathrm{rms}(X[i_{x}, i_{n}])\\, \\mathrm{rms}(k[i_{x}, i_{n}])}, '
            '\\quad g[i_{x}, i_{n}] = (1 - \\mathrm{mod}[i_{x}])\\, '
            '\\sigma\\Big( \\mathrm{sign}(y) \\sqrt{\\max(\\lvert y \\rvert '
            '\\lvert m \\rvert^{-1/2}, \\varepsilon_{\\mathrm{g}})} \\Big)'),
        description=text.GATE_EVERY_STREAM_DESCRIPTION,
        references=(model_lines(347, 348), model_lines(356, 362),
                    model_lines(363, 364)))


def engram_with_the_modality(layer: int) -> cat.BroadcastedCategory:
    '''The module of `layer` with its three operands open: the four-stream residual,
    the token identifiers and the modality of every token. The result is the residual
    with the gated value added to every stream.'''
    require_engram_layer(layer)
    identifiers_to_key_and_value = (
        map_tokens_to_compressed_identifiers()
        @ address_table_rows(layer) @ read_key_and_value(layer))
    return ((hold(X) * identifiers_to_key_and_value * hold(MODALITY_OF_EVERY_TOKEN))
            @ route((0, 0, 1, 3, 2), (X, KEY, VALUE, MODALITY_OF_EVERY_TOKEN))
            @ (hold(X) * gate_every_stream(layer) * hold(VALUE))
            @ (hold(X) * ops.Einops.template('x n, x m -> x n m'))
            @ over((x, n, m), ops.AdditionOp.template()))


def engram_block(layer: int) -> cat.Block:
    '''The module of `layer` as one titled block from the residual to the residual.
    The two grabs stand at the top level of the block, where
    `ParaBlockOperator.expose_tape_as_ports` looks for them.'''
    return cat.Block.template(
        (hold(X)
         * Para.Grab(tape=SLOT_IDS, size=TOKEN_IDS)
         * Para.Grab(tape=SLOT_MODALITY, size=MODALITY_OF_EVERY_TOKEN))
        @ engram_with_the_modality(layer),
        title=f'\\text{{Engram of Layer {layer}}}', fill_color=ENGRAM_COLOUR,
        formula=('X\'[i_{x}, i_{n}, i_{m}] = X[i_{x}, i_{n}, i_{m}] '
                 '+ g[i_{x}, i_{n}]\\, v[i_{x}, i_{m}]'),
        description=text.ENGRAM_BLOCK_DESCRIPTION.format(layer=layer),
        references=(model_lines(328, 365), model_lines(1249, 1252),
                    model_lines(1262, 1263), inference_config_lines(43, 48),
                    inference_config_lines(63, 64)))


def engram_of_layer(layer: int) -> ParaBlockOperator.WrappedBox:
    '''The module of `layer` as a box from the four-stream residual to the four-stream
    residual, which stands between two layer blocks beside the collapse vector. The
    box records the two slots it grabs.'''
    return para_boxed(engram_block(layer), named_for_layer(ENGRAM_BOX, layer))
