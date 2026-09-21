'''The mechanisms of DeepSeek-V4.1-Flash that `whole_model.v41_flash` leaves out, each
stated over explicit shapes.

Written by Claude Fable 5.1, reasoning effort 80.

Every mechanism is written with the standard operators wherever the standard set states
it, and with one `ops.GenericOperator` at exactly the operation the standard set cannot
state. The generic operator carries a name and the shape of what it reads and writes,
and nothing else, so what it shows is the functional shape of the mechanism: which axes
it consumes, which axes it is broadcast over, and which datatype each operand carries.
`validate_omitted_mechanisms.py` holds the claims made about each mechanism here.

    generic_operator           an operator over explicit arrays, broadcast over a degree
    the pinned block           the block holding the query's newest reachable entry,
                               forced into the candidate pool
    Engram                     the module of one layer, with the token map as a table
                               that returns a `Natural`, the n-gram hash written out
                               over the integer operators of `ops`, and the lookback,
                               the lookup, the projections and the gate as standard
                               ones
    DSpark                     the arrays the drafter passes between its parts. The
                               drafter itself is `dspark_draft_chain`, with the draft
                               trunk as its one generic operator, and the sampler every
                               draft step ends in is `gumbel_max_sampler`, with the
                               exponential draw as its one generic operator
    the vision pathway         the arrays the pathway passes between its parts. The
                               pathway itself is `vision_pathway`, with the vision
                               encoder as its one generic operator, and the write of a
                               cell at the token it occupies is
                               `write_at_token_positions`
    the FP4 cache              the four-bit datatype and the axis of the scale groups
                               that `quantised_caches` rounds a cached array through
    the compressor's state     one decode step of the compressor as a generic operator

`clamped_mixture_of_experts` states the SwiGLU clamps, the router's temperature and the
correction bias chosen by the token's modality, inside the mixture that applies them,
and `mhc_with_epsilons` states the epsilons of the Sinkhorn normalisation inside the
prediction of the mixing coefficients. The notebook draws those three mechanisms from
those two modules.

`rotary_embedding` states every rotation site of the released model, each as a table of
turns inside a box, and defines that table, the YaRN frequencies and the YaRN ramp.
This module declares the complex datatype its pairs carry, because the pairs and the
rotated channels are shapes several mechanisms share. `scaled_attention_core` states the
scale on an attention score inside the core that applies it, and `rotated_indexer` the
scale on the weight of an indexer head inside the scoring box that applies it.
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst
import quantization.data_structure.Quantization as Quantization

from websocket_transfer.auxiliary_information import OperatorRole

import notebooks.sota.DeepSeekV41Flash.candidate_pool as candidate_pool
import notebooks.sota.DeepSeekV41Flash.whole_model as whole_model
from notebooks.display.explain_operators import OperatorExplanation
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.DeepSeekV41Flash.custom_operations import weights
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    P, Q, R, X, a, c, m, n, u, x)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    engram_lines, inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.released_constants import (
    ENGRAM_GATE_FLOOR, NORM_EPSILON)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

t = cat.RawAxis.named('t')
z = fd.DynamicName('z').capture(cat.RawAxis(_size=nm.Integer(2) * t.local_size()))
zbar = fd.DynamicName('\\bar{z}').capture(
    cat.RawAxis(_size=c.local_size() - z.local_size()))
L = cat.RawAxis.named('L')
G = cat.RawAxis.named('G')
K = cat.RawAxis.named('K')
D = cat.RawAxis.named('D')
S = cat.RawAxis.named('S')
Z = cat.RawAxis.named('Z')
U = cat.RawAxis.named('U')
V = fd.DynamicName('V').capture(cat.RawAxis(_size=U.local_size()))
F = cat.RawAxis.named('F')
M = cat.RawAxis.named('M')
Hp = cat.RawAxis.named('H\'')
Wp = cat.RawAxis.named('W\'')
H = fd.DynamicName('H').capture(cat.RawAxis(_size=U.local_size() * Hp.local_size()))
W = fd.DynamicName('W').capture(cat.RawAxis(_size=V.local_size() * Wp.local_size()))
E = cat.RawAxis.named('E')

COMPLEX = dst.Complex(R)
TOKEN_IDS = whole_model.v41_flash.dom()[0]
VOCABULARY = whole_model.v41_flash.cod()[0].shape()[1]
COMPRESSED_IDS = cat.Natural.template('\\hat{v}')
FP4 = Quantization.E2M1
LATENT = cat.Array(R, (c,))


def generic_operator(
    name: str,
    inputs: tuple[cat.Array, ...],
    outputs: tuple[cat.Array, ...],
    degree: tuple[cat.Axis, ...] = (),
) -> cat.Broadcasted:
    '''An `ops.GenericOperator` consuming the whole shape of each input and producing
    the whole shape of each output, broadcast over `degree`, which leads every weave.
    A tiled operator is the same function at every index of its degree, so an axis the
    mechanism is a function of stands in the arrays and never in `degree`.'''
    tiled = (cat.WeaveMode.TILED,) * len(degree)
    return cat.Broadcasted(
        operator=ops.GenericOperator(name=fd.DynamicName.from_str(name)),
        input_weaves=tuple(cat.Weave(array.datatype, tiled + tuple(array.shape()))
                           for array in inputs),
        output_weaves=tuple(cat.Weave(array.datatype, tiled + tuple(array.shape()))
                            for array in outputs),
        reindexings=tuple(cat.ProdObject(degree).identity() for _ in inputs),
        backup_degree=cat.ProdObject(degree) if not inputs and degree else None)


# The pinned block

def maximum_over_offsets() -> cat.BroadcastedCategory:
    '''Each block scored by its best offset, as `candidate_pool` writes it.'''
    return over((P,), cat.Broadcasted(
        operator=ops.Maximum(),
        input_weaves=(cat.Weave(R, (u,)),),
        output_weaves=(cat.Weave(R, ()),),
        reindexings=(cat.ProdObject().identity(),)))


# Engram

LOOKBACK = sc.StrideMorphism(
    _dom=(x, L),
    _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
    name=fd.DynamicName('ngram'))
LOOKBACK_VIEW = mark_sparse_domains.guarded_view(
    reindexing=(LOOKBACK,), base=COMPRESSED_IDS, name='ngram')
L_reach = LOOKBACK_VIEW.cod()[0].shape()[1]
NGRAM_FEATURES = cat.Array(R, (x, G, K, D))
KEY = cat.Array(R, (x, n, m))
VALUE = cat.Array(R, (x, m))
ENGRAM_LAYERS: tuple[int, ...] = (1, 14)
TABLE_ROWS_OF_LAYER: dict[int, int] = {1: 384_006_168, 14: 384_016_682}
TOKEN_MAP_NAME = '\\mathrm{cmp}'
TABLE_NAME = '\\mathrm{ng}'
KEY_PROJECTION_NAME = 'W^{K}'
VALUE_PROJECTION_NAME = 'W^{V}'
STREAM_WEIGHT_NAME = '\\mathrm{w}'
MULTIPLIERS_NAME = '\\alpha'
PRIMES_NAME = 'p'
GATE_NAME = '\\mathrm{gate}'
ENGRAM_COLOUR = '#E4EFD9'
TOKEN_MAP_COLOUR = '#FCE0E1'
TABLE_COLOUR = '#D9E7F5'
GATE_COLOUR = '#F7E0EF'


class LayerHoldsNoEngram(ValueError):
    '''An Engram module asked for at a layer the released configuration gives none.'''


def require_engram_layer(layer: int) -> None:
    if layer not in ENGRAM_LAYERS:
        raise LayerHoldsNoEngram(
            f'layer {layer} holds no Engram module, and the layers that hold one are '
            f'{ENGRAM_LAYERS}')


def named_for_layer(body: str, layer: int) -> str:
    '''`body` with the number of the layer as its subscript, which keeps the
    multipliers, the primes, the table and the weights of the two modules apart in a
    figure and on the tape.'''
    return f'{body}_{{{layer}}}'


def table_row_count(layer: int) -> nm.FreeNumeric:
    '''The symbol for the number of rows of the table of `layer`. The two tables have
    different numbers of rows, so each layer has a symbol of its own, `T` with the
    number of its layer as its subscript. The subscript is a `fd.DynamicName` of its
    own rather than markup inside the body, so that the size a configuration assigns
    joins the subscript after a colon and the label reads `T_{1:384006168}`. A body
    holding its own subscript takes a second one beside it, which is not LaTeX.'''
    return nm.FreeNumeric.named(fd.DynamicName('T', fd.DynamicName(str(layer))))


def table_row_of_layer(layer: int) -> cat.Natural:
    '''A row number of the table of `layer`, which the hash of that layer returns and
    the table of that layer is read at.'''
    return cat.Natural(table_row_count(layer))


def map_tokens_to_compressed_identifiers() -> cat.Block:
    '''The token map of the released code, `cmp`, as a lookup whose result is a
    `Natural`. `para.processing.show_grabbed_parameters.parameter_arrays` grabs no
    weight for an `ops.Embedding`, so the figure shows no learned array for the fixed
    table.'''
    return cat.Block.template(
        over((x,), ops.Embedding.template(
            TOKEN_IDS.datatype, (), name=TOKEN_MAP_NAME, datatype=COMPRESSED_IDS)),
        title=text.TOKEN_MAP_TITLE, fill_color=TOKEN_MAP_COLOUR,
        formula='\\mathrm{cmp}(\\mathrm{id})[i_{x}] = M[\\mathrm{id}[i_{x}]]',
        description=text.MAP_TOKENS_TO_COMPRESSED_IDENTIFIERS_DESCRIPTION,
        references=(engram_lines(17, 61), engram_lines(164),
                    inference_config_lines(48)))


INT64 = cat.Natural(nm.Power(base=nm.Integer(2), exponent=nm.Integer(63)))
MULTIPLIER = cat.Natural(INT64.max_value / COMPRESSED_IDS.max_value)
PRODUCT = cat.Natural(
    nm.cancel_reciprocal_factors(COMPRESSED_IDS.max_value * MULTIPLIER.max_value))
PRIME_BOUND = nm.FreeNumeric.named('\\bar{p}')
PRIME = cat.Natural(PRIME_BOUND)
Lp = fd.DynamicName("L'").capture(cat.RawAxis(_size=L.local_size()))
PREFIX = sc.StrideMorphism(
    _dom=(G, Lp),
    _cod_stride_shift=((L, (nm.Integer(1), nm.Integer(1)),
                        nm.Integer(1) - G.local_size()),),
    name=fd.DynamicName('prefix'))
PREFIX_VIEW = mark_sparse_domains.guarded_view(
    reindexing=(hold(x) * PREFIX,), base=INT64, name='prefix')
L_prefix = PREFIX_VIEW.cod()[0].shape()[2]
GK = fd.DynamicName('GK').capture(cat.RawAxis(_size=G.local_size() * K.local_size()))
GKp = fd.DynamicName("GK'").capture(cat.RawAxis(_size=GK.local_size()))
PRIMES = cat.Array(PRIME, (GK,))
BEFORE = sc.StrideMorphism(
    _dom=(GK, GKp),
    _cod_stride_shift=((GK, (nm.Integer(1), nm.Integer(1)),
                        nm.Integer(-1) * GK.local_size()),),
    name=fd.DynamicName('before'))
BEFORE_VIEW = mark_sparse_domains.guarded_view(
    reindexing=(BEFORE,), base=PRIME, name='before')
GK_before = BEFORE_VIEW.cod()[0].shape()[1]
PAIR = sc.StrideMorphism(
    _dom=(G, K),
    _cod_stride_shift=((GK, (K.local_size(), nm.Integer(1)), nm.Integer(0)),),
    name=fd.DynamicName('pair'))
HASH_COLOUR = '#F5E6D9'
HASH_BOX = 'Hash'


def remainder_plus_offset(layer: int) -> cat.Natural:
    '''The bound the addition of a remainder and an offset declares, which is the sum
    of the prime bound and the row count of `layer`, because a bound on a sum of two
    naturals is the sum of their bounds.'''
    return cat.Natural(PRIME_BOUND + table_row_count(layer))


def offsets_of_layer(layer: int) -> cat.Array:
    '''The offset of every order and head, which is a row number of the table of
    `layer`.'''
    return cat.Array(table_row_of_layer(layer), (G, K))


def multiply_by_multipliers(layer: int) -> cat.BroadcastedCategory:
    '''Each identifier times the odd multiplier that `layer` gives its slot. A
    multiplier is below `2^63 / v`, where `v` bounds the identifiers, so a product is
    below `2^63` and `PRODUCT` is `INT64` once the identifier bound is cancelled
    against its reciprocal.'''
    multipliers = ops.FixedArray.template(
        named_for_layer(MULTIPLIERS_NAME, layer), (L,), MULTIPLIER)
    product = ops.with_datatypes(
        ops.Einops.template('x L, L -> x L'), (COMPRESSED_IDS, MULTIPLIER), (PRODUCT,))
    return (hold(LOOKBACK_VIEW.cod()[0]) * multipliers) @ product


def xor_prefixes() -> cat.BroadcastedCategory:
    '''The exclusive or of the products of the newest two, three and four slots, one
    per order. The view named prefix reads slot `i_L' + i_G + 1 - |G|` for the order
    `i_G`, so the slots of an order below the last read before the first slot and
    hold the unit, which the exclusive or ignores.'''
    return PREFIX_VIEW @ over((x, G), ops.BitwiseXor.template(L_prefix, INT64))


def sum_primes_before(layer: int) -> cat.BroadcastedCategory:
    '''The offset of every pair of an order and a head, which is the sum of the primes
    before its own in the order the reference draws them. The view named before
    reads, for the pair `i_GK`, the prime at `i_GK' + i_GK - |GK|`, so the primes
    before the pair fill the last `i_GK` slots and the other slots read before the
    first prime and hold the unit. The sum is below the row count of `layer`, because
    that count is the sum of every prime of the layer.'''
    return BEFORE_VIEW @ ops.with_datatypes(
        ops.Einops.template('GK GKp -> GK'), (PRIME,), (table_row_of_layer(layer),))


def read_pairs_as_orders_and_heads(datatype: cat.Datatype) -> cat.BroadcastedCategory:
    '''An array over the `|G| |K|` pairs read as an array over the orders and the
    heads, the pair `|K| i_G + i_K` standing at the order `i_G` and the head `i_K`.'''
    return ops.View.template(base=datatype, reindexing=(PAIR,), name='pair')


def primes_and_offsets(layer: int) -> cat.BroadcastedCategory:
    '''The prime of every order and head beside the offset of that pair, both read
    off one fixed array of the `|G| |K|` primes of `layer`.'''
    return (ops.FixedArray.template(
                named_for_layer(PRIMES_NAME, layer), (GK,), PRIME)
            @ route((0, 0), (PRIMES,))
            @ (hold(PRIMES) * sum_primes_before(layer))
            @ (read_pairs_as_orders_and_heads(PRIME)
               * read_pairs_as_orders_and_heads(table_row_of_layer(layer))))


def reduce_modulo_primes() -> cat.BroadcastedCategory:
    '''The remainder of each order's exclusive or after division by the prime of the
    order and the head, which is below that prime.'''
    return ops.Modulo.template('x G, G K -> x G K', INT64, PRIME)


def offset_into_table(layer: int) -> cat.BroadcastedCategory:
    '''The remainder plus the offset of the order and the head, which is the number of
    the first row of the table that belongs to that pair. The sum is below the sum of
    the primes up to and including the pair, which is at most the row count of
    `layer`, so the cast into a row number loses nothing.'''
    row = table_row_of_layer(layer)
    sum_of_bounds = remainder_plus_offset(layer)
    add = ops.with_datatypes(
        ops.AdditionOp.template('x G K, G K -> x G K', PRIME),
        (PRIME, row), (sum_of_bounds,))
    return add @ over((x, G, K), ops.Cast.template(sum_of_bounds, to=row))


def hash_formula(layer: int) -> str:
    '''The formula the hash block of `layer` shows, with the layer as the subscript of
    the hash, the multipliers, the primes and the offsets.'''
    return (
        f'\\mathrm{{hash}}_{{{layer}}}(c)[i_{{G}}, i_{{K}}] = \\Big( '
        '\\bigoplus_{i_{L} \\in L,\\; i_{L} \\leq i_{G} + 1} '
        f'\\alpha_{{{layer}}}[i_{{L}}]\\, c[i_{{L}}] \\Big) '
        f'\\bmod p_{{{layer}}}[i_{{G}}, i_{{K}}] + o_{{{layer}}}[i_{{G}}, i_{{K}}]')


def hash_ngrams(
    layer: int,
) -> cat.Broadcasted[cat.Datatype, cat.Axis, ops.BlockOperator[cat.Datatype, cat.Axis]]:
    '''The rows of the table of `layer` that the 2-gram, the 3-gram and the 4-gram
    ending at a token address, one row per order and hash head, written out over the
    integer operators inside one box. The figure draws the box, and its inspection box
    draws the body.'''
    require_engram_layer(layer)
    return boxed(cat.Block.template(
        multiply_by_multipliers(layer) @ xor_prefixes()
        @ (hold(cat.Array(INT64, (x, G))) * primes_and_offsets(layer))
        @ (reduce_modulo_primes() * hold(offsets_of_layer(layer)))
        @ offset_into_table(layer),
        title=text.HASH_TITLE, fill_color=HASH_COLOUR,
        formula=hash_formula(layer),
        description=(text.HASH_WRITTEN_OUT_DESCRIPTION + ' '
                     + text.HASH_OF_LAYER_SENTENCE.format(
                         layer=layer, rows=f'{TABLE_ROWS_OF_LAYER[layer]:,}')),
        references=(engram_lines(169, 175), engram_lines(179, 184),
                    engram_lines(64, 83), engram_lines(108, 118),
                    engram_lines(148, 149))),
        named_for_layer(HASH_BOX, layer))


def address_table_rows(layer: int) -> cat.BroadcastedCategory:
    '''From the compressed identifier of every token to the rows of the table of
    `layer` that the token's three n-grams address. The lookback stands in front of
    the hash, because it is the one operation that reads across tokens.'''
    return LOOKBACK_VIEW @ hash_ngrams(layer)


def look_up_rows(layer: int) -> cat.BroadcastedCategory:
    return over((x, G, K), ops.Embedding.template(
        table_row_of_layer(layer), (D,), name=named_for_layer(TABLE_NAME, layer)))


def project_key_and_value(layer: int) -> cat.BroadcastedCategory:
    '''One key per residual stream and one value shared by the streams, both read off
    the looked-up rows of every order and head together. The released code holds one
    weight and cuts its result in two, and a linear map whose result is cut into parts
    is one linear map per part.'''
    return (route((0, 0), (NGRAM_FEATURES,))
            @ (over((x,), ops.Linear.template(
                (G, K, D), (n, m), named_for_layer(KEY_PROJECTION_NAME, layer)))
               * over((x,), ops.Linear.template(
                   (G, K, D), (m,), named_for_layer(VALUE_PROJECTION_NAME, layer)))))


def read_key_and_value(layer: int) -> cat.Block:
    '''From the rows a token addresses to the token's keys and value.'''
    require_engram_layer(layer)
    return cat.Block.template(
        look_up_rows(layer) @ project_key_and_value(layer),
        title=text.TABLE_TITLE, fill_color=TABLE_COLOUR,
        description=text.ENGRAM_TABLE_DESCRIPTION.format(
            layer=layer, rows=f'{TABLE_ROWS_OF_LAYER[layer]:,}'),
        references=(model_lines(309, 310), model_lines(317, 321),
                    model_lines(343, 345), model_lines(353, 355),
                    inference_config_lines(45)))


def normalise_over_the_channels() -> cat.Broadcasted:
    '''Every stream of every token divided by its root mean square over the channels,
    with the epsilon the released code adds under the root. The released normalisation
    multiplies by no learned gain and adds no bias, so the operator declares
    neither.'''
    return over((x, n), ops.Normalize.template(
        (m,), gain=False, bias=False, epsilon=NORM_EPSILON))


def weighted_dot_product(layer: int) -> cat.BroadcastedCategory:
    '''Each stream against its key, summed over the channels under the learned weight
    of `layer`, which is the product of the released `q_weight` and `k_weight`.'''
    stream_weight = weights(named_for_layer(STREAM_WEIGHT_NAME, layer), (n, m))
    return ((hold(X) * stream_weight * hold(KEY))
            @ ops.Einops.template('x n m, n m, x n m -> x n'))


def normalised_dot_product(layer: int) -> cat.BroadcastedCategory:
    '''Each stream of every token against its key under the learned weight of `layer`,
    with the stream and the key each divided by its own root mean square over the
    channels first. Dividing the two operands before the sum gives the same number as
    dividing the sum by the two root mean squares, because both are constant along the
    channels the sum runs over.'''
    return ((normalise_over_the_channels() * normalise_over_the_channels())
            @ weighted_dot_product(layer))


def signed_root_gate() -> cat.Broadcasted:
    '''The gate of one stream from its normalised dot product: the sign of the product
    times the square root of its magnitude held at or above epsilon, through a
    sigmoid.'''
    scaled = nm.x * nm.Power.template(m.local_size(), nm.Integer(-1) / nm.Integer(2))
    magnitude = nm.LargerOf(nm.AbsoluteValue(scaled), ENGRAM_GATE_FLOOR)
    return ops.Arithmetic.template(
        nm.Sigmoid(nm.Sign(scaled) * nm.SquareRoot(magnitude)), name=GATE_NAME)


def gate_every_stream(layer: int) -> cat.Block:
    '''From the residual and the keys to one gate per token and stream.'''
    return cat.Block.template(
        normalised_dot_product(layer) @ over((x, n), signed_root_gate()),
        title=text.ENGRAM_GATE_TITLE, fill_color=GATE_COLOUR,
        formula=(
            'y[i_{x}, i_{n}] = \\frac{\\sum_{i_{m} \\in m} X[i_{x}, i_{n}, i_{m}]\\, '
            'w[i_{n}, i_{m}]\\, k[i_{x}, i_{n}, i_{m}]}'
            '{\\mathrm{rms}(X[i_{x}, i_{n}])\\, \\mathrm{rms}(k[i_{x}, i_{n}])}, '
            '\\quad g[i_{x}, i_{n}] = '
            '\\sigma\\Big( \\mathrm{sign}(y) \\sqrt{\\max(\\lvert y \\rvert '
            '\\lvert m \\rvert^{-1/2}, \\varepsilon_{\\mathrm{g}})} \\Big)'),
        description=text.ENGRAM_GATE_WITHOUT_THE_MODALITY_DESCRIPTION,
        references=(model_lines(347, 348), model_lines(356, 362),
                    model_lines(363, 364)))


def engram_expression(layer: int) -> cat.BroadcastedCategory:
    '''The module of `layer` with its two operands open: the four-stream residual and
    the token identifiers. The result is the residual with the gated value added to
    every stream.'''
    require_engram_layer(layer)
    identifiers_to_key_and_value = (
        map_tokens_to_compressed_identifiers()
        @ address_table_rows(layer) @ read_key_and_value(layer))
    return ((hold(X) * identifiers_to_key_and_value)
            @ route((0, 0, 1, 2), (X, KEY, VALUE))
            @ (hold(X) * gate_every_stream(layer) * hold(VALUE))
            @ (hold(X) * ops.Einops.template('x n, x m -> x n m'))
            @ over((x, n, m), ops.AdditionOp.template()))


def engram_module(layer: int) -> cat.Block:
    '''The four-stream residual of `layer` with the gated n-gram value added to every
    stream, as one titled block.'''
    return cat.Block.template(
        engram_expression(layer),
        title=f'\\text{{Engram of Layer {layer}}}', fill_color=ENGRAM_COLOUR,
        formula=('X\'[i_{x}, i_{n}, i_{m}] = X[i_{x}, i_{n}, i_{m}] '
                 '+ g[i_{x}, i_{n}]\\, v[i_{x}, i_{m}]'),
        description=text.ENGRAM_MODULE_OF_LAYER_DESCRIPTION.format(layer=layer),
        references=(model_lines(328, 365), inference_config_lines(43, 48),
                    inference_config_lines(63, 64)))


def fixed_array_roles(layer: int) -> dict[
        str, tuple[str, str, fd.Prod[cat.CodeReference]]]:
    '''The rows an inspection box shows over the two fixed arrays of the hash of
    `layer`, keyed by the text of each array's name. The bound of an array's datatype
    is a bound on every entry, because a wire carries one datatype.'''
    rows = f'{TABLE_ROWS_OF_LAYER[layer]:,}'
    return {
        fd.DynamicName.from_str(
            named_for_layer(MULTIPLIERS_NAME, layer)).to_bodies(): (
            f'\\alpha_{{{layer}}}[i_{{L}}] < 2^{{63}} / \\hat{{v}}, \\quad '
            f'\\alpha_{{{layer}}}[i_{{L}}] \\text{{ odd}}',
            text.ENGRAM_MULTIPLIERS_ROLE.format(layer=layer),
            (engram_lines(64, 83),)),
        fd.DynamicName.from_str(named_for_layer(PRIMES_NAME, layer)).to_bodies(): (
            f'p_{{{layer}}}[0] < p_{{{layer}}}[1] < \\dots '
            f'< p_{{{layer}}}[|G||K| - 1] < \\bar{{p}}, \\quad '
            f'p_{{{layer}}}[i_{{GK}}] \\text{{ prime}}, \\quad '
            f'T_{{{layer}}} = \\sum_{{i_{{GK}} \\in GK}} p_{{{layer}}}[i_{{GK}}]',
            text.ENGRAM_PRIMES_ROLE.format(layer=layer, rows=rows),
            (engram_lines(102, 118),)),
    }


def bound_name(natural: cat.Natural) -> str | None:
    '''The bodies of the name of the free numeric bounding `natural`, which a figure
    writing the assigned size into the name leaves as they are, and `None` for a
    bound that is not a named numeric.'''
    bound = natural.max_value
    if isinstance(bound, nm.FreeNumeric) and bound.uid._name is not None:
        return bound.uid._name.to_bodies()
    return None


def cast_roles(layer: int) -> dict[str, str]:
    '''The row an inspection box shows over the one cast of the hash of `layer`, keyed
    by the name of the bound the cast narrows to, per `bound_name`.'''
    row_name = bound_name(table_row_of_layer(layer))
    if row_name is None:
        raise LayerHoldsNoEngram(f'the row count of layer {layer} has no name')
    return {row_name: text.HASH_CAST_DESCRIPTION.format(layer=layer)}


def weight_roles(layer: int) -> dict[str, OperatorRole]:
    '''The roles of the three learned arrays of the Engram module of `layer`.'''
    return {
        fd.DynamicName.from_str(
            named_for_layer(KEY_PROJECTION_NAME, layer)).to_bodies(): OperatorRole(
            role=(
                text.ENGRAM_KEY_PROJECTION_ROLE.format(layer=layer)),
            references=(model_lines(345), model_lines(353, 355))),
        fd.DynamicName.from_str(
            named_for_layer(VALUE_PROJECTION_NAME, layer)).to_bodies(): OperatorRole(
            role=(
                text.ENGRAM_VALUE_PROJECTION_ROLE.format(layer=layer)),
            references=(model_lines(345), model_lines(353, 354))),
        fd.DynamicName.from_str(
            named_for_layer(STREAM_WEIGHT_NAME, layer)).to_bodies(): OperatorRole(
            role=(
                text.ENGRAM_GATE_WEIGHT_ROLE.format(layer=layer)),
            references=(model_lines(347, 348), model_lines(356))),
    }


def table_explanation(layer: int) -> OperatorExplanation:
    '''The box over the n-gram table of the Engram module of `layer`.'''
    return OperatorExplanation(
        title=text.TABLE_TITLE,
        formula=f'E_{{{layer}}}(r)[i_{{D}}] = N_{{{layer}}}[r, i_{{D}}]',
        description=text.NGRAM_TABLE_ROW_DESCRIPTION.format(
            layer=layer, rows=f'{TABLE_ROWS_OF_LAYER[layer]:,}'),
        references=(model_lines(296, 325), inference_config_lines(45),
                    inference_config_lines(64)))


TOKEN_MAP_EXPLANATION = OperatorExplanation(
    title=text.TOKEN_MAP_TITLE,
    formula=r'\mathrm{cmp}(\mathrm{id})[i_{x}] = M[\mathrm{id}[i_{x}]]',
    description=text.TOKEN_MAP_DESCRIPTION,
    references=(engram_lines(17, 61), engram_lines(164), inference_config_lines(48)))

ENGRAM_FIXED_ARRAY_ROLES: dict[str, tuple[str, str, fd.Prod[cat.CodeReference]]] = {
    key: row for layer in ENGRAM_LAYERS
    for key, row in fixed_array_roles(layer).items()}

ENGRAM_CAST_ROLES: dict[str, str] = {
    key: row for layer in ENGRAM_LAYERS for key, row in cast_roles(layer).items()}

ENGRAM_WEIGHT_ROLES: dict[str, OperatorRole] = {
    key: row for layer in ENGRAM_LAYERS for key, row in weight_roles(layer).items()}

ENGRAM_REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    'ngram': ReindexingExplanation(
        description=text.LOOKBACK_VIEW_DESCRIPTION,
        references=(engram_lines(171, 172),)),
    'prefix': ReindexingExplanation(
        description=text.PREFIX_VIEW_DESCRIPTION,
        references=(engram_lines(179, 184),)),
    'before': ReindexingExplanation(
        description=text.PRIMES_BEFORE_VIEW_DESCRIPTION,
        references=(engram_lines(108, 118),)),
    'pair': ReindexingExplanation(
        description=text.PAIR_VIEW_DESCRIPTION,
        references=(engram_lines(108, 118),)),
}


def embedding_key(name: str) -> str:
    '''The key an explanation table gives an `ops.Embedding` named `name`, whose
    `template` puts the name under the body `E`.'''
    return f'E{fd.DynamicName.from_str(name).to_bodies()}'


ENGRAM_EMBEDDING_EXPLANATIONS: dict[str, OperatorExplanation] = {
    embedding_key(TOKEN_MAP_NAME): TOKEN_MAP_EXPLANATION,
    **{embedding_key(named_for_layer(TABLE_NAME, layer)): table_explanation(layer)
       for layer in ENGRAM_LAYERS},
}


# DSpark

DRAFT_STATES = cat.Array(R, (S, m))
DRAFT_STATE = cat.Array(R, (m,))
LOGITS = cat.Array(R, (VOCABULARY,))
MARKOV_EMBEDDING = cat.Array(R, (Z,))


# The vision pathway

PATCHES = cat.Array(R, (H, W, F))
PATCH_FEATURES = cat.Array(R, (H, W, M))
CELL_FEATURES = cat.Array(R, (Hp, Wp, m))
IMAGE_POSITIONS = cat.Array(cat.Natural(x.local_size()), (Hp, Wp))
STATE = cat.Array(R, (x, m))


# The compressor's state

PENDING_GROUP = cat.Array(R, (a, c))


def compress_step() -> cat.Broadcasted:
    '''One decode step of the compressor at ratio 2. The token's value and gate logit
    go into the slot of the pending group the step number names, and the entry is
    emitted on the step that completes the group.'''
    return generic_operator('\\mathrm{step}', (LATENT, LATENT, PENDING_GROUP, PENDING_GROUP),
                            (PENDING_GROUP, PENDING_GROUP, LATENT))
