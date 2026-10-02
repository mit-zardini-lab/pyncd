# Claude Opus 5.5 (1M context), effort 40.
'''Kimi Delta Attention, the linear attention of 69 of the 93 layers of Kimi K3.

The attention is `KimiDeltaAttention` of the reference. It holds no cache of keys and
values. Every head carries a state of 128 by 128 numbers from each token to the next,
and `delta_rule_scan` writes the recurrence that updates it. Everything around the scan
is computed once per token:

    queries, keys, values   three projections of the hidden state onto 96 heads of 128
                            channels, each passed through a causal convolution of four
                            taps and a SiLU, the `ShortConvolution` of `fla`
    decays                  a projection onto a rank of 128 and back onto every key
                            channel of every head, turned into the decay of the channel
                            by the gate of `fla`
    strengths               a projection onto one number per head and a sigmoid
    normalisation           the queries and the keys divided by their length over the
                            key channels, and the queries scaled by one over the square
                            root of the number of key channels
    output                  the output of the scan normalised over the value channels of
                            each head, multiplied by the sigmoid of a third projection
                            of the hidden state, and projected back onto the hidden width

The reference passes the raw decay, the raw strengths and the unnormalised queries and
keys to `chunk_kda` with `use_gate_in_kernel`, `use_beta_sigmoid_in_kernel` and
`use_qk_l2norm_in_kernel`, and the kernel computes the gate, the sigmoid and the
normalisation before the recurrence. The expression computes them before the scan, as
operations broadcast over the tokens, which is where the kernel computes them.

The gate of `fla` with a lower bound writes the logarithm of the decay of key channel
`i_d` of head `i_j` as

    log alpha[i_x, i_j, i_d] = lambda sigma(e^{A[i_j]} (f[i_x, i_j, i_d] + b[i_j, i_d]))

with `lambda` the `gate_lower_bound` of -5, `A` the parameter `A_log` of one number per
head, `b` the parameter `dt_bias` of one number per channel, and `f` the projection. The
logarithm lies between -5 and 0, so a channel keeps between `e^{-5}` of its state and
all of it from one token to the next.

A weight read at the channel of the value it multiplies is a `Linear` that produces the
channel axis followed by the diagonal that keeps the entries where the produced channel
is the channel of the value, per `obsidian/06-practice/Representing Models.md`. The taps
of the convolution and the bias `b` are written that way. `A_log` passes through an
exponential before it is read, so it is a parameter array, a `Linear` with no inputs.
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import algebra.discovering_broadcasts as discovering_broadcasts
import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.classic.shared_mechanisms import contract
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    sigmoid, sigmoid_weighted_input, weights)
from notebooks.sota.KimiK3.declared_axes import (
    DELTA_HEAD_SCALARS, DELTA_KEY_CHANNELS, DELTA_VALUE_CHANNELS, R, STATE, d, j, m, w,
    x, z)
from notebooks.sota.KimiK3.delta_rule_scan import DELTA_RULE_SCAN
from notebooks.sota.KimiK3.reference_links import (
    GATE, GATED_NORM, GATED_NORM_KERNEL, KERNEL_COMPUTES_THE_GATE,
    KERNEL_NORMALISES_QUERIES_AND_KEYS, KERNEL_SCALE, KERNEL_SIGMOID_OF_BETA, L2_NORM,
    L2_NORM_KERNEL, SHORT_CONVOLUTION, checkpoint_config_lines, modeling_lines)
from notebooks.sota.KimiK3.released_constants import (
    GATE_LOWER_BOUND, L2_NORM_EPSILON, NORM_EPSILON)
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

PROJECTION_COLOUR = '#E8DFF0'
CONVOLUTION_COLOUR = '#B8D8CE'
GATE_COLOUR = '#FFE2BB'
NORM_COLOUR = '#DDE8D6'
OUTPUT_COLOUR = '#C1E8F7'
ATTENTION_COLOUR = '#C5BEDF'
CONVOLUTION_BOX = 'Conv'
L2_NORM_BOX = 'L2'
ATTENTION_BOX = 'KDA'
TAPS_VIEW_NAME = '\\mathrm{Taps}'
DEPTHWISE_VIEW_NAME = '\\mathrm{Depthwise}'
DECAY_NAME = 'e^{\\lambda \\sigma(x)}'
INVERSE_LENGTH_NAME = '(x + \\varepsilon_{\\mathrm{q}})^{-1/2}'
QUERY_SCALE_NAME = 'x / \\sqrt{\\lvert d \\rvert}'
EXPONENTIAL_NAME = 'e^{x}'

decay_rank = fd.DynamicName('s', code_form='decay_rank').capture(
    cat.RawAxis(_size=d.local_size()))
'''The rank of the projection that gives the decays, which the reference sets to the
width of a head, `head_dim`.'''

PROJECTION_REFERENCES = (modeling_lines(495, 502), modeling_lines(580, 582),
                         modeling_lines(605, 607), checkpoint_config_lines(94),
                         checkpoint_config_lines(166))
CONVOLUTION_REFERENCES = (modeling_lines(504, 518), modeling_lines(583, 600),
                          SHORT_CONVOLUTION, checkpoint_config_lines(167))
DECAY_REFERENCES = (modeling_lines(520, 527), modeling_lines(601, 602),
                    modeling_lines(532), modeling_lines(609, 627), GATE,
                    KERNEL_COMPUTES_THE_GATE, checkpoint_config_lines(93))
STRENGTH_REFERENCES = (modeling_lines(529), modeling_lines(603), KERNEL_SIGMOID_OF_BETA)
L2_NORM_REFERENCES = (KERNEL_NORMALISES_QUERIES_AND_KEYS, L2_NORM, L2_NORM_KERNEL,
                      modeling_lines(620))
QUERY_SCALE_REFERENCES = (KERNEL_SCALE,)
OUTPUT_REFERENCES = (modeling_lines(531, 541), modeling_lines(651, 659), GATED_NORM,
                     GATED_NORM_KERNEL, checkpoint_config_lines(168))
ATTENTION_REFERENCES = (modeling_lines(477, 663), checkpoint_config_lines(66, 169))


TAPS_READ = mark_sparse_domains.mark_sparse_domain(sc.StrideMorphism(
    _dom=(x, w),
    _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
    name=fd.DynamicName(TAPS_VIEW_NAME)))
'''The read of token `i_x - i_w` at tap `i_w` of each token `i_x`, with the tap axis
marked `w|x`, live where `i_x - i_w >= 0`. The read is marked once, so the convolutions
of the queries, the keys and the values carry one tap axis, and the CausalSlide finds
one read asked of the hidden state by the three of them.'''
taps = TAPS_READ._dom[1]


def read_back_over_the_taps[A: cat.Axis](channels: tuple[A, ...]) -> cat.Broadcasted:
    '''An array over `(x, *channels)` read at `i_x - i_w` for every tap `i_w`, which
    returns `[x, w|x, *channels]`. A tap before the first token reads the universal
    unit.'''
    return ops.View.template(
        reindexing=(TAPS_READ, cat.ProdObject(channels).identity()), name=TAPS_VIEW_NAME)


def weigh_the_taps[A: cat.Axis](channels: tuple[A, ...], taps_name: str
                                 ) -> cat.Broadcasted:
    '''`R[x, w|x, *channels] -> R[x, *channels, *channels]`: a `Linear` from the taps
    onto every channel, computed once per token and channel. The tap axis stands second
    in the operand, so the weave is written by position.'''
    T = cat.WeaveMode.TILED
    degree = (x, *channels)
    return cat.Broadcasted(
        operator=ops.Linear(name=fd.DynamicName.from_str(taps_name).reconstruct(
            settings=fd.DynamicNameSettings(bold=True))),
        input_weaves=(cat.Weave(R, (T, taps, *(T for _ in channels))),),
        output_weaves=(cat.Weave(R, (*(T for _ in degree), *channels)),),
        reindexings=(cat.ProdObject(degree).identity(),))


def keep_the_diagonal[A: cat.Axis](channels: tuple[A, ...]) -> cat.Broadcasted:
    '''An array over `(x, *channels, *channels)` read where the two copies of the
    channels agree.'''
    positions = tuple(range(1, 1 + len(channels)))
    return ops.View.template(
        reindexing=cat.Rearrangement((0, *positions, *positions), (x, *channels)),
        name=DEPTHWISE_VIEW_NAME)


def causal_convolution[A: cat.Axis](channels: tuple[A, ...], taps_name: str
                                     ) -> cat.Broadcasted:
    '''The depthwise causal convolution of `|w|` taps over every channel, then the SiLU,
    as one box. Tap `i_w` of token `i_x` reads token `i_x - i_w`, which holds the
    universal unit before the first token, as the causal padding of the reference
    does.'''
    return boxed(cat.Block.template(
        read_back_over_the_taps(channels)
        @ weigh_the_taps(channels, taps_name)
        @ keep_the_diagonal(channels)
        @ over((x, *channels), sigmoid_weighted_input()),
        title=text.CONVOLUTION_TITLE, fill_color=CONVOLUTION_COLOUR,
        description=text.CONVOLUTION_DESCRIPTION,
        formula=('y[i_{x}, i_{j}, i_{d}] = \\mathrm{SiLU}\\Big(\\sum_{i_{w} \\in w} '
                 'W^{C}[i_{w}, i_{j}, i_{d}]\\, v[i_{x} - i_{w}, i_{j}, i_{d}]\\Big)'),
        references=CONVOLUTION_REFERENCES), CONVOLUTION_BOX)


def project_and_convolve[A: cat.Axis](
    channels: tuple[A, ...], projection_name: str, taps_name: str,
) -> cat.BroadcastedCategory:
    return ((x >> ops.Linear.template((m,), channels, projection_name))
            @ causal_convolution(channels, taps_name))


def l2_normalisation_body() -> cat.Block:
    '''One head of one token divided by its length over the key channels, with
    `\\varepsilon_{\\mathrm{q}}` added to the square of the length.'''
    head = cat.Array(R, (d,))
    return cat.Block.template(
        route((0, 0, 0), (head,))
        @ (contract(((d,), (d,)), ()) * hold(head))
        @ (ops.Arithmetic.template(
            nm.Integer(1) / nm.SquareRoot(nm.x + L2_NORM_EPSILON),
            name=fd.DynamicName(INVERSE_LENGTH_NAME)) * hold(head))
        @ contract(((), (d,)), (d,)),
        title=text.L2_NORM_TITLE, fill_color=NORM_COLOUR,
        description=text.L2_NORM_DESCRIPTION,
        formula=('y[i_{d}] = v[i_{d}] \\Big/ \\sqrt{\\sum_{j_{d} \\in d} v[j_{d}]^{2} + '
                 '\\varepsilon_{\\mathrm{q}}}'),
        references=L2_NORM_REFERENCES)


def l2_normalisation() -> cat.Broadcasted:
    '''The L2 normalisation of every head of every token, one box computed once per
    token and head.'''
    return discovering_broadcasts.broadcast_block_over_axes(
        l2_normalisation_body(), (x, j), ((0, 1),), L2_NORM_BOX)


def scale_the_queries() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.x / nm.SquareRoot(d.local_size()),
                                   name=QUERY_SCALE_NAME)


def decays() -> cat.Block:
    '''`STATE[x, m] -> R[x, j, d]`: the decay of every key channel of every head, from
    a projection through a rank of `|d|`, the bias `b^{\\Delta}` of the channel, the
    factor `e^{A}` of the head and the gate `e^{\\lambda \\sigma(x)}`.'''
    channels = (x, j, d)
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (decay_rank,), 'W^{Fa}'))
        @ (x >> ops.Linear.template((decay_rank,), (j, d), 'W^{Fb}'))
        @ (hold(DELTA_KEY_CHANNELS) * weights('b^{\\Delta}', (j, d)))
        @ einops_simplification.einsum((channels, (j, d)), channels, R, ops.AdditionOp())
        @ (hold(DELTA_KEY_CHANNELS)
           * (weights('A_{\\log}', (j,))
              @ ops.Arithmetic.template(nm.E ** nm.x, base=cat.Array(R, (j,)),
                                        name=EXPONENTIAL_NAME)))
        @ contract((channels, (j,)), channels)
        @ ops.Arithmetic.template(
            nm.E ** (GATE_LOWER_BOUND * nm.Sigmoid(nm.x)), base=DELTA_KEY_CHANNELS,
            name=fd.DynamicName(DECAY_NAME)),
        title=text.DECAY_TITLE, fill_color=GATE_COLOUR,
        description=text.DECAY_DESCRIPTION,
        formula=('\\alpha[i_{x}, i_{j}, i_{d}] = \\exp\\Big(\\lambda\\, \\sigma\\big('
                 'e^{A[i_{j}]} (f[i_{x}, i_{j}, i_{d}] + b^{\\Delta}[i_{j}, i_{d}])'
                 '\\big)\\Big)'),
        references=DECAY_REFERENCES)


def strengths() -> cat.Block:
    '''`STATE[x, m] -> R[x, j]`: the strength of the write of every head.'''
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (j,), 'W^{\\beta}')) @ sigmoid(),
        title=text.STRENGTH_TITLE, fill_color=GATE_COLOUR,
        description=text.STRENGTH_DESCRIPTION, references=STRENGTH_REFERENCES)


def queries_keys_and_values() -> cat.Block:
    '''`STATE[x, m] -> R[x, j, d], R[x, j, d], R[x, j, z]`: the three projections, each
    convolved over the last four tokens, and the queries and the keys normalised to
    length one over the key channels, the queries scaled.'''
    return cat.Block.template(
        route((0, 0, 0), (STATE,))
        @ ((project_and_convolve((j, d), 'W^{Q}', 'W^{Cq}') @ l2_normalisation()
            @ scale_the_queries())
           * (project_and_convolve((j, d), 'W^{K}', 'W^{Ck}') @ l2_normalisation())
           * project_and_convolve((j, z), 'W^{V}', 'W^{Cv}')),
        title=text.DELTA_QKV_TITLE, fill_color=PROJECTION_COLOUR,
        description=text.DELTA_QKV_DESCRIPTION,
        references=(*PROJECTION_REFERENCES, *L2_NORM_REFERENCES,
                    *QUERY_SCALE_REFERENCES))


def normalise_and_gate_the_output() -> cat.Block:
    '''`R[x, j, z], STATE[x, m] -> STATE[x, m]`: the output of the scan normalised over
    the value channels of each head with a gain shared by the heads, multiplied by the
    sigmoid of the gate projection, and projected back onto the hidden width.'''
    return cat.Block.template(
        (over((x, j), ops.Normalize.template((z,), epsilon=NORM_EPSILON))
         * ((x >> ops.Linear.template((m,), (j, z), 'W^{Gk}')) @ sigmoid()))
        @ ops.Einops.template('x j z, x j z -> x j z')
        @ (x >> ops.Linear.template((j, z), (m,), 'W^{Ok}')),
        title=text.DELTA_OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.DELTA_OUTPUT_DESCRIPTION, references=OUTPUT_REFERENCES)


def delta_attention() -> cat.Block:
    '''`STATE[x, m] -> STATE[x, m]`: the queries, the keys and the values, the decays
    and the strengths, the scan over the tokens, and the gated output.'''
    return cat.Block.template(
        route((0, 0, 0, 0), (STATE,))
        @ (queries_keys_and_values() * decays() * strengths() * hold(STATE))
        @ (DELTA_RULE_SCAN * hold(STATE))
        @ normalise_and_gate_the_output(),
        title=text.KDA_TITLE, fill_color=ATTENTION_COLOUR,
        description=text.KDA_DESCRIPTION, references=ATTENTION_REFERENCES)


DELTA_ATTENTION = boxed(delta_attention(), ATTENTION_BOX)
