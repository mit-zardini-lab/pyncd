# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The mechanisms that two or three of the classic models write the same way.

`contract` builds a contraction over declared axes through
`einops_simplification.einsum`, because a letter of `ops.Einops.template` mints a fresh
axis, and a fresh axis composed onto a declared one can take the declared axis's name.

`read_back_from_every_position` is the causal mask of a decoder, written as a read. Slot
`i_w` of position `i_x` reads position `i_x - i_w`, so slot 0 is the position itself and
no slot reads a later one. A slot before the first position names a negative position
and holds the universal unit, and `mark_sparse_domains.guarded_view` marks the slot axis
with the affine form `i_x - i_w >= 0`. A slot axis as long as the sequence is the causal
mask, and a shorter one is a sliding window.

`rotate_channel_pairs` is a rotary embedding. It reads adjacent channels as complex
numbers, multiplies each by the entry a `dst.Rotary` table holds for its position and
its pair, and writes the pairs back as channels. A model boxes it over one vector per
token as the operator `RoPE`, whose body is drawn once and opens in the inspection box,
and `broadcast_between_positions_and_channels` computes the box once per head. The
function is the one `notebooks/sota/GLM53/rotary_embedding.py` and
`notebooks/sota/DeepSeekV41Flash/rotary_embedding.py` each hold.
`residual_connection` adds the output of a sublayer to its input.

`obsidian/06-practice/Representing Models.md` states the rules these follow, and
`obsidian/02-categories/Padding and Masks as Sparse Axes.md` the reads that produce an
affine form.
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst
import deepseek.registries.standard_expansions  # noqa: F401 - the rotary table's rule

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route

R = cat.Reals()
COMPLEX = dst.Complex(R)
ROTARY_BOX = 'RoPE'


def contract[A: cat.Axis](
    inputs: tuple[tuple[einops_simplification.Entry, ...], ...],
    output: tuple[einops_simplification.Entry, ...],
) -> cat.Broadcasted[cat.Reals, A]:
    '''A contraction over declared axes, or over index variables of them where one
    axis is read at two positions, so no axis is minted or renamed.'''
    return einops_simplification.einsum(inputs, output, R)


def read_back_from_every_position[A: cat.Axis](
    positions: A, slots: A, carried: tuple[A, ...], name: str,
) -> cat.Broadcasted:
    '''An array `[positions, *carried]` read at `i_x - i_w` for every slot `i_w` of
    `slots`, which returns `[positions, slots|positions, *carried]`.'''
    back = sc.StrideMorphism(
        _dom=(positions, slots),
        _cod_stride_shift=((positions, (nm.Integer(1), nm.Integer(-1)),
                            nm.Integer(0)),),
        name=fd.DynamicName(name))
    return mark_sparse_domains.guarded_view(
        reindexing=(back, cat.ProdObject(carried).identity()), name=name)


def rotate_channel_pairs[A: cat.Axis](
    table: cat.Broadcasted[dst.Complex, A, dst.Rotary],
    positions: A, carried: tuple[A, ...], channels: A, pairs: A,
) -> cat.BroadcastedCategory:
    '''An array `[positions, *carried, channels]` with every adjacent pair of channels
    read as one complex number, multiplied by the entry `table` holds for its position
    and its pair, and written back as two channels. `table` is over
    `(positions, pairs)`.'''
    lifted = (positions, *carried)
    pair_shape = (*lifted, pairs)
    return ((over(lifted, dst.PairsAsComplex.template(
                base=R, channels=channels, pairs=pairs))
             * table)
            @ einops_simplification.einsum(
                (pair_shape, (positions, pairs)), pair_shape, COMPLEX)
            @ over(lifted, dst.Decomplex.template(
                base=R, pairs=pairs, channels=channels)))


def broadcast_between_positions_and_channels[A: cat.Axis](
    box: cat.Broadcasted,
    between: tuple[A, ...],
) -> cat.Broadcasted:
    '''A box over `[positions, channels]` computed once per index of `between`, on
    arrays `[positions, *between, channels]`. `over` prefixes the axes it broadcasts
    over, which would put them ahead of the positions.'''
    tiled = (cat.WeaveMode.TILED,) * len(between)

    def with_tiled_slots(weave: cat.Weave) -> cat.Weave:
        positions, channels = weave._shape
        return cat.Weave(weave.datatype, (positions, *tiled, channels))

    return cat.Broadcasted(
        operator=box.operator,
        input_weaves=tuple(map(with_tiled_slots, box.input_weaves)),
        output_weaves=tuple(map(with_tiled_slots, box.output_weaves)),
        reindexings=tuple(cat.ProdObject(tuple(between)).identity()
                          for _ in box.input_weaves))


def residual_connection[A: cat.Axis](
    sublayer: cat.BroadcastedCategory, state: cat.Array[cat.Reals, A],
    title: str, description: str, colour: str,
    references: tuple[cat.CodeReference, ...],
) -> cat.Block:
    '''`z + Sublayer(z)`, with the copy of `z` carried past the sublayer.'''
    return cat.Block.template(
        route((0, 0), (state,)) @ (sublayer * hold(state)) @ ops.AdditionOp.template(),
        title=title, description=description, fill_color=colour,
        references=references)


def sigmoid_weighted_input() -> cat.Broadcasted:
    '''`x \\sigma(x)`, which Mistral and DeepSeek call the SiLU. It prints as its
    formula, per the ruling that a name is given only where it is shorter than the
    formula and denotes one function.'''
    return ops.Arithmetic.template(nm.Sigmoid(nm.x) * nm.x)
