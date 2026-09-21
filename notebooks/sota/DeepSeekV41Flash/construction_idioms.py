'''The construction idioms every module of DeepSeek-V4.1-Flash is written with.

Written by Claude Opus 5, effort high.

`hold`, `route` and `over` are the three wiring idioms the notebooks under
`notebooks/sota/` share. `l1_norm_over` is the fourth, and builds the normalisation
`y = x / sum(x)` over one position of a shape. `boxed` turns a `cat.Block` into an
`ops.BlockOperator`, which draws as one named box whose body is rendered once, and
`para_boxed` does the same for a block holding grabs and drops. `axes`,
`node_with_operator` and `node_with_box_named` read a morphism for the assertions the
notebook makes. A box computed once per index of a degree is built by
`algebra.discovering_broadcasts`, and the slots a morphism reads and writes are read by
`para.data_structure.ParaBlockOperator`.

The other modules of this package import these names directly rather than through a
module alias, and the same holds for the axes of `declared_axes`. An expression
written out of `construction_idioms.over` and `declared_axes.x` cannot be read against
the mathematics it states, and `over`, `hold` and `route` are the names
`obsidian/02-categories/Construction Helpers.md` already gives these three idioms.

`obsidian/06-practice/Representing Models.md` states the rules these follow.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import construction_helpers.lift as chl
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.processing.Hypergraph2Morphism as h2m
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as ParaBlockOperator
import term_utilities.term_utilities as tutil


def hold[B: cat.Datatype, A: cat.Axis](obj: cat.Array[B, A]) -> cat.Morphism:
    '''The identity on a single object, which is a wire passing straight through.'''
    return cat.ProdObject((obj,)).identity()


def route[L](mapping: tuple[int, ...], objs: tuple[L, ...]) -> cat.Rearrangement[L]:
    '''A rearrangement: codomain position j carries domain object mapping[j].'''
    return cat.Rearrangement(tuple(mapping), tuple(objs))


def over[B: cat.Datatype, A: cat.Axis](
    lift_axes: tuple[A, ...],
    morphism: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`morphism` tiled over `lift_axes`, which are prefixed onto every object.'''
    return chl.morphism_object_lift(morphism, cat.ProdObject(tuple(lift_axes)))


def l1_norm_over[B: cat.Datatype, A: cat.Axis](
    shape: tuple[A, ...],
    normalised_position: int,
    datatype: B = cat.Reals(),
    epsilon: nm.Numeric = ops.NO_EPSILON,
) -> cat.Broadcasted[B, A]:
    '''An `ops.L1Norm` of an array of `shape`, dividing by the sum over the axis
    at `normalised_position` plus `epsilon`, and broadcast over every other axis.

    The normalised axis is named by its position rather than by the axis itself,
    because a weave is a sequence of positions and the position says which
    of them the operation consumes. `over` cannot state this, since it
    prefixes its axes onto the degree and the Sinkhorn column normalisation
    consumes the first of two positions.

    `epsilon` is the number the released code adds to the sum before it divides. It
    is `ops.NO_EPSILON` for the exact model, which divides by the sum itself.
    '''
    weave = cat.Weave(datatype, tuple(
        axis if position == normalised_position else cat.WeaveMode.TILED
        for position, axis in enumerate(shape)))
    degree = cat.ProdObject(tuple(
        axis for position, axis in enumerate(shape)
        if position != normalised_position))
    return cat.Broadcasted(
        operator=ops.L1Norm(epsilon=epsilon),
        input_weaves=(weave,),
        output_weaves=(weave,),
        reindexings=(degree.identity(),))


def recycled_block[B: cat.Datatype, A: cat.Axis](
    block: cat.Block[B, A],
) -> cat.Block[B, A]:
    '''`block` with its wiring normalised and its own tag restored. Recycling lifts a
    wire that passes through a block out of that block, so the recycled body is given
    the block's tag again.'''
    recycled = h2m.recycle(block)
    body = recycled.body if isinstance(recycled, cat.Block) else recycled
    return cat.Block(body=body, block_tag=block.block_tag)


def boxed[B: cat.Datatype, A: cat.Axis](
    block: cat.Block[B, A],
    short_name: str,
) -> cat.Broadcasted[B, A, ops.BlockOperator[B, A]]:
    '''A block as a named operator, drawn as one compact box whose body is rendered
    once beside the main figure. The block's own title names the thing in words and
    the box carries `short_name`.'''
    return ops.BlockOperator.template(recycled_block(block), short_name)


def para_boxed[B: cat.Datatype, A: cat.Axis](
    block: cat.Block[B, A],
    short_name: str,
) -> ParaBlockOperator.WrappedBox[B, A]:
    '''A block holding grabs and drops as a named operator that records them, so that
    the slots a box reads and writes are readable without searching its body. Each grab
    stands at a leading operand of the box and each drop at a trailing result, and the
    `para.data_structure.ParaWrap.ParaWrap` the box is returned inside carries the slot
    of each of those ports, so the wrap has the domain and the codomain of `block`.'''
    return ParaBlockOperator.ParaBlockOperator.template(
        recycled_block(block), short_name)


def axis_name[A: cat.Axis](axis: A) -> str:
    '''The bodies of the axis's name, and for a concatenated axis the names of its
    parts joined, as `w|x + s|x`.'''
    if isinstance(axis, AxisConcatenation.ConcatenatedAxis):
        return AxisConcatenation.PART_SEPARATOR.join(
            axis_name(part) for part in axis.parts)
    return axis.uid._name.to_bodies() if axis.uid._name else '?'


def axes(morphism: cat.Morphism) -> tuple[list[list[str]], list[list[str]]]:
    '''Axis names of the domain and codomain, for checking shapes by eye.'''
    def names(objs) -> list[list[str]]:
        return [[axis_name(axis) for axis in obj.shape()] for obj in objs]
    return names(morphism.dom()), names(morphism.cod())


def node_with_operator[O: cat.Operator](
    kind: type[O],
    morphism: cat.Morphism,
) -> cat.Broadcasted:
    '''The one node of `morphism` whose operator is a `kind`, which raises where the
    morphism holds none or holds two.'''
    node, = (root for root in tutil.type_search(cat.Broadcasted, morphism)
             if isinstance(root.operator, kind))
    return node


def node_with_box_named[B: cat.Datatype, A: cat.Axis](
    short_name: str,
    morphism: cat.Morphism,
) -> cat.Broadcasted[B, A, ops.BlockOperator[B, A]]:
    '''The one node of `morphism` whose operator is an `ops.BlockOperator` carrying
    `short_name`, which raises where the morphism holds none or holds two. An expression
    holding a box inside a box is read with this where `node_with_operator` cannot say
    which of the two is wanted.'''
    node, = (root for root in tutil.type_search(cat.Broadcasted, morphism)
             if isinstance(root.operator, ops.BlockOperator)
             and root.operator.name is not None
             and root.operator.name.to_bodies() == short_name)
    return node


def named_slot(body: str, subscript: str | None = None) -> Para.TapeSlot:
    subscript = fd.DynamicName.from_str(subscript) if subscript else None
    return fd.DynamicName(body, subscript).capture(Para.TapeSlot())
