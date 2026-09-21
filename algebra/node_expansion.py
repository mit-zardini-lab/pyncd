# Factoring the reindexings out of a Broadcasted.
#
# A `Broadcasted` carries three things at once: an operator, the weaves that
# say which positions it is broadcast over, and a reindexing per input saying
# where in the degree each input is read from. `expand_to_nodes` separates the
# third from the first two. It returns a product of `View` morphisms, one
# per input, each carrying that input's reindexing and nothing else, plus a
# `new_core` that is the same operator with its reindexings replaced by the
# degree identity.
#
# Two consumers need it, for different reasons:
#
#   - `algebra.linear_expansion` needs the core: once a Linear's
#     reindexings are trivial, the weight can be split off as a 0-input array
#     and contracted back in through a real reindexing that the dependency
#     graph carries.
#   - `para` needs the nodes: a reindexing is an affine index map, so as a
#     linear map it is a 0/1 matrix, and it is the one part of a Broadcasted
#     whose transpose is completely determined. Factored out, each reindexing
#     can be reversed on its own and the core's derivative is taken on a
#     cleanly broadcast operator, meaning one whose inputs are already in the degree.
#
# It lives here rather than in either of them because it is a change of
# presentation rather than a step in any derivation.
from __future__ import annotations
import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Operators as ops

from construction_helpers import simple_helper as chsh


def expand_to_nodes[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A]
) -> tuple[cat.BroadcastedCategory[B, A], cat.Broadcasted[B, A]]:
    degree = target.degree()
    input_weaves = tuple(
        cat.Weave(weave.datatype, (cat.WeaveMode.TILED,) * len(degree) + tuple(weave.target().shape()))
        for weave in target.input_weaves
    )
    nodes_reindexings: fd.Prod[cat.StrideCategory[A]] = tuple(
        chsh.make_composed(
            chsh.make_product(reindexing, weave.target().shape().identity()),
            weave.rearrangement(reindexing.cod())
        )
        for weave, reindexing in zip(target.input_weaves, target.reindexings)
    )
    nodes = tuple(
        cat.Broadcasted(
            ops.View(),
            input_weaves=(cat.Weave(weave.datatype, (cat.WeaveMode.TILED,) * len(weave._shape)),),
            # Every slot tiled, target axes included: they are in this node's
            # own degree, where `nodes_reindexings` put them, and a weave
            # whose tiled count disagrees with `degree()` misaligns anything
            # that pairs the two up positionally. The object is unchanged,
            # since `imprint_to_degree` fills those slots with the same axes.
            output_weaves=(cat.Weave(weave.datatype, (cat.WeaveMode.TILED,) * len(reindexing.dom())),),
            reindexings=(reindexing,)
        )
        for weave, reindexing in zip(target.input_weaves, nodes_reindexings)
    )
    nodes_identity_reduction: cat.BroadcastedCategory[B, A] = chsh.make_product(
        *(node if not ops.is_identity(node) else cat.Rearrangement((0,), tuple(node.dom()))
        for node in nodes)
    )
    new_core = cat.Broadcasted(
        target.operator,
        input_weaves=input_weaves,
        output_weaves=target.output_weaves,
        reindexings=(target.degree().identity(),) * len(target.input_weaves),
        backup_degree=target.backup_degree,
    )
    return nodes_identity_reduction, new_core
