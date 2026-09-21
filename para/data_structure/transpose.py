'''The two transposes a reverse pass needs for a morphism that is already linear.

A linear map needs no residual, because its reverse derivative is its transpose
and the transpose is built from the same data read the other way. Nothing is
taped, nothing is recomputed, and the forward pass is left as it was written.

`Transpose` is the transpose of a linear operator, `Linear` above all.
`ReindexTranspose` is the transpose of an affine reindexing, which sums over the
fibre and which the broadcasted category has no other way to write.

`obsidian/07-para/Derivatives.md` gives the mathematics, the reason a reindexing
transposes into an operator rather than into another reindexing, and the reason
the transpose does not factor out the context it is broadcast over.
'''
from __future__ import annotations
from dataclasses import dataclass

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Operators as ops
import term_utilities.term_utilities as tutil


@dataclass(frozen=True)
class Transpose[O: cat.Operator](cat.Operator):
    '''The transpose of a linear operator, holding the operator it transposes.

    `operator` is kept whole rather than reduced to a name. `transpose` unwraps
    it rather than nesting, which makes the transpose an involution, and a later
    pass can read that a `Linear` and its transpose are the same weight.
    `linear_expansion.ExpandLinear` does not read the field. It splits a forward
    `Linear` into a weight array and a contraction, so applied to a transpose it
    would produce a second weight unrelated to the first.

    `name` is the transposed operator's own name, carried unchanged, so the
    listing reads `Transpose<L>` and the glyph is drawn mirrored rather than
    under a second label. A transpose does not rename a matrix.
    '''
    operator: O


@dataclass(frozen=True)
class ReindexTranspose[A: cat.Axis](cat.Operator):
    '''The transpose of a node, summing over the fibre of its reindexing.

    The reindexing is an argument of the operator rather than a `reindexings`
    entry, because a `reindexings` entry always points from the output back to
    the input and a transpose runs the other way.
    `obsidian/07-para/Derivatives.md` gives the argument in full.

    A `ReindexTranspose` is not the `Scatter` that
    `obsidian/07-para/Selection and the Reverse Pass.md` records as missing.
    `Scatter` is the transpose of an index map that depends on data, meaning
    `IndexSelect`, whose indices arrive on a wire rather than in the term. A
    reindexing is affine, so `reindexing` holds the whole of it.
    '''
    reindexing: cat.StrideCategory[A]


def is_cleanly_broadcast[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> bool:
    '''Every reindexing is the degree identity, which is what `expand_to_nodes`
    leaves behind.
    '''
    return all(tutil.is_identity(eta) for eta in target.reindexings)


class OperatorIsNotCleanlyBroadcast(Exception):
    '''An operator carrying a live reindexing handed to `transpose`.'''


def transpose[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.Broadcasted[B, A]:
    '''A cleanly broadcast linear operator, with its weaves swapped.

    Every reindexing is the degree identity, so `dom` is `input_weaves`
    imprinted with the degree and `cod` is `output_weaves` imprinted with the
    same degree. Exchanging the two weaves therefore exchanges the domain and
    the codomain exactly. The tilings need no handling, because a weave keeps
    whatever TILED positions it had, on the other side.

    `backprop` establishes the precondition by running `expand_to_nodes` first.
    A `Linear` with a live reindexing would transpose into a morphism whose
    domain and codomain are wrong without saying so, so `transpose` raises
    rather than transposing it.
    '''
    if not is_cleanly_broadcast(target):
        raise OperatorIsNotCleanlyBroadcast(
            f'{target.operator} carries a live reindexing. Run '
            'expand_to_nodes before transposing it.')
    degree = target.degree()
    operator = target.operator
    inner = (operator.operator if isinstance(operator, Transpose)
             else Transpose(name=operator.name, operator=operator))
    return cat.Broadcasted(
        operator=inner,
        input_weaves=target.output_weaves,
        output_weaves=target.input_weaves,
        reindexings=(degree.identity(),) * len(target.output_weaves),
    )


def _reindexing_name[A: cat.Axis](
    reindexing: cat.StrideCategory[A],
) -> fd.DynamicName | None:
    '''The name of the one `StrideMorphism` in a reindexing, where exactly one
    of them is named.

    A convolution window is written
    `StrideMorphism.from_matrix((1, 1), ..., name='+')`. Carrying that name onto
    the transpose makes the listing read `ReindexTranspose<+>` rather than
    leaving the reader to open the term. Where several are named, none of them
    describes the composite, and the transpose is left unnamed.
    '''
    names = [found.name
             for found in tutil.type_search(cat.StrideMorphism, reindexing)
             if found.name is not None]
    return names[0] if len(names) == 1 else None


class MorphismIsNotANode(Exception):
    '''A morphism with other than one reindexing handed to a node transpose.'''


def transpose_reindexing[B: cat.Datatype, A: cat.Axis](
    node: cat.Broadcasted[B, A],
) -> cat.Broadcasted[B, A]:
    '''The transpose of a node, mapping `[X, P] -> [X, Q]` where the node mapped
    `[X, Q] -> [X, P]`.

    The shapes are read off the node's own domain and codomain rather than
    rebuilt from the reindexing, so the two cannot disagree. The degree is empty
    and both weaves are entirely target, which
    `obsidian/07-para/Derivatives.md` records as a deliberate limit.
    '''
    if len(node.reindexings) != 1:
        raise MorphismIsNotANode(
            f'{node.operator} carries {len(node.reindexings)} reindexings, and '
            'a node carries one.')
    source = tuple(node.dom())[0]
    result = tuple(node.cod())[0]
    reindexing = node.reindexings[0]
    return cat.Broadcasted(
        operator=ReindexTranspose(name=_reindexing_name(reindexing),
                                  reindexing=reindexing),
        input_weaves=(cat.Weave(result.datatype, tuple(result.shape())),),
        output_weaves=(cat.Weave(source.datatype, tuple(source.shape())),),
        reindexings=(cat.ProdObject().identity(),),
    )


class MorphismIsNotAReindexTranspose(Exception):
    '''A morphism whose operator is not a `ReindexTranspose`, handed to
    `untranspose_reindexing`.'''


def untranspose_reindexing[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.Broadcasted[B, A]:
    '''The node a `ReindexTranspose` is the transpose of, so that a reverse pass
    can itself be reversed.

    A repeat, which is a node whose reindexing drops an axis, reverses back into
    the sum it is dual to in the same way.
    '''
    operator = target.operator
    if not isinstance(operator, ReindexTranspose):
        raise MorphismIsNotAReindexTranspose(
            f'{operator} is not a transposed reindexing.')
    source = tuple(target.dom())[0]
    result = tuple(target.cod())[0]
    return cat.Broadcasted(
        operator=ops.View(),
        input_weaves=(cat.Weave(result.datatype,
                                (cat.WeaveMode.TILED,) * len(result.shape())),),
        output_weaves=(cat.Weave(source.datatype,
                                 (cat.WeaveMode.TILED,) * len(source.shape())),),
        reindexings=(operator.reindexing,),
    )
