# Absorbing a node's reindexing into the morphism that reads it.
#
# `node_expansion` factors a `Broadcasted`'s reindexings *out*, into standalone
# `View` morphisms, because a node is a reindexing and nothing else. The pass
# here is the
# move back: a node whose result is read once is folded into its consumer's
# own reindexing for that operand.
#
# It is not simply the inverse of `expand_to_nodes`, and that is the point. A
# reverse pass writes nodes that the forward pass never had - `para` transposes
# each node on its own, then puts the transposed core between them, so after
# `backprop` the nodes are new, and the consumer they belong to is a
# different morphism from the one they were factored out of. Absorbing them is
# what turns
#
#     %5 = View(%3[q, d])                : R[q, v, d]     a broadcast along v
#     %8 = Einops(%0[q, v], %5[q, v, d])     : R[q, v, d]
#
# into the one line that states the same thing, with the broadcast written where
# the algebra already had somewhere to put it, which is the operand's
# reindexing:
#
#     %8 = Einops(%0[q, v], %3[q, d])        : R[q, v, d]
#
# ## A repeat is a node
#
# A **node** is a `View` carrying a reindexing, and a reindexing that does
# not name one of the degree axes is a broadcast along it, so a repeat, the
# dual of a sum over an axis, is a node too: `View` at degree `(q, x)`
# whose one reindexing reads `(q,)`. `einops_simplification.einsum` writes it
# that way, and it is absorbed here exactly as a broadcast node is, the reader
# dropping the axis from its own reindexing. There is no separate repeat
# operator and no separate rule for one.
#
# ## A node is copied over a fan-out
#
# A node read by several morphisms is folded into each of them. The copy of
# the node's output commutes with the node, because a reindexing is natural in
# the array it reads: copying then reindexing each copy reads the same
# elements as reindexing then copying. Each reader then absorbs its own copy,
# and the node is removed once nothing reads it. A repeat is therefore written
# at the last point it is needed, on each branch separately, which is where a
# kernel would broadcast it. `merge_into_consumer.merge_producers_into_every_consumer`
# is the search that admits a shared producer, and it is offered this rule
# alone, because a copied node costs nothing and a copied contraction does not.
#
# ## What a merge costs
#
# Nothing, in indices: the composite index map is `rho . sigma`, and both are
# affine, so the result is affine. What has to be checked is that it stays
# expressible. A `Broadcasted` states that these positions are degree and those
# are
# the operator's" per operand, and the producer is allowed to have shuffled
# that distinction. Hence the refusals in `_absorb`.
from __future__ import annotations
import data_structure.Category as cat
import data_structure.Operators as ops
import term_utilities.term_utilities as tutil

import algebra.merge_into_consumer as merge_into_consumer


def is_node(target: cat.Morphism) -> bool:
    '''A `Broadcasted` that is a reindexing and nothing else.

    Both weaves are fully tiled, so the operator receives single elements and
    every axis is degree, with one operand in and one out. `expand_to_nodes`
    writes
    exactly this, and `Derivatives`' node transpose reverses it into one.
    '''
    return (isinstance(target, cat.Broadcasted)
            and isinstance(target.operator, ops.View)
            and len(target.input_weaves) == 1
            and len(target.output_weaves) == 1
            and _all_tiled(target.input_weaves[0])
            and _all_tiled(target.output_weaves[0]))


def _all_tiled(weave: cat.Weave) -> bool:
    return all(isinstance(entry, cat.WeaveMode) for entry in weave._shape)


def _mapping(target: cat.Morphism) -> tuple[int, ...] | None:
    '''The reindexing as "codomain position i reads domain position m[i]".

    `None` for a genuinely strided map, since a convolution window sends several
    domain axes to one codomain axis, so there is no such tuple, and the merge
    below would have nothing to permute the weave by.
    '''
    try:
        return tuple(tutil.get_mapping(target))
    except (ValueError, KeyError):
        return None


def _tiled_positions(weave: cat.Weave) -> tuple[int, ...]:
    return tuple(position for position, entry in enumerate(weave._shape)
                 if isinstance(entry, cat.WeaveMode))


def _origin(producer: cat.Broadcasted) -> tuple[int, ...] | None:
    """
    Where each axis of the producer's input sits in the array it writes.

    The producer's output weave has a TILED slot per degree axis, and its
    reindexing states which degree axis each axis of the input is. Composing the
    two gives, for input axis `k`, the position in the *output array* holding
    it, which is the position the reader's weave is indexed by, since the
    reader reads exactly that array. Any output position not named here is one
    the producer invented: a degree axis it broadcast along, or an axis it
    produced by repeating.
    """
    if len(producer.input_weaves) != 1 or len(producer.output_weaves) != 1:
        return None
    if not _all_tiled(producer.input_weaves[0]):
        return None
    rho = _mapping(producer.reindexings[0])
    if rho is None:
        return None
    tiled = _tiled_positions(producer.output_weaves[0])
    if any(degree_index >= len(tiled) for degree_index in rho):
        return None
    return tuple(tiled[degree_index] for degree_index in rho)


def _absorb(
    producer: cat.Broadcasted,
    consumer: cat.Broadcasted,
    port: int,
) -> cat.Broadcasted | None:
    """
    Rewrite the consumer to read the producer's input directly, with the
    producer's reindexing folded into its own for that operand.

    The whole merge is one substitution. `_origin` states which position of the
    reader's weave each axis of the input arrives at, so selecting the weave's
    entries through it is the new weave, and it names the same axes either
    way: an entry is either an axis the operator consumes, carried over
    untouched, or a TILED slot whose axis both sides already agree on. The new
    reindexing follows the same route, degree position by degree position.
    Positions `_origin` does not name are dropped, which is precisely the
    broadcasting the producer was doing, now written where the algebra already
    had somewhere to put it.

    Three refusals, all of them a case where the substitution is not a
    substitution:

    - the producer's own weaves draw the degree/operator line somewhere the
      mapping cannot move (`_origin` returns `None`);
    - either reindexing is strided rather than a rearrangement, so there is no
      position-to-position mapping to compose;
    - an axis the consumer's *operator* reads is not, or is not uniquely, an
      axis of the producer's input. The producer broadcast it into existence,
      or read the same axis twice. Absorbing would silently change what the
      operator receives. A sum over an axis the producer broadcast along is the
      readable case, and it is a multiplication by that axis' size, which is
      arithmetic and not a rewrite.
    """
    if port >= len(consumer.input_weaves):
        return None
    weave = consumer.input_weaves[port]
    if weave.datatype != producer.input_weaves[0].datatype:
        return None
    origin = _origin(producer)
    sigma = _mapping(consumer.reindexings[port])
    if origin is None or sigma is None:
        return None
    if any(position >= len(weave._shape) for position in origin):
        return None

    # Position in the consumer's weave -> its index among the TILED slots,
    # which is what the consumer's reindexing is indexed by.
    tiled_rank = {position: rank for rank, position
                  in enumerate(_tiled_positions(weave))}
    for position, entry in enumerate(weave._shape):
        if position not in tiled_rank and origin.count(position) != 1:
            return None

    new_weave = cat.Weave(weave.datatype,
                          tuple(weave._shape[position] for position in origin))
    new_reindexing = cat.Rearrangement(
        tuple(sigma[tiled_rank[position]]
              for position in origin if position in tiled_rank),
        tuple(consumer.degree()))
    return consumer.reconstruct(
        input_weaves=(*consumer.input_weaves[:port], new_weave,
                      *consumer.input_weaves[port + 1:]),
        reindexings=(*consumer.reindexings[:port], new_reindexing,
                     *consumer.reindexings[port + 1:]))


def absorb(
    node: cat.Broadcasted,
    consumer: cat.Broadcasted,
    port: int,
) -> cat.Broadcasted | None:
    """A node, meaning a reindexing and nothing else, folded into its reader."""
    if not is_node(node):
        return None
    return _absorb(node, consumer, port)


def absorb_nodes[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    '''Every node folded into each reader that takes it, copied over a fan-out.'''
    return merge_into_consumer.merge_producers_into_every_consumer(target, absorb)
