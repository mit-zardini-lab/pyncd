# Claude Opus 5 (1M context), high effort.
'''Recycling an expression at every level, including inside a box.

`graphs.processing.Hypergraph2Morphism.recycle` converts a morphism to a
hypergraph and back. The conversion normalises the wiring: a rearrangement that
permutes and copies becomes the wires it describes, a value read twice is one
wire with two consumers, and the composition comes back in the order the
dependencies force rather than the order it was written in. `recycle` is defined
on any product category, so it descends through a `cat.Block`, which every
product category has, and treats a seed morphism as one node.

An `ops.BlockOperator` is a seed morphism of **Br** that carries a whole
sub-expression in its `block`. The generic recycling therefore normalises the
level it was called at and leaves every box exactly as it was written.
`BroadcastedRecycle` is the recycling of **Br**, which knows that a block
operator holds an expression and recycles that expression too.

The V4.1-Flash package recycles a body at the moment it boxes one, in
`notebooks.sota.DeepSeekV41Flash.construction_idioms.recycled_block`, and a box
built by `algebra.discovering_broadcasts.broadcast_block_over_axes` skips that
step, which is why the attention core drew as it was written.

The descent is a walk over the term rather than over the category, for three
reasons. A `para.data_structure.ParaWrap.ParaWrap` is a morphism that
`graphs.processing.hypergraph_functor` has no case for, so a category functor
reads one as a leaf and never sees the boxes inside it, and the attention modes
of DeepSeek-V4.1-Flash put the attention core inside exactly that. A term walk
rebuilds from the leaves up, so a box's body already holds its own recycled
boxes by the time the box is reached, and each body is recycled once. `algebra/`
may not import `para/`, which a walk over the category would have to.

The body is recycled and the block is rebuilt around it, rather than the block
being recycled whole. Recycling a `cat.Block` lifts a wire that passes straight
through it out of the block, which changes the block's domain and codomain, and a
block operator's weaves are built from those. Recycling the body alone leaves
them where they were.
'''
from __future__ import annotations
from dataclasses import dataclass, field

import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m


@dataclass
class BroadcastedRecycle:
    '''The recycling of **Br**: the wiring normalised at every level of an
    expression, down to the body of the innermost box.

    `recycled_nodes` records what each node of the term was rewritten into,
    against the identity of the node it was rewritten from, so a subterm reached
    by many paths is visited once and the sharing of the term survives the
    rewrite.
    '''
    recycled_nodes: dict[int, object] = field(default_factory=dict)

    def __call__[T: fd.GeneralTerm](self, target: T | hg.Hypergraph) -> T:
        '''`target` recycled, and the body of every `ops.BlockOperator` in it
        recycled in turn. A hypergraph is converted first, as the diagram
        transport converts it.'''
        morphism = (h2m.hypergraph_to_morphism(target)
                    if isinstance(target, hg.Hypergraph) else target)
        return h2m.recycle(self.recycle_inside_every_block(morphism))

    def recycle_inside_every_block[T](self, target: T) -> T:
        '''`target` with the body of every `ops.BlockOperator` in it recycled,
        the innermost first, and `target` itself where nothing changed.'''
        if id(target) in self.recycled_nodes:
            return self.recycled_nodes[id(target)]
        rebuilt = fd.deep_reconstruct(target, self.recycle_inside_every_block)
        if isinstance(rebuilt, ops.BlockOperator):
            rebuilt = self.recycled_block_operator(rebuilt)
        self.recycled_nodes[id(target)] = rebuilt
        return rebuilt

    def recycled_block_operator[O: ops.BlockOperator](self, operator: O) -> O:
        '''`operator` holding a recycled body, and `operator` itself where the
        recycling changed nothing.

        A `para.data_structure.ParaBlockOperator.ParaBlockOperator` carries one
        grab per leading operand of its body and one drop per trailing result,
        aligned with those ports by position. Recycling a body preserves its
        domain and codomain, so the alignment holds.
        '''
        block = operator.block
        body = h2m.recycle(block.body)
        if body == block.body:
            return operator
        return operator.reconstruct(block=block.reconstruct(body=body))


def recycle_broadcasted[T: fd.GeneralTerm](target: T | hg.Hypergraph) -> T:
    '''`target` recycled at every level, through the body of every box.'''
    return BroadcastedRecycle()(target)
