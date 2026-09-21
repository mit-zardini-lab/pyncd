# Claude Opus 5 (1M context), high effort.
'''Choosing whether a box's body is normalised before the figure is drawn.

A morphism is written in the order a person composes it, and the wiring that
carries a value to two readers is written as a copy and a permutation.
`graphs.processing.Hypergraph2Morphism.recycle` converts the morphism to a
hypergraph and back, which replaces that wiring with the wires themselves and
returns the composition in the order the dependencies force. The diagram
transport already recycles what it is handed, through `recycle` on
`websocket_transfer.send_morphism.to_morphism`.

That recycling reaches one level. An `ops.BlockOperator` carries a whole
expression in its `block`, and the transport draws the body as a sub-diagram
without recycling it, so a body written by hand is drawn as it was written.
`RECYCLED` passes the term through
`algebra.broadcasted_recycle.recycle_broadcasted` before anything else is done
to it, which recycles the body of every box and the body of every box inside
those, so the sub-diagram of the attention core is drawn from the same normal
form as the figure that holds it.

`AS_WRITTEN` is the default and hands the term to the transport untouched.

Recycling keeps each block's tag, so the record
`notebooks/display/remember_drawn_blocks.py` keeps of the bodies already
delivered still names the same bodies.
'''
from __future__ import annotations

import enum

import algebra.broadcasted_recycle as broadcasted_recycle
import data_structure.Term as fd


class BlockRecycling(enum.Enum):
    AS_WRITTEN = 'as_written'
    RECYCLED = 'recycled'


def present[T: fd.GeneralTerm](term: T, recycling: BlockRecycling) -> T:
    '''`term` with its boxes normalised where `recycling` asks. `AS_WRITTEN`
    returns it as it stands.'''
    if recycling is BlockRecycling.AS_WRITTEN:
        return term
    return broadcasted_recycle.recycle_broadcasted(term)
