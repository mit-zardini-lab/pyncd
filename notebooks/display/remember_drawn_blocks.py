# Claude Opus 5 (1M context), high effort.
'''Which `BlockOperator` bodies tsncd has already drawn beside a figure.

tsncd draws the body of every `BlockOperator` it builds a box for as a
sub-diagram beside the main figure, and it holds no record of what an earlier
message drew: the render target is wiped at the start of every message, and a
capture draws into a second target that never saw the first. The record
therefore belongs to the process that has been sending, which is the notebook
kernel, and this module is where it is kept.

`SubBlocks` is the setting a figure carries. `EVERY_BODY` draws the body of
every `BlockOperator` the figure holds, whatever an earlier figure drew.
`BODIES_NOT_YET_DRAWN`, the default, leaves out a body the record already
names. `NO_BODIES` draws none of them, which gives the high-level view alone.

`DRAWN_BLOCK_TAGS` is the record, a set of `BlockTag` uid ids that lives for as
long as the kernel does. `notebook_diagrams.show_diagram` sends it as the
`drawnBlockTags` display setting and calls `remember(term)` once a figure has
been delivered. A body whose tag is in the set is left out of the next figure
asked for with `BODIES_NOT_YET_DRAWN`, and that figure still holds the box the
body belongs to.

`forget_drawn_blocks()` empties the record, so a cell that wants every body
drawn again calls it. Restarting the kernel empties it as well.

The record outlives a notebook, because the kernel does. A notebook run a
second time in the kernel that ran it before starts with the record its first
run filled, so each of its figures draws the boxes and none of the bodies. A
notebook whose figures must come out the same on every run asks for
`SubBlocks.EVERY_BODY`, or empties the record in its setup cell.

The set the sender must hold is not every tag in the term. A body tsncd leaves
out is never built, so the `BlockOperator`s inside it are never made pending and
their bodies are not drawn either. `newly_drawn_tags` walks the term the way the
renderer does: it takes the `BlockOperator`s that stand in the term outside any
block, marks each one's tag, and descends into the body of the ones it marked.
'''

from __future__ import annotations

import enum

import data_structure.Operators as ops
import data_structure.Term as fd

DRAWN_BLOCK_TAGS: set[fd.IDType] = set()


class SubBlocks(enum.Enum):
    EVERY_BODY = 'every_body'
    BODIES_NOT_YET_DRAWN = 'bodies_not_yet_drawn'
    NO_BODIES = 'no_bodies'


def draws_any_body(sub_blocks: SubBlocks) -> bool:
    '''The `subBlocks` display setting tsncd reads.'''
    return sub_blocks is not SubBlocks.NO_BODIES


def forget_drawn_blocks() -> None:
    '''Empty the record, so that every body is drawn again.'''
    DRAWN_BLOCK_TAGS.clear()


def block_operators_outside_blocks(
    target: fd.GeneralTerm,
) -> tuple[ops.BlockOperator, ...]:
    '''The `BlockOperator`s of `target` that stand outside the block of another.

    These are the ones tsncd builds a box for while it draws `target`, and so
    the ones whose bodies it makes pending. A `BlockOperator` nested in another
    one's body is reached by drawing that body and not by drawing `target`.
    '''
    found: list[ops.BlockOperator] = []
    visited: set[int] = set()

    def descend(value) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, ops.BlockOperator):
            found.append(value)
            return
        if isinstance(value, tuple):
            for member in value:
                descend(member)
            return
        if isinstance(value, fd.Term):
            for field in value.dict().values():
                descend(field)

    descend(target)
    return tuple(found)


def newly_drawn_tags(
    target: fd.GeneralTerm,
    already_drawn: set[fd.IDType],
) -> set[fd.IDType]:
    '''The tags whose bodies tsncd draws beside `target`, given `already_drawn`.

    A tag in `already_drawn` is skipped and its body is not descended into,
    which is what the renderer does with a body it leaves out.
    '''
    drawn: set[fd.IDType] = set()
    pending = list(block_operators_outside_blocks(target))
    while pending:
        operator = pending.pop()
        tag = operator.block.block_tag.uid._id
        if tag in already_drawn or tag in drawn:
            continue
        drawn.add(tag)
        pending.extend(block_operators_outside_blocks(operator.block.body))
    return drawn


def tags_to_skip(sub_blocks: SubBlocks) -> list[fd.IDType]:
    '''The `drawnBlockTags` display setting tsncd reads.

    `SubBlocks.EVERY_BODY` names no tag, so tsncd draws the body of every
    `BlockOperator` it builds a box for however many figures were sent before.
    '''
    if sub_blocks is SubBlocks.EVERY_BODY:
        return []
    return sorted(DRAWN_BLOCK_TAGS)


def remember(target: fd.GeneralTerm) -> None:
    '''Record the bodies tsncd drew beside `target`.'''
    DRAWN_BLOCK_TAGS.update(newly_drawn_tags(target, DRAWN_BLOCK_TAGS))
