'''Choosing how a term's grabs and drops are presented.

A `Para` holds its `Grab` and `Drop` seeds as morphisms of their own, which is
the form the algebra works on. `para_wrap.to_para_wrap` re-expresses the same
term with each grab absorbed onto the operand port it feeds and each drop onto
the result it saves, per `obsidian/07-para/Para Wrap.md`. The two say the same
thing, so the choice between them belongs to the display, and a notebook makes
it once in its `DiagramSettings`. The absorbed form is the default of a diagram
since 2026-09-12, and a term holding no grab and no drop is returned as it
stands under either presentation, so a hand-built morphism drawn with
`recycle=False` is not recycled by the choice.

`to_para_wrap` rewrites siblings in one scope and never enters an operator, and
tsncd draws the block of every `ops.BlockOperator` as a sub-diagram beside the
main figure. `wrap_inside_boxes` therefore applies the same rule to the block of
each box, so a grab standing inside a boxed layer is drawn on the operand it
feeds there as one standing beside the box is. The block keeps its tag, so
`remember_drawn_blocks` recognises a body it has already delivered. A
`ParaBlockOperator` carries its tape at the box's ports as well, per
`obsidian/07-para/Para Block Operator.md`, and that is part of the expression
rather than a presentation of it, so it is left alone here.

A grab standing beside a plain box and read by that box alone is absorbed onto
the box's port, and the block inside the box still reads the operand on a wire,
so the body drawn for the box shows no tape. The reviewer found that on
2026-09-17 in the Lightning Indexer and the Entry Gather of the Reindex layer of
DeepSeek-V4.1-Flash. `wrap_inside_boxes` therefore rebuilds every plain box that
a wrap tapes as the `ParaBlockOperator` whose block holds those tapes as seeds,
through `ParaBlockOperator.box_with_wrapped_tapes_as_seeds`, and wraps the body of
that block in turn. The box keeps the tape at its port and its body shows the same
tape on the operation that reads it, as a box the model wrote as a
`ParaBlockOperator` does.
'''
from __future__ import annotations
import enum

import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import para.data_structure.MultiCategory as multi_category
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as para_block_operator
import para.data_structure.ParaWrap as para_wrap
import term_utilities.term_utilities as tutil


class TapePresentation(enum.Enum):
    BOXED = 'boxed'
    ABSORBED = 'absorbed'


def holds_tape_operations(term: object) -> bool:
    '''Whether `term` holds a grab, a drop or a wrap of either.'''
    return any(True for _ in tutil.type_search(
        (Para.ParaMorphism, para_wrap.ParaWrap), term))


def holds_bare_seeds(term: object) -> bool:
    '''Whether `term` holds a `Grab` or a `Drop` standing as a morphism of its
    own, which is what `to_para_wrap` has something to rewrite.'''
    return any(True for _ in tutil.type_search(Para.ParaMorphism, term))


def tapes_a_plain_box(node: object) -> bool:
    '''Whether `node` is a wrap that tapes a port of a boxed block whose block holds
    no seed for that tape. A `ParaBlockOperator` holds the seeds already, and a block
    drawn in place of its operator has no body to draw them in.'''
    if not isinstance(node, para_wrap.ParaWrap):
        return False
    box = node.body
    if not (isinstance(box, cat.Broadcasted)
            and isinstance(box.operator, ops.BlockOperator)):
        return False
    if isinstance(box.operator, para_block_operator.ParaBlockOperator):
        return False
    aesthetics = box.operator.block.aesthetics
    if aesthetics is not None and aesthetics.drawing is cat.BlockDrawing.BODY_IN_PLACE:
        return False
    return any(entry is not None for entry in (*node.grabs, *node.drops))


def wrap_inside_boxes[T: fd.GeneralTerm](term: T) -> T:
    '''`term` with the block of every boxed operator wrapped as its siblings
    are, and with every plain box a wrap tapes rebuilt as the `ParaBlockOperator`
    whose block holds those tapes as seeds.

    Each node is rewritten once, by identity, so the sharing of the term is kept
    and a box standing in a repeated block has its body wrapped once. The
    innermost box is reached first. Wrapping a body can absorb a grab onto a
    plain box standing in it, so the wrapped body is walked again, and the block
    of a rebuilt box holds bare seeds, so the rebuilt box is walked again. Each
    of the two moves a tape one box deeper, so the walk ends at the depth of the
    innermost box. A node is held beside its rewriting, because a body wrapped
    during the walk is reachable from nothing else, and the id of a collected node
    is given out again.
    '''
    rewritten: dict[int, tuple[object, object]] = {}

    def write(node: object) -> object:
        if id(node) in rewritten:
            return rewritten[id(node)][1]
        rebuilt = fd.deep_reconstruct(node, write)
        if (isinstance(rebuilt, ops.BlockOperator)
                and holds_bare_seeds(rebuilt.block)):
            rebuilt = rebuilt.reconstruct(block=rebuilt.block.reconstruct(
                body=write(para_wrap.to_para_wrap(rebuilt.block.body))))
        elif tapes_a_plain_box(rebuilt):
            rebuilt = write(
                para_block_operator.box_with_wrapped_tapes_as_seeds(rebuilt))
        rewritten[id(node)] = (node, rebuilt)
        return rebuilt

    return write(term)  # type: ignore[return-value]


def present[L, M: cat.Morphism](
    term: cat.ProdCategory[L, M] | hg.Hypergraph[L, M]
    | multi_category.MultiCategory[L, M],
    presentation: TapePresentation,
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M] | multi_category.MultiCategory[L, M]:
    '''`term` in the presentation asked for. `BOXED` returns it as it stands,
    and so does `ABSORBED` when `term` holds no tape operation.'''
    if presentation is TapePresentation.BOXED or not holds_tape_operations(term):
        return term
    return wrap_inside_boxes(_wrap_outside_boxes(term))


def _wrap_outside_boxes[L, M: cat.Morphism](
    term: cat.ProdCategory[L, M] | hg.Hypergraph[L, M]
    | multi_category.MultiCategory[L, M],
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M] | multi_category.MultiCategory[L, M]:
    '''`to_para_wrap` applied to every scope of `term` that stands outside an
    operator, which is each row of a `MultiCategory` and the morphism a
    hypergraph converts to.'''
    match term:
        case multi_category.MultiCategory():
            return para_wrap.to_para_wrap_rows(term)
        case hg.Hypergraph():
            return para_wrap.to_para_wrap(h2m.hypergraph_to_morphism(term))
        case _:
            return para_wrap.to_para_wrap(term)
