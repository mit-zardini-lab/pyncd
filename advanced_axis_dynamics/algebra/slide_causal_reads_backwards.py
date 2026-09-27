# Claude Opus 5.5 (1M context), effort 40.
'''The CausalSlide: every causal read of an expression moved backwards as far as it goes.

A causal read is a view whose reindexing reads an axis at a position of that axis
minus an offset that is never negative. The causal mask of a decoder is the case:
`notebooks/classic/shared_mechanisms.read_back_from_every_position` reads token
`i_x - i_w` for every slot `i_w`. An operator broadcast over the axis the read reads
computes the same function at every position of that axis, so the read can stand
before the operator without changing the function the expression computes. That rule
is the Yoneda trick of `obsidian/06-practice/Yoneda and Cartesian Tricks.md`.

`slide_causal_reads_backwards` takes each causal read as the pending read of its
operand and carries it back with `move_reads_backwards.ReadCrawler`. The operators it
passes are rebuilt over the axes the read returns, so a key projection that followed
the mask is written over the tokens and the slots. The read stops at the copy whose
other branches read the operand unmasked, where the crawl writes it as one view. The
key and value branches of an attention ask for the same read, so they share that
view. The user named the result the CausalSlide on 2026-09-26 and ruled it the
standard form in which an expression is displayed, because the mask then stands next to
the copy it masks and every operator after it reads the positions the mask allows.

The slide enters the body of every box and every `ParaWrap`. A causal read inside a
body slides to a copy inside it, or to the domain of the body. A read at the domain
of a box's body leaves the box as the pending read of the operand, composed with the
reindexing of that operand, and slides on outside, so the box then reads the array the
read returns. A read at a domain position of a wrap that the wrap grabs from the tape
stops there, because a slot holds its array whole.

The position of the causal read on its path is the placement of a cache. The operators
before the read compute once for every token, and a cache of the array the read reads
serves every later pass. The operators after the read compute once for every token and
slot. `caching.algebra.derive_cached_pass` places the cache on the operand of the
causal read, so on the CausalSlide it caches the array at the copy, and every other
placement slides the read forward past the operators computed once per token.
`obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` states the
correspondence.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import construction_helpers.simple_helper as chsh
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as ParaBlockOperator
import para.data_structure.ParaWrap as para_wrap

import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards


def grabbed_operands(operator: ops.BlockOperator) -> fd.Prod[Para.ParaMorphism]:
    '''The grabs a box holding tape seeds carries as its leading operands, which its
    block holds as seeds rather than as domain arrays, and none for any other box.'''
    if isinstance(operator, ParaBlockOperator.ParaBlockOperator):
        return tuple(operator.grabs)
    return ()


def is_causal_row(row_axis: sc.Axis, strides: tuple[nm.Numeric, ...], shift: nm.Numeric,
                  domain: tuple[sc.Axis, ...]) -> bool:
    '''Whether a row reads its own axis at unit stride and at least one other axis at a
    negative stride, with no other axis read at a positive stride and a shift of at
    most zero, so that it reads a position no later than the one it computes.'''
    read = move_reads_backwards.positions_read(strides)
    own = tuple(position for position in read if domain[position] == row_axis)
    others = tuple(position for position in read if position not in own)
    return (len(own) == 1 and strides[own[0]] == nm.Integer(1) and bool(others)
            and all(isinstance(strides[position], nm.Integer)
                    and strides[position]._value <= 0 for position in others)
            and any(strides[position]._value < 0 for position in others)
            and isinstance(shift, nm.Integer) and shift._value <= 0)


def causal_read_of(target: cat.BroadcastedCategory) -> sc.StrideMorphism | None:
    '''The reindexing of `target` as one stride morphism where `target` is a view with
    a causal row, and `None` otherwise.'''
    if not (isinstance(target, cat.Broadcasted) and isinstance(target.operator, ops.View)
            and len(target.reindexings) == 1):
        return None
    read = move_reads_backwards.as_stride_morphism(target.reindexings[0])
    if any(is_causal_row(axis, strides, shift, tuple(read._dom))
           for axis, strides, shift in read._cod_stride_shift):
        return read
    return None


@dataclass
class CausalReadCrawler[B: cat.Datatype, A: cat.Axis](
        move_reads_backwards.ReadCrawler[B, A]):
    '''The read crawl, started at every causal read of an expression and carried past
    every operator, or past the operators of `operators_passed` alone where it names
    any, a box named there passing its whole body.'''
    operators_passed: frozenset[cat.Operator] | None = None

    def slid_inside_wrap(self, wrap: para_wrap.ParaWrap
                         ) -> tuple[para_wrap.ParaWrap, Sequence[move_reads_backwards.Read]]:
        '''`wrap` with the causal reads of its body slid, and the reads that reach an
        operand staying on its wire carried out of the wrap. A read reaching a grabbed
        operand stops at the grab, because the slot holds its array whole.'''
        body, body_reads = self.propagate_category(
            wrap.body, tuple(None for _ in wrap.body.cod()))
        stopped_at_a_grab = tuple(read is not None and not Para.is_kept(grab)
                                  for read, grab in zip(body_reads, wrap.grabs))
        if any(stopped_at_a_grab):
            body = chsh.make_composed(chsh.make_product(*(
                move_reads_backwards.view_of(array, read) if stops
                else cat.ProdObject((slid_array,)).identity()
                for array, slid_array, read, stops
                in zip(wrap.body.dom(), body.dom(), body_reads, stopped_at_a_grab))),
                body)
        return (wrap.reconstruct(body=body),
                tuple(read for read, grab in zip(body_reads, wrap.grabs)
                      if Para.is_kept(grab)))

    def slid_inside_box(self, box: cat.Broadcasted[B, A, ops.BlockOperator]
                        ) -> tuple[cat.Broadcasted[B, A],
                                   Sequence[move_reads_backwards.Read]]:
        '''`box` with the causal reads of its body slid. A read reaching the domain of
        the body leaves the box as the pending read of the operand, composed with the
        reindexing of that operand, so that it goes on sliding outside.'''
        block, block_reads = self.propagate_category(
            box.operator.block, tuple(None for _ in box.operator.block.cod()))
        if block == box.operator.block:
            return box, (None,) * len(box.input_weaves)
        body_reads = (*(None for _ in grabbed_operands(box.operator)), *block_reads)
        operator = box.operator.reconstruct(block=block)
        degree = tuple(box.degree())
        degree_read = sc.StrideMorphism(
            _dom=degree, _cod_stride_shift=tuple(
                move_reads_backwards.unit_row(axis, len(degree), position)
                for position, axis in enumerate(degree)))
        carried = tuple(
            move_reads_backwards.CarriedRead(reindexing, weave, None)
            if body_read is None
            else move_reads_backwards.carry_through_operand(
                degree_read, reindexing, weave, body_read)
            for reindexing, weave, body_read
            in zip(box.reindexings, box.input_weaves, body_reads))
        return (box.reconstruct(operator=operator,
                                input_weaves=tuple(carry.weave for carry in carried),
                                reindexings=tuple(carry.reindexing for carry in carried)),
                tuple(carry.read for carry in carried))

    def root_processor(self, target: cat.Broadcasted[B, A],
                       guide: Sequence[move_reads_backwards.Read]
                       ) -> tuple[cat.BroadcastedCategory[B, A],
                                  Sequence[move_reads_backwards.Read]]:
        reads_nothing = all(read is None for read in guide)
        causal_read = causal_read_of(target)
        if causal_read is not None and reads_nothing:
            return cat.ProdObject(tuple(target.cod())).identity(), (causal_read,)
        if reads_nothing and isinstance(target, para_wrap.ParaWrap):
            return self.slid_inside_wrap(target)
        if (reads_nothing and isinstance(target, cat.Broadcasted)
                and isinstance(target.operator, ops.BlockOperator)):
            return self.slid_inside_box(target)
        if self.operators_passed is None or reads_nothing:
            return super().root_processor(target, guide)
        if target.operator not in self.operators_passed:
            return self.stopped(target, guide)
        if isinstance(target.operator, ops.BlockOperator):
            return CausalReadCrawler[B, A]().root_processor(target, guide)
        return super().root_processor(target, guide)


def with_causal_reads_slid[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A], crawl: CausalReadCrawler[B, A]
        ) -> cat.BroadcastedCategory[B, A]:
    '''`morphism` with its causal reads carried back by `crawl`, and a view on every
    domain array a read reached.'''
    expression, domain_reads = crawl.propagate_category(
        morphism, tuple(None for _ in morphism.cod()))
    if all(read is None for read in domain_reads):
        return expression
    return chsh.make_composed(
        chsh.make_product(*(
            cat.ProdObject((array,)).identity() if read is None
            else move_reads_backwards.view_of(array, read)
            for array, read in zip(morphism.dom(), domain_reads))),
        expression)


def slide_causal_reads_backwards[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A]) -> cat.BroadcastedCategory[B, A]:
    '''`morphism` in the CausalSlide form: every causal read moved back to the copy
    whose other branches read its operand unmasked, or to the domain.'''
    return with_causal_reads_slid(morphism, CausalReadCrawler[B, A]())


def slide_causal_reads_back_past[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A], operators: frozenset[cat.Operator]
        ) -> cat.BroadcastedCategory[B, A]:
    '''`morphism` with every causal read moved back past the operators `operators`
    alone, which places it where `derive_cached_pass` places a cache when the same
    operators are computed over the cache.'''
    return with_causal_reads_slid(morphism, CausalReadCrawler[B, A](operators_passed=operators))
