# Claude Opus 5.5 (1M context), effort 40.
'''The operations of every linear map and contraction of a pass, symbolic in its sizes.

`operations_of_a_pass` walks a term in the order its parts compose, multiplies in the
repetition of every block and the degree of every box around each operation, and
yields every `ops.Linear` and every contraction of two operands with its operations.
`read_symbolic_work` of `performance_modeling/morphism_work.py` counts two operations
per multiply-add, and so does everything here. The notebook package
`notebooks/caching/CachedGLM53/` wrote the walk first, for the cached GLM-5.3, and
`caching.algebra.cost_cache_placements` costs every placement of the caches of a
derived pass with it.
'''
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.data_structure.ParaWrap as para_wrap
import performance_modeling.morphism_work as morphism_work

import caching.algebra.cache_contents as cache_contents


@dataclass(frozen=True)
class NodeOperations:
    '''The operations of one linear map or contraction in one pass, with the names of
    the boxes around it from the outermost in.'''
    node: cat.Broadcasted
    boxes: tuple[str, ...]
    operations: nm.Numeric


def box_name(node: cat.Broadcasted) -> str:
    name = node.operator.name
    return name.to_bodies() if name is not None else ''


def operations_of_a_pass(
    term: object,
    multiplier: nm.Numeric = nm.Integer(1),
    boxes: tuple[str, ...] = (),
) -> Iterator[NodeOperations]:
    '''Every `ops.Linear` and every contraction of two operands in `term`, with its
    operations in one pass.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from operations_of_a_pass(part, multiplier, boxes)
        case cat.Block(body=body, block_tag=block_tag):
            yield from operations_of_a_pass(
                body, multiplier * block_tag.repetition, boxes)
        case para_wrap.ParaWrap(body=body):
            yield from operations_of_a_pass(body, multiplier, boxes)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block)):
            degree_size = nm.Multiplication.template(
                *(axis.local_size() for axis in term.degree()))
            yield from operations_of_a_pass(
                block, multiplier * degree_size, (*boxes, box_name(term)))
        case cat.Broadcasted(operator=ops.Linear()) | cat.Broadcasted(
                operator=ops.Einops()) if len(term.input_weaves) >= 1:
            if isinstance(term.operator, ops.Einops) and len(term.input_weaves) != 2:
                return
            work = morphism_work.read_symbolic_work(term)
            yield NodeOperations(term, boxes, multiplier * work.operations)


def operations_at_sizes(nodes: Iterator[NodeOperations] | tuple[NodeOperations, ...],
                        sizes: Mapping[str, int]) -> int:
    '''The operations of `nodes` summed, with every size bound by the bodies of its
    name.'''
    return sum(cache_contents.size_of(node.operations, sizes) for node in nodes)
