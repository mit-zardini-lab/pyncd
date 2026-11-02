'''Check conversion of a morphism to a hypergraph and back when blocks pass wires
through untouched. The recurrence checks compare structure and computed values;
random UIDs make a listing comparison unsuitable.
'''
from __future__ import annotations

import torch

import construction_helpers as ch
import construction_helpers.lift as lift
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.generate_config as generate_config
import term_utilities.term_utilities as tutil
import torch_compile.torch_compile as torch_compile

reals = cat.Reals()
width = cat.RawAxis.named('n')
array = cat.Array(reals, (width,))


def identity(count: int) -> cat.Morphism:
    return cat.ProdObject((array,) * count).identity()


def add() -> cat.Morphism:
    return lift.morphism_object_lift(ops.AdditionOp.template(), cat.ProdObject((width,)))


def recurrence_step(title: str) -> cat.Block:
    '''`(g, p, c, s) -> (g, c, t, s + t)` with `t = (g + c) + p`.'''
    recur = (cat.Rearrangement((0, 2, 0, 2, 1, 3), (array,) * 4)
             @ (identity(2) * ((add() * identity(1)) @ add()) * identity(1)))
    accumulate = cat.Rearrangement((0, 1, 2, 3, 2), (array,) * 4) @ (identity(3) * add())
    return cat.Block.template(recur @ accumulate, title=title, fill_color='#E02C6D')


def block_titles(term: cat.Morphism) -> list[str]:
    return sorted(block.aesthetics.title
                  for block in tutil.type_search(cat.Block, term))


def compiled_at(term: cat.Morphism, size: int) -> torch_compile.ConstructedModule:
    config = generate_config.NumericConfig.template(term)
    config.assign_values(n=size)
    return torch_compile.ConstructedModule.construct(config.apply_context(term))


def check_a_wire_passed_on_by_the_next_block() -> None:
    '''The next step passes a wire written by the first block.'''
    for kept in ((3,), (2,)):
        term = (recurrence_step('first') @ recurrence_step('second')
                @ cat.Rearrangement(kept, (array,) * 4))
        recycled = h2m.recycle(term)
        assert recycled.dom() == term.dom() and recycled.cod() == term.cod(), kept
        assert block_titles(recycled) == ['first', 'second'], block_titles(recycled)
        inputs = [torch.arange(3, dtype=torch.float64) + i for i in range(4)]
        assert torch.allclose(compiled_at(term, 3)(*inputs)[0],
                              compiled_at(recycled, 3)(*inputs)[0]), kept
        print(f'passed-through computed wire, kept {kept}: passed')


def check_a_block_of_wiring_alone_is_kept() -> None:
    '''A block of wiring is still drawn as that block.'''
    for body in (cat.Rearrangement((1, 0), (array, array)),
                 cat.Rearrangement((0, 0), (array,)),
                 identity(2)):
        block = cat.Block.template(body, title='wiring', fill_color='#E02C6D')
        assert block_titles(h2m.recycle(block)) == ['wiring']
        print(f'wiring block title retained for {body.mapping}: passed')


def check_a_passenger_beside_an_operation() -> None:
    '''Two blocks pass one wire through and write the other.'''
    double = width >> ops.Arithmetic.template(2 * nm.x)
    term = (cat.Block.template(identity(1) * double, title='first', fill_color='#E02C6D')
            @ cat.Block.template(identity(1) * double, title='second',
                                 fill_color='#E02C6D'))
    assert block_titles(h2m.recycle(term)) == ['first', 'second']
    print('passenger beside two operations: passed')


if __name__ == '__main__':
    check_a_wire_passed_on_by_the_next_block()
    check_a_block_of_wiring_alone_is_kept()
    check_a_passenger_beside_an_operation()
