'''Check recomputation through a native block and the following tape conversion.'''

import construction_helpers as ch  # noqa: F401 - installs composition operators
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import term_utilities.term_utilities as term_utilities

import algebra.einops_simplification as einops_simplification
import notebooks.website.tutorial.derive_training_step as derive_training_step


def products_model() -> cat.Morphism:
    row = cat.RawAxis.named('r')
    column = cat.RawAxis.named('c')
    row_index = einops_simplification.IndexVariable('row', row)
    column_index = einops_simplification.IndexVariable('column', column)
    array = cat.Array(cat.Reals(), (row, column))
    hold = cat.ProdObject((array,)).identity()
    def lift(operation: cat.Morphism) -> cat.Morphism:
        return (row, column) >> operation
    multiply = einops_simplification.einsum(
        ((row_index, column_index), (row_index, column_index)),
        (row_index, column_index), cat.Reals())
    exp = lift(ops.Arithmetic.template(nm.E ** nm.x))
    add = lift(ops.AdditionOp.template())

    shared_sum = (cat.Rearrangement((2, 3, 0, 1), (array,) * 4)
                  @ (add * hold * hold)
                  @ cat.Rearrangement((0, 1, 0, 1, 0, 2), (array,) * 3))
    first = (hold * exp) @ multiply
    second = ((hold * exp * hold * exp)
              @ (multiply * multiply) @ add)
    model = shared_sum @ (first * second)
    assert tuple(model.dom()) == (array,) * 4
    assert tuple(model.cod()) == (array,) * 2
    return model


def check_unblocked_and_blocked_training_step() -> None:
    model = products_model()
    unblocked = derive_training_step.derive_training_step(model)
    blocked = cat.Block.template(model, title='Products', fill_color='#2c78bf')
    training_step = derive_training_step.derive_training_step(blocked)

    for label, step in (('unblocked', unblocked), ('blocked', training_step)):
        assert tuple(step.recomputed.forward.dom()) == tuple(model.dom()), label
        assert tuple(step.recomputed.forward.cod()) == tuple(model.cod()), label
        assert tuple(step.recomputed.backward.dom()) == tuple(model.cod()), label
        assert tuple(step.recomputed.backward.cod()) == tuple(model.dom()), label

    forward_blocks = tuple(term_utilities.type_search(
        cat.Block, training_step.recomputed.forward))
    assert any(block.block_tag == blocked.block_tag for block in forward_blocks)
    backward_blocks = tuple(term_utilities.type_search(
        cat.Block, training_step.recomputed.backward))
    assert backward_blocks
    training_step.forward_over_backward()


if __name__ == '__main__':
    check_unblocked_and_blocked_training_step()
    print('ok unblocked and blocked recomputation, signatures, blocks, and ParaWrap')
