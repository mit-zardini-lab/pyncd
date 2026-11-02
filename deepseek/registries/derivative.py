'''The reverse derivative of a selection, registered into `para`'s rule table.

`para.registries.derivative.RULES` keys by operator type and `register` is its
decorator, so an operator declared outside `para` gets a reverse derivative by
adding a row rather than by editing the reverse functor. Import this module for
its side effect, as `import deepseek.registries.derivative`, wherever a
selection has to be differentiated.

`deepseek` imports `para` and not the other way round, which is the layer order
`CLAUDE.md` lists. It imports `advanced_axis_dynamics.registries.derivative` as well,
because `ds.merge_selected_axis` builds an `aops.CovariantView` and a model holding one
has to differentiate it.

`obsidian/07-para/Selection and the Reverse Pass.md` states what a selection
does to the reverse functor, and what the rule below stands in for.
'''
from __future__ import annotations

import data_structure.Category as cat
import deepseek.data_structure as ds
import para.data_structure.Para as Para
import para.data_structure.inject as inject
import para.data_structure.transpose as transpose
import para.registries.derivative as derivative
import advanced_axis_dynamics.registries.derivative  # the reverse of a merge
from construction_helpers import simple_helper as chsh


class SelectionFormHasNoReverseRule(Exception):
    '''A `TopK` whose form hands out no positions the reverse pass can inject at.'''


class SelectionOverPositionsHasNoReverseRule(Exception):
    '''A `TopK` selecting over entries whose positions arrive as an operand. Its
    cotangent returns to the entries, and the slot holds the positions of the
    wider axis it reported instead.'''


@derivative.register(ds.TopK)
def top_k(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''`Inject`, which puts each surviving cotangent back on the parent axis.

    A selection maps `[R, n]` to `[R, k]` in either spelling, so its reverse
    derivative maps `[R, k]` to `[R, n]`. `para.data_structure.inject.Inject`
    is the operator with that shape, and it takes the index as its first
    operand, because an injection has to say which of the `n` positions each
    of the `k` cotangents lands on. The forms of `ds.SelectionForm` differ in
    where the index comes from.

    In the `WEIGHTS` form the one output rides a `SparseAxis` and the index is
    not on a wire, so the rule grabs it from the slot `inject.selection_slot`
    derives from the axis, which is the slot a forward expansion drops it to.

    In the `WEIGHTS_SELECT` form the index is the second output, whose
    `Natural` datatype has no cotangent. The rule declares that output as the
    residual, so `backprop` drops it in the forward pass and grabs it in the
    reverse. A model that already drops the index to a slot of its own ends with
    two drops of one wire, and `pathway_collapse.dedup_slots` keeps the model's
    slot and points the reverse pass's grab at it.

    The `ONLY_WEIGHTS` form hands out values and no positions, so no rule can say
    where their cotangents land. The `ONLY_SELECTION` form hands out positions
    alone, and a `Natural` has no cotangent, so its rule would return zero to the
    scores. No rule is written for either, and `obsidian/06-practice/Open Gaps.md`
    lists both. A selection over entries
    whose positions arrive as an operand, per `ds.TopK.template(positions_of=)`,
    has no rule either. Its cotangent has to return to the entries, and the
    positions its slot holds are positions of the wider axis.

    The rule was the identity until `Inject` existed.
    `obsidian/07-para/Selection and the Reverse Pass.md` records what that
    cost.
    '''
    ds.check_form_matches_outputs(target)
    if ds.selects_over_positions(target):
        raise SelectionOverPositionsHasNoReverseRule(
            f'a TopK over {target.input_weaves[0].target().shape()} reports positions '
            f'of {target.input_weaves[1].datatype}, and no rule returns its cotangent '
            'to the entries it selected over')
    match target.operator.form:
        case ds.SelectionForm.WEIGHTS:
            return compressed_top_k(target)
        case ds.SelectionForm.WEIGHTS_SELECT:
            return complete_top_k(target)
    raise SelectionFormHasNoReverseRule(
        f'a TopK in the {target.operator.form.name} form emits no positions the '
        'reverse pass can inject at, and no slot holds them')


def compressed_top_k(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''The `WEIGHTS` form: the index is grabbed from the selection's slot.'''
    sparse, = (axis for axis in target.output_weaves[0].target().shape()
               if isinstance(axis, ds.SparseAxis))
    injected = inject.inject(target, cat.Natural(sparse._size))
    index, values = tuple(injected.dom())
    return derivative.Residual(), chsh.make_composed(
        chsh.make_product(
            Para.Grab(tape=inject.selection_slot(sparse), size=index),
            cat.ProdObject((values,)).identity()),
        injected)


def complete_top_k(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''The `WEIGHTS_SELECT` form: the index output is the residual.'''
    index = target.output_weaves[1].target().datatype
    return derivative.Residual(outputs=(1,)), inject.inject(target, index)


@derivative.register(ds.IndexSelect)
def index_select(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''Scatter the scalar cotangent onto the indexed position of the payload.'''
    if not transpose.is_cleanly_broadcast(target):
        raise ValueError('IndexSelect reverse expects an expanded clean core')
    index, payload = target.input_weaves
    value, = target.output_weaves
    payload_shape = tuple(payload.target().shape())
    if (tuple(index.target().shape()) or len(payload_shape) != 1
            or tuple(value.target().shape())
            or value.datatype != payload.datatype
            or index.datatype != cat.Natural(payload_shape[0].local_size())):
        raise NotImplementedError(
            'IndexSelect reverse supports scalar index and vector payload')
    return derivative.Residual(inputs=(0,)), inject.inject_onto(
        target.degree(), index, value, payload)


@derivative.register(inject.Inject)
def scalar_inject(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''Read the cotangent of an injected scalar at its saved index.'''
    if not transpose.is_cleanly_broadcast(target):
        raise ValueError('Inject reverse expects an expanded clean core')
    index, value = target.input_weaves
    parent, = target.output_weaves
    parent_shape = tuple(parent.target().shape())
    if (tuple(index.target().shape()) or tuple(value.target().shape())
            or len(parent_shape) != 1
            or index.datatype != cat.Natural(parent_shape[0].local_size())
            or value.datatype != parent.datatype):
        raise NotImplementedError(
            'Inject reverse supports only rank-0 index and scalar value into a '
            'vector; slab/selected-axis forms are unsupported')
    selected = cat.Broadcasted(
        operator=ds.IndexSelect(),
        input_weaves=(index, parent),
        output_weaves=(value,),
        reindexings=(target.degree().identity(),) * 2,
    )
    return derivative.Residual(inputs=(0,)), selected
