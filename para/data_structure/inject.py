'''
Putting a selection's cotangent back on the axis it was selected from.

A selection maps `[R, n]` to `[R, k/n]`. Its reverse derivative has to map
`[R, k/n]` to `[R, n]`, and `Inject` is that morphism: it writes each surviving
cotangent at the position its value came from and zero everywhere else.

    TopK    : [R, n]   -> [R, k/n]
    Inject  : [R, k/n] -> [R, n]

## Why it is an operator and not a contraction

The three ways this package writes a map between two axes are a reindexing, an
`Einops` and an operator. A reindexing is affine, and the map from a slot to
the position it selected is runtime data. An `Einops` signature describes a
contraction, and injecting is not one: a contraction sums over an axis, where
an injection leaves `n - k` entries untouched. What is left is an operator, and
the same three sentences are why `deepseek.Select` and `deepseek.IndexSelect`
exist in the forward direction.

## The two forms of a selection, and the slab it writes

`inject` reverses a `TopK` in either spelling. Compressed, the values arrive on
the sparse axis and the index is loaded from the slot `selection_slot` names.
Complete, the values arrive on the dense `k` and the index is the second
output the `TopK` emitted, which `backprop` tapes as the operator's residual.
The morphism is the same either way: the index first, the values second, and
the parent axis out.

`inject_slab` is the same operator with a rank-0 index. A `Linear` that selects
one of `n` weights has a weight gradient that lands on the one slab the index
named, so its reverse writes the outer product at that position of a fresh `n`
and zero at every other. Summing the result over the degree, which the weight's
own node transpose does, accumulates every token's slab into the expert bank.
The scatter-add `obsidian/07-para/Selection and the Reverse Pass.md` records
as missing is therefore `Inject` followed by the sum a node transpose already
writes.

## What it fixes

Before `Inject` the reverse of a `TopK` was the identity on the incoming
cotangent. The identity typechecks only if `[R, k/n]` and `[R, n]` are the same
object, so composing it made the sparse axis and its parent one axis with two
names, and it left two links of the derived backward pass whose codomain and
domain disagree. Both showed in the backward pass of a mixture of experts.
`Inject` has the shape the reverse derivative's signature asks for, so the axes
stay apart and the compositions agree.
'''
from __future__ import annotations
from dataclasses import dataclass

import data_structure.Term as fd
import data_structure.Category as cat
import para.data_structure.Para as Para
import para.data_structure.transpose as transpose


def selection_slot(axis: cat.Axis) -> Para.TapeSlot:
    '''The slot a selection over `axis` saves its index to.

    The slot's identity is derived from the axis, through `fd.hash_id`, so two
    passes that never meet agree on it. The forward pass drops the index at the
    `TopK` that produced the axis, and the reverse of that `TopK` grabs it here,
    with nothing shared between them but the axis they both name.
    `nm.FreeNumeric.named` derives an id the same way and for the same reason.
    Every other id in the package is random per process, per
    `obsidian/06-practice/Invariants.md` under *Axis identity*.
    '''
    axis_name = axis.uid._name
    name = fd.DynamicName(
        'idx', subscript=axis_name,
        code_form=fd.join_code_forms(
            'index', axis_name.code_form_or_identifier() if axis_name is not None else None))
    return Para.TapeSlot(fd.UID(Para.TapeSlot, fd.hash_id(axis.uid), name))


@dataclass(frozen=True)
class Inject(cat.Operator):
    '''The reverse derivative of a selection, from `k/n` back onto `n`.

    It carries no field beyond its name. The sparse axis on its input weave
    already states how many of the parent's entries survived, through
    `SparseAxis.activity`, so an operator field repeating the count would be a
    second copy of a number the wire carries.
    '''
    name: fd.DynamicName | None = fd.DynamicName('Inject')


def inject_onto[B: cat.Datatype, A: cat.Axis](
    degree: cat.ProdObject[A],
    index: cat.Weave[cat.Natural, A],
    values: cat.Weave[B, A],
    parent: cat.Weave[B, A],
) -> cat.Broadcasted[B, A]:
    '''`Inject` cleanly broadcast over `degree`, writing `values` at the
    positions `index` names and zero elsewhere.'''
    return cat.Broadcasted(
        operator=Inject(),
        input_weaves=(index, values),
        output_weaves=(parent,),
        reindexings=(degree.identity(),) * 2,
    )


def inject[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
    index: cat.Datatype,
) -> cat.Broadcasted[B, A]:
    '''The reverse of a cleanly broadcast selection, from the selected axis
    back onto the parent.

    The output weaves are the selection's input weaves and the values weave is
    the selection's first output weave, for the reason `transpose.transpose`
    gives: every reindexing of a cleanly broadcast operator is the degree
    identity, so exchanging the input and output weaves exchanges the domain
    and the codomain exactly. A complete `TopK` has a second output carrying
    the index, and it is left out, because the index arrives as an operand.

    `index` is the datatype of the selector, meaning `Natural(n)` for a
    selection over `n`. It arrives on the selected axis, one index per
    surviving slot, and it is the first operand, as it is for an expanded
    `Linear` and for `deepseek.IndexSelect`. Without it an injection states
    where the cotangent came from and not where it goes, and the entries the
    selection dropped cannot be told from the entries it kept.

    The precondition is asserted rather than handled. A selection with a live
    reindexing reaching here would produce a morphism whose domain and codomain
    are silently wrong, which is the failure this operator was written to
    remove.
    '''
    assert transpose.is_cleanly_broadcast(target), (
        'inject needs a cleanly broadcast selection. Run expand_to_nodes '
        f'first: {target.operator}')
    values_weave = target.output_weaves[0]
    parent_weave, = target.input_weaves
    return inject_onto(
        target.degree(),
        cat.Weave(index, values_weave._shape), values_weave, parent_weave)


def inject_slab[B: cat.Datatype, A: cat.Axis](
    degree: cat.ProdObject[A],
    index: cat.Weave[cat.Natural, A],
    slab: cat.Weave[B, A],
    selected: A,
    position: int,
) -> cat.Broadcasted[B, A]:
    '''`Inject` with a rank-0 index, writing `slab` at the position the index
    names along `selected`, inserted at `position` of the slab's target.

    `index` is the weave of the index operand, tiled over the whole degree with
    an empty target. The result has the slab's target with `selected` inserted,
    which is the shape of the weight a selecting `Linear` holds.
    '''
    tiled = tuple(entry for entry in slab._shape if isinstance(entry, cat.WeaveMode))
    target = tuple(slab.target().shape())
    parent = cat.Weave(
        slab.datatype,
        (*tiled, *target[:position], selected, *target[position:]))
    return inject_onto(degree, index, slab, parent)
