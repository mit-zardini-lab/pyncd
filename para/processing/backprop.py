'''
Morphism -> (Forward, Backward).

The construction, in the terms of the sketch this file grew out of. For
`F : X -> Y`,

    Forward   = (0,0) @ (hold(X) * drop(X)) @ F      : X    -> Y
    Backward  = (grab(X) * hold(T*[Y])) @ R[F]       : T*[Y] -> T*[X]

The forward pass is `F` with the residual copied onto a tape. The backward pass
loads the same slot and runs the reverse derivative. **The tape is the only
coupling**: no wire crosses between the two morphisms, which is what lets each
be fused, kernelised and costed on its own
(`obsidian/07-para/Training.md`).

`Grab` and `Drop` are composition-neutral, so `Forward` has the same domain
and codomain as
`F` - the tape does not show up in a signature.

Structure recurses as the contravariant category is read, and for the same reasons:

    Composed         reverse the order, back-propagate each factor
    Product          back-propagate each factor in place
    Block            a block, with the body back-propagated
    Rearrangement    copy dualises to ADDITION, permutation to permutation,
                     deletion to ZERO. For each domain segment, sum every
                     codomain segment that mapping sent to it. None of them is the zero
                     constant, one is the identity.
    Broadcasted      node-expand, then a rule declared a-priori (`derivative.py`)

The difference from `Contravariant.Contravariant` is only at the leaves. A
`Contravariant` holds the whole forward expression and leaves every leaf symbolic.
This module interprets each leaf, so the result is an ordinary morphism in the broadcasted category that the
fusion algebra can then take as input.

The seed case is where `algebra.node_expansion.expand_to_nodes` earns its
place. A `Broadcasted` mixes a reindexing with an operator and their duals are
computed in completely different ways. A reindexing is affine, so its transpose
is determined, and an operator needs a rule declared a-priori, so the two are
separated before anything is differentiated. The nodes are transposed
generically, and the core is
cleanly broadcast and its rule never mentions broadcasting
(`obsidian/07-para/Derivatives.md`).

The residual is taped as the **unexpanded** input and pushed back through the
node in the backward pass. Taping the expanded one would store the broadcast
copy, at `q x d` floats where `q d` were written, and a reindexing is free to
redo.
'''
from __future__ import annotations
from typing import Iterable

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Operators as ops
import construction_helpers as ch  # for the `*` product on ProdObject
from construction_helpers import simple_helper as chsh
import term_utilities.term_utilities as tutil
import algebra.einops_simplification as einops_simplification
import algebra.node_expansion as ne

import para.data_structure.Para as para
import para.data_structure.Contravariant as contravariant
import para.data_structure.MultiCategory as multi_category
import para.data_structure.contraction as pcon
import para.registries.derivative as drv
import para.algebra.tangent as tng


class Taped[L, M: cat.Morphism](multi_category.MultiCategory[L, M]):
    '''A `MultiCategory` of the two passes, coupled by the slots inside them
    and nothing else.

    The first row is the forward pass, covariant. The second row is the
    backward pass, held in a `contravariant.Contravariant` because the pass
    reads from the cotangent of the codomain back to the cotangent of the
    domain. The two properties hand back each pass as the executable covariant
    morphism, which is what every rewrite in `para/algebra` works on.
    '''

    @classmethod
    def from_passes(
        cls,
        forward: cat.ProdCategory[L, M],
        backward: cat.ProdCategory[L, M],
    ) -> Taped[L, M]:
        return cls((forward, contravariant.Contravariant(backward)))

    @property
    def forward(self) -> cat.ProdCategory[L, M]:
        return contravariant.covariant_body(self.content[0])

    @property
    def backward(self) -> cat.ProdCategory[L, M]:
        return contravariant.covariant_body(self.content[1])


def _identity_on(arrays: Iterable[cat.Array]) -> cat.Rearrangement:
    return cat.ProdObject.from_iter(arrays).identity()


def _unit_factors[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M], count: int
) -> fd.Prod[cat.ProdCategory[L, M]]:
    '''
    Split a parallel product of `count` one-in one-out morphisms back into its
    factors. `expand_to_nodes` returns its nodes already producted together, and
    `make_product` collapses an all-identity product into a single
    `Rearrangement` over every wire, so the two shapes it can come back in are
    both handled here rather than at each use.
    '''
    if count == 1:
        return (target,)
    match target:
        case cat.ProductOfMorphisms(content=ms) if len(ms) == count:
            return ms
    if tutil.is_identity(target):
        return tuple(_identity_on((array,)) for array in target.dom())
    raise ValueError(f'expected {count} single-wire factors, got {target}')


def _transpose_node[L, M: cat.Morphism](
    factor: cat.ProdCategory[L, M]
) -> cat.ProdCategory[L, M]:
    '''A node is linear and needs no tape, so its dual is the rule alone.'''
    match factor:
        case cat.Broadcasted():
            residual, back = drv.rule_for(factor.operator)(factor)
            assert residual == drv.Residual(), (
                'a node transpose is producted straight onto the cotangent, so '
                'a rule declaring a residual here would be wired to the wrong '
                f'wire: {factor.operator}')
            return back
        case _:
            # `expand_to_nodes` writes a node whose reindexing does nothing as
            # an identity `Rearrangement`, and an identity transposes to itself.
            assert tutil.is_identity(factor), f'not a node: {factor}'
            return factor


# ==========================================================================
# The cases.
# ==========================================================================
def _broadcasted(target: cat.Broadcasted) -> Taped:
    nodes, core = ne.expand_to_nodes(target)
    dom, cod = tuple(target.dom()), tuple(target.cod())
    factors = _unit_factors(nodes, len(dom))
    residual, reverse = drv.rule_for(core.operator)(core)

    # One slot per taped array, inputs first, which is the order `Residual`
    # states.
    taped = (*(dom[i] for i in residual.inputs),
             *(cod[j] for j in residual.outputs))
    slots = tuple(para.new_slot() for _ in taped)
    in_slots = slots[:len(residual.inputs)]
    out_slots = slots[len(residual.inputs):]

    def save(indices, boundary, chosen):
        '''Copy the taped segments off, then drop them onto their slots.'''
        if not indices:
            return ()
        return (
            cat.Rearrangement((*indices, *range(len(boundary))), boundary),
            chsh.make_product(
                *(para.Drop(tape=slot, size=array)
                  for slot, array in zip(chosen, (boundary[i] for i in indices))),
                _identity_on(boundary)))

    forward = chsh.make_composed(
        *save(residual.inputs, dom, in_slots),
        nodes, core,
        *save(residual.outputs, cod, out_slots))

    cotangent = tuple(tng.obj(cod))
    backward = chsh.make_composed(
        # load the residual alongside the incoming cotangent
        chsh.make_product(
            *(para.Grab(tape=slot, size=array)
              for slot, array in zip(slots, taped)),
            _identity_on(cotangent)),
        # push the taped inputs back through their nodes, so that `reverse` -
        # which is a rule on the core, is given the operands it was written for
        chsh.make_product(
            *(factors[i] for i in residual.inputs),
            _identity_on((*(cod[j] for j in residual.outputs), *cotangent))),
        reverse,
        # and transpose the nodes, which is where a broadcast becomes a sum.
        # An input with no tangent, meaning an index, has no cotangent for a
        # node to transpose, and `reverse` emits nothing for it.
        chsh.make_product(*(
            _transpose_node(factor)
            for factor, array in zip(factors, dom)
            if tng.array(array) is not None)))
    return Taped.from_passes(forward, backward)


def _rearrangement(target: cat.Rearrangement) -> Taped:
    '''
    Copy -> addition, permutation -> permutation, deletion -> zero. All three
    are the one rule: for each domain segment, add up every codomain segment
    that `mapping` sent to it. Over zero segments that is the zero constant;
    over one it is the identity.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    dom_at, cod_at = tng.positions(dom), tng.positions(cod)
    tangent_dom, tangent_cod = tuple(tng.obj(dom)), tuple(tng.obj(cod))
    # A segment with no tangent, meaning an index wire, is absent from both
    # sides, so
    # it is dropped here rather than zeroed.
    groups = tuple(
        tuple(cod_at[j] for j, source in enumerate(target.mapping)
              if source == i and cod_at[j] is not None)
        for i in range(len(dom)))

    parts = []
    for i, group in enumerate(groups):
        if dom_at[i] is None:
            continue
        array = tangent_dom[dom_at[i]]
        match len(group):
            case 0:
                parts.append(pcon.zeros(array))
            case 1:
                parts.append(_identity_on((array,)))
            case width:
                shape = einops_simplification.positional_shape(array)
                parts.append(pcon.contract(
                    (shape,) * width, shape,
                    datatype=array.datatype, operator=ops.AdditionOp()))

    return Taped.from_passes(target, chsh.make_composed(
        cat.Rearrangement(
            mapping=tuple(j for i, group in enumerate(groups)
                          if dom_at[i] is not None for j in group),
            _dom=tangent_cod),
        chsh.make_product(*parts)))


_gradient_slots: dict[para.TapeSlot, para.TapeSlot] = {}


def gradient_slot(slot: para.TapeSlot) -> para.TapeSlot:
    '''
    The slot a parameter's cotangent is written to: `dW_Q` for `W_Q`. Memoised
    per derivation, so the same parameter grabbed twice, which is a tied
    weight, drops its gradient to one slot, whose two writes accumulate. A
    slot's write is `+=` because copying is a comonoid
    (`obsidian/07-para/Training.md`).
    '''
    if slot not in _gradient_slots:
        name = slot.uid._name
        gradient_name = (
            fd.DynamicName(body='d' + (name.body or ''),
                           subscript=name.subscript, settings=name.settings,
                           code_form=fd.join_code_forms('grad', name.code_form)
                           if name.code_form is not None else None)
            if name is not None else fd.DynamicName('d?'))
        _gradient_slots[slot] = gradient_name.capture(para.TapeSlot())
    return _gradient_slots[slot]


def _para_seed(target: para.ParaMorphism) -> Taped:
    '''
    R swaps `Grab` and `Drop` - they are each other's transpose
    (`obsidian/07-para/Training.md`) - onto the gradient slot: reversing
    `Grab<W>` DROPS `dW`, reversing `Drop<t>` grabs `dt`. No residual and
    nothing taped, because the tape itself is linear. A wire with no tangent,
being an
    index) reverses to nothing at all, the same absence `tangent` writes
    everywhere else.
    '''
    tangent = tng.array(target.size)
    if tangent is None:
        return Taped.from_passes(target, cat.ProdObject().identity())
    slot = gradient_slot(target.tape)
    if isinstance(target, para.Grab):
        return Taped.from_passes(target, para.Drop(tape=slot, size=tangent))
    return Taped.from_passes(target, para.Grab(tape=slot, size=tangent))


def _block(target: cat.Block) -> Taped:
    inner = _recurse(target.body)
    aesthetics = target.block_tag.aesthetics
    title = aesthetics.title if aesthetics is not None else None
    # A fresh tag, so the two blocks are not the same block seen twice. The
    # repetition carries across, which is only right for a loop whose tape is
    # read back in the order it was written. *Gaps* records it. The tag's name
    # carries across too, because it is the index of the loop, and the reverse
    # loop reads the members the forward loop wrote under the same index.
    backward_tag = cat.BlockTag(
        repetition=target.block_tag.repetition,
        aesthetics=(
            aesthetics if title is None
            else aesthetics.reconstruct(title=f'R[{title}]')))
    index_name = target.block_tag.uid._name
    if index_name is not None:
        backward_tag = index_name.capture(backward_tag)
    return Taped.from_passes(
        forward=target.reconstruct(body=inner.forward),
        backward=cat.Block(body=inner.backward, block_tag=backward_tag))


# ==========================================================================
# The transform.
# ==========================================================================
def forward_backward[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M]
) -> Taped[L, M]:
    '''
    The forward pass, with its residual taped, and the backward pass that reads
    that tape. `forward.dom() == target.dom()`, `forward.cod() == target.cod()`,
    and `backward` goes from the tangent of the codomain to the tangent of the
    domain.

    The slot counter is reset here, so the same expression derives byte-
    identical listings in two processes, and the gradient-slot memo with it,
    so two derivations do not share slots.
    '''
    para.reset_slots()
    _gradient_slots.clear()
    return _recurse(target)


def _recurse[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M]
) -> Taped[L, M]:
    match target:
        case cat.Block():
            return _block(target)
        case cat.Composed(content=ms):
            parts = tuple(_recurse(m) for m in ms)
            return Taped.from_passes(
                chsh.make_composed(*(part.forward for part in parts)),
                chsh.make_composed(*(part.backward for part in reversed(parts))))
        case cat.ProductOfMorphisms(content=ms):
            parts = tuple(_recurse(m) for m in ms)
            return Taped.from_passes(
                chsh.make_product(*(part.forward for part in parts)),
                chsh.make_product(*(part.backward for part in parts)))
        case cat.Rearrangement():
            return _rearrangement(target)
        case para.Grab() | para.Drop():
            return _para_seed(target)
        case cat.Broadcasted():
            return _broadcasted(target)
    raise ValueError(f'no backpropagation rule for {type(target).__qualname__}')
