# Claude Opus 5 (1M context), effort high.
'''Every quantisation removed from a model, leaving the mathematics underneath.

`quantization.processing.quantise_model` writes a quantisation onto every wire of a
model and puts a `TypeConvert` named `cast` wherever an operation requires another. The
functor here is the other direction. It takes the quantised model back to the
expression in the reals it was made from: every `Quantified` wrapper, and the
`BlockScale` carried by one, is taken off every datatype of every wire and every weight,
and every `TypeConvert` reading one quantisation of a value into another quantisation of
the same value is deleted, with the operations reading its result redirected onto the
wire its operand came from.

A quantisation-free view is how the arithmetic of a model is read once the formats have
been settled. Two figures of one mechanism, one carrying the formats and one carrying
none, differ in the casts alone, and the second is the expression a derivation runs on,
because a cast is a change of representation and contributes no arithmetic.

The deletion is the functor's own rule rather than a splice. `apply_root` returns the
identity `cat.Rearrangement` on the wire the cast reads, and `construction_helpers`
drops an identity out of a composition and out of a product, so the operations after
the cast read the wire in front of it and no operation stands between them. The
datatypes are taken off by `_Unquantified`, which is `fd.deep_reconstruct` memoised on
object identity, so a subterm reachable along many paths is rewritten once and a subterm
holding no quantisation comes back as the object it went in as.

No rule here differs by operator, so the package registers nothing. The one question
asked of an operation is whether it is a conversion between two quantisations of one
value, and `converts_between_two_quantisations` answers it from the source and the
target the conversion carries.

`obsidian/04-quantization/Stripping Quantisations.md` states the functor, and
`obsidian/04-quantization/Quantization.md` states the pass it inverts.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import construction_helpers.simple_helper as chsh
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.processing.hypergraph_functor as functor
import para.data_structure.ParaWrap as para_wrap
import quantization.data_structure.Quantization as Quantization
import term_utilities.term_utilities as tutil


@dataclass
class _Unquantified:
    '''A term with every `Quantified` replaced by the datatype wrapped by it.

    The walk is memoised on object identity, because terms form a directed acyclic
    graph with heavy sharing and a walk that follows paths rather than nodes visits a
    subterm reachable fifty ways fifty times. Each entry holds the original term beside
    its replacement, which keeps the original alive, so no identity is recycled while
    the walk runs.
    '''
    _rewritten_by_identity: dict[int, tuple[Any, Any]] = field(default_factory=dict)

    def __call__[T: fd.GeneralTerm](self, target: T) -> T:
        found = self._rewritten_by_identity.get(id(target))
        if found is not None:
            return found[1]
        rewritten = self._without_quantisations(target)
        self._rewritten_by_identity[id(target)] = (target, rewritten)
        return rewritten

    def _without_quantisations[T: fd.GeneralTerm](self, target: T) -> T:
        if isinstance(target, Quantization.Quantified):
            return self(target.wraps)  # type: ignore[return-value]
        return fd.deep_reconstruct(target, self)


def without_quantisations[T: fd.GeneralTerm](target: T) -> T:
    '''`target` with every `Quantified` replaced by the datatype wrapped by it, and
    `target` itself where it holds none.

    A quantisation of the reals is taken off a real number and the `cat.Reals` under it
    is returned, a quantisation of the naturals is taken off an index and the
    `cat.Natural` under it keeps its own bound, and a wrapper around a quantised value,
    as `deepseek.data_structure.Complex` is, keeps its own form around the datatype now
    under it.
    '''
    return _Unquantified()(target)


def converts_between_two_quantisations(convert: Quantization.TypeConvert) -> bool:
    '''Whether `convert` reads one quantisation of a value into another quantisation of
    the same value.

    Both sides carry a quantisation, and the two datatypes are the same once the
    quantisations are taken off them, so the conversion changes the format of the value
    and changes no part of the mathematics. A conversion into a datatype carrying no
    quantisation, and a conversion between two different mathematical values, fail one
    of the two tests and are left in place.
    '''
    return (Quantization.quantisation_of(convert.source) is not None
            and Quantization.quantisation_of(convert.target) is not None
            and without_quantisations(convert.source)
            == without_quantisations(convert.target))


def is_a_conversion_between_two_quantisations(morphism: cat.Morphism) -> bool:
    '''Whether `morphism` is one operation whose operator is such a conversion.'''
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, Quantization.TypeConvert)
            and converts_between_two_quantisations(morphism.operator))


@dataclass
class StripQuantisations(functor.Endofunctor[cat.Array, cat.Broadcasted]):
    '''The functor taking a quantised model to the expression in the reals under it.

    `apply_object` takes the quantisation off the array carried by a wire.
    `apply_root` returns the identity on that array for a conversion between two
    quantisations, so the composition it stood in loses it, and returns every other
    operation with the quantisations taken off its weaves, its operator and the slot of
    the tape it reads or writes. The body of a box is a morphism of its own and is taken
    through the functor, so a cast the pass wrote inside a box is removed there too.

    `unquantified` is shared by every step, so one Grab written inside a box and
    recorded a second time on the operator of that box is rewritten once and the two
    occurrences stay the same term.
    '''
    unquantified: _Unquantified = field(default_factory=_Unquantified)

    def apply_object(self, target: cat.Array) -> cat.Array:
        return self.unquantified(target)

    def apply_root(self, target: cat.Broadcasted) -> cat.ProdCategory[
            cat.Array, cat.Broadcasted]:
        if is_a_conversion_between_two_quantisations(target):
            return self.apply_prod_object(target.dom()).identity()
        match target:
            case para_wrap.ParaWrap(body=cat.Broadcasted() as box):
                return self.unquantified(target.reconstruct(body=self._box(box)))
            case cat.Broadcasted(operator=ops.BlockOperator()):
                return self.unquantified(self._box(target))
        return self.unquantified(target)

    def _box(self, target: cat.Broadcasted) -> cat.Broadcasted:
        '''`target` with the body of the block it holds taken through the functor.'''
        operator = target.operator
        if not isinstance(operator, ops.BlockOperator):
            return target
        block = operator.block
        body = self.apply_category(block.body)
        if body is block.body:
            return target
        return target.reconstruct(
            operator=operator.reconstruct(block=block.reconstruct(body=body)))

    def apply_category(self, target: cat.ProdCategory[cat.Array, cat.Broadcasted]
                       ) -> cat.ProdCategory[cat.Array, cat.Broadcasted]:
        '''`target` with the functor applied to its parts, and `target` itself where
        every part came back as the object it went in as.

        `Functor.apply_category` rebuilds a composition, a product and a block from the
        parts returned to it. Rebuilding unconditionally turns the directed acyclic
        graph of a term into a tree, and every later pass then walks the term once per
        path through it, which is the rule `fd.deep_reconstruct` states. The cases below
        are the cases of `Functor.apply_category` with that rule applied to each.
        '''
        match target:
            case cat.Composed(content=parts):
                rewritten = fd.deep_reconstruct(parts, self.apply_category)
                return (target if rewritten is parts
                        else chsh.make_composed(*rewritten))
            case cat.ProductOfMorphisms(content=parts):
                rewritten = fd.deep_reconstruct(parts, self.apply_category)
                return (target if rewritten is parts
                        else chsh.make_product(*rewritten))
            case cat.Block(body=body):
                rewritten = self.apply_category(body)
                return (target if rewritten is body
                        else target.reconstruct(body=rewritten))
            case cat.Rearrangement(_dom=dom):
                rewritten = fd.deep_reconstruct(dom, self.apply_object)
                return (target if rewritten is dom
                        else target.reconstruct(_dom=rewritten))
        return self.apply_root(target)  # type: ignore[arg-type]


def holds_a_quantisation(morphism: cat.Morphism) -> bool:
    '''Whether any datatype of `morphism` carries a quantisation.'''
    return any(True for _ in tutil.type_search(Quantization.Quantified, morphism))


def strip_quantisations(morphism: cat.Morphism) -> cat.Morphism:
    '''`morphism` with every quantisation removed, and `morphism` itself where it
    carries none.

    Every wire and every weight loses the `Quantified` wrapper it carried, together
    with the `BlockScale` that wrapper held, and every conversion between two
    quantisations of one value is deleted, so the operations that read the value read
    the wire its operand came from. A model carrying no quantisation is returned as it
    stands, which also makes the functor idempotent: the model it returns carries none.
    '''
    if not holds_a_quantisation(morphism):
        return morphism
    return StripQuantisations()(morphism)
