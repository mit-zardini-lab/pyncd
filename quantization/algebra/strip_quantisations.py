# Claude Opus 5 (1M context), effort high. Restated by Claude Opus 5.5 (1M context),
# effort 40, on 2026-09-27, as the three steps the user set out for dequantisation.
'''Every quantisation removed from a model, leaving the mathematics underneath.

`quantization.processing.quantise_model` writes a quantisation onto every wire of a
model and puts a `TypeConvert` named `cast` wherever an operation requires another. The
functor here is the other direction. It takes the quantised model back to the
expression in the reals it was made from, in the three steps the user set out on
2026-09-27:

1. `without_quantisations` takes every `Quantified` wrapper, and the `BlockScale`
   carried by one, off every datatype of every wire and every weight.
2. `turn_casts_into_identities` writes the identity on its operand in place of every
   `TypeConvert` that then reads and writes one datatype. A conversion that still
   converts, such as one between two different mathematical values, stays.
3. `algebra.remove_identities.remove_identities` takes the identities out of the
   compositions, the products, the blocks and the boxes that hold them, from the leaves
   upwards, so no identity is left where the categorical laws remove it.

tsncd applies the same functor in the browser, as `dequantise`, to draw the unquantised
variant of a page, and a page applies it to the figure as presented, where a cast may
stand in a box explaining it or as the body of a `ParaWrap`. The explaining box is a
block holding the identity once its cast is one, and step 3 removes it. A `ParaWrap`
that grabs or drops keeps the identity as its body.

Every step is `fd.deep_reconstruct` memoised on object identity, so a subterm reachable
along many paths is rewritten once and a subterm the step does not change comes back as
the object it went in as. No rule here differs by operator, so the package registers
nothing.

`obsidian/04-quantization/Stripping Quantisations.md` states the functor, and
`obsidian/04-quantization/Quantization.md` states the pass it inverts.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import algebra.remove_identities as remove_identities
import data_structure.Category as cat
import data_structure.Term as fd
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


def converts_nothing_once_dequantised(convert: Quantization.TypeConvert) -> bool:
    '''Whether `convert` reads and writes the same datatype once the quantisations are
    taken off its source and its target. A cast between two quantisations of one value
    does, and so does a conversion from a quantised value into the same value
    unquantised. A conversion between two different mathematical values does not.'''
    return without_quantisations(convert.source) == without_quantisations(convert.target)


def reads_and_writes_one_datatype(morphism: object) -> bool:
    '''Whether `morphism` is one operation whose operator is a `TypeConvert` with the
    same source and target, reading and writing the same arrays through identity
    reindexings.'''
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, Quantization.TypeConvert)
            and morphism.operator.source == morphism.operator.target
            and morphism.dom() == morphism.cod()
            and all(tutil.is_identity(reindexing)
                    for reindexing in morphism.reindexings))


@dataclass
class _CastsAsIdentities:
    '''The walk of `turn_casts_into_identities`, memoised on object identity as
    `_Unquantified` is.'''
    _rewritten_by_identity: dict[int, tuple[Any, Any]] = field(default_factory=dict)

    def __call__[T](self, target: T) -> T:
        found = self._rewritten_by_identity.get(id(target))
        if found is not None:
            return found[1]
        rewritten = fd.deep_reconstruct(target, self)
        if reads_and_writes_one_datatype(rewritten):
            rewritten = remove_identities.identity_on_the_domain_of(rewritten)
        self._rewritten_by_identity[id(target)] = (target, rewritten)
        return rewritten


def turn_casts_into_identities[T](target: T) -> T:
    '''`target` with the identity on its operand written in place of every operation
    `reads_and_writes_one_datatype` finds, inside the body of every box and every
    `ParaWrap` as well. Applied after `without_quantisations`, it turns into the
    identity every cast `converts_nothing_once_dequantised` finds.'''
    return _CastsAsIdentities()(target)


def holds_a_quantisation(morphism: cat.Morphism) -> bool:
    '''Whether any datatype of `morphism` carries a quantisation.'''
    return any(True for _ in tutil.type_search(Quantization.Quantified, morphism))


def strip_quantisations(morphism: cat.Morphism) -> cat.Morphism:
    '''`morphism` with every quantisation removed, every cast that then converts
    nothing turned into the identity, and every identity removed where the categorical
    laws allow, and `morphism` itself where it carries no quantisation.

    The functor is idempotent for the second reason: the model it returns carries no
    quantisation, so stripping it again returns it.
    '''
    if not holds_a_quantisation(morphism):
        return morphism
    return remove_identities.remove_identities(
        turn_casts_into_identities(without_quantisations(morphism)))
