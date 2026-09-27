# Claude Opus 5.5 (1M context), effort 40.
'''Removing the identities of a morphism, keeping the morphism it denotes.

A rewrite that turns an operation into the identity, as the dequantisation of a cast
does, leaves the identity standing where the operation stood: in a composition, in a
product, as the body of a block or of a box, or as the body of a `ParaWrap`.
`remove_identities` takes each one out wherever the laws of the category allow, from
the leaves upwards, so a block left holding the identity becomes the identity and
its removal can leave the block around it holding the identity in turn.

A composition drops each member that is an identity, and becomes the identity when
every member is one. A product of identities becomes the identity on its domain. A
block whose body is the identity is the identity whatever its tag, repetition, title
or colour, because a block only groups and a repeated identity is the identity. A box,
an `ops.BlockOperator` operation, whose block is the identity and which reads and
writes the same arrays through identity reindexings, is the identity as well. A
`ParaWrap` that grabs or drops acts on the tape and is never the identity, so it keeps
the identity as its body. `obsidian/04-quantization/Stripping Quantisations.md` states the
dequantisation that uses the rules.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import para.data_structure.ParaWrap as para_wrap
import term_utilities.term_utilities as tutil


def is_identity_on_its_domain(morphism: object) -> bool:
    '''Whether `morphism` is the identity on its domain: a rearrangement taking every
    wire to itself, a composition or a product of identities, a block whose body is
    one, a box whose block is one and which reads and writes the same arrays through
    identity reindexings, or a `ParaWrap` of one that grabs and drops nothing.'''
    match morphism:
        case cat.Rearrangement():
            return tutil.is_identity(morphism)
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return all(is_identity_on_its_domain(part) for part in parts)
        case cat.Block(body=body):
            return is_identity_on_its_domain(body)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block)):
            return (is_identity_on_its_domain(block)
                    and morphism.dom() == morphism.cod()
                    and all(tutil.is_identity(reindexing)
                            for reindexing in morphism.reindexings))
        case para_wrap.ParaWrap(body=body, grabs=grabs, drops=drops):
            return (all(entry is None for entry in (*grabs, *drops))
                    and is_identity_on_its_domain(body))
    return False


def identity_on_the_domain_of[L](morphism: cat.Morphism[L]) -> cat.Rearrangement[L]:
    return cat.ProdObject.from_iter(morphism.dom()).identity()


def written_as_identity[T](morphism: T) -> T | cat.Rearrangement:
    '''The identity rearrangement on the domain of `morphism` where `morphism` is an
    identity written another way, and `morphism` otherwise.'''
    if (is_identity_on_its_domain(morphism)
            and not isinstance(morphism, cat.Rearrangement)):
        return identity_on_the_domain_of(morphism)  # type: ignore[arg-type]
    return morphism


def with_identities_taken_out[T](target: T) -> T:
    '''`target`, whose parts are already simplified, with the identities among its
    own parts taken out or written as the identity rearrangement.'''
    match target:
        case cat.Composed(content=parts):
            kept = tuple(part for part in parts if not is_identity_on_its_domain(part))
            if not kept:
                return identity_on_the_domain_of(target)  # type: ignore[return-value]
            if len(kept) == 1:
                return kept[0]
            return target if len(kept) == len(parts) else target.reconstruct(
                content=kept)
        case cat.ProductOfMorphisms(content=parts):
            if all(is_identity_on_its_domain(part) for part in parts):
                return identity_on_the_domain_of(target)  # type: ignore[return-value]
            written = fd.deep_reconstruct(parts, written_as_identity)
            return target if written is parts else target.reconstruct(content=written)
        case cat.Block(body=body) | para_wrap.ParaWrap(body=body):
            written = written_as_identity(body)
            return target if written is body else target.reconstruct(body=written)
    return target


@dataclass
class _IdentitiesRemoved:
    '''The walk of `remove_identities`, rewriting the parts of a term before the term
    and memoised on object identity, so a subterm reachable along many paths is
    rewritten once. Each entry holds the original term beside its replacement, which
    keeps the original alive, so no identity is recycled while the walk runs.'''
    _rewritten_by_identity: dict[int, tuple[Any, Any]] = field(default_factory=dict)

    def __call__[T](self, target: T) -> T:
        found = self._rewritten_by_identity.get(id(target))
        if found is not None:
            return found[1]
        rewritten = with_identities_taken_out(fd.deep_reconstruct(target, self))
        self._rewritten_by_identity[id(target)] = (target, rewritten)
        return rewritten


def remove_identities[T](target: T) -> T | cat.Rearrangement:
    '''`target` with every identity taken out where the laws of the category allow,
    from the leaves upwards, and `target` itself where there is none. A `target` that
    is an identity written another way is returned as the identity rearrangement.'''
    return written_as_identity(_IdentitiesRemoved()(target))
