'''The contravariant category, in which each morphism exchanges its domain and codomain.

`Contravariant` holds the expression it was built from in `body` and exchanges the two
ends. `Contravariant` is a construction rule in the sense of *Weaves, Wires, and
Morphisms*: the builder is recovered by reading `body`. The transformation is applied to
the interpretation of the recovered body rather than to the body itself. `dom` and `cod`
are the methods that rule needs. Each reads the opposite end of `body`.

"Covariant" and "contravariant" name the direction a categorical construction reads.
"Forward" and "backward" name the two training passes, and appear in `backprop.py` and
`ParaWrap.py` rather than here.

A `Contravariant` says nothing about the value of what it holds. The body keeps the
classes it had. A copy in the body is therefore a `cat.Rearrangement`, and the enclosing
`Contravariant` is what makes the copy denote an addition. A caller that needs the value
supplies that reading itself.

Composing two contravariant expressions composes their bodies in the opposite order. A
product of contravariant expressions keeps its order. `compose_column` and
`product_column`, in `para/data_structure/MultiCategory.py`, apply both.

`obsidian/07-para/Derivatives.md` states what the reverse of each operator is.
'''
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import data_structure.Category as cat


@dataclass(frozen=True)
class Contravariant[L, M: cat.Morphism](cat.Morphism[L]):
    body: M

    def dom(self) -> cat.ProdObject[L]: return self.body.cod()
    def cod(self) -> cat.ProdObject[L]: return self.body.dom()


type ContravariantCategory[L, M: cat.Morphism] = (
    Contravariant[L, cat.ProdCategory[L, M]])


class CovariantOrContravariant(Enum):
    COVARIANT = 'COVARIANT'
    CONTRAVARIANT = 'CONTRAVARIANT'


def covariant_or_contravariant[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M] | ContravariantCategory[L, M],
) -> CovariantOrContravariant:
    if isinstance(target, Contravariant):
        return CovariantOrContravariant.CONTRAVARIANT
    return CovariantOrContravariant.COVARIANT


def covariant_body[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M] | ContravariantCategory[L, M],
) -> cat.ProdCategory[L, M]:
    '''The expression `target` was built from. A covariant `target` is its own body.'''
    return target.body if isinstance(target, Contravariant) else target
