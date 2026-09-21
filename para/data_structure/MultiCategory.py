'''Several expressions read side by side, each one forward or reversed.

A `MultiCategory` holds one expression per row. A covariant row is a `cat.ProdCategory`.
A contravariant row is a `contravariant.Contravariant`.
`contravariant.covariant_or_contravariant` reports which of the two a row is by testing
whether it is wrapped.

Composition and the product apply to the columns of several `MultiCategory`s laid in a
line. `(F, G) @ (H, K)` is `(F @ H, G @ K)`. A column of contravariant rows composes its
bodies in the opposite order, which `compose_column` applies.
'''
from __future__ import annotations
from typing import Iterable, Sequence
from dataclasses import dataclass
import data_structure.Term as fd
import data_structure.Category as cat

import para.data_structure.Contravariant as contravariant

import utilities.utilities as util

import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m


type MultiCategoryElement[L, M: cat.Morphism] = (
    cat.ProdCategory[L, M] | contravariant.ContravariantCategory[L, M])

type MultiObject[L] = cat.ProdObject[cat.ProdObject[L]]


class DirectionsDisagree(Exception):
    '''A column mixing a forward expression with a reversed one.'''


@dataclass(frozen=True)
class MultiCategory[L, M: cat.Morphism](
        cat.Morphism[cat.ProdObject[L]], Sequence[MultiCategoryElement[L, M]]):
    content: fd.Prod[MultiCategoryElement[L, M]]

    @classmethod
    def from_iter(
        cls, xs: Iterable[MultiCategoryElement[L, M]],
    ) -> MultiCategory[L, M]:
        return cls(tuple(xs))

    def __len__(self) -> int:
        return len(self.content)
    def __getitem__(self, idx):
        return self.content[idx]
    def dom(self) -> cat.ProdObject[cat.ProdObject[L]]:
        return cat.ProdObject.from_iter(m.dom() for m in self.content)
    def cod(self) -> cat.ProdObject[cat.ProdObject[L]]:
        return cat.ProdObject.from_iter(m.cod() for m in self.content)

    def covariant_or_contravariant(
            self) -> fd.Prod[contravariant.CovariantOrContravariant]:
        return tuple(
            contravariant.covariant_or_contravariant(m) for m in self)

    def to_graph(self, reduce: bool = True) -> MultiCategoryGraph[L, M]:
        return MultiCategoryGraph.from_multicategory(self, reduce=reduce)

def check_column_direction[L, M: cat.Morphism](
    column: fd.Prod[MultiCategoryElement[L, M]],
) -> contravariant.CovariantOrContravariant:
    direction = util.iallequals(
        (contravariant.covariant_or_contravariant(m) for m in column),
        fallback=None)
    if direction is None:
        raise DirectionsDisagree(
            'A column holds both a covariant expression and a contravariant one. '
            'Every element of a column must read the same way.')
    return direction


def compose_column[L, M: cat.Morphism](
    column: fd.Prod[MultiCategoryElement[L, M]],
) -> MultiCategoryElement[L, M]:
    '''The bodies of a contravariant `column` compose in the opposite order.'''
    bodies = tuple(contravariant.covariant_body(m) for m in column)
    direction = check_column_direction(column)
    if direction is contravariant.CovariantOrContravariant.CONTRAVARIANT:
        return contravariant.Contravariant(
            cat.Composed.from_iter(reversed(bodies)))
    return cat.Composed.from_iter(bodies)


def product_column[L, M: cat.Morphism](
    column: fd.Prod[MultiCategoryElement[L, M]],
) -> MultiCategoryElement[L, M]:
    bodies = tuple(contravariant.covariant_body(m) for m in column)
    direction = check_column_direction(column)
    if direction is contravariant.CovariantOrContravariant.CONTRAVARIANT:
        return contravariant.Contravariant(
            cat.ProductOfMorphisms.from_iter(bodies))
    return cat.ProductOfMorphisms.from_iter(bodies)


def compose_multicategory[L, M: cat.Morphism](
    *ms: MultiCategory[L, M],
) -> MultiCategory[L, M]:
    '''Compose column by column. `(F, G) @ (H, K)` is `(F @ H, G @ K)`.

    `construction_helpers` has no case for `contravariant.Contravariant`, so the
    composition is written out here rather than delegated to the autocomposer.
    '''
    return MultiCategory.from_iter(compose_column(column) for column in zip(*ms))


def product_multicategory[L, M: cat.Morphism](
    *ms: MultiCategory[L, M],
) -> MultiCategory[L, M]:
    return MultiCategory.from_iter(product_column(column) for column in zip(*ms))


def stack_multicategory[L, M: cat.Morphism](
    *ms: MultiCategory[L, M],
) -> MultiCategory[L, M]:
    return MultiCategory(util.concat(ms))


@dataclass
class MultiCategoryGraph[L, M: cat.Morphism]:
    content: fd.Prod[
        tuple[contravariant.CovariantOrContravariant, hg.Hypergraph[L, M]]]

    def to_multicategory(self) -> MultiCategory[L, M]:
        morphisms = ((direction, h2m.hypergraph_to_morphism(graph))
                     for direction, graph in self.content)
        contra = contravariant.CovariantOrContravariant.CONTRAVARIANT
        return MultiCategory.from_iter(
            contravariant.Contravariant(m) if direction == contra else m
            for direction, m in morphisms
        )

    @classmethod
    def from_multicategory(
        cls, mc: MultiCategory[L, M], reduce: bool = True,
    ) -> MultiCategoryGraph[L, M]:
        return cls(tuple(
            (contravariant.covariant_or_contravariant(m),
             hg.Multigraph.from_morphism(
                 contravariant.covariant_body(m), reduce=reduce))
            for m in mc
        ))
