'''Splitting a reindexing into the independent maps it holds.

Written by Claude Opus 5 (1M context), reasoning effort high.

A row of a `sc.StrideMorphism` reads the domain axes it takes a stride along and no
others, so a morphism whose rows read disjoint sets of axes is several maps written as
one. `disentangle_reindexing` returns the product of them: one factor per connected
component of the rows and the domain axes they read, an identity where a component is
the identity on its axis, and the rearrangements that carry the domain and the codomain
between their own order and the grouped one. The result is equal to the input as a map,
and it draws as one pentagon per factor with a straight wire through every axis the
factor does not touch.

The recurring case is a map that mixes one real reindexing with axes that pass through.
The degree reindexing of the lightning indexer's merge re-guides the slot axis and names
the key width on a row of its own, and the key width is unchanged by the merge, so the
whole morphism drew as a two-row hexagon where the re-guiding belongs on the slot wire
alone. `aops.degree_reindexing` disentangles what it builds for that reason.

`ops.View.template` is left alone. Splitting a view's reindexing changes the term every
later rewrite reads, which is a decision of its own.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the rule and what is left
undecided. `algebra/einops_rearrange.disentangle_einops` performs the
matching split for a contraction, over the operands that share a contraction group.
'''
from __future__ import annotations
from dataclasses import dataclass

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.ProductCategory as pc
import data_structure.StrideCategory as sc


@dataclass(frozen=True)
class ReindexingComponent:
    '''One independent map inside a reindexing: the positions of the domain axes its
    rows read, and the codomain rows that read them, each in increasing order.'''
    domain_positions: fd.Prod[int]
    rows: fd.Prod[int]

    def order(self) -> tuple[int, int]:
        '''The key the components are sorted by, which is the first domain axis a
        component reads, and the first row of a component that reads none.'''
        if self.domain_positions:
            return (0, self.domain_positions[0])
        return (1, self.rows[0])


def connected_components[A: sc.Axis](
    morphism: sc.StrideMorphism[A]) -> fd.Prod[ReindexingComponent]:
    '''The domain positions and the codomain rows of `morphism` grouped into connected
    components, a row joined to every domain axis it takes a stride along.

    A domain axis no row reads is a component of its own, which is a deletion, and a row
    that reads no axis is a component of its own, which writes its shift. The components
    are sorted by `ReindexingComponent.order`, so a morphism whose components stand in
    one order in the domain and in the codomain keeps both orders and needs no
    rearrangement.
    '''
    dom = tuple(morphism._dom)
    rows = morphism._cod_stride_shift
    parent = list(range(len(dom) + len(rows)))

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def join(one: int, other: int) -> None:
        one_root, other_root = root(one), root(other)
        if one_root != other_root:
            parent[one_root] = other_root

    for index, (_, strides, _) in enumerate(rows):
        for position, stride in enumerate(strides):
            if not nm.is_zero(stride):
                join(position, len(dom) + index)
    members: dict[int, list[int]] = {}
    for item in range(len(dom) + len(rows)):
        members.setdefault(root(item), []).append(item)
    components = [
        ReindexingComponent(
            domain_positions=tuple(item for item in group if item < len(dom)),
            rows=tuple(item - len(dom) for item in group if item >= len(dom)))
        for group in members.values()]
    return tuple(sorted(components, key=ReindexingComponent.order))


type Rows[A: sc.Axis] = fd.Prod[tuple[A, fd.Prod[nm.Numeric], nm.Numeric]]


def states_the_identity[A: sc.Axis](dom: fd.Prod[A], rows: Rows[A]) -> bool:
    '''Whether `rows` are the identity on `dom`: one row per domain axis, in order, onto
    that same axis, at unit stride and with no shift.'''
    if len(rows) != len(dom):
        return False
    return all(
        axis == dom[position] and nm.is_zero(shift)
        and strides[position] == nm.Integer(1)
        and all(nm.is_zero(stride) for index, stride in enumerate(strides)
                if index != position)
        for position, (axis, strides, shift) in enumerate(rows))


def sliced_component[A: sc.Axis](
    morphism: sc.StrideMorphism[A],
    component: ReindexingComponent) -> tuple[fd.Prod[A], Rows[A]]:
    '''The domain axes and the rows of `component`, with each row's strides cut down to
    the component's domain positions.'''
    dom = tuple(morphism._dom)
    rows = morphism._cod_stride_shift
    return (tuple(dom[position] for position in component.domain_positions),
            tuple((rows[row][0],
                   tuple(rows[row][1][position]
                         for position in component.domain_positions),
                   rows[row][2])
                  for row in component.rows))


def factor_of_component[A: sc.Axis](
    morphism: sc.StrideMorphism[A],
    component: ReindexingComponent,
    names_the_whole_map: bool,
) -> sc.StrideCategory[A]:
    '''The map `component` states, as an identity where it is the identity on its axis
    and as a fresh `sc.StrideMorphism` otherwise.

    The morphism's name names the whole map, so a factor takes it only where every other
    factor is an identity and the factor is therefore that map beside axes it leaves
    alone.
    '''
    factor_dom, factor_rows = sliced_component(morphism, component)
    if states_the_identity(factor_dom, factor_rows):
        return pc.ProdObject(factor_dom).identity()
    return sc.StrideMorphism(
        _dom=factor_dom, _cod_stride_shift=factor_rows,
        name=morphism.name if names_the_whole_map else None)


def disentangled_stride_morphism[A: sc.Axis](
    morphism: sc.StrideMorphism[A]) -> sc.StrideCategory[A]:
    '''`morphism` as the product of its independent factors, with the rearrangements its
    components need where they interleave.

    A morphism of one component that is not an identity is returned as the same object,
    which keeps the sharing every other pass depends on.
    '''
    dom = tuple(morphism._dom)
    rows = morphism._cod_stride_shift
    components = connected_components(morphism)
    identities = [states_the_identity(*sliced_component(morphism, component))
                  for component in components]
    factors = tuple(
        factor_of_component(morphism, component,
                            names_the_whole_map=all(
                                identity for other, identity
                                in zip(components, identities)
                                if other is not component))
        for component in components)
    if len(factors) == 1 and not identities[0]:
        return morphism
    domain_order = tuple(position for component in components
                         for position in component.domain_positions)
    row_order = tuple(row for component in components for row in component.rows)
    product = (factors[0] if len(factors) == 1
               else pc.ProductOfMorphisms.from_iter(factors))
    pieces: list[sc.StrideCategory[A]] = []
    if domain_order != tuple(range(len(dom))):
        pieces.append(pc.Rearrangement(domain_order, dom))
    pieces.append(product)
    if row_order != tuple(range(len(rows))):
        grouped = tuple(rows[row][0] for row in row_order)
        pieces.append(pc.Rearrangement(
            tuple(row_order.index(row) for row in range(len(rows))), grouped))
    if len(pieces) == 1:
        return pieces[0]
    return pc.Composed.from_iter(pieces)


def disentangle_reindexing[A: sc.Axis](
    morphism: sc.StrideCategory[A]) -> sc.StrideCategory[A]:
    '''`morphism` with every `sc.StrideMorphism` in it replaced by the product of its
    independent factors.

    A `pc.Rearrangement` holds no arithmetic to split and is returned as it stands, and
    a composite is disentangled factor by factor. A composite nothing changed in is
    returned as the same object.
    '''
    match morphism:
        case sc.StrideMorphism():
            return disentangled_stride_morphism(morphism)
        case pc.ProductOfMorphisms(content=content):
            factors = tuple(disentangle_reindexing(factor) for factor in content)
            if all(before is after for before, after in zip(content, factors)):
                return morphism
            return pc.ProductOfMorphisms.from_iter(factors)
        case pc.Composed(content=content):
            factors = tuple(disentangle_reindexing(factor) for factor in content)
            if all(before is after for before, after in zip(content, factors)):
                return morphism
            return pc.Composed.from_iter(factors)
    return morphism
