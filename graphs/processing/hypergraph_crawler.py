# The crawler is a general utility operation for propagating a "guide"
# alongside a category/hypergraph, top-down (ForwardCrawler) or
# bottom-up (ReverseCrawler).

from __future__ import annotations
from typing import Callable, Sequence, overload
from dataclasses import dataclass
import data_structure.Numeric as nm
import data_structure.Term as fd
import data_structure.Category as cat
import utilities.utilities as util
import graphs.data_structure.Hypergraph as hg
import graphs.processing.HypergraphAnalysis as hga

from abc import ABC, abstractmethod
import display as dpl

def realign_guide[G](
    guide: Sequence[G],
    wires: Sequence[hg.HypergraphObject],
    new_wires: Sequence[hg.HypergraphObject],
) -> fd.Prod[G]:
    """`guide`, one entry per entry of `wires`, re-read against `new_wires`.
    The two sequences name the same wires but need not agree on order or on
    how many times a wire is listed, because a HypergraphBlock deduplicates its
    dom by wire and its body does not. A guide crossing between them is keyed by
    wire, and a wire listed twice must have been given the same guide both
    times."""
    by_wire = util.Multidict(zip(wires, guide))
    return tuple(util.iallequals(by_wire[wire]) for wire in new_wires)


## CRAWLERS
@dataclass
class Crawler[L, M: cat.Morphism, G](ABC):
    """Shared machinery for the top-down (ForwardCrawler) and bottom-up
    (ReverseCrawler) directions. Subclasses supply the direction-specific
    traversals, propagate_category and propagate_multigraph, and the functions
    that rewrite a root and an object."""

    @abstractmethod
    def root_processor(self, target: M, guide: Sequence[G]) -> tuple[M, Sequence[G]]: ...

    @abstractmethod
    def object_processor(self, target: L, guide: G) -> tuple[L, G]: ...

    @abstractmethod
    def propagate_category(self, target: cat.ProdCategory[L, M], guide: Sequence[G]) -> tuple[cat.ProdCategory[L, M], Sequence[G]]: ...

    @abstractmethod
    def propagate_multigraph(self, target: hg.Multigraph[L, M], guide: Sequence[G]) -> tuple[hg.Multigraph[L, M], Sequence[G]]: ...

    def generate_guide_for_zeros(self, target: L) -> G:
        raise NotImplementedError("This method should be implemented in subclasses.")

    # The side a guide arrives at and the one it leaves from: dom then cod
    # for the forward direction, cod then dom for the reverse.
    @abstractmethod
    def entry_nodes(self, target: hg.Hypergraph[L, M]) -> Sequence[hg.HypergraphObject[L]]: ...

    @abstractmethod
    def exit_nodes(self, target: hg.Hypergraph[L, M]) -> Sequence[hg.HypergraphObject[L]]: ...

    @overload
    def propagate(self, target: cat.ProdCategory[L, M], guide: Sequence[G] | None = None) -> tuple[cat.ProdCategory[L, M], Sequence[G]]: ...
    @overload
    def propagate(self, target: hg.Hypergraph[L, M], guide: Sequence[G] | None = None) -> tuple[hg.Hypergraph[L, M], Sequence[G]]: ...
    @abstractmethod
    def propagate(self, target: cat.ProdCategory[L, M] | hg.Hypergraph[L, M], guide: Sequence[G] | None = None) -> None:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def propagate_graph(self, target: hg.Hypergraph[L, M], guide: Sequence[G]) -> tuple[hg.Hypergraph[L, M], Sequence[G]]:
        match target:
            case hg.HypergraphRoot(wraps=wraps):
                new_wraps, new_guide = self.root_processor(wraps, guide)
                new_graph = hg.HypergraphRoot.template(
                    new_wraps,
                    target.dom,
                    target.cod
                )
                return new_graph, new_guide
            case hg.HypergraphBlock(body=body, block_tag=block_tag):
                # A block's dom is deduplicated by wire (HypergraphBlock.
                # template) while its body's is not, so a guide crossing the
                # block's dom is realigned by wire, never by position: a
                # wire that enters the block twice has one guide at the
                # block's dom and two, which must agree, at the body's.
                new_body, new_guide = self.propagate_graph(
                    body,
                    realign_guide(guide, self.entry_nodes(target), self.entry_nodes(body)))
                new_graph = hg.HypergraphBlock.template(
                    body=new_body,
                    block_tag=block_tag
                )
                new_guide = realign_guide(
                    new_guide, self.exit_nodes(new_body), self.exit_nodes(new_graph))
                assert (block_tag.repetition == nm.Integer(1)) or guide == new_guide
                return new_graph, new_guide
            case hg.Multigraph():
                return self.propagate_multigraph(target, guide)
            case _:
                raise NotImplementedError(f"Unknown graph type: {type(target)}")


@dataclass
class ForwardCrawler[L, M: cat.Morphism, G](Crawler[L, M, G]):

    def entry_nodes(self, target: hg.Hypergraph[L, M]) -> Sequence[hg.HypergraphObject[L]]:
        return target.dom

    def exit_nodes(self, target: hg.Hypergraph[L, M]) -> Sequence[hg.HypergraphObject[L]]:
        return target.cod

    def propagate(self, target: cat.ProdCategory[L, M] | hg.Hypergraph[L, M], guide: Sequence[G] | None = None): # type: ignore
        match target:
            case cat.Morphism():
                guide = guide or tuple(self.generate_guide_for_zeros(obj) for obj in target.dom())
                return self.propagate_category(target, guide)
            case hg.Hypergraph():
                guide = guide or tuple(self.generate_guide_for_zeros(obj.obj) for obj in target.cod)
                return self.propagate_graph(target, guide)

    def propagate_category(self, target: cat.ProdCategory[L, M], guide: Sequence[G]) -> tuple[cat.ProdCategory[L, M], Sequence[G]]:
        match target:
            case cat.Composed(content=ms):
                new_content = []
                for m in ms:
                    new_m, guide = self.propagate_category(m, guide)
                    new_content.append(new_m)
                return cat.Composed.from_iter(new_content), guide
            case cat.ProductOfMorphisms(content=ms):
                morphisms_guides = tuple(
                    self.propagate_category(m, guide_partition)
                    for m, guide_partition in target.partition(guide)
                )
                return (
                    cat.ProductOfMorphisms.from_iter(m for m, _ in morphisms_guides),
                    util.concat(guide for _, guide in morphisms_guides)
                )
            case cat.Rearrangement(mapping=mapping, _dom=dom):
                doms_guides = tuple(self.object_processor(d, g) for d, g in zip(dom, guide))
                return (
                    cat.Rearrangement(mapping=mapping, _dom=tuple(d for d, _ in doms_guides)),
                    target.apply(tuple(g for _, g in doms_guides))
                )
            case cat.Block():
                new_body, new_guide = self.propagate_category(target.body, guide)
                assert (target.block_tag.repetition == nm.Integer(1)) or guide == new_guide
                return cat.Block(body=new_body, block_tag=target.block_tag), new_guide
            case _:
                return self.root_processor(target, guide)

    def propagate_multigraph(self, target: hg.Multigraph[L, M], guide: Sequence[G]) -> tuple[hg.Multigraph[L, M], Sequence[G]]:
        analyzer = hga.HypergraphAnalysis(target)

        # Keep track of the graphs
        processed_subgraphs: set[hga.GraphTag[L, M]] = set((hga.HypergraphSpecialTag.LEFT, hga.HypergraphSpecialTag.RIGHT))
        unprocessed_subgraphs = set(target._subgraphs)
        # We associate each wire with a guide. A wire's equality reads the uid
        # alone, so a key retrieves the entry whichever `.obj` its occurrence
        # carries: `.obj` is exactly what is being resolved here, and can
        # legitimately differ between two occurrences of one wire (e.g. before
        # and after a functor pass).
        node_guide = dict(zip(target.dom, guide))
        object_guide: dict[hg.HypergraphObject[L], L | None] = {
            obj: None
            for obj in (*target.dom, *target.cod)
        }

        # This keeps the order the same
        new_graph_key: dict[fd.UID[hg.Hypergraph], hg.Hypergraph[L, M]] = {}

        while unprocessed_subgraphs:
            processable_subgraphs = set(
                subgraph for subgraph in unprocessed_subgraphs
                if set(analyzer.left_subgraphs(subgraph)) <= processed_subgraphs
            )
            if not processable_subgraphs:
                raise ValueError("No processable subgraphs found. There may be a cycle or missing dependencies.")
            for subgraph in processable_subgraphs:
                unprocessed_subgraphs.remove(subgraph)
                subgraph_guide = tuple(node_guide[wire] for wire in subgraph.dom)
                new_subgraph, new_guide = self.propagate_graph(subgraph, subgraph_guide)
                node_guide.update(zip(new_subgraph.cod, new_guide))
                processed_subgraphs.add(subgraph)
                new_graph_key[subgraph.uid] = new_subgraph

                # Update the object guide. We align original -> final objects
                # by wire rather than by position: a HypergraphBlock can dedupe
                # its dom/cod to a different length than it had before this
                # subgraph was reconstructed, so a positional zip is not
                # reliable, but wire identity is preserved throughout.
                final_by_wire = {
                    final_object: final_object
                    for final_object in (*new_subgraph.dom, *new_subgraph.cod)
                }
                for original_object in (*subgraph.dom, *subgraph.cod):
                    if original_object in object_guide:
                        final_object = final_by_wire[original_object]
                        assert object_guide[original_object] is None
                        object_guide[original_object] = final_object.obj

        new_dom: fd.Prod[hg.HypergraphObject[L]] = tuple(
            dom.reconstruct(
                obj=object_guide[dom] or self.object_processor(dom.obj, node_guide[dom])[0]
            )
            for dom in target.dom
        )
        new_cod: fd.Prod[hg.HypergraphObject[L]] = tuple(
            cod.reconstruct(
                obj=object_guide[cod] or self.object_processor(cod.obj, node_guide[cod])[0]
            )
            for cod in target.cod
        )
        new_subgraphs = tuple(
            new_graph_key[subgraph.uid]
            for subgraph in target._subgraphs
        )
        new_guide = tuple(node_guide[cod] for cod in target.cod)
        return hg.Multigraph.template(
            dom=new_dom,
            cod=new_cod,
            subgraphs=new_subgraphs
        ), new_guide


@dataclass
class BuiltForwardCrawler[L, M: cat.Morphism, G](ForwardCrawler[L, M, G]):
    _root_processor: Callable[[M, Sequence[G]], M]
    _object_processor: Callable[[L, G], L]
    _object_to_guide: Callable[[L], G]
    def root_processor(self, target: M, guide: Sequence[G]) -> tuple[M, Sequence[G]]:
        new_target = self._root_processor(target, guide)
        new_guide = tuple(self._object_to_guide(obj) for obj in new_target.cod())
        return new_target, new_guide
    def object_processor(self, target: L, guide: G) -> tuple[L, G]:
        new_target = self._object_processor(target, guide)
        new_guide = self._object_to_guide(new_target)
        return new_target, new_guide

@dataclass
class ReverseCrawler[L, M: cat.Morphism, G](Crawler[L, M, G]):

    def entry_nodes(self, target: hg.Hypergraph[L, M]) -> Sequence[hg.HypergraphObject[L]]:
        return target.cod

    def exit_nodes(self, target: hg.Hypergraph[L, M]) -> Sequence[hg.HypergraphObject[L]]:
        return target.dom

    def propagate(self, target: cat.ProdCategory[L, M] | hg.Hypergraph[L, M], guide: Sequence[G] | None = None): # type: ignore
        match target:
            case cat.Morphism():
                guide = guide or tuple(self.generate_guide_for_zeros(obj) for obj in target.cod())
                return self.propagate_category(target, guide)
            case hg.Hypergraph():
                guide = guide or tuple(self.generate_guide_for_zeros(obj.obj) for obj in target.cod)
                return self.propagate_graph(target, guide)

    def propagate_category(self, target: cat.ProdCategory[L, M], guide: Sequence[G]) -> tuple[cat.ProdCategory[L, M], Sequence[G]]:
        match target:
            case cat.Composed(content=ms):
                new_content = []
                for m in reversed(ms):
                    new_m, guide = self.propagate_category(m, guide)
                    new_content.append(new_m)
                return cat.Composed.from_iter(reversed(new_content)), guide
            case cat.ProductOfMorphisms(content=ms):
                morphisms_guides = tuple(
                    self.propagate_category(m, guide_partition)
                    for m, guide_partition in target.partition_codomain(guide)
                )
                return (
                    cat.ProductOfMorphisms.from_iter(m for m, _ in morphisms_guides),
                    util.concat(guide for _, guide in morphisms_guides)
                )
            case cat.Rearrangement(mapping=mapping, _dom=dom):
                dom_to_cod = util.Multidict(target.pairwise())
                guides = tuple(
                    util.iallequals(guide[j] for j in dom_to_cod[i])
                    if dom_to_cod[i]
                    else self.generate_guide_for_zeros(dom[i])
                    for i, _ in enumerate(dom)
                )
                dom_guides = tuple(self.object_processor(d, g) for d, g in zip(dom, guides))
                new_dom = tuple(d for d, _ in dom_guides)
                new_guide = tuple(g for _, g in dom_guides)
                return (
                    cat.Rearrangement(mapping=mapping, _dom=new_dom),
                    new_guide
                )
            case cat.Block():
                new_body, new_guide = self.propagate_category(target.body, guide)
                assert (target.block_tag.repetition == nm.Integer(1)) or guide == new_guide
                return cat.Block(body=new_body, block_tag=target.block_tag), new_guide
            case _:
                return self.root_processor(target, guide)

    def propagate_multigraph(self, target: hg.Multigraph[L, M], guide: Sequence[G], verbose: bool = False) -> tuple[hg.Multigraph[L, M], Sequence[G]]:
        analyzer = hga.HypergraphAnalysis(target)

        # Keep track of the graphs
        processed_subgraphs: set[hga.GraphTag[L, M]] = set((hga.HypergraphSpecialTag.LEFT, hga.HypergraphSpecialTag.RIGHT))
        unprocessed_subgraphs = set(target._subgraphs)

        if len(unprocessed_subgraphs) == 0:
            new_cod_guides = tuple(
                self.object_processor(c.obj, g) for c, g in zip(target.cod, guide)
            )
            dom_connections = tuple(
                tuple(new_cod_guides[i] for i, cod in enumerate(target.cod) if cod == dom)
                for dom in target.dom
            )
            new_dom_guides = tuple(
                util.iallequals(dc) if dc
                else self.object_processor(dom.obj, self.generate_guide_for_zeros(dom.obj))
                for dom, dc in zip(target.dom, dom_connections)
            )
            return hg.Multigraph.template(
                dom=hg.HypergraphObject.template(tuple(d for d, _ in new_dom_guides), target.dom),
                cod=hg.HypergraphObject.template(tuple(c for c, _ in new_cod_guides), target.cod),
            ), tuple(g for _, g in new_dom_guides)

        # We associate each wire with a guide. A wire's equality reads the uid
        # alone, so a key retrieves the entry whichever `.obj` its occurrence
        # carries: `.obj` is exactly what is being resolved here, and can
        # legitimately differ between two occurrences of one wire (e.g. before
        # and after a functor pass).
        node_guide = util.Multidict(zip(target.cod, guide))
        object_guide: dict[hg.HypergraphObject[L], L | None] = {
            obj: None
            for obj in (*target.dom, *target.cod)
        }
        original_by_wire: dict[hg.HypergraphObject[L], hg.HypergraphObject[L]] = {
            obj: obj
            for obj in (*target.dom, *target.cod)
        }

        # This keeps the order the same
        new_graph_key: dict[fd.UID[hg.Hypergraph], hg.Hypergraph[L, M]] = {}

        while unprocessed_subgraphs:
            processable_subgraphs = set(
                subgraph for subgraph in unprocessed_subgraphs
                if set(analyzer.right_subgraphs(subgraph)) <= processed_subgraphs
            )
            if not processable_subgraphs:
                raise ValueError("No processable subgraphs found. There may be a cycle or missing dependencies.")
            for subgraph in processable_subgraphs:
                unprocessed_subgraphs.remove(subgraph)
                subgraph_guide = tuple(
                    util.iallequals(node_guide[hypergraph_obj])
                    if node_guide[hypergraph_obj]
                    else self.generate_guide_for_zeros(hypergraph_obj.obj)
                    for hypergraph_obj in subgraph.cod
                )
                new_subgraph, new_guide = self.propagate_graph(subgraph, subgraph_guide)
                node_guide.update(zip(new_subgraph.dom, new_guide))
                processed_subgraphs.add(subgraph)
                new_graph_key[subgraph.uid] = new_subgraph

                # Update the object guide. We align original -> final objects
                # by wire rather than by position: a HypergraphBlock can dedupe
                # its dom/cod to a different length than it had before this
                # subgraph was reconstructed, so a positional zip is not
                # reliable, but wire identity is preserved throughout.
                final_by_wire = {
                    final_object: final_object
                    for final_object in (*new_subgraph.dom, *new_subgraph.cod)
                }
                for original_object in (*subgraph.dom, *subgraph.cod):
                    if original_object in object_guide:
                        final_object = final_by_wire[original_object]
                        assert object_guide[original_object] in (None, final_object.obj), f"Assigned object\n{final_object.obj}\nDoes not match registered\n{object_guide[original_object]}\n"
                        object_guide[original_object] = final_object.obj

        new_guide = tuple(
            util.iallequals(node_guide[dom])
            if node_guide[dom]
            else self.generate_guide_for_zeros(dom.obj)
            for dom in target.dom
        )
        new_dom: fd.Prod[hg.HypergraphObject[L]] = tuple(
            dom.reconstruct(obj=(
                    object_guide[dom] or
                    self.object_processor(dom.obj, g)[0]
            ))
            for dom, g in zip(target.dom, new_guide)
        )
        # NOTE: 0/0 is just to raise an error if the cod object is not found.
        if verbose:
            print('Target')
            print(dpl.graph.display_graph(target))
            print('OBJECT GUIDE')
            print(dpl.Box.Horizontal.from_iter(
                dpl.Box.Vertical((
                    dpl.cat.display_uterm(cod),
                    dpl.cat.display_array(cod.obj)
                ))
                for cod in (*target.dom, *target.cod)
            ))
            for wire, final in object_guide.items():
                if final is not None:
                    original = original_by_wire[wire]
                    print(dpl.Box.Horizontal((
                        dpl.Box.Vertical((
                            dpl.cat.display_uterm(original),
                            dpl.cat.display_array(original.obj)
                        )),
                        dpl.Box.TextBox(': '),
                        dpl.cat.display_array(final)
                    )))
        new_cod: fd.Prod[hg.HypergraphObject[L]] = tuple(
            cod.reconstruct(
                obj=object_guide[cod] or 0/0
            )
            for cod in target.cod
        )
        new_subgraphs = tuple(
            new_graph_key[subgraph.uid]
            for subgraph in target._subgraphs
        )
        return hg.Multigraph.template(
            dom=new_dom,
            cod=new_cod,
            subgraphs=new_subgraphs
        ), new_guide


@dataclass
class BuiltReverseCrawler[L, M: cat.Morphism, G](ReverseCrawler[L, M, G]):
    _root_processor: Callable[[M, Sequence[G]], M]
    _object_processor: Callable[[L, G], L]
    _object_to_guide: Callable[[L], G]
    _generate_guide_for_zeros: Callable[[L], G] | None = None
    def root_processor(self, target: M, guide: Sequence[G]) -> tuple[M, Sequence[G]]:
        new_target = self._root_processor(target, guide)
        new_guide = tuple(self._object_to_guide(obj) for obj in new_target.dom())
        return new_target, new_guide
    def object_processor(self, target: L, guide: G) -> tuple[L, G]:
        new_target = self._object_processor(target, guide)
        new_guide = self._object_to_guide(new_target)
        return new_target, new_guide

    def generate_guide_for_zeros(self, target: L) -> G:
        if self._generate_guide_for_zeros is not None:
            return self._generate_guide_for_zeros(target)
        raise NotImplementedError("This method should be implemented in subclasses.")
