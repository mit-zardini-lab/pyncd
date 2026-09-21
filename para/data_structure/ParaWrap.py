'''A morphism carrying its grabs and drops on itself.

`Para.py` implements the Para construction with the two seed morphisms `Grab`
and `Drop`, which is the number of cases the algebra needs. A picture needs
more. A tape that arrives beside an operator and a tape that arrives at a box
of its own, a rearrangement away, are two different things to read. `ParaWrap`
is the presentation that says which operands of `body` come off the tape and
which results go back onto it. It adds no mathematical value, and `to_base`
writes it back out as the grabs, the body and the drops it stands for.

`to_para_wrap` is the layering rule that puts a `Para` into that form for
display. It works on the hypergraph, for the reason
`algebra.merge_into_consumer` gives. "Read once, and by that one" is a question
about wires, and a morphism has none. Three rewrites drive it, applied to a
fixed point, between siblings in one scope:

    Grab ; seed          the grab merges into the seed it feeds. The seed
                         becomes a `ParaWrap` with that operand grabbed, or
                         gains the grab when it is a `ParaWrap` already. The
                         case arises in a backward pass, where a taped residual
                         feeds an `Einops` and is drawn entering it.

    seed ; Drop          the drop merges into the seed that produces its wire,
                         when nothing else reads that wire. The seed writes the
                         slot where it would write a result. The case arises
                         wherever a value is computed only to be saved: a
                         weight gradient, and the index a `TopK` emits beside
                         its values.

    (0,0) ; (id * Drop)  a drop on a wire that is also read elsewhere becomes a
                         `ParaWrap` over the copy, with the second output
                         dropped. The case arises in a forward pass, where a
                         value is both passed on and saved. The copy is
                         attached to the wire's producer: the producer is
                         renamed to write a fresh wire, the wrap reads that
                         wire and writes the wire the other readers name, so
                         every reader, the scope's codomain included, is
                         unchanged. A wire that leaves a block is therefore
                         copied inside the block, one copy per wire, and the
                         morphism reads `(copy ; (id * drop)) * (copy ; (id *
                         drop))` where it read one rearrangement `[0,0,1,1]`
                         followed by a row of identities and drops.

A grab whose wire enters a block is loaded inside the block instead. A grab
read by several siblings is split into one `Grab` of the same slot per reader,
each on a wire of its own, so that each can then merge into the morphism it
feeds. The slot is what says the copies are one parameter, because the tape is
the only coupling there is.

A morphism that reads an operand through a reindexing beyond a rearrangement
absorbs neither a grab nor a drop. The wrap draws a grabbed operand on the
operand's own port, and a reindexing that is more than a permutation, a copy or
a deletion of degree axes reads that port at indices the port does not name. The
grab or the drop stays a box of its own instead, and the reindexing is drawn
between the box and the operator.

A grab or drop that no rewrite reaches becomes a `ParaWrap` over an identity,
which is the old `Grab` or `Drop` box exactly. The two cases are a drop whose
wire is produced outside its own scope, and a grabbed value that is itself an
output. The renderer then has one box to draw, and the two old boxes are
instances of it.

`obsidian/07-para/Para Wrap.md` describes the rule and how a wrap is drawn.
'''
from __future__ import annotations
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import data_structure.Term as fd # for 'foundations'
import data_structure.Category as cat
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import para.data_structure.Contravariant as contravariant
import para.data_structure.MultiCategory as multi_category
import para.data_structure.Para as Para
import construction_helpers.simple_helper as sh

@dataclass(frozen=True)
class ParaWrap[L, M: cat.Morphism](cat.Morphism[L]):
    '''`body` with a slot named against each operand and result it tapes.

    `grabs[i]` is the slot that operand `i` of `body` comes off, or `None` when
    that operand arrives on a wire. `drops[j]` is the slot that result `j` goes
    onto, or `None` when that result leaves on a wire. An entry that is a
    `Para.StreamSlot` stands for a `StreamGrab` or a `StreamDrop`, the seeds of a
    loop variable a stream loop carries, a `Para.LoopSlot` for a `LoopGrab` or a
    `LoopDrop`, the seeds of a slot indexed by the iteration of a repeated block,
    and a `Para.ReductionSlot` for a `ReductionGrab` or a
    `ReductionDrop`, the seeds of an exchange. The wrap's own `dom` and `cod` are the operands and results whose
    entry is `None`.
    '''
    body: M | cat.Rearrangement[L]
    grabs: fd.Prod[Para.SlotEntry]
    drops: fd.Prod[Para.SlotEntry]

    def dom(self) -> cat.ProdObject[L]:
        return cat.ProdObject.from_iter(
            d for d, g in zip(self.body.dom(), self.grabs) if g is None)

    def cod(self) -> cat.ProdObject[L]:
        return cat.ProdObject.from_iter(
            c for c, d in zip(self.body.cod(), self.drops) if d is None)

    def to_base(self) -> Para.Para[L, M]:
        input_grabs = sh.make_product(
            *(
                cat.ProdObject((d,)).identity() if grab is None
                else Para.grab_of(grab, d)
                for d, grab in zip(self.body.dom(), self.grabs)
            )
        )
        core = self.body
        output_grabs = sh.make_product(
            *(
                cat.ProdObject((c,)).identity() if drop is None
                else Para.drop_of(drop, c)
                for c, drop in zip(self.body.cod(), self.drops)
            )
        )
        return sh.make_composed(
            input_grabs, core, output_grabs
        ) # type: ignore

    # The positions in `body.dom()` and `body.cod()` that the wrap's own `dom`
    # and `cod` expose. A reader indexing `dom()` is indexing these.
    def kept_inputs(self) -> fd.Prod[int]:
        return tuple(i for i, g in enumerate(self.grabs) if g is None)

    def kept_outputs(self) -> fd.Prod[int]:
        return tuple(i for i, d in enumerate(self.drops) if d is None)

type ParaWrapped[L, M: cat.Morphism] = Para.Para[L, M | ParaWrap[L, M]]


# ==========================================================================
# The layering rule.
# ==========================================================================
WRAP_LIMIT = 1024


def to_para_wrap[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    '''A `Para` with its grabs and drops written onto the morphisms they touch.

    The conversion goes through the hypergraph and back, so the result is
    recycled as well as wrapped.
    '''
    return h2m.hypergraph_to_morphism(
        para_wrap_graph(hg.Multigraph.from_morphism(target)))


def to_para_wrap_rows[L, M: cat.Morphism](
    target: multi_category.MultiCategory[L, M],
) -> multi_category.MultiCategory[L, M]:
    '''Each row wrapped by `to_para_wrap`, keeping the row's direction and the
    class of the whole, so a `backprop.Taped` comes back a `Taped`.'''
    def wrap_row(
        row: multi_category.MultiCategoryElement[L, M],
    ) -> multi_category.MultiCategoryElement[L, M]:
        wrapped = to_para_wrap(contravariant.covariant_body(row))
        if isinstance(row, contravariant.Contravariant):
            return contravariant.Contravariant(wrapped)
        return wrapped
    return type(target).from_iter(wrap_row(row) for row in target)


def para_wrap_graph[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
) -> hg.Hypergraph[L, M]:
    '''The rewrites, applied until no grab and no drop is left bare.'''
    for _ in range(WRAP_LIMIT):
        rebuilt = _wrap_once(graph, is_outermost=True)
        if rebuilt is None:
            return graph
        graph = rebuilt
    raise RuntimeError(f'{WRAP_LIMIT} rewrites without reaching a fixed point')


def _wrap_once(graph: hg.Hypergraph, is_outermost: bool) -> hg.Hypergraph | None:
    '''The first rewrite anywhere in `graph`, innermost scopes first, or `None`
    when no rewrite applies.'''
    match graph:
        case hg.HypergraphRoot():
            return None
        case hg.HypergraphBlock(body=body):
            rebuilt = _wrap_once(body, is_outermost=False)
            if rebuilt is None:
                return None
            return hg.HypergraphBlock.template(rebuilt, graph.block_tag)
        case hg.Multigraph():
            for i, subgraph in enumerate(graph._subgraphs):
                rebuilt = _wrap_once(subgraph, is_outermost=False)
                if rebuilt is not None:
                    return hg.Multigraph.template(
                        graph.dom, graph.cod,
                        tuple(rebuilt if k == i else sub
                              for k, sub in enumerate(graph._subgraphs)))
            return _wrap_here(graph, is_outermost)
    raise NotImplementedError(f'cannot rewrite {type(graph).__name__}')


def _reindexings_of(target: cat.Morphism) -> Iterator[cat.StrideCategory]:
    '''Every reindexing of every `cat.Broadcasted` in `target`.

    The walk follows the construction rules alone and never enters an object,
    so it costs the size of the morphism's own tree. `tutil.type_search` would
    descend into every array, axis and size symbol as well, and `_wrap_here`
    asks this of each candidate on each rewrite.
    '''
    match target:
        case cat.Broadcasted(reindexings=reindexings):
            yield from reindexings
        case (ParaWrap(body=body) | cat.Block(body=body)
              | contravariant.Contravariant(body=body)):
            yield from _reindexings_of(body)
        case cat.Composed(content=content) | cat.ProductOfMorphisms(content=content):
            for part in content:
                yield from _reindexings_of(part)


def _is_rearrangement(target: cat.StrideCategory) -> bool:
    '''Whether `target` permutes, copies and deletes its domain and does
    nothing further.'''
    match target:
        case cat.Rearrangement():
            return True
        case cat.Composed(content=content) | cat.ProductOfMorphisms(content=content):
            return all(_is_rearrangement(part) for part in content)
        case cat.Block(body=body):
            return _is_rearrangement(body)
        case _:
            return False


def _has_non_rearrangement_reindexing(target: cat.Morphism) -> bool:
    '''Whether a `cat.Broadcasted` inside `target` reads an operand through an
    affine index map that a `cat.Rearrangement` cannot express.'''
    return any(not _is_rearrangement(reindexing)
               for reindexing in _reindexings_of(target))


def _is_root_of(subgraph: hg.Hypergraph, *kinds: type) -> bool:
    return (isinstance(subgraph, hg.HypergraphRoot)
            and isinstance(subgraph.wraps, kinds))


def _readers(graph: hg.Multigraph) -> Counter:
    '''How many times each wire is read within this scope, counting a read by a
    sibling subgraph and a read by the scope's own codomain.

    A block counts once for each entry of its domain that names the wire.
    '''
    readers: Counter = Counter()
    for subgraph in graph.subgraphs():
        for node in subgraph.dom:
            readers[node] += 1
    for obj in graph.cod:
        readers[obj] += 1
    return readers


def _with_grab(
    target: cat.Morphism, port: int, slot: Para.NamedEntry,
) -> ParaWrap:
    '''`target` with the operand at `port` grabbed.

    `port` counts the operands the morphism still exposes in its own `dom()`,
    so on an existing `ParaWrap` it is translated through `kept_inputs`.
    '''
    if isinstance(target, ParaWrap):
        index = target.kept_inputs()[port]
        return target.reconstruct(grabs=tuple(
            slot if i == index else grab
            for i, grab in enumerate(target.grabs)))
    return ParaWrap(
        body=target,
        grabs=tuple(slot if i == port else None
                    for i in range(len(target.dom()))),
        drops=(None,) * len(target.cod()))


def _with_drop(
    target: cat.Morphism, port: int, slot: Para.NamedEntry,
) -> ParaWrap:
    '''`target` with the result at `port` dropped.

    `port` counts the results the morphism still exposes in its own `cod()`,
    so on an existing `ParaWrap` it is translated through `kept_outputs`.
    '''
    if isinstance(target, ParaWrap):
        index = target.kept_outputs()[port]
        return target.reconstruct(drops=tuple(
            slot if i == index else drop
            for i, drop in enumerate(target.drops)))
    return ParaWrap(
        body=target,
        grabs=(None,) * len(target.dom()),
        drops=tuple(slot if i == port else None
                    for i in range(len(target.cod()))))


def _wrap_over_identity(target: Para.ParaMorphism) -> ParaWrap:
    '''A grab or a drop as a wrap over the identity, which is the old box.'''
    identity = cat.ProdObject((target.size,)).identity()
    entry = Para.entry_of(target)
    if isinstance(target, Para.Grab):
        return ParaWrap(body=identity, grabs=(entry,), drops=(None,))
    return ParaWrap(body=identity, grabs=(None,), drops=(entry,))


def _rebuild(
    graph: hg.Multigraph,
    replacements: dict[int, hg.Hypergraph | None],
    cod: fd.Prod[hg.HypergraphObject] | None = None,
) -> hg.Hypergraph:
    '''`graph` with subgraph `k` replaced by `replacements[k]`, and removed
    where the replacement is `None`.'''
    rebuilt = tuple(
        replacements.get(k, subgraph)
        for k, subgraph in enumerate(graph._subgraphs)
        if replacements.get(k, subgraph) is not None)
    return hg.Multigraph.template(graph.dom, cod or graph.cod, rebuilt)


def _wrap_here(graph: hg.Multigraph, is_outermost: bool) -> hg.Hypergraph | None:
    readers = _readers(graph)
    subgraphs = graph.subgraphs()
    for i, subgraph in enumerate(subgraphs):
        if _is_root_of(subgraph, Para.Grab):
            return _wrap_grab(graph, i, readers)
        if _is_root_of(subgraph, Para.Drop):
            return _wrap_drop(graph, i, readers, is_outermost)
    return None


def _wrap_grab(
    graph: hg.Multigraph, i: int, readers: Counter,
) -> hg.Hypergraph:
    grab = graph.subgraphs()[i]
    slot = Para.entry_of(grab.wraps)
    wire = grab.cod[0]
    for k, subgraph in enumerate(graph.subgraphs()):
        if isinstance(subgraph, hg.HypergraphBlock) and wire in subgraph.dom:
            return _push_into_block(graph, i, k, readers[wire] == 1)
    if readers[wire] > 1:
        return _split_grab(graph, i)
    if readers[wire] == 1:
        for j, consumer in enumerate(graph.subgraphs()):
            if j == i or not isinstance(consumer, hg.HypergraphRoot):
                continue
            if isinstance(consumer.wraps, (Para.Grab, Para.Drop)):
                continue
            if _has_non_rearrangement_reindexing(consumer.wraps):
                continue
            ports = [p for p, node in enumerate(consumer.dom)
                     if node == wire]
            if len(ports) != 1:
                continue
            port = ports[0]
            wrapped = _with_grab(consumer.wraps, port, slot)
            root = hg.HypergraphRoot.template(
                wrapped,
                (*consumer.dom[:port], *consumer.dom[port + 1:]),
                consumer.cod)
            return _rebuild(graph, {i: None, j: root})
    root = hg.HypergraphRoot.template(_wrap_over_identity(grab.wraps), grab.dom, grab.cod)
    return _rebuild(graph, {i: root})


def _push_into_block(
    graph: hg.Multigraph, i: int, k: int, sole: bool,
) -> hg.Hypergraph:
    '''Load a grab inside the block its wire enters.

    A `Grab` root producing the same node joins the block's body, and the node
    leaves the block's domain. The inner scope then splits and merges that
    grab among the block's own readers, so a residual read by two contractions
    inside the block is loaded beside each of them. The outer grab is removed
    when the block was its only reader, which `sole` reports.
    '''
    grab = graph.subgraphs()[i]
    block = graph.subgraphs()[k]
    wire = grab.cod[0]
    body = block.body
    assert isinstance(body, hg.Multigraph)
    inner = hg.HypergraphRoot.template(grab.wraps, (), (wire,))
    new_body = hg.Multigraph.template(
        tuple(obj for obj in body.dom if obj != wire),
        body.cod,
        (inner, *body._subgraphs))
    new_block = hg.HypergraphBlock.template(new_body, block.block_tag)
    return _rebuild(graph, {k: new_block, **({i: None} if sole else {})})


def _split_grab(graph: hg.Multigraph, i: int) -> hg.Hypergraph:
    '''One grab per sibling reader of its wire, each on a fresh node, with the
    reader redirected onto it.

    The original grab stays when the scope's own codomain reads the wire. That
    reader cannot be given a node of its own without changing the scope's
    codomain, which a sibling rewrite may not do.
    '''
    grab = graph.subgraphs()[i]
    wire = grab.cod[0]
    replacements: dict[int, hg.Hypergraph | None] = {}
    added: list[hg.Hypergraph] = []
    for k, subgraph in enumerate(graph._subgraphs):
        if k == i or wire not in subgraph.dom:
            continue
        copy = hg.HypergraphRoot.template(grab.wraps, ())
        redirect = fd.Context([
            fd.UIDRenaming.set_canonical(copy.cod[0], wire)])
        replacements[k] = redirect.apply(subgraph)
        added.append(copy)
    if not any(obj == wire for obj in graph.cod):
        replacements[i] = None
    rebuilt = tuple(
        replacements.get(k, subgraph)
        for k, subgraph in enumerate(graph._subgraphs)
        if replacements.get(k, subgraph) is not None)
    return hg.Multigraph.template(graph.dom, graph.cod, (*added, *rebuilt))


def _wrap_drop(
    graph: hg.Multigraph, i: int, readers: Counter, is_outermost: bool,
) -> hg.Hypergraph:
    drop = graph.subgraphs()[i]
    slot = Para.entry_of(drop.wraps)
    array = drop.wraps.size
    wire = drop.dom[0]
    # The drop merges into the seed that produces its wire, when that seed is a
    # sibling and nothing else reads the wire.
    if readers[wire] == 1:
        merged = _merge_drop_into_producer(graph, i, slot, wire)
        if merged is not None:
            return merged
    producer = _producer_of(graph, wire)
    if readers[wire] > 1 and producer is not None:
        return _copy_at_producer(graph, i, producer, slot, array, wire)
    # The identity case, where the wire comes from outside this scope and the
    # drop is its only reader, or the scope is a block's body whose codomain
    # carries the wire, which a sibling rewrite may not change.
    passes_through = not is_outermost and any(obj == wire for obj in graph.cod)
    if readers[wire] == 1 or passes_through:
        root = hg.HypergraphRoot.template(_wrap_over_identity(drop.wraps), drop.dom, ())
        return _rebuild(graph, {i: root})
    # The copy of a wire the scope reads on its domain, with one output passed
    # on and the other dropped. Every other reader of the wire is redirected
    # onto the new output node, whether it is a sibling root, a block or the
    # graph's own codomain. In a block's body the wire is not in the codomain,
    # so the body's ports are untouched and the copy stands inside the block.
    root = hg.HypergraphRoot.template(_copy_wrap(array, slot), (wire,))
    passed = root.cod[0]
    redirect = fd.Context([fd.UIDRenaming.set_canonical(passed, wire)])
    replacements: dict[int, hg.Hypergraph | None] = {i: root}
    for k, subgraph in enumerate(graph.subgraphs()):
        if k != i and wire in subgraph.dom:
            replacements[k] = redirect.apply(subgraph)
    cod = tuple(redirect.apply(obj) for obj in graph.cod)
    return _rebuild(graph, replacements, cod)


def _copy_wrap(array: object, slot: Para.NamedEntry) -> ParaWrap:
    '''The copy of one wire with its second output dropped onto `slot`.'''
    return ParaWrap(
        body=cat.Rearrangement(mapping=(0, 0), _dom=(array,)),
        grabs=(None,),
        drops=(None, slot))


def _producer_of(graph: hg.Multigraph, wire: hg.HypergraphObject) -> int | None:
    '''The index of the sibling, a root or a block, that produces `wire`, or
    `None` when the wire arrives on the scope's domain.'''
    for k, subgraph in enumerate(graph.subgraphs()):
        if any(obj == wire for obj in subgraph.cod):
            return k
    return None


def _copy_at_producer(
    graph: hg.Multigraph, i: int, producer: int,
    slot: Para.NamedEntry, array: object, wire: hg.HypergraphObject,
) -> hg.Hypergraph:
    '''The drop at `i` as a copy at the output of the sibling at `producer`.

    The producer is renamed to write a fresh wire, and the copy reads it and
    writes the wire every other reader names, so the readers, the blocks among
    them and the scope's own codomain are untouched. A producer that is a
    block is renamed throughout, which keeps its body and its codomain
    agreeing.
    '''
    source = hg.HypergraphObject()
    renamed = fd.Context([fd.UIDRenaming.set_canonical(source, wire)]).apply(
        graph.subgraphs()[producer])
    root = hg.HypergraphRoot.template(_copy_wrap(array, slot), (source,), (wire,))
    return _rebuild(graph, {i: root, producer: renamed})


def _merge_drop_into_producer(
    graph: hg.Multigraph, i: int, slot: Para.NamedEntry,
    wire: hg.HypergraphObject,
) -> hg.Hypergraph | None:
    '''The drop at `i` written onto the sibling seed that produces its wire.

    `None` where no sibling root produces the wire at exactly one result, which
    is the case for a wire arriving from a block or from the scope's own
    domain. A producer reindexing beyond a rearrangement is passed over as
    well.
    '''
    for j, producer in enumerate(graph.subgraphs()):
        if j == i or not isinstance(producer, hg.HypergraphRoot):
            continue
        if isinstance(producer.wraps, (Para.Grab, Para.Drop)):
            continue
        if _has_non_rearrangement_reindexing(producer.wraps):
            continue
        ports = [p for p, node in enumerate(producer.cod) if node == wire]
        if len(ports) != 1:
            continue
        port = ports[0]
        root = hg.HypergraphRoot.template(
            _with_drop(producer.wraps, port, slot),
            producer.dom,
            (*producer.cod[:port], *producer.cod[port + 1:]))
        return _rebuild(graph, {i: None, j: root})
    return None
