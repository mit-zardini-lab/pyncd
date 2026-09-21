'''Unpacking a compressed selection into an explicit gather.

A `SparseAxis` compresses two wires into one: the surviving values on a
genuinely k-sized axis, and `Natural(n)` indices on that axis saying which of
the n each value came from. The two are the pair `torch.topk` returns.

The compressed form is what the notebooks under `notebooks/sota/` draw, using
one wire instead of two through every consumer. The expanded form is what
`example_notebooks/DeepSeekV3.ipynb` builds by hand, in its section on writing the
mixture of experts out.

This module rewrites between the two. `expand_sparse` takes a morphism, or a
hypergraph, written in the compressed form and unpacks every selection by
manipulating the graph. It is not a local splice, because every consumer of the
sparse axis changes shape, so the rewrite works on the hypergraph, where the new
index wire can be routed from the `TopK` to each consumer by node identity,
extending a block's domain and codomain as it crosses them.

`obsidian/02-categories/Sparse Expansion.md` is the full account.

The rules, with S the sparse axis, k its fresh dense replacement and n its
parent extent:

    TopK      [R; e] -> [R; S]           becomes  [R; e] -> [R; k], [Nat(e); k]

The second output is the index wire, routed through the graph to every consumer
below. The operator's form moves from `WEIGHTS` to `WEIGHTS_SELECT`, per
`ds.SelectionForm`. A `TopK` already in a dense form holds no sparse axis, so the
pass leaves it alone.

    Linear    W: X -> [Y; S]             becomes  W: [Nat(e); k], X -> [Y; k]

The sparse axis leaves the weight's target and becomes a broadcast axis in the
degree. The index enters as a rank-0 target operand, so the implicit weight
tensor is indexed by the chosen expert.

If S was already in the degree, which happens in the idiom where a projection
produces the axis and a diagonal selects from it, so that W^D at (x, S) produces
(S, m), the two collapse into one k. The collapse is what the diagonal asserted,
so the diagonal downstream becomes an identity and is spliced out.

    Select    [R; S], [R; e] -> [R; S]   becomes  hold[R; k]
                                                  x IndexSelect([Nat(e); k], [R; e] -> [R; k])
                                                  ; einops('k, k -> k')

The held selection values are multiplied back onto the gathered payload.

    TopK      [R; C], [Nat(B); C] -> [R; S]   becomes  [R; C] -> [R; k], [Nat(C); k]
                                                  ; hold[R; k]
                                                  x IndexSelect([Nat(C); k], [Nat(B); C] -> [Nat(B); k])

A selection over entries whose positions in a wider axis arrive as an operand,
per `ds.TopK.template(positions_of=)`, hands out a sparse axis over the wider
one. Its index wire is the operand read at the slots it chose.

For every other operator, S is substituted by k in weaves and reindexings alike.
The reindexings keep their mapping and strides, with only the axis object
swapped, so everything affine stays intact.

One repair is needed. A reindexing that copied the sparse axis, meaning one
whose mapping references its position twice, which is the diagonal, loses the
copies, because its producer no longer duplicates the axis.

The index wire binds each consumer to the `TopK` of the same expression. A
`cat.Block` gains the wire on its domain and codomain as the wire crosses it.
A `BlockOperator` whose
body holds a selection is expanded recursively, provided the selection is
self-contained inside it, meaning no sparse axis appears in the body's domain or
codomain. A sparse axis with no compressed `TopK` in scope is left alone,
because there is nothing to expand it with.
'''

from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Any

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import term_utilities.term_utilities as tutil
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import graphs.processing.hypergraph_functor as functor
import construction_helpers.lift as chl
from construction_helpers import simple_helper as chsh
import advanced_axis_dynamics.data_structure.Operators as aops

import deepseek.data_structure as ds


# ---------------------------------------------------------------------------
# The index wire of one selection
# ---------------------------------------------------------------------------

@dataclass
class SparseWire:
    '''Everything one selection needs to expand: the sparse axis, the dense
    k axis replacing it, the wire the index travels on, and (once the TopK is
    seen) the index array as the TopK emits it.'''
    sparse: ds.SparseAxis
    k_axis: cat.RawAxis
    node: hg.HypergraphObject = field(default_factory=hg.HypergraphObject)
    array: cat.Array | None = None      # set at the producing TopK

    @classmethod
    def template(cls, sparse: ds.SparseAxis) -> SparseWire:
        # The k axis takes the activity as its size, which is the same symbol
        # the TopK operator's `k` field carries, so one NumericConfig
        # assignment reaches both. It also takes the activity's name, such as
        # 'k' or 's'.
        k_axis = ds.selected_axis_name(sparse).capture(
            cat.RawAxis(_size=sparse.activity))
        return cls(sparse=sparse, k_axis=k_axis)

    def natural(self) -> cat.Natural:
        '''The index datatype: which of the parent's n.'''
        return cat.Natural(self.sparse._size)

    def hyper_object(self) -> hg.HypergraphObject[Any]:
        assert self.array is not None, (
            f'sparse axis {self.sparse.uid.to_latex()} is consumed but no '
            'compressed TopK produces it in this expression')
        return self.node.reconstruct(obj=self.array)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _target_sparse(weaves: fd.Prod[cat.Weave]) -> ds.SparseAxis | None:
    for weave in weaves:
        for axis in weave.target().shape():
            if isinstance(axis, ds.SparseAxis):
                return axis
    return None


def classify(target: cat.Broadcasted) -> tuple[str, ds.SparseAxis | None]:
    '''What role this seed morphism plays in the expansion.'''
    operator = target.operator
    if isinstance(operator, ops.BlockOperator):
        if any(True for _ in tutil.type_search(ds.SparseAxis, operator.block)):
            return 'block_operator', None
        return 'generic', None
    if isinstance(operator, aops.CovariantView):
        produced = _target_sparse(target.output_weaves)
        if produced is not None and _target_sparse(target.input_weaves) is not None:
            return 'merge', produced
        return 'generic', None
    if isinstance(operator, ds.TopK):
        if (operator.form is ds.SelectionForm.WEIGHTS
                and (sparse := _target_sparse(target.output_weaves)) is not None):
            if _target_sparse(target.input_weaves) is not None:
                return 'topk_over_sparse', sparse
            if ds.selects_over_positions(target):
                return 'topk_over_positions', sparse
            return 'topk', sparse
        return 'generic', None
    if isinstance(operator, ops.Linear):
        if (sparse := _target_sparse(target.output_weaves)) is not None:
            return 'linear', sparse
        return 'generic', None
    if isinstance(operator, ds.Select):
        if (sparse := _target_sparse(target.input_weaves[:1])) is not None:
            return 'select', sparse
        return 'generic', None
    return 'generic', None


def _walk_roots(graph: hg.Hypergraph):
    match graph:
        case hg.HypergraphRoot():
            yield graph
        case hg.HypergraphBlock(body=body):
            yield from _walk_roots(body)
        case hg.Multigraph():
            for subgraph in graph.subgraphs():
                yield from _walk_roots(subgraph)


# ---------------------------------------------------------------------------
# The diagonal repair
# ---------------------------------------------------------------------------

def collapse_copies(
    target: cat.Broadcasted,
    k_axes: set[cat.RawAxis],
) -> cat.Broadcasted:
    '''Drop repeated references to an expanded axis from input reindexings.

    A reindexing that copies the sparse axis, which is the diagonal with
    mapping (0, 1, 1, 2), exists to relate the slot axis to the produced
    expert axis. The expanded `Linear` produces only one k, so the copy has
    nothing to refer to. Keep the first reference, drop the rest, and drop the
    matching TILED slots from the weave.

    This applies only to axes this expansion created, so legitimate copies of
    other axes are untouched.
    '''
    new_weaves, new_reindexings = [], []
    changed = False
    for weave, reindexing in zip(target.input_weaves, target.reindexings):
        if not isinstance(reindexing, cat.Rearrangement):
            new_weaves.append(weave); new_reindexings.append(reindexing)
            continue
        seen: set[int] = set()
        keep_cod: list[bool] = []
        for dom_idx in reindexing.mapping:
            duplicate = (dom_idx in seen
                         and reindexing._dom[dom_idx] in k_axes)
            keep_cod.append(not duplicate)
            seen.add(dom_idx)
        if all(keep_cod):
            new_weaves.append(weave); new_reindexings.append(reindexing)
            continue
        changed = True
        new_mapping = tuple(
            dom_idx for dom_idx, keep in zip(reindexing.mapping, keep_cod)
            if keep)
        # cod position j is the j-th TILED slot of the weave, in order.
        keep_iter = iter(keep_cod)
        new_shape = tuple(
            entry for entry in weave._shape
            if not isinstance(entry, cat.WeaveMode) or next(keep_iter))
        new_weaves.append(cat.Weave(weave.datatype, new_shape))
        new_reindexings.append(
            cat.Rearrangement(new_mapping, reindexing._dom))
    if not changed:
        return target
    return target.reconstruct(
        input_weaves=tuple(new_weaves),
        reindexings=tuple(new_reindexings))


# ---------------------------------------------------------------------------
# The root rewrites
# ---------------------------------------------------------------------------

def expand_topk(
    target: cat.Broadcasted,
    wire: SparseWire,
) -> cat.Broadcasted:
    '''[R; e] -> [R; S]  becomes  [R; e] -> [R; k], [Nat(e); k], which is the
    `WEIGHTS_SELECT` form of `ds.SelectionForm`.'''
    values_weave = target.output_weaves[0]
    index_weave = cat.Weave(wire.natural(), values_weave._shape)
    return target.reconstruct(
        operator=target.operator.reconstruct(
            form=ds.SelectionForm.WEIGHTS_SELECT),
        output_weaves=(*target.output_weaves, index_weave))


def expand_linear(
    target: cat.Broadcasted,
    wire: SparseWire,
) -> cat.Broadcasted:
    '''W: X -> [Y; k of e]  becomes  W: [Nat(e); k], X -> [Y; k].

    `target` arrives already substituted, so the sparse axis reads as `k`
    here. This function moves it out of the output target into the degree, and
    gives the operator the index operand.
    '''
    k = wire.k_axis
    degree = tuple(target.degree())
    extend = k not in degree
    new_degree = (*degree, k) if extend else degree

    new_reindexings = []
    for reindexing in target.reindexings:
        if not extend:
            new_reindexings.append(reindexing)
        elif isinstance(reindexing, cat.Rearrangement):
            new_reindexings.append(cat.Rearrangement(
                reindexing.mapping, new_degree))
        else:
            raise NotImplementedError(
                'sparse expansion: cannot extend a non-Rearrangement '
                f'reindexing on Linear {target.operator.name}')

    new_output_weaves = []
    for weave in target.output_weaves:
        new_shape: list[cat.Axis | cat.WeaveMode] = []
        tiles_before = 0
        replaced = not extend       # when k is already a degree axis, every
        for entry in weave._shape:  # target occurrence simply drops
            if isinstance(entry, cat.WeaveMode):
                tiles_before += 1
                new_shape.append(entry)
            elif entry == k:
                if not replaced:
                    assert tiles_before == len(degree), (
                        'sparse expansion: the sparse target must follow '
                        'every broadcast slot of the weave')
                    new_shape.append(cat.WeaveMode.TILED)
                    replaced = True
                # further occurrences drop: one k is the whole statement
            else:
                new_shape.append(entry)
        new_output_weaves.append(cat.Weave(weave.datatype, tuple(new_shape)))

    # The index operand goes FIRST, as in the V3 complete form: the drawn
    # box shows the selection on top, and the weight's own operands after.
    index_weave = cat.Weave(
        wire.natural(),
        (cat.WeaveMode.TILED,) * len(wire.array.shape()))  # type: ignore
    index_reindexing = cat.Rearrangement(
        tuple(new_degree.index(axis) for axis in wire.array.shape()),  # type: ignore
        new_degree)

    return target.reconstruct(
        input_weaves=(index_weave, *target.input_weaves),
        output_weaves=tuple(new_output_weaves),
        reindexings=(index_reindexing, *new_reindexings))


def expand_select(
    target: cat.Broadcasted,
    wire: SparseWire,
) -> cat.BroadcastedCategory:
    '''Select: [R; S], [R; e] -> [R; S]  becomes
    hold[R; k] x IndexSelect([Nat(e)], [R; e] -> [R]) ; einops('k, k -> k').

    The `IndexSelect` is elementwise in the index. Its degree is the whole
    output array, which for CSA is (x, k, c), with every surviving axis
    broadcast. Its targets are the rank-0 `Nat(e)` index and the payload's
    parent axis, and the index reindexing is the identity prefix reading the
    index wire at its own axes.

    That mirrors how an expanded `Linear` takes its index, and it keeps k a
    broadcast axis everywhere rather than a target the gather consumes and
    re-emits.

    Returns a composite morphism whose domain is the values, the index and the
    payload. The caller wires the index slot to the `TopK`'s index node.
    '''
    values_weave, payload_weave = target.input_weaves
    output_weave = target.output_weaves[0]
    values_reindexing, payload_reindexing = target.reindexings
    degree = target.degree()

    values_array = values_weave.imprint_to_degree(values_reindexing.cod())
    output_array = output_weave.imprint_to_degree(degree)
    new_degree = tuple(output_array.shape())

    # The index wire, read at its own axes: an identity-prefix map into the
    # new degree, with a rank-0 Nat(e) target.
    index_axes = tuple(values_array.shape())
    index_weave = cat.Weave(
        wire.natural(), (cat.WeaveMode.TILED,) * len(index_axes))
    index_reindexing = cat.Rearrangement(
        tuple(new_degree.index(axis) for axis in index_axes), new_degree)

    # The payload keeps its parent-axis target, and its broadcast slots re-read
    # off the new degree (in CSA, just c).
    new_payload_reindexing = cat.Rearrangement(
        tuple(new_degree.index(axis) for axis in payload_reindexing.cod()),
        new_degree)

    grab = cat.Broadcasted(
        operator=ds.IndexSelect(),
        input_weaves=(index_weave, payload_weave),
        output_weaves=(
            cat.Weave(output_array.datatype,
                      (cat.WeaveMode.TILED,) * len(new_degree)),),
        reindexings=(index_reindexing, new_payload_reindexing))
    # einops('k, k -> k'): elementwise product of the held selection values
    # with the gathered payload, broadcast over the output's whole shape.
    product = cat.Broadcasted(
        operator=ops.Einops(signature=((), ())),
        input_weaves=(
            cat.Weave(values_array.datatype,
                      (cat.WeaveMode.TILED,) * len(values_array.shape())),
            cat.Weave(output_array.datatype,
                      (cat.WeaveMode.TILED,) * len(output_array.shape()))),
        output_weaves=(
            cat.Weave(output_array.datatype,
                      (cat.WeaveMode.TILED,) * len(output_array.shape())),),
        reindexings=(
            cat.Rearrangement(
                tuple(tuple(output_array.shape()).index(axis)
                      for axis in values_array.shape()),
                tuple(output_array.shape())),
            output_array.shape().identity()))

    hold = cat.ProdObject((values_array,)).identity()
    return chsh.make_composed(chsh.make_product(hold, grab), product)


# ---------------------------------------------------------------------------
# A selection merged into a wider axis, and a selection over a selection
# ---------------------------------------------------------------------------

PRODUCERS = ('topk', 'topk_over_positions', 'merge', 'topk_over_sparse')
CONSUMERS = ('linear', 'select')
CHAINED = ('merge', 'topk_over_sparse')


def consumed_sparse(target: cat.Broadcasted) -> ds.SparseAxis | None:
    '''The sparse axis on an input target of `target`, or None.'''
    return _target_sparse(target.input_weaves)


def dense_reindexing(
    reindexing: cat.StrideMorphism,
    dom_axes: fd.Prod[cat.Axis],
    cod_axis: cat.Axis,
) -> cat.StrideMorphism:
    '''`reindexing` between other axes, keeping its strides and shift: the same
    split between the dense replacements of the axes it split.'''
    _, strides, shift = reindexing._cod_stride_shift[0]
    return cat.StrideMorphism(
        _dom=tuple(dom_axes),
        _cod_stride_shift=((cod_axis, strides, shift),),
        name=reindexing.name)


def expand_merge(
    target: cat.Broadcasted,
    consumed: SparseWire,
    produced: SparseWire,
) -> cat.BroadcastedCategory:
    '''aops.CovariantView : [R; p/P, u] -> [R; c/B]  becomes the product of

        aops.CovariantView : [R; p, u] -> [R; c]

    for the values and, for the index wire of the merged axis,

        MergedPositions : [Nat(P); p] -> [Nat(B); p, u]
        ; aops.CovariantView : [Nat(B); p, u] -> [Nat(B); c].

    The merged axis is a selection whose positions are the split applied to the
    block numbers at every offset. `target` arrives substituted, so the sparse axes
    read as `p` and `c`. Returns a morphism whose domain is the values and the index
    of the split axis, and whose codomain is the merged values and the index of the
    merged axis.
    '''
    operator: aops.CovariantView = target.operator
    split = operator.reindexing
    values_weave, = target.input_weaves
    merged_weave, = target.output_weaves
    dom_axes = tuple(values_weave.target().shape())
    merged_axis, = tuple(merged_weave.target().shape())
    selected_position = dom_axes.index(consumed.k_axis)
    values = target.reconstruct(operator=operator.reconstruct(
        reindexing=dense_reindexing(split, dom_axes, merged_axis)))
    positions = ds.merge_selected_positions(
        split, consumed.k_axis, target.degree(), merged_axis, selected_position)
    return chsh.make_product(values, positions)


def select_then_compose(
    topk: cat.Broadcasted,
    inner_axis: cat.Axis,
    positions: cat.Array,
    produced: SparseWire,
) -> cat.BroadcastedCategory:
    '''`topk`, a selection over `inner_axis` in the `WEIGHTS_SELECT` form, followed
    by the `IndexSelect` reading `positions`, the positions of the parent that the
    entries of `inner_axis` sit at, at the slots the selection chose. Returns a
    morphism whose domain is the scores and `positions`, and whose codomain is the
    values and the positions of the parent that were chosen.'''
    T = cat.WeaveMode.TILED
    inner = cat.Natural(inner_axis.local_size())
    degree = tuple(topk.degree())
    values_array, inner_index = tuple(topk.cod())
    selected = tuple(inner_index.shape())
    compose = cat.Broadcasted(
        operator=ds.IndexSelect(),
        input_weaves=(cat.Weave(inner, (T,) * len(selected)),
                      cat.Weave(produced.natural(),
                                (*(T,) * len(degree), inner_axis))),
        output_weaves=(cat.Weave(produced.natural(), (T,) * len(selected)),),
        reindexings=(cat.ProdObject(selected).identity(),
                     cat.Rearrangement(tuple(range(len(degree))), selected)))
    return chsh.make_composed(
        chsh.make_product(topk, cat.ProdObject((positions,)).identity()),
        chsh.make_product(cat.ProdObject((values_array,)).identity(), compose))


def expand_topk_over_sparse(
    target: cat.Broadcasted,
    consumed: SparseWire,
    produced: SparseWire,
) -> cat.BroadcastedCategory:
    '''TopK : [R; c/B] -> [R; s/B]  becomes

        TopK : [R; c] -> [R; s], [Nat(c); s]
        ; hold [R; s] x IndexSelect([Nat(c); s], [Nat(B); c] -> [Nat(B); s]).

    A selection over an axis that is itself a selection hands out positions of the
    inner axis, and the positions of the parent are the inner selection's positions
    read at them. `target` arrives substituted. Returns a morphism whose domain is
    the scores and the inner index, and whose codomain is the values and the outer
    index.
    '''
    values_weave = target.output_weaves[0]
    inner = cat.Natural(consumed.k_axis.local_size())
    topk = target.reconstruct(
        operator=target.operator.reconstruct(form=ds.SelectionForm.WEIGHTS_SELECT),
        output_weaves=(values_weave, cat.Weave(inner, values_weave._shape)))
    return select_then_compose(topk, consumed.k_axis, consumed.array, produced)


def expand_topk_over_positions(
    target: cat.Broadcasted,
    produced: SparseWire,
) -> cat.BroadcastedCategory:
    '''TopK : [R; C], [Nat(B); C] -> [R; s/B]  becomes

        TopK : [R; C] -> [R; s], [Nat(C); s]
        ; hold [R; s] x IndexSelect([Nat(C); s], [Nat(B); C] -> [Nat(B); s]).

    A selection over entries whose positions arrive as an operand, per
    `ds.TopK.template(positions_of=)`, is the selection over a selection with the
    inner positions on a wire of the model instead of on the index wire of a sparse
    axis. `target` arrives substituted. Returns a morphism whose domain is the
    scores and the positions, and whose codomain is the values and the outer index.
    '''
    scores_weave, positions_weave = target.input_weaves
    scores_reindexing, positions_reindexing = target.reindexings
    inner_axis, = tuple(scores_weave.target().shape())
    positions = positions_weave.imprint_to_degree(positions_reindexing.cod())
    values_weave = target.output_weaves[0]
    topk = target.reconstruct(
        operator=target.operator.reconstruct(form=ds.SelectionForm.WEIGHTS_SELECT),
        input_weaves=(scores_weave,),
        reindexings=(scores_reindexing,),
        output_weaves=(values_weave,
                       cat.Weave(cat.Natural(inner_axis.local_size()),
                                 values_weave._shape)))
    return select_then_compose(topk, inner_axis, positions, produced)


# ---------------------------------------------------------------------------
# The traversal
# ---------------------------------------------------------------------------

@dataclass
class _Expansion:
    wires: dict[fd.UID, SparseWire] = field(default_factory=dict)
    kinds: dict[int, tuple[str, ds.SparseAxis | None]] = field(default_factory=dict)
    scopes: dict[int, tuple[set[fd.UID], Counter]] = field(default_factory=dict)
    substitute: fd.Context = field(default_factory=fd.Context)
    splices: fd.Context = field(default_factory=fd.Context)
    total_consumed: Counter = field(default_factory=Counter)

    # -- phase 0: find the selections, mint their wires -------------------
    def survey(self, graph: hg.Hypergraph) -> None:
        roots = tuple(_walk_roots(graph))
        for root in roots:
            self.kinds[id(root)] = classify(root.wraps)
        for root in roots:
            kind, sparse = self.kinds[id(root)]
            if kind in PRODUCERS and sparse.uid not in self.wires:
                self.wires[sparse.uid] = SparseWire.template(sparse)
        self._drop_chains_without_a_source(roots)
        self.substitute = fd.Context([
            fd.EqualityClass(
                _type=ds.SparseAxis,
                bucket={wire.sparse.uid},
                canonical=wire.k_axis)
            for wire in self.wires.values()
        ])
        # A consumer whose sparse axis has no TopK in scope stays compressed.
        for root in roots:
            kind, sparse = self.kinds[id(root)]
            if kind in CONSUMERS and sparse.uid not in self.wires:
                self.kinds[id(root)] = ('generic', None)
        # The index array, as the producing node emits it.
        for root in roots:
            kind, sparse = self.kinds[id(root)]
            if kind in PRODUCERS:
                wire = self.wires[sparse.uid]
                assert wire.array is None, (
                    f'two nodes produce {sparse.uid.to_latex()} in one scope, '
                    'which would make the index wire ambiguous')
                substituted = self.substitute.apply(root.wraps)
                wire.array = substituted.output_weaves[0].imprint_to_degree(
                    substituted.degree())
                wire.array = cat.Array(wire.natural(),
                                       tuple(wire.array.shape()))

    def _drop_chains_without_a_source(self, roots) -> None:
        '''A merge, or a selection over a selection, whose own input axis has no
        producer in scope stays compressed, and so does everything produced from
        it.'''
        changed = True
        while changed:
            changed = False
            for root in roots:
                kind, sparse = self.kinds[id(root)]
                if kind in CHAINED and sparse.uid in self.wires and (
                        consumed_sparse(root.wraps).uid not in self.wires):
                    del self.wires[sparse.uid]
                    self.kinds[id(root)] = ('generic', None)
                    changed = True

    # -- phase 1: who produces and consumes, per subtree ------------------
    def analyze(self, graph: hg.Hypergraph) -> tuple[set[fd.UID], Counter]:
        produced: set[fd.UID] = set()
        consumed: Counter = Counter()
        match graph:
            case hg.HypergraphRoot():
                kind, sparse = self.kinds[id(graph)]
                if kind in PRODUCERS:
                    produced.add(sparse.uid)
                if kind in CONSUMERS:
                    consumed[sparse.uid] += 1
                if kind in CHAINED:
                    consumed[consumed_sparse(graph.wraps).uid] += 1
            case hg.HypergraphBlock(body=body):
                sub_produced, sub_consumed = self.analyze(body)
                produced |= sub_produced
                consumed += sub_consumed
            case hg.Multigraph():
                for subgraph in graph.subgraphs():
                    sub_produced, sub_consumed = self.analyze(subgraph)
                    produced |= sub_produced
                    consumed += sub_consumed
        self.scopes[id(graph)] = (produced, consumed)
        return produced, consumed

    # -- phase 2: rebuild -------------------------------------------------
    def _sorted_wires(self, uids) -> tuple[SparseWire, ...]:
        return tuple(sorted(
            (self.wires[uid] for uid in uids),
            key=lambda wire: wire.sparse.uid._id))

    def _substituted(self, objects):
        return tuple(self.substitute.apply(obj) for obj in objects)

    def _index_wires_first(self, objects, extra_wires):
        '''For a domain: the index wires ahead of the substituted objects.

        A block that imports a selection then shows it as its first input, the
        way an expanded `Linear` takes it as its first operand.
        '''
        wires = tuple(wire.hyper_object() for wire in extra_wires)
        return (*wires, *self._substituted(objects))

    def _index_wires_last(self, objects, extra_wires):
        '''For a codomain: the index wires after the substituted objects.'''
        wires = tuple(wire.hyper_object() for wire in extra_wires)
        return (*self._substituted(objects), *wires)

    def _rebuild(self, graph: hg.Hypergraph) -> hg.Hypergraph:
        match graph:
            case hg.HypergraphRoot():
                return self._rebuild_root(graph)
            case hg.HypergraphBlock(body=body):
                if self._crossing(graph) and graph.block_tag.repetition != nm.Integer(1):
                    raise NotImplementedError(
                        'sparse expansion: an index wire would cross into a '
                        'repeated block')
                new_body = self._rebuild(body)
                return hg.HypergraphBlock.template(new_body, graph.block_tag)
            case hg.Multigraph():
                new_subgraphs = tuple(
                    rebuilt
                    for subgraph in graph._subgraphs
                    if (rebuilt := self._rebuild(subgraph)) is not None)
                produced, consumed = self.scopes[id(graph)]
                enters = self._sorted_wires(
                    uid for uid in consumed if uid not in produced)
                exits = self._sorted_wires(
                    uid for uid in produced
                    if self.total_consumed[uid] > consumed[uid])
                new_dom = self._index_wires_first(graph.dom, enters)
                new_cod = self._index_wires_last(graph.cod, exits)
                return hg.Multigraph.template(new_dom, new_cod, new_subgraphs)
        raise NotImplementedError(f'cannot rebuild {type(graph).__name__}')

    def _crossing(self, graph: hg.Hypergraph) -> bool:
        produced, consumed = self.scopes[id(graph)]
        entering = any(uid not in produced for uid in consumed)
        exiting = any(self.total_consumed[uid] > consumed[uid]
                      for uid in produced)
        return entering or exiting

    def _rebuild_root(self, root: hg.HypergraphRoot) -> hg.Hypergraph:
        kind, sparse = self.kinds[id(root)]
        wraps = root.wraps
        match kind:
            case 'topk':
                wire = self.wires[sparse.uid]
                new_wraps = expand_topk(self.substitute.apply(wraps), wire)
                return hg.HypergraphRoot.template(
                    new_wraps,
                    dom=root.dom,
                    cod=(*root.cod, wire.node))
            case 'linear':
                wire = self.wires[sparse.uid]
                new_wraps = expand_linear(self.substitute.apply(wraps), wire)
                return hg.HypergraphRoot.template(
                    new_wraps,
                    dom=(wire.node, *root.dom),
                    cod=root.cod)
            case 'select':
                wire = self.wires[sparse.uid]
                replacement = expand_select(
                    self.substitute.apply(wraps), wire)
                values_node, payload_node = root.dom
                return self._convert_onto(
                    replacement, (values_node, wire.node, payload_node), root.cod)
            case 'topk_over_positions':
                wire = self.wires[sparse.uid]
                replacement = expand_topk_over_positions(
                    self.substitute.apply(wraps), wire)
                return self._convert_onto(
                    replacement, root.dom, (*root.cod, wire.node))
            case 'merge' | 'topk_over_sparse':
                return self._rebuild_chained(root, kind, sparse)
            case 'block_operator':
                return self._rebuild_block_operator(root)
        # generic: substitute the axis, repair copies of it, and, should the
        # node collapse to a pure identity (the diagonal) - splice it out.
        new_wraps = self.substitute.apply(wraps)
        if isinstance(new_wraps, cat.Broadcasted):
            new_wraps = collapse_copies(
                new_wraps,
                {wire.k_axis for wire in self.wires.values()})
            if (ops.is_identity(new_wraps)
                    and len(root.dom) == 1 and len(root.cod) == 1):
                # The node became a pure identity (the diagonal, after its
                # copies collapsed): splice it out by making its output node
                # an alias of its input node.
                self.splices.append_bucket(fd.UIDRenaming.set_canonical(
                    root.dom[0], root.cod[0]))
                return None
        if new_wraps is wraps:
            return root
        return hg.HypergraphRoot.template(
            new_wraps, dom=root.dom, cod=root.cod)

    def _rebuild_chained(
        self,
        root: hg.HypergraphRoot,
        kind: str,
        produced_axis: ds.SparseAxis,
    ) -> hg.Hypergraph:
        '''A merge, or a selection over a selection: the index wire of the axis it
        consumes enters, the index wire of the axis it produces leaves, and the
        composite it expands to is converted onto the root's own nodes.'''
        consumed = self.wires[consumed_sparse(root.wraps).uid]
        produced = self.wires[produced_axis.uid]
        rule = expand_merge if kind == 'merge' else expand_topk_over_sparse
        replacement = rule(self.substitute.apply(root.wraps), consumed, produced)
        values_node, = root.dom
        return self._convert_onto(
            replacement, (values_node, consumed.node), (*root.cod, produced.node))

    def _convert_onto(
        self,
        replacement: cat.BroadcastedCategory,
        dom_nodes: fd.Prod[hg.HypergraphObject],
        cod_nodes: fd.Prod[hg.HypergraphObject],
    ) -> hg.Hypergraph:
        '''`replacement` as a graph reading `dom_nodes`, with the wires it produces
        renamed onto `cod_nodes`, so that the consumers of the node it replaces
        read from it.'''
        as_graph = hg.Multigraph.from_morphism(
            replacement,
            dom=hg.HypergraphObject.template(replacement.dom(), dom_nodes))
        self.splices.append_buckets(
            fd.UIDRenaming.set_canonical(new, old)
            for new, old in zip(as_graph.cod, cod_nodes))
        return as_graph

    def _rebuild_block_operator(self, root: hg.HypergraphRoot) -> hg.Hypergraph:
        wraps = root.wraps
        operator: ops.BlockOperator = wraps.operator
        boundary = (*wraps.input_weaves, *wraps.output_weaves)
        assert _target_sparse(boundary) is None and not any(
            isinstance(axis, ds.SparseAxis)
            for weave in boundary for axis in weave._shape
            if isinstance(axis, cat.Axis)), (
            'sparse expansion: a sparse axis crosses a BlockOperator '
            f'boundary ({operator.name}); expand before boxing, or use a '
            'cat.Block, whose domain and codomain the index wire can extend')
        new_block = expand_sparse(operator.block)
        if new_block is operator.block:
            return root
        new_wraps = wraps.reconstruct(
            operator=operator.reconstruct(block=new_block))
        return hg.HypergraphRoot.template(
            new_wraps, dom=root.dom, cod=root.cod)

    def run(self, graph: hg.Hypergraph) -> hg.Hypergraph:
        self.survey(graph)
        if not self.wires and not any(
                kind == 'block_operator' for kind, _ in self.kinds.values()):
            return graph
        _, self.total_consumed = self.analyze(graph)
        produced, consumed = self.scopes[id(graph)]
        unmet = [uid for uid in consumed if uid not in produced]
        assert not unmet, (
            'sparse axes consumed with no TopK in scope: '
            + ', '.join(uid.to_latex() for uid in unmet))
        new_graph = self._rebuild(graph)
        if self.splices.equality_classes:
            new_graph = functor.reduce_nodes(self.splices, new_graph)
        return new_graph


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def expand_sparse[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M] | hg.Hypergraph[L, M],
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M]:
    '''Expand every compressed selection in `target`.

    Accepts a morphism or a hypergraph and returns the same kind. A target
    with no sparse content comes back unchanged (the same object).'''
    if not any(True for _ in tutil.type_search(ds.SparseAxis, target)):
        return target
    as_morphism = not isinstance(target, hg.Hypergraph)
    graph = (hg.Multigraph.from_morphism(target)
             if as_morphism else target)
    new_graph = _Expansion().run(graph)
    if new_graph is graph and as_morphism:
        return target
    return h2m.hypergraph_to_morphism(new_graph) if as_morphism else new_graph
