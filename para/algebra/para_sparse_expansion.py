'''Expanding a selection onto the tape, with nothing wired between the pieces.

`deepseek.sparse_expansion.expand_sparse` unpacks a compressed selection and
routes the index as a wire. Every consumer of the sparse axis gains an operand,
and the wire has to reach each of them, so the rewrite extends the domain and
the codomain of every block it crosses and works on the hypergraph in three
phases.

This module unpacks the same selection onto the tape. The rules are:

    TopK        the index output is dropped to a slot
    a parametric operation
                the index is grabbed from that slot as an additional input
    Select      the gather becomes a held value beside an `IndexSelect` that
                grabs the index from the slot, then the product of the two
    a merge of a selection, an `aops.CovariantView` over a sparse axis
                the index of the split axis is grabbed, the positions of the
                merged axis are computed from it, and they are dropped to the
                merged axis's slot
    a selection over a selection
                the index of the inner axis is grabbed, the outer positions
                are read at the selected inner ones, and they are dropped
    a selection over entries whose positions arrive as an operand
                the positions are read at the selected entries and dropped,
                as a `TopK`'s index is
    a BlockOperator holding a selection
                its body takes the same pass and its weaves the substitution,
                because the tape crosses the box without a wire
    the axis    `k/n` is substituted by a genuinely `k`-sized axis, in every
                weave, reindexing and tape operation that names it

`Grab` and `Drop` are composition-neutral, per
`obsidian/07-para/Para Category.md`, so the index enters no signature. Each
seed morphism is therefore rewritten on its own, no wire is routed, and no
block's domain or codomain moves. The result is an expression whose pieces are
independent, connected by the slot's name and by nothing else, and the pass is
an ordinary `functor.Endofunctor` for that reason.

`para.algebra.tie_tapes` connects them again. It gives every grab and drop of
one slot the same node, removes the tape operations, and puts the node on a
scope's domain or codomain wherever only one half of the slot is inside it. The
expression it produces is the one `expand_sparse` reaches by routing the wire
from the start, line for line.

`hypergraph_to_morphism` converts an expression this pass produces. It builds
each sink into a branch of its own, and places that branch in the column of the
rightmost branch that reads a wire of the sink's domain, per
`obsidian/03-hypergraphs/Hypergraph to Morphism.md`. Converting moves a drop to
the end of the scope holding it. A listing that reads better with the drop
beside the `TopK` writing it is therefore printed with `recycle=False`.

## Why the index belongs on a tape

`obsidian/07-para/Selection and the Reverse Pass.md` states that an index wire
is forward-only: it is saved, loaded, and never differentiated, because
`Natural` has no cotangent. `obsidian/07-para/Training.md` classifies a slot by
which pass writes it, and a value one pass writes and another reads with no wire
between them is what a slot is.

## The boxes the tape crosses

`expand_sparse` refuses a sparse axis on a `BlockOperator`'s domain or codomain,
because its index wire would have to extend the box. The tape crosses a box
without a wire, so a `TopK` inside one boxed layer and a `Select` inside another
expand here, each on its own, and the whole of the
DeepSeek-V4.1-Flash model in `notebooks/sota/DeepSeekV41Flash/`, whose layers are
boxed and whose selections leave them on a sparse axis, expands on the tape.

A morphism that already holds a `Grab` or a `Drop` is in Para, and this is the
pass that expands it. `expand_sparse` has no case for a tape operation.

Both passes read the seed rules from `deepseek.sparse_expansion`, meaning
`classify`, `expand_topk`, `expand_linear`, `expand_select`, `expand_merge`,
`expand_topk_over_sparse`, `expand_topk_over_positions` and `collapse_copies`.
The rules are what a selection does to one operator, and the routing is the
only thing that differs between the two passes.

`para` sits below `deepseek` in the layer order `CLAUDE.md` lists, and this
module imports upward. The alternative is to put a para-shaped pass inside
`deepseek/`, which splits the tape's rules across two features. Nothing imports
in the other direction from here, so there is no cycle.
'''
from __future__ import annotations
from dataclasses import dataclass, field

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Operators as ops
import deepseek.data_structure as ds
import deepseek.sparse_expansion as sparse_expansion
import graphs.processing.hypergraph_functor as functor
import para.data_structure.Para as Para
import para.data_structure.inject as inject
import para.data_structure.transpose as transpose
import term_utilities.term_utilities as tutil
from construction_helpers import simple_helper as chsh


class SelectionHasNoTapeExpansion(Exception):
    '''A selection whose consumers this pass has no rule for.'''


def reads_a_weight(operator: cat.Operator) -> bool:
    """True for a `Linear` and for the `Transpose` of one.

    A `Transpose` is built from the weight its forward `Linear` read, per
    `para/data_structure/transpose.py`, so a `Linear` that selects one of `n`
    weights reverses to a `Transpose` that reads the same slab. The index says
    which slab, and both operators need it.
    """
    if isinstance(operator, transpose.Transpose):
        return isinstance(operator.operator, ops.Linear)
    return isinstance(operator, ops.Linear)


def selecting_input(target: cat.Broadcasted) -> ds.SparseAxis | None:
    """The sparse axis a weight-reading operator consumes, or None.

    `deepseek.sparse_expansion.classify` reports a `Linear` whose *output*
    target carries the sparse axis, which is the up-projection idiom, written
    `m -> (e, f)`. A down-projection may be written the other way round, as
    `(k/e, f) -> m`, and the axis is then on an operand. The operator selects a
    weight either way, so it grabs the index either way. A `Transpose` reads
    the axis on an operand for the same reason its forward `Linear` produced
    it, which `reads_a_weight` is what admits.
    """
    if not reads_a_weight(target.operator):
        return None
    for weave in target.input_weaves:
        for axis in weave.target().shape():
            if isinstance(axis, ds.SparseAxis):
                return axis
    return None


def expand_selecting_input(
    target: cat.Broadcasted,
    wire: sparse_expansion.SparseWire,
) -> cat.Broadcasted:
    """`W: [X; k of n] -> Y` becomes `W: [Nat(n); k], [X; k] -> Y`.

    The mirror of `deepseek.sparse_expansion.expand_linear`, which moves the
    axis out of the *output* target and into the degree. Here the axis stays
    where it is, because a real input of a `Linear` is contracted and the
    operation sums over the selected slots. What the operator gains is the
    index, which says which `k` of the `n` weights the sum runs over.

    The index arrives on the same slots the sparse operand does, so it takes
    that operand's reindexing and its tiled positions, with the target cut down
    to the sparse axis alone. It is the first operand, as it is in
    `expand_linear` and in `deepseek.IndexSelect`.

    `target` arrives already substituted, so the sparse axis reads as `k`.
    """
    k = wire.k_axis
    for weave, reindexing in zip(target.input_weaves, target.reindexings):
        if k not in weave.target().shape():
            continue
        index_weave = cat.Weave(wire.natural(), tuple(
            entry for entry in weave._shape
            if isinstance(entry, cat.WeaveMode) or entry == k))
        return target.reconstruct(
            input_weaves=(index_weave, *target.input_weaves),
            reindexings=(reindexing, *target.reindexings))
    raise NotImplementedError(
        'para sparse expansion: no operand of '
        f'{target.operator.name} carries the sparse axis')


@dataclass
class ExpandSparseOntoTape[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every compressed selection in a morphism, with its index on the tape.

    `survey` runs first and has to see the whole expression, because a consumer
    is rewritten only where a `TopK` in the same expression produces its axis.
    A consumer whose sparse axis has no producer in scope is left compressed,
    which is the rule `expand_sparse` follows for the same reason.
    '''
    wires: dict[fd.UID, sparse_expansion.SparseWire] = field(
        default_factory=dict)
    slots: dict[fd.UID, Para.TapeSlot] = field(default_factory=dict)
    substitution: fd.Context = field(default_factory=fd.Context)

    def survey(self, target: cat.ProdCategory) -> None:
        '''Find the selections, mint a k axis and a slot for each.'''
        seeds = tuple(tutil.type_search(cat.Broadcasted, target))
        for seed in seeds:
            kind, sparse = sparse_expansion.classify(seed)
            if kind not in sparse_expansion.PRODUCERS or sparse.uid in self.wires:
                continue
            self.wires[sparse.uid] = sparse_expansion.SparseWire.template(sparse)
            self.slots[sparse.uid] = inject.selection_slot(sparse)
        self._drop_chains_without_a_source(seeds)
        self.substitution = fd.Context([
            fd.EqualityClass(_type=ds.SparseAxis,
                             bucket={wire.sparse.uid},
                             canonical=wire.k_axis)
            for wire in self.wires.values()
        ])
        for seed in seeds:
            kind, sparse = sparse_expansion.classify(seed)
            if kind in sparse_expansion.PRODUCERS and sparse.uid in self.wires:
                self._record_index_array(seed, self.wires[sparse.uid])

    def _drop_chains_without_a_source(self, seeds) -> None:
        '''A merge, or a selection over a selection, whose own input axis has no
        producer in scope stays compressed, and so does everything produced
        from it.'''
        changed = True
        while changed:
            changed = False
            for seed in seeds:
                kind, sparse = sparse_expansion.classify(seed)
                if kind in sparse_expansion.CHAINED and sparse.uid in self.wires and (
                        sparse_expansion.consumed_sparse(seed).uid not in self.wires):
                    del self.wires[sparse.uid]
                    del self.slots[sparse.uid]
                    changed = True

    def _record_index_array(
        self,
        topk: cat.Broadcasted,
        wire: sparse_expansion.SparseWire,
    ) -> None:
        '''The index array, at the degree the producing `TopK` emits it.'''
        substituted = self.substitution.apply(topk)
        values = substituted.output_weaves[0].imprint_to_degree(
            substituted.degree())
        wire.array = cat.Array(wire.natural(), tuple(values.shape()))

    def apply_object(self, target: cat.Array[B, cat.RawAxis]):
        return self.substitution.apply(target)

    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        if isinstance(target, Para.ParaMorphism):
            return self._substitute_slot_array(target)
        if not isinstance(target, cat.Broadcasted):
            return target
        kind, sparse = sparse_expansion.classify(target)
        if kind == 'block_operator':
            return self._expand_inside_the_box(target)
        if sparse is None and (sparse := selecting_input(target)) is not None:
            kind = 'linear_input'
        wire = None if sparse is None else self.wires.get(sparse.uid)
        if wire is None:
            return self._substitute_and_repair(target)
        if kind in sparse_expansion.CHAINED and self.wires.get(
                sparse_expansion.consumed_sparse(target).uid) is None:
            return self._substitute_and_repair(target)
        match kind:
            case 'topk':
                return self._drop_the_index(
                    target, wire, sparse, sparse_expansion.expand_topk)
            case 'topk_over_positions':
                return self._drop_the_index(
                    target, wire, sparse, sparse_expansion.expand_topk_over_positions)
            case 'linear':
                return self._grab_the_index(
                    target, wire, sparse, sparse_expansion.expand_linear)
            case 'linear_input':
                return self._grab_the_index(
                    target, wire, sparse, expand_selecting_input)
            case 'select':
                return self._grab_the_index_at_the_gather(target, wire, sparse)
            case 'merge':
                return self._chain_on_the_tape(
                    target, sparse, sparse_expansion.expand_merge)
            case 'topk_over_sparse':
                return self._chain_on_the_tape(
                    target, sparse, sparse_expansion.expand_topk_over_sparse)
        raise SelectionHasNoTapeExpansion(
            f'{type(target.operator).__name__} consumes the sparse axis '
            f'{sparse.uid.to_latex()} and this pass has a rule for TopK, for '
            'a parametric operation, for Select, for a merge, for a selection '
            'over a selection and for a selection over positions alone. '
            'deepseek.sparse_expansion expands it as a wire.')

    def _expand_inside_the_box(self, target: cat.Broadcasted[B, cat.RawAxis]):
        '''A `BlockOperator` whose body holds a selection, with the body taken
        through this pass and the box's own weaves substituted.

        The tape is the only coupling between a selection inside the box and a
        consumer outside it, so no wire has to cross the box, and a sparse axis
        on the box's domain or codomain is substituted like any other.
        `deepseek.sparse_expansion` refuses that axis, because its index wire
        would have to extend the box.
        '''
        operator = target.operator
        expanded_block = self.apply_category(operator.block)
        return target.reconstruct(
            operator=operator.reconstruct(block=expanded_block),
            input_weaves=tuple(
                self.substitution.apply(weave) for weave in target.input_weaves),
            output_weaves=tuple(
                self.substitution.apply(weave) for weave in target.output_weaves),
            reindexings=tuple(
                self.substitution.apply(reindexing)
                for reindexing in target.reindexings))

    def _substitute_slot_array(self, target: Para.ParaMorphism):
        '''A `Grab` or a `Drop`, with the sparse axis swapped in its array.

        `apply_object` reaches an object a morphism names in its domain or its
        codomain, and a tape operation carries its array in a field instead, so
        the substitution has to be applied here. A grab left unsubstituted
        loads at `k/n` from a slot the drop saved at `k`, and the two ends of
        one tape then disagree.
        '''
        return target.reconstruct(size=self.substitution.apply(target.size))

    def _substitute_and_repair(self, target: cat.Broadcasted[B, cat.RawAxis]):
        '''An operation the selection does not parametrise, with the axis swapped.

        `collapse_copies` is the repair `deepseek.sparse_expansion` documents. A
        reindexing that copied the sparse axis, meaning the diagonal that
        related a slot to the expert it named, loses the copies, because the
        expanded operation above it produces one `k` where the compressed one
        produced two. A diagonal whose copies have collapsed is the identity,
        and it is returned as one so that no `View` doing nothing survives.
        '''
        substituted = self.substitution.apply(target)
        if not isinstance(substituted, cat.Broadcasted):
            return substituted
        repaired = sparse_expansion.collapse_copies(
            substituted, {wire.k_axis for wire in self.wires.values()})
        if repaired is not substituted and ops.is_identity(repaired):
            return repaired.dom().identity()
        return repaired

    def _drop_the_index(
        self,
        target: cat.Broadcasted[B, cat.RawAxis],
        wire: sparse_expansion.SparseWire,
        sparse: ds.SparseAxis,
        expand,
    ):
        '''`[R; n] -> [R; k]`, with the index saved to the slot beside it.

        `expand` is `expand_topk` for a selection over an axis, and
        `expand_topk_over_positions` for one whose positions arrive as an
        operand. Both hand out the values and the index, in that order.
        '''
        expanded = expand(self.substitution.apply(target), wire)
        values, index = tuple(expanded.cod())
        return chsh.make_composed(
            expanded,
            chsh.make_product(
                cat.ProdObject((values,)).identity(),
                Para.Drop(tape=self.slots[sparse.uid], size=index)))

    def _grab_the_index(
        self,
        target: cat.Broadcasted[B, cat.RawAxis],
        wire: sparse_expansion.SparseWire,
        sparse: ds.SparseAxis,
        expand,
    ):
        '''The operation with its index operand, loaded from the slot.

        `expand` is the rule for the side the sparse axis is on, meaning
        `expand_linear` where the operator produces the axis and
        `expand_selecting_input` where it consumes one. Both put the index
        first, as the complete form of `example_notebooks/DeepSeekV3.ipynb` does,
        so the grab is the first factor of the product feeding it.
        '''
        expanded = expand(self.substitution.apply(target), wire)
        index, *carried = tuple(expanded.dom())
        return chsh.make_composed(
            chsh.make_product(
                Para.Grab(tape=self.slots[sparse.uid], size=index),
                cat.ProdObject(tuple(carried)).identity()),
            expanded)

    def _grab_the_index_at_the_gather(
        self,
        target: cat.Broadcasted[B, cat.RawAxis],
        wire: sparse_expansion.SparseWire,
        sparse: ds.SparseAxis,
    ):
        '''`Select` as `hold x IndexSelect ; einops('k, k -> k')`, with the
        `IndexSelect` reading the index from the slot.

        `expand_select` returns the composite with the values, the index and
        the payload on its domain, in that order, so the grab is the middle
        factor of the product feeding it.
        '''
        expanded = sparse_expansion.expand_select(
            self.substitution.apply(target), wire)
        values, index, payload = tuple(expanded.dom())
        return chsh.make_composed(
            chsh.make_product(
                cat.ProdObject((values,)).identity(),
                Para.Grab(tape=self.slots[sparse.uid], size=index),
                cat.ProdObject((payload,)).identity()),
            expanded)


    def _chain_on_the_tape(
        self,
        target: cat.Broadcasted[B, cat.RawAxis],
        produced_axis: ds.SparseAxis,
        rule,
    ):
        '''A merge, or a selection over a selection, grabbing the index of the
        axis it consumes and dropping the index of the axis it produces.

        `rule` is `expand_merge` or `expand_topk_over_sparse`. Both return the
        composite with the values and the consumed index on its domain, and the
        values and the produced index on its codomain, in that order.
        '''
        consumed_axis = sparse_expansion.consumed_sparse(target)
        consumed = self.wires[consumed_axis.uid]
        produced = self.wires[produced_axis.uid]
        expanded = rule(self.substitution.apply(target), consumed, produced)
        values, index_in = tuple(expanded.dom())
        values_out, index_out = tuple(expanded.cod())
        return chsh.make_composed(
            chsh.make_product(
                cat.ProdObject((values,)).identity(),
                Para.Grab(tape=self.slots[consumed_axis.uid], size=index_in)),
            expanded,
            chsh.make_product(
                cat.ProdObject((values_out,)).identity(),
                Para.Drop(tape=self.slots[produced_axis.uid], size=index_out)))


def expand_sparse_onto_tape[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    '''Expand every compressed selection in `target`, taping the index.

    An expression with no compressed selection comes back unchanged, as the
    same object, so the pass is free to run on anything.
    '''
    pass_over = ExpandSparseOntoTape()
    pass_over.survey(target)
    if not pass_over.wires:
        return target
    return pass_over(target)
