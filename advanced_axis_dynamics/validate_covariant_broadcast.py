'''Check the merge that broadcasts its input over the domain axes it does not carry.

Written by Claude Opus 5 (1M context), reasoning effort high.

    python advanced_axis_dynamics/validate_covariant_broadcast.py

The fixture is the lightning indexer of DeepSeek-V4.1-Flash, built here on fresh axes at
a compression ratio of one, two, three and four, so no notebook is imported. The keys
are read `r` entries back from every entry and one
`aops.CovariantView.broadcast_over_absent_axes_and_merge` writes each entry's slots onto
every query whose newest reachable entry that entry is.

Six things are checked. The `back` view marks the slots `r|b`, live where
`i_b - i_r >= 0`. The merge reads the entries alone and leaves the slots guided by the
query as `r|x`, live where `i_x - ratio * i_r - (ratio - 1) >= 0`. The query axis stays
dense, and the same merge with no slot axis to carry marks it, which is what makes the
slots the reason it stays dense. The entries each query reads through the whole chain
are the entries the reference's `compress_lens` admits, counted at every query of a
twelve-entry sequence at each of the four ratios. Reading the keys one entry further
back leaves the last query of the sequence written by no group, which is why the chain
reads at `i_b - i_r`. And the scoring after the merge is one box computed once per
query, which `discovering_broadcasts.confirm_broadcast_expansion` establishes by
expanding the box back out. Three further checks cover
`disentangle_reindexings.disentangle_reindexing`, which the merge's degree reindexing
goes through: the indexer's degree reindexing is the re-guiding of the slots beside an
identity on the key width, a morphism holding two independent maps splits into two
factors, and one whose maps interleave gains the rearrangements that group its domain
and restore its codomain and sends every domain point where the original does. A last
check exercises the two constructors' refusals.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the operator and the chain
the checks read.
'''
from __future__ import annotations
from dataclasses import dataclass

# Run as `python advanced_axis_dynamics/validate_covariant_broadcast.py` (from the
# repository root). Python prepends the script's own directory, whose `data_structure`,
# `algebra` and `registries` folders would shadow the packages of those names - drop it.
import os, sys
sys.path = [p for p in sys.path
            if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_display as ad
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @ and * overloads
import construction_helpers.lift as chl
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.ProductCategory as pc
import data_structure.Term as fd
import advanced_axis_dynamics.algebra.disentangle_reindexings as disentangle_reindexings
import advanced_axis_dynamics.algebra.mark_sparse_codomains as mark_sparse_codomains
import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import advanced_axis_dynamics.data_structure.Operators as aops

R = cat.Reals()
ENTRIES = 12
RATIOS = (1, 2, 3, 4)


@dataclass(frozen=True)
class IndexerChain:
    '''The indexer's reads at one compression ratio, on axes of this fixture's own.

    `read_back` reads the keys at entry `i_b - i_r` and carries the slots `back_reach`.
    `merge` writes each entry's slots onto every query of its group, `keys_per_query`
    composes the two, and `reach` is the slot axis guided by the query that comes out.
    `sizes` binds every size symbol, `queries` is the number of queries those sizes
    give, and `group_of_query` is read off the merge's own row.
    '''
    ratio: int
    ratio_size: nm.Numeric
    entry_axis: cat.RawAxis
    query_axis: cat.RawAxis
    read_back: cat.Broadcasted
    back_reach: AffineGuards.AffineSparseAxis
    merge: cat.Broadcasted
    keys_per_query: cat.BroadcastedCategory
    reach: AffineGuards.AffineSparseAxis
    sizes: dict[nm.Numeric, int]

    def queries(self) -> int:
        return nm.evaluate_integer(self.query_axis.local_size(), self.sizes)

    def group_of_query(self, query: int) -> int:
        '''The group whose newest reachable entry serves `query`, from the strides and
        the shift of the merge's own row.'''
        _, strides, shift = self.merge.operator.reindexing._cod_stride_shift[0]
        return ((query - nm.evaluate_integer(shift, self.sizes))
                // nm.evaluate_integer(strides[0], self.sizes))

    def entries_read_at(self, query: int) -> list[int]:
        '''The entries `query` reads: its group's newest reachable entry less each live
        distance.'''
        group = self.group_of_query(query)
        live = mark_sparse_domains.live_positions(
            self.reach, (query,), self.sizes)
        return sorted(group - distance for distance in live)

    def queries_written(self) -> set[int]:
        '''The queries some group and offset of the merge's row lands on.'''
        _, strides, shift = self.merge.operator.reindexing._cod_stride_shift[0]
        stride = nm.evaluate_integer(strides[0], self.sizes)
        landed = nm.evaluate_integer(shift, self.sizes)
        return {stride * group + offset + landed
                for group in range(ENTRIES) for offset in range(self.ratio)}


def indexer_chain(ratio: int, key_read_shift: int = 0) -> IndexerChain:
    '''The keys read at entry `i_b - i_r + key_read_shift` and merged onto the queries
    whose newest reachable entry each group holds.

    At `key_read_shift` of zero the slot zero of group `i_b` is the entry `i_b` itself,
    so group `i_b` holds the queries from `ratio * i_b + ratio - 1` to
    `ratio * i_b + 2 * ratio - 2` and the merge's row carries the shift `ratio - 1`. A
    shift of `-1` on the key read moves every group one entry back and takes
    `ratio * (key_read_shift + 1) - 1` as the row's shift, which
    `check_a_key_read_one_entry_back_drops_the_last_query` rejects.
    '''
    entry_axis = cat.RawAxis.named('b')
    channels = cat.RawAxis.named('d')
    offset_axes = () if ratio == 1 else (cat.RawAxis.named('a'),)
    ratio_size = (nm.Integer(1) if ratio == 1
                  else offset_axes[0].local_size())
    query_axis = fd.DynamicName('x').capture(cat.RawAxis(
        _size=nm.Multiplication.template(ratio_size, entry_axis.local_size())))
    read_back = mark_sparse_domains.guarded_view(
        reindexing=(cat.StrideMorphism(
            _dom=(entry_axis,
                  fd.DynamicName('r').capture(
                      cat.RawAxis(_size=entry_axis.local_size()))),
            _cod_stride_shift=((entry_axis, (nm.Integer(1), nm.Integer(-1)),
                                nm.Integer(key_read_shift)),),
            name=fd.DynamicName('back')),
            cat.ProdObject((channels,)).identity()),
        name='back')
    back_reach = read_back.cod()[0].shape()[1]
    merge = aops.CovariantView.broadcast_over_absent_axes_and_merge(
        cat.StrideMorphism(
            _dom=(entry_axis, *offset_axes),
            _cod_stride_shift=((query_axis,
                                (ratio_size, *(nm.Integer(1),) * len(offset_axes)),
                                ratio_size * nm.Integer(key_read_shift + 1)
                                - nm.Integer(1)),),
            name=fd.DynamicName('pos')),
        input_axes=(entry_axis,),
        degree=(back_reach, channels))
    keys_per_query = read_back @ merge
    sizes = {entry_axis.local_size(): ENTRIES}
    for axis in offset_axes:
        sizes[axis.local_size()] = ratio
    return IndexerChain(
        ratio=ratio, ratio_size=ratio_size,
        entry_axis=entry_axis, query_axis=query_axis,
        read_back=read_back, back_reach=back_reach, merge=merge,
        keys_per_query=keys_per_query,
        reach=keys_per_query.cod()[0].shape()[1], sizes=sizes)


def reachable_entry_count(query: int, ratio: int) -> int:
    '''The reference's `compress_lens`, the number of compressed entries query `query`
    has passed the last token of at compression ratio `ratio`.'''
    return (query + 1) // ratio


def check_the_keys_are_read_back_from_each_entry() -> None:
    '''The `back` view marks the slot axis `r|b`, live where `i_b - i_r >= 0`, with its
    empty end last and the entry count as its extent, because the row names the entry
    axis on both sides.'''
    i_b, i_r = (nm.FreeNumeric.named(name) for name in ('i_b', 'i_r'))
    for ratio in RATIOS:
        chain = indexer_chain(ratio)
        assert tuple(chain.read_back.cod()[0].shape())[0] == chain.entry_axis
        assert isinstance(chain.back_reach, AffineGuards.AffineSparseAxis)
        assert chain.back_reach.guides == (chain.entry_axis,)
        assert chain.back_reach.extent == chain.entry_axis.local_size()
        assert chain.back_reach.empty_end() is AffineGuards.EmptyEnd.LAST
        assert chain.back_reach.guard_form((i_b,), i_r) == nm.collect_like_terms(
            i_b - i_r), chain.back_reach.guard_form((i_b,), i_r)
        for entry in range(ENTRIES):
            assert mark_sparse_domains.live_positions(
                chain.back_reach, (entry,), chain.sizes) == list(range(entry + 1))


def check_the_merge_reads_the_entries_and_broadcasts_over_the_offsets() -> None:
    '''The merge's input weave carries the entries alone where its reindexing merges the
    entries and the offsets, the absent offsets enter the mixed-radix test, and the
    degree leaves as the re-guided slots beside the channels.'''
    for ratio in RATIOS:
        chain = indexer_chain(ratio)
        reindexing = chain.merge.operator.reindexing
        target = chain.merge.input_weaves[0].target().shape()
        assert tuple(target) == (chain.entry_axis,)
        assert tuple(reindexing.cod()) == (chain.query_axis,)
        assert len(tuple(reindexing.dom())) == (1 if ratio == 1 else 2)
        assert tuple(chain.merge.degree()) == (chain.reach,
                                               chain.read_back.cod()[0].shape()[2])
        groups = mark_sparse_codomains.merge_groups(
            mark_sparse_codomains.merge_carrying(
                reindexing, (chain.back_reach,)))
        assert [(group.row, group.digits, group.signs) for group in groups] == (
            [(0, (0,), (1,)), (1, (1,), (1,))] if ratio == 1
            else [(0, (1, 0), (1, 1)), (1, (2,), (1,))]), \
            'the query row writes the offsets then the entries at the radices ' \
            '(1, ratio), and the slots pass'


def check_the_slots_leave_guided_by_the_query() -> None:
    '''The slots leave the merge as `r|x`, live where
    `i_x - ratio * i_r - (ratio - 1) >= 0`, which is the reachability of the entry at
    that distance, and the live count at each query is the reference's.'''
    i_x, i_r = (nm.FreeNumeric.named(name) for name in ('i_x', 'i_r'))
    for ratio in RATIOS:
        chain = indexer_chain(ratio)
        assert isinstance(chain.reach, AffineGuards.AffineSparseAxis)
        assert chain.reach.guides == (chain.query_axis,)
        assert chain.reach.empty_end() is AffineGuards.EmptyEnd.LAST
        assert chain.reach.guard_form((i_x,), i_r) == nm.collect_like_terms(
            i_x - chain.ratio_size * i_r - chain.ratio_size + nm.Integer(1)), \
            chain.reach.guard_form((i_x,), i_r).to_latex()
        for query in range(chain.queries()):
            assert mark_sparse_domains.live_positions(
                chain.reach, (query,), chain.sizes) == list(
                    range(reachable_entry_count(query, ratio)))
        selected = chain.reach.selected_slots(nm.FreeNumeric.named('s'), 's')
        assert selected.guard_form((i_x,), i_r) == chain.reach.guard_form(
            (i_x,), i_r), \
            'a selection over the slots fills a prefix of its own under the same form'


def check_the_query_axis_stays_dense() -> None:
    '''The merge writes no query below `ratio - 1` and writes past the last query, and
    the query axis stays dense because every slot of an unwritten query is already
    empty. The same merge with no slot axis to carry marks the query axis, so the slots
    are the reason it stays dense rather than an accident of the row.'''
    i_x = nm.FreeNumeric.named('i_x')
    for ratio in RATIOS:
        chain = indexer_chain(ratio)
        assert chain.keys_per_query.cod()[0].shape()[0] == chain.query_axis
        assert isinstance(chain.query_axis, cat.RawAxis)
        written = chain.queries_written()
        unwritten = [query for query in range(chain.queries())
                     if query not in written]
        assert unwritten == list(range(ratio - 1)), unwritten
        for query in unwritten:
            assert not mark_sparse_domains.live_positions(
                chain.reach, (query,), chain.sizes), \
                'a query the merge writes nothing to reaches no entry'
        without_slots = aops.CovariantView.broadcast_over_absent_axes_and_merge(
            chain.merge.operator.reindexing, input_axes=(chain.entry_axis,))
        marked = without_slots.cod()[0].shape()[0]
        if ratio == 1:
            assert marked == chain.query_axis, 'a merge at ratio one fills its codomain'
            continue
        assert isinstance(marked, AffineGuards.AffineSparseAxis), \
            'with no slot axis to carry the form, the merge marks the query axis'
        assert marked.guard_form((), i_x) == nm.collect_like_terms(
            i_x - chain.ratio_size + nm.Integer(1)), \
            marked.guard_form((), i_x).to_latex()


def check_the_entries_read_are_those_compress_lens_admits() -> None:
    '''The entries each query reads through the whole chain, its group's newest
    reachable entry less each live distance, are the entries at or before
    `compress_lens`.'''
    for ratio in RATIOS:
        chain = indexer_chain(ratio)
        for query in range(chain.queries()):
            assert chain.entries_read_at(query) == list(
                range(reachable_entry_count(query, ratio))), (ratio, query)


def check_a_key_read_one_entry_back_drops_the_last_query() -> None:
    '''Reading the keys at `i_b - i_r - 1`, which pairs with the shift `-1` on the
    merge's row, leaves the last query of the sequence written by no group, and that
    query does reach entries. The chain therefore reads the keys at `i_b - i_r` and
    carries the shift `ratio - 1`.'''
    for ratio in (2, 3, 4):
        chain = indexer_chain(ratio, key_read_shift=-1)
        written = chain.queries_written()
        unwritten = [query for query in range(chain.queries()) if query not in written]
        assert unwritten == [chain.queries() - 1], unwritten
        assert mark_sparse_domains.live_positions(
            chain.reach, (unwritten[0],), chain.sizes), (
            'the query no group writes to reaches entries, so that shift loses '
            'its scores')


def score_every_query(chain: IndexerChain) -> ops.BBlock:
    '''Every query's score against each entry it has reached, as the indexer computes
    it: one rectified score per head and a per-head weight read off the hidden state.'''
    heads, width, low_rank, residual = (
        cat.RawAxis.named(name) for name in ('i', 'd', 'q', 'm'))
    keys = cat.Array(R, (chain.query_axis, chain.reach, width))
    head_weights = cat.Array(R, (chain.query_axis, heads))
    lifted = (chain.query_axis,)
    return cat.Block.template(
        (chl.morphism_object_lift(
            ops.Linear.template((low_rank,), (heads, width), 'q^{I}'),
            cat.ProdObject(lifted))
         * cat.ProdObject((keys,)).identity()
         * chl.morphism_object_lift(
             ops.Linear.template((residual,), (heads,), 'w^{I}'),
             cat.ProdObject(lifted)))
        @ ((ops.Einops.template('x i d, x r d -> x i r')
            @ ops.Arithmetic.template(nm.FreeInput() * nm.IsPositive()))
           * cat.ProdObject((head_weights,)).identity())
        @ ops.Einops.template('x i r, x i -> x r'))


def check_the_scoring_is_one_box_per_query() -> None:
    '''The scoring after the merge carries the query axis at the head of every array, so
    `discovering_broadcasts.discover_broadcast_over_axes` writes it as one body computed
    once per query and confirms the box by expanding it back out.'''
    for ratio in RATIOS:
        chain = indexer_chain(ratio)
        discovered = discovering_broadcasts.discover_broadcast_over_axes(
            score_every_query(chain), (chain.query_axis,), 'Sco')
        assert discovered.is_confirmed(), \
            discovered.confirmation.named_difference
        assert tuple(discovered.candidate.degree()) == (chain.query_axis,)
        assert tuple(discovered.candidate.cod()[0].shape()) == (
            chain.query_axis, chain.reach)


def codomain_positions(
    morphism: cat.StrideCategory, positions: fd.Prod[int],
    sizes: dict[nm.Numeric, int]) -> fd.Prod[int]:
    '''The codomain positions `morphism` sends the integer domain `positions` to, with
    every size symbol bound by `sizes`.'''
    match morphism:
        case cat.StrideMorphism():
            return tuple(
                sum(nm.evaluate_integer(stride, sizes) * position
                    for stride, position in zip(strides, positions))
                + nm.evaluate_integer(shift, sizes)
                for _, strides, shift in morphism._cod_stride_shift)
        case pc.Rearrangement(mapping=mapping):
            return tuple(positions[index] for index in mapping)
        case pc.ProductOfMorphisms(content=content):
            written: list[int] = []
            taken = 0
            for factor in content:
                width = len(factor.dom())
                written.extend(codomain_positions(
                    factor, positions[taken:taken + width], sizes))
                taken += width
            return tuple(written)
        case pc.Composed(content=content):
            carried = positions
            for factor in content:
                carried = codomain_positions(factor, carried, sizes)
            return carried
    raise AssertionError(f'{morphism} is no reindexing this check evaluates')


def two_independent_maps() -> cat.StrideMorphism:
    '''A morphism whose first two rows read its first two axes and whose third row
    reads its third, which is two maps written as one.'''
    first, second, third = (cat.RawAxis.named(name) for name in ('p', 'q', 'u'))
    return cat.StrideMorphism(
        _dom=(first, second, third),
        _cod_stride_shift=(
            (cat.RawAxis.named('S'), (nm.Integer(1), nm.Integer(1), nm.Integer(0)),
             nm.Integer(0)),
            (cat.RawAxis.named('D'), (nm.Integer(1), nm.Integer(-1), nm.Integer(0)),
             nm.Integer(0)),
            (cat.RawAxis.named('T'), (nm.Integer(0), nm.Integer(0), nm.Integer(2)),
             nm.Integer(1))),
        name=fd.DynamicName('two'))


def interleaved_maps() -> cat.StrideMorphism:
    '''A morphism whose two maps interleave in the domain and in the codomain: the
    first and the third axis are read by the first and the third row, and the second
    axis by the second row.'''
    first, second, third = (cat.RawAxis.named(name) for name in ('p', 'u', 'q'))
    return cat.StrideMorphism(
        _dom=(first, second, third),
        _cod_stride_shift=(
            (cat.RawAxis.named('S'), (nm.Integer(1), nm.Integer(0), nm.Integer(1)),
             nm.Integer(0)),
            (cat.RawAxis.named('U'), (nm.Integer(0), nm.Integer(1), nm.Integer(0)),
             nm.Integer(3)),
            (cat.RawAxis.named('P'), (nm.Integer(2), nm.Integer(0), nm.Integer(0)),
             nm.Integer(-1))),
        name=fd.DynamicName('mixed'))


def check_the_degree_reindexing_is_disentangled() -> None:
    '''The merge re-guides the slots and leaves the key width alone, so its degree
    reindexing is the product of a one-row map onto the arriving slots and an identity
    on the key width, and the key width's wire runs straight through the figure.'''
    for ratio in RATIOS:
        chain = indexer_chain(ratio)
        channels = chain.read_back.cod()[0].shape()[2]
        reindexing = chain.merge.reindexings[0]
        assert isinstance(reindexing, pc.ProductOfMorphisms), type(reindexing).__name__
        reguiding, passed_through = reindexing.content
        assert isinstance(reguiding, cat.StrideMorphism)
        assert tuple(reguiding.dom()) == (chain.reach,)
        assert tuple(reguiding.cod()) == (chain.back_reach,)
        assert len(reguiding._cod_stride_shift) == 1, \
            'the re-guiding is one row, so it draws as a pentagon on the slot wire'
        assert isinstance(passed_through, pc.Rearrangement)
        assert tuple(passed_through.dom()) == (channels,)
        assert tuple(passed_through.cod()) == (channels,)


def check_two_independent_maps_split_into_two_factors() -> None:
    '''A morphism holding two maps that read disjoint axes is returned as the product of
    them, in the order its domain holds them, with no rearrangement.'''
    morphism = two_independent_maps()
    disentangled = disentangle_reindexings.disentangle_reindexing(morphism)
    assert isinstance(disentangled, pc.ProductOfMorphisms), type(disentangled).__name__
    coupled, alone = disentangled.content
    assert len(tuple(coupled.dom())) == 2 and len(tuple(coupled.cod())) == 2
    assert len(tuple(alone.dom())) == 1 and len(tuple(alone.cod())) == 1
    assert tuple(disentangled.dom()) == tuple(morphism.dom())
    assert tuple(disentangled.cod()) == tuple(morphism.cod())
    for positions in ((0, 0, 0), (1, 0, 2), (3, 2, 1), (5, 4, 7)):
        assert codomain_positions(disentangled, positions, {}) == codomain_positions(
            morphism, positions, {}), positions


def check_interleaved_maps_gain_their_rearrangements() -> None:
    '''Where the two maps interleave, the result is the product between the
    rearrangement that groups the domain and the one that restores the codomain, and it
    sends every domain point where the original does.'''
    morphism = interleaved_maps()
    disentangled = disentangle_reindexings.disentangle_reindexing(morphism)
    assert isinstance(disentangled, pc.Composed), type(disentangled).__name__
    grouping, product, restoring = disentangled.content
    assert isinstance(grouping, pc.Rearrangement) and grouping.mapping == (0, 2, 1)
    assert isinstance(product, pc.ProductOfMorphisms)
    assert isinstance(restoring, pc.Rearrangement) and restoring.mapping == (0, 2, 1)
    assert tuple(disentangled.dom()) == tuple(morphism.dom())
    assert tuple(disentangled.cod()) == tuple(morphism.cod())
    for positions in ((0, 0, 0), (1, 2, 3), (4, 1, 0), (2, 5, 6)):
        assert codomain_positions(disentangled, positions, {}) == codomain_positions(
            morphism, positions, {}), positions


def check_input_axes_that_are_not_the_leading_domain_axes_are_rejected() -> None:
    '''`template` merges an input carrying one axis per domain axis and refuses an input
    carrying fewer, and the broadcasting constructor refuses input axes that are not the
    first of the domain.'''
    chain = indexer_chain(2)
    reindexing = chain.merge.operator.reindexing
    offsets = tuple(reindexing.dom())[1]
    for call in (
            lambda: aops.CovariantView.template(
                reindexing, input_axes=(chain.entry_axis,)),
            lambda: aops.CovariantView.broadcast_over_absent_axes_and_merge(
                reindexing, input_axes=(offsets,)),
            lambda: aops.CovariantView.broadcast_over_absent_axes_and_merge(
                reindexing, input_axes=(chain.entry_axis, offsets, offsets))):
        try:
            call()
        except aops.InputAxesDoNotMatchTheDomain:
            continue
        raise AssertionError(
            'a merge whose input axes are not the leading domain axes is rejected')


CHECKS = (
    check_the_keys_are_read_back_from_each_entry,
    check_the_merge_reads_the_entries_and_broadcasts_over_the_offsets,
    check_the_slots_leave_guided_by_the_query,
    check_the_query_axis_stays_dense,
    check_the_entries_read_are_those_compress_lens_admits,
    check_a_key_read_one_entry_back_drops_the_last_query,
    check_the_scoring_is_one_box_per_query,
    check_the_degree_reindexing_is_disentangled,
    check_two_independent_maps_split_into_two_factors,
    check_interleaved_maps_gain_their_rearrangements,
    check_input_axes_that_are_not_the_leading_domain_axes_are_rejected,
)


if __name__ == '__main__':
    chain = indexer_chain(2)
    print('==== the keys read back from every entry and merged onto the queries ====')
    print(ad.listing(chain.keys_per_query))
    for ratio in RATIOS:
        at_ratio = indexer_chain(ratio)
        print(f'  ratio {ratio}: query 0 reads '
              f'{at_ratio.entries_read_at(0)}, query {at_ratio.queries() - 1} reads '
              f'{len(at_ratio.entries_read_at(at_ratio.queries() - 1))} entries')
    for check in CHECKS:
        check()
        print(f'  ok    {check.__name__}')
    print('all covariant broadcast checks passed')
