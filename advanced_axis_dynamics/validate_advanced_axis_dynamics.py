'''Check the guard a read derives and the expansion of a concatenation.

Written by Claude Opus 5 (1M context), reasoning effort medium.

    python advanced_axis_dynamics/validate_advanced_axis_dynamics.py

Four groups of checks, all on fixtures built here, so none of them needs a notebook.

  * the reads of the DeepSeek-V4.1-Flash model, written in
    `notebooks/sota/DeepSeekV41Flash/`, on fresh axes: the window
    view, the relative indexer and the block split each mark an `AffineSparseAxis` on
    their output, the merge re-guides the slots it is broadcast over, a read of a
    guarded axis pulls its form back, a fold leaves the form at the folded axis's most
    favourable position, a selection over a prefix-live axis fills a prefix of its
    slots, and the live sets at concrete sizes are the reference's. These checks were
    `check_causal_reads` in `deepseek/validate_sparse.py` until 2026-09-15, and moved
    here with the code they check;

  * the attention core of the same notebook, written with the window latents and the
    selected-entry latents concatenated along the slot axis: the expansion of the
    concatenation is the two-branch core the notebook writes by hand, compared as
    `agent_display` listings, and so is the expansion of the same core with its body
    in a box computed once per head, whose broadcast
    `discovering_broadcasts.confirm_broadcast_expansion` confirms against the core
    written out over every head;

  * a concatenation onto an axis the model declared, which the deconcatenation of
    that axis composes with to one domain and codomain, and whose consumers the
    expansion rewrites as it rewrites the consumers of a `ConcatenatedAxis`;

  * the reverse derivative of a merge and of a concatenation, whose domain and
    codomain are the codomain and the domain of the operator they reverse.

Structural assertions rather than a listing diff, except for the expansion, where the
two forms are built in one process and the listings are therefore comparable.
'''
from __future__ import annotations
from dataclasses import dataclass

# Run as `python advanced_axis_dynamics/validate_advanced_axis_dynamics.py` (from the
# repository root). Python prepends the script's own directory, whose `data_structure`,
# `algebra` and `registries` folders would shadow the packages of those names - drop it.
import os, sys
sys.path = [p for p in sys.path
            if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_display as ad
import construction_helpers as ch  # noqa: F401 - @ auto-alignment
import construction_helpers.lift as chl
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.processing.Hypergraph2Morphism as h2m
import algebra.discovering_broadcasts as discovering_broadcasts
import para.registries.derivative as derivative
import term_utilities.term_utilities as tutil
import advanced_axis_dynamics.algebra.concatenation_expansion as concatenation_expansion
import advanced_axis_dynamics.algebra.mark_sparse_codomains as mark_sparse_codomains
import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import advanced_axis_dynamics.data_structure.Operators as aops
import advanced_axis_dynamics.registries.derivative  # noqa: F401 - the reverse rules
import advanced_axis_dynamics.registries.standard_expansions  # noqa: F401 - the deconcatenation's rule
import algebra.registries.standard_expansions as standard_expansions

R = cat.Reals()


def over(axes, morphism):
    return chl.morphism_object_lift(morphism, cat.ProdObject(tuple(axes)))


def hold(array):
    return cat.ProdObject((array,)).identity()


def the(iterable):
    (item,) = tuple(iterable)
    return item


def exponential():
    return ops.Arithmetic.template(nm.E ** nm.x)


def reciprocal():
    return ops.Arithmetic.template(nm.Integer(1) / nm.x, name='z^{-1}')


def guarded_reads():
    '''The window view, the relative indexer and the block split of
    `notebooks/sota/DeepSeekV41Flash/`, built on fresh axes.

    The window view reads the last `|w|` tokens through `win` at `i_x - i_w`, slot `w`
    counting back from the current token, so the guard arises from the negative stride
    on `w`. The query axis is sized `|a| |b|`.

    The indexer reads the queries through `grp` at
    `|a| i_{b_0} - i_a + 2 |a| - 2`, group `b_0` holding the queries whose newest
    reachable entry is `b_0`, and the keys through `back` at `i_b = i_{b_0} - i_r`,
    entry `r` back from that entry. The `pos` merge writes the scores at `(b_0, a, r)`
    back to the query, broadcast over `r`, whose guide it consumes.
    '''
    w, b, a, u, P, d, c = (cat.RawAxis.named(name) for name in 'wbauPdc')
    x = fd.DynamicName('x').capture(
        cat.RawAxis(_size=a.local_size() * b.local_size()))
    window = cat.StrideMorphism(
        _dom=(x, w),
        _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
        name=fd.DynamicName('win'))
    window_view = mark_sparse_domains.guarded_view(
        reindexing=(window, cat.ProdObject((c,)).identity()), name='win')
    group = fd.DynamicName.from_str('b_0').capture(
        cat.RawAxis(_size=b.local_size()))
    back = fd.DynamicName('r').capture(cat.RawAxis(_size=b.local_size()))
    ratio = a.local_size()
    query_groups = cat.StrideMorphism(
        _dom=(group, a),
        _cod_stride_shift=((x, (ratio, nm.Integer(-1)),
                            nm.Integer(2) * ratio - nm.Integer(2)),),
        name=fd.DynamicName('grp'))
    entries_back = cat.StrideMorphism(
        _dom=(group, back),
        _cod_stride_shift=((b, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
        name=fd.DynamicName('back'))
    read_queries = mark_sparse_domains.guarded_view(
        reindexing=(query_groups, cat.ProdObject((d,)).identity()), name='grp')
    read_keys = mark_sparse_domains.guarded_view(
        reindexing=(entries_back, cat.ProdObject((d,)).identity()), name='back')
    offset_reach = read_queries.cod()[0].shape()[1]
    back_reach = read_keys.cod()[0].shape()[1]
    restore = aops.CovariantView.template(
        query_groups.reconstruct(name=fd.DynamicName('pos')), degree=(back_reach,))
    indexer = ((read_queries * read_keys)
               @ ops.Einops.template('b0 a d, b0 r d -> b0 a r')
               @ restore)
    r_reach = indexer.cod()[0].shape()[1]
    split = cat.StrideMorphism(
        _dom=(P, u),
        _cod_stride_shift=((r_reach, (u.local_size(), nm.Integer(1)), nm.Integer(0)),),
        name=fd.DynamicName('blk'))
    block_view = mark_sparse_domains.guarded_view(
        reindexing=(split,), name='blk')
    group_view = mark_sparse_domains.guarded_view(
        reindexing=(cat.StrideMorphism(
            _dom=(b, a),
            _cod_stride_shift=((x, (a.local_size(), nm.Integer(1)), nm.Integer(0)),),
            name=fd.DynamicName('grp')),),
        name='grp')
    return dict(x=x, w=w, b=b, a=a, u=u, P=P, group=group, back=back,
                window_view=window_view, offset_reach=offset_reach,
                back_reach=back_reach, restore=restore, indexer=indexer,
                r_reach=r_reach, split=split, block_view=block_view,
                group_view=group_view)


def check_guarded_reads() -> None:
    '''A row with a negative shift or stride marks the last axis it reads as an
    `AffineSparseAxis` guided by the others, a row onto an axis sized in its own symbols
    that runs past the end marks it too, a merge re-guides a broadcast axis whose guide
    it consumes and leaves the rest dense, a row reading a guarded axis pulls its form
    back, a fold leaves the form at the folded axis's most favourable position, a
    selection over a prefix-live axis fills a prefix of its slots, the live sets at
    concrete sizes are the reference's, and an unshifted row marks nothing.'''
    r = guarded_reads()
    x, w, b, a, u, P = (r[name] for name in 'xwbauP')
    i_x, i_P, i_u, j, i_g, i_r, i_a = (nm.FreeNumeric.named(name)
                                       for name in ('i_x', 'i_P', 'i_u', 'j', 'i_g',
                                                    'i_r', 'i_a'))
    w_axis = r['window_view'].cod()[0].shape()[1]
    assert isinstance(w_axis, AffineGuards.AffineSparseAxis) and w_axis.guides == (x,)
    assert w_axis.empty_end() is AffineGuards.EmptyEnd.LAST
    assert w_axis.guard_form((i_x,), j) == nm.collect_like_terms(i_x - j)
    sizes = {a.local_size(): 2, b.local_size(): 20, w.local_size(): 8}
    for query in range(40):
        live = mark_sparse_domains.live_positions(w_axis, (query,), sizes)
        assert live == list(range(min(query + 1, 8))), (query, live)
        assert [query - slot for slot in live] == list(
            range(query, max(-1, query - 8), -1)), (query, live)
    assert all(isinstance(axis, cat.RawAxis)
               for axis in r['group_view'].cod()[0].shape()), \
        'an unshifted group view onto the query axis sized |a| |b| marks nothing'
    offset_reach = r['offset_reach']
    assert isinstance(offset_reach, AffineGuards.AffineSparseAxis)
    assert offset_reach.guides == (r['group'],)
    assert offset_reach.extent == x.local_size()
    assert offset_reach.empty_end() is AffineGuards.EmptyEnd.FIRST
    assert offset_reach.guard_form((i_g,), i_a) == nm.collect_like_terms(
        a.local_size() * i_g - i_a + nm.Integer(2) * a.local_size() - nm.Integer(2))
    few = {a.local_size(): 2, b.local_size(): 4}
    assert [mark_sparse_domains.live_positions(offset_reach, (group,), few)
            for group in range(4)] == [[0, 1], [0, 1], [0, 1], [1]], \
        'the last group of queries holds the last token alone'
    back_reach = r['back_reach']
    assert isinstance(back_reach, AffineGuards.AffineSparseAxis)
    assert back_reach.guides == (r['group'],)
    assert back_reach.empty_end() is AffineGuards.EmptyEnd.LAST
    assert back_reach.guard_form((i_g,), i_r) == nm.collect_like_terms(i_g - i_r), \
        'the keys read back from the newest reachable entry mark the slots past it'
    restore = r['restore']
    assert tuple(restore.operator.reindexing.cod()) == (x,)
    r_reach, = restore.degree()
    assert r_reach is r['r_reach']
    assert tuple(r['indexer'].cod()[0].shape()) == (x, r_reach)
    assert isinstance(r_reach, AffineGuards.AffineSparseAxis) and r_reach.guides == (x,)
    assert r_reach.empty_end() is AffineGuards.EmptyEnd.LAST
    reach_form = nm.collect_like_terms(
        i_x - a.local_size() * i_r - a.local_size() + nm.Integer(1))
    assert r_reach.guard_form((i_x,), i_r) == reach_form, \
        'the slots leave the merge guided by the query'
    assert isinstance(restore.reindexings[0], cat.StrideMorphism), \
        'the degree reindexing pairs the re-guided slots with the arriving ones'
    groups = mark_sparse_codomains.merge_groups(mark_sparse_codomains.merge_carrying(
        restore.operator.reindexing, (back_reach,)))
    assert [(group.row, group.digits, group.signs) for group in groups] == [
        (0, (1, 0), (-1, 1)), (1, (2,), (1,))], \
        'the query row writes (a, b_0) at the signed radices (-1, |a|), and the ' \
        'slots pass'
    s_reach = r_reach.selected_slots(nm.FreeNumeric.named('s'), 's')
    assert isinstance(s_reach, AffineGuards.AffineSparseAxis)
    assert s_reach.guard_form((i_x,), i_r) == reach_form, \
        'a selection over a prefix-live axis fills a prefix of its slots under the ' \
        'same form'
    sizes = {a.local_size(): 2, b.local_size(): 30, s_reach.local_size(): 8}
    for query in range(60):
        reachable = (query + 1) // 2
        assert mark_sparse_domains.live_positions(
            r_reach, (query,), sizes) == list(range(reachable))
        assert mark_sparse_domains.live_positions(
            s_reach, (query,), sizes) == list(range(min(8, reachable)))
    plain_split = cat.StrideMorphism(
        _dom=(P, u),
        _cod_stride_shift=((cat.RawAxis.named('B'), (u.local_size(), nm.Integer(1)),
                            nm.Integer(0)),),
        name=fd.DynamicName('blk'))
    assert mark_sparse_codomains.mark_sparse_codomain(plain_split) is plain_split, \
        'a merge that fills its codomain marks nothing'
    try:
        mark_sparse_codomains.merge_groups(cat.StrideMorphism(
            _dom=(x, w),
            _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(1)), nm.Integer(0)),),
            name=None))
    except mark_sparse_codomains.NotAMixedRadixSplit:
        pass
    else:
        raise AssertionError('a sum of two axes is not an injection and is rejected')
    u_reach = r['block_view'].cod()[0].shape()[1]
    assert isinstance(u_reach, AffineGuards.AffineSparseAxis)
    assert u_reach.guides == (x, P)
    pulled = nm.collect_like_terms(
        i_x - a.local_size() * u.local_size() * i_P - a.local_size() * i_u
        - a.local_size() + nm.Integer(1))
    assert u_reach.guard_form((i_x, i_P), i_u) == pulled
    assert mark_sparse_domains.pull_back_sparse_axis(
        r_reach, r['split'])[1].guard_form((i_x, i_P), i_u) == pulled
    P_reach = mark_sparse_domains.sparse_axis_after_fold(u_reach)
    assert P_reach.guides == (x,)
    assert P_reach.empty_end() is AffineGuards.EmptyEnd.LAST
    assert P_reach.guard_form((i_x,), i_P) == nm.collect_like_terms(
        i_x - a.local_size() * u.local_size() * i_P - a.local_size() + nm.Integer(1))


HEAD, QUERIES, CHANNELS, WINDOW, SELECTED = (
    cat.RawAxis.named(name) for name in ('h', 'x', 'c', 'w', 's'))
JOIN_LATENTS = aops.ConcatenateAxes.template(
    ((QUERIES, WINDOW, CHANNELS), (QUERIES, SELECTED, CHANNELS)), name='slots')
SLOTS = the(JOIN_LATENTS.output_weaves).target().shape()[0]

QUERY = cat.Array(R, (HEAD, QUERIES, CHANNELS))
WINDOW_LATENTS = cat.Array(R, (QUERIES, WINDOW, CHANNELS))
SELECTED_LATENTS = cat.Array(R, (QUERIES, SELECTED, CHANNELS))
LATENTS = cat.Array(R, (QUERIES, SLOTS, CHANNELS))
WINDOW_SCORES = cat.Array(R, (HEAD, QUERIES, WINDOW))
SELECTED_SCORES = cat.Array(R, (HEAD, QUERIES, SELECTED))
SCORES = cat.Array(R, (HEAD, QUERIES, SLOTS))
DENOMINATOR = cat.Array(R, (HEAD, QUERIES))
QUERY_PER_HEAD = cat.Array(R, (QUERIES, CHANNELS))
SCORES_PER_HEAD = cat.Array(R, (QUERIES, SLOTS))
DENOMINATOR_PER_HEAD = cat.Array(R, (QUERIES,))


def attention_axes() -> fd.Prod[cat.Axis]:
    '''The axes the attention cores share: the heads, the queries, the channels, the
    window slots and the selected slots.'''
    return (HEAD, QUERIES, CHANNELS, WINDOW, SELECTED)


@dataclass(frozen=True)
class AttentionCoreForms:
    '''The four spellings of the attention core that the expansion relates.

    They are built together because the box and the expansion of the box have to hold
    one body: a comparison reads the tag of every block, so a second call to the
    body's constructor gives a block the confirmation reports as a difference.

    The sink logit of `notebooks/sota/DeepSeekV41Flash/attention_core.py` is left out
    of all four, because it enters the denominator alone and says nothing about the
    concatenation.
    '''
    two_branch: cat.BroadcastedCategory
    concatenated: cat.BroadcastedCategory
    concatenated_in_a_box: cat.BroadcastedCategory
    written_over_heads: cat.BroadcastedCategory
    box: cat.Broadcasted


def slot_scores(letter: str) -> cat.Broadcasted:
    '''Every query head against one kind of slot.'''
    return ops.Einops.template(f'h x c, x {letter} c -> h x {letter}')


def branch_scores(slot_axis: cat.Axis, letter: str) -> cat.BroadcastedCategory:
    '''One kind of slot's scores, exponentiated.'''
    return (slot_scores(letter)
            @ over((HEAD, QUERIES, slot_axis), exponential()))


def single_branch_core_over_heads() -> cat.BroadcastedCategory:
    '''The attention core the requester wrote on 2026-09-15, over every head at once:
    one contraction of the queries against the concatenated latents, one exponential,
    one denominator sum over the concatenated slots, one contraction of the weights
    against the same latents, and the reciprocal of the denominator against it.

    The queries carry the head axis and the concatenated latents do not, because one
    latent per slot is shared by every head.
    '''
    return (cat.Rearrangement((0, 1, 1), (QUERY, LATENTS))
            @ (ops.Einops.template('h x c, x t c -> h x t') * hold(LATENTS))
            @ (over((HEAD, QUERIES, SLOTS), exponential()) * hold(LATENTS))
            @ cat.Rearrangement((0, 0, 1), (SCORES, LATENTS))
            @ (ops.Einops.template('h x t -> h x') * hold(SCORES) * hold(LATENTS))
            @ (over((HEAD, QUERIES), reciprocal()) * hold(SCORES) * hold(LATENTS))
            @ (hold(DENOMINATOR) * ops.Einops.template('h x t, x t c -> h x c'))
            @ ops.Einops.template('h x, h x c -> h x c'))


def single_branch_core_per_head() -> ops.BBlock:
    '''The same core for one head, which is the body the per-head box holds. It
    carries the head axis nowhere.'''
    return cat.Block.template(
        cat.Rearrangement((0, 1, 1), (QUERY_PER_HEAD, LATENTS))
        @ (ops.Einops.template('x c, x t c -> x t') * hold(LATENTS))
        @ (over((QUERIES, SLOTS), exponential()) * hold(LATENTS))
        @ cat.Rearrangement((0, 0, 1), (SCORES_PER_HEAD, LATENTS))
        @ (ops.Einops.template('x t -> x') * hold(SCORES_PER_HEAD) * hold(LATENTS))
        @ (over((QUERIES,), reciprocal()) * hold(SCORES_PER_HEAD) * hold(LATENTS))
        @ (hold(DENOMINATOR_PER_HEAD)
           * ops.Einops.template('x t, x t c -> x c'))
        @ ops.Einops.template('x, x c -> x c'),
        title='\\text{Attention Core}')


def two_branch_attention_core() -> cat.BroadcastedCategory:
    '''The same core with the two kinds of slot written out, which is the form
    `attend_over_all_heads_with_entries` holds in the notebook: each branch
    exponentiates its own scores, sums its own denominator and contracts against its
    own latents, and the two denominators and the two numerators are added.'''
    return h2m.recycle(
        cat.Rearrangement((0, 1, 0, 2, 1, 2),
                          (QUERY, WINDOW_LATENTS, SELECTED_LATENTS))
        @ (branch_scores(WINDOW, 'w') * branch_scores(SELECTED, 's')
           * hold(WINDOW_LATENTS) * hold(SELECTED_LATENTS))
        @ cat.Rearrangement(
            (0, 0, 1, 1, 2, 3),
            (WINDOW_SCORES, SELECTED_SCORES, WINDOW_LATENTS, SELECTED_LATENTS))
        @ (ops.Einops.template('h x w -> h x') * hold(WINDOW_SCORES)
           * ops.Einops.template('h x s -> h x') * hold(SELECTED_SCORES)
           * hold(WINDOW_LATENTS) * hold(SELECTED_LATENTS))
        @ cat.Rearrangement(
            (0, 2, 1, 3, 4, 5),
            (DENOMINATOR, WINDOW_SCORES, DENOMINATOR, SELECTED_SCORES,
             WINDOW_LATENTS, SELECTED_LATENTS))
        @ (over((HEAD, QUERIES), ops.AdditionOp.template())
           * hold(WINDOW_SCORES) * hold(SELECTED_SCORES)
           * hold(WINDOW_LATENTS) * hold(SELECTED_LATENTS))
        @ (over((HEAD, QUERIES), reciprocal())
           * hold(WINDOW_SCORES) * hold(SELECTED_SCORES)
           * hold(WINDOW_LATENTS) * hold(SELECTED_LATENTS))
        @ cat.Rearrangement(
            (0, 1, 3, 2, 4),
            (DENOMINATOR, WINDOW_SCORES, SELECTED_SCORES,
             WINDOW_LATENTS, SELECTED_LATENTS))
        @ (hold(DENOMINATOR) * ops.Einops.template('h x w, x w c -> h x c')
           * ops.Einops.template('h x s, x s c -> h x c'))
        @ (hold(DENOMINATOR)
           * over((HEAD, QUERIES, CHANNELS), ops.AdditionOp.template()))
        @ ops.Einops.template('h x, h x c -> h x c'))


def attention_core_forms() -> AttentionCoreForms:
    '''The two-branch core, the concatenated core written over every head, the same
    core as one box computed once per head, and the two composed behind the
    concatenation of the latents.

    The concatenation stands outside the box in both concatenated forms. It is the
    same array at every head, so a box holding it would concatenate the latents once
    per head.
    '''
    written_over_heads = single_branch_core_over_heads()
    box = discovering_broadcasts.broadcast_block_over_axes(
        single_branch_core_per_head(), (HEAD,), ((0,), ()),
        name='\\text{Attention Core}')
    join = hold(QUERY) * JOIN_LATENTS
    return AttentionCoreForms(
        two_branch=two_branch_attention_core(),
        concatenated=h2m.recycle(join @ written_over_heads),
        concatenated_in_a_box=join @ box,
        written_over_heads=written_over_heads,
        box=box)


def check_concatenation_expansion() -> None:
    '''The expansion of the concatenated core is the two-branch core, listing for
    listing: each branch contracts the queries against its own latents, exponentiates
    its own scores and sums its own denominator, the denominator is the sum of the two
    branches' sums, and the numerator is the sum of the two branches' contractions.'''
    forms = attention_core_forms()
    expanded = concatenation_expansion.expand_concatenations(forms.concatenated)
    assert not tuple(
        root for root in expanded_operators(expanded)
        if isinstance(root, aops.ConcatenateAxes)), \
        'a concatenation survived the expansion'
    assert tuple(expanded.dom()) == tuple(forms.concatenated.dom())
    assert tuple(expanded.cod()) == tuple(forms.concatenated.cod())
    assert ad.listing(h2m.recycle(expanded)) == ad.listing(forms.two_branch), (
        ad.listing(h2m.recycle(expanded)) + '\n\n' + ad.listing(forms.two_branch))


def check_concatenation_expansion_through_a_block() -> None:
    '''The same core with its body in a box computed once per head expands to the same
    two-branch core.

    The concatenation feeds the box, so the rewrite reaches its consumers only once
    the box is written out: the body is lifted over the head axis and each operand is
    read through the `View` of its own reindexing. The latents are shared by every
    head, so each part arrives through a repeat, and the comparison runs after
    `discovering_broadcasts.normalise_for_comparison`, which absorbs a repeat into
    every operation that reads it.
    '''
    forms = attention_core_forms()
    expanded = concatenation_expansion.expand_concatenations_through_blocks(
        forms.concatenated_in_a_box)
    assert not tuple(
        root for root in expanded_operators(expanded)
        if isinstance(root, (aops.ConcatenateAxes, ops.BlockOperator))), \
        'a concatenation or a block operator survived the expansion'
    assert tuple(expanded.dom()) == tuple(forms.two_branch.dom())
    assert tuple(expanded.cod()) == tuple(forms.two_branch.cod())
    derived = ad.listing(
        discovering_broadcasts.normalise_for_comparison(expanded))
    by_hand = ad.listing(
        discovering_broadcasts.normalise_for_comparison(forms.two_branch))
    assert derived == by_hand, derived + '\n\n' + by_hand


def check_the_box_states_the_core_over_every_head() -> None:
    '''The box computed once per head and the same core written out over every head
    are one expression, which `discovering_broadcasts.confirm_broadcast_expansion`
    establishes by expanding the box and comparing.

    The concatenated latents stand on the domain of both, so the concatenated form and
    the broadcast form are the same expression and the expansion of either gives the
    two-branch core.
    '''
    forms = attention_core_forms()
    confirmation = discovering_broadcasts.confirm_broadcast_expansion(
        forms.written_over_heads, forms.box)
    assert confirmation.is_confirmed(), confirmation.named_difference
    degree = tuple(forms.box.degree())
    assert degree == (HEAD,), 'the box is computed once per head'
    assert tuple(forms.box.dom()[1].shape()) == (QUERIES, SLOTS, CHANNELS), \
        'the concatenated latents stand on the domain of the box'


def expanded_operators(target):
    '''The operator of every seed morphism of `target`.'''
    return tuple(root.operator for root in tutil.type_search(cat.Broadcasted, target))


def check_parts_fill_the_axis() -> None:
    '''A concatenation states one axis per input, laid out in order, filling the axis
    they are written onto. Shapes that differ at two positions name no concatenated
    axis, and parts that leave a gap fill nothing.'''
    head, queries, channels, window, selected = attention_axes()
    joined = aops.ConcatenateAxes.template(
        ((head, queries, window), (head, queries, selected)), name='slots')
    slots = the(joined.output_weaves).target().shape()[0]
    assert isinstance(slots, AxisConcatenation.ConcatenatedAxis)
    assert slots.parts == (window, selected), 'the axis references its parts'
    assert slots.local_size() == nm.Addition.template(
        window.local_size(), selected.local_size())
    assert ad.morphism_ir.axis_name(slots) == 'w + s'
    assert joined.operator.parts() == (window, selected)
    assert joined.operator.concatenated_axis() == slots
    try:
        aops.ConcatenateAxes.template(
            ((head, queries, window), (queries, head, selected)))
    except aops.PartsDisagreeOutsideTheConcatenatedAxis:
        pass
    else:
        raise AssertionError('two shapes differing at three positions are rejected')
    try:
        aops.ConcatenateAxes.template(
            ((head, queries, selected), (head, queries, window)), concatenated=slots)
    except aops.ConcatenatedAxisHasOtherParts:
        pass
    else:
        raise AssertionError('an axis handed over in the other order is rejected')
    gapped = cat.StrideMorphism(
        _dom=(selected,),
        _cod_stride_shift=((slots, (nm.Integer(1),),
                            nm.Addition.template(window.local_size(),
                                                 nm.Integer(1))),))
    try:
        aops.check_parts_fill_the_axis(
            (joined.operator.part_reindexings[0], gapped))
    except aops.PartsDoNotFillTheAxis:
        pass
    else:
        raise AssertionError('a part starting past the one before it is rejected')


def check_the_concatenated_axis_follows_its_parts() -> None:
    '''A concatenation written over a raw axis meets the guarded axis a view produces,
    composition replaces the part, and the concatenated axis, which references its
    parts, is labelled by the guarded part and sized by the sum. A configuration that
    sizes the parts sizes the concatenated axis to the integer sum.'''
    head, queries, channels, window, selected = attention_axes()
    window_read = cat.StrideMorphism(
        _dom=(queries, window),
        _cod_stride_shift=((queries, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
        name=fd.DynamicName('win'))
    guarded_window = mark_sparse_domains.guarded_view(
        reindexing=(window_read, cat.ProdObject((channels,)).identity()), name='win')
    model = ((guarded_window * hold(SELECTED_LATENTS))
             @ aops.ConcatenateAxes.template(
                 ((queries, window, channels), (queries, selected, channels))))
    slots = the(model.cod()).shape()[1]
    guarded_part = the(guarded_window.cod()).shape()[1]
    assert isinstance(slots, AxisConcatenation.ConcatenatedAxis)
    assert isinstance(guarded_part, AffineGuards.AffineSparseAxis)
    assert [part.uid for part in slots.parts] == [guarded_part.uid, selected.uid]
    assert isinstance(slots.parts[0], AffineGuards.AffineSparseAxis), \
        'the concatenated axis carries the guarded part composition put in its place'
    assert ad.morphism_ir.axis_name(slots) == 'w|x + s'
    assert slots.local_size() == nm.Addition.template(
        window.local_size(), selected.local_size())
    sizes = fd.Context([
        fd.EqualityClass(_type=nm.FreeNumeric, bucket={window.local_size().uid},
                         canonical=nm.Integer(128)),
        fd.EqualityClass(_type=nm.FreeNumeric, bucket={selected.local_size().uid},
                         canonical=nm.Integer(512))])
    sized_slots = the(sizes.apply(model).cod()).shape()[1]
    assert sized_slots.local_size() == nm.Integer(640), \
        'the sum of two assigned sizes is the integer they sum to'
    assert sized_slots.uid == slots.uid


def check_a_concatenation_onto_a_declared_axis() -> None:
    '''A concatenation may fill an axis the model declared, where the sizes of the
    parts sum to the size of that axis. The deconcatenation of a latent axis `c`
    followed by the concatenation of the same parts onto `c` then has one domain and
    codomain, which is how a rotary embedding returns the channels to the axis every
    other wire carries. A declared axis of another size is refused. The expansion of
    concatenations reads the parts off the weaves, so a streamed consumer concatenates
    again onto `c` and a folded consumer leaves no concatenation.'''
    head, queries, channels, _, _ = attention_axes()
    pairs = cat.RawAxis.named('t')
    rotated = fd.DynamicName('z').capture(
        cat.RawAxis(_size=nm.Integer(2) * pairs.local_size()))
    unrotated = fd.DynamicName('\\bar{z}').capture(
        cat.RawAxis(_size=channels.local_size() - rotated.local_size()))
    shapes = ((head, queries, unrotated), (head, queries, rotated))
    cut = aops.DeconcatenateAxes.template(shapes, concatenated=channels)
    joined = aops.ConcatenateAxes.template(shapes, concatenated=channels)
    assert joined.operator.concatenated_axis() is channels
    assert joined.operator.parts() == (unrotated, rotated)
    assert joined.operator.part_reindexings == cut.operator.part_reindexings
    assert tuple(the(joined.cod()).shape()) == (head, queries, channels)
    round_trip = cut @ joined
    assert tuple(round_trip.dom()) == tuple(round_trip.cod()) == tuple(cut.dom()), \
        'the cut followed by the concatenation returns the array to the declared axis'
    try:
        aops.ConcatenateAxes.template(shapes, concatenated=cat.RawAxis.named('q'))
    except aops.PartsDoNotFillTheAxis:
        pass
    else:
        raise AssertionError('a declared axis of another size is rejected')
    streamed = h2m.recycle(joined @ over((head, queries, channels), exponential()))
    restreamed = concatenation_expansion.expand_concatenations(streamed)
    assert tuple(restreamed.cod()) == tuple(streamed.cod()), \
        'a streamed consumer concatenates its parts onto the declared axis again'
    assert [type(operator) for operator in expanded_operators(restreamed)].count(
        aops.ConcatenateAxes) == 1
    folded = h2m.recycle(streamed @ ops.Einops.template('h x c -> h x'))
    unfolded = concatenation_expansion.expand_concatenations(folded)
    assert not tuple(operator for operator in expanded_operators(unfolded)
                     if isinstance(operator, aops.ConcatenateAxes)), \
        'a concatenation onto a declared axis survived the expansion'
    assert tuple(unfolded.dom()) == tuple(folded.dom())
    assert tuple(unfolded.cod()) == tuple(folded.cod())
    residual, reverse = derivative.rule_for(joined.operator)(joined)
    assert residual == derivative.Residual()
    assert isinstance(reverse.operator, aops.DeconcatenateAxes)
    assert tuple(reverse.cod()) == tuple(joined.dom())


def check_a_softmax_refuses_the_expansion() -> None:
    '''A `SoftMax` over a concatenated axis reads every position of it to write every
    position, so its result over a part is no part of its result over the whole, and
    the expansion refuses it rather than splitting the softmax in two.'''
    head, queries, channels, window, selected = attention_axes()
    joined = aops.ConcatenateAxes.template(
        ((head, queries, window), (head, queries, selected)), name='slots')
    slots = the(joined.output_weaves).target().shape()[0]
    window_scores = cat.Array(R, (head, queries, window))
    selected_scores = cat.Array(R, (head, queries, selected))
    normalised = over((head, queries), ops.SoftMax.template())
    model = h2m.recycle(
        cat.Rearrangement((0, 1), (window_scores, selected_scores))
        @ joined
        @ normalised)
    try:
        concatenation_expansion.expand_concatenations(model)
    except concatenation_expansion.ConcatenatedAxisIsNotStreamed:
        return
    raise AssertionError('a softmax over a concatenated axis is rejected')


def check_reverse_rules() -> None:
    '''The reverse derivative of a merge and of a concatenation each run from the
    codomain of the operator to its domain, which is the shape the axiom of a reverse
    derivative category states, and neither declares a residual. A concatenation
    reverses into a deconcatenation, whose standard expansion is one `View` per part,
    and the deconcatenation reverses into the concatenation.'''
    head, queries, channels, window, selected = attention_axes()
    joined = aops.ConcatenateAxes.template(
        ((head, queries, window), (head, queries, selected)), name='slots')
    residual, reverse = derivative.rule_for(joined.operator)(joined)
    assert residual == derivative.Residual(), 'a concatenation is linear'
    assert tuple(reverse.dom()) == tuple(joined.cod())
    assert tuple(reverse.cod()) == tuple(joined.dom())
    assert isinstance(reverse.operator, aops.DeconcatenateAxes),         'a concatenation reverses into the deconcatenation of the same parts'
    views = tuple(operator for operator in expanded_operators(
                      standard_expansions.expand_standard(reverse))
                  if isinstance(operator, ops.View))
    assert len(views) == 2, 'the expansion reads the cotangent of each part by a View'
    residual, forward_again = derivative.rule_for(reverse.operator)(reverse)
    assert residual == derivative.Residual(), 'a deconcatenation is linear'
    assert isinstance(forward_again.operator, aops.ConcatenateAxes)
    assert forward_again.operator.part_reindexings == joined.operator.part_reindexings
    assert tuple(forward_again.dom()) == tuple(joined.dom())
    assert tuple(forward_again.cod()) == tuple(joined.cod())
    blocks, offsets = (cat.RawAxis.named(name) for name in ('P', 'u'))
    split = cat.StrideMorphism(
        _dom=(blocks, offsets),
        _cod_stride_shift=((cat.RawAxis.named('B'),
                            (offsets.local_size(), nm.Integer(1)), nm.Integer(0)),),
        name=fd.DynamicName('blk'))
    merge = aops.CovariantView.template(split, name='mrg')
    residual, reverse = derivative.rule_for(merge.operator)(merge)
    assert residual == derivative.Residual(), 'a merge is linear'
    assert tuple(reverse.dom()) == tuple(merge.cod())
    assert tuple(reverse.cod()) == tuple(merge.dom())


CHECKS = (
    check_guarded_reads,
    check_concatenation_expansion,
    check_concatenation_expansion_through_a_block,
    check_the_box_states_the_core_over_every_head,
    check_parts_fill_the_axis,
    check_the_concatenated_axis_follows_its_parts,
    check_a_concatenation_onto_a_declared_axis,
    check_a_softmax_refuses_the_expansion,
    check_reverse_rules,
)


if __name__ == '__main__':
    forms = attention_core_forms()
    print('==== the core, with the latents concatenated ====')
    print(ad.listing(forms.concatenated))
    print('==== the same core, expanded ====')
    print(ad.listing(
        concatenation_expansion.expand_concatenations(forms.concatenated)))
    print('==== the core in a box computed once per head, expanded ====')
    print(ad.listing(concatenation_expansion.expand_concatenations_through_blocks(
        forms.concatenated_in_a_box)))
    for check in CHECKS:
        check()
        print(f'  ok    {check.__name__}')
    print('all advanced axis dynamics checks passed')
