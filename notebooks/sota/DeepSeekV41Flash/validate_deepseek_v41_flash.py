'''Check every claim this package makes about DeepSeek-V4.1-Flash.

Written by Claude Opus 5, effort high.

    python notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash.py

The modules beside this one write the expression and the claims made about them live
here, one `check_` function per part of the model. The notebook that draws the model
keeps the prose and the figures, so a reader who wants the evidence for a sentence has
one place to look and the notebook reads as an account rather than as a test.

Every check is structural. The axes, the degrees, the reindexings, the weaves, the slots
and the affine forms are compared, and no listing is diffed except the two spellings of
the attention core, which `discovering_broadcasts.normalise_for_comparison` makes
comparable. A uid is random per process, so a listing is stable within a run and not
across runs, where a shape is stable in both.
'''
from __future__ import annotations

import dataclasses
import pathlib
import sys
from collections.abc import Callable

# The repository root, for a run by path rather than through
# `validations/run_validations.py`, which puts it on PYTHONPATH itself.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.algebra.concatenation_expansion as concatenation_expansion  # noqa: E402
import advanced_axis_dynamics.algebra.mark_sparse_codomains as mark_sparse_codomains  # noqa: E402
import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa: E402
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation  # noqa: E402
import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import agent_display as ad  # noqa: E402
import algebra.discovering_broadcasts as discovering_broadcasts  # noqa: E402
import algebra.write_axis_exponents as write_axis_exponents  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import para.algebra.para_sparse_expansion as para_sparse_expansion  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import para.data_structure.ParaBlockOperator as para_block_operator  # noqa: E402
import para.data_structure.ParaWrap as para_wrap  # noqa: E402
import data_transfer.broadcast_occurrences as broadcast_occurrences  # noqa: E402
import term_utilities.generate_config as gc  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
import utilities.wording_json as wording_json  # noqa: E402
import websocket_transfer.auxiliary_information as auxiliary_information  # noqa: E402
import websocket_transfer.send_morphism as send_morphism  # noqa: E402
from para.data_structure.ParaBlockOperator import (  # noqa: E402
    bare_box_of, slots_dropped, slots_grabbed)

import notebooks.display.axis_sizes as axis_sizes  # noqa: E402
import notebooks.display.expand_with_parameters as expand_with_parameters  # noqa: E402
import notebooks.display.explain_operators as explain_operators  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.tape_presentation as tape_presentation  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.attention_core as attention_core  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions as block_titles_and_descriptions  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.attention_modes as attention_modes  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.candidate_pool as candidate_pool  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.custom_operations as custom_operations  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.grouped_output as grouped_output  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.layer_stack as layer_stack  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.lightning_indexer as lightning_indexer  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.mixture_of_experts as mixture_of_experts  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.operator_explanations as operator_explanations  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.reference_links as reference_links  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.single_pass_mhc as single_pass_mhc  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.token_compressors as token_compressors  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.whole_model as whole_model  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (  # noqa: E402
    axes, axis_name, node_with_box_named, node_with_operator, over)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (  # noqa: E402
    B, C, CKVd, CKVe, COLLAPSE, GROUP_COUNTER, KId, KIe, N, P, Q, QR, R, SLOT_CKVd,
    SLOT_CKVe, SLOT_KId, SLOT_POOL, SLOT_SELd, SLOT_SELe, WKV, X, a, b, c, d, e, f, g,
    h, i, j, kexp, m, n, npool, nsel, o, q, s, state, u, w, x)

i_x, i_b, i_r, i_P, i_u, j_w, j_s, j_p = (
    nm.FreeNumeric.named(name) for name in
    ('i_x', 'i_b', 'i_r', 'i_P', 'i_u', 'j_w', 'j_s', 'j_p'))

WINDOW_SIZES = {a.local_size(): 2, b.local_size(): 150, w.local_size(): 128}
FOUR_TOKENS = {a.local_size(): 2, b.local_size(): 4}
LONG_CONTEXT = {a.local_size(): 2, b.local_size(): 600, nsel: 512}
BLOCK_SIZES = {a.local_size(): 2, b.local_size(): 600, u.local_size(): 8,
               P.local_size(): 150, npool: 64}
RUNTIME_SLOTS = {SLOT_CKVe, SLOT_SELe, SLOT_CKVd, SLOT_KId, SLOT_POOL, SLOT_SELd}


def reference_window_tokens(query: int, window: int) -> list[int]:
    '''The tokens `get_window_topk_idxs` names for one prefill query: the row starts at
    the window's start clamped to zero, and a slot past the query is marked -1.'''
    start = max(0, query - window + 1)
    return [token for token in range(start, start + window) if token <= query]


def sparse_axis_names(target: fd.GeneralTerm) -> set[str]:
    '''The name of every `AffineGuards.AffineSparseAxis` in `target`.'''
    return {axis_name(axis) for axis in
            tutil.type_search(AffineGuards.AffineSparseAxis, target)}


def degree_names(target: fd.GeneralTerm) -> set[str]:
    '''The name of every axis any operation of `target` is broadcast over.'''
    return {axis_name(axis)
            for operation in tutil.type_search(cat.Broadcasted, target)
            for axis in operation.degree()}


def check_the_declared_axes() -> None:
    '''*Axes and objects*: the query axis is sized by the ratio and the encoder's
    entries, the decoder's entries by the query axis, every other size is a free symbol,
    and the six tape slots are six.'''
    assert x.local_size() == a.local_size() * b.local_size()
    assert B.local_size() == x.local_size()
    assert all(isinstance(axis.local_size(), nm.FreeNumeric) for axis in
               (m, h, c, q, o, g, j, w, a, b, u, P, C, i, d, e, f, n, N))
    assert all(isinstance(count, nm.FreeNumeric) for count in (nsel, kexp, npool))
    assert [count.to_latex() for count in (nsel, kexp, npool)] == ['|s|', '|k|', '|p|'], \
        'a selection count is named the way a size is named, between absolute bars'
    assert s.uid._name.to_latex() == 's', \
        'the dense axis the selection hands its values out on takes the letter alone'
    assert len(RUNTIME_SLOTS) == 6


def check_the_custom_operations() -> None:
    '''*Custom operations*: a parameter array is a `Linear` with no inputs, an
    elementwise product is an `Einops` along a shared letter, every pointwise map reads
    one number, and the model holds no opaque operator.'''
    sink = custom_operations.weights('\\mathrm{sink}', (h,))
    assert sink.has_empty_domain() and axes(sink) == ([], [['h']])
    assert isinstance(sink.operator, ops.Linear)
    assert axes(custom_operations.multiply_along(f)) == ([['f'], ['f']], [['f']])
    for pointwise in (custom_operations.indicator(), custom_operations.exponential(),
                      custom_operations.reciprocal(), custom_operations.sigmoid(),
                      custom_operations.doubled_sigmoid(),
                      custom_operations.sigmoid_weighted_input(),
                      custom_operations.sqrt_softplus()):
        assert isinstance(pointwise.operator, ops.Arithmetic)
        assert axes(pointwise) == ([[]], [[]]), 'a pointwise map reads one number'
    assert not list(tutil.type_search(ops.GenericOperator, whole_model.v41_flash)), \
        'every operation of the model is one the standard operator set states'


def check_the_compressors() -> None:
    '''*The compressor at two ratios*: both read the hidden state and return the
    compressed entries of their own entry axis.'''
    assert axes(token_compressors.pool_tokens_into_entries(b)) == ([['x', 'm']],
                                                                   [['b', 'c']])
    assert axes(token_compressors.project_tokens_into_entries(B)) == ([['x', 'm']],
                                                                      [['B', 'c']])


def check_the_sliding_window() -> None:
    '''*The sliding window and the attention core*: the window's row is `i_x - j_w`, its
    slot axis is `w|x` with its empty positions at the end, and the tokens its live
    slots read at every query of a 300-token sequence are the tokens the reference's own
    row names.'''
    window_axis = attention_core.window_view().cod()[0].shape()[1]
    assert isinstance(window_axis, AffineGuards.AffineSparseAxis)
    assert window_axis.guides == (x,)
    assert window_axis.stride == nm.Integer(-1) and window_axis.shift == nm.Integer(0)
    assert window_axis.guard_form((i_x,), j_w) == nm.collect_like_terms(i_x - j_w)
    assert window_axis.empty_end() is AffineGuards.EmptyEnd.LAST
    assert window_axis.local_size() == w.local_size()
    for query in range(300):
        live = mark_sparse_domains.live_positions(window_axis, (query,), WINDOW_SIZES)
        assert live == list(range(min(128, query + 1))), 'slot 0 is the current token'
        assert sorted(query - slot for slot in live) == reference_window_tokens(
            query, 128)
    assert axes(attention_core.window_view()) == ([['x', 'c']], [['x', 'w|x', 'c']])
    assert axes(attention_core.window_kv()) == ([['x', 'm']], [['x', 'w|x', 'c']])


def check_the_concatenated_core() -> None:
    '''*The sliding window and the attention core*: the two kinds of latent are
    concatenated along one dense slot axis that references its two parts and is sized
    `|w| + |s|`, and the expansion of that form through the box is the two-branch
    core.'''
    concatenated = attention_core.attend_over_all_heads_with_concatenated_slots()
    two_branch = attention_core.attend_over_all_heads_with_entries()
    concatenation = node_with_operator(aops.ConcatenateAxes, concatenated).operator
    slots = concatenation.concatenated_axis()
    assert [axis_name(axis) for axis in concatenation.parts()] == ['w', 's']
    assert isinstance(slots, AxisConcatenation.ConcatenatedAxis)
    assert [part.uid for part in slots.parts] == [w.uid, s.uid], \
        'the concatenated axis references the two slot axes'
    assert axis_name(slots) == 'w + s', 'the core alone holds the raw slot axes'
    assert slots.local_size() == nm.Addition.template(w.local_size(), s.local_size())
    assert not isinstance(slots, AffineGuards.AffineSparseAxis), \
        'the concatenated axis is dense'
    assert axes(concatenated) == axes(two_branch)
    expanded = concatenation_expansion.expand_concatenations_through_blocks(
        concatenated)
    assert not [root for root in tutil.type_search(cat.Broadcasted, expanded)
                if isinstance(root.operator,
                              (aops.ConcatenateAxes, ops.BlockOperator))], \
        'the expansion writes the box out and leaves no concatenation'
    derived = ad.listing(discovering_broadcasts.normalise_for_comparison(expanded))
    by_hand = ad.listing(discovering_broadcasts.normalise_for_comparison(two_branch))
    assert derived == by_hand, derived + '\n' + by_hand


def check_the_boxed_core() -> None:
    '''*The sliding window and the attention core*: the core is one box computed once
    per head, reading the sink logit and the query at the head and the concatenated
    latents whole, and both boxes are confirmed against the core written out over every
    head.'''
    core = attention_core.attend_over_window_and_entries()
    core_box = node_with_operator(ops.BlockOperator, core)
    sink_target, query_target, latent_target = (
        weave.target().shape() for weave in core_box.input_weaves)
    assert axes(core) == ([['h', 'x', 'c'], ['x', 'w', 'c'], ['x', 's', 'c']],
                          [['h', 'x', 'c']])
    assert axes(attention_core.attend_over_window()) == (
        [['h', 'x', 'c'], ['x', 'w', 'c']], [['h', 'x', 'c']])
    assert tuple(core_box.degree()) == (h,), 'the core is computed once per head'
    assert [reindexing.mapping for reindexing in core_box.reindexings] == [
        (0,), (0,), ()], 'the sink and the query read the head and the latents do not'
    assert not tuple(sink_target), 'the sink logit is one number per head'
    assert [axis_name(axis) for axis in query_target] == ['x', 'c']
    assert [axis_name(axis) for axis in latent_target] == ['x', 'w + s', 'c'], \
        'the concatenated slots stand between the queries and the channels'
    assert latent_target[1].local_size() == nm.Addition.template(w.local_size(),
                                                                 s.local_size())
    assert all(weave._shape[0] is cat.WeaveMode.TILED
               for weave in core_box.output_weaves), 'the result carries the head'
    assert 'h' not in {axis_name(axis) for axis in
                       tutil.type_search(cat.Axis, core_box.operator.block)}
    for discovered in (attention_core.CORE_ON_CONCATENATED_SLOTS,
                       attention_core.CORE_ON_THE_WINDOW):
        assert discovered.is_confirmed(), discovered.confirmation.named_difference


def check_the_grouped_output_projection() -> None:
    '''*The grouped low-rank output projection*: the head split is an affine view, the
    weight produces the group and the rank beside the group it is read at, the
    diagonalisation keeps the entries where the two groups agree, and the box is
    computed once per query.'''
    projection = grouped_output.output_projection()
    projection_box = node_with_operator(ops.BlockOperator, projection)
    group_weight = node_with_operator(
        ops.Linear, grouped_output.contract_each_group_against_its_weight())
    diagonal = node_with_operator(
        ops.View, grouped_output.contract_each_group_against_its_weight())
    assert axes(grouped_output.split_heads_into_groups()) == ([['h', 'c']],
                                                              [['g', 'j', 'c']])
    assert axes(projection) == ([['x', 'h', 'c']], [['x', 'm']])
    assert tuple(projection_box.degree()) == (x,), 'the projection runs once per query'
    assert [axis_name(axis) for axis in group_weight.cod()[0].shape()] == [
        'g', 'g', 'o'], \
        'the weight produces the group and the rank beside the group it is read at'
    assert diagonal.reindexings[0].mapping == (0, 0, 1), \
        'the diagonal keeps the entries where the two groups agree'
    assert grouped_output.PROJECTION_CONFIRMATION.is_confirmed(), \
        grouped_output.PROJECTION_CONFIRMATION.named_difference


def check_the_lightning_indexer() -> None:
    '''*The lightning indexer*: both halves read their keys back from each entry, merge
    each entry's slots onto the queries it serves, and score one query at a time, and
    the forms the two reads leave are the reference's reachability condition.'''
    for box, indexer, entry_axis, reach, selected, ratio in (
            (lightning_indexer.INDEXER_e, lightning_indexer.ENCODER_INDEXER, b,
             lightning_indexer.reach_e, lightning_indexer.s_e, a.local_size()),
            (lightning_indexer.INDEXER_d, lightning_indexer.DECODER_INDEXER, B,
             lightning_indexer.reach_d, lightning_indexer.s_d, nm.Integer(1))):
        assert tuple(box.dom()) == (QR, cat.Array(R, (entry_axis, d)), state)
        assert tuple(box.cod()) == (cat.Array(R, (x, reach)),)
        assert box.cod()[0].shape()[0] == x, 'the query axis stays dense'
        entry = axis_name(entry_axis)
        assert sparse_axis_names(box) == {f'r|{entry}', 'r|x'}
        back_reach, = (axis for axis in
                       tutil.type_search(AffineGuards.AffineSparseAxis, box)
                       if axis_name(axis) == f'r|{entry}')
        assert back_reach.guides == (entry_axis,)
        assert back_reach.empty_end() is AffineGuards.EmptyEnd.LAST
        assert back_reach.guard_form((i_b,), i_r) == nm.collect_like_terms(i_b - i_r)
        assert isinstance(reach, AffineGuards.AffineSparseAxis)
        assert reach.guides == (x,)
        assert reach.empty_end() is AffineGuards.EmptyEnd.LAST
        assert reach.guard_form((i_x,), i_r) == nm.collect_like_terms(
            i_x - ratio * i_r - ratio + nm.Integer(1))
        assert isinstance(selected, AffineGuards.AffineSparseAxis)
        assert selected.guard_form((i_x,), j_s) == nm.collect_like_terms(
            i_x - ratio * j_s - ratio + nm.Integer(1))
        assert selected.local_size() == nsel
        merge = node_with_operator(aops.CovariantView, box)
        assert tuple(merge.operator.reindexing.cod()) == (x,)
        assert [axis_name(axis) for axis in merge.cod()[0].shape()] == [
            'x', 'r|x', 'd'], \
            'the merge writes the slots and the key width of each entry onto the queries'
        assert tuple(merge.input_weaves[0].target().shape()) == (entry_axis,), \
            'the merge reads the entries and broadcasts over the offsets of each group'
        assert tuple(merge.degree()) == (reach, d), \
            'the merge is broadcast over the slots and the channels'
        scoring = node_with_operator(ops.BlockOperator, box.operator.block)
        assert tuple(scoring.degree()) == (x,), 'the scoring runs once per query'
        assert indexer.scoring.is_confirmed(), \
            indexer.scoring.confirmation.named_difference
    assert axes(lightning_indexer.index_keys(b)) == ([['b', 'c']], [['b', 'd']])


def check_the_entry_gather() -> None:
    '''*The lightning indexer*: each half's Top-512 hands out its distances on that
    half's slot axis, and the gather reads one compressed entry per slot, broadcast over
    the slots.'''
    assert tuple(over((x,), lightning_indexer.ENCODER_SELECT).cod()) == (
        lightning_indexer.SELe,)
    assert tuple(over((x,), lightning_indexer.DECODER_SELECT).cod()) == (
        lightning_indexer.SELd,)
    encoder_gather = lightning_indexer.gather(b, (a,), lightning_indexer.s_e)
    assert tuple(encoder_gather.dom()) == (lightning_indexer.SELe, CKVe)
    assert tuple(encoder_gather.cod()) == (
        cat.Array(R, (x, lightning_indexer.s_e, c)),)
    read = node_with_operator(dst.IndexSelect, encoder_gather)
    assert not list(tutil.type_search(dst.Select, encoder_gather)), \
        'the gather holds one selection and it is an IndexSelect'
    assert [axis_name(axis) for axis in read.degree()] == ['x', 's|x', 'c'], \
        'the slot axis is in the degree, so the gather is broadcast over the slots'
    assert not tuple(read.input_weaves[0].target().shape()), \
        'the index is a rank-0 Natural read at the query and the slot'
    assert isinstance(read.input_weaves[0].datatype, cat.Natural)
    payload_target, = read.input_weaves[1].target().shape()
    assert axis_name(payload_target) == 'r|x', \
        'the payload keeps the entries a query has reached as its target'
    assert not tuple(read.output_weaves[0].target().shape())
    assert [reindexing.mapping for reindexing in read.reindexings] == [(0, 1), (0, 2)], \
        'the positions are read at the query and the slot and the payload at the query ' \
        'and the channel'
    assert sparse_axis_names(encoder_gather) == {'r|b', 'r|x', 's|x'}
    assert tuple(lightning_indexer.gather(B, (), lightning_indexer.s_d).dom()) == (
        lightning_indexer.SELd, CKVd)


def check_the_reachable_counts() -> None:
    '''*The lightning indexer*: the entries and the slots a query reaches are the
    reference's `compress_lens` at both ratios, at four tokens and at 1200.'''
    assert [mark_sparse_domains.live_positions(
        lightning_indexer.reach_e, (query,), FOUR_TOKENS)
        for query in range(4)] == [[], [0], [0], [0, 1]], \
        'at four tokens and ratio 2 the first query reaches no entry and the last ' \
        'reaches both'
    for query in (0, 1, 2, 3, 100, 511, 512, 1022, 1023, 1024, 1025, 1199):
        reachable_count = (query + 1) // 2
        assert mark_sparse_domains.live_positions(
            lightning_indexer.reach_e, (query,), LONG_CONTEXT) == list(
                range(reachable_count))
        assert mark_sparse_domains.live_positions(
            lightning_indexer.s_e, (query,), LONG_CONTEXT) == list(
                range(min(512, reachable_count)))
        assert mark_sparse_domains.live_positions(
            lightning_indexer.reach_d, (query,), LONG_CONTEXT) == list(
                range(query + 1))
        assert mark_sparse_domains.live_positions(
            lightning_indexer.s_d, (query,), LONG_CONTEXT) == list(
                range(min(512, query + 1)))


def check_the_candidate_blocks() -> None:
    '''*The hierarchical sparse indexer*: the block split pulls the distances' form back
    onto the offsets, the fold derives the blocks' form from it, and a query reaches the
    blocks its distances cover.'''
    assert isinstance(candidate_pool.u_reach, AffineGuards.AffineSparseAxis)
    assert candidate_pool.u_reach.guides == (x, P)
    assert candidate_pool.u_reach.guard_form((i_x, i_P), i_u) == nm.collect_like_terms(
        i_x - u.local_size() * i_P - i_u)
    pulled_offsets = mark_sparse_domains.pull_back_sparse_axis(
        candidate_pool.reach_d, candidate_pool.BLOCK_SPLIT)[1]
    assert pulled_offsets.guard_form((i_x, i_P), i_u) == (
        candidate_pool.u_reach.guard_form((i_x, i_P), i_u))
    assert isinstance(candidate_pool.P_reach, AffineGuards.AffineSparseAxis)
    assert candidate_pool.P_reach.guides == (x,)
    assert candidate_pool.P_reach.empty_end() is AffineGuards.EmptyEnd.LAST
    assert candidate_pool.P_reach.guard_form((i_x,), i_P) == nm.collect_like_terms(
        i_x - u.local_size() * i_P)
    assert isinstance(candidate_pool.p_axis, AffineGuards.AffineSparseAxis)
    assert candidate_pool.p_axis.guard_form((i_x,), j_p) == nm.collect_like_terms(
        i_x - u.local_size() * j_p)
    for query in (0, 1, 7, 8, 9, 63, 500, 511, 512, 1199):
        reachable_blocks = -(-(query + 1) // 8)
        assert mark_sparse_domains.live_positions(
            candidate_pool.P_reach, (query,), BLOCK_SIZES) == list(
                range(reachable_blocks))
        assert mark_sparse_domains.live_positions(
            candidate_pool.p_axis, (query,), BLOCK_SIZES) == list(
                range(min(64, reachable_blocks)))


def check_the_boxed_candidate_pool() -> None:
    '''*The hierarchical sparse indexer*: the pool is one box computed once per token,
    holding no token axis in any degree and carrying positions and no values, and a
    Reindex layer reads its scores at the distances it holds.'''
    pool = candidate_pool.CANDIDATE_POOL
    assert candidate_pool.KEEP_BLOCKS.operator.form is dst.SelectionForm.ONLY_SELECTION
    assert isinstance(candidate_pool.KEEP_BLOCKS.cod()[0].datatype, cat.Natural)
    assert axes(candidate_pool.POOL_BODY) == ([['r|x']], [['C']])
    assert axes(pool) == ([['x', 'r|x']], [['x', 'C']])
    assert tuple(pool.degree()) == (x,), 'the pool runs once per token'
    assert [reindexing.mapping for reindexing in pool.reindexings] == [(0,)]
    assert 'x' not in degree_names(candidate_pool.POOL_BODY), \
        'no operation of the body is broadcast over the tokens'
    assert candidate_pool.POOL_CONFIRMATION.is_confirmed(), \
        candidate_pool.POOL_CONFIRMATION.named_difference
    assert pool.operator.block.aesthetics.title == '\\text{Candidate Pool}', \
        'the block carries the long title'
    assert (pool.operator.name.to_bodies() == 'Pool'
            and pool.operator.name.settings.bold), \
        'the box carries the short name, in bold'
    assert isinstance(pool.cod()[0].datatype, cat.Natural)
    assert not list(tutil.type_search(dst.SparseAxis, pool))
    assert not list(tutil.type_search(ops.Arithmetic, pool))
    assert axes(candidate_pool.restrict_to_candidates()) == (
        [['x', 'r|x'], ['x', 'C']], [['x', 'C']])
    assert axes(over((x,), candidate_pool.REINDEX_SELECT)) == (
        [['x', 'C'], ['x', 'C']], [['x', 's|x']])
    assert tuple(over((x,), candidate_pool.REINDEX_SELECT).cod()) == (
        lightning_indexer.SELd,)


MODES = (
    ('SWA', attention_modes.window_attention(), set(), set(), 0, {'w|x'}),
    ('encoder Full', attention_modes.encoder_full_attention(),
     {SLOT_CKVe, SLOT_SELe}, set(), 1, {'w|x', 'r|b', 'r|x', 's|x'}),
    ('decoder Full', attention_modes.decoder_full_attention(),
     {SLOT_CKVd, SLOT_KId, SLOT_POOL, SLOT_SELd}, set(), 2,
     {'w|x', 'r|B', 'r|x', 'u|x,P', 'P|x', 'p|x', 's|x'}),
    ('Reindex', attention_modes.reindex_attention(), {SLOT_SELd},
     {SLOT_CKVd, SLOT_KId, SLOT_POOL}, 1, {'w|x', 'r|B', 'r|x', 's|x'}),
    ('encoder Reuse', attention_modes.encoder_reuse_attention(), set(),
     {SLOT_CKVe, SLOT_SELe}, 0, {'w|x', 'r|b', 'r|x', 's|x'}),
    ('decoder Reuse', attention_modes.decoder_reuse_attention(), set(),
     {SLOT_CKVd, SLOT_SELd}, 0, {'w|x', 'r|B', 'r|x', 's|x'}),
    ('Reindex group Reuse', attention_modes.reindex_group_reuse_attention(), set(),
     {SLOT_CKVd, SLOT_SELd}, 0, {'w|x', 'r|B', 'r|x', 's|x'}),
)


def concatenated_slot_labels(target: fd.GeneralTerm) -> set[str]:
    '''The label of the axis every `aops.ConcatenateAxes` of `target` produces, read
    off its output weave.'''
    return {axis_name(node.output_weaves[0].target().shape()[0])
            for node in tutil.type_search(cat.Broadcasted, target)
            if isinstance(node.operator, aops.ConcatenateAxes)}


def check_the_concatenated_slots_follow_their_parts() -> None:
    '''*The sliding window and the attention core*: a mode composes the guarded window
    slots `w|x` and the guarded selected slots `s|x` into the core written over the raw
    axes `w` and `s`, and the concatenated axis, which references its parts, is
    labelled `w|x + s|x` in every mode that holds entries and appears in no mode that
    holds the window alone.'''
    for name, mode, _, _, _, sparse_axes in MODES:
        expected = {'w|x + s|x'} if 's|x' in sparse_axes else set()
        assert concatenated_slot_labels(mode) == expected, name


def check_the_attention_modes() -> None:
    '''*CSA2*: every mode reads the hidden state and returns the hidden state, touches
    the slots its table row names, holds as many selections as that row names, and reads
    each entry it gathers with an `IndexSelect` broadcast over the slots.'''
    for name, mode, drops, grabs, selections, sparse_axes in MODES:
        assert tuple(mode.dom()) == tuple(mode.cod()) == (state,), name
        assert slots_dropped(mode) == drops, name
        assert slots_grabbed(mode) == grabs, name
        assert len(list(tutil.type_search(dst.TopK, mode))) == selections, name
        assert sparse_axis_names(mode) == sparse_axes, name
        assert not list(tutil.type_search(dst.SparseAxis, mode)), name
        assert not list(tutil.type_search(Para.StreamGrab, mode)), name
        assert not list(tutil.type_search(dst.Select, mode)), name
        for read in tutil.type_search(cat.Broadcasted, mode):
            if isinstance(read.operator, dst.IndexSelect):
                assert not tuple(read.output_weaves[0].target().shape()), name
                assert isinstance(read.input_weaves[0].datatype, cat.Natural), name
                assert not tuple(read.input_weaves[0].target().shape()), name


def check_the_expert_gate() -> None:
    '''*The MoE*: the gate is one box reading a token's hidden state and handing out the
    gates on the sparse expert axis, the axis the box hands out is the axis the experts
    consume, and the mixture's body holds that one nested box.'''
    gate, expert_slots = mixture_of_experts.expert_gate()
    assert isinstance(gate.operator, ops.BlockOperator)
    assert tuple(gate.dom()) == (mixture_of_experts.TOKEN_STATE,)
    assert tuple(gate.cod()) == (cat.Array(R, (expert_slots,)),)
    assert axes(gate) == ([['m']], [['k/e']])
    assert gate.cod()[0].shape()[0].uid == expert_slots.uid, \
        'recycling the block keeps the axis the experts are selected on'
    assert isinstance(expert_slots, dst.SparseAxis)
    assert gate.operator.block.aesthetics.title == '\\text{Expert Gate}', \
        'the block carries the long title'
    assert (gate.operator.name.to_bodies() == mixture_of_experts.GATE_BOX
            and gate.operator.name.settings.bold), \
        'the box carries the short name, in bold'
    assert not tuple(gate.degree()), 'the gate reads one token and is broadcast over ' \
        'nothing'
    nested = [root for root in tutil.type_search(
        cat.Broadcasted, mixture_of_experts.MIXTURE_BODY)
        if isinstance(root.operator, ops.BlockOperator)]
    assert [box.operator.name.to_bodies() for box in nested] == [
        mixture_of_experts.GATE_BOX], 'the body holds the gate box and no other'
    assert node_with_box_named(
        mixture_of_experts.GATE_BOX, mixture_of_experts.MIXTURE) is nested[0]
    assert {axis_name(axis) for axis in
            tutil.type_search(dst.SparseAxis, nested[0].operator.block)} == {'k/e'}
    assert [type(root.operator).__name__ for root in tutil.type_search(
        cat.Broadcasted, nested[0].operator.block)
        if isinstance(root.operator, (dst.TopK, dst.Select))] == ['TopK', 'Select'], \
        'the selection and the gather sit inside the gate box'


def check_the_mixture() -> None:
    '''*The MoE*: the mixture is one box computed once per token, its body holds no
    token axis, the bias addition is broadcast over the experts with both operands read
    at the expert, and the box is confirmed against the mixture written out over every
    token.'''
    picked_gates, expert_slots = mixture_of_experts.router()
    bias_addition = node_with_operator(ops.AdditionOp,
                                       mixture_of_experts.add_correction_bias())
    assert axes(picked_gates) == ([['m']], [['k/e']])
    assert axes(mixture_of_experts.normalise_gates(expert_slots)) == ([['k/e']],
                                                                      [['k/e']])
    assert [axis_name(axis) for axis in bias_addition.degree()] == ['e'], \
        'the bias addition is broadcast over the experts'
    assert not tuple(bias_addition.output_weaves[0].target().shape())
    assert [reindexing.mapping for reindexing in bias_addition.reindexings] == [
        (0,), (0,)], 'both operands of the addition are read at the expert'
    assert axes(mixture_of_experts.MIXTURE_BODY) == ([['m']], [['m']])
    assert axes(mixture_of_experts.MIXTURE) == ([['x', 'm']], [['x', 'm']])
    assert tuple(mixture_of_experts.MIXTURE.degree()) == (x,), \
        'the mixture runs once per token'
    assert [reindexing.mapping
            for reindexing in mixture_of_experts.MIXTURE.reindexings] == [(0,)]
    assert 'x' not in {axis_name(axis) for axis in
                       tutil.type_search(cat.Axis,
                                         mixture_of_experts.MIXTURE_BODY)}
    assert mixture_of_experts.MIXTURE_CONFIRMATION.is_confirmed(), \
        mixture_of_experts.MIXTURE_CONFIRMATION.named_difference


def check_single_pass_mhc() -> None:
    '''*Single-Pass mHC*: the coefficients are one box computed once per token, a
    sublayer reads and returns the four streams and the collapse vector, and a body that
    does not read and write the hidden state alone is refused.'''
    coefficients_box = node_with_operator(ops.BlockOperator,
                                          single_pass_mhc.mixing_coefficients())
    assert axes(single_pass_mhc.predict_one_token()) == ([['n', 'm']],
                                                         [['n'], ['n'], ['n', 'N']])
    assert axes(single_pass_mhc.mixing_coefficients()) == (
        [['x', 'n', 'm']], [['x', 'n'], ['x', 'n'], ['x', 'n', 'N']])
    assert tuple(coefficients_box.degree()) == (x,), \
        'the prediction runs once per token'
    assert 'x' not in {axis_name(axis) for axis in tutil.type_search(
        cat.Axis, single_pass_mhc.predict_one_token())}
    assert single_pass_mhc.COEFFICIENTS_CONFIRMATION.is_confirmed(), \
        single_pass_mhc.COEFFICIENTS_CONFIRMATION.named_difference
    body = single_pass_mhc.predict_one_token()
    maps = [node for node in tutil.type_search(cat.Broadcasted, body)
            if isinstance(node.operator, ops.Linear)]
    assert sorted((node.operator.name.to_bodies(), node.operator.bias,
                   [axis_name(axis) for axis in node.cod()[0].shape()])
                  for node in maps) == [
        ('H0', True, ['n']), ('H1', True, ['n']), ('H2', True, ['n', 'N'])], \
        'each set of coefficients has a biased linear map of its own'
    assert all([axis_name(axis) for axis in node.dom()[0].shape()] == ['n', 'm']
               for node in maps), 'each map reads the four normalised streams'
    assert not any(isinstance(node.operator, (ops.View, ops.SoftMax))
                   for node in tutil.type_search(cat.Broadcasted, body)), \
        'no slice cuts a projection and no softmax stands before the Sinkhorn rounds'
    rounds, = (block for block in tutil.type_search(cat.Block, body)
               if block.block_tag.repetition != nm.Integer(1))
    assert rounds.block_tag.repetition == nm.Integer(20), \
        'the released hc_sinkhorn_iters rounds follow one exponential'
    assert [type(node.operator) for node in tutil.type_search(cat.Broadcasted, rounds)
            ] == [ops.L1Norm, ops.L1Norm]
    sublayer = single_pass_mhc.mhc_sublayer(layer_stack.SWA)
    assert tuple(sublayer.dom()) == tuple(sublayer.cod()) == (COLLAPSE, X)
    try:
        single_pass_mhc.mhc_sublayer(attention_core.window_kv())
        latent_body_accepted = True
    except ValueError as refusal:
        latent_body_accepted = False
        assert str(axes(attention_core.window_kv())) in str(refusal)
    assert not latent_body_accepted


def check_the_layer_plan() -> None:
    '''*The encoder, the decoder and the Causal Encoder-Decoder*: every repeated block
    returns its own domain, the counts add to forty, every slot dropped in a group is
    grabbed in it, and a group's own slots carry the counter of that group.'''
    for block in (layer_stack.swa_block, layer_stack.encoder_reuse_block,
                  layer_stack.encoder_group, layer_stack.decoder_reuse_block,
                  layer_stack.reindex_reuse_block, layer_stack.reindex_group):
        assert tuple(block.dom()) == tuple(block.cod()) == (COLLAPSE, X), (
            f'{axes(block)} is a repeated block carrying more than the residual and '
            'the collapse vector')
    assert (layer_stack.SWA_LAYERS
            + layer_stack.ENCODER_GROUPS * (1 + layer_stack.ENCODER_REUSE)
            + (1 + layer_stack.DECODER_REUSE)
            + layer_stack.DECODER_GROUPS * (1 + layer_stack.DECODER_REUSE)) == 40
    assert len(list(tutil.type_search(dst.TopK, layer_stack.encoder_group))) == 2
    assert not slots_dropped(layer_stack.swa_block)
    assert not slots_grabbed(layer_stack.swa_block)
    assert slots_dropped(layer_stack.encoder_group) == slots_grabbed(
        layer_stack.encoder_group) == {SLOT_CKVe, SLOT_SELe}
    assert slots_dropped(layer_stack.decoder) == slots_grabbed(
        layer_stack.decoder) == {SLOT_CKVd, SLOT_KId, SLOT_POOL, SLOT_SELd}
    assert slots_grabbed(layer_stack.decoder_reuse_block) == {SLOT_CKVd, SLOT_SELd}
    assert not slots_dropped(layer_stack.decoder_reuse_block)
    assert slots_grabbed(layer_stack.reindex_group) == {
        SLOT_CKVd, SLOT_KId, SLOT_POOL, SLOT_SELd}
    assert slots_dropped(layer_stack.reindex_group) == {SLOT_SELd}
    assert [len(bare_box_of(box).operator.grabs)
            for box in (layer_stack.REINDEX, layer_stack.REUSED)] == [3, 2], \
        'a Para box lists the grabs of its body'
    assert [len(bare_box_of(box).operator.drops)
            for box in (layer_stack.FULLD, layer_stack.REINDEX)] == [4, 1], \
        'a Para box lists the drops of its body'
    for group in (layer_stack.encoder_group, layer_stack.reindex_group):
        loop_seeds = [*tutil.type_search(Para.LoopGrab, group),
                      *tutil.type_search(Para.LoopDrop, group)]
        assert loop_seeds, 'a group writes and reads its slots at its own iteration'
        assert {seed.index for seed in loop_seeds} == {GROUP_COUNTER}
        assert not list(tutil.type_search(Para.StreamGrab, group))
    assert not list(tutil.type_search(
        Para.LoopGrab, layer_stack.decoder_reuse_block)), \
        'the Reuse layers under the decoder Full layer read what no iteration wrote'


def sized_model() -> tuple[gc.NumericConfig, cat.Morphism]:
    '''The configuration of every size the released `config.json` states, and the model
    it binds.'''
    config = gc.NumericConfig.template(whole_model.v41_flash)
    config.assign_values(m=5120, h=64, c=512, q=1280, o=1024, g=8, j=8, w=128,
                         a=2, u=8, i=32, d=128, e=384, f=2304, n=4,
                         k=6, s=512, p=2048, C=16384)
    return config, config(whole_model.v41_flash)


def check_the_whole_model() -> None:
    '''*The whole model*: the model reads token identifiers and returns probabilities,
    every slot it drops it grabs, the initial collapse vector is a covariant read of one
    index with no operands, and the configuration binds every size the release states.'''
    collapse_vector = whole_model.initial_collapse()
    assert axes(whole_model.expand_into_streams()) == ([['x', 'm']], [['x', 'n', 'm']])
    assert axes(collapse_vector) == ([], [['x', 'n']])
    assert isinstance(collapse_vector.operator, aops.CovariantView)
    assert tuple(collapse_vector.operator.reindexing.dom()) == ()
    assert tuple(collapse_vector.operator.reindexing.cod()) == (n,)
    assert tuple(collapse_vector.backup_degree) == (x,)
    try:
        aops.CovariantView.template(collapse_vector.operator.reindexing, degree=(x,))
        template_states_the_injection = True
    except mark_sparse_codomains.NotAMixedRadixSplit:
        template_states_the_injection = False
    assert not template_states_the_injection, \
        'the template asks for a bijection, and an injection out of the empty product ' \
        'is none'
    tokens, = whole_model.v41_flash.dom()
    probabilities, = whole_model.v41_flash.cod()
    assert isinstance(tokens.datatype, cat.Natural)
    assert isinstance(probabilities.datatype, cat.Reals)
    assert slots_dropped(whole_model.v41_flash) == slots_grabbed(
        whole_model.v41_flash) == RUNTIME_SLOTS
    config, _ = sized_model()
    names = {term.uid._name.to_bodies() for term in config.terms if term.uid._name}
    assert {'b', 'P'} <= names
    assert not {'x', 'B'} & names, \
        'the query axis and the decoder entries are sized |a| |b|'


def check_the_compound_axis_labels() -> None:
    '''*The whole model*: the assignments are written onto the symbolic model, so a label
    holding several symbols keeps every letter and gives each its own size as an
    exponent.'''
    config, _ = sized_model()
    assigned = config.assigned_integers_by_name()
    assert assigned['k'] == 6 and assigned['e'] == 384
    assert 'b' not in assigned, 'the encoder entries are left symbolic'
    routing = write_axis_exponents.write_assigned_size_exponents(
        mixture_of_experts.MIXTURE, assigned)
    router = list(tutil.type_search(dst.SparseAxis, routing))
    assert {axis.activity.to_latex() for axis in router} == {'|k|^{6}'}
    assert {axis.uid._name.to_latex() for axis in router} == {'k/e^{384}'}, \
        'the parent size rides the sparse axis name, and tsncd draws it on the letter ' \
        'after the slash as `e^{384}`'
    pool = write_axis_exponents.write_assigned_size_exponents(
        candidate_pool.CANDIDATE_POOL, assigned)
    assert '|a|^{2} |b|' in {natural.max_value.to_latex()
                             for natural in tutil.type_search(cat.Natural, pool)}, \
        'a size assigned in part carries the exponent on the factor that was assigned'
    reindex = write_axis_exponents.write_assigned_size_exponents(
        attention_modes.reindex_attention(), assigned)
    guarded = {axis.uid._name.to_latex()
               for axis in tutil.type_search(AffineGuards.AffineSparseAxis, reindex)
               if axis.uid._name}
    assert {'w|x^{128}', 's|x^{512}'} <= guarded, \
        'the exponent rides the whole guarded name, and tsncd moves it onto the letter ' \
        'before the bar'


def check_the_sparse_expansion() -> None:
    '''*Expanding the selections onto the tape*: the router's selection expands onto one
    index slot, the gate box drops it and the three expert projections outside the box
    grab it, every selection by position comes back unchanged, and the whole model
    expands with the router's slot beside the six of the runtime.'''
    routing = mixture_of_experts.MIXTURE
    routing_expanded = para_sparse_expansion.expand_sparse_onto_tape(routing)
    assert not list(tutil.type_search(dst.SparseAxis, routing_expanded))
    assert routing_expanded.dom() == routing.dom()
    assert routing_expanded.cod() == routing.cod()
    index_slot, = slots_dropped(routing_expanded)
    assert slots_dropped(routing_expanded) == slots_grabbed(routing_expanded)
    expanded_gate = node_with_box_named(mixture_of_experts.GATE_BOX, routing_expanded)
    assert axes(expanded_gate) == ([['m']], [['k']]), \
        'the expanded gate hands out the gates on a genuinely six-wide axis'
    assert {drop.tape for drop in tutil.type_search(
        Para.Drop, expanded_gate.operator.block)} == {index_slot}, \
        'the selection inside the gate box drops the index'
    indexed_weights = sorted(
        root.operator.name.to_bodies()
        for root in tutil.type_search(cat.Broadcasted, routing_expanded)
        if isinstance(root.operator, ops.Linear)
        and any(isinstance(array.datatype, cat.Natural) for array in root.dom()))
    assert indexed_weights == ['W^{D}', 'W^{G}', 'W^{U}'], \
        'the three expert projections read the index the gate box dropped'
    for selecting_by_position in (
            attention_modes.encoder_full_attention(),
            attention_modes.decoder_full_attention(),
            attention_modes.reindex_attention(),
            attention_modes.decoder_reuse_attention(),
            candidate_pool.CANDIDATE_POOL,
            single_pass_mhc.mhc_sublayer(layer_stack.FULLE)):
        assert not list(tutil.type_search(dst.SparseAxis, selecting_by_position))
        assert (para_sparse_expansion.expand_sparse_onto_tape(selecting_by_position)
                is selecting_by_position)
    model_expanded = para_sparse_expansion.expand_sparse_onto_tape(
        whole_model.v41_flash)
    assert not list(tutil.type_search(dst.SparseAxis, model_expanded))
    assert model_expanded.dom() == whole_model.v41_flash.dom()
    assert model_expanded.cod() == whole_model.v41_flash.cod()
    assert RUNTIME_SLOTS < slots_grabbed(model_expanded) == slots_dropped(
        model_expanded)
    assert len(slots_dropped(model_expanded) - RUNTIME_SLOTS) == 1


def check_the_code_references() -> None:
    '''Every block of the model carries a description in sentences and the pinned
    links of its mechanism: each block has at least one reference, and every
    reference is a link under the released commit with a line. No reference names a file of this package,
    as the reviewer ruled on 2026-09-16.'''
    blocks = list(tutil.type_search(cat.Block, whole_model.v41_flash))
    tags: dict[int, cat.Block] = {b.block_tag.uid._id: b for b in blocks}
    if len(tags) < 20:
        raise AssertionError(f'{len(tags)} distinct blocks')
    for block in tags.values():
        aesthetics = block.block_tag.aesthetics
        if aesthetics is None or not aesthetics.references:
            raise AssertionError(f'{aesthetics and aesthetics.title} carries no reference')
        if not aesthetics.description or not aesthetics.description.endswith('.'):
            raise AssertionError(f'{aesthetics.title} carries no description in sentences')
        for reference in aesthetics.references:
            if reference.url is None or not reference.url.startswith(
                    reference_links.BASE_URL):
                raise AssertionError(f'{reference.label} is not pinned: {reference.url}')
            if reference.line is None:
                raise AssertionError(f'{reference.label} names no line')
    pinned = sum(1 for b in tags.values() for r in b.block_tag.aesthetics.references
                 if r.url is not None)
    if pinned < 40:
        raise AssertionError(f'{pinned} pinned links')


def check_the_auxiliary_information() -> None:
    '''The auxiliary information sent beside the sized model holds a legend row for
    every named axis with the assigned size, the information of every block with its
    references, and an expansion for every softmax, L1 norm and RMSNorm, keyed by
    the number tsncd's importer gives the node. Each block record carries the
    block's description, which the information box shows under the title.'''
    config = gc.NumericConfig.template(whole_model.v41_flash)
    config.assign_values(m=5120, h=64, c=512, q=1280, o=1024, g=8, j=8, w=128, a=2,
                         u=8, i=32, d=128, e=384, f=2304, n=4, k=6, s=512,
                         p=2048, C=16384)
    assigned = config.assigned_integers_by_name()
    morphism = send_morphism.to_morphism(whole_model.v41_flash, recycle=True)
    auxiliary = auxiliary_information.auxiliary_information(
        morphism, assigned_sizes=assigned, code_link_base=reference_links.BASE_URL)
    legend = {row['text']: row for row in auxiliary['legend']}
    distinct = {tuple(sorted((k, v) for k, v in row.items() if k != 'uids'))
                for row in auxiliary['legend']}
    if len(distinct) != len(auxiliary['legend']) or len(auxiliary['legend']) > 40:
        raise AssertionError(
            f'{len(auxiliary["legend"])} rows for {len(distinct)} distinct labels')
    for name, size in (('m', 5120), ('h', 64), ('c', 512), ('e', 384), ('w|x', 128),
                       ('s|x', 512), ('p|x', 2048)):
        if legend[name]['size'] != size:
            raise AssertionError(f'{name} listed at {legend[name]["size"]}')
    if legend['x']['size'] is not None:
        raise AssertionError('the token axis is symbolic')
    if legend['m']['codeName'] != 'hidden_width' or legend['x']['codeName'] != 'tokens':
        raise AssertionError('the declared axes carry their code forms')
    tags = {str(b.block_tag.uid._id) for b in tutil.type_search(cat.Block, morphism)}
    if set(auxiliary['blocks']) != tags:
        raise AssertionError('one block record per tag')
    if any(not record['references'] for record in auxiliary['blocks'].values()):
        raise AssertionError('every block record carries its references')
    if any(not record['description'] for record in auxiliary['blocks'].values()):
        raise AssertionError('every block record carries its description')
    numbered = broadcast_occurrences.number_broadcasts_in_import_order(morphism)
    expandable = {k for k, b in enumerate(numbered)
                  if isinstance(b.operator, (ops.SoftMax, ops.L1Norm, ops.L2Norm,
                                             ops.Normalize))}
    if {int(k) for k in auxiliary['expansions']} != expandable:
        raise AssertionError('one expansion per softmax, L1 norm, L2 norm and RMSNorm')
    for key, expansion in auxiliary['expansions'].items():
        if type(numbered[int(key)].operator).__name__ != expansion['operator']:
            raise AssertionError(f'expansion {key} names {expansion["operator"]}')
    icons = {reference['icon'] for record in auxiliary['blocks'].values()
             for reference in record['references']}
    if icons != {'huggingface'}:
        raise AssertionError(f'every block reference links into Hugging Face: {icons}')


def check_the_operator_explanations() -> None:
    '''The figure the model is drawn as explains every operator. Each top-k selection, read at selected positions, embedding,
    concatenation, covariant view, merged position and elementwise map that hides part
    of its formula is wrapped in a block drawn as the operator alone, which carries a
    formula and a description, and the model handed over is not edited. The four
    top-k selections carry four formulas, one per form and operand, and the one the
    Reindex layer feeds from the tape is wrapped inside its `ParaWrap`. A plain box a
    wrap tapes holds the same slot as a seed inside its block. Each RMSNorm and each
    `Linear` is written out with its parameters on the tape, each read through a box
    named after the weight that the grab is absorbed onto, and no `Linear` that reads
    data left in it, a learned array such as the sink logit as the weight box of
    that array alone, and its box links the released `RMSNorm` and `linear`. The box over a `Linear` says the role of that weight in
    the model first and links the line that declares it, and its formula names a bias
    only where the map has one. Each named reindexing of a view is wrapped in a block
    drawn as the reindexing alone. No reference names a file of this package.'''
    settings = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
        operator_explanations=operator_explanations.OPERATOR_EXPLANATIONS,
        operator_references=operator_explanations.OPERATOR_REFERENCES,
        operator_roles=operator_explanations.OPERATOR_ROLES,
        reindexing_explanations=operator_explanations.REINDEXING_EXPLANATIONS)
    sent, _, auxiliary = notebook_diagrams.package_auxiliary(
        notebook_diagrams.present_each_side(whole_model.v41_flash, settings), settings)
    for node in tutil.type_search(cat.Broadcasted, sent):
        for reindexing in node.reindexings:
            for stride in tutil.type_search(cat.StrideMorphism, reindexing):
                if stride.name is not None and not any(
                        block.body is stride
                        for block in tutil.type_search(cat.Block, reindexing)):
                    raise AssertionError(
                        f'the reindexing {stride.name.to_latex()} is not explained')
    in_place = [b for b in tutil.type_search(cat.Block, sent)
                if b.block_tag.aesthetics is not None
                and b.block_tag.aesthetics.drawing is cat.BlockDrawing.BODY_IN_PLACE
                and isinstance(b.body, cat.Broadcasted)]
    wrapped = {id(b.body) for b in in_place}
    for node in tutil.type_search(cat.Broadcasted, sent):
        explained = explain_operators.explanation_for(
            node, operator_explanations.OPERATOR_EXPLANATIONS)
        if explained is not None and id(node) not in wrapped:
            raise AssertionError(f'{type(node.operator).__name__} is not explained')
    top_k_formulas = {b.block_tag.aesthetics.formula for b in in_place
                      if isinstance(b.body.operator, dst.TopK)}
    if len(top_k_formulas) != 4:
        raise AssertionError(
            f'the four top-k selections carry {len(top_k_formulas)} formulas')
    taped = [wrap.body.operator for wrap in tutil.type_search(para_wrap.ParaWrap, sent)
             if isinstance(wrap.body, cat.Broadcasted)
             and isinstance(wrap.body.operator, ops.BlockOperator)]
    if not any(isinstance(operator.block.body, cat.Broadcasted)
               and dst.selects_over_positions(operator.block.body)
               for operator in taped):
        raise AssertionError(
            'the top-k selection the Reindex layer feeds from the tape is explained')
    hidden = {b.body.operator.name.to_bodies() for b in in_place
              if isinstance(b.body.operator, ops.Arithmetic)}
    if hidden != set(operator_explanations.ARITHMETIC_ROLES):
        raise AssertionError(
            f'the elementwise maps that open a box are {sorted(hidden)}')
    for operator in taped:
        aesthetics = operator.block.block_tag.aesthetics
        if aesthetics is not None and aesthetics.drawing is cat.BlockDrawing.BODY_IN_PLACE:
            continue
        if not isinstance(operator, para_block_operator.ParaBlockOperator):
            raise AssertionError(
                f'{aesthetics.title} is taped at a port and holds no seed for it')
    for block in in_place:
        record = auxiliary['blocks'][str(block.block_tag.uid._id)]
        if not record['formula'] or not record['description'] or not record['title']:
            raise AssertionError(f'{record["title"]} carries no formula or description')
    if any(b.block_tag.aesthetics is not None and b.block_tag.aesthetics.drawing is not None
           for b in tutil.type_search(cat.Block, whole_model.v41_flash)):
        raise AssertionError('the model is not edited by the display pass')
    numbered = broadcast_occurrences.number_broadcasts_in_import_order(sent)
    for number, node in enumerate(numbered):
        if not isinstance(node.operator, (ops.Normalize, ops.Linear)):
            continue
        expansion = auxiliary['expansions'].get(str(number))
        if expansion is None:
            raise AssertionError(f'{node.operator.name.to_latex()} is not written out')
        written = expand_with_parameters.write_out_under(
            settings.expanded_parameters, settings.tape)(node)
        weight_arrays = [b for b in tutil.type_search(cat.Broadcasted, written)
                         if isinstance(b.operator, ops.Linear) and not b.dom()]
        holds_a_parameter = not (
            isinstance(node.operator, ops.Normalize)
            and not node.operator.gain and not node.operator.bias)
        if ((holds_a_parameter and not weight_arrays)
                or tape_presentation.holds_tape_operations(written)):
            raise AssertionError(
                f'{expansion["operator"]} {number} reads a parameter that no weight '
                'array names, or reads the tape')
        if weight_arrays and not holds_a_parameter:
            raise AssertionError(
                f'{expansion["operator"]} {number} declares neither a gain nor a bias '
                'and is written out with a weight array')
        if any(isinstance(b.operator, ops.Linear) and b not in weight_arrays
               for b in tutil.type_search(cat.Broadcasted, written)):
            raise AssertionError(
                f'{node.operator.name.to_latex()} is written out with a Linear that '
                'reads data left in it')
        if {r['icon'] for r in expansion['references']} != {'huggingface'}:
            raise AssertionError(f'{expansion["operator"]} {number} links no released code')
        if not isinstance(node.operator, ops.Linear):
            continue
        role = operator_explanations.OPERATOR_ROLES.get(node.operator.name.to_bodies())
        if role is None or not expansion['description'].startswith(role.role):
            raise AssertionError(
                f'{node.operator.name.to_latex()} has no role, or its box does not say '
                'the role first')
        if (' + b_' in expansion['formula']) != node.operator.bias:
            raise AssertionError(
                f'{node.operator.name.to_latex()} has bias {node.operator.bias} and the '
                f'formula {expansion["formula"]}')
    paths = [reference['path']
             for record in (*auxiliary['blocks'].values(), *auxiliary['expansions'].values())
             for reference in record['references']]
    if any(path is not None and not path.startswith(('inference/', 'config.json'))
           for path in paths):
        raise AssertionError('a reference names a file outside the released repository')


def check_the_size_placement() -> None:
    '''A figure drawn under `AxisSizes.SUBSCRIPT` lowers every assigned size into
    the subscript of the name it belongs to, so the residual width reads `m_{5120}`
    where `EXPONENT` reads `m^{5120}`, a size symbol keeps its bars outside the
    value, `|k|_{6}`, and a listing writes the value after a colon, `m:5120`. The
    bodies of every name are unchanged, so the legend keys its rows as before.'''
    config = gc.NumericConfig.template(whole_model.v41_flash)
    config.assign_values(m=5120, k=6, w=128)
    assigned = config.assigned_integers_by_name()
    lowered = axis_sizes.present(mixture_of_experts.MIXTURE, axis_sizes.AxisSizes.SUBSCRIPT,
                                 assigned)
    raised = axis_sizes.present(mixture_of_experts.MIXTURE, axis_sizes.AxisSizes.EXPONENT,
                                assigned)
    def names_of(kind: type, term: object) -> dict[str, fd.DynamicName]:
        return {name.to_bodies(): name for target in tutil.type_search(kind, term)
                if (name := target.uid._name) is not None}

    axes_lowered, axes_raised = names_of(cat.Axis, lowered), names_of(cat.Axis, raised)
    sizes_lowered = names_of(nm.FreeNumeric, lowered)
    sizes_raised = names_of(nm.FreeNumeric, raised)
    if axes_lowered['m'].to_latex() != 'm_{5120}' or axes_raised['m'].to_latex() != 'm^{5120}':
        raise AssertionError(
            f'{axes_lowered["m"].to_latex()} and {axes_raised["m"].to_latex()}')
    if sizes_lowered['k'].to_latex() != '|k|_{6}' or sizes_raised['k'].to_latex() != '|k|^{6}':
        raise AssertionError(
            f'{sizes_lowered["k"].to_latex()} and {sizes_raised["k"].to_latex()}')
    if axes_lowered['m'].to_text() != 'm:5120' or axes_raised['m'].to_text() != 'm^5120':
        raise AssertionError(
            f'{axes_lowered["m"].to_text()} and {axes_raised["m"].to_text()}')
    if axes_lowered.keys() != axes_raised.keys() or sizes_lowered.keys() != sizes_raised.keys():
        raise AssertionError('the bodies of the names differ between the placements')
    window = names_of(cat.Axis, axis_sizes.present(
        layer_stack.SWA, axis_sizes.AxisSizes.SUBSCRIPT, assigned))
    if window['w|x'].to_latex() != 'w|x_{128}':
        raise AssertionError(f'the guarded window reads {window["w|x"].to_latex()}')



def check_the_wording_file_loads_into_its_dataclass() -> None:
    '''Every field of `BlockTitlesAndDescriptions` is an entry of the wording file
    beside it, every entry is a field, each is a string once resolved, and loading the
    file again gives the wording the modules were built with.'''
    fields = {field.name for field in
              dataclasses.fields(block_titles_and_descriptions.BlockTitlesAndDescriptions)}
    file = wording_json.read_wording_file(block_titles_and_descriptions.WORDING_FILE)
    assert set(file.wordings) == fields, (
        sorted(set(file.wordings) ^ fields))
    loaded = block_titles_and_descriptions.BlockTitlesAndDescriptions.load()
    assert loaded == block_titles_and_descriptions.TEXT
    assert all(isinstance(getattr(loaded, name), str) for name in fields)
    assert loaded.CORE_TITLE == '\\text{Attention Core}'
    assert loaded.SCALED_CORE_DESCRIPTION.startswith(loaded.CORE_DESCRIPTION)
    assert loaded.SCALED_CORE_DESCRIPTION.endswith(loaded.SCORE_SCALE_SENTENCE)

CHECKS: tuple[Callable[[], None], ...] = (
    check_the_declared_axes,
    check_the_custom_operations,
    check_the_compressors,
    check_the_sliding_window,
    check_the_concatenated_core,
    check_the_boxed_core,
    check_the_grouped_output_projection,
    check_the_lightning_indexer,
    check_the_entry_gather,
    check_the_reachable_counts,
    check_the_candidate_blocks,
    check_the_boxed_candidate_pool,
    check_the_attention_modes,
    check_the_concatenated_slots_follow_their_parts,
    check_the_expert_gate,
    check_the_mixture,
    check_single_pass_mhc,
    check_the_layer_plan,
    check_the_whole_model,
    check_the_compound_axis_labels,
    check_the_sparse_expansion,
    check_the_code_references,
    check_the_operator_explanations,
    check_the_auxiliary_information,
    check_the_size_placement,
    check_the_wording_file_loads_into_its_dataclass,
)


def run_checks() -> int:
    '''Every check, one line each, and the number that failed.'''
    failures = 0
    for check in CHECKS:
        try:
            check()
        except Exception as reason:
            failures += 1
            print(f'FAILED  {check.__name__}: {type(reason).__name__}: {reason}')
        else:
            print(f'ok      {check.__name__}')
    return failures


if __name__ == '__main__':
    failed = run_checks()
    print(f'{len(CHECKS) - failed} of {len(CHECKS)} DeepSeek-V4.1-Flash checks passed')
    sys.exit(1 if failed else 0)
