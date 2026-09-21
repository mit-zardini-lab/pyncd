'''CSA2, the three compressed sparse attention modes of DeepSeek-V4.1-Flash.

Written by Claude Opus 5, effort high.

Every mode reads a 128-token sliding window and a set of selected compressed entries
under one softmax, and computes its own query, its own window latents and its own output
projection from its own hidden state. The three differ in where the compressed entries,
the indexer keys and the selection come from, and each mode grabs what it reads from a
tape slot and drops what it publishes onto one, so every mode reads and returns the
hidden state alone.

The modes and the slots they read and write, with `[l]` marking a slot carrying the
counter of the repeated group the mode stands in:

    window_attention()              : no slots
    encoder_full_attention()        : drops ckv_b[l] and sel_b[l]
    decoder_full_attention()        : drops ckv_B, ik_B, pool and sel_B
    reindex_attention()             : grabs ckv_B, ik_B and pool, drops sel_B[l]
    encoder_reuse_attention()       : grabs ckv_b[l] and sel_b[l]
    decoder_reuse_attention()       : grabs ckv_B and sel_B
    reindex_group_reuse_attention() : grabs ckv_B and sel_B[l]

A slot one iteration of a group writes and the same iteration reads carries the group's
counter, so the Full layer and the Reuse layers of one encoder group are linked by the
iteration, as a Reindex layer and the Reuse layers below it are. The decoder's Full layer
stands outside every group, so what it writes is read without an iteration, and the
decoder therefore has two Reuse modes: the three layers under the Full layer read what it
wrote, and the three layers of a Reindex group read the selection that group's Reindex
layer wrote.

The attention core is computed once per head and the output projection once per query, so
the head axis leads the arrays the core reads and writes and the query axis leads the
arrays the projection reads and writes. The two transpose views carry the query's heads
into the leading position and back.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops
import para.data_structure.Para as Para

from notebooks.sota.DeepSeekV41Flash.attention_core import (
    attend_over_window, attend_over_window_and_entries, window_kv)
from notebooks.sota.DeepSeekV41Flash.candidate_pool import (
    CANDIDATE_POOL, POOL, REINDEX_SELECT, restrict_to_candidates)
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, over, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    CKVd, CKVe, GROUP_COUNTER, KId, KIe, Q, QR, R, SLOT_CKVd, SLOT_CKVe, SLOT_KId,
    SLOT_POOL, SLOT_SELd, SLOT_SELe, WKV, B, a, b, c, h, m, q, state, x)
from notebooks.sota.DeepSeekV41Flash.grouped_output import output_projection
from notebooks.sota.DeepSeekV41Flash.lightning_indexer import (
    DECODER_SELECT, ENCODER_SELECT, INDEXER_d, INDEXER_e, SELd, SELe, gather,
    index_keys, reach_d, s_d, s_e)
from notebooks.sota.DeepSeekV41Flash.token_compressors import (
    pool_tokens_into_entries, project_tokens_into_entries)
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

COMPRESSOR_BOX = 'Comp'

DECODER_SCORES = cat.Array(R, (x, reach_d))


def move_heads_ahead_of_queries() -> cat.Broadcasted:
    '''The query with its head axis brought to the front, for the core, which is computed
    once per head.'''
    return ops.View.template(reindexing=cat.Rearrangement((1, 0, 2), (h, x, c)),
                             name='tr')


def move_queries_ahead_of_heads() -> cat.Broadcasted:
    '''What the core returned with the query axis brought to the front, for the output
    projection, which is computed once per query.'''
    return ops.View.template(reindexing=cat.Rearrangement((1, 0, 2), (x, h, c)),
                             name='tr')


def query_path() -> cat.BroadcastedCategory:
    '''The low-rank query, copied so that the indexer reads the rank the attention
    reads.'''
    return ((x >> ops.Linear.template((m,), (q,), 'W^{Qa}'))
            @ over((x,), ops.Normalize.template((q,)))
            @ route((0, 0), (QR,))
            @ (((x >> ops.Linear.template((q,), (h, c), 'W^{Qb}'))
                @ move_heads_ahead_of_queries()) * hold(QR)))


def attention_query() -> cat.BroadcastedCategory:
    '''The query alone, for a mode that runs no indexer.'''
    return ((x >> ops.Linear.template((m,), (q,), 'W^{Qa}'))
            @ over((x,), ops.Normalize.template((q,)))
            @ (x >> ops.Linear.template((q,), (h, c), 'W^{Qb}'))
            @ move_heads_ahead_of_queries())


def project_attended_window_and_entries() -> cat.BroadcastedCategory:
    '''The attention core, the query's heads gathered, and the output projection.'''
    return (attend_over_window_and_entries()
            @ move_queries_ahead_of_heads()
            @ output_projection())


def project_attended_window() -> cat.BroadcastedCategory:
    '''The same for a layer with no compressed entries.'''
    return (attend_over_window()
            @ move_queries_ahead_of_heads()
            @ output_projection())


def encoder_full_attention() -> cat.Block:
    '''The encoder's Full mode: it mints the compressed entries, the indexer keys and the
    selection, and drops the entries and the selection onto their slots.'''
    return cat.Block.template(
        route((0, 0, 0, 0), (state,))
        @ (query_path() * window_kv()
           * boxed(pool_tokens_into_entries(b), COMPRESSOR_BOX) * hold(state))
        @ route((0, 1, 2, 3, 3, 4), (Q, QR, WKV, CKVe, state))
        @ (hold(Q) * hold(QR) * hold(WKV) * hold(CKVe) * index_keys(b) * hold(state))
        @ route((0, 2, 3, 1, 4, 5), (Q, QR, WKV, CKVe, KIe, state))
        @ (hold(Q) * hold(WKV) * hold(CKVe) * INDEXER_e)
        @ (hold(Q) * hold(WKV) * hold(CKVe) * over((x,), ENCODER_SELECT))
        @ route((0, 1, 3, 2, 2, 3), (Q, WKV, CKVe, SELe))
        @ (hold(Q) * hold(WKV) * gather(b, (a,), s_e) * hold(CKVe) * hold(SELe))
        @ (project_attended_window_and_entries() * hold(CKVe) * hold(SELe))
        @ (hold(state)
           * Para.LoopDrop(tape=SLOT_CKVe, size=CKVe, index=GROUP_COUNTER)
           * Para.LoopDrop(tape=SLOT_SELe, size=SELe, index=GROUP_COUNTER)),
        title=text.FULL_TITLE, fill_color='#C5BEDF',
        description=text.ENCODER_FULL_ATTENTION_DESCRIPTION,
        references=(model_lines(410, 424), model_lines(700, 720), model_lines(1166, 1176)))


def decoder_full_attention() -> cat.Block:
    '''The decoder's Full mode: the encoder-decoder projection, the shared indexer keys,
    the candidate pool and a fresh Top-512, all four dropped onto their slots.'''
    return cat.Block.template(
        route((0, 0, 0, 0), (state,))
        @ (query_path() * window_kv()
           * boxed(project_tokens_into_entries(B), COMPRESSOR_BOX) * hold(state))
        @ route((0, 1, 2, 3, 3, 4), (Q, QR, WKV, CKVd, state))
        @ (hold(Q) * hold(QR) * hold(WKV) * hold(CKVd) * index_keys(B) * hold(state))
        @ route((0, 2, 3, 4, 1, 4, 5), (Q, QR, WKV, CKVd, KId, state))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * hold(KId) * INDEXER_d)
        @ route((0, 1, 2, 3, 4, 4), (Q, WKV, CKVd, KId, DECODER_SCORES))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * hold(KId) * CANDIDATE_POOL
           * over((x,), DECODER_SELECT))
        @ route((0, 1, 5, 2, 2, 3, 4, 5), (Q, WKV, CKVd, KId, POOL, SELd))
        @ (hold(Q) * hold(WKV) * gather(B, (), s_d) * hold(CKVd) * hold(KId)
           * hold(POOL) * hold(SELd))
        @ (project_attended_window_and_entries() * hold(CKVd) * hold(KId)
           * hold(POOL) * hold(SELd))
        @ (hold(state) * Para.Drop(tape=SLOT_CKVd, size=CKVd)
           * Para.Drop(tape=SLOT_KId, size=KId)
           * Para.Drop(tape=SLOT_POOL, size=POOL)
           * Para.Drop(tape=SLOT_SELd, size=SELd)),
        title=text.FULL_TITLE, fill_color='#C5BEDF',
        description=text.DECODER_FULL_ATTENTION_DESCRIPTION,
        references=(model_lines(410, 424), model_lines(700, 720), model_lines(569, 575),
            model_lines(1166, 1176)))


def reindex_attention() -> cat.Block:
    '''Reindex mode: the entries, the keys and the candidate pool are grabbed from the
    slots the decoder's Full layer wrote, the scores are read at the pool's distances, a
    fresh Top-512 is taken over them, and the selection is the one array the mode
    drops.'''
    return cat.Block.template(
        (hold(state) * Para.Grab(tape=SLOT_CKVd, size=CKVd)
         * Para.Grab(tape=SLOT_KId, size=KId)
         * Para.Grab(tape=SLOT_POOL, size=POOL))
        @ route((0, 0, 0, 1, 2, 3), (state, CKVd, KId, POOL))
        @ (query_path() * window_kv() * hold(state) * hold(CKVd) * hold(KId)
           * hold(POOL))
        @ route((0, 2, 4, 1, 5, 3, 6), (Q, QR, WKV, state, CKVd, KId, POOL))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * INDEXER_d * hold(POOL))
        @ route((0, 1, 2, 3, 4, 4), (Q, WKV, CKVd, DECODER_SCORES, POOL))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * restrict_to_candidates() * hold(POOL))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * over((x,), REINDEX_SELECT))
        @ route((0, 1, 3, 2, 3), (Q, WKV, CKVd, SELd))
        @ (hold(Q) * hold(WKV) * gather(B, (), s_d) * hold(SELd))
        @ (project_attended_window_and_entries() * hold(SELd))
        @ (hold(state)
           * Para.LoopDrop(tape=SLOT_SELd, size=SELd, index=GROUP_COUNTER)),
        title=text.REINDEX_TITLE, fill_color='#D9E7F5',
        description=text.REINDEX_ATTENTION_DESCRIPTION,
        references=(model_lines(563, 565), model_lines(569, 575), model_lines(578, 580)))


def reuse_attention[A: cat.Axis](
    entry_axis: A,
    offset_axes: tuple[A, ...],
    selected: A,
    grabbed_entries: Para.NamedEntry,
    grabbed_selection: Para.NamedEntry,
) -> cat.Block:
    '''Reuse mode: the entries and the selection are grabbed from the slots a preceding
    layer wrote, and the layer computes its query, its window and its output projection.
    A `Para.LoopSlot` in place of a slot grabs the member one iteration of the enclosing
    group wrote.'''
    CKV = cat.Array(R, (entry_axis, c))
    SEL = cat.Array(cat.Natural(entry_axis.local_size()), (x, selected))
    return cat.Block.template(
        (hold(state) * Para.grab_of(grabbed_entries, CKV)
         * Para.grab_of(grabbed_selection, SEL))
        @ route((0, 0, 2, 1), (state, CKV, SEL))
        @ (attention_query() * window_kv() * gather(entry_axis, offset_axes, selected))
        @ project_attended_window_and_entries(),
        title=text.REUSE_TITLE, fill_color='#B8D8CE',
        description=text.REUSE_ATTENTION_DESCRIPTION,
        references=(model_lines(1166, 1176), model_lines(1180)))


def encoder_reuse_attention() -> cat.Block:
    '''The Reuse mode of an encoder group, reading the entries and the selection the Full
    layer of the same iteration of the group wrote.'''
    return reuse_attention(b, (a,), s_e,
                           Para.LoopSlot(SLOT_CKVe, GROUP_COUNTER),
                           Para.LoopSlot(SLOT_SELe, GROUP_COUNTER))


def decoder_reuse_attention() -> cat.Block:
    '''The Reuse mode of the three layers under the decoder's Full layer, reading the
    entries and the selection that layer wrote. Neither slot carries an iteration,
    because the layer that wrote them stands outside every repeated group.'''
    return reuse_attention(B, (), s_d, SLOT_CKVd, SLOT_SELd)


def reindex_group_reuse_attention() -> cat.Block:
    '''The Reuse mode of a Reindex group, reading the entries the decoder's Full layer
    wrote and the selection the Reindex layer of the same iteration of the group
    wrote.'''
    return reuse_attention(B, (), s_d, SLOT_CKVd,
                           Para.LoopSlot(SLOT_SELd, GROUP_COUNTER))


def window_attention() -> cat.Block:
    '''Layers 0 and 1: the same query, window and sink, with no compressed branch.'''
    return cat.Block.template(
        route((0, 0), (state,))
        @ (attention_query() * window_kv())
        @ project_attended_window(),
        title=text.WINDOW_TITLE, fill_color='#DFF0D8',
        description=text.WINDOW_ATTENTION_DESCRIPTION,
        references=(model_lines(410, 424), model_lines(700, 720)))
