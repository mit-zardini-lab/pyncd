'''The seven attention modes of DeepSeek-V4.1-Flash with the rotary embedding, the
rounding of the caches, the pinned pool and the score scales written in.

Written by Claude Fable 5.1, reasoning effort 80.

Each mode has the routing of the mode of the same name in
`notebooks.sota.DeepSeekV41Flash.attention_modes`, reads the hidden state alone and
returns the hidden state alone. The sub-expressions are those the released layer
computes, in the released order:

    the query            `W^{Qa}`, its norm and `W^{Qb}`, then the rotation at the
                         token's position, once per head
    the window latents   `W^{KV}` and its norm, the rotation at the token's position,
                         the FP8 round trip, then the window view. The rotation stands
                         ahead of the view, because a slot of the view reads the token
                         `i_x - i_w`, whose position the view no longer carries
    the entries          the compressor, then two copies. The indexer keys are made from
                         the unrotated copy. The other copy is rotated at the first
                         token of each entry's group and rounded through FP4, and that
                         copy reaches the gather and the slot
    the indexer keys     `k^{I}` and its norm, the rotation, the FP4 round trip, then
                         the indexer and, in the decoder, the slot
    the pool             `pinned_candidate_pool.PINNED_CANDIDATE_POOL`, in the decoder's
                         Full mode
    the core             `scaled_attention_core`, with the scale `|c|^{-1/2}`
    the output           the clockwise rotation at the query's position, once per head,
                         between the core and the grouped output projection

Every slot holds reals. The entries and the keys a slot holds are rotated and carry the
rounding, as the released caches do, so a Reuse mode and the Reindex mode use what they
grab as it comes.

The modes and the slots they read and write:

    window_attention()              : no slots
    encoder_full_attention()        : drops ckv_b[l] and sel_b[l]
    decoder_full_attention()        : drops ckv_B, ik_B, pool and sel_B
    reindex_attention()             : grabs ckv_B, ik_B and pool, drops sel_B[l]
    encoder_reuse_attention()       : grabs ckv_b[l] and sel_b[l]
    decoder_reuse_attention()       : grabs ckv_B and sel_B
    reindex_group_reuse_attention() : grabs ckv_B and sel_B[l]

A mode that touches a slot of a repeated group takes the entry of that slot as an
argument, a `Para.NamedEntry`, whose default carries the counter of the group. A group
written alone, outside its repeated block, passes a `Para.LoopSlot` with a constant
member in its place.

Layers 0 and 1 rotate with the plain table, `RotaryKind.PLAIN`, and hold no entries, so
`window_attention` is the one mode of that kind. Every other mode rotates with
`RotaryKind.YARN`. `SWA`, `FULLE`, `FULLD`, `REINDEX`, `REUSEE`, `REUSED` and
`REUSED_IN_GROUP` are the seven modes boxed the way
`notebooks.sota.DeepSeekV41Flash.layer_stack` boxes them, each with the domain and the
codomain `(state,)`.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import para.data_structure.Para as Para

import notebooks.sota.DeepSeekV41Flash.scaled_attention_core as scaled_attention_core
from notebooks.sota.DeepSeekV41Flash.attention_core import (
    project_window_latents, window_view)
from notebooks.sota.DeepSeekV41Flash.attention_modes import (
    COMPRESSOR_BOX, DECODER_SCORES, attention_query, move_queries_ahead_of_heads, query_path)
from notebooks.sota.DeepSeekV41Flash.candidate_pool import (
    POOL, REINDEX_SELECT, restrict_to_candidates)
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, over, para_boxed, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    CKVd, CKVe, GROUP_COUNTER, KId, KIe, Q, QR, R, SLOT_CKVd, SLOT_CKVe, SLOT_KId,
    SLOT_POOL, SLOT_SELd, SLOT_SELe, WKV, B, a, b, c, h, state, x)
from notebooks.sota.DeepSeekV41Flash.grouped_output import output_projection
from notebooks.sota.DeepSeekV41Flash.lightning_indexer import (
    DECODER_SELECT, ENCODER_SELECT, SELd, SELe, gather, s_d, s_e)
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.token_compressors import (
    pool_tokens_into_entries, project_tokens_into_entries)
from notebooks.sota.DeepSeekV41Flash.pinned_candidate_pool import (
    PINNED_CANDIDATE_POOL)
from notebooks.sota.DeepSeekV41Flash.quantised_caches import (
    ENTRY_ROUND_TRIP, WINDOW_ROUND_TRIP)
from notebooks.sota.DeepSeekV41Flash.rotary_embedding import (
    ROTATE_ATTENTION_OUTPUT_BACK, ROTATE_DECODER_ENTRIES, ROTATE_ENCODER_ENTRIES,
    ROTATE_TOKEN_LATENTS, RotaryKind)
from notebooks.sota.DeepSeekV41Flash.rotated_indexer import (
    INDEXER_d, INDEXER_e, decoder_index_keys, encoder_index_keys)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

FORWARD_REFERENCE = model_lines(765, 789)
WINDOW_REFERENCE = model_lines(700, 720)
ENTRY_CACHE_REFERENCE = model_lines(739, 763)


def rotated_query_path(kind: RotaryKind) -> cat.BroadcastedCategory:
    '''The rotated query of every head beside the unrotated query low rank, which the
    indexer reads.'''
    return query_path() @ (over((h,), ROTATE_TOKEN_LATENTS[kind]) * hold(QR))


def rotated_attention_query(kind: RotaryKind) -> cat.BroadcastedCategory:
    '''The rotated query alone, for a mode that runs no indexer.'''
    return attention_query() @ over((h,), ROTATE_TOKEN_LATENTS[kind])


def cached_window_kv(kind: RotaryKind) -> cat.BroadcastedCategory:
    '''This layer's own sliding-window latents as the released window cache holds
    them: rotated at each token's position and rounded through FP8, and then read per
    query by the window view.'''
    return (project_window_latents()
            @ ROTATE_TOKEN_LATENTS[kind]
            @ over((x,), WINDOW_ROUND_TRIP)
            @ window_view())


def cached_entries[A: cat.Axis](
    entry_axis: A,
    rotate_entries: cat.Broadcasted,
) -> cat.BroadcastedCategory:
    '''The compressed entries as the released entry cache holds them: rotated by
    `rotate_entries` and rounded through FP4.'''
    return rotate_entries @ over((entry_axis,), ENTRY_ROUND_TRIP)


def project_attended_window_and_entries(kind: RotaryKind) -> cat.BroadcastedCategory:
    '''The scaled attention core, the rotation of its result back, the query's heads
    gathered, and the output projection.'''
    return (scaled_attention_core.attend_over_window_and_entries()
            @ over((h,), ROTATE_ATTENTION_OUTPUT_BACK[kind])
            @ move_queries_ahead_of_heads()
            @ output_projection())


def project_attended_window(kind: RotaryKind) -> cat.BroadcastedCategory:
    '''The scaled attention core over the window alone, the rotation of its result
    back, the query's heads gathered, and the output projection, for a layer with no
    compressed entries.'''
    return (scaled_attention_core.attend_over_window()
            @ over((h,), ROTATE_ATTENTION_OUTPUT_BACK[kind])
            @ move_queries_ahead_of_heads()
            @ output_projection())


def encoder_full_attention(
    dropped_entries: Para.NamedEntry = Para.LoopSlot(SLOT_CKVe, GROUP_COUNTER),
    dropped_selection: Para.NamedEntry = Para.LoopSlot(SLOT_SELe, GROUP_COUNTER),
) -> cat.Block:
    '''The Full mode of the encoder, which computes the compressed entries, the indexer
    keys and the selection, and drops the rotated and rounded entries and the selection
    onto `dropped_entries` and `dropped_selection`.'''
    return cat.Block.template(
        route((0, 0, 0, 0), (state,))
        @ (rotated_query_path(RotaryKind.YARN) * cached_window_kv(RotaryKind.YARN)
           * boxed(pool_tokens_into_entries(b), COMPRESSOR_BOX) * hold(state))
        @ route((0, 1, 2, 3, 3, 4), (Q, QR, WKV, CKVe, state))
        @ (hold(Q) * hold(QR) * hold(WKV) * cached_entries(b, ROTATE_ENCODER_ENTRIES)
           * encoder_index_keys() * hold(state))
        @ route((0, 2, 3, 1, 4, 5), (Q, QR, WKV, CKVe, KIe, state))
        @ (hold(Q) * hold(WKV) * hold(CKVe) * INDEXER_e)
        @ (hold(Q) * hold(WKV) * hold(CKVe) * over((x,), ENCODER_SELECT))
        @ route((0, 1, 3, 2, 2, 3), (Q, WKV, CKVe, SELe))
        @ (hold(Q) * hold(WKV) * gather(b, (a,), s_e) * hold(CKVe) * hold(SELe))
        @ (project_attended_window_and_entries(RotaryKind.YARN) * hold(CKVe)
           * hold(SELe))
        @ (hold(state) * Para.drop_of(dropped_entries, CKVe)
           * Para.drop_of(dropped_selection, SELe)),
        title=text.FULL_TITLE, fill_color='#C5BEDF',
        description=text.SCALED_ENCODER_FULL_ATTENTION_DESCRIPTION,
        references=(model_lines(410, 424), WINDOW_REFERENCE, ENTRY_CACHE_REFERENCE,
                    FORWARD_REFERENCE, model_lines(1166, 1176)))


def decoder_full_attention() -> cat.Block:
    '''The Full mode of the decoder, which computes the encoder-decoder projection, the
    shared indexer keys, the candidate pool with its newest block pinned and a fresh
    Top-512, and drops all four onto their slots.'''
    return cat.Block.template(
        route((0, 0, 0, 0), (state,))
        @ (rotated_query_path(RotaryKind.YARN) * cached_window_kv(RotaryKind.YARN)
           * boxed(project_tokens_into_entries(B), COMPRESSOR_BOX) * hold(state))
        @ route((0, 1, 2, 3, 3, 4), (Q, QR, WKV, CKVd, state))
        @ (hold(Q) * hold(QR) * hold(WKV) * cached_entries(B, ROTATE_DECODER_ENTRIES)
           * decoder_index_keys() * hold(state))
        @ route((0, 2, 3, 4, 1, 4, 5), (Q, QR, WKV, CKVd, KId, state))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * hold(KId) * INDEXER_d)
        @ route((0, 1, 2, 3, 4, 4), (Q, WKV, CKVd, KId, DECODER_SCORES))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * hold(KId) * PINNED_CANDIDATE_POOL
           * over((x,), DECODER_SELECT))
        @ route((0, 1, 5, 2, 2, 3, 4, 5), (Q, WKV, CKVd, KId, POOL, SELd))
        @ (hold(Q) * hold(WKV) * gather(B, (), s_d) * hold(CKVd) * hold(KId)
           * hold(POOL) * hold(SELd))
        @ (project_attended_window_and_entries(RotaryKind.YARN) * hold(CKVd)
           * hold(KId) * hold(POOL) * hold(SELd))
        @ (hold(state) * Para.Drop(tape=SLOT_CKVd, size=CKVd)
           * Para.Drop(tape=SLOT_KId, size=KId)
           * Para.Drop(tape=SLOT_POOL, size=POOL)
           * Para.Drop(tape=SLOT_SELd, size=SELd)),
        title=text.FULL_TITLE, fill_color='#C5BEDF',
        description=text.SCALED_DECODER_FULL_ATTENTION_DESCRIPTION,
        references=(model_lines(410, 424), WINDOW_REFERENCE, ENTRY_CACHE_REFERENCE,
                    model_lines(569, 575), FORWARD_REFERENCE, model_lines(1166, 1176)))


def reindex_attention(
    dropped_selection: Para.NamedEntry = Para.LoopSlot(SLOT_SELd, GROUP_COUNTER),
) -> cat.Block:
    '''The Reindex mode, in which the entries, the keys and the candidate pool are
    grabbed from the slots the decoder's Full layer wrote, the scores are read at the
    pool's distances, a fresh Top-512 is taken over them, and the selection is the one
    array the mode drops, onto `dropped_selection`.'''
    return cat.Block.template(
        (hold(state) * Para.Grab(tape=SLOT_CKVd, size=CKVd)
         * Para.Grab(tape=SLOT_KId, size=KId)
         * Para.Grab(tape=SLOT_POOL, size=POOL))
        @ route((0, 0, 0, 1, 2, 3), (state, CKVd, KId, POOL))
        @ (rotated_query_path(RotaryKind.YARN) * cached_window_kv(RotaryKind.YARN)
           * hold(state) * hold(CKVd) * hold(KId) * hold(POOL))
        @ route((0, 2, 4, 1, 5, 3, 6), (Q, QR, WKV, state, CKVd, KId, POOL))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * INDEXER_d * hold(POOL))
        @ route((0, 1, 2, 3, 4, 4), (Q, WKV, CKVd, DECODER_SCORES, POOL))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * restrict_to_candidates() * hold(POOL))
        @ (hold(Q) * hold(WKV) * hold(CKVd) * over((x,), REINDEX_SELECT))
        @ route((0, 1, 3, 2, 3), (Q, WKV, CKVd, SELd))
        @ (hold(Q) * hold(WKV) * gather(B, (), s_d) * hold(SELd))
        @ (project_attended_window_and_entries(RotaryKind.YARN) * hold(SELd))
        @ (hold(state) * Para.drop_of(dropped_selection, SELd)),
        title=text.REINDEX_TITLE, fill_color='#D9E7F5',
        description=text.SCALED_REINDEX_ATTENTION_DESCRIPTION,
        references=(model_lines(563, 565), model_lines(569, 575), model_lines(578, 580),
                    WINDOW_REFERENCE, FORWARD_REFERENCE))


def reuse_attention[A: cat.Axis](
    entry_axis: A,
    offset_axes: tuple[A, ...],
    selected: A,
    grabbed_entries: Para.NamedEntry,
    grabbed_selection: Para.NamedEntry,
) -> cat.Block:
    '''The Reuse mode, in which the rotated and rounded entries and the selection are
    grabbed from the slots a preceding layer wrote, and the layer computes its query,
    its window and its output projection. A `Para.LoopSlot` in place of a slot grabs
    the member one iteration of the enclosing group wrote.'''
    CKV = cat.Array(R, (entry_axis, c))
    SEL = cat.Array(cat.Natural(entry_axis.local_size()), (x, selected))
    return cat.Block.template(
        (hold(state) * Para.grab_of(grabbed_entries, CKV)
         * Para.grab_of(grabbed_selection, SEL))
        @ route((0, 0, 2, 1), (state, CKV, SEL))
        @ (rotated_attention_query(RotaryKind.YARN) * cached_window_kv(RotaryKind.YARN)
           * gather(entry_axis, offset_axes, selected))
        @ project_attended_window_and_entries(RotaryKind.YARN),
        title=text.REUSE_TITLE, fill_color='#B8D8CE',
        description=text.SCALED_REUSE_ATTENTION_DESCRIPTION,
        references=(model_lines(1166, 1176), model_lines(1180), WINDOW_REFERENCE,
                    model_lines(762, 763), FORWARD_REFERENCE))


def encoder_reuse_attention(
    grabbed_entries: Para.NamedEntry = Para.LoopSlot(SLOT_CKVe, GROUP_COUNTER),
    grabbed_selection: Para.NamedEntry = Para.LoopSlot(SLOT_SELe, GROUP_COUNTER),
) -> cat.Block:
    '''The Reuse mode of an encoder group, reading the entries and the selection the
    Full layer of the same iteration of the group wrote.'''
    return reuse_attention(b, (a,), s_e, grabbed_entries, grabbed_selection)


def decoder_reuse_attention() -> cat.Block:
    '''The Reuse mode of the three layers under the decoder's Full layer, reading the
    entries and the selection that layer wrote. Neither slot carries an iteration,
    because the layer that wrote them stands outside every repeated group.'''
    return reuse_attention(B, (), s_d, SLOT_CKVd, SLOT_SELd)


def reindex_group_reuse_attention(
    grabbed_selection: Para.NamedEntry = Para.LoopSlot(SLOT_SELd, GROUP_COUNTER),
) -> cat.Block:
    '''The Reuse mode of a Reindex group, reading the entries the decoder's Full layer
    wrote and the selection the Reindex layer of the same iteration of the group
    wrote.'''
    return reuse_attention(B, (), s_d, SLOT_CKVd, grabbed_selection)


def window_attention() -> cat.Block:
    '''The mode of layers 0 and 1, which has the query, the window and the sink of
    every other mode, no compressed branch, and the plain rotary table.'''
    return cat.Block.template(
        route((0, 0), (state,))
        @ (rotated_attention_query(RotaryKind.PLAIN)
           * cached_window_kv(RotaryKind.PLAIN))
        @ project_attended_window(RotaryKind.PLAIN),
        title=text.WINDOW_TITLE, fill_color='#DFF0D8',
        description=text.SCALED_WINDOW_ATTENTION_DESCRIPTION,
        references=(model_lines(410, 424), WINDOW_REFERENCE, model_lines(680, 687),
                    FORWARD_REFERENCE))


SWA = boxed(window_attention(), 'SWA')
FULLE = para_boxed(encoder_full_attention(), 'Full')
FULLD = para_boxed(decoder_full_attention(), 'Full')
REINDEX = para_boxed(reindex_attention(), 'Rex')
REUSEE = para_boxed(encoder_reuse_attention(), 'Reu')
REUSED = para_boxed(decoder_reuse_attention(), 'Reu')
REUSED_IN_GROUP = para_boxed(reindex_group_reuse_attention(), 'Reu')
