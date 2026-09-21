'''The lightning indexer of DeepSeek-V4.1-Flash, and the gather its selection drives.

Written by Claude Opus 5 (1M context), reasoning effort high.

The indexer scores every compressed entry a query has reached, with 32 heads of width
128, and a Top-512 keeps the best of them. Causality is written on the keys alone. The
`back` view reads the shared keys `r` entries back from each entry, so a slot past the
first entry is a negative index and reads the universal unit, and the `pos` merge writes
each entry's slots onto every query whose newest reachable entry that entry is, turning
`[b, r|b, d]` into `[x, r|x, d]`. The query low rank and the hidden state are read
as they arrive, and the scoring is one box computed once per query.

`obsidian/02-categories/Padding and Masks as Sparse Axes.md` states why causality is
written as the sign of an index.
'''
from __future__ import annotations

from dataclasses import dataclass

import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst

from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    QR, R, B, a, b, c, d, i, m, nsel, q, state, x)
from notebooks.sota.DeepSeekV41Flash.reference_links import kernel_lines, model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

INDEXER_COLOUR = '#D9E7F5'
INDEXER_BOX = 'Idx'
SCORE_BOX = 'Sco'
GATHER_BOX = 'Gth'


def index_keys[A: cat.Axis](entry_axis: A) -> cat.BroadcastedCategory:
    '''One shared indexer key per compressed entry, from the compressed latent.'''
    return ((entry_axis >> ops.Linear.template((c,), (d,), 'k^{I}'))
            @ over((entry_axis,), ops.Normalize.template((d,))))


def entries_back_axis[A: cat.Axis](entry_axis: A) -> cat.RawAxis:
    '''The axis counting entries back from a query's newest reachable entry.'''
    return fd.DynamicName('r').capture(cat.RawAxis(_size=entry_axis.local_size()))


def rectify() -> cat.Broadcasted:
    '''The rectifier, written as the input times the indicator of its sign, so that a
    derived backward pass carries the indicator rather than a primed name.'''
    return ops.Arithmetic.template(nm.FreeInput() * nm.IsPositive())


def query_of_group_and_offset[A: cat.Axis](
    entry_axis: A,
    offset_axes: tuple[A, ...],
    name: str,
) -> sc.StrideMorphism:
    '''The row writing group `i_b` and offset `i_a` to the query
    `ratio * i_b + i_a + (ratio - 1)`, where `ratio` is the product of the offset sizes
    and the offsets form a mixed-radix number.

    A query has reached a compressed entry once it has passed the entry's last token, so
    the queries whose newest reachable entry is `i_b` are those from
    `ratio * i_b + ratio - 1` to `ratio * i_b + 2 * ratio - 2`, and group `i_b` is that
    run of `ratio` queries. The first `ratio - 1` queries reach no entry and are in no
    group, and the newest `ratio - 1` offsets of the last group are written past the end
    of the query axis, where nothing receives them.
    '''
    radices = []
    ratio = nm.Integer(1)
    for axis in reversed(offset_axes):
        radices.insert(0, ratio)
        ratio = nm.Multiplication.template(ratio, axis.local_size())
    return sc.StrideMorphism(
        _dom=(entry_axis, *offset_axes),
        _cod_stride_shift=((x, (ratio, *radices), ratio - nm.Integer(1)),),
        name=fd.DynamicName(name))


def read_back_from_each_entry[A: cat.Axis](
    entry_axis: A,
    back_axis: A,
    rest: tuple[A, ...],
    base: cat.Datatype = R,
) -> cat.Broadcasted:
    '''An array on the entries read at entry `i_b - i_r`, `r` entries back from each
    entry, times the identity on `rest`.

    A slot counting back past the first entry is a negative index and reads the unit, so
    `mark_sparse_domains.guarded_view` marks the slot axis `r|b`, live where
    `i_b - i_r >= 0`, with its empty end last.
    '''
    return mark_sparse_domains.guarded_view(
        base=base,
        reindexing=(sc.StrideMorphism(
            _dom=(entry_axis, back_axis),
            _cod_stride_shift=((entry_axis, (nm.Integer(1), nm.Integer(-1)),
                                nm.Integer(0)),),
            name=fd.DynamicName('back')),
            cat.ProdObject(rest).identity()),
        name='back')


def write_to_every_query_of_the_group[A: cat.Axis](
    entry_axis: A,
    offset_axes: tuple[A, ...],
    degree: tuple[A, ...],
) -> cat.Broadcasted:
    '''The `pos` merge writing each entry's `degree` to every query whose newest
    reachable entry that entry is.

    The input carries the entries and not the offsets, so the merge broadcasts it over
    the offsets: the `ratio` queries of a group all read the same entry at the same
    distance. A slot axis of `degree` guided by the entries leaves guided by the
    queries, so `r|b` becomes `r|x`, live where `i_x - ratio * i_r - (ratio - 1) >= 0`.
    '''
    return aops.CovariantView.broadcast_over_absent_axes_and_merge(
        query_of_group_and_offset(entry_axis, offset_axes, 'pos'),
        input_axes=(entry_axis,),
        degree=tuple(degree))


def read_keys_per_query[A: cat.Axis](
    entry_axis: A,
    offset_axes: tuple[A, ...],
) -> cat.BroadcastedCategory:
    '''The shared indexer keys as each query's own array of reachable keys: `[b, d]`
    read back from every entry and merged onto the queries as `[x, r|x, d]`.'''
    read_back = read_back_from_each_entry(
        entry_axis, entries_back_axis(entry_axis), (d,))
    return read_back @ write_to_every_query_of_the_group(
        entry_axis, offset_axes, (read_back.cod()[0].shape()[1], d))


def score_every_query[A: cat.Axis](reach: A) -> cat.Block:
    '''Every query's combined score against each entry it has reached, from the query
    low rank, the keys already read per query and the hidden state.

    Each of the 32 indexer heads scores the query against the entry and the score is
    rectified. A per-token per-head weight, the reference's `weights_proj` read off the
    5120-wide hidden state, says how much each head's score counts, so the combine is a
    contraction between two wires.
    '''
    keys = cat.Array(R, (x, reach, d))
    head_weights = cat.Array(R, (x, i))
    return cat.Block.template(
        (over((x,), ops.Linear.template((q,), (i, d), 'q^{I}')) * hold(keys)
         * over((x,), ops.Linear.template((m,), (i,), 'w^{I}')))
        @ ((ops.Einops.template('x i d, x r d -> x i r') @ rectify())
           * hold(head_weights))
        @ ops.Einops.template('x i r, x i -> x r'),
        title=text.SCORE_TITLE, fill_color=INDEXER_COLOUR,
        description=text.SCORE_EVERY_QUERY_DESCRIPTION,
        references=(model_lines(515), model_lines(555, 557)))


@dataclass(frozen=True)
class RelativeIndexer:
    '''The indexer over one entry axis.

    `keys_per_query` reads the shared keys back from every entry and merges them onto
    the queries, `reach` is the slot axis its output carries, `scoring` is the box that
    scores one query against those slots with the confirmation that expanding the box
    over the queries gives the written-out scoring back, and `block` composes the two.
    '''
    keys_per_query: cat.BroadcastedCategory
    reach: AffineGuards.AffineSparseAxis
    scoring: discovering_broadcasts.DiscoveredBroadcast
    block: cat.Block


def relative_indexer[A: cat.Axis](entry_axis: A,
                                  offset_axes: tuple[A, ...]) -> RelativeIndexer:
    '''The indexer reading its keys relative to each query's newest reachable entry and
    scoring one query at a time.'''
    keys_per_query = read_keys_per_query(entry_axis, offset_axes)
    reach = keys_per_query.cod()[0].shape()[1]
    scoring = discovering_broadcasts.discover_broadcast_over_axes(
        score_every_query(reach), (x,), SCORE_BOX)
    return RelativeIndexer(
        keys_per_query=keys_per_query,
        reach=reach,
        scoring=scoring,
        block=cat.Block.template(
            (hold(QR) * keys_per_query * hold(state)) @ scoring.candidate,
            title=text.INDEXER_TITLE, fill_color=INDEXER_COLOUR,
            description=text.RELATIVE_INDEXER_DESCRIPTION,
            references=(model_lines(563, 565), model_lines(567), model_lines(577, 580))))


ENCODER_INDEXER = relative_indexer(b, (a,))
DECODER_INDEXER = relative_indexer(B, ())
INDEXER_e = boxed(ENCODER_INDEXER.block, INDEXER_BOX)
INDEXER_d = boxed(DECODER_INDEXER.block, INDEXER_BOX)
reach_e = ENCODER_INDEXER.reach
reach_d = DECODER_INDEXER.reach


def select_entries[A: cat.Axis](reachable_axis: A) -> cat.Broadcasted:
    '''The Top-512 over the entries a query reaches, in the `ONLY_SELECTION` form. The
    indexer's scores decide which entries are read and are not read after the selection,
    so it hands out the distances of the selected entries alone, on the slot axis `s|x`
    that is live for as many slots as the query reaches.'''
    return dst.TopK.template(k=nsel, axis=reachable_axis,
                             form=dst.SelectionForm.ONLY_SELECTION)


ENCODER_SELECT = select_entries(reach_e)
DECODER_SELECT = select_entries(reach_d)
s_e, = ENCODER_SELECT.cod()[0].shape()
s_d, = DECODER_SELECT.cod()[0].shape()
SELe = cat.Array(cat.Natural(b.local_size()), (x, s_e))
SELd = cat.Array(cat.Natural(B.local_size()), (x, s_d))


def select_each_slots_entry[A: cat.Axis](selected: A, reach: A,
                                         entries: nm.Numeric) -> cat.Broadcasted:
    '''The latent of the entry at each slot's distance, broadcast over the queries, the
    slots and the channels: `[x, s], [x, r|x, c] -> [x, s, c]`, with the slot axis in
    the degree.

    A gather reads one entry per slot, so it is elementwise in the slot. The targets of
    `dst.IndexSelect` are the rank-0 `Natural` index and the entries a query has
    reached, and the slot axis is broadcast over, so a diagram carries the slot wire
    through the gather.
    '''
    T = cat.WeaveMode.TILED
    degree = (x, selected, c)
    return cat.Broadcasted(
        operator=dst.IndexSelect(),
        input_weaves=(cat.Weave(cat.Natural(entries), (T, T)),
                      cat.Weave(R, (T, reach, T))),
        output_weaves=(cat.Weave(R, (T, T, T)),),
        reindexings=(cat.Rearrangement((0, 1), degree),
                     cat.Rearrangement((0, 2), degree)))


def gather[A: cat.Axis](entry_axis: A, offset_axes: tuple[A, ...],
                        selected: A) -> cat.Broadcasted:
    '''The compressed entries read at the distances the selection holds.

    The entries go through the chain the indexer reads its keys through, so each query
    holds the latent of every entry it has reached, counted back from its newest
    reachable entry. A `dst.IndexSelect` broadcast over the slots reads the entry at
    each slot's distance. An empty slot reads the unit.
    '''
    read_back = read_back_from_each_entry(
        entry_axis, entries_back_axis(entry_axis), (c,))
    entries_per_query = read_back @ write_to_every_query_of_the_group(
        entry_axis, offset_axes, (read_back.cod()[0].shape()[1], c))
    selection = cat.Array(cat.Natural(entry_axis.local_size()), (x, selected))
    return boxed(cat.Block.template(
        (hold(selection) * entries_per_query)
        @ select_each_slots_entry(selected, entries_per_query.cod()[0].shape()[1],
                                  entry_axis.local_size()),
        title=text.GATHER_TITLE, fill_color=INDEXER_COLOUR,
        description=text.GATHER_DESCRIPTION,
        references=(model_lines(578, 580), kernel_lines(362, 364))), GATHER_BOX)
