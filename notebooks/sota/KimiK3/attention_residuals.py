# Claude Opus 5.5 (1M context), effort 40.
'''The attention residuals of Kimi K3, which replace the residual stream with a softmax
over the outputs of earlier blocks of layers.

A layer of a transformer usually reads the hidden state, which is the embedding plus the
output of every sublayer so far. `_forward_attn_residual` of the reference keeps the
same sums in pieces. The configuration sets `attn_res_block_size` to 12, which cuts the
93 layers into seven blocks of twelve and one of nine. The reference keeps the embedding
and the sum of the outputs of every finished block in the list `block_residual`, and the
sum of the outputs of the current block in `prefix_sum`. Before each sublayer,
`_apply_attn_res` computes the input of the sublayer as a weighted mean of those
entries:

    k[i_b, i_m] = v[i_b, i_m] / sqrt(mean over m of v[i_b]^2 + epsilon)
    s[i_b]      = sum over i_m of gamma[i_m] w[i_m] k[i_b, i_m]
    y[i_m]      = sum over i_b of softmax_b(s)[i_b] v[i_b, i_m]

The entries `v` are the list followed by the current sum. The weights of the mean are a
softmax of a score read off each entry through a learned vector, the gain `gamma` of an
RMS normalisation times the weight `w` of a projection onto one number, so a sublayer
chooses by content how much of each earlier block it reads. Every sublayer holds its
own pair of vectors, `self_attention_res_norm` with `self_attention_res_proj` before the
attention and `mlp_res_norm` with `mlp_res_proj` before the feed-forward map, and the
model holds a last pair before its output head.

The expression carries the list and the current sum as one array over the axis `b` of
nine positions. Position 0 holds the embedding and position `1 + i` holds the sum of the
outputs of block `i`. A sublayer of block `i` adds its output at position `1 + i`, so
the current sum is always the last live position, and the sum of a finished block stays
where it was written. The reference appends the current sum to the list when the next
block begins, at position `1 + i` of the list, and puts it last in the concatenation
before every mix, so the two hold the same entries in the same order. A position after
the current block holds the universal unit, which the softmax and the sum over `b` pass
over, so every mix reads the entries the reference concatenates.

The first layer of the model reads the embedding directly, because the reference applies
no mix while the list is empty. A mix over the embedding alone would return the embedding
with a weight of one.

    read_the_embedding()          R[x, b, m] -> R[x, m]
    mix_of(name)                  R[x, b, m] -> R[x, m], one box computed per token
    add_at_block(position)        R[x, b, m], R[x, m] -> R[x, b, m]
    start_the_entries()           R[x, m] -> R[x, b, m]
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, route
from notebooks.sota.KimiK3.declared_axes import (
    R, RESIDUAL_ENTRIES, STATE, TOKEN_RESIDUAL_ENTRIES, b, m, x)
from notebooks.sota.KimiK3.reference_links import checkpoint_config_lines, modeling_lines
from notebooks.sota.KimiK3.released_constants import NORM_EPSILON
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

MIX_COLOUR = '#FCE0E1'
MIX_BOX = 'Res'
EMBEDDING_ENTRY = nm.Integer(0)

MIX_REFERENCES = (modeling_lines(1075, 1088), modeling_lines(906, 917),
                  checkpoint_config_lines(26))
LIST_REFERENCES = (modeling_lines(973, 1046), modeling_lines(1188, 1192))


def row_at_entry(position: nm.Numeric) -> sc.StrideMorphism:
    '''The row with no domain axis that selects entry `position` of `b`.'''
    return sc.StrideMorphism(
        _dom=(), _cod_stride_shift=((b, (), position),),
        name=fd.DynamicName(position.to_latex()))


def read_the_embedding() -> cat.Broadcasted:
    '''`R[x, b, m] -> R[x, m]`: the entry at position 0, which holds the embedding.'''
    return ops.View.template(
        reindexing=(cat.ProdObject((x,)).identity(), row_at_entry(EMBEDDING_ENTRY),
                    cat.ProdObject((m,)).identity()),
        name=EMBEDDING_ENTRY.to_latex())


def write_at_entry(position: nm.Numeric) -> cat.Broadcasted:
    '''`R[x, m] -> R[x, b, m]`: an array written at entry `position` of `b`, holding the
    universal unit at every other entry. It is the covariant reading of the row that
    selects the entry.'''
    T = cat.WeaveMode.TILED
    return cat.Broadcasted(
        operator=aops.CovariantView(reindexing=row_at_entry(position),
                                    name=fd.DynamicName(position.to_latex())),
        input_weaves=(cat.Weave(R, (T, T)),),
        output_weaves=(cat.Weave(R, (T, b, T)),),
        reindexings=(cat.ProdObject((x, m)).identity(),))


def add_at_entry(position: nm.Numeric) -> cat.BroadcastedCategory:
    '''`R[x, b, m], R[x, m] -> R[x, b, m]`: the output of a sublayer added to the entry
    at `position`.'''
    shape = (x, b, m)
    return ((hold(RESIDUAL_ENTRIES) * write_at_entry(position))
            @ einops_simplification.einsum((shape, shape), shape, R, ops.AdditionOp()))


def start_the_entries() -> cat.Broadcasted:
    '''`R[x, m] -> R[x, b, m]`: the embedding written at entry 0, with every later entry
    empty.'''
    return write_at_entry(EMBEDDING_ENTRY)


def mix_body(weight_name: str) -> cat.Block:
    '''One token's entries to their weighted mean: every entry normalised, scored by the
    projection `weight_name`, the scores passed through a softmax over the entries, and
    the entries summed under the softmax.'''
    scores = cat.Array(R, (b,))
    return cat.Block.template(
        route((0, 0), (TOKEN_RESIDUAL_ENTRIES,))
        @ (((b >> ops.Normalize.template((m,), epsilon=NORM_EPSILON))
            @ (b >> ops.Linear.template((m,), (), weight_name))
            @ ops.SoftMax.template())
           * hold(TOKEN_RESIDUAL_ENTRIES))
        @ ops.Einops.template('b, b m -> m'),
        title=text.MIX_TITLE, fill_color=MIX_COLOUR,
        description=text.MIX_DESCRIPTION,
        formula=('y[i_{m}] = \\sum_{i_{b} \\in b} \\mathrm{softmax}_{b}\\Big('
                 '\\sum_{j_{m} \\in m} w[j_{m}]\\, \\mathrm{RMSNorm}(v[i_{b}])[j_{m}]'
                 '\\Big)[i_{b}]\\, v[i_{b}, i_{m}]'),
        references=MIX_REFERENCES)


def mix_of(weight_name: str) -> cat.Broadcasted:
    '''The mix as one box computed once per token.'''
    return discovering_broadcasts.broadcast_block_over_axes(
        mix_body(weight_name), (x,), ((0,),), MIX_BOX)


ATTENTION_INPUT_WEIGHT = 'w^{Ra}'
FEED_FORWARD_INPUT_WEIGHT = 'w^{Rf}'
OUTPUT_INPUT_WEIGHT = 'w^{Ro}'
