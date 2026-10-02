# Claude Opus 5.5 (1M context), effort 40.
'''The axes and the arrays of MiMo-V2.6-Pro.

Every structural axis is declared once here and passed to the constructions that need
it, because an axis is identified by its uid rather than by its name. No size read from
the configuration appears in an expression. Each axis carries a free symbol, and
`whole_model.RELEASED_SIZES` binds the symbols by the body of their names. A size fixed
by two others is written as their sum or product:

    p = 2 t         the channels of a query head or a key head the rotary embedding
                    turns, 64, as 32 pairs, each pair two adjacent channels
    a = p + n       the width of a query head and of a key head, 64 + 128
    r = x           the distances a query counts back over, one per token

The reference holds the 64 turned channels as two halves of 32, the real parts of the
pairs and then their imaginary parts, which is the layout `rotate_half` reads. The axis
`c` names the half, 0 for the real parts and 1 for the imaginary parts, so the turned
rows of a projection are written over `(c, t)` and channel `i_t + |t| i_c` of the
reference is entry `(i_c, i_t)`. Its size is the number 2 rather than a symbol of the
configuration.

The axes stand for the widths named in the configuration of the checkpoint: `m` is
`hidden_size`, `h` is `num_key_value_heads`, `g` is `num_attention_heads` divided by
`num_key_value_heads`, `a` is `head_dim`, `u` is `v_head_dim`, `t` is half of the
rotary width `int(head_dim * partial_rotary_factor)`, `w` is `sliding_window`, `e` is
`n_routed_experts`, `f` is `moe_intermediate_size` and `d` is `intermediate_size`. The
sliding window layers declare the same widths under `swa_head_dim`,
`swa_v_head_dim`, `swa_num_attention_heads` and `swa_num_key_value_heads`, so one set of
axes serves both kinds of layer.

The reference numbers its 128 query heads so that heads `16 i_h` to `16 i_h + 15` read
key-value head `i_h`, which is the pairing made by `repeat_kv`. The queries are written
over `(h, g)`, head `i_g` of group `i_h`, per the ruling on groupings in
`obsidian/06-practice/Representing Models.md`.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.declared_axes import selection_count

R = cat.Reals()

x = cat.RawAxis.named('x', code_form='tokens')
m = cat.RawAxis.named('m', code_form='hidden_width')
h = cat.RawAxis.named('h', code_form='key_value_heads')
g = cat.RawAxis.named('g', code_form='query_heads_per_key_value_head')
t = cat.RawAxis.named('t', code_form='rotary_pairs')
p = fd.DynamicName('p', code_form='rotated_head_width').capture(
    cat.RawAxis(_size=nm.Integer(2) * t.local_size()))
c = fd.DynamicName('c', code_form='real_and_imaginary_halves').capture(
    cat.RawAxis(_size=nm.Integer(2)))
n = cat.RawAxis.named('n', code_form='unrotated_head_width')
a = fd.DynamicName('a', code_form='query_key_width').capture(
    cat.RawAxis(_size=p.local_size() + n.local_size()))
u = cat.RawAxis.named('u', code_form='value_width')
w = cat.RawAxis.named('w', code_form='window_slots')
r = fd.DynamicName('r', code_form='distances').capture(
    cat.RawAxis(_size=x.local_size()))
e = cat.RawAxis.named('e', code_form='experts')
f = cat.RawAxis.named('f', code_form='expert_width')
d = cat.RawAxis.named('d', code_form='dense_width')

selected_experts = selection_count('k', 'selected_experts_size')

vocab = fd.DynamicName(
    'v', settings=fd.DynamicNameSettings(overline=True), code_form='vocabulary')

STATE = cat.Array(R, (x, m))
TOKEN_STATE = cat.Array(R, (m,))
QUERIES = cat.Array(R, (x, h, g, a))
KEYS = cat.Array(R, (x, h, a))
VALUES = cat.Array(R, (x, h, u))
HEAD_OUTPUTS = cat.Array(R, (x, h, g, u))
EXPONENTIATED_SINKS = cat.Array(R, (h, g))
