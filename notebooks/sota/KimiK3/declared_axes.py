# Claude Opus 5.5 (1M context), effort 40.
'''The axes, the arrays and the counters of Kimi K3.

Every structural axis is declared once here and passed to the constructions that need
it, because an axis is identified by its uid rather than by its name. No size appears
in an expression. Each axis carries a free symbol, and `whole_model.RELEASED_SIZES`
binds them by the body of their names. A size fixed by two others is written as their
sum, so the configuration binds the two and the third follows:

    a = n + p       the width of one query head and one key head of the latent
                    attention, 128 + 64
    r = x           the distances a query counts back over, one per token

The axes stand for the widths named in the configuration of the reference. For the
latent attention, `h` is `num_attention_heads`, `q` is `q_lora_rank`, `c` is
`kv_lora_rank`, `n` is `qk_nope_head_dim`, `p` is `qk_rope_head_dim` and `u` is
`v_head_dim`. For the delta attention, `j` is `linear_attn_config.num_heads`, `d` and
`z` are both `linear_attn_config.head_dim`, one for the key channels of a head and one
for its value channels, and `w` is `short_conv_kernel_size`. For the feed-forward maps,
`e` is `num_experts`, `l` is `routed_expert_hidden_size`, `f` is
`moe_intermediate_size`, `t` is `moe_intermediate_size` times `num_shared_experts` and
`g` is `intermediate_size`.

The key channels and the value channels of a delta-attention head have the same size in
the checkpoint and are two axes here, because the state of a head is a matrix with one
row per key channel and one column per value channel, and a matrix over one axis read
twice would be a diagonal.

`b` indexes the entries of the attention residuals. Its first position holds the
embedding, position `1 + i` holds the sum of the outputs of block `i` of twelve
layers, and the configuration sets `attn_res_block_size` to 12. The 93 layers fill
eight blocks, so the axis has nine positions. `BLOCK_COUNTER` counts the six blocks
written as one repeated block, and `TOKEN_COUNTER` counts the tokens of the scan of
the delta attention.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd

R = cat.Reals()

x = cat.RawAxis.named('x', code_form='tokens')
m = cat.RawAxis.named('m', code_form='hidden_width')

h = cat.RawAxis.named('h', code_form='attention_heads')
q = cat.RawAxis.named('q', code_form='query_rank')
c = cat.RawAxis.named('c', code_form='latent_width')
n = cat.RawAxis.named('n', code_form='latent_key_width')
p = cat.RawAxis.named('p', code_form='shared_key_width')
a = fd.DynamicName('a', code_form='query_key_width').capture(
    cat.RawAxis(_size=n.local_size() + p.local_size()))
u = cat.RawAxis.named('u', code_form='value_width')
r = fd.DynamicName('r', code_form='distances').capture(
    cat.RawAxis(_size=x.local_size()))

j = cat.RawAxis.named('j', code_form='delta_heads')
d = cat.RawAxis.named('d', code_form='delta_key_width')
z = cat.RawAxis.named('z', code_form='delta_value_width')
w = cat.RawAxis.named('w', code_form='convolution_taps')

e = cat.RawAxis.named('e', code_form='experts')
l = cat.RawAxis.named('l', code_form='expert_latent_width')
f = cat.RawAxis.named('f', code_form='expert_width')
t = cat.RawAxis.named('t', code_form='shared_expert_width')
g = cat.RawAxis.named('g', code_form='dense_width')

b = cat.RawAxis.named('b', code_form='residual_entries')


def selection_count(letter: str, code_form: str) -> nm.FreeNumeric:
    '''How many entries a selection keeps, named as `cat.Axis.named` names a size: the
    letter between absolute bars, with `code_form`.'''
    return nm.FreeNumeric.named(fd.DynamicName(
        letter, settings=fd.DynamicNameSettings(absolute=True), code_form=code_form))


selected_experts = selection_count('k', 'selected_experts_size')

vocab = fd.DynamicName(
    'v', settings=fd.DynamicNameSettings(overline=True), code_form='vocabulary')

STATE = cat.Array(R, (x, m))
TOKEN_STATE = cat.Array(R, (m,))
RESIDUAL_ENTRIES = cat.Array(R, (x, b, m))
TOKEN_RESIDUAL_ENTRIES = cat.Array(R, (b, m))

QUERY_LOW_RANK = cat.Array(R, (x, q))
LATENT = cat.Array(R, (x, c))
QUERIES = cat.Array(R, (x, h, a))
KEYS = cat.Array(R, (x, h, a))
VALUES = cat.Array(R, (x, h, u))
LATENT_KEYS = cat.Array(R, (x, h, n))
SHARED_KEYS = cat.Array(R, (x, h, p))
HEAD_OUTPUTS = cat.Array(R, (x, h, u))

DELTA_KEY_CHANNELS = cat.Array(R, (x, j, d))
DELTA_VALUE_CHANNELS = cat.Array(R, (x, j, z))
DELTA_GATE_RANK = cat.Array(R, (x, d))
DELTA_HEAD_SCALARS = cat.Array(R, (x, j))
DELTA_STATE = cat.Array(R, (j, d, z))

BLOCK_COUNTER_NAME = 'o'
BLOCK_COUNTER = nm.FreeNumeric.named(BLOCK_COUNTER_NAME)
TOKEN_COUNTER_NAME = 'i_x'
TOKEN_COUNTER = nm.FreeNumeric.named(TOKEN_COUNTER_NAME)
