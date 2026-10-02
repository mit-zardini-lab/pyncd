# Claude Opus 5.5 (1M context), effort 40.
'''The axes, the arrays and the tape slot of GLM-5.3.

Every structural axis is declared once here and passed to the constructions that need
it, because an axis is identified by its uid rather than by its name. No size appears
in an expression. Each axis carries a free symbol, and `whole_model.RELEASED_SIZES`
binds them by the body of their names. A size fixed by two others is written as their
sum or product, so the configuration binds the two and the third follows:

    p = 2 t         the channels the rotary embedding turns, 64, as 32 pairs
    a = n + p       the width of one query head and one key head, 192 + 64
    d_bar = d - p   the channels of an indexer vector that are not turned, 128 - 64
    r = x           the distances a query counts back over, one per token

The axes stand for the widths named in the configuration of the reference: `n` is
`qk_nope_head_dim`, `t` is half of `qk_rope_head_dim`, `u` is `v_head_dim`, `q` is
`q_lora_rank`, `c` is `kv_lora_rank`, `i` and `d` are `index_n_heads` and
`index_head_dim`, `f` is `moe_intermediate_size` and `g` is `intermediate_size`.

`SLOT_SELECTION` is the tape slot of the top-2048 selection. A layer that runs its own
indexer drops its selection onto the slot, and every layer that shares the selection
grabs it from the slot. `GROUP_COUNTER` is the counter of the repeated IndexShare
group, so the eighteen iterations of the group write eighteen members of the slot and
each iteration reads its own member.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import named_slot

R = cat.Reals()

x = cat.RawAxis.named('x', code_form='tokens')
m = cat.RawAxis.named('m', code_form='hidden_width')
h = cat.RawAxis.named('h', code_form='heads')
q = cat.RawAxis.named('q', code_form='query_rank')
c = cat.RawAxis.named('c', code_form='latent_width')
n = cat.RawAxis.named('n', code_form='unrotated_head_width')
t = cat.RawAxis.named('t', code_form='rotary_pairs')
p = fd.DynamicName('p', code_form='rotated_head_width').capture(
    cat.RawAxis(_size=nm.Integer(2) * t.local_size()))
a = fd.DynamicName('a', code_form='query_key_width').capture(
    cat.RawAxis(_size=n.local_size() + p.local_size()))
u = cat.RawAxis.named('u', code_form='value_width')
i = cat.RawAxis.named('i', code_form='indexer_heads')
d = cat.RawAxis.named('d', code_form='indexer_width')
dbar = fd.DynamicName('\\bar{d}', code_form='unrotated_indexer_width').capture(
    cat.RawAxis(_size=d.local_size() - p.local_size()))
r = fd.DynamicName('r', code_form='distances').capture(
    cat.RawAxis(_size=x.local_size()))
e = cat.RawAxis.named('e', code_form='experts')
f = cat.RawAxis.named('f', code_form='expert_width')
g = cat.RawAxis.named('g', code_form='dense_width')


def selection_count(letter: str, code_form: str) -> nm.FreeNumeric:
    '''How many entries a selection keeps, named as `cat.Axis.named` names a size: the
    letter between absolute bars, with `code_form`. The count is the size of the axis
    the selection hands its results out on, so it is drawn as a size is drawn.'''
    return nm.FreeNumeric.named(fd.DynamicName(
        letter, settings=fd.DynamicNameSettings(absolute=True), code_form=code_form))


selected_tokens = selection_count('s', 'selected_tokens_size')
selected_experts = selection_count('k', 'selected_experts_size')

vocab = fd.DynamicName(
    'v', settings=fd.DynamicNameSettings(overline=True), code_form='vocabulary')

STATE = cat.Array(R, (x, m))
TOKEN_STATE = cat.Array(R, (m,))
QUERY_LOW_RANK = cat.Array(R, (x, q))
LATENT = cat.Array(R, (x, c))
QUERIES = cat.Array(R, (x, h, a))
KEYS = cat.Array(R, (x, h, a))
VALUES = cat.Array(R, (x, h, u))
UNROTATED_KEYS = cat.Array(R, (x, h, n))
ROTATED_KEY_SHARED_BY_THE_HEADS = cat.Array(R, (x, p))

GROUP_COUNTER_NAME = 'l'
GROUP_COUNTER = nm.FreeNumeric.named(GROUP_COUNTER_NAME)

SLOT_SELECTION = named_slot('\\mathrm{sel}')
