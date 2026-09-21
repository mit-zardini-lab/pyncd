'''The axes, the arrays and the tape slots of DeepSeek-V4.1-Flash.

Written by Claude Opus 5, effort high.

Every structural axis is declared once here and passed to the constructions that
need it, because an axis is identified by its uid rather than by its name. No size
appears in an expression: each axis carries a free symbol, and the one
`gc.NumericConfig` in the notebook's last cell binds them all by name.

The six tape slots stand for the six fields of the shared attention runtime the
reference passes between its layers. `GROUP_COUNTER` is the counter of the two repeated
groups of the layer plan, an encoder group and a Reindex group. A slot one iteration of a
group writes and the same iteration reads carries that counter, so the Full layer and the
Reuse layers of one group are linked by the iteration.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import deepseek.data_structure as dst

import notebooks.sota.DeepSeekV41Flash.construction_idioms as construction_idioms

R = cat.Reals()

m = cat.RawAxis.named('m', code_form='hidden_width')
h = cat.RawAxis.named('h', code_form='heads')
c = cat.RawAxis.named('c', code_form='latent_width')
q = cat.RawAxis.named('q', code_form='query_rank')
o = cat.RawAxis.named('o', code_form='output_rank')
g = cat.RawAxis.named('g', code_form='head_groups')
j = cat.RawAxis.named('j', code_form='heads_per_group')
w = cat.RawAxis.named('w', code_form='window')
a = cat.RawAxis.named('a', code_form='compression_ratio')
b = cat.RawAxis.named('b', code_form='entries')
x = fd.DynamicName('x', code_form='tokens').capture(cat.RawAxis(_size=a.local_size() * b.local_size()))
B = fd.DynamicName('B', code_form='decoder_entries').capture(cat.RawAxis(_size=x.local_size()))
u = cat.RawAxis.named('u', code_form='block_size')
P = cat.RawAxis.named('P', code_form='blocks')
C = cat.RawAxis.named('C', code_form='candidates')
i = cat.RawAxis.named('i', code_form='indexer_heads')
d = cat.RawAxis.named('d', code_form='indexer_width')
e = cat.RawAxis.named('e', code_form='experts')
f = cat.RawAxis.named('f', code_form='expert_width')
n = cat.RawAxis.named('n', code_form='streams')
N = cat.RawAxis.named('N', code_form='combined_streams')


def selection_count(letter: str) -> nm.FreeNumeric:
    '''How many entries a selection keeps, named the way `cat.Axis.named` names
    a size: the letter between absolute bars. The count is the size of the dense
    axis the selection hands its values out on, so it is drawn as a size is, and
    `dst.dense_selected_axis` drops the bars to name that axis.'''
    return nm.FreeNumeric.named(
        fd.DynamicName(letter, settings=fd.DynamicNameSettings(absolute=True)))


nsel = selection_count('s')
kexp = selection_count('k')
npool = selection_count('p')
s = dst.dense_selected_axis(nsel)

vocab = fd.DynamicName('v', settings=fd.DynamicNameSettings(overline=True))

state = cat.Array(R, (x, m))
X = cat.Array(R, (x, n, m))
COLLAPSE = cat.Array(R, (x, n))
QR = cat.Array(R, (x, q))
Q = cat.Array(R, (h, x, c))
WKV = cat.Array(R, (x, w, c))
CKVe = cat.Array(R, (b, c))
CKVd = cat.Array(R, (B, c))
KIe = cat.Array(R, (b, d))
KId = cat.Array(R, (B, d))

GROUP_COUNTER_NAME = 'l'
GROUP_COUNTER = nm.FreeNumeric.named(GROUP_COUNTER_NAME)

SLOT_CKVe = construction_idioms.named_slot('\\mathrm{ckv}', 'b')
SLOT_SELe = construction_idioms.named_slot('\\mathrm{sel}', 'b')
SLOT_CKVd = construction_idioms.named_slot('\\mathrm{ckv}', 'B')
SLOT_KId = construction_idioms.named_slot('\\mathrm{ik}', 'B')
SLOT_POOL = construction_idioms.named_slot('\\mathrm{pool}')
SLOT_SELd = construction_idioms.named_slot('\\mathrm{sel}', 'B')
