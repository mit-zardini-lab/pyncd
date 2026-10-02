# Claude Opus 5.5 (1M context), effort 40.
'''The axes of one Mamba layer, the arrays that pass between its steps, and the counter
of the loop over the tokens.

The reference names its sizes `d_model`, `d_inner = expand * d_model`, `d_state`,
`dt_rank` and `d_conv`, per `reference_links.SIZES_OF_THE_LAYER`. Here they are the
axes `m`, `d`, `n`, `r` and `w`, and the tokens are `x`. Every size is a free symbol,
bound by name where the layer is evaluated.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm

R = cat.Reals()

x = cat.RawAxis.named('x')
m = cat.RawAxis.named('m')
d = cat.RawAxis.named('d')
n = cat.RawAxis.named('n')
r = cat.RawAxis.named('r')
w = cat.RawAxis.named('w')

TOKEN_COUNTER_NAME = 'i_x'
TOKEN_COUNTER = nm.FreeNumeric.named(TOKEN_COUNTER_NAME)

HIDDEN = cat.Array(R, (x, m))
CHANNELS = cat.Array(R, (x, d))
STATE_MAP = cat.Array(R, (x, n))
STEP_SIZES = cat.Array(R, (x, d))

SMALL_SIZES: dict[str, int] = {'m': 4, 'd': 8, 'n': 3, 'r': 1, 'w': 4}
'''The sizes the layer is evaluated at. They follow the defaults of the
reference's constructor for `d_model = 4`: `expand = 2`, `d_conv = 4` and
`dt_rank = ceil(4 / 16)`, with `d_state` lowered from 16 to 3.'''
