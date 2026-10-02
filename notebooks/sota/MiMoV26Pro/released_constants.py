# Claude Opus 5.5 (1M context), effort 40.
'''The named constants of MiMo-V2.6-Pro, with the value each has in the reference.

`term_utilities.generate_config.NumericConfig.assign_values` matches a symbol by the
body of its name, so every constant has a body no axis uses. A constant written with a
subscript holds the subscript inside its body, because `fd.DynamicName.from_str` cuts a
name at its first underscore and would leave two constants with one body.

The reference builds two tables of turns. `MiMoV2RotaryEmbedding` reads `rope_theta` for
the full attention layers and `swa_rope_theta` for the sliding window layers, so the two
bases are two constants. Every RMS normalisation of the text model adds
`layernorm_epsilon`. The router adds a number written in its code to the sum of the
eight kept gates, and the attention multiplies every value by `attention_value_scale`.
The router multiplies the normalised gates by `routed_scaling_factor`, which the
configuration leaves as `null` and the router replaces by 1.0, so the expression holds
no factor for it.
'''
from __future__ import annotations

from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd

from notebooks.sota.MiMoV26Pro.reference_links import (
    checkpoint_config_lines, modeling_lines)

ROTARY_BASE = nm.FreeNumeric.named('\\beta')
WINDOW_ROTARY_BASE = nm.FreeNumeric.named(fd.DynamicName('\\beta_{\\mathrm{w}}'))
NORM_EPSILON = nm.FreeNumeric.named('\\varepsilon')
ROUTER_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{r}}'))
VALUE_SCALE = nm.FreeNumeric.named('\\lambda')


@dataclass(frozen=True)
class ReleasedConstant:
    '''The value a named constant has in the reference, with the line that sets it.'''
    symbol: nm.FreeNumeric
    released_value: str
    meaning: str
    reference: cat.CodeReference


RELEASED_CONSTANTS: tuple[ReleasedConstant, ...] = (
    ReleasedConstant(
        symbol=ROTARY_BASE, released_value='10^{7}',
        meaning='the rotary base of the full attention layers, `rope_theta`, which '
                'sets how slowly the slowest pair of channels turns',
        reference=checkpoint_config_lines(365)),
    ReleasedConstant(
        symbol=WINDOW_ROTARY_BASE, released_value='10^{4}',
        meaning='the rotary base of the sliding window layers, `swa_rope_theta`',
        reference=checkpoint_config_lines(373)),
    ReleasedConstant(
        symbol=NORM_EPSILON, released_value='10^{-5}',
        meaning='the number added under the square root of every RMS normalisation, '
                '`layernorm_epsilon`',
        reference=checkpoint_config_lines(122)),
    ReleasedConstant(
        symbol=ROUTER_EPSILON, released_value='10^{-20}',
        meaning='the number added to the sum of the eight kept gates before each '
                'gate is divided by that sum',
        reference=modeling_lines(181)),
    ReleasedConstant(
        symbol=VALUE_SCALE, released_value='0.612',
        meaning='the factor `attention_value_scale`, which multiplies every value '
                'before the attention reads it',
        reference=checkpoint_config_lines(11)),
)

RELEASED_CONSTANTS_TABLE_HEADER: tuple[str, str] = (
    '| symbol | value in the reference | meaning | line |',
    '|---|---|---|---|')


def released_constant_row(constant: ReleasedConstant) -> str:
    '''One markdown table row for `constant`, with its symbol and value set as
    mathematics and its line as a link.'''
    reference = constant.reference
    return (f'| ${constant.symbol.to_latex()}$ | ${constant.released_value}$ | '
            f'{constant.meaning} | [{reference.label}]({reference.url}) |')


def released_constants_table() -> str:
    '''The markdown table of every named constant of the model, for a notebook cell.'''
    return '\n'.join((*RELEASED_CONSTANTS_TABLE_HEADER,
                      *map(released_constant_row, RELEASED_CONSTANTS)))
