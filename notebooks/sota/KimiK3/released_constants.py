# Claude Opus 5.5 (1M context), effort 40.
'''The named constants of Kimi K3, with the value each has in the reference.

`term_utilities.generate_config.NumericConfig.assign_values` matches a symbol by the
body of its name, so every constant has a body no axis uses. A constant written with a
subscript holds the subscript inside its body, because `fd.DynamicName.from_str` cuts a
name at its first underscore and would leave two constants with one body.

The reference adds four different numbers under a square root or to a sum. Every RMS
normalisation built with `config.rms_norm_eps` adds `\\varepsilon`, which covers the
normalisations of the residual, of the attention residuals, of the output of the delta
attention and of the latent of the experts. The RMS normalisations of the query low rank
and of the key-value latent of the latent attention are built with no epsilon argument,
so they add the default of `KimiRMSNorm`. The L2 normalisation of the queries and the
keys of the delta attention adds the default of `l2norm_fwd` in `fla`. The router adds
a number written at the division of the gates.

`routed_scaling_factor` is 1 in the configuration, so the multiplication of the gates by
it changes no value and the expression leaves it out.
'''
from __future__ import annotations

from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd

from notebooks.sota.KimiK3.reference_links import (
    L2_NORM, checkpoint_config_lines, modeling_lines)

NORM_EPSILON = nm.FreeNumeric.named('\\varepsilon')
LATENT_NORM_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{l}}'))
L2_NORM_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{q}}'))
ROUTER_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{r}}'))
GATE_LOWER_BOUND = nm.FreeNumeric.named('\\lambda')
SITU_GATE_BOUND = nm.FreeNumeric.named('\\beta')
SITU_UP_BOUND = nm.FreeNumeric.named(fd.DynamicName('\\beta_{\\mathrm{u}}'))


@dataclass(frozen=True)
class ReleasedConstant:
    '''The value a named constant has in the reference, with the line that sets it.'''
    symbol: nm.FreeNumeric
    released_value: str
    meaning: str
    reference: cat.CodeReference


RELEASED_CONSTANTS: tuple[ReleasedConstant, ...] = (
    ReleasedConstant(
        symbol=NORM_EPSILON, released_value='10^{-5}',
        meaning='the number added under the square root of every RMS normalisation '
                'built with `rms_norm_eps`',
        reference=checkpoint_config_lines(245)),
    ReleasedConstant(
        symbol=LATENT_NORM_EPSILON, released_value='10^{-6}',
        meaning='the number added under the square root of the RMS normalisations of '
                'the query low rank and of the key-value latent',
        reference=modeling_lines(227)),
    ReleasedConstant(
        symbol=L2_NORM_EPSILON, released_value='10^{-6}',
        meaning='the number added under the square root of the L2 normalisation of '
                'the queries and the keys of the delta attention',
        reference=L2_NORM),
    ReleasedConstant(
        symbol=GATE_LOWER_BOUND, released_value='-5',
        meaning='`gate_lower_bound`, the smallest logarithm of the decay of a key '
                'channel in one token',
        reference=checkpoint_config_lines(93)),
    ReleasedConstant(
        symbol=SITU_GATE_BOUND, released_value='4',
        meaning='`activation_situ_beta`, the bound of the hyperbolic tangent on the '
                'gate branch of SiTU',
        reference=checkpoint_config_lines(20)),
    ReleasedConstant(
        symbol=SITU_UP_BOUND, released_value='25',
        meaning='`activation_situ_linear_beta`, the bound of the hyperbolic tangent on '
                'the up branch of SiTU',
        reference=checkpoint_config_lines(21)),
    ReleasedConstant(
        symbol=ROUTER_EPSILON, released_value='10^{-20}',
        meaning='the number added to the sum of the sixteen kept gates before each '
                'gate is divided by that sum',
        reference=modeling_lines(754)),
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


RELEASED_CONSTANT_VALUES: dict[str, float] = {
    '\\varepsilon': 1e-5, '\\varepsilon_{\\mathrm{l}}': 1e-6,
    '\\varepsilon_{\\mathrm{q}}': 1e-6, '\\varepsilon_{\\mathrm{r}}': 1e-20,
    '\\lambda': -5.0, '\\beta': 4.0, '\\beta_{\\mathrm{u}}': 25.0}
'''The value of every named constant, keyed by the body of its name, for an evaluation
of the expression.'''
