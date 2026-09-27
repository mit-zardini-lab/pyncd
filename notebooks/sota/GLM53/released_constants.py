# Claude Opus 5.5 (1M context), effort 40.
'''The named constants of GLM-5.3, with the value each has in the reference.

`term_utilities.generate_config.NumericConfig.assign_values` matches a symbol by the
body of its name, so every constant has a body no axis uses. A constant written with a
subscript holds the subscript inside its body, because `fd.DynamicName.from_str` cuts a
name at its first underscore and would leave two constants with one body.

The reference sets three epsilons. The RMS normalisations of the residual and of the
final hidden state add `rms_norm_eps` from the configuration. The RMS normalisations of
the query low rank and of the key-value latent are built with no epsilon argument, so
they add the default of `GlmMoeDsaRMSNorm`. The layer normalisation of the indexer keys
adds the number written at its declaration.
'''
from __future__ import annotations

from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd

from notebooks.sota.GLM53.reference_links import (
    checkpoint_config_lines, modeling_lines)

ROTARY_BASE = nm.FreeNumeric.named('\\beta')
NORM_EPSILON = nm.FreeNumeric.named('\\varepsilon')
LATENT_NORM_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{l}}'))
KEY_NORM_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{k}}'))
ROUTER_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{r}}'))
ROUTE_SCALE = nm.FreeNumeric.named('\\gamma')


@dataclass(frozen=True)
class ReleasedConstant:
    '''The value a named constant has in the reference, with the line that sets it.'''
    symbol: nm.FreeNumeric
    released_value: str
    meaning: str
    reference: cat.CodeReference


RELEASED_CONSTANTS: tuple[ReleasedConstant, ...] = (
    ReleasedConstant(
        symbol=ROTARY_BASE, released_value='8 \\times 10^{6}',
        meaning='the rotary base, `rope_theta`, which sets how slowly the slowest '
                'pair of channels turns',
        reference=checkpoint_config_lines(211, 214)),
    ReleasedConstant(
        symbol=NORM_EPSILON, released_value='10^{-5}',
        meaning='the number added under the square root of the RMS normalisations of '
                'the residual and of the final hidden state',
        reference=checkpoint_config_lines(209)),
    ReleasedConstant(
        symbol=LATENT_NORM_EPSILON, released_value='10^{-6}',
        meaning='the number added under the square root of the RMS normalisations of '
                'the query low rank and of the key-value latent',
        reference=modeling_lines(49)),
    ReleasedConstant(
        symbol=KEY_NORM_EPSILON, released_value='10^{-6}',
        meaning='the number added to the variance in the layer normalisation of the '
                'indexer keys',
        reference=modeling_lines(191)),
    ReleasedConstant(
        symbol=ROUTER_EPSILON, released_value='10^{-20}',
        meaning='the number added to the sum of the eight kept gates before each '
                'gate is divided by that sum',
        reference=modeling_lines(516)),
    ReleasedConstant(
        symbol=ROUTE_SCALE, released_value='2.5',
        meaning='the factor `routed_scaling_factor`, which multiplies every '
                'normalised gate',
        reference=checkpoint_config_lines(215)),
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
