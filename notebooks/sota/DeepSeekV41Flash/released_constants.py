'''The named constants of DeepSeek-V4.1-Flash that its omitted mechanisms read, with
the value each has in the released configuration.

Written by Claude Fable 5.1, reasoning effort 80, as part of
`notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots`. Moved here by
Claude Fable 5.1, reasoning effort 25, on 2026-09-19, so that the omitted mechanisms and
the integrated model read one set of constants.

`term_utilities.generate_config.NumericConfig.assign_values` matches a symbol by the
body of its name. A constant whose body is the letter of an axis would therefore take
the size assigned to that axis. Every constant here has a body no axis uses. A constant
written with a subscript holds the subscript inside its body, because
`fd.DynamicName.from_str` cuts a name at its first underscore and would leave two
constants with one body. A body that holds its own subscript takes a second script
beside it once a size is written onto the name, which is not LaTeX, and a constant is
never assigned a size, so none of these takes one. A symbol that is assigned one
carries its subscript as a `fd.DynamicName` instead, as
`engram_modules.table_row_count` does. `RELEASED_CONSTANTS` gives the
value each constant has in the released configuration, with the line that sets it, and
the explanations print those values beside the formulas.
'''
from __future__ import annotations

from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines)

MODALITY = cat.Natural.template('\\mu')

ROTARY_BASE = nm.FreeNumeric.named('\\beta')
WINDOW_ROTARY_BASE = nm.FreeNumeric.named(fd.DynamicName('\\beta_{0}'))
YARN_FACTOR = nm.FreeNumeric.named('\\kappa')
YARN_RAMP_START = nm.FreeNumeric.named('\\mathrm{lo}')
YARN_RAMP_END = nm.FreeNumeric.named('\\mathrm{hi}')
SWIGLU_LIMIT = nm.FreeNumeric.named('\\lambda')
ROUTER_TEMPERATURE = nm.FreeNumeric.named('\\tau')
NORM_EPSILON = nm.FreeNumeric.named('\\varepsilon')
MHC_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{hc}}'))
ROUTER_EPSILON = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{r}}'))
ENGRAM_GATE_FLOOR = nm.FreeNumeric.named(fd.DynamicName('\\varepsilon_{\\mathrm{g}}'))


@dataclass(frozen=True)
class ReleasedConstant:
    '''The value a named constant has in the released configuration.'''
    symbol: nm.FreeNumeric
    released_value: str
    meaning: str
    reference: cat.CodeReference


RELEASED_CONSTANTS: tuple[ReleasedConstant, ...] = (
    ReleasedConstant(
        symbol=ROTARY_BASE, released_value='160000',
        meaning='the rotary base of every layer that holds compressed entries',
        reference=inference_config_lines(51)),
    ReleasedConstant(
        symbol=WINDOW_ROTARY_BASE, released_value='10000',
        meaning='the rotary base of the two sliding-window layers',
        reference=inference_config_lines(30)),
    ReleasedConstant(
        symbol=YARN_FACTOR, released_value='16',
        meaning='the factor YaRN divides the slow rotary frequencies by',
        reference=inference_config_lines(31)),
    ReleasedConstant(
        symbol=YARN_RAMP_START, released_value='15',
        meaning='the last channel pair YaRN leaves at its own frequency',
        reference=model_lines(382, 383)),
    ReleasedConstant(
        symbol=YARN_RAMP_END, released_value='25',
        meaning='the first channel pair whose frequency YaRN divides by the whole '
                'factor',
        reference=model_lines(382, 383)),
    ReleasedConstant(
        symbol=SWIGLU_LIMIT, released_value='10',
        meaning='the limit the two branches of an expert are clamped at',
        reference=inference_config_lines(19)),
    ReleasedConstant(
        symbol=ROUTER_TEMPERATURE, released_value='1',
        meaning='the temperature the router divides its logits by',
        reference=model_lines(67)),
    ReleasedConstant(
        symbol=NORM_EPSILON, released_value='10^{-20}',
        meaning='the number added under the square root of every RMS normalisation',
        reference=inference_config_lines(23)),
    ReleasedConstant(
        symbol=MHC_EPSILON, released_value='10^{-6}',
        meaning='the number added in the Sinkhorn normalisations and to the '
                'collapse coefficients',
        reference=inference_config_lines(42)),
    ReleasedConstant(
        symbol=ROUTER_EPSILON, released_value='10^{-20}',
        meaning='the number added to the sum the kept gates are divided by',
        reference=model_lines(825)),
    ReleasedConstant(
        symbol=ENGRAM_GATE_FLOOR, released_value='10^{-6}',
        meaning='the smallest magnitude the Engram gate takes the square root of',
        reference=model_lines(341)),
)
