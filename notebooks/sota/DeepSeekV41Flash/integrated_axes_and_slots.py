'''The tape slots that the integrated DeepSeek-V4.1-Flash adds to
`notebooks.sota.DeepSeekV41Flash.declared_axes`, and the constants it reads.

Written by Claude Fable 5.1, reasoning effort 80.

The axes of the omitted mechanisms are declared once, in
`notebooks.sota.DeepSeekV41Flash.omitted_mechanisms`, and the named constants once, in
`notebooks.sota.DeepSeekV41Flash.released_constants`, and both are imported from there,
so a wire of the integrated model carries the same axis and the same symbol as the wire
of the mechanism drawn on its own. The constants stood in this module until 2026-09-19
and are re-exported here under the names the integrated modules read them by.
'''
from __future__ import annotations

from notebooks.sota.DeepSeekV41Flash.construction_idioms import named_slot
from notebooks.sota.DeepSeekV41Flash.released_constants import (
    ENGRAM_GATE_FLOOR as ENGRAM_GATE_FLOOR,
    MHC_EPSILON as MHC_EPSILON,
    MODALITY as MODALITY,
    NORM_EPSILON as NORM_EPSILON,
    RELEASED_CONSTANTS as RELEASED_CONSTANTS,
    ROTARY_BASE as ROTARY_BASE,
    ROUTER_EPSILON as ROUTER_EPSILON,
    ROUTER_TEMPERATURE as ROUTER_TEMPERATURE,
    SWIGLU_LIMIT as SWIGLU_LIMIT,
    WINDOW_ROTARY_BASE as WINDOW_ROTARY_BASE,
    YARN_FACTOR as YARN_FACTOR,
    YARN_RAMP_END as YARN_RAMP_END,
    YARN_RAMP_START as YARN_RAMP_START,
    ReleasedConstant as ReleasedConstant,
)

SLOT_IDS = named_slot('\\mathrm{ids}')
SLOT_MODALITY = named_slot('\\mathrm{mod}')
SLOT_STREAM_MEAN = named_slot('\\mathrm{mean}')
