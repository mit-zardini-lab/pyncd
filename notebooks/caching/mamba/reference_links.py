# Claude Opus 5.5 (1M context), effort 40.
'''The pinned links into the reference implementation of Mamba.

The layer is read from `mamba_ssm/modules/mamba_simple.py` and the reference scan from
`selective_scan_ref` in `mamba_ssm/ops/selective_scan_interface.py`, both at the commit
`e9594ce1c732d97440f0332fdc43170a2294dbfa` of `state-spaces/mamba`, read on 2026-09-26.
'''
from __future__ import annotations

import data_structure.Category as cat
from term_utilities.code_references import pinned_link

MAMBA_REPOSITORY = 'state-spaces/mamba'
MAMBA_COMMIT = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
MAMBA_URL = f'https://github.com/{MAMBA_REPOSITORY}/blob/{MAMBA_COMMIT}/'
MAMBA_READ_ON = '2026-09-26'
LAYER_PATH = 'mamba_ssm/modules/mamba_simple.py'
SCAN_PATH = 'mamba_ssm/ops/selective_scan_interface.py'


def line_label(file: str, line: int, end_line: int | None) -> str:
    '''The file name and its lines, which is the label a link carries.'''
    lines = (f'L{line}' if end_line is None or end_line == line
             else f'L{line}-L{end_line}')
    return f'{file} {lines}'


def layer_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`mamba_simple.py` at the pinned commit.'''
    return pinned_link(MAMBA_URL, LAYER_PATH, line, end_line,
                       label=line_label('mamba_simple.py', line, end_line))


def scan_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`selective_scan_interface.py` at the pinned commit.'''
    return pinned_link(MAMBA_URL, SCAN_PATH, line, end_line,
                       label=line_label('selective_scan_interface.py', line, end_line))


SIZES_OF_THE_LAYER = layer_lines(32, 60)
INPUT_PROJECTION = layer_lines(135, 141)
BRANCHES_CUT_APART = layer_lines(162)
CONVOLUTION_DECLARED = layer_lines(64, 72)
CONVOLUTION_APPLIED = layer_lines(169)
STATE_MAPS_PROJECTED = layer_lines(182, 187)
DECAY_FROM_ITS_LOGARITHM = layer_lines(143)
DECAY_LOGARITHM_DECLARED = layer_lines(103, 111)
SKIP_DECLARED = layer_lines(114)
SCAN_CALLED = layer_lines(189, 203)
OUTPUT_PROJECTION = layer_lines(204, 205)
STEP_SIZE_SOFTPLUS = scan_lines(145, 148)
DISCRETISATION = scan_lines(160, 167)
RECURRENCE = scan_lines(173, 188)
SKIP_AND_GATE = scan_lines(189, 191)
CONVOLUTION_STATE_AT_PREFILL = layer_lines(164, 167)
CONVOLUTION_STEP = layer_lines(214, 221)
STATE_STEP = layer_lines(237, 246)
STATES_ALLOCATED = layer_lines(255, 266)
