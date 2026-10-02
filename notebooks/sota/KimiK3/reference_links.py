# Claude Opus 5.5 (1M context), effort 40.
'''The pinned links into the reference implementation of Kimi K3 that the blocks carry.

Moonshot AI ships the inference code of Kimi K3 inside the model repository
`moonshotai/Kimi-K3` on Hugging Face, as remote code named by the `auto_map` of the
configuration. The text model is `KimiLinearForCausalLM` in `modeling_kimi_linear.py`,
configured by `KimiLinearConfig` in `configuration_kimi_k3.py`, and
`KimiK3ForConditionalGeneration` in `modeling_kimi_k3.py` wraps it with the vision
encoder. The four files and the `config.json` of the checkpoint were read at the commit
`f831ab668142` of `moonshotai/Kimi-K3` on 2026-09-28.

The delta attention of the reference calls three kernels and two modules of the
`fla-core` package, which the modeling file imports from `fla` and asks to be installed
with `pip install -U fla-core`. The package was read at the tag `v0.5.2`, the release
published on the day the weights were released, 2026-07-27, and the newest release on
2026-09-28. `naive.py` of that release holds a reference loop for the delta rule, and
`gate.py` holds the reference form of the gate the kernel computes, so the recurrence
and the gate are cited from those two functions and from the arguments `chunk_kda`
reads.

`term_utilities.code_references.pinned_link` writes the links, so a line number stays
valid after either repository moves on.
'''
from __future__ import annotations

import data_structure.Category as cat
from term_utilities.code_references import pinned_link
from websocket_transfer.auxiliary_information import OperatorRole

CHECKPOINT_REPOSITORY = 'moonshotai/Kimi-K3'
CHECKPOINT_COMMIT = 'f831ab66814297da540d832a5235f8e904f29d06'
READ_ON = '2026-09-28'
CHECKPOINT_URL = (
    f'https://huggingface.co/{CHECKPOINT_REPOSITORY}/blob/{CHECKPOINT_COMMIT}/')

FLA_REPOSITORY = 'fla-org/flash-linear-attention'
FLA_COMMIT = '9c8e42e762fce087c27b673af4922795d9edb85e'
FLA_TAG = 'v0.5.2'
FLA_URL = f'https://github.com/{FLA_REPOSITORY}/blob/{FLA_COMMIT}/'

MODELING_FILE = 'modeling_kimi_linear.py'
CONFIGURATION_FILE = 'configuration_kimi_k3.py'
WRAPPER_FILE = 'modeling_kimi_k3.py'


def line_label(file: str, line: int, end_line: int | None) -> str:
    '''The file name and its lines, which is the label a link carries.'''
    lines = (f'L{line}' if end_line is None or end_line == line
             else f'L{line}-L{end_line}')
    return f'{file} {lines}'


def modeling_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`modeling_kimi_linear.py` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, MODELING_FILE, line, end_line,
                       label=line_label(MODELING_FILE, line, end_line))


def configuration_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`configuration_kimi_k3.py` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, CONFIGURATION_FILE, line, end_line,
                       label=line_label(CONFIGURATION_FILE, line, end_line))


def wrapper_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`modeling_kimi_k3.py` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, WRAPPER_FILE, line, end_line,
                       label=line_label(WRAPPER_FILE, line, end_line))


def checkpoint_config_lines(
    line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''The `config.json` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, 'config.json', line, end_line)


def checkpoint_file(path: str) -> cat.CodeReference:
    '''A file of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, path)


def fla_lines(path: str, line: int, end_line: int | None = None) -> cat.CodeReference:
    '''A file of `fla-core` at the tag `v0.5.2`.'''
    return pinned_link(FLA_URL, path, line, end_line,
                       label=line_label(f'fla {path.rsplit("/", 1)[-1]}', line, end_line))


def declared_at(role: str, line: int, end_line: int | None = None) -> OperatorRole:
    '''The role of a weight with the line of `modeling_kimi_linear.py` that declares
    it.'''
    return OperatorRole(role=role, references=(modeling_lines(line, end_line),))


RECURRENCE = fla_lines('fla/ops/kda/naive.py', 12, 66)
'''`naive_recurrent_kda`, the loop over the tokens that the chunked kernel computes in
blocks of 64 tokens.'''
GATE = fla_lines('fla/ops/kda/gate.py', 58, 70)
'''`naive_kda_lowerbound_gate`, the form of the decay that `chunk_kda` computes when it is
given a lower bound.'''
KERNEL_ARGUMENTS = fla_lines('fla/ops/kda/chunk.py', 178, 262)
'''The signature and the documented arguments of `chunk_kda`.'''
KERNEL_SCALE = fla_lines('fla/ops/kda/chunk.py', 414, 415)
'''The scale `K ** -0.5` that `chunk_kda` takes when it is given none.'''
KERNEL_NORMALISES_QUERIES_AND_KEYS = fla_lines('fla/ops/kda/chunk.py', 54, 58)
KERNEL_SIGMOID_OF_BETA = fla_lines('fla/ops/kda/chunk.py', 60, 62)
KERNEL_COMPUTES_THE_GATE = fla_lines('fla/ops/kda/chunk_fwd.py', 43, 56)
L2_NORM = fla_lines('fla/modules/l2norm.py', 148, 160)
L2_NORM_KERNEL = fla_lines('fla/modules/l2norm.py', 27, 46)
SHORT_CONVOLUTION = fla_lines('fla/modules/conv/short_conv.py', 24, 69)
GATED_NORM = fla_lines('fla/modules/fused_norm_gate.py', 1001, 1061)
GATED_NORM_KERNEL = fla_lines('fla/modules/fused_norm_gate.py', 98, 104)
