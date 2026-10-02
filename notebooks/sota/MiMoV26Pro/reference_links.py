# Claude Opus 5.5 (1M context), effort 40.
'''The pinned links into the reference implementation of MiMo-V2.6-Pro that the blocks
carry.

Xiaomi ships the reference implementation inside the checkpoint. The `auto_map` of the
`config.json` of `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` names
`modeling_mimo_v2.MiMoV2ForCausalLM` and `configuration_mimo_v2.MiMoV2Config`, and both
files stand beside the weights. The checkpoint was read at the commit `adea8e2c5373` on
Hugging Face on 2026-09-28. It is the MOPD upgrade of `XiaomiMiMo/MiMo-V2.6-Pro-RL`,
read at the commit `73875d00b30a` on the same date, and the two commits hold
byte-identical copies of `config.json`, `configuration_mimo_v2.py` and
`modeling_mimo_v2.py`. The technical report stands in the repository of the RL
checkpoint alone.

The remote code imports the attention masks and the cache from the Hugging Face
`transformers` library, and the `config.json` of the checkpoint was written by version
5.3.0. The two files of `transformers` that decide what a sliding window reads and what
its cache keeps are linked at the commit of the tag `v5.3.0`, read on 2026-09-28.

Every link is pinned to one of those commits, so a line number stays valid after the
repositories move on. `term_utilities.code_references.pinned_link` writes the links, and
each block of the model carries the links of its own mechanism in its aesthetics.
'''
from __future__ import annotations

import data_structure.Category as cat
from term_utilities.code_references import pinned_link
from websocket_transfer.auxiliary_information import OperatorRole

CHECKPOINT_REPOSITORY = 'XiaomiMiMo/MiMo-V2.6-Pro-MOPD'
CHECKPOINT_COMMIT = 'adea8e2c5373181e5a973fa1ecb343cb31af214b'
RL_CHECKPOINT_REPOSITORY = 'XiaomiMiMo/MiMo-V2.6-Pro-RL'
RL_CHECKPOINT_COMMIT = '73875d00b30a89ef8cc353a0b60b0e9f9561952d'
TRANSFORMERS_REPOSITORY = 'huggingface/transformers'
TRANSFORMERS_COMMIT = 'aad13b87ed59f2afcfaebc985f403301887a35fc'
TRANSFORMERS_VERSION = '5.3.0'
READ_ON = '2026-09-28'
CHECKPOINT_URL = (
    f'https://huggingface.co/{CHECKPOINT_REPOSITORY}/blob/{CHECKPOINT_COMMIT}/')
RL_CHECKPOINT_URL = (
    f'https://huggingface.co/{RL_CHECKPOINT_REPOSITORY}/blob/{RL_CHECKPOINT_COMMIT}/')
TRANSFORMERS_URL = (
    f'https://github.com/{TRANSFORMERS_REPOSITORY}/blob/{TRANSFORMERS_COMMIT}/')
LIBRARY_FOLDER = 'src/transformers/'
MODELING_FILE = 'modeling_mimo_v2.py'
CONFIGURATION_FILE = 'configuration_mimo_v2.py'
TECHNICAL_REPORT = 'MiMo_V2_6_technical_report.pdf'
BIAS_SHARD = 'model_pp0_ep0_shard0.safetensors'
'''The shard of the checkpoint holding the correction biases of the router of every
mixture of experts, as its header states.'''


def line_label(file: str, line: int, end_line: int | None) -> str:
    '''The file name and its lines, which is the label a link carries.'''
    lines = (f'L{line}' if end_line is None or end_line == line
             else f'L{line}-L{end_line}')
    return f'{file} {lines}'


def modeling_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`modeling_mimo_v2.py` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, MODELING_FILE, line, end_line,
                       label=line_label(MODELING_FILE, line, end_line))


def configuration_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`configuration_mimo_v2.py` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, CONFIGURATION_FILE, line, end_line,
                       label=line_label(CONFIGURATION_FILE, line, end_line))


def checkpoint_config_lines(
    line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''The `config.json` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, 'config.json', line, end_line)


def generation_config_lines(
    line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''The `generation_config.json` of the checkpoint at the pinned commit.'''
    return pinned_link(CHECKPOINT_URL, 'generation_config.json', line, end_line)


def checkpoint_file(path: str) -> cat.CodeReference:
    '''A file of the checkpoint at its pinned commit, such as a shard whose header gives
    the datatype and the place of each tensor it holds.'''
    return pinned_link(CHECKPOINT_URL, path)


def technical_report() -> cat.CodeReference:
    '''The technical report, in the repository of the RL checkpoint at its pinned
    commit.'''
    return pinned_link(RL_CHECKPOINT_URL, TECHNICAL_REPORT)


def library_lines(
    path: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file under `src/transformers/` at the commit of the tag `v5.3.0`.'''
    return pinned_link(TRANSFORMERS_URL, LIBRARY_FOLDER + path, line, end_line,
                       label=line_label(path, line, end_line))


def declared_at(role: str, line: int, end_line: int | None = None) -> OperatorRole:
    '''The role of a weight with the line of `modeling_mimo_v2.py` that declares it.'''
    return OperatorRole(role=role, references=(modeling_lines(line, end_line),))


SLIDING_WINDOW_MASK_REFERENCES: tuple[cat.CodeReference, ...] = (
    library_lines('masking_utils.py', 74, 77),
    library_lines('masking_utils.py', 90, 99),
    library_lines('masking_utils.py', 114, 118),
    library_lines('masking_utils.py', 1099))
'''The causal mask, the window laid over it, which keeps key `k` for query `q` where
`k > q - sliding_window`, the two joined, and the line reading the width of the window
from the configuration.'''

SLIDING_WINDOW_CACHE_REFERENCES: tuple[cat.CodeReference, ...] = (
    library_lines('cache_utils.py', 168, 217),
    library_lines('cache_utils.py', 958, 979))
'''The cache layer of a sliding window, which keeps the last `sliding_window - 1`
tokens, and the lines of `DynamicCache` that give that layer to every layer the
configuration names `sliding_attention`.'''

EAGER_ATTENTION_REFERENCES: tuple[cat.CodeReference, ...] = (
    library_lines('modeling_utils.py', 1145),
    library_lines('modeling_utils.py', 1910, 1942))
'''The default of `_supports_sdpa`, which `MiMoV2ForCausalLM` does not override, and the
lines that fall back to the eager attention when SDPA is not supported.'''
