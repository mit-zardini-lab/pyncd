# Claude Fable 5.1, effort 80.
'''The pinned links into the released DeepSeek-V4.1-Flash code that the blocks carry.

The released code is the git repository `deepseek-ai/DeepSeek-V4.1-Flash` on Hugging
Face, read at commit `dba1be0a40aa` on 2026-09-11. Every link is pinned to that commit,
so a line number stays valid after the repository moves on. Each
block of the model carries the links of its own mechanism in its aesthetics, so an
inspection box opened over the block on the tsncd page links to the code it expresses.
`term_utilities.code_references.pinned_link` writes the links. A block carries no
reference into this package, as the reviewer ruled on 2026-09-16. The lines of the
embedding and of the output head in `whole_model.py`, and the lines
`operator_explanations.py` cites, were read from `inference/model.py` at the same commit
on 2026-09-16. The lines `quantised_text_only_model.py` cites for the quantisation of each
tensor and each weight were read from `inference/model.py`, `inference/kernel.py`,
`inference/convert.py` and `inference/generate.py` at the same commit on 2026-09-20.
'''
from __future__ import annotations

import data_structure.Category as cat
from term_utilities.code_references import pinned_link
from websocket_transfer.auxiliary_information import OperatorRole


REPOSITORY = 'deepseek-ai/DeepSeek-V4.1-Flash'
COMMIT = 'dba1be0a40aa45a94ad051997016db3960a90277'
READ_ON = '2026-09-11'
BASE_URL = f'https://huggingface.co/{REPOSITORY}/blob/{COMMIT}/'


def model_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/model.py` at the released commit.'''
    return pinned_link(BASE_URL, 'inference/model.py', line, end_line)


def kernel_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/kernel.py` at the released commit.'''
    return pinned_link(BASE_URL, 'inference/kernel.py', line, end_line)


def inference_config_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/config.json`, the layer plan, at the released commit.'''
    return pinned_link(BASE_URL, 'inference/config.json', line, end_line)


def top_level_config_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''The top-level `config.json` at the released commit.'''
    return pinned_link(BASE_URL, 'config.json', line, end_line)


def engram_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/engram.py`, the token map and the n-gram hash, at the released
    commit.'''
    return pinned_link(BASE_URL, 'inference/engram.py', line, end_line)


def vision_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/vision.py`, the vision encoder and its projector, at the released
    commit.'''
    return pinned_link(BASE_URL, 'inference/vision.py', line, end_line)


def image_processor_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/image_processor.py`, which cuts an image into patches, at the
    released commit.'''
    return pinned_link(BASE_URL, 'inference/image_processor.py', line, end_line)


def convert_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/convert.py`, which rewrites the checkpoint's weights into the quantisations
    the model loads, at the released commit.'''
    return pinned_link(BASE_URL, 'inference/convert.py', line, end_line)


def generate_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/generate.py`, which loads the model and sets the default quantisation, at
    the released commit.'''
    return pinned_link(BASE_URL, 'inference/generate.py', line, end_line)


def declared_at(role: str, line: int, end_line: int | None = None) -> OperatorRole:
    '''The role of a weight with the released line of `model.py` that declares it.'''
    return OperatorRole(role=role, references=(model_lines(line, end_line),))
