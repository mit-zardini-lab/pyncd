# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The pinned links into the papers and the released code the three classic models are
checked against.

Each model's blocks carry the links of their own mechanism in their aesthetics, so an
inspection box opened over a block on the interactive page links to the text or the code
the block expresses. The links were read on 2026-09-23.

*Attention Is All You Need* is cited from version 7 of arXiv 1706.03762 by section,
because a PDF has no line to link. The paper links its code, `tensorflow/tensor2tensor`,
which is cited at commit `bafdc1b` for the settings the paper leaves unstated.

Mixtral-8x7B is cited from `mistralai/mistral-inference` at commit `9eaeb91`, which is
the reference implementation Mistral AI publishes, and from the `config.json` of
`mistralai/Mixtral-8x7B-v0.1` on Hugging Face at commit `fc7ac94`. The Hugging Face
repository holds the sizes of the released weights.

DeepSeek-V3 is cited from `deepseek-ai/DeepSeek-V3` at commit `9b4e978`, whose
`inference/model.py` is the released implementation and whose `inference/configs/`
holds the configurations.
'''
from __future__ import annotations

import data_structure.Category as cat
from term_utilities.code_references import pinned_link
from websocket_transfer.auxiliary_information import OperatorRole

READ_ON = '2026-09-23'

TRANSFORMER_PAPER_URL = 'https://arxiv.org/pdf/1706.03762v7'
TENSOR2TENSOR_COMMIT = 'bafdc1b67730430d38d6ab802cbd51f9d053ba2e'
TENSOR2TENSOR_BASE = (
    f'https://github.com/tensorflow/tensor2tensor/blob/{TENSOR2TENSOR_COMMIT}/')

MISTRAL_INFERENCE_COMMIT = '9eaeb91c17450e09021b6065a1d5cc69876507c8'
MISTRAL_INFERENCE_BASE = (
    f'https://github.com/mistralai/mistral-inference/blob/{MISTRAL_INFERENCE_COMMIT}/')
MIXTRAL_WEIGHTS_COMMIT = 'fc7ac94680e38d7348cfa806e51218e6273104b0'
MIXTRAL_WEIGHTS_BASE = (
    'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/'
    f'{MIXTRAL_WEIGHTS_COMMIT}/')
MIXTRAL_PAPER_URL = 'https://arxiv.org/pdf/2401.04088v1'

DEEPSEEK_V3_COMMIT = '9b4e9788e4a3a731f7567338ed15d3ec549ce03b'
DEEPSEEK_V3_BASE = (
    f'https://github.com/deepseek-ai/DeepSeek-V3/blob/{DEEPSEEK_V3_COMMIT}/')
DEEPSEEK_V3_PAPER_URL = 'https://arxiv.org/pdf/2412.19437v2'


def transformer_section(section: str) -> cat.CodeReference:
    '''A section of *Attention Is All You Need*, version 7.'''
    return cat.CodeReference(
        label=f'Attention Is All You Need, Section {section}',
        url=TRANSFORMER_PAPER_URL)


def tensor2tensor_lines(
    path: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file of `tensorflow/tensor2tensor` at the pinned commit.'''
    return pinned_link(TENSOR2TENSOR_BASE, f'tensor2tensor/{path}', line, end_line)


def transformer_role(role: str, *sections: str) -> OperatorRole:
    '''The role of a weight of the transformer with the sections that state it.'''
    return OperatorRole(
        role=role,
        references=tuple(transformer_section(section) for section in sections))


def mistral_lines(
    path: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file of `src/mistral_inference/` at the pinned commit.'''
    return pinned_link(
        MISTRAL_INFERENCE_BASE, f'src/mistral_inference/{path}', line, end_line)


def mixtral_config_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''The `config.json` of the released Mixtral-8x7B weights at the pinned commit.'''
    return pinned_link(MIXTRAL_WEIGHTS_BASE, 'config.json', line, end_line)


def mixtral_role(role: str, path: str, line: int,
                 end_line: int | None = None) -> OperatorRole:
    '''The role of a weight of Mixtral-8x7B with the line that declares it.'''
    return OperatorRole(role=role, references=(mistral_lines(path, line, end_line),))


def deepseek_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/model.py` of DeepSeek-V3 at the pinned commit.'''
    return pinned_link(DEEPSEEK_V3_BASE, 'inference/model.py', line, end_line)


def deepseek_config_lines(
    name: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A configuration under `inference/configs/` of DeepSeek-V3 at the pinned
    commit.'''
    return pinned_link(DEEPSEEK_V3_BASE, f'inference/configs/{name}', line, end_line)


def small_config_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`inference/configs/config_16B.json` of DeepSeek-V3, the sizes the hand-drawn
    diagram labels.'''
    return deepseek_config_lines('config_16B.json', line, end_line)


def released_config_lines(
    line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''`inference/configs/config_671B.json` of DeepSeek-V3, the released model.'''
    return deepseek_config_lines('config_671B.json', line, end_line)


def deepseek_role(role: str, line: int, end_line: int | None = None) -> OperatorRole:
    '''The role of a weight of DeepSeek-V3 with the line of `model.py` that declares
    it.'''
    return OperatorRole(role=role, references=(deepseek_lines(line, end_line),))


DEEPSEEK_V3_CHECKPOINT_COMMIT = 'e815299b0bcbac849fa540c768ef21845365c9eb'
DEEPSEEK_V3_CHECKPOINT_BASE = (
    'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/'
    f'{DEEPSEEK_V3_CHECKPOINT_COMMIT}/')
PYTORCH_2_4_1_COMMIT = 'ee1b6804381c57161c477caa380a840a84167676'
PYTORCH_2_4_1_BASE = (
    f'https://github.com/pytorch/pytorch/blob/{PYTORCH_2_4_1_COMMIT}/')
'''The sources of the quantisations of the released DeepSeek-V3, read on 2026-09-27
with the code above. `deepseek-ai/DeepSeek-V3` on Hugging Face holds the FP8 checkpoint
that `inference/convert.py` converts, and the header of each of its shards gives the
datatype of every tensor the shard holds. `inference/requirements.txt` pins PyTorch
2.4.1, whose tag names the commit above.'''


def deepseek_inference_lines(
    name: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file under `inference/` of DeepSeek-V3 at the pinned commit, such as
    `kernel.py` or `generate.py`.'''
    return pinned_link(DEEPSEEK_V3_BASE, f'inference/{name}', line, end_line)


def deepseek_repository_lines(
    path: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file of the repository of DeepSeek-V3 at the pinned commit, outside
    `inference/`.'''
    return pinned_link(DEEPSEEK_V3_BASE, path, line, end_line)


def deepseek_checkpoint_file(
    path: str, line: int | None = None, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file of the released checkpoint `deepseek-ai/DeepSeek-V3` at its pinned
    commit.'''
    return pinned_link(DEEPSEEK_V3_CHECKPOINT_BASE, path, line, end_line)


def pytorch_lines(
    path: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file of PyTorch at the commit of the tag `v2.4.1`, the version
    `inference/requirements.txt` of DeepSeek-V3 pins.'''
    return pinned_link(PYTORCH_2_4_1_BASE, path, line, end_line,
                       label=(f'PyTorch 2.4.1 {path} L{line}' if end_line is None
                              else f'PyTorch 2.4.1 {path} L{line}-L{end_line}'))
