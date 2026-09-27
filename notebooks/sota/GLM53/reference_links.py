# Claude Opus 5.5 (1M context), effort 40.
'''The pinned links into the reference implementation of GLM-5.3 that the blocks carry.

Z.ai ships no inference code in the model repository `zai-org/GLM-5.3` on Hugging Face.
Its model card names `GlmMoeDsaForCausalLM` as the architecture, and the reference
implementation of that class is the `glm_moe_dsa` model of the Hugging Face
`transformers` library. `modeling_glm_moe_dsa.py` and `configuration_glm_moe_dsa.py`
were read at the commit `7cd73d9df0c1` of `huggingface/transformers` on GitHub on
2026-09-23, which was the newest commit to touch the modeling file on that date. The
`config.json` of the checkpoint was read at the commit `aca966e4e027` of
`zai-org/GLM-5.3` on the same date. Every link is pinned to one of those two commits, so
a line number stays valid after either repository moves on.

The modeling file is generated from `modular_glm_moe_dsa.py` by the `transformers`
tooling, and the links name the generated file, because the generated file holds every
class the model runs in one place. `term_utilities.code_references.pinned_link` writes
the links, and each block of the model carries the links of its own mechanism in its
aesthetics, so an inspection box opened over the block on the page links to the lines
the block expresses.
'''
from __future__ import annotations

import data_structure.Category as cat
from term_utilities.code_references import pinned_link
from websocket_transfer.auxiliary_information import OperatorRole


TRANSFORMERS_REPOSITORY = 'huggingface/transformers'
TRANSFORMERS_COMMIT = '7cd73d9df0c14b151c684b708a9f27d8d0349dfe'
CHECKPOINT_REPOSITORY = 'zai-org/GLM-5.3'
CHECKPOINT_COMMIT = 'aca966e4e02791568aa6a4ced368624b3d897f42'
READ_ON = '2026-09-23'
MODEL_FOLDER = 'src/transformers/models/glm_moe_dsa/'
TRANSFORMERS_URL = (
    f'https://github.com/{TRANSFORMERS_REPOSITORY}/blob/{TRANSFORMERS_COMMIT}/')
CHECKPOINT_URL = (
    f'https://huggingface.co/{CHECKPOINT_REPOSITORY}/blob/{CHECKPOINT_COMMIT}/')


def modeling_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`modeling_glm_moe_dsa.py` at the pinned `transformers` commit.'''
    return pinned_link(TRANSFORMERS_URL, MODEL_FOLDER + 'modeling_glm_moe_dsa.py',
                       line, end_line,
                       label=line_label('modeling_glm_moe_dsa.py', line, end_line))


def configuration_lines(line: int, end_line: int | None = None) -> cat.CodeReference:
    '''`configuration_glm_moe_dsa.py` at the pinned `transformers` commit.'''
    return pinned_link(TRANSFORMERS_URL, MODEL_FOLDER + 'configuration_glm_moe_dsa.py',
                       line, end_line,
                       label=line_label('configuration_glm_moe_dsa.py', line, end_line))


def checkpoint_config_lines(
    line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''The `config.json` of the checkpoint at the pinned commit of `zai-org/GLM-5.3`.'''
    return pinned_link(CHECKPOINT_URL, 'config.json', line, end_line)


LIBRARY_FOLDER = 'src/transformers/'
BF16_CHECKPOINT_REPOSITORY = 'zai-org/GLM-5.3-BF16'
BF16_CHECKPOINT_COMMIT = '9d2398f478cab2de883137db3a36ad2c96205e24'
DEEP_GEMM_REPOSITORY = 'kernels-community/deep-gemm'
DEEP_GEMM_COMMIT = '9590415046fa95a187af7ea03391d4782047170d'
DEEP_GEMM_BUILD = 'build/torch210-cxx11-cu128-x86_64-linux/'
FINEGRAINED_FP8_REPOSITORY = 'kernels-community/finegrained-fp8'
FINEGRAINED_FP8_COMMIT = '3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8'
FINEGRAINED_FP8_BUILD = 'build/torch-cuda/'
QUANTISATION_READ_ON = '2026-09-26'
BF16_CHECKPOINT_URL = (
    f'https://huggingface.co/{BF16_CHECKPOINT_REPOSITORY}/blob/{BF16_CHECKPOINT_COMMIT}/')
DEEP_GEMM_URL = f'https://huggingface.co/{DEEP_GEMM_REPOSITORY}/blob/{DEEP_GEMM_COMMIT}/'
FINEGRAINED_FP8_URL = (
    f'https://huggingface.co/{FINEGRAINED_FP8_REPOSITORY}/blob/{FINEGRAINED_FP8_COMMIT}/')
'''The sources of the quantised model, read on 2026-09-26. `zai-org/GLM-5.3` at the
pinned commit is the FP8 checkpoint, and `zai-org/GLM-5.3-BF16` is the same model in
BF16. The FP8 checkpoint is run by the fine-grained FP8 integration of `transformers`,
read at the pinned `transformers` commit. That integration loads its matrix multiplies
from two kernel repositories on the Hugging Face hub, `kernels-community/deep-gemm` at
version 2 and `kernels-community/finegrained-fp8` at version 4, per
`integrations/hub_kernels.py` lines 726 and 727. The two commits are the ones the
branches `v2` and `v4` named on 2026-09-26, and the two builds are one CUDA build of
each, whose Python sources are the same in every build. DeepGEMM at that commit ships
builds for PyTorch 2.10, 2.11 and 2.12 alone, so the build named here is the one for
the PyTorch release pinned below, with CUDA 12.8.'''


def library_lines(path: str, line: int, end_line: int | None = None) -> cat.CodeReference:
    '''A file under `src/transformers/` at the pinned `transformers` commit.'''
    return pinned_link(TRANSFORMERS_URL, LIBRARY_FOLDER + path, line, end_line,
                       label=line_label(path, line, end_line))


def checkpoint_file(path: str) -> cat.CodeReference:
    '''A file of the FP8 checkpoint at its pinned commit, such as a shard whose header
    gives the datatype of each tensor it holds.'''
    return pinned_link(CHECKPOINT_URL, path)


def bf16_checkpoint_file(path: str) -> cat.CodeReference:
    '''A file of the BF16 checkpoint at its pinned commit.'''
    return pinned_link(BF16_CHECKPOINT_URL, path)


def deep_gemm_lines(path: str, line: int, end_line: int | None = None) -> cat.CodeReference:
    '''A file of the DeepGEMM build loaded by `transformers`, at the pinned commit.'''
    return pinned_link(DEEP_GEMM_URL, DEEP_GEMM_BUILD + path, line, end_line,
                       label=line_label(f'deep-gemm {path}', line, end_line))


def finegrained_fp8_lines(
    path: str, line: int, end_line: int | None = None,
) -> cat.CodeReference:
    '''A file of the Triton FP8 build loaded by `transformers`, at the pinned
    commit.'''
    return pinned_link(FINEGRAINED_FP8_URL, FINEGRAINED_FP8_BUILD + path, line, end_line,
                       label=line_label(f'finegrained-fp8 {path}', line, end_line))


PYTORCH_REPOSITORY = 'pytorch/pytorch'
PYTORCH_COMMIT = '449b1768410104d3ed79d3bcfe4ba1d65c7f22c0'
PYTORCH_URL = f'https://github.com/{PYTORCH_REPOSITORY}/blob/{PYTORCH_COMMIT}/'
'''PyTorch at the commit of the tag `v2.10.0`, read on 2026-09-27, in its wheel for
CUDA 12.8. The pinned `transformers` requires `torch>=2.5` and does not name one
version. Two facts of the quantised model are computed inside PyTorch rather than in
`transformers`: the kernel that `scaled_dot_product_attention` runs, and the
accumulator of `torch.sum` over a BF16 tensor. The release is the oldest for which
DeepGEMM ships a build. Its wheel for CUDA 12.8 installs cuDNN 9.10.2.21, and PyTorch
puts cuDNN attention first only above cuDNN 9.15.0, so the memory-efficient kernel
runs. The wheel for CUDA 13.0 installs cuDNN 9.15.1.9, under which cuDNN attention,
whose source is closed, runs on a Hopper GPU.'''


def pytorch_lines(path: str, line: int, end_line: int | None = None) -> cat.CodeReference:
    '''A file of PyTorch at the pinned commit.'''
    return pinned_link(PYTORCH_URL, path, line, end_line,
                       label=line_label(f'pytorch {path.rsplit("/", 1)[-1]}', line, end_line))


def line_label(file: str, line: int, end_line: int | None) -> str:
    '''The file name and its lines, which is the label a link carries. The folder of
    the model is left out of the label, because every link of the modeling code names
    the same folder.'''
    lines = (f'L{line}' if end_line is None or end_line == line
             else f'L{line}-L{end_line}')
    return f'{file} {lines}'


def declared_at(role: str, line: int, end_line: int | None = None) -> OperatorRole:
    '''The role of a weight with the line of `modeling_glm_moe_dsa.py` that declares
    it.'''
    return OperatorRole(role=role, references=(modeling_lines(line, end_line),))


PYTORCH_ENVIRONMENT_REFERENCES: tuple[cat.CodeReference, ...] = (
    pytorch_lines('.github/scripts/generate_binary_build_matrix.py', 75),
    pytorch_lines('aten/src/ATen/native/transformers/cuda/sdp_utils.cpp', 78, 116))
'''The line pinning cuDNN 9.10.2.21 in the wheel of PyTorch 2.10.0 for CUDA 12.8, and
the lines putting cuDNN attention first only above cuDNN 9.15.0.'''
