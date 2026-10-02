# Claude Opus 5.5 (1M context), effort 40.
'''The delta attention and the mix of the attention residuals of Kimi K3, run in
PyTorch at small sizes, and their weights handed to the expression.

`KimiDeltaAttention.forward` of `modeling_kimi_linear.py` calls `ShortConvolution`,
`chunk_kda` and `FusedRMSNormGated` of `fla`, whose kernels are written in Triton and
run on a GPU. `fla` ships a PyTorch statement of the two functions the kernel computes,
`naive_kda_lowerbound_gate` in `fla/ops/kda/gate.py` and `naive_recurrent_kda` in
`fla/ops/kda/naive.py`, and both are copied here from the tag `v0.5.2`, under the MIT
licence of `fla`, with the batch axis kept. The convolution, the L2 normalisation and
the gated normalisation are written from the lines of `fla` that `reference_links` pins:
a depthwise `conv1d` padded by three zeros in front and cut to the length of the
sequence, followed by the SiLU; `x / sqrt(sum(x^2) + 1e-6)`; and
`x / sqrt(mean(x^2) + 1e-5) * weight * sigmoid(g)`. `_apply_attn_res` is copied from
`modeling_kimi_linear.py`. Every tensor is `torch.float64`.

    kimi_delta_attention        `KimiDeltaAttention.forward` on one sequence
    apply_attn_res              `_apply_attn_res`, lines 1075 to 1088
    delta_weights_of_the_expression
                                the reference's weights, reshaped into the weight of
                                every `Linear` of the expression
'''
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from notebooks.caching.mamba.evaluate_numerically import DTYPE, LinearValues

GATE_LOWER_BOUND = -5.0
NORM_EPSILON = 1e-5
L2_NORM_EPSILON = 1e-6


def naive_kda_lowerbound_gate(g, A_log, dt_bias=None, lower_bound=-5.0,
                              output_dtype=torch.float32):
    '''Copied from `fla/ops/kda/gate.py`, lines 58 to 70, at `v0.5.2`, with the cast
    to `torch.float` replaced by a cast to `torch.float64`.'''
    H, _ = g.shape[-2:]
    g = g.to(DTYPE)
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None,
                        output_final_state=False):
    '''Copied from `fla/ops/kda/naive.py`, lines 12 to 66, at `v0.5.2`, with the cast
    to `torch.float` replaced by a cast to `torch.float64`.'''
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(DTYPE), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def short_convolution(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    '''`ShortConvolution` on one sequence `x` of shape `[T, C]`, with the weight of
    shape `[C, 1, K]` of the `nn.Conv1d` it inherits: a depthwise convolution padded by
    `K - 1` zeros in front, cut to the length of the sequence, and the SiLU.'''
    length, channels = x.shape
    convolved = F.conv1d(x.T[None], weight, padding=weight.shape[-1] - 1, groups=channels)
    return F.silu(convolved[0, :, :length].T)


def l2_normalised(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + L2_NORM_EPSILON)


def rms_norm_gated(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    '''`FusedRMSNormGated` with the sigmoid activation.'''
    return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPSILON)
            * weight * torch.sigmoid(g))


@dataclass(frozen=True)
class DeltaAttentionParameters:
    '''The parameters of `KimiDeltaAttention`, with the shapes the reference declares
    them in.'''
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_conv1d: torch.Tensor
    k_conv1d: torch.Tensor
    v_conv1d: torch.Tensor
    A_log: torch.Tensor
    f_a_proj: torch.Tensor
    f_b_proj: torch.Tensor
    dt_bias: torch.Tensor
    b_proj: torch.Tensor
    g_proj: torch.Tensor
    o_norm: torch.Tensor
    o_proj: torch.Tensor

    @property
    def heads(self) -> int:
        return self.A_log.shape[0]

    @property
    def head_width(self) -> int:
        return self.f_a_proj.shape[0]


def random_delta_attention_parameters(hidden: int, heads: int, head_width: int,
                                      taps: int, generator: torch.Generator
                                      ) -> DeltaAttentionParameters:
    '''Parameters at the shapes of `KimiDeltaAttention.__init__`, drawn at random.'''
    projection = heads * head_width

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=DTYPE) / shape[-1] ** 0.5

    return DeltaAttentionParameters(
        q_proj=draw(projection, hidden), k_proj=draw(projection, hidden),
        v_proj=draw(projection, hidden), q_conv1d=draw(projection, 1, taps),
        k_conv1d=draw(projection, 1, taps), v_conv1d=draw(projection, 1, taps),
        A_log=torch.log(torch.empty(heads, dtype=DTYPE).uniform_(1, 16, generator=generator)),
        f_a_proj=draw(head_width, hidden), f_b_proj=draw(projection, head_width),
        dt_bias=draw(projection), b_proj=draw(heads, hidden), g_proj=draw(projection, hidden),
        o_norm=1 + draw(head_width), o_proj=draw(hidden, projection))


def kimi_delta_attention(hidden_states: torch.Tensor, p: DeltaAttentionParameters
                         ) -> torch.Tensor:
    '''`KimiDeltaAttention.forward` on one sequence of shape `[T, hidden]`, on the path
    it takes with no cache and no padding mask, lines 580 to 663.'''
    length = hidden_states.shape[0]
    heads, head_width = p.heads, p.head_width
    q = short_convolution(hidden_states @ p.q_proj.T, p.q_conv1d)
    k = short_convolution(hidden_states @ p.k_proj.T, p.k_conv1d)
    v = short_convolution(hidden_states @ p.v_proj.T, p.v_conv1d)
    g = (hidden_states @ p.f_a_proj.T @ p.f_b_proj.T).view(length, heads, head_width)
    beta = hidden_states @ p.b_proj.T
    q, k, v = (tensor.view(length, heads, head_width) for tensor in (q, k, v))
    g = naive_kda_lowerbound_gate(g, p.A_log, p.dt_bias, GATE_LOWER_BOUND, DTYPE)
    o, _ = naive_recurrent_kda(
        l2_normalised(q)[None], l2_normalised(k)[None], v[None], g[None],
        torch.sigmoid(beta)[None])
    gate = (hidden_states @ p.g_proj.T).view(length, heads, head_width)
    o = rms_norm_gated(o[0], gate, p.o_norm)
    return o.reshape(length, heads * head_width) @ p.o_proj.T


def delta_weights_of_the_expression(p: DeltaAttentionParameters
                                    ) -> tuple[dict[str, LinearValues],
                                               dict[str, torch.Tensor]]:
    '''The weight of every `Linear` of the delta attention, over the axes of its input
    followed by the axes of its output, and the gain of its normalisation. Tap `i_w` of
    the expression reads token `i_x - i_w`, and `nn.Conv1d` stores the weight of that
    token at position `K - 1 - i_w`, so the taps are reversed.'''
    heads, head_width = p.heads, p.head_width
    hidden = p.q_proj.shape[1]

    def onto_heads(weight: torch.Tensor) -> torch.Tensor:
        return weight.T.reshape(weight.shape[1], heads, head_width)

    def taps(conv: torch.Tensor) -> torch.Tensor:
        return conv[:, 0, :].flip(-1).T.reshape(conv.shape[-1], heads, head_width)

    weights = {
        'W^{Q}': LinearValues(onto_heads(p.q_proj)),
        'W^{K}': LinearValues(onto_heads(p.k_proj)),
        'W^{V}': LinearValues(onto_heads(p.v_proj)),
        'W^{Cq}': LinearValues(taps(p.q_conv1d)),
        'W^{Ck}': LinearValues(taps(p.k_conv1d)),
        'W^{Cv}': LinearValues(taps(p.v_conv1d)),
        'W^{Fa}': LinearValues(p.f_a_proj.T),
        'W^{Fb}': LinearValues(onto_heads(p.f_b_proj)),
        'b^{\\Delta}': LinearValues(p.dt_bias.view(heads, head_width)),
        'A_{\\log}': LinearValues(p.A_log),
        'W^{\\beta}': LinearValues(p.b_proj.T),
        'W^{Gk}': LinearValues(onto_heads(p.g_proj)),
        'W^{Ok}': LinearValues(p.o_proj.T.reshape(heads, head_width, hidden)),
    }
    return weights, {'RMSNorm': p.o_norm}


def apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    '''Copied from `_apply_attn_res` of `modeling_kimi_linear.py`, lines 1075 to 1088,
    with the projection and the normalisation passed as their weights.'''
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.to(DTYPE)
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.to(DTYPE) * proj_weight.squeeze(0).to(DTYPE)
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)
