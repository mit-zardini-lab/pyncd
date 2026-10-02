# Claude Opus 5.5 (1M context), effort 40.
'''The SiTU-GLU feed-forward map of Kimi K3, which is the dense MLP of layer 0, the
shared experts of every later layer, and each routed expert.

The map is `KimiMLP` of the reference, and `KimiBlockSparseMLP` for a routed expert. It
projects one token's hidden state twice, onto a gate and an up branch, joins the two
with `SituAndMul`, and projects the result back. `SituAndMul` bounds both branches with
a scaled hyperbolic tangent and multiplies them:

    SiTU(gate, up) = beta tanh(gate / beta) sigma(gate) * beta_u tanh(up / beta_u)

with `beta` the `activation_situ_beta` of 4 and `beta_u` the
`activation_situ_linear_beta` of 25. A bounded tangent is close to the identity for an
input much smaller than its bound and never leaves the interval between minus the bound
and the bound, so SiTU behaves as a SwiGLU on small activations and caps the large ones.
`nm` has no hyperbolic tangent, and `tanh(y) = 2 sigma(2 y) - 1` writes it with the
sigmoid the package has.

The reference joins the two projections with `torch.cat` and splits the result in half
inside `SituAndMul`, which is the pair of branches read one after the other. The
expression keeps the two branches on two wires.

Every operation of the dense MLP is the same at every token, so it is one box computed
once per token, built by `discovering_broadcasts.broadcast_block_over_axes`.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import over, route
from notebooks.sota.DeepSeekV41Flash.custom_operations import multiply_along
from notebooks.sota.KimiK3.declared_axes import R, g, m, x
from notebooks.sota.KimiK3.reference_links import checkpoint_config_lines, modeling_lines
from notebooks.sota.KimiK3.released_constants import SITU_GATE_BOUND, SITU_UP_BOUND
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

DENSE_COLOUR = '#DFF0D8'
DENSE_BOX = 'MLP'
SITU_GATE_NAME = '\\beta \\tanh(x / \\beta)\\, \\sigma(x)'
SITU_UP_NAME = '\\beta_{\\mathrm{u}} \\tanh(x / \\beta_{\\mathrm{u}})'

SITU_REFERENCES = (modeling_lines(64, 91), checkpoint_config_lines(20, 21))
FEED_FORWARD_REFERENCES = (modeling_lines(273, 301), *SITU_REFERENCES)


def bounded_tangent(bound: nm.Numeric) -> nm.Numeric:
    '''`bound tanh(x / bound)`, written with the sigmoid as
    `bound (2 sigma(2 x / bound) - 1)`.'''
    return bound * (nm.Integer(2) * nm.Sigmoid(nm.Integer(2) * nm.x / bound)
                    - nm.Integer(1))


def situ_gate() -> cat.Broadcasted:
    '''The gate branch of SiTU: the bounded tangent of the gate times its sigmoid.'''
    return ops.Arithmetic.template(
        bounded_tangent(SITU_GATE_BOUND) * nm.Sigmoid(nm.x),
        name=fd.DynamicName(SITU_GATE_NAME))


def situ_up() -> cat.Broadcasted:
    '''The up branch of SiTU: the bounded tangent of the up projection.'''
    return ops.Arithmetic.template(bounded_tangent(SITU_UP_BOUND),
                                   name=fd.DynamicName(SITU_UP_NAME))


def situ_glu[A: cat.Axis](
    source: tuple[A, ...],
    width: A,
    gate_name: str,
    up_name: str,
    down_name: str,
    target: tuple[A, ...] | None = None,
) -> cat.BroadcastedCategory:
    '''One token's array over `source` through a gate projection and an up projection
    onto `width`, the two branches bounded and multiplied, and the down projection
    onto `target`, which is `source` where it is not given.'''
    token = cat.Array(R, source)
    return (route((0, 0), (token,))
            @ ((ops.Linear.template(source, (width,), gate_name) @ situ_gate())
               * (ops.Linear.template(source, (width,), up_name) @ situ_up()))
            @ multiply_along(width)
            @ ops.Linear.template((width,), target if target is not None else source,
                                  down_name))


DENSE_BODY = cat.Block.template(
    situ_glu((m,), g, 'W^{G0}', 'W^{U0}', 'W^{D0}'),
    title=text.DENSE_MLP_TITLE, fill_color=DENSE_COLOUR,
    description=text.DENSE_MLP_DESCRIPTION,
    references=(*FEED_FORWARD_REFERENCES, modeling_lines(893, 900),
                checkpoint_config_lines(46), checkpoint_config_lines(56)))
DENSE_MLP = discovering_broadcasts.broadcast_block_over_axes(
    DENSE_BODY, (x,), ((0,),), DENSE_BOX)
DENSE_MLP_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    over((x,), DENSE_BODY), DENSE_MLP)
