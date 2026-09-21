'''Single-Pass mHC, the four-stream residual of DeepSeek-V4.1-Flash.

Written by Claude Opus 5, effort high. Revised by Claude Fable 5.1, effort 80.

The residual stream is four parallel copies of the hidden state. Each sublayer predicts
a collapse vector of four numbers, an output vector of four numbers and a four-by-four
combine matrix made doubly stochastic by Sinkhorn normalisation, each by a linear map
of its own from the normalised streams. The released code divides the streams by their
root mean square and multiplies by no learned weight, so `normalise_streams` writes an
`ops.Normalize` that declares neither a gain nor a bias. The sublayer reads the four streams collapsed by
the vector the sublayer before it predicted, so the coefficient prediction and the input
mixing do not depend on each other, which is what Single-Pass means.

The released code holds the three maps as one weight of 24 rows, `hc_attn_fn` or
`hc_ffn_fn`, and cuts the 24 results into 4, 4 and 16. A linear map followed by a slice of
its results is the linear map whose weight is that slice of the rows, so the reviewer
ruled on 2026-09-17 that the model writes three maps, `H_0`, `H_1` and `H_2`, and no
slice. The released scale of each part, one entry of `hc_attn_scale`, multiplies one of
the three weights and is part of it.

The released kernel takes a softmax along each row of the combine matrix, normalises the
columns, and then runs nineteen rounds of a row and a column normalisation. A softmax is
the exponential of every entry followed by the division of each row by its sum, so the
same function is the exponential followed by twenty rounds, which is the form the reviewer
asked for the same day and the count `hc_sinkhorn_iters` of the released configuration.
The equality holds because the model leaves the epsilons out, as
`mhc_with_epsilons.py` records: the released code adds one after the softmax and one to
every later sum.

The three sets are predicted from one token's own streams alone, so the prediction is one
box computed once per token. `predict_one_token` is the body and
`predict_over_all_tokens` writes the same prediction out over every token, so that
`discovering_broadcasts.confirm_broadcast_expansion` can check the box against it.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    axes, hold, l1_norm_over, over, route)
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    doubled_sigmoid, exponential, sigmoid)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    COLLAPSE, N, R, X, m, n, state, x)
from notebooks.sota.DeepSeekV41Flash.reference_links import kernel_lines, model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

SINKHORN_ITERATIONS = 20


def sinkhorn() -> cat.BroadcastedCategory:
    '''One token's combine matrix made doubly stochastic: the exponential of every
    entry, then twenty rounds of a row normalisation and a column normalisation.

    The combine matrix is `(n, N)`, so the row normalisation divides by the sum
    over position 1 and the column normalisation by the sum over position 0. The
    exponential and the first row normalisation are the row softmax of the released
    kernel.
    '''
    rows = l1_norm_over((n, N), 1)
    columns = l1_norm_over((n, N), 0)
    return (over((n, N), exponential())
            @ cat.Block.template(rows @ columns, title=text.SINKHORN_TITLE,
                                 description=text.SINKHORN_DESCRIPTION,
                                 repetition=SINKHORN_ITERATIONS,
                                 references=(model_lines(104), kernel_lines(436, 448),
                                             kernel_lines(450, 458))))


def normalise_streams() -> cat.Broadcasted:
    '''One token's four streams divided by the root of the mean of their squares over
    every stream and channel, with no learned weight after the division.

    The `ops.Normalize` takes neither a gain nor a bias, because the released code
    multiplies by no learned weight and adds none here, and it adds no epsilon under
    the root, because this module writes the exact function.
    `notebooks.sota.DeepSeekV41Flash.mhc_with_epsilons` supplies the epsilon
    the released kernel adds.
    '''
    return ops.Normalize.template((n, m), gain=False, bias=False,
                                  epsilon=ops.NO_EPSILON)


def predict_one_token() -> cat.Block:
    '''One token's collapse vector, output vector and doubly stochastic combine matrix,
    each from a linear map of its own that reads the token's four normalised streams.'''
    streams = cat.Array(R, (n, m))
    return cat.Block.template(
        normalise_streams()
        @ route((0, 0, 0), (streams,))
        @ ((ops.Linear.template((n, m), (n,), 'H_0', bias=True) @ sigmoid())
           * (ops.Linear.template((n, m), (n,), 'H_1', bias=True) @ doubled_sigmoid())
           * (ops.Linear.template((n, m), (n, N), 'H_2', bias=True) @ sinkhorn())),
        title=text.COEFFICIENTS_TITLE, fill_color='#E8DFF5',
        description=text.PREDICT_ONE_TOKEN_DESCRIPTION,
        references=(model_lines(907, 915), model_lines(941, 946),
                    kernel_lines(426, 431)))


COEFFICIENTS_BODY = predict_one_token()


def predict_over_all_tokens() -> cat.BroadcastedCategory:
    '''The same prediction written out over every token, which the box is confirmed
    against. It lifts the body the box holds rather than building a second one, because
    `cat.Block.template` mints a fresh tag per call and the Sinkhorn loop's tag is part
    of the expression the comparison reads.'''
    return over((x,), COEFFICIENTS_BODY)


COEFFICIENTS = discovering_broadcasts.broadcast_block_over_axes(
    COEFFICIENTS_BODY, (x,), ((0,),), 'Coef')
COEFFICIENTS_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    predict_over_all_tokens(), COEFFICIENTS)


def mixing_coefficients() -> cat.Broadcasted:
    '''The three sets of coefficients as one box computed once per token.'''
    return COEFFICIENTS


def collapse_streams() -> cat.Broadcasted:
    '''The four streams weighted by the collapse vector and summed into one.'''
    return over((x,), ops.Einops.template('n, n m -> m'))


def write_streams_back() -> cat.BroadcastedCategory:
    '''The output vector times the sublayer's result, plus the combine matrix applied to
    the four streams the sublayer came in on.'''
    return (((over((x,), ops.Einops.template('m, n -> n m')))
             * (over((x,), ops.Einops.template('n m, n N -> N m'))))
            @ ops.AdditionOp.template())


def mhc_sublayer(body: cat.BroadcastedCategory) -> cat.Block:
    '''One sublayer inside the four-stream residual. Position 0 is the residual and
    position 1 is the collapse vector the previous sublayer predicted. `body` reads the
    collapsed and normalised hidden state and returns the hidden state, so anything it
    shares with another layer goes through the tape.'''
    if tuple(body.dom()) != (state,) or tuple(body.cod()) != (state,):
        raise ValueError(
            f'a sublayer body reads and writes the hidden state alone, and this one '
            f'has the domain and codomain {axes(body)}')
    collapse, output, combine_matrix = COEFFICIENTS.cod()
    return cat.Block.template(
        route((1, 1, 1, 0), (COLLAPSE, X))
        @ (COEFFICIENTS * hold(X) * hold(X) * hold(COLLAPSE))
        @ route((0, 1, 2, 3, 5, 4), (collapse, output, combine_matrix, X, X, COLLAPSE))
        @ (hold(collapse) * hold(output) * hold(combine_matrix) * hold(X)
           * collapse_streams())
        @ (hold(collapse) * hold(output) * hold(combine_matrix) * hold(X)
           * (ops.Normalize.template() @ body))
        @ route((0, 4, 1, 3, 2), (collapse, output, combine_matrix, X, state))
        @ (hold(collapse) * write_streams_back()),
        title=text.MHC_TITLE, fill_color='#F1F4C1',
        description=text.MHC_SUBLAYER_DESCRIPTION,
        references=(model_lines(938), model_lines(955, 959)))
