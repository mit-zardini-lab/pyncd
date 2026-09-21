'''Single-Pass mHC with the epsilons the released kernel adds.

Written by Claude Fable 5.1, reasoning effort 80.

`notebooks.sota.DeepSeekV41Flash.single_pass_mhc` writes the combine matrix as an
exponential followed by twenty exact rounds of Sinkhorn normalisation, which equals the
released function only with every epsilon left out. This module writes the order the
released kernel runs. The first round is the exponential, an exact division of each row
by its sum, the addition of the epsilon to every entry, and a division of each column by
its sum plus the epsilon. Nineteen rounds follow, and each divides every row and then
every column by its sum plus the epsilon. The collapse vector gains the same epsilon
after its sigmoid. `released_constants.MHC_EPSILON` is that epsilon.

An `ops.L1Norm` divides by the sum over one axis plus the epsilon it carries, so a row
normalisation and a column normalisation are each one operator at `MHC_EPSILON`, and
the exact rounds of `single_pass_mhc` are the same operator at `ops.NO_EPSILON`. An
`ops.Normalize` divides by the root of the mean of the squares plus the epsilon it
carries and takes the gain and the bias its fields declare, so the normalisation of the
four streams is one operator at `released_constants.NORM_EPSILON` with neither.

    divide_rows_by_sum_plus_epsilon, divide_columns_by_sum_plus_epsilon
                                    the two `ops.L1Norm` operators of a Sinkhorn round
    sinkhorn                        the first round and the nineteen later rounds
    normalise_streams_without_gain  the RMS normalisation of one token's streams
    predict_one_token, COEFFICIENTS the three sets of coefficients of one token, and the
                                    box that computes them once per token
    mhc_sublayer                    one sublayer inside the four-stream residual, given
                                    its body and the coefficients box

`collapse_streams` and `write_streams_back` are imported from `single_pass_mhc`
unchanged. `ARITHMETIC_ROLES` and `ARITHMETIC_REFERENCES` say what each elementwise map
this module names is for, by the text of its name, in the form
`notebooks.sota.DeepSeekV41Flash.operator_explanations` holds its own.

An elementwise map is named with a `fd.DynamicName` built from the whole string.
`fd.DynamicName.from_str` cuts a string at its first underscore, which would put the
closing bracket of `x + \\varepsilon_{\\mathrm{hc}}` inside the subscript.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    axes, hold, l1_norm_over, over, route)
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    doubled_sigmoid, exponential, sigmoid)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    COLLAPSE, N, R, X, m, n, state, x)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, kernel_lines, model_lines)
import notebooks.sota.DeepSeekV41Flash.single_pass_mhc as single_pass_mhc
from notebooks.sota.DeepSeekV41Flash.single_pass_mhc import (
    SINKHORN_ITERATIONS, collapse_streams, write_streams_back)
from notebooks.sota.DeepSeekV41Flash.released_constants import (
    MHC_EPSILON, NORM_EPSILON)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

LATER_SINKHORN_ROUNDS = SINKHORN_ITERATIONS - 1
SINKHORN_COLOUR = '#F4EEFB'
COEFFICIENTS_BOX = 'Coef'

COMBINE_MATRIX = cat.Array(R, (n, N))
STREAMS_OF_ONE_TOKEN = cat.Array(R, (n, m))

ADD_MHC_EPSILON_NAME = 'x + \\varepsilon_{\\mathrm{hc}}'


def divide_rows_by_sum_plus_epsilon() -> cat.Broadcasted:
    '''Every row of the combine matrix divided by its sum plus the epsilon of the
    hyper-connections. The matrix is `(n, N)`, so a row sum runs over position 1.'''
    return l1_norm_over((n, N), 1, epsilon=MHC_EPSILON)


def divide_columns_by_sum_plus_epsilon() -> cat.Broadcasted:
    '''Every column of the combine matrix divided by its sum plus the epsilon of the
    hyper-connections. The matrix is `(n, N)`, so a column sum runs over position 0.'''
    return l1_norm_over((n, N), 0, epsilon=MHC_EPSILON)


def add_mhc_epsilon() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.x + MHC_EPSILON,
                                   name=fd.DynamicName(ADD_MHC_EPSILON_NAME))


def first_sinkhorn_round() -> cat.Block:
    '''The round the released kernel writes as a row softmax with the epsilon added,
    followed by one column normalisation. The division of each row by its sum is exact,
    and the epsilon is added to the entries after that division.'''
    return cat.Block.template(
        over((n, N), exponential())
        @ l1_norm_over((n, N), 1)
        @ over((n, N), add_mhc_epsilon())
        @ divide_columns_by_sum_plus_epsilon(),
        title=text.FIRST_ROUND_TITLE, fill_color=SINKHORN_COLOUR,
        description=text.FIRST_SINKHORN_ROUND_DESCRIPTION,
        references=(kernel_lines(436, 443), kernel_lines(445, 448),
                    inference_config_lines(42)))


def later_sinkhorn_rounds() -> cat.Block:
    return cat.Block.template(
        divide_rows_by_sum_plus_epsilon() @ divide_columns_by_sum_plus_epsilon(),
        title=text.LATER_ROUND_TITLE, fill_color=SINKHORN_COLOUR,
        description=text.LATER_SINKHORN_ROUNDS_DESCRIPTION,
        repetition=LATER_SINKHORN_ROUNDS,
        references=(kernel_lines(450, 458), inference_config_lines(41),
                    inference_config_lines(42)))


def sinkhorn() -> cat.BroadcastedCategory:
    '''One token's combine matrix made nearly doubly stochastic, in the order the
    released kernel runs: the first round, then nineteen rounds of a row and a column
    normalisation. The combine matrix is `(n, N)`, so a row sum runs over position 1
    and a column sum over position 0.'''
    return first_sinkhorn_round() @ later_sinkhorn_rounds()


def normalise_streams_without_gain() -> cat.Broadcasted:
    '''The RMS normalisation of one token's four streams, over every stream and every
    channel together, with the epsilon the released code adds under the root and with
    neither a gain nor a bias. It is
    `notebooks.sota.DeepSeekV41Flash.single_pass_mhc.normalise_streams` with the
    epsilon supplied.'''
    return ops.Normalize.template((n, m), gain=False, bias=False,
                                  epsilon=NORM_EPSILON)


def predict_one_token() -> cat.Block:
    '''One token's collapse vector, output vector and combine matrix, each from a
    linear map of its own that reads the token's four normalised streams. The collapse
    vector gains the epsilon after its sigmoid, as the released kernel writes it.'''
    return cat.Block.template(
        normalise_streams_without_gain()
        @ route((0, 0, 0), (STREAMS_OF_ONE_TOKEN,))
        @ ((ops.Linear.template((n, m), (n,), 'H_0', bias=True) @ sigmoid()
            @ over((n,), add_mhc_epsilon()))
           * (ops.Linear.template((n, m), (n,), 'H_1', bias=True) @ doubled_sigmoid())
           * (ops.Linear.template((n, m), (n, N), 'H_2', bias=True) @ sinkhorn())),
        title=text.COEFFICIENTS_TITLE, fill_color='#E8DFF5',
        description=text.EPSILON_PREDICT_ONE_TOKEN_DESCRIPTION,
        references=(model_lines(948, 955), model_lines(941, 946),
                    kernel_lines(426, 431)))


COEFFICIENTS_BODY = predict_one_token()


def predict_over_all_tokens() -> cat.BroadcastedCategory:
    '''The same prediction written out over every token, which the box is confirmed
    against. It lifts the body the box holds, because `cat.Block.template` mints a
    fresh tag per call and the comparison reads the tags of the Sinkhorn blocks.'''
    return over((x,), COEFFICIENTS_BODY)


COEFFICIENTS = discovering_broadcasts.broadcast_block_over_axes(
    COEFFICIENTS_BODY, (x,), ((0,),), COEFFICIENTS_BOX)
COEFFICIENTS_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    predict_over_all_tokens(), COEFFICIENTS)


class SublayerBodyIsNotOnTheHiddenState(ValueError):
    '''A sublayer body whose domain or codomain is not the hidden state alone.'''


def mhc_sublayer(
    body: cat.BroadcastedCategory,
    coefficients: cat.Broadcasted = COEFFICIENTS,
) -> cat.Block:
    '''One sublayer inside the four-stream residual, with `coefficients` predicting the
    three sets of mixing coefficients. Position 0 is the collapse vector the previous
    sublayer predicted and position 1 is the residual. `body` reads the collapsed and
    normalised hidden state and returns the hidden state, so anything it shares with
    another layer goes through the tape.'''
    if tuple(body.dom()) != (state,) or tuple(body.cod()) != (state,):
        raise SublayerBodyIsNotOnTheHiddenState(
            f'a sublayer body reads and writes the hidden state alone, and this one '
            f'has the domain and codomain {axes(body)}')
    collapse, output, combine_matrix = coefficients.cod()
    return cat.Block.template(
        route((1, 1, 1, 0), (COLLAPSE, X))
        @ (coefficients * hold(X) * hold(X) * hold(COLLAPSE))
        @ route((0, 1, 2, 3, 5, 4), (collapse, output, combine_matrix, X, X, COLLAPSE))
        @ (hold(collapse) * hold(output) * hold(combine_matrix) * hold(X)
           * collapse_streams())
        @ (hold(collapse) * hold(output) * hold(combine_matrix) * hold(X)
           * (ops.Normalize.template() @ body))
        @ route((0, 4, 1, 3, 2), (collapse, output, combine_matrix, X, state))
        @ (hold(collapse) * write_streams_back()),
        title=text.MHC_TITLE, fill_color='#F1F4C1',
        description=text.EPSILON_MHC_SUBLAYER_DESCRIPTION,
        references=(model_lines(957, 966), model_lines(981, 994)))


ARITHMETIC_ROLES: dict[str, str] = {
    ADD_MHC_EPSILON_NAME: (
        text.MHC_EPSILON_ROLE),
}

REPLACED_ARITHMETIC_ROLES: tuple[str, ...] = ()
'''The rows of the base model's table that this module's mechanisms replace. Both
normalisations here are operators of their own, which carry their epsilon and open
their own inspection box, so this module leaves no row of the base table unused.'''


ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    ADD_MHC_EPSILON_NAME: (kernel_lines(427), kernel_lines(443)),
}
