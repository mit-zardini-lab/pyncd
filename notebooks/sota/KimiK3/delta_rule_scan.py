# Claude Opus 5.5 (1M context), effort 40.
'''The gated delta rule of Kimi Delta Attention as a loop over the tokens.

`chunk_kda` of `fla` computes the recurrence in blocks of 64 tokens, and
`naive_recurrent_kda` in `fla/ops/kda/naive.py` states the same recurrence one token at
a time. Every head `j` holds a state `S`, a matrix with one row per key channel `d` and
one column per value channel `z`. At token `t` the state is decayed row by row, the
value the decayed state predicts for the key is read out, the difference between the
value and the prediction is written back along the key with the strength `beta`, and
the query reads the new state:

    S'   = alpha_t[d] S[d, z]
    S_t  = S' + beta_t k_t[d] (v_t[z] - sum_d k_t[d] S'[d, z])
    o_t  = sum_d q_t[d] S_t[d, z]

The write is the delta rule of DeltaNet, which moves the state towards storing the
value at the key rather than adding the value on top of what the state holds. The decay
`alpha_t` has one number per key channel, which is what Kimi Delta Attention adds to the
gated delta rule of Gated DeltaNet, whose decay has one number per head.

The loop is a `cat.Block` repeated once per token, whose counter `i_x` is the name its
tag carries, as `notebooks/caching/mamba/selective_scan.py` writes the selective scan of
Mamba. The state is a loop variable. A `Para.StreamGrab` of the slot `S` reads the state
the iteration starts from, and a `Para.StreamDrop` writes the state the next iteration
starts from. The output over every token is a second loop variable. Each iteration
writes `o_t` at position `i_x` of an array over the tokens, which holds the universal
unit at every other position, and adds the write to the output. The state starts at
zero, as the reference's `initial_state` of `None` does, and the output starts at zero.

The queries, the keys, the values, the decays and the strengths of every token are
computed before the loop by operators broadcast over the tokens. They enter the loop on
wires, and iteration `i_x` reads its token through a view whose row selects position
`i_x`. `obsidian/08-caching/Carrying the State of a Scan Between Passes.md` states why
the per-token inputs reach the loop this way.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import para.data_structure.Para as Para

from notebooks.caching.mamba.selective_scan import (
    add_over, read_at, row_at, write_at, zeros_over)
from notebooks.classic.shared_mechanisms import contract
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, route
from notebooks.sota.KimiK3.declared_axes import (
    DELTA_STATE, R, TOKEN_COUNTER, TOKEN_COUNTER_NAME, d, j, x, z)
from notebooks.sota.KimiK3.reference_links import (
    KERNEL_ARGUMENTS, RECURRENCE, modeling_lines)
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

SCAN_COLOUR = '#DBDFEF'
SCAN_BOX = 'Scan'
LOOP_COLOUR = '#F1F4C1'
STATE_SLOT = fd.DynamicName('S').capture(Para.TapeSlot())
OUTPUT_SLOT = fd.DynamicName('O').capture(Para.TapeSlot())
NEGATE_NAME = '-x'
RECURRENCE_FORMULA = (
    'S_{t} = \\alpha_{t} S_{t-1} + \\beta_{t} k_{t} \\big(v_{t} - k_{t}^{\\top} '
    '\\alpha_{t} S_{t-1}\\big)^{\\top}, \\quad o_{t} = S_{t}^{\\top} q_{t}')
SCAN_REFERENCES = (RECURRENCE, KERNEL_ARGUMENTS, modeling_lines(609, 645))

KEY_SHAPE = (j, d)
VALUE_SHAPE = (j, z)
STATE_SHAPE = (j, d, z)
OUTPUT = cat.Array(R, (x, j, z))
KEY_ROWS = cat.Array(R, KEY_SHAPE)
VALUE_ROW = cat.Array(R, VALUE_SHAPE)


def negate() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.Integer(-1) * nm.x, name=NEGATE_NAME)


def read_this_token() -> cat.BroadcastedCategory:
    '''Token `i_x` of the queries, the keys, the values, the decays and the strengths,
    beside the state and the output the iteration starts from.'''
    return (read_at(x, TOKEN_COUNTER, KEY_SHAPE) * read_at(x, TOKEN_COUNTER, KEY_SHAPE)
            * read_at(x, TOKEN_COUNTER, VALUE_SHAPE)
            * read_at(x, TOKEN_COUNTER, KEY_SHAPE) * read_at(x, TOKEN_COUNTER, (j,))
            * Para.StreamGrab(tape=STATE_SLOT, size=DELTA_STATE)
            * Para.StreamGrab(tape=OUTPUT_SLOT, size=OUTPUT))


def decay_the_state() -> cat.Broadcasted:
    '''Every row of the state multiplied by the decay of its key channel.'''
    return contract((KEY_SHAPE, STATE_SHAPE), STATE_SHAPE)


def predict_the_value() -> cat.Broadcasted:
    '''The value the decayed state holds at the key.'''
    return contract((KEY_SHAPE, STATE_SHAPE), VALUE_SHAPE)


def correction_along_the_key() -> cat.Broadcasted:
    '''The strength times the key times the difference between the value and the
    prediction, which is the outer product the delta rule adds to the state.'''
    return contract(((j,), KEY_SHAPE, VALUE_SHAPE), STATE_SHAPE)


def read_the_state() -> cat.Broadcasted:
    '''The state read by the query.'''
    return contract((KEY_SHAPE, STATE_SHAPE), VALUE_SHAPE)


def next_state_and_output() -> cat.BroadcastedCategory:
    '''Token `i_x`, the state and the output so far to the next state and to the
    output with `o_t` written at `i_x`.

    The operands after `read_this_token` stand in the order query, key, value, decay,
    strength, state, output.'''
    strengths = cat.Array(R, (j,))
    state = DELTA_STATE
    return (
        read_this_token()
        @ route((0, 1, 1, 2, 4, 3, 5, 6),
                (KEY_ROWS, KEY_ROWS, VALUE_ROW, KEY_ROWS, strengths, state, OUTPUT))
        @ (hold(KEY_ROWS) * hold(KEY_ROWS) * hold(KEY_ROWS) * hold(VALUE_ROW)
           * hold(strengths) * decay_the_state() * hold(OUTPUT))
        @ route((0, 1, 3, 2, 5, 4, 5, 6),
                (KEY_ROWS, KEY_ROWS, KEY_ROWS, VALUE_ROW, strengths, state, OUTPUT))
        @ (hold(KEY_ROWS) * hold(KEY_ROWS) * hold(VALUE_ROW)
           * (predict_the_value() @ negate()) * hold(strengths) * hold(state)
           * hold(OUTPUT))
        @ (hold(KEY_ROWS) * hold(KEY_ROWS) * add_over(VALUE_SHAPE) * hold(strengths)
           * hold(state) * hold(OUTPUT))
        @ route((0, 3, 1, 2, 4, 5), (KEY_ROWS, KEY_ROWS, VALUE_ROW, strengths, state,
                                     OUTPUT))
        @ (hold(KEY_ROWS) * correction_along_the_key() * hold(state) * hold(OUTPUT))
        @ (hold(KEY_ROWS) * route((1, 0), (state, state)) * hold(OUTPUT))
        @ (hold(KEY_ROWS) * add_over(STATE_SHAPE) * hold(OUTPUT))
        @ route((1, 0, 1, 2), (KEY_ROWS, state, OUTPUT))
        @ (hold(state) * read_the_state() * hold(OUTPUT))
        @ (hold(state) * write_at(x, TOKEN_COUNTER, VALUE_SHAPE) * hold(OUTPUT))
        @ (hold(state) * route((1, 0), (OUTPUT, OUTPUT)))
        @ (hold(state) * add_over((x, *VALUE_SHAPE))))


def one_token_of_the_scan() -> cat.BroadcastedCategory:
    '''The body of the loop: the next state and the next output dropped for the next
    iteration, and the output handed out.'''
    return (next_state_and_output()
            @ (Para.StreamDrop(tape=STATE_SLOT, size=DELTA_STATE)
               * route((0, 0), (OUTPUT,)))
            @ (Para.StreamDrop(tape=OUTPUT_SLOT, size=OUTPUT) * hold(OUTPUT)))


def repeated_over_the_tokens(body: cat.BroadcastedCategory) -> cat.Block:
    return cat.Block.template(
        body, title=text.SCAN_STEP_TITLE, fill_color=LOOP_COLOUR,
        repetition=x.local_size(), index_name=TOKEN_COUNTER_NAME,
        formula=RECURRENCE_FORMULA, description=text.SCAN_STEP_DESCRIPTION,
        references=SCAN_REFERENCES)


def start_the_state_and_the_output() -> cat.BroadcastedCategory:
    '''The zero state and the zero output, each dropped on its slot.'''
    return ((zeros_over(STATE_SHAPE) @ Para.Drop(tape=STATE_SLOT, size=DELTA_STATE))
            * (zeros_over((x, *VALUE_SHAPE)) @ Para.Drop(tape=OUTPUT_SLOT, size=OUTPUT)))


SCAN_INPUTS: tuple[cat.Array, ...] = (
    cat.Array(R, (x, *KEY_SHAPE)), cat.Array(R, (x, *KEY_SHAPE)),
    cat.Array(R, (x, *VALUE_SHAPE)), cat.Array(R, (x, *KEY_SHAPE)),
    cat.Array(R, (x, j)))
'''The queries, the keys, the values, the decays and the strengths of every token, in
the order the loop reads them.'''


def delta_rule_scan() -> cat.Block:
    '''The scan over the tokens: the state and the output started at zero, then the
    loop.'''
    return cat.Block.template(
        (start_the_state_and_the_output() * cat.ProdObject(SCAN_INPUTS).identity())
        @ repeated_over_the_tokens(one_token_of_the_scan()),
        title=text.SCAN_TITLE, fill_color=SCAN_COLOUR, formula=RECURRENCE_FORMULA,
        description=text.SCAN_DESCRIPTION, references=SCAN_REFERENCES)


DELTA_RULE_SCAN = boxed(delta_rule_scan(), SCAN_BOX)
