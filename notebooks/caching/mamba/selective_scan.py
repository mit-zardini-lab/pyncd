# Claude Opus 5.5 (1M context), effort 40.
'''The selective scan of Mamba as a loop over the tokens.

The reference runs the recurrence `h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t u_t` and reads
`y_t = C_t · h_t` at every token, per `reference_links.RECURRENCE`. The loop here is a
`cat.Block` repeated once per token, whose counter `i_x` is the name its tag carries.

The state `h` is a loop variable. A `Para.StreamGrab` of the slot `h` reads the state
the iteration starts from, and a `Para.StreamDrop` of the slot writes the state the
next iteration starts from, per `obsidian/07-para/Para Category.md`. The slots `h` and
`Y` are inner, so the scan lifted over a batch axis carries one state for every
sequence, per `obsidian/07-para/Outer and Inner Tape Slots.md`. The output `Y`
over every token is a second loop variable. Each iteration writes `y_t` at position
`i_x` of an array over the tokens, which is the covariant reading of the row that
selects position `i_x` and holds the universal unit at every other position, and adds
the write to `Y`. The two loop variables start from arrays of zeros that a `Para.Drop`
puts on each slot before the loop, and the initializers, the drops and the loop are
grouped in one block.

The loop reads the decays, the writes and the read-out map of token `i_x` through a
view whose row has no domain axis and the shift `i_x`, times the identity on the other
axes. The three arrays enter the loop on wires over every token, and every iteration
reads its own token from them. `obsidian/08-caching/Carrying the State of a Scan
Between Passes.md` states why the per-token inputs reach the loop this way.
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import para.data_structure.Para as Para

from notebooks.classic.shared_mechanisms import contract
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, route

from notebooks.caching.mamba import reference_links
from notebooks.caching.mamba.declared_axes import R, d, n

SCAN_COLOUR = '#DBDFEF'
LOOP_COLOUR = '#F1F4C1'
STATE_SLOT = fd.DynamicName('h').capture(Para.TapeSlot())
OUTPUT_SLOT = fd.DynamicName('Y').capture(Para.TapeSlot())
RECURRENCE_FORMULA = (
    'h_{t} = \\bar{A}_{t} h_{t-1} + \\bar{B}_{t} u_{t}, \\quad '
    'y_{t} = \\sum_{i_{n} \\in n} C_{t}[i_{n}]\\, h_{t}[i_{n}]')


def row_at(tokens: cat.Axis, position: nm.Numeric) -> sc.StrideMorphism:
    '''The row with no domain axis that selects `position` of `tokens`.'''
    return sc.StrideMorphism(
        _dom=(), _cod_stride_shift=((tokens, (), position),),
        name=fd.DynamicName(position.to_latex()))


def read_at(tokens: cat.Axis, position: nm.Numeric,
            rest: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    '''An array `[tokens, *rest]` read at `position` of `tokens`, which returns
    `[*rest]`.'''
    return ops.View.template(
        reindexing=(row_at(tokens, position), cat.ProdObject(rest).identity()),
        name=position.to_latex())


def write_at(tokens: cat.Axis, position: nm.Numeric,
             rest: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    '''An array `[*rest]` written at `position` of `tokens`, which returns
    `[tokens, *rest]` holding the universal unit at every other position of
    `tokens`. It is the covariant reading of the row `read_at` reads.'''
    tiled = (cat.WeaveMode.TILED,) * len(rest)
    return cat.Broadcasted(
        operator=aops.CovariantView(reindexing=row_at(tokens, position),
                                    name=fd.DynamicName(position.to_latex())),
        input_weaves=(cat.Weave(R, tiled),),
        output_weaves=(cat.Weave(R, (tokens, *tiled)),),
        reindexings=(cat.ProdObject(rest).identity(),))


def zeros_over(shape: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    '''The array of zeros over `shape`, which is where a sum starts.'''
    return cat.Broadcasted(
        operator=ops.ConstantOp(value=nm.Integer(0)),
        input_weaves=(), output_weaves=(cat.Weave(R, (cat.WeaveMode.TILED,) * len(shape)),),
        reindexings=(), backup_degree=cat.ProdObject(shape))


def add_over(shape: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    return einops_simplification.einsum((shape, shape), shape, R, ops.AdditionOp())


def next_state_and_output(tokens: cat.Axis, counter: nm.Numeric) -> cat.BroadcastedCategory:
    '''Token `counter` of the decays, the writes and the read-out map, the state the
    iteration starts from and the output written so far, to the next state and to the
    output with `y_t` written at `counter`.'''
    state = cat.Array(R, (d, n))
    output = cat.Array(R, (tokens, d))
    read_out = cat.Array(R, (n,))
    decayed = contract(((d, n), (d, n)), (d, n))
    read_the_state = contract(((d, n), (n,)), (d,))
    return (
        (read_at(tokens, counter, (d, n)) * read_at(tokens, counter, (d, n))
         * read_at(tokens, counter, (n,))
         * Para.StreamGrab(tape=STATE_SLOT, size=state)
         * Para.StreamGrab(tape=OUTPUT_SLOT, size=output))
        @ route((0, 3, 1, 2, 4), (state, state, read_out, state, output))
        @ (decayed * hold(state) * hold(read_out) * hold(output))
        @ (add_over((d, n)) * hold(read_out) * hold(output))
        @ route((0, 0, 1, 2), (state, read_out, output))
        @ (hold(state) * read_the_state * hold(output))
        @ (hold(state) * write_at(tokens, counter, (d,)) * hold(output))
        @ (hold(state) * route((1, 0), (output, output)))
        @ (hold(state) * add_over((tokens, d))))


def one_step_of_the_recurrence(tokens: cat.Axis, counter: nm.Numeric) -> cat.BroadcastedCategory:
    '''The body of the loop of the model: the next state and the next output dropped
    for the next iteration, and the output handed out.'''
    state = cat.Array(R, (d, n))
    output = cat.Array(R, (tokens, d))
    return (next_state_and_output(tokens, counter)
            @ (Para.StreamDrop(tape=STATE_SLOT, size=state) * route((0, 0), (output,)))
            @ (Para.StreamDrop(tape=OUTPUT_SLOT, size=output) * hold(output)))


def one_step_handing_out_the_state(
        tokens: cat.Axis, counter: nm.Numeric) -> cat.BroadcastedCategory:
    '''The body of a loop whose final state is read after it: the next state and the
    next output dropped for the next iteration, and both handed out, the output
    first.'''
    state = cat.Array(R, (d, n))
    output = cat.Array(R, (tokens, d))
    return (next_state_and_output(tokens, counter)
            @ (route((0, 0), (state,)) * route((0, 0), (output,)))
            @ (Para.StreamDrop(tape=STATE_SLOT, size=state) * hold(state)
               * Para.StreamDrop(tape=OUTPUT_SLOT, size=output) * hold(output))
            @ route((1, 0), (state, output)))


def repeated_over_the_tokens(
        body: cat.BroadcastedCategory, tokens: cat.Axis, counter_name: str) -> cat.Block:
    '''`body` repeated once per token of `tokens`, with the counter named
    `counter_name`.'''
    return cat.Block.template(
        body, title='\\text{One Token}', fill_color=LOOP_COLOUR,
        repetition=tokens.local_size(), index_name=counter_name,
        formula=RECURRENCE_FORMULA, references=(reference_links.RECURRENCE,))


def start_the_output(tokens: cat.Axis) -> cat.BroadcastedCategory:
    '''The zero output over `tokens`, dropped on its slot.'''
    return (zeros_over((tokens, d))
            @ Para.Drop(tape=OUTPUT_SLOT, size=cat.Array(R, (tokens, d))))


def start_the_state_at_zero() -> cat.BroadcastedCategory:
    '''The zero state, dropped on its slot, which is where the reference starts the
    scan.'''
    return zeros_over((d, n)) @ Para.Drop(tape=STATE_SLOT, size=cat.Array(R, (d, n)))


def scan_inputs(tokens: cat.Axis) -> tuple[cat.Array, ...]:
    '''The decays, the writes and the read-out maps of every token.'''
    return (cat.Array(R, (tokens, d, n)), cat.Array(R, (tokens, d, n)),
            cat.Array(R, (tokens, n)))


def selective_scan(tokens: cat.Axis, counter_name: str) -> cat.Block:
    '''The scan over `tokens`: the state and the output started at zero, then the
    loop.'''
    counter = nm.FreeNumeric.named(counter_name)
    return cat.Block.template(
        (start_the_state_at_zero() * start_the_output(tokens)
         * cat.ProdObject(scan_inputs(tokens)).identity())
        @ repeated_over_the_tokens(
            one_step_of_the_recurrence(tokens, counter), tokens, counter_name),
        title='\\text{Selective Scan}', fill_color=SCAN_COLOUR,
        formula=RECURRENCE_FORMULA,
        references=(reference_links.RECURRENCE,))
