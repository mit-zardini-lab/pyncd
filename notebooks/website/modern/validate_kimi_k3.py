# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim `notebooks/website/modern/KimiK3.ipynb` makes about Kimi K3.

    python notebooks/website/modern/validate_kimi_k3.py

The notebook is public facing and holds the prose and the figures alone. The claims live
here, one `check_` function per claim, in the order the notebook makes them: the ends of
the model and its sizes, the delta attention and its scan, the latent attention, the
attention residuals, the feed-forward maps, the layer plan, the CausalSlide, and the
page. Two claims are checked on numbers. `check_the_delta_attention_computes_the_reference`
evaluates the delta attention at small sizes and compares it with `KimiDeltaAttention`
run in PyTorch from the reference loop of `fla`, and
`check_the_mix_computes_the_reference` compares one mix of the attention residuals with
`_apply_attn_res`. The script prints one line per check and the time the run took, and
it exits non-zero on a failure.
'''
from __future__ import annotations

import pathlib
import sys
import time
from collections.abc import Callable, Iterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
import torch_compile.torch_compile as torch_compile  # noqa: E402

import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.sota_figures as figures  # noqa: E402
import notebooks.sota.KimiK3.assemble_page_variants as assemble_page_variants  # noqa: E402
import notebooks.sota.KimiK3.attention_residuals as attention_residuals  # noqa: E402
import notebooks.sota.KimiK3.declared_axes as declared_axes  # noqa: E402
import notebooks.sota.KimiK3.delta_attention as delta_attention  # noqa: E402
import notebooks.sota.KimiK3.delta_rule_scan as delta_rule_scan  # noqa: E402
import notebooks.sota.KimiK3.evaluate_numerically as evaluate_numerically  # noqa: E402
import notebooks.sota.KimiK3.feed_forward as feed_forward  # noqa: E402
import notebooks.sota.KimiK3.latent_attention as latent_attention  # noqa: E402
import notebooks.sota.KimiK3.latent_mixture_of_experts as latent_mixture_of_experts  # noqa: E402
import notebooks.sota.KimiK3.layer_stack as layer_stack  # noqa: E402
import notebooks.sota.KimiK3.released_constants as released_constants  # noqa: E402
import notebooks.sota.KimiK3.run_the_reference as run_the_reference  # noqa: E402
import notebooks.sota.KimiK3.slide_causal_reads as slide_causal_reads  # noqa: E402
import notebooks.sota.KimiK3.whole_model as whole_model  # noqa: E402
from notebooks.caching.mamba.evaluate_numerically import LinearValues  # noqa: E402
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text  # noqa: E402

MODEL: cat.Morphism = whole_model.kimi_k3
DECODE: cat.Morphism = slide_causal_reads.kimi_k3_slid
ASSIGNED_SIZES: dict[str, int] = whole_model.released_assigned_sizes()
PAGE = assemble_page_variants.page_settings(figures.DiagramMode.HTML)
VARIANTS = assemble_page_variants.page_variants(PAGE)

RELEASED_AXIS_WIDTHS: dict[str, int] = {
    'm': 7168, 'h': 96, 'q': 1536, 'c': 512, 'n': 128, 'p': 64, 'u': 128, 'j': 96,
    'd': 128, 'z': 128, 'w': 4, 'e': 896, 'k': 16, 'l': 3584, 'f': 3072, 't': 6144,
    'g': 33792, 'b': 9, 'v': 163840}
'''The widths of `config.json` at the pinned commit: `hidden_size`,
`num_attention_heads`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
`qk_rope_head_dim`, `v_head_dim`, `linear_attn_config.num_heads`, its `head_dim` twice
and its `short_conv_kernel_size`, `num_experts`, `num_experts_per_token`,
`routed_expert_hidden_size`, `moe_intermediate_size`, `moe_intermediate_size` times
`num_shared_experts`, `intermediate_size`, the nine entries of the attention residuals,
and `vocab_size`.'''

RELEASED_CONSTANT_VALUES: dict[nm.FreeNumeric, str] = {
    released_constants.NORM_EPSILON: '10^{-5}',
    released_constants.LATENT_NORM_EPSILON: '10^{-6}',
    released_constants.L2_NORM_EPSILON: '10^{-6}',
    released_constants.GATE_LOWER_BOUND: '-5',
    released_constants.SITU_GATE_BOUND: '4',
    released_constants.SITU_UP_BOUND: '25',
    released_constants.ROUTER_EPSILON: '10^{-20}'}
'''`rms_norm_eps` of `config.json`, the default epsilon of `KimiRMSNorm`, the default
epsilon of `l2norm_fwd` in `fla`, `gate_lower_bound`, `activation_situ_beta`,
`activation_situ_linear_beta`, and the number the router adds to the sum of the
gates.'''

KDA_LAYERS_OF_THE_CONFIGURATION: tuple[int, ...] = (
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30,
    31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58,
    59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86,
    87, 89, 90, 91)
'''`linear_attn_config.kda_layers` of `config.json`, lines 95 to 165, which counts the
layers from one.'''
LAYERS = 93
ATTENTION_RESIDUAL_BLOCK = 12
FIRST_DENSE_LAYERS = 1

ATTENTION_BOXES = (delta_attention.ATTENTION_BOX, latent_attention.ATTENTION_BOX)
FEED_FORWARD_BOXES = (feed_forward.DENSE_BOX, latent_mixture_of_experts.MIXTURE_BOX)
SUBLAYER_BOXES = (*ATTENTION_BOXES, *FEED_FORWARD_BOXES)


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def box_named(name: str, term: cat.Morphism) -> cat.Broadcasted:
    return whole_model.part_named(name, term)


def body_of(name: str, term: cat.Morphism = MODEL) -> cat.Morphism:
    return box_named(name, term).operator.block


def title_of(block: cat.Block) -> str | None:
    aesthetics = block.block_tag.aesthetics
    return aesthetics.title if aesthetics is not None else None


def operations_outside_boxes(term: cat.Morphism) -> Iterator[cat.Broadcasted]:
    '''Every operation of `term` that stands in no box, in the order of the term.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from operations_outside_boxes(part)
        case cat.Block(body=body):
            yield from operations_outside_boxes(body)
        case cat.Broadcasted():
            yield term


def names_of_boxes(nodes: tuple[cat.Broadcasted, ...]) -> tuple[str, ...]:
    return tuple(node.operator.name.to_bodies() for node in nodes
                 if isinstance(node.operator, ops.BlockOperator)
                 and node.operator.name is not None)


def counter_of(block: cat.Block) -> nm.FreeNumeric | None:
    name = block.block_tag.uid._name
    return nm.FreeNumeric.named(name) if name is not None else None


def sublayers_in_order(term: cat.Morphism, bindings: dict[nm.Numeric, int]
                       ) -> Iterator[tuple[str, int, str]]:
    '''Every sublayer of `term` in the order it runs, with every repeated block written
    out: the name of its box, the entry it adds its output at, and the name of the box
    or view that gives its input.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from sublayers_in_order(part, bindings)
        case cat.Block(body=body, block_tag=block_tag):
            if title_of(term) == text.RESIDUAL_TITLE:
                yield read_sublayer(body, bindings)
                return
            counter = counter_of(term)
            for iteration in range(block_tag.repetition._value):
                inner = dict(bindings)
                if counter is not None:
                    inner[counter] = iteration
                yield from sublayers_in_order(body, inner)


def read_sublayer(body: cat.Morphism, bindings: dict[nm.Numeric, int]
                  ) -> tuple[str, int, str]:
    nodes = tuple(operations_outside_boxes(body))
    sublayer, = (name for name in names_of_boxes(nodes) if name in SUBLAYER_BOXES)
    write, = (node for node in nodes if isinstance(node.operator, aops.CovariantView))
    (_, _, position), = write.operator.reindexing._cod_stride_shift
    reading = ('mix' if attention_residuals.MIX_BOX in names_of_boxes(nodes)
               else next(node.operator.name.to_bodies() for node in nodes
                         if isinstance(node.operator, ops.View)))
    return sublayer, nm.evaluate_integer(position, bindings), reading


def layer_plan(term: cat.Morphism) -> list[tuple[str, str, int, str]]:
    '''One entry per layer: the attention box, the feed-forward box, the entry both add
    their outputs at, and the input of the attention.'''
    sublayers = list(sublayers_in_order(term, {}))
    return [(attention[0], feed_forward_part[0], attention[1], attention[2])
            for attention, feed_forward_part in zip(sublayers[::2], sublayers[1::2])]


def released_plan() -> list[tuple[str, str, int, str]]:
    '''The plan the reference runs: layer `i` runs the delta attention when `i + 1` is
    listed in `kda_layers`, the dense MLP when `i < first_k_dense_replace`, adds its
    outputs at entry `1 + i // 12`, and reads the embedding in its attention alone
    when it is layer 0.'''
    return [(delta_attention.ATTENTION_BOX if layer + 1 in KDA_LAYERS_OF_THE_CONFIGURATION
             else latent_attention.ATTENTION_BOX,
             feed_forward.DENSE_BOX if layer < FIRST_DENSE_LAYERS
             else latent_mixture_of_experts.MIXTURE_BOX,
             1 + layer // ATTENTION_RESIDUAL_BLOCK,
             attention_residuals.EMBEDDING_ENTRY.to_latex() if layer == 0
             else 'mix')
            for layer in range(LAYERS)]


# ==========================================================================
# The model, its sizes and its constants.
# ==========================================================================
def check_the_model_reads_identifiers_and_returns_logits() -> None:
    '''The model reads the token identifiers and returns one logit per entry of the
    vocabulary for every token.'''
    identifiers, = MODEL.dom()
    logits, = MODEL.cod()
    require(isinstance(identifiers.datatype, cat.Natural)
            and tuple(identifiers.shape()) == (declared_axes.x,),
            f'the model reads {identifiers}')
    require(isinstance(logits.datatype, cat.Reals) and len(logits.shape()) == 2
            and logits.shape()[0] == declared_axes.x,
            f'the model returns {logits}')


def check_the_axes_have_the_released_sizes() -> None:
    '''Every axis of the table takes the size of the configuration, a key and a query
    of the latent attention are 192 channels wide, and the distance axis is as long as
    the prompt.'''
    differing = {name: ASSIGNED_SIZES.get(name) for name, width
                 in RELEASED_AXIS_WIDTHS.items() if ASSIGNED_SIZES.get(name) != width}
    require(not differing, f'the configuration sizes {differing}')
    width = nm.evaluate_integer(
        declared_axes.a.local_size(),
        {symbol: ASSIGNED_SIZES[symbol.uid._name.to_bodies()]
         for symbol in (declared_axes.n.local_size(), declared_axes.p.local_size())})
    require(width == 192, f'a key of the latent attention is {width} wide')
    require(declared_axes.r.local_size() == declared_axes.x.local_size(),
            'the distance axis is not as long as the prompt')


def check_the_constants_have_the_released_values() -> None:
    '''The table of constants gives each constant the value of the reference.'''
    table = {constant.symbol: constant.released_value
             for constant in released_constants.RELEASED_CONSTANTS}
    require(table == RELEASED_CONSTANT_VALUES, f'the constants are {table.values()}')


# ==========================================================================
# The delta attention.
# ==========================================================================
def check_the_convolution_reads_four_taps_back() -> None:
    '''Tap `i_w` of token `i_x` reads token `i_x - i_w`, and the tap axis holds a value
    where `i_x - i_w >= 0`.'''
    (row_axis, strides, shift), = delta_attention.TAPS_READ._cod_stride_shift
    require(row_axis == declared_axes.x and strides == (nm.Integer(1), nm.Integer(-1))
            and shift == nm.Integer(0), 'the taps read another token')
    require(isinstance(delta_attention.taps, AffineGuards.AffineSparseAxis)
            and delta_attention.taps.empty_end() is AffineGuards.EmptyEnd.LAST,
            'the tap axis marks no empty taps before the first token')
    require(ASSIGNED_SIZES['w'] == 4, f"the convolution has {ASSIGNED_SIZES['w']} taps")


def check_the_scan_is_a_loop_over_the_tokens() -> None:
    '''The scan is a block repeated once per token with the counter `i_x`, whose
    iteration grabs the state of every head, `[j, d, z]`, and the output so far, and
    drops both for the next iteration.'''
    loops = [block for block in tutil.type_search(cat.Block, delta_rule_scan.delta_rule_scan())
             if block.block_tag.repetition != nm.Integer(1)]
    require(len(loops) == 1, f'the scan holds {len(loops)} loops')
    loop, = loops
    require(loop.block_tag.repetition == declared_axes.x.local_size()
            and counter_of(loop) == declared_axes.TOKEN_COUNTER,
            'the loop runs otherwise than once per token')
    grabs = {grab.tape.uid._name.to_bodies(): tuple(grab.size.shape())
             for grab in tutil.type_search(Para.StreamGrab, loop)}
    drops = {drop.tape.uid._name.to_bodies() for drop in tutil.type_search(
        Para.StreamDrop, loop)}
    require(grabs == {'S': (declared_axes.j, declared_axes.d, declared_axes.z),
                      'O': (declared_axes.x, declared_axes.j, declared_axes.z)}
            and drops == set(grabs), f'the loop grabs {grabs} and drops {drops}')


def delta_attention_on_numbers(term: cat.Morphism) -> float:
    '''The largest difference between `term`, evaluated on numbers, and
    `KimiDeltaAttention` run in PyTorch, over seven tokens at small sizes.'''
    generator = torch.Generator().manual_seed(0)
    parameters = run_the_reference.random_delta_attention_parameters(
        hidden=5, heads=2, head_width=3, taps=4, generator=generator)
    hidden = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    weights, gains = run_the_reference.delta_weights_of_the_expression(parameters)
    got, = evaluate_numerically.evaluate(
        term, (hidden,), weights, gains,
        {'x': 7, 'm': 5, 'j': 2, 'd': 3, 'z': 3, 'w': 4})
    expected = run_the_reference.kimi_delta_attention(hidden, parameters)
    return (got - expected).abs().max().item()


def check_the_delta_attention_computes_the_reference() -> None:
    '''The delta attention as built and in the CausalSlide both compute
    `KimiDeltaAttention` of the reference, with the gate and the recurrence of the
    reference loop of `fla`, to within `1e-12` in `torch.float64`.'''
    for form, term in (('as built', delta_attention.DELTA_ATTENTION.operator.block),
                       ('in the CausalSlide', body_of(delta_attention.ATTENTION_BOX,
                                                      DECODE))):
        difference = delta_attention_on_numbers(term)
        require(difference < 1e-12, f'the delta attention {form} differs by {difference}')


# ==========================================================================
# The latent attention.
# ==========================================================================
def check_the_latent_attention_turns_no_channel() -> None:
    '''No channel of a query or a key of the latent attention is turned by a rotary
    embedding, and the scores are scaled by one over the square root of 192.'''
    body = body_of(latent_attention.ATTENTION_BOX)
    turns = [node for node in tutil.type_search(cat.Broadcasted, body)
             if isinstance(node.operator, (dst.Rotary, dst.PairsAsComplex))]
    require(not turns, f'the latent attention holds {len(turns)} rotations')
    scales = [node for node in tutil.type_search(cat.Broadcasted, body)
              if isinstance(node.operator, ops.Arithmetic)
              and node.operator.name.to_bodies() == latent_attention.SCORE_SCALE_NAME]
    require(len(scales) == 1, f'the core holds {len(scales)} scales')


def check_the_key_joins_a_latent_part_and_a_shared_part() -> None:
    '''A key of a head joins 128 channels expanded from the latent and the 64 channels
    shared by every head onto the 192 channels of `a`.'''
    join, = (node for node in tutil.type_search(
        cat.Broadcasted, body_of(latent_attention.ATTENTION_BOX))
        if isinstance(node.operator, aops.ConcatenateAxes))
    require(tuple(join.operator.parts()) == (declared_axes.n, declared_axes.p)
            and join.operator.concatenated_axis() == declared_axes.a,
            'the key is joined otherwise')


def check_the_boxes_are_computed_once_per_token_or_head() -> None:
    '''The dense MLP and the mixture are computed once per token, and the core of the
    latent attention once per query and head, each confirmed against its body written
    over every token.'''
    for name, confirmation in (
            ('dense MLP', feed_forward.DENSE_MLP_CONFIRMATION),
            ('mixture', latent_mixture_of_experts.MIXTURE_CONFIRMATION),
            ('core', latent_attention.CORE_CONFIRMATION)):
        require(confirmation.named_difference is None,
                f'the {name} differs from its body: {confirmation.named_difference}')


# ==========================================================================
# The attention residuals.
# ==========================================================================
def check_the_entries_have_nine_positions() -> None:
    '''The attention residuals have nine entries: the embedding and the sums of eight
    blocks of at most twelve layers.'''
    blocks = -(-LAYERS // ATTENTION_RESIDUAL_BLOCK)
    require(ASSIGNED_SIZES['b'] == 1 + blocks == 9,
            f"the entries are {ASSIGNED_SIZES['b']} for {blocks} blocks")


def check_the_mix_computes_the_reference() -> None:
    '''One mix of the attention residuals computes `_apply_attn_res` of the reference
    when every entry holds a value, to within `1e-12` in `torch.float64`.'''
    generator = torch.Generator().manual_seed(1)
    tokens, entries, width = 3, 4, 5
    values = torch.randn(tokens, entries, width, generator=generator, dtype=torch.float64)
    projection = torch.randn(1, width, generator=generator, dtype=torch.float64)
    gain = 1 + torch.randn(width, generator=generator, dtype=torch.float64) / 4
    expected = run_the_reference.apply_attn_res(
        values[:, -1], values[:, :-1], projection, gain, run_the_reference.NORM_EPSILON)
    got, = evaluate_numerically.evaluate(
        attention_residuals.mix_of(attention_residuals.ATTENTION_INPUT_WEIGHT),
        (values,), {attention_residuals.ATTENTION_INPUT_WEIGHT: LinearValues(projection[0])},
        {'RMSNorm': gain}, {'x': tokens, 'b': entries, 'm': width})
    difference = (got - expected).abs().max().item()
    require(difference < 1e-12, f'the mix differs by {difference}')


def check_the_embedding_starts_the_entries() -> None:
    '''The embedding is written at entry 0, and the output head reads a last mix of the
    entries.'''
    embedding = whole_model.embed()
    writes = [node for node in tutil.type_search(cat.Broadcasted, embedding)
              if isinstance(node.operator, aops.CovariantView)]
    require(len(writes) == 1, f'the embedding holds {len(writes)} writes')
    (_, _, position), = writes[0].operator.reindexing._cod_stride_shift
    require(position == nm.Integer(0), f'the embedding is written at {position}')
    output_boxes = names_of_boxes(tuple(operations_outside_boxes(whole_model.output_logits())))
    require(output_boxes == (attention_residuals.MIX_BOX,),
            f'the output head reads {output_boxes}')


# ==========================================================================
# The feed-forward maps.
# ==========================================================================
def check_the_mixture_keeps_sixteen_of_896_experts_on_the_latent() -> None:
    '''The router keeps 16 of the 896 experts, the routed experts read and write the
    latent of 3584 channels, the sum of the experts is normalised over the latent, and
    the shared experts read the hidden state at a width of 6144.'''
    body = latent_mixture_of_experts.MIXTURE_BODY
    top_k, = (node for node in tutil.type_search(cat.Broadcasted, body)
              if isinstance(node.operator, dst.TopK))
    require(top_k.operator.k == declared_axes.selected_experts,
            'the router keeps another count')
    latent = whole_model.part_titled(text.LATENT_TITLE, body)
    normalisations = [node for node in tutil.type_search(cat.Broadcasted, latent)
                      if isinstance(node.operator, ops.Normalize)]
    require(len(normalisations) == 1
            and tuple(normalisations[0].input_weaves[0].target().shape())
            == (declared_axes.l,), 'the latent is normalised otherwise')
    linears = {node.operator.name.to_bodies(): (
        tuple(node.input_weaves[0].target().shape()),
        tuple(node.output_weaves[0].target().shape()))
        for node in tutil.type_search(cat.Broadcasted, latent)
        if isinstance(node.operator, ops.Linear)}
    require(linears['W^{Ld}'] == ((declared_axes.m,), (declared_axes.l,))
            and linears['W^{Lu}'] == ((declared_axes.l,), (declared_axes.m,)),
            'the latent projections map otherwise')
    require(ASSIGNED_SIZES['t'] == 2 * ASSIGNED_SIZES['f'],
            'the shared width is not twice the expert width')


def check_situ_bounds_both_branches() -> None:
    '''The gate branch of SiTU is `4 tanh(x / 4) sigma(x)` and the up branch is
    `25 tanh(x / 25)`, each written with the sigmoid.'''
    points = torch.linspace(-60.0, 60.0, 241, dtype=torch.float64)
    gate = evaluate_numerically.with_the_released_constants(feed_forward.situ_gate())
    up = evaluate_numerically.with_the_released_constants(feed_forward.situ_up())
    got_gate = torch_compile.numeric_torch(gate.operator.formula, points)
    got_up = torch_compile.numeric_torch(up.operator.formula, points)
    expected_gate = 4 * torch.tanh(points / 4) * torch.sigmoid(points)
    expected_up = 25 * torch.tanh(points / 25)
    difference = max((got_gate - expected_gate).abs().max().item(),
                     (got_up - expected_up).abs().max().item())
    require(difference < 1e-12, f'SiTU differs by {difference}')


# ==========================================================================
# The layer plan.
# ==========================================================================
def check_the_stack_holds_the_released_layers() -> None:
    '''The stack runs 93 layers, 69 with the delta attention and 24 with the latent
    attention.'''
    counts = (layer_stack.delta_layer_count(MODEL), layer_stack.latent_layer_count(MODEL))
    require(counts == (69, 24), f'the stack runs {counts[0]} delta and {counts[1]} '
                                'latent layers')


def check_the_layer_plan_is_the_released_plan() -> None:
    '''Layer `i` runs the attention `kda_layers` gives it, the dense MLP in layer 0 and
    the mixture after it, adds both outputs at entry `1 + i // 12`, and its attention
    reads a mix of the entries, except in layer 0, which reads the embedding.'''
    plan = layer_plan(layer_stack.layer_stack)
    released = released_plan()
    differing = [layer for layer, (drawn, expected) in enumerate(zip(plan, released))
                 if drawn != expected]
    require(len(plan) == LAYERS and not differing,
            f'the plan holds {len(plan)} layers and differs at layers {differing[:8]}')


def check_every_block_of_the_plan_returns_the_entries() -> None:
    '''Every block of the plan reads and returns the entries of the attention residuals,
    so the entries are the only wire between the layers.'''
    blocks = (layer_stack.first_block, layer_stack.repeated_blocks, layer_stack.last_block)
    require(all(tuple(block.dom()) == (declared_axes.RESIDUAL_ENTRIES,) == tuple(block.cod())
                for block in blocks), 'a block reads or returns another array')


# ==========================================================================
# The CausalSlide.
# ==========================================================================
def views_named(name: str, term: cat.Morphism) -> list[cat.Broadcasted]:
    return [node for node in operations_outside_boxes(term)
            if isinstance(node.operator, ops.View) and node.operator.name is not None
            and node.operator.name.to_bodies() == name]


def check_the_slide_reads_the_hidden_state_back_once_per_attention() -> None:
    '''In the CausalSlide the latent attention reads the hidden state back from every
    query once, at the copy that feeds the queries, the keys, the values and the gate,
    and the delta attention reads the hidden state back over the taps once, at the copy
    that feeds the three projections, the decays, the strengths and the gate.'''
    for box, name in ((latent_attention.ATTENTION_BOX, latent_attention.BACK_VIEW_NAME),
                      (delta_attention.ATTENTION_BOX, delta_attention.TAPS_VIEW_NAME)):
        body = body_of(box, DECODE)
        reads = [node for node in tutil.type_search(cat.Broadcasted, body)
                 if isinstance(node.operator, ops.View) and node.operator.name is not None
                 and node.operator.name.to_bodies() == name]
        require(len(reads) == 1, f'{box} reads back {len(reads)} times')
        read, = reads
        require(tuple(read.dom()[0].shape()) == (declared_axes.x, declared_axes.m),
                f'{box} reads back an array other than the hidden state')


def check_the_scan_passes_the_slide_whole() -> None:
    '''The slide leaves the scan of the delta rule as it was built.'''
    require(box_named(delta_rule_scan.SCAN_BOX, DECODE)
            == box_named(delta_rule_scan.SCAN_BOX, MODEL), 'the slide changed the scan')


# ==========================================================================
# The page.
# ==========================================================================
def as_sent(term: cat.Morphism, settings: notebook_diagrams.DiagramSettings
            ) -> cat.Morphism:
    presented = notebook_diagrams.present_each_side(term, settings)
    sent, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
    return sent


def check_the_page_holds_the_model_in_the_reals() -> None:
    '''The page holds one variant, the model reading every token of the prompt in the
    reals, drawn in the CausalSlide.'''
    notebook_diagrams.check_page_variants(VARIANTS, assemble_page_variants.INITIAL_VARIANT)
    require([(entry.identifier, entry.group.title, entry.title) for entry in VARIANTS]
            == [('decode-unquantised', 'Decode', 'Unquantised')],
            'the variants are named otherwise')
    require(VARIANTS[0].term is DECODE, 'the variant draws another term')


def check_every_legend_row_carries_a_code_name() -> None:
    '''Every row of the legend carries the name of its axis in generated code, and a row
    whose size is one named symbol carries the code name of the size.'''
    legends = notebook_diagrams.page_variant_legends(VARIANTS, PAGE)
    for identifier, rows in legends.items():
        missing = [row['text'] for row in rows if not row['codeName']]
        require(rows and not missing, f'{identifier}: the rows {missing} have no code name')
    for axis in tutil.type_search(cat.Axis, DECODE):
        size = axis.local_size()
        if (axis.uid._name is not None and isinstance(size, nm.FreeNumeric)
                and size.uid._name is not None):
            require(bool(size.uid._name.code_form),
                    f'the size of {axis.uid._name.to_bodies()} has no code name')


def check_every_view_opens_a_box() -> None:
    '''Every named view of the page opens an inspection box.'''
    by_identifier = {entry.identifier: entry for entry in VARIANTS}
    for entry in VARIANTS:
        settings = notebook_diagrams.settings_of_page_variant(entry, by_identifier, PAGE)
        unexplained = explain_reindexings.names_of_unexplained_views(
            as_sent(entry.term, settings))
        require(not unexplained, f'{entry.identifier}: no box opens over {unexplained}')


def check_every_weight_has_a_role() -> None:
    '''Every weight of the model, and the embedding table, has a row saying what it is
    for.'''
    roles = PAGE.operator_roles or {}
    names = {node.operator.name.to_bodies() for node in tutil.type_search(
        cat.Broadcasted, MODEL) if isinstance(node.operator, (ops.Linear, ops.Embedding))
        and node.operator.name is not None}
    missing = sorted(names - set(roles))
    require(not missing, f'the weights {missing} have no role')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_model_reads_identifiers_and_returns_logits,
    check_the_axes_have_the_released_sizes,
    check_the_constants_have_the_released_values,
    check_the_convolution_reads_four_taps_back,
    check_the_scan_is_a_loop_over_the_tokens,
    check_the_delta_attention_computes_the_reference,
    check_the_latent_attention_turns_no_channel,
    check_the_key_joins_a_latent_part_and_a_shared_part,
    check_the_boxes_are_computed_once_per_token_or_head,
    check_the_entries_have_nine_positions,
    check_the_mix_computes_the_reference,
    check_the_embedding_starts_the_entries,
    check_the_mixture_keeps_sixteen_of_896_experts_on_the_latent,
    check_situ_bounds_both_branches,
    check_the_stack_holds_the_released_layers,
    check_the_layer_plan_is_the_released_plan,
    check_every_block_of_the_plan_returns_the_entries,
    check_the_slide_reads_the_hidden_state_back_once_per_attention,
    check_the_scan_passes_the_slide_whole,
    check_the_page_holds_the_model_in_the_reals,
    check_every_legend_row_carries_a_code_name,
    check_every_view_opens_a_box,
    check_every_weight_has_a_role,
)


def report_each_check() -> int:
    '''Every check, with one line printed for each and a line of totals, returning how
    many of them failed.'''
    started = time.perf_counter()
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'ok      {check.__name__}', flush=True)
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures += 1
            print(f'FAILED  {check.__name__}: {type(error).__name__}: {error}', flush=True)
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} checks of the Kimi K3 website '
          f'notebook passed in {time.perf_counter() - started:.0f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
