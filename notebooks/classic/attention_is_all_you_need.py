# Claude Opus 5.5 (1M context), reasoning effort 40. The view of the mask was renamed
# Mask, set upright, by Claude Opus 5.5 (1M context), effort 40, on 2026-09-27.
'''The encoder-decoder transformer of *Attention Is All You Need*, as a morphism in Br.

The model reads a source sentence and a target sentence as token identifiers and returns
the probability of every token of the vocabulary at every target position. It is the
base model of Table 3 of the paper, written from Sections 3 and 5.4.

The encoder embeds the source, adds the positional encoding and runs a stack of `N`
layers, each a self-attention sublayer and a feed-forward sublayer. Every sublayer
stands inside an Add & Norm block, which applies dropout to the output of the sublayer,
adds the input of the sublayer and applies a layer normalisation. The output of the
encoder is dropped onto the tape slot `A`. The decoder embeds the target, adds the
positional encoding and runs its own stack of `N` layers, each a masked self-attention
sublayer, a cross-attention sublayer that grabs `A` from the tape, and a feed-forward
sublayer. A linear map and a softmax over the vocabulary finish the model.

The key positions of a self-attention are the query positions read a second time. The
encoder projects its keys and its values from the same copy of the source as its
queries, so the score contraction carries the source axis at two positions, one for
the query and one for the key, and is written in index variables. The decoder reads
its keys at `i_y - i_w`, the target
position `i_w` before the query, through the view named Mask, built by
`mark_sparse_domains.guarded_view`. Slot `i_w` of a query before position `i_w` names a
negative position and holds the unit, so the view marks the slot axis as `w|y`, live
where `i_y - i_w >= 0`, and no mask operator is written. The slot axis has the size of
the target, so the view reads every earlier position.

The positional encoding is a table with no operands, boxed as `PE`, whose body is drawn
once and opens in the inspection box. `dst.Rotary` holds the complex
number `e^{i i_x theta[i_t]}` at position `i_x` and frequency pair `i_t`, with
`theta[i_t] = beta^{-i_t / |t|}` and `|m| = 2 |t|`, which is the angle of Section 3.5.
The paper puts the sine at channel `2 i_t` and the cosine at channel `2 i_t + 1`. The
product of the imaginary unit and the conjugate of the table is `sin + i cos`, and
`dst.Decomplex` writes each complex number as its real part followed by its imaginary
part, so the channels come out in the order of the paper.

Section 3.4 shares one weight matrix between the two embeddings and the output
projection, and the expression names all three after `E`. The expression holds three
operators, and nothing in it states that their weights are one array.
'''
from __future__ import annotations

import algebra.einops_simplification as einops_simplification
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst
import deepseek.registries.standard_expansions  # noqa: F401 - the rotary table's rule
import para.data_structure.Para as Para
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.classic.attention_is_all_you_need_wording import TEXT as text
from notebooks.classic.reference_links import (
    tensor2tensor_lines, transformer_role, transformer_section)
from notebooks.classic.shared_mechanisms import (
    contract, read_back_from_every_position)
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation, explain_named_arithmetic)
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, named_slot, route)

R = cat.Reals()
MINUS_HALF = nm.Integer(-1) / nm.Integer(2)
HALF = nm.Integer(1) / nm.Integer(2)

x = cat.RawAxis.named('x', code_form='source_positions')
y = cat.RawAxis.named('y', code_form='target_positions')
w = fd.DynamicName('w', code_form='earlier_targets').capture(
    cat.RawAxis(_size=y.local_size()))
m = cat.RawAxis.named('m', code_form='model_width')
h = cat.RawAxis.named('h', code_form='heads')
k = cat.RawAxis.named('k', code_form='head_width')
f = cat.RawAxis.named('f', code_form='feed_forward_width')
t = fd.DynamicName('t', code_form='frequency_pairs').capture(
    cat.RawAxis(_size=m.local_size() * HALF))
v = cat.RawAxis.named(fd.DynamicName(
    'v', settings=fd.DynamicNameSettings(overline=True), code_form='vocabulary'))

LAYER_COUNT = nm.FreeNumeric.named('N')
POSITIONAL_BASE = nm.FreeNumeric.named('\\beta')
DROPOUT_RATE = nm.FreeNumeric.named(fd.DynamicName('P_{\\mathrm{drop}}'))
NORM_EPSILON = nm.FreeNumeric.named('\\epsilon')

TOKEN = cat.Natural(v.local_size())
SOURCE_TOKENS = cat.Array(TOKEN, (x,))
TARGET_TOKENS = cat.Array(TOKEN, (y,))
SOURCE_STATE = cat.Array(R, (x, m))
TARGET_STATE = cat.Array(R, (y, m))
COMPLEX = dst.Complex(R)

ENCODED_INPUT = named_slot('A')

TRANSFORMER_COLOUR = '#FFFFFF'
EMBEDDING_COLOUR = '#FCE0E1'
POSITIONAL_COLOUR = '#FFFFFF'
STACK_COLOUR = '#F3F3F4'
ATTENTION_COLOUR = '#FFE2BB'
CORE_COLOUR = '#C5BEDF'
ADD_NORM_COLOUR = '#F1F4C1'
FEED_FORWARD_COLOUR = '#C1E8F7'
OUTPUT_COLOUR = '#DBDFEF'
SOFTMAX_COLOUR = '#CCE7CF'

OUTPUT_PROJECTION_NAME = 'E^{\\top}'
EMBEDDING_SCALE_NAME = '|m|^{1/2} x'
SCORE_SCALE_NAME = '|k|^{-1/2} x'
POSITIONAL_ENCODING_NAME = '\\mathrm{PE}'
POSITIONAL_ENCODING_BOX = 'PE'
SINE_FIRST_NAME = '\\mathrm{i}\\,\\overline{x}'
MASK_NAME = '\\mathrm{Mask}'
RELU_NAME = '\\mathrm{ReLU}(x)'
DROPOUT_NAME = '↯'

CORE_FORMULA = (
    r'\mathrm{Attention}(Q, K, V) = '
    r'\mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{|k|}}\right) V')
FEED_FORWARD_FORMULA = (
    r'\mathrm{FFN}(z) = \max(0,\, z W_{1} + b_{1})\, W_{2} + b_{2}')
ADD_NORM_FORMULA = (
    r'\mathrm{LayerNorm}\big(z + \mathrm{Dropout}(\mathrm{Sublayer}(z))\big)')
DROPOUT_FORMULA = (
    r'y = \frac{b\, x}{1 - P_{\mathrm{drop}}},\quad '
    r'b \sim \mathrm{Bernoulli}(1 - P_{\mathrm{drop}})')
EMBEDDING_FORMULA = r'y[i_{m}] = E[x, i_{m}]'
DECOMPLEX_FORMULA = r'y[2 i_{t}] + \mathrm{i}\, y[2 i_{t} + 1] = z[i_{t}]'


def positional_encoding_formula[A: cat.Axis](positions: A) -> str:
    '''The two channels of pair `i_t` at a position of `positions`, as Section 3.5
    writes them, in the index notation of the package.'''
    letter = positions.uid._name.to_latex()
    angle = rf'i_{{{letter}}}\, \beta^{{-i_{{t}} / |t|}}'
    return (rf'\mathrm{{PE}}[i_{{{letter}}}, 2 i_{{t}}] = \sin({angle}),\quad '
            rf'\mathrm{{PE}}[i_{{{letter}}}, 2 i_{{t}} + 1] = \cos({angle})')


def dropout() -> cat.Broadcasted:
    '''Dropout at the rate `P_drop`, named with the lightning bolt the hand-drawn
    diagram draws. The bolt is the Unicode character, because KaTeX has no
    `\\lightning`. The operator is an elementwise map with no formula, because what it
    computes is random during training and the identity at inference.'''
    return ops.Dropout.template(name=DROPOUT_NAME)


def relu() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.RectifiedLinear(nm.x))


def scale_by(factor: nm.Numeric, name: str) -> cat.Broadcasted:
    '''Every value multiplied by `factor`, named so that a root prints as a power.'''
    return ops.Arithmetic.template(nm.x * factor, name=name)


# ==========================================================================
# Embedding and positional encoding.
# ==========================================================================
def positional_encoding[A: cat.Axis](positions: A) -> cat.Broadcasted:
    '''The sines and cosines of Section 3.5 over `positions` and the model width, as a
    table with no operands, boxed as `PE`.'''
    table = dst.Rotary.template(positions, t, base=POSITIONAL_BASE,
                                name=POSITIONAL_ENCODING_NAME)
    sine_first = ops.Arithmetic.template(
        nm.Constant(nm.ConstantSymbol.IMAGINARY_UNIT) * nm.Conjugate(nm.x),
        base=table.cod()[0], name=SINE_FIRST_NAME)
    return boxed(cat.Block.template(
        table @ sine_first
        @ (positions >> dst.Decomplex.template(base=R, pairs=t, channels=m)),
        title=text.POSITIONAL_ENCODING_TITLE, fill_color=POSITIONAL_COLOUR,
        description=text.POSITIONAL_ENCODING_DESCRIPTION,
        formula=positional_encoding_formula(positions),
        references=(transformer_section('3.5'),)), POSITIONAL_ENCODING_BOX)


def embed[A: cat.Axis](positions: A, title: str, description: str) -> cat.Block:
    '''The tokens of one sentence embedded, scaled by the root of the model width, added
    to the positional encoding and passed through dropout.'''
    embedded = ((positions >> ops.Embedding.template(TOKEN, (m,)))
                @ scale_by(m.local_size() ** HALF, EMBEDDING_SCALE_NAME))
    return cat.Block.template(
        (positional_encoding(positions) * embedded)
        @ ops.AdditionOp.template() @ dropout(),
        title=title, fill_color=EMBEDDING_COLOUR, description=description,
        references=(transformer_section('3.4'), transformer_section('3.5'),
                    transformer_section('5.4')))


# ==========================================================================
# Attention.
# ==========================================================================
def project_heads[A: cat.Axis](positions: A, name: str) -> cat.BroadcastedCategory:
    '''Every position projected into one vector of the head width per head.'''
    return positions >> ops.Linear.template((m,), (h, k), name)


def read_earlier_targets() -> cat.Broadcasted:
    '''The keys and values of the target read back from every query, slot `i_w` holding
    the position `i_w` before it, at `i_y - i_w`. A slot before the first position
    reads the unit, which marks `w` as `w|y`.'''
    return read_back_from_every_position(y, w, (h, k), MASK_NAME)


def scaled_dot_product_attention[A: cat.Axis](
    queries: A, keys: tuple[A, ...], slots: tuple[A, ...],
) -> cat.Block:
    '''Attention(Q, K, V) = softmax(Q K^T / |k|^{1/2}) V for every head. `keys` is the
    shape of a key and of a value, and `slots` the axes of it a query reads across.
    The contractions are written in index variables, one for the query position and
    one per slot, because in the self-attention of the encoder the query axis is the
    slot, and the scores then carry that axis at two positions.'''
    query = einops_simplification.IndexVariable('query', queries)
    slot_variables = tuple(
        einops_simplification.IndexVariable(f'slot{i}', axis)
        for i, axis in enumerate(slots))
    head = einops_simplification.IndexVariable('head', h)
    channel = einops_simplification.IndexVariable('channel', k)

    def variable_of(axis: A) -> einops_simplification.IndexVariable:
        for slot, slot_variable in zip(slots, slot_variables):
            if axis is slot:
                return slot_variable
        if axis is queries:
            return query
        if axis is h:
            return head
        if axis is k:
            return channel
        raise ValueError(
            f'{axis} is neither the query axis, a slot, the heads nor the head width')

    key_shape = tuple(variable_of(axis) for axis in keys)
    query_shape = (query, head, channel)
    scores = (head, query, *slot_variables)
    return cat.Block.template(
        ((contract((query_shape, key_shape), scores)
          @ scale_by(k.local_size() ** MINUS_HALF, SCORE_SCALE_NAME)
          @ ops.SoftMax.template())
         * hold(cat.Array(R, keys)))
        @ contract((scores, key_shape), query_shape),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, formula=CORE_FORMULA,
        references=(transformer_section('3.2.1'),))


def output_projection[A: cat.Axis](queries: A) -> cat.BroadcastedCategory:
    return queries >> ops.Linear.template((h, k), (m,), 'W^{O}')


def encoder_self_attention() -> cat.Block:
    keys = (x, h, k)
    return cat.Block.template(
        route((0, 0, 0), (SOURCE_STATE,))
        @ (project_heads(x, 'W^{Q}') * project_heads(x, 'W^{K}')
           * project_heads(x, 'W^{V}'))
        @ scaled_dot_product_attention(x, keys, (x,))
        @ output_projection(x),
        title=text.SELF_ATTENTION_TITLE, fill_color=ATTENTION_COLOUR,
        description=text.SELF_ATTENTION_DESCRIPTION,
        references=(transformer_section('3.2.2'), transformer_section('3.2.3')))


def decoder_masked_self_attention() -> cat.Block:
    keys = (y, w, h, k)
    key_path = ((project_heads(y, 'W^{K}') @ read_earlier_targets())
                * (project_heads(y, 'W^{V}') @ read_earlier_targets()))
    return cat.Block.template(
        route((0, 0, 0), (TARGET_STATE,))
        @ (project_heads(y, 'W^{Q}') * key_path)
        @ scaled_dot_product_attention(y, keys, (w,))
        @ output_projection(y),
        title=text.MASKED_SELF_ATTENTION_TITLE, fill_color=ATTENTION_COLOUR,
        description=text.MASKED_SELF_ATTENTION_DESCRIPTION,
        references=(transformer_section('3.2.3'),))


def decoder_cross_attention() -> cat.Block:
    '''The target positions query the encoded input, which is grabbed from the tape.'''
    keys = (x, h, k)
    encoded = Para.Grab(tape=ENCODED_INPUT, size=SOURCE_STATE)
    key_path = route((0, 0), (SOURCE_STATE,)) @ (
        project_heads(x, 'W^{K}') * project_heads(x, 'W^{V}'))
    return cat.Block.template(
        (project_heads(y, 'W^{Q}') * (encoded @ key_path))
        @ scaled_dot_product_attention(y, keys, (x,))
        @ output_projection(y),
        title=text.CROSS_ATTENTION_TITLE, fill_color=ATTENTION_COLOUR,
        description=text.CROSS_ATTENTION_DESCRIPTION,
        references=(transformer_section('3.2.3'),))


# ==========================================================================
# Feed-forward, Add & Norm and the two stacks.
# ==========================================================================
def feed_forward[A: cat.Axis](positions: A) -> cat.Block:
    '''FFN(z) = max(0, z W_1 + b_1) W_2 + b_2 at every position.'''
    return cat.Block.template(
        (positions >> ops.Linear.template((m,), (f,), 'W_{1}', bias=True))
        @ relu()
        @ (positions >> ops.Linear.template((f,), (m,), 'W_{2}', bias=True)),
        title=text.FEED_FORWARD_TITLE, fill_color=FEED_FORWARD_COLOUR,
        description=text.FEED_FORWARD_DESCRIPTION, formula=FEED_FORWARD_FORMULA,
        references=(transformer_section('3.3'),))


def add_and_norm[A: cat.Axis](
    sublayer: cat.BroadcastedCategory, positions: A,
) -> cat.Block:
    '''LayerNorm(z + Dropout(Sublayer(z))) at every position.'''
    state = cat.Array(R, (positions, m))
    return cat.Block.template(
        route((0, 0), (state,))
        @ ((sublayer @ dropout()) * hold(state))
        @ ops.AdditionOp.template()
        @ (positions >> ops.LayerNorm.template((m,), epsilon=NORM_EPSILON)),
        title=text.ADD_NORM_TITLE, fill_color=ADD_NORM_COLOUR,
        description=text.ADD_NORM_DESCRIPTION, formula=ADD_NORM_FORMULA,
        references=(transformer_section('3.1'), transformer_section('5.4'),
                    tensor2tensor_lines('layers/common_hparams.py', 130, 133)))


def encoder_stack() -> cat.Block:
    return cat.Block.template(
        add_and_norm(encoder_self_attention(), x)
        @ add_and_norm(feed_forward(x), x),
        title=text.ENCODER_TITLE, fill_color=STACK_COLOUR, repetition=LAYER_COUNT,
        description=text.ENCODER_DESCRIPTION,
        references=(transformer_section('3.1'),))


def decoder_stack() -> cat.Block:
    return cat.Block.template(
        add_and_norm(decoder_masked_self_attention(), y)
        @ add_and_norm(decoder_cross_attention(), y)
        @ add_and_norm(feed_forward(y), y),
        title=text.DECODER_TITLE, fill_color=STACK_COLOUR, repetition=LAYER_COUNT,
        description=text.DECODER_DESCRIPTION,
        references=(transformer_section('3.1'),))


def output_probabilities() -> cat.Block:
    return cat.Block.template(
        (y >> ops.Linear.template((m,), (v,), OUTPUT_PROJECTION_NAME))
        @ ops.SoftMax.template(),
        title=text.OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.OUTPUT_DESCRIPTION,
        references=(transformer_section('3.4'),))


def encode() -> cat.BroadcastedCategory:
    '''The source sentence through the encoder, its output dropped onto the tape.'''
    return (embed(x, text.INPUT_EMBEDDING_TITLE, text.INPUT_EMBEDDING_DESCRIPTION)
            @ encoder_stack()
            @ Para.Drop(tape=ENCODED_INPUT, size=SOURCE_STATE))


def decode() -> cat.BroadcastedCategory:
    return (embed(y, text.OUTPUT_EMBEDDING_TITLE, text.OUTPUT_EMBEDDING_DESCRIPTION)
            @ decoder_stack()
            @ output_probabilities())


def transformer() -> cat.Block:
    return cat.Block.template(
        (encode() * hold(TARGET_TOKENS)) @ decode(),
        title=text.TRANSFORMER_TITLE, fill_color=TRANSFORMER_COLOUR,
        description=text.TRANSFORMER_DESCRIPTION,
        references=(transformer_section('3'),))


# ==========================================================================
# What the inspection boxes show over the operators and the views.
# ==========================================================================
ARITHMETIC_ROLES: dict[str, str] = {
    EMBEDDING_SCALE_NAME: text.EMBEDDING_SCALE_ROLE,
    SCORE_SCALE_NAME: text.SCORE_SCALE_ROLE,
    RELU_NAME: text.RELU_ROLE,
    SINE_FIRST_NAME: text.SINE_FIRST_ROLE,
}

ARITHMETIC_REFERENCES: dict[str, tuple[cat.CodeReference, ...]] = {
    EMBEDDING_SCALE_NAME: (transformer_section('3.4'),),
    SCORE_SCALE_NAME: (transformer_section('3.2.1'),),
    RELU_NAME: (transformer_section('3.3'),),
    SINE_FIRST_NAME: (transformer_section('3.5'),),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    ops.Dropout: OperatorExplanation(
        title=r'\text{Dropout}', formula=DROPOUT_FORMULA,
        description=text.DROPOUT_DESCRIPTION,
        references=(transformer_section('5.4'),)),
    ops.Embedding: OperatorExplanation(
        title=r'\text{Embedding}', formula=EMBEDDING_FORMULA,
        description=text.EMBEDDING_OPERATOR_DESCRIPTION,
        references=(transformer_section('3.4'),)),
    dst.Decomplex: OperatorExplanation(
        title=r'\text{Decomplex}', formula=DECOMPLEX_FORMULA,
        description=text.DECOMPLEX_DESCRIPTION,
        references=(transformer_section('3.5'),)),
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
}

OPERATOR_ROLES: dict[str, OperatorRole] = {
    'W^{Q}': transformer_role(text.QUERY_ROLE, '3.2.2'),
    'W^{K}': transformer_role(text.KEY_ROLE, '3.2.2'),
    'W^{V}': transformer_role(text.VALUE_ROLE, '3.2.2'),
    'W^{O}': transformer_role(text.OUTPUT_ROLE, '3.2.2'),
    'W_{1}': transformer_role(text.FIRST_FEED_FORWARD_ROLE, '3.3'),
    'W_{2}': transformer_role(text.SECOND_FEED_FORWARD_ROLE, '3.3'),
    OUTPUT_PROJECTION_NAME: transformer_role(text.OUTPUT_PROJECTION_ROLE, '3.4'),
    'LayerNorm': OperatorRole(
        role=text.LAYER_NORM_ROLE,
        references=(transformer_section('3.1'),
                    tensor2tensor_lines('layers/common_hparams.py', 142, 144))),
    POSITIONAL_ENCODING_NAME: transformer_role(text.POSITIONAL_TABLE_ROLE, '3.5'),
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    MASK_NAME: ReindexingExplanation(
        description=text.MASK_VIEW_DESCRIPTION,
        references=(transformer_section('3.2.3'),)),
}

OPERATOR_REFERENCES: dict[type[cat.Operator], tuple[cat.CodeReference, ...]] = {
    ops.SoftMax: (transformer_section('3.2.1'), transformer_section('3.4')),
}
