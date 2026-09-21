# Claude Fable 5.1, effort 80.
'''What an inspection box shows over an operator of DeepSeek-V4.1-Flash.

A block of the model carries its own title, description and references. An operator
carries none, and neither does a reindexing, so the four tables here supply them to
the display.

`OPERATOR_EXPLANATIONS` explains each operator that no rule writes out in primitives: the
top-k selection, the two reads at selected positions, the embedding, the concatenation of
two axes, the covariant view, the merged positions and the elementwise maps whose name
hides part of their formula. `notebooks/display/explain_operators.py` wraps each such
operator in a block that tsncd draws as the operator alone, and the box over it shows the
formula, the description and the references given here.

The reviewer asked on 2026-09-17 for the box over a top-k selection to depend on the form
the selection hands its results out in. `explain_top_k` writes the formula and the
sentences from the `dst.SelectionForm` of the operator and from whether a second operand
carries the positions of the entries selected over, naming the operator's own axes. The
router selects in the `WEIGHTS` form, the indexer and the candidate pool in the
`ONLY_SELECTION` form, and the Reindex layer in the `ONLY_SELECTION` form over positions
held as data. The same day the reviewer asked for the elementwise maps that hide their
formula to open a box, and `ARITHMETIC_ROLES` says what each of them is for, by the text
of its name.

`OPERATOR_REFERENCES` gives the released code that an expanded operator stands for. An
RMSNorm and a linear map are written out by `algebra.registries.standard_expansions`, and
their boxes draw the expansion, so they need no explanation here and take their links
from this table.

`OPERATOR_ROLES` says what each weight of the model is for, by the text of the weight's
name, with the released line that declares it. The reviewer asked on 2026-09-17 for the
box over a weight to explain the role of that weight. The box says the role first, then
what a linear map is, under a formula `para.processing.write_linear_formula` writes from
the weight's own axes. The declarations were read from `inference/model.py` at the pinned
commit on 2026-09-17.

`REINDEXING_EXPLANATIONS` explains each named reindexing of a view, by the text of its
name. The three slices of the mixing coefficients were among them until 2026-09-17, when
the projection `H` of the hyper-connections became the three maps `H_0`, `H_1` and `H_2`
and the slices went. `notebooks/display/explain_reindexings.py` wraps each in a block drawn as the
reindexing alone and writes the formula from the reindexing's own rows, so the table
holds the sentences and the released lines.

Every link is pinned to the commit `reference_links.py` names. The lines of the router,
the indexer and the compressor are the ones the blocks already cite. The lines of
`RMSNorm`, `linear`, `F.embedding`, the gather of the gate and the concatenation of the
window indices with the compressed indices were read from `inference/model.py` at that
commit on 2026-09-16.
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.write_index_notation as write_index_notation
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst
import para.data_structure.inject as inject
import quantization.data_structure.Quantization as Quantization
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation, explain_named_arithmetic)
import notebooks.display.explain_reindexings as explain_reindexings
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    declared_at, engram_lines, kernel_lines, model_lines)
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
from notebooks.sota.DeepSeekV41Flash import (
    clamped_mixture_of_experts, dspark_draft_chain, gumbel_max_sampler, mhc_with_epsilons,
    quantised_caches, rotary_embedding, rotated_indexer, scaled_attention_core,
    vision_pathway, write_at_token_positions)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text


def explain_top_k(target: cat.Broadcasted) -> OperatorExplanation:
    '''What the box over a top-k selection shows, from the form the selection hands its
    results out in. `s` is the scores, `q` the positions a second operand carries, `y`
    the values handed out and `p` the positions handed out.'''
    operator = target.operator
    scored, = write_index_notation.axis_letters(
        target.input_weaves[0].target().shape())
    selected, = write_index_notation.axis_letters(
        target.output_weaves[0].target().shape())
    count = operator.k.to_latex()
    score = write_index_notation.read_at('s', (scored,))
    kept = rf'{score} \text{{ is among the }} {count} \text{{ largest of }} s'
    position = write_index_notation.read_at('p', (selected,))
    value = write_index_notation.read_at('y', (selected,))
    over_positions = dst.selects_over_positions(target)
    source = (write_index_notation.read_at('q', (scored,)) if over_positions
              else write_index_notation.index_of(scored))
    held_as_data = (
        ' ' + text.TOP_K_OVER_POSITIONS_SENTENCE if over_positions else '')
    match operator.form:
        case dst.SelectionForm.WEIGHTS:
            formula = (rf'{write_index_notation.read_at("y", (scored,))} = {score} '
                       rf'\text{{ where }} {kept}')
            description = (
                text.TOP_K_WEIGHTS_DESCRIPTION)
        case dst.SelectionForm.WEIGHTS_SELECT:
            formula = (rf'\{{({position}, {value})\}} = '
                       rf'\{{({source}, {score}) : {kept}\}}')
            description = (
                text.TOP_K_WEIGHTS_SELECT_DESCRIPTION)
        case dst.SelectionForm.ONLY_WEIGHTS:
            formula = rf'\{{{value}\}} = \{{{score} : {kept}\}}'
            description = (
                text.TOP_K_ONLY_WEIGHTS_DESCRIPTION)
        case dst.SelectionForm.ONLY_SELECTION:
            formula = rf'\{{{position}\}} = \{{{source} : {kept}\}}'
            description = (
                text.TOP_K_ONLY_SELECTION_DESCRIPTION)
    return OperatorExplanation(
        title=r'\text{TopK}', formula=formula,
        description=description + held_as_data,
        references=top_k_references(target))


def explain_covariant_view(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over a covariant view, written from the rows of the reindexing the
    operator holds. The reviewer asked on 2026-09-19 for the position written to be
    the row itself and no symbol standing for it. The axes the input carries stand at
    the head of the reindexing's domain, per
    `aops.CovariantView.broadcast_over_absent_axes_and_merge`, so the input is read at
    the first letters and the output is written at every row.'''
    reindexing = target.operator.reindexing
    letters = write_index_notation.axis_letters(tuple(reindexing._dom))
    indices = tuple(write_index_notation.index_of(letter) for letter in letters)
    written = ', '.join(explain_reindexings.position_read(strides, shift, indices)
                        for _, strides, shift in reindexing._cod_stride_shift)
    carried = (letters[:len(target.input_weaves[0].target().shape())]
               if target.input_weaves else ())
    source = write_index_notation.read_at('x', carried) if target.input_weaves else '1'
    broadcast = (
        '' if len(carried) == len(letters) else text.COVARIANT_VIEW_BROADCAST_SENTENCE)
    writes = (
        text.COVARIANT_VIEW_WRITES_EACH_SENTENCE if carried else
        text.COVARIANT_VIEW_WRITES_ONE_SENTENCE if not target.input_weaves
        else text.COVARIANT_VIEW_WRITES_INPUT_SENTENCE)
    return OperatorExplanation(
        formula=f'y[{written}] = {source}',
        description=' '.join(sentence for sentence in (
            writes, broadcast, text.COVARIANT_VIEW_UNIT_SENTENCE) if sentence))


def top_k_references(target: cat.Broadcasted) -> fd.Prod[cat.CodeReference]:
    '''The released lines a top-k selection stands for. The router is the one
    selection that hands out scores. A selection over positions held as data is the
    Reindex layer's, which masks its scores to the candidates before the top-k every
    indexer takes, and the candidate pool takes its own top-k over blocks.'''
    if target.operator.form is not dst.SelectionForm.ONLY_SELECTION:
        return (model_lines(822),)
    if dst.selects_over_positions(target):
        return (model_lines(574, 575), model_lines(578, 579))
    return (model_lines(578, 579), model_lines(607))


ARITHMETIC_ROLES: dict[str, str] = {
    '\\sqrt{s^{+}}': (
        text.SQRT_SOFTPLUS_ROLE),
    '\\tfrac{3}{2}x': (
        text.ROUTE_SCALE_ROLE),
    'x \\sigma(x)': (
        text.SILU_ROLE),
    '\\sigma': (
        text.COLLAPSE_SIGMOID_ROLE),
    '2\\sigma': (
        text.OUTPUT_DOUBLED_SIGMOID_ROLE),
    'z^{-1}': (
        text.NORMALISING_RECIPROCAL_ROLE),
}

OMITTED_MECHANISM_ARITHMETIC_ROLES: dict[str, str] = {
    omitted_mechanisms.GATE_NAME: (
        text.ENGRAM_STREAM_GATE_ROLE),
    **scaled_attention_core.ARITHMETIC_ROLES,
    **rotated_indexer.ARITHMETIC_ROLES,
    **quantised_caches.ARITHMETIC_ROLES,
    **clamped_mixture_of_experts.ARITHMETIC_ROLES,
    **mhc_with_epsilons.ARITHMETIC_ROLES,
    **vision_pathway.ARITHMETIC_ROLES,
}

OMITTED_MECHANISM_ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    **scaled_attention_core.ARITHMETIC_REFERENCES,
    **rotated_indexer.ARITHMETIC_REFERENCES,
    **quantised_caches.ARITHMETIC_REFERENCES,
    **clamped_mixture_of_experts.ARITHMETIC_REFERENCES,
    **mhc_with_epsilons.ARITHMETIC_REFERENCES,
    **vision_pathway.ARITHMETIC_REFERENCES,
}


def explain_fixed_array(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The box over a fixed array of Engram\'s hash, by the text of its name. The
    multipliers and the primes are the two arrays the released layout draws, and
    the bound of each array\'s datatype is a bound on every entry, because a wire
    carries one datatype.'''
    name = target.operator.name.to_bodies()
    if name not in FIXED_ARRAY_ROLES:
        return None
    formula, description, references = FIXED_ARRAY_ROLES[name]
    return OperatorExplanation(
        formula=formula, description=description, references=references)


FIXED_ARRAY_ROLES: dict[str, tuple[str, str, fd.Prod[cat.CodeReference]]] = (
    omitted_mechanisms.ENGRAM_FIXED_ARRAY_ROLES)


POSITIVE_INFINITY_EXPLANATION = OperatorExplanation(
    title=r'\text{Positive Infinity}',
    formula=r'J = \infty',
    description=text.POSITIVE_INFINITY_DESCRIPTION,
    references=(model_lines(602, 605), model_lines(607)))


def explain_positive_infinity(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `ops.ConstantOp`: the constant the pinned block writes, and no
    other. `integrated_explanations.explain_constant` adds the ones of the image
    span before it.'''
    if target.operator.value == nm.Constant(nm.ConstantSymbol.INFINITY):
        return POSITIVE_INFINITY_EXPLANATION
    return None


def natural_under_quantisation(datatype: cat.Datatype) -> cat.Natural | None:
    '''The natural `datatype` is, or holds under a quantisation, and `None` for a
    datatype holding no natural.'''
    held = (datatype.wraps if isinstance(datatype, Quantization.Quantified)
            else datatype)
    return held if isinstance(held, cat.Natural) else None


def held_in_sentence(datatype: cat.Datatype) -> str:
    '''The sentence naming the format a quantised natural is held in, and no sentence
    for a natural with no quantisation.'''
    quantisation = Quantization.quantisation_of(datatype)
    if quantisation is None:
        return ''
    return ' ' + text.HELD_IN_SENTENCE.format(
        format=Quantization.format_name(quantisation))


def explain_cast(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The box over a cast between two naturals, which reads a value into another
    bound and changes no value. The formula names the two bounds, and the sentence for
    the cast is found by the name of the bound the cast narrows to, which a figure
    writing the assigned size into the name leaves as it is. A natural held under a
    quantisation, as every index of the quantised model is, opens the same box, with
    the format named.'''
    source = natural_under_quantisation(target.input_weaves[0].datatype)
    result = natural_under_quantisation(target.output_weaves[0].datatype)
    if source is None or result is None:
        return None
    row_name = omitted_mechanisms.bound_name(result)
    role = None if row_name is None else CAST_ROLES.get(row_name)
    if role is None:
        return None
    return OperatorExplanation(
        formula=(rf'y = x, \quad \mathrm{{Nat}}({source.max_value.to_latex()}) '
                 rf'\to \mathrm{{Nat}}({result.max_value.to_latex()})'),
        description=role + held_in_sentence(target.output_weaves[0].datatype),
        references=(engram_lines(184),))


CAST_ROLES: dict[str, str] = omitted_mechanisms.ENGRAM_CAST_ROLES

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    '\\sqrt{s^{+}}': (model_lines(817),),
    '\\tfrac{3}{2}x': (model_lines(826),),
    'x \\sigma(x)': (model_lines(848),),
}


OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    dst.TopK: explain_top_k,
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
    dst.Select: OperatorExplanation(
        title=r'\text{Select}',
        formula=(r'\mathrm{Select}(w, p)[j] = w[j]\, p[j] '
                 r'\text{ at each live position } j'),
        description=text.SELECT_DESCRIPTION,
        references=(model_lines(823),)),
    dst.IndexSelect: OperatorExplanation(
        title=r'\text{IndexSelect}',
        formula=r'\mathrm{IndexSelect}(i, p) = p[i]',
        description=text.INDEX_SELECT_DESCRIPTION,
        references=(kernel_lines(362, 364),)),
    ops.Embedding: OperatorExplanation(
        title=r'\text{Embedding}',
        formula=r'E(t) = W[t]',
        description=text.EMBEDDING_DESCRIPTION,
        references=(model_lines(174),)),
    aops.ConcatenateAxes: OperatorExplanation(
        title=r'\text{Concatenation}',
        formula=r'y = x^{(1)} \,\Vert\, x^{(2)}, \quad |t| = |w| + |s|',
        description=text.CONCATENATE_AXES_DESCRIPTION,
        references=(model_lines(778),)),
    aops.CovariantView: explain_covariant_view,
    dst.MergedPositions: OperatorExplanation(
        formula=r'i_{B} = |u|\, i_{P} + i_{u}',
        description=text.MERGED_POSITIONS_DESCRIPTION),
    ops.FixedArray: explain_fixed_array,
    ops.ConstantOp: explain_positive_infinity,
    ops.Cast: explain_cast,
}

def table_key(name: str) -> str:
    '''The text the tables key an operator or a reindexing by, from the LaTeX of its
    name. `fd.DynamicName.to_bodies` joins a subscript with no underscore, so
    `W^{K}_{1}` is keyed `W^{K}{1}`.'''
    return fd.DynamicName.from_str(name).to_bodies()


def explanation_from_row(
    row: ExplanationOfOperator, target: cat.Broadcasted,
) -> OperatorExplanation | None:
    return row if isinstance(row, OperatorExplanation) else row(target)


def explain_concatenation(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `aops.ConcatenateAxes` in the figures of the omitted mechanisms.
    The join that closes a rotation box, onto the declared channel axis, takes
    `rotary_embedding.explain_channel_join`, and every other concatenation takes the
    row of the base model.'''
    channel_join = rotary_embedding.explain_channel_join(target)
    if channel_join is not None:
        return channel_join
    return explanation_from_row(OPERATOR_EXPLANATIONS[aops.ConcatenateAxes], target)


def explain_any_cast(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for a cast of either kind: `explain_cast` writes the row of an
    `ops.Cast` between two naturals, which narrows a remainder plus an offset to a row
    number of Engram's table, and `quantised_caches.explain_cast` writes the row of a
    `Quantization.TypeConvert` into a stored form of a round trip or back out of one.'''
    return explain_cast(target) or quantised_caches.explain_cast(target)


OMITTED_MECHANISM_GENERIC_OPERATOR_EXPLANATIONS: dict[str, OperatorExplanation] = {
    quantised_caches.CEILING: quantised_caches.CEILING_EXPLANATION,
    **vision_pathway.GENERIC_OPERATOR_EXPLANATIONS,
    **dspark_draft_chain.GENERIC_OPERATOR_EXPLANATIONS,
    **gumbel_max_sampler.GENERIC_OPERATOR_EXPLANATIONS,
}


def explain_generic_operator(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `ops.GenericOperator`, chosen by the text of the operator's name,
    because one class stands for every operation the package cannot state. A generic
    operator the table does not name is left unexplained.'''
    name = target.operator.name
    return (None if name is None
            else OMITTED_MECHANISM_GENERIC_OPERATOR_EXPLANATIONS.get(name.to_bodies()))


OMITTED_MECHANISM_EMBEDDING_EXPLANATIONS: dict[str, ExplanationOfOperator] = {
    table_key('E'): OPERATOR_EXPLANATIONS[ops.Embedding],
    **omitted_mechanisms.ENGRAM_EMBEDDING_EXPLANATIONS,
    **dspark_draft_chain.EMBEDDING_EXPLANATIONS,
}


def explain_omitted_embedding(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `ops.Embedding`, chosen by the text of the operator's name, because
    the token map, the n-gram tables and the Markov embedding each take a sentence of
    their own and the row of the base model describes the token embedding.'''
    name = target.operator.name
    if name is None:
        return None
    row = OMITTED_MECHANISM_EMBEDDING_EXPLANATIONS.get(name.to_bodies())
    return None if row is None else explanation_from_row(row, target)


def explain_omitted_constant(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `ops.ConstantOp`: the ones a write injects beside its values, and
    the positive infinity of the pinned block.'''
    return (write_at_token_positions.explain_ones(target)
            or explain_positive_infinity(target))


OMITTED_MECHANISM_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    **OPERATOR_EXPLANATIONS,
    **rotary_embedding.OPERATOR_EXPLANATIONS,
    aops.ConcatenateAxes: explain_concatenation,
    ops.Arithmetic: explain_named_arithmetic(
        {**ARITHMETIC_ROLES, **OMITTED_MECHANISM_ARITHMETIC_ROLES},
        {**ARITHMETIC_REFERENCES, **OMITTED_MECHANISM_ARITHMETIC_REFERENCES}),
    ops.Cast: explain_any_cast,
    Quantization.TypeConvert: explain_any_cast,
    ops.GenericOperator: explain_generic_operator,
    ops.Embedding: explain_omitted_embedding,
    ops.ConstantOp: explain_omitted_constant,
    inject.Inject: write_at_token_positions.INJECT_EXPLANATION,
}


OPERATOR_REFERENCES: dict[type[cat.Operator], fd.Prod[cat.CodeReference]] = {
    ops.Normalize: (model_lines(281, 293),),
    ops.Linear: (model_lines(181, 207),),
}


OPERATOR_ROLES: dict[str, OperatorRole] = {
    'W^{Qa}': declared_at(
        text.QUERY_DOWN_PROJECTION_ROLE, 640),
    'W^{Qb}': declared_at(
        text.QUERY_UP_PROJECTION_ROLE, 642),
    'W^{KV}': declared_at(
        text.WINDOW_LATENT_PROJECTION_ROLE, 643),
    '\\mathrm{sink}': declared_at(
        text.SINK_LOGIT_ROLE, 639),
    'W^{Oa}': declared_at(
        text.OUTPUT_PROJECTION_FIRST_HALF_ROLE, 645, 649),
    'W^{Ob}': declared_at(
        text.OUTPUT_PROJECTION_SECOND_HALF_ROLE, 650),
    'W^{R}': declared_at(
        text.ROUTER_ROLE, 805),
    '\\mathrm{bias}': declared_at(
        text.CORRECTION_BIAS_ROLE, 806),
    'W^{G}': declared_at(
        text.EXPERT_GATE_PROJECTION_ROLE, 836),
    'W^{U}': declared_at(
        text.EXPERT_UP_PROJECTION_ROLE, 838),
    'W^{D}': declared_at(
        text.EXPERT_DOWN_PROJECTION_ROLE, 837),
    'W^{Gs}': declared_at(
        text.SHARED_GATE_PROJECTION_ROLE, 887),
    'W^{Us}': declared_at(
        text.SHARED_UP_PROJECTION_ROLE, 887),
    'W^{Ds}': declared_at(
        text.SHARED_DOWN_PROJECTION_ROLE, 887),
    'W^{C}': declared_at(
        text.COMPRESSOR_VALUE_PROJECTION_ROLE, 446),
    'W^{Z}': declared_at(
        text.COMPRESSOR_GATE_PROJECTION_ROLE, 448),
    'k^{I}': declared_at(
        text.INDEXER_KEY_PROJECTION_ROLE, 518),
    'q^{I}': declared_at(
        text.INDEXER_QUERY_PROJECTION_ROLE, 514),
    'w^{I}': declared_at(
        text.INDEXER_HEAD_WEIGHTS_ROLE, 515),
    'H0': declared_at(
        text.COLLAPSE_PROJECTION_ROLE,
        941, 946),
    'H1': declared_at(
        text.OUTPUT_VECTOR_PROJECTION_ROLE, 941, 946),
    'H2': declared_at(
        text.COMBINE_PROJECTION_ROLE, 941, 946),
    'L': declared_at(
        text.OUTPUT_HEAD_ROLE, 1006),
}


REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    'win': ReindexingExplanation(
        description=text.WINDOW_VIEW_DESCRIPTION,
        references=(model_lines(410, 424),)),
    'grp': ReindexingExplanation(
        description=text.GROUP_VIEW_DESCRIPTION,
        references=(model_lines(473, 474), model_lines(787))),
    'back': ReindexingExplanation(
        description=text.BACK_VIEW_DESCRIPTION),
    'pos': ReindexingExplanation(
        description=text.POSITION_VIEW_DESCRIPTION),
    'blk': ReindexingExplanation(
        description=text.BLOCK_VIEW_DESCRIPTION,
        references=(model_lines(599),)),
}


OMITTED_MECHANISM_OPERATOR_ROLES: dict[str, OperatorRole] = {
    **OPERATOR_ROLES,
    **rotary_embedding.TABLE_ROLES,
    **omitted_mechanisms.ENGRAM_WEIGHT_ROLES,
    **{key: OperatorRole(
        role=role,
        references=clamped_mixture_of_experts.WEIGHT_REFERENCES.get(key, ()))
       for key, role in clamped_mixture_of_experts.WEIGHT_ROLES.items()},
    **dspark_draft_chain.OPERATOR_ROLES,
    **vision_pathway.OPERATOR_ROLES,
}


OMITTED_MECHANISM_REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    **REINDEXING_EXPLANATIONS,
    **quantised_caches.REINDEXING_EXPLANATIONS,
    **omitted_mechanisms.ENGRAM_REINDEXING_EXPLANATIONS,
    **dspark_draft_chain.REINDEXING_EXPLANATIONS,
    **gumbel_max_sampler.REINDEXING_EXPLANATIONS,
}
