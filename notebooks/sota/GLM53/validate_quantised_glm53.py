# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim made about the quantised GLM-5.3.

    python notebooks/sota/GLM53/validate_quantised_glm53.py

`quantised_whole_model` writes the quantisations of the FP8 checkpoint, as
`transformers` runs it, onto `whole_model.glm53`. `notebooks/website/modern/GLM53.ipynb`
states the policy, the quantisation of every weight, the casts and the departures of
eight boxes, and draws the model. Each claim is one `check_` function here, and
`notebooks/website/modern/validate_glm53.py` runs every one of them but
`check_every_wire_of_the_page_carries_a_quantisation`, whose page draws the quantised
model alone. The script prints one line per check and exits non-zero on a failure, and
`check_every_claim_of_the_notebook` runs the same checks and raises when one of them
fails, in the shape of `validate_glm53.py`.
'''
from __future__ import annotations

import dataclasses
import pathlib
import sys
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import agent_display as ad  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import graphs.processing.Hypergraph2Morphism as h2m  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import para.data_structure.ParaWrap as para_wrap  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import quantization.registries.operator_quantisations as operator_quantisations  # noqa
import term_utilities.term_utilities as tutil  # noqa: E402

import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.sota_figures as figures  # noqa: E402
import notebooks.sota.GLM53.operator_explanations as operator_explanations  # noqa
import notebooks.sota.GLM53.quantised_whole_model as quantised_whole_model  # noqa
import notebooks.sota.GLM53.whole_model as whole_model  # noqa: E402
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text  # noqa

MODEL: cat.Morphism = quantised_whole_model.glm53_quantised
STRIPPED: cat.Morphism = quantised_whole_model.glm53_without_quantisations
UNQUANTISED: cat.Morphism = whole_model.glm53

BF16 = Quantization.BF16
FP32 = Quantization.FP32
INT64 = Quantization.INT64
INT32 = Quantization.INT32
FP8_BLOCK_WEIGHT = quantised_whole_model.FP8_BLOCK_WEIGHT
FP8_ROUNDED_OPERAND = quantised_whole_model.FP8_ROUNDED_OPERAND

FP8_WEIGHTS = (
    'W^{Qa}', 'W^{Qn}', 'W^{Qr}', 'W^{KVa}', 'W^{Kr}', 'W^{Kb}', 'W^{Vb}', 'W^{O}',
    'q^{I}', 'k^{I}', 'W^{Gd}', 'W^{Ud}', 'W^{Dd}', 'W^{Gs}', 'W^{Us}', 'W^{Ds}',
    'W^{G}', 'W^{U}', 'W^{D}')
RELEASED_WEIGHT_QUANTISATIONS: dict[str, Quantization.Quantified] = {
    **{name: FP8_BLOCK_WEIGHT for name in FP8_WEIGHTS},
    'E': BF16, 'W^{L}': BF16, 'w^{I}': BF16, 'W^{R}': FP32, '\\mathrm{bias}': FP32}

ROUNDED = Quantization.format_name(FP8_ROUNDED_OPERAND)
RELEASED_CAST_COUNTS: dict[tuple[str, str], int] = {
    ('BF16', ROUNDED): 33,
    ('BF16', 'FP32'): 4,
    ('FP32', 'BF16'): 3,
    ('INT32', 'INT64'): 2,
    ('INT64', 'INT32'): 2,
}
'''The casts written in the quantised model, each counted once per written operation.
The 33 roundings are 5 in each of the three Full bodies and the two Shared bodies, 2 in
the indexer box, 2 in the dense MLP and 4 in the mixture. The four upcasts are the
router's operand, the indexer queries and keys, and the indexer head weights. The three
downcasts are the attention probabilities, the attention output and the gates.'''

PAGE_TITLE = 'GLM-5.3 (Quantised, Reference Implementation)'


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the quantised expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def quantisation_of(datatype: cat.Datatype) -> Quantization.Quantified | None:
    return Quantization.quantisation_of(datatype)


def box_body(name: str, model: cat.Morphism = MODEL) -> cat.Morphism:
    '''The body of the first box of `model` named `name`, looking through the wrap
    around a box holding a grab or a drop.'''
    part = whole_model.part_named(name, model)
    box = part.body if isinstance(part, para_wrap.ParaWrap) else part
    return box.operator.block.body


def box_named(name: str) -> cat.Broadcasted:
    part = whole_model.part_named(name, MODEL)
    return part.body if isinstance(part, para_wrap.ParaWrap) else part


def operations(morphism: cat.Morphism, kind: type[cat.Operator]
               ) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in quantise_model.operations_of(morphism)
                 if isinstance(node.operator, kind))


def operations_outside_boxes(morphism: cat.Morphism) -> tuple[cat.Broadcasted, ...]:
    '''Every operation of `morphism` itself, with a box read as one operation and its
    body left unentered.'''
    return tuple(leaf.wraps for leaf in quantise_model.graph_leaves(morphism)
                 if isinstance(leaf.wraps, cat.Broadcasted))


def cast_pairs(morphism: cat.Morphism) -> list[tuple[str, str]]:
    '''The format read and the format written by every cast of `morphism`, the casts
    inside its boxes included.'''
    return [(Quantization.format_name(quantisation_of(cast.operator.source)),
             Quantization.format_name(quantisation_of(cast.operator.target)))
            for cast in quantise_model.casts_of(morphism)]


def page_settings() -> notebook_diagrams.DiagramSettings:
    '''The settings under which an HTML page draws the quantised model, with
    `SubBlocks.NO_BODIES`, the released sizes and `AdvancedDisplay.INTERACTIVE`.'''
    return quantised_whole_model.with_quantised_explanation_tables(
        figures.DiagramSettings(
            mode=figures.DiagramMode.HTML,
            tape=figures.TapePresentation.ABSORBED,
            axis_sizes=figures.AxisSizes.SUBSCRIPT,
            axis_label_font_size=0.8,
            block_recycling=figures.BlockRecycling.RECYCLED,
            advanced_display=figures.AdvancedDisplay.INTERACTIVE,
            expanded_parameters=figures.ExpandedParameters.WEIGHT_ARRAYS,
            clean_quantisation_labels=False,
            sub_blocks=figures.SubBlocks.NO_BODIES,
            assigned_sizes=quantised_whole_model.released_assigned_sizes(),
            title=PAGE_TITLE))


def as_sent_to_the_page(settings: notebook_diagrams.DiagramSettings) -> cat.Morphism:
    '''The quantised model as `show_diagram` sends it under `settings`, after every
    presentation pass and the wrapping of the explained operators.'''
    presented = notebook_diagrams.present_each_side(MODEL, settings)
    sent, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
    return sent


# ==========================================================================
# The quantisations of the weights and of the wires.
# ==========================================================================
def check_every_weight_has_the_quantisation_of_the_checkpoint() -> None:
    '''Nineteen projections are E4M3 with one FP32 scale per 128 by 128 block. The
    embedding, the output head and the indexer head weights are BF16, and the router
    weight and the correction bias are read in FP32.'''
    held = dict(quantised_whole_model.WEIGHT_QUANTISATIONS)
    require(held == RELEASED_WEIGHT_QUANTISATIONS,
            f'the weights differ at '
            f'{sorted(set(held.items()) ^ set(RELEASED_WEIGHT_QUANTISATIONS.items()))}')
    require(Quantization.format_name(FP8_BLOCK_WEIGHT) == 'E4M3 with FP32 per 128x128',
            f'the FP8 weight is named {Quantization.format_name(FP8_BLOCK_WEIGHT)}')


def check_every_wire_carries_a_quantisation() -> None:
    '''Every wire of the quantised model holding a number carries a quantisation, and
    every operator of the model has a rule in the registry.'''
    unwritten = quantised_whole_model.unquantised_weaves()
    require(not unwritten, f'{len(unwritten)} wires carry no quantisation')
    without = quantised_whole_model.operators_without_a_rule()
    require(not without, f'the operators {without} have no rule')


def check_every_wire_of_the_page_carries_a_quantisation() -> None:
    '''Every wire of the model as it is sent to the page carries a quantisation, which
    holds because the page keeps the quantisation label of every wire. The cleaning
    of quantisation labels would take the quantisation off a wire carrying the
    quantisation of every operand of the operation writing it.'''
    settings = page_settings()
    sent = as_sent_to_the_page(settings)
    unwritten = quantise_model.unquantised_weaves(sent)
    require(not unwritten, f'{len(unwritten)} wires of the page carry no quantisation')
    cleaned = as_sent_to_the_page(
        dataclasses.replace(settings, clean_quantisation_labels=True))
    require(quantise_model.unquantised_weaves(cleaned),
            'the cleaning of quantisation labels keeps every quantisation, so the page '
            'no longer needs it turned off')


# ==========================================================================
# The rounding in front of an FP8 projection.
# ==========================================================================
def check_an_fp8_projection_reads_a_rounded_operand() -> None:
    '''Every projection whose weight is FP8 reads its operand as E4M3 with one FP32
    scale per 128 channels and returns BF16, every cast into that format reads BF16,
    and every other projection reads and returns the quantisation of its weight.'''
    policy = quantised_whole_model.RELEASED_POLICY
    for projection in operations(MODEL, ops.Linear):
        name = projection.operator.name.to_bodies()
        weight = quantised_whole_model.WEIGHT_QUANTISATIONS[name]
        operands = {quantisation_of(weave.datatype) for weave in projection.input_weaves}
        result = quantisation_of(projection.output_weaves[0].datatype)
        if operator_quantisations.has_fewer_bits_than_the_activations(weight, policy):
            require(operands <= {FP8_ROUNDED_OPERAND} and result == BF16,
                    f'{name} reads {operands} and returns {result}')
        else:
            require(operands <= {weight} and result == weight,
                    f'{name} reads {operands} and returns {result}')
    for cast in quantise_model.casts_of(MODEL):
        if quantisation_of(cast.operator.target) == FP8_ROUNDED_OPERAND:
            require(quantisation_of(cast.operator.source) == BF16,
                    f'a rounding reads {quantisation_of(cast.operator.source)}')


def check_the_casts_of_the_model() -> None:
    '''The quantised model writes 44 casts in five pairs of formats, 33 of them the
    rounding in front of an FP8 projection.'''
    counts = dict(quantised_whole_model.cast_counts())
    require(counts == RELEASED_CAST_COUNTS, f'the casts are {counts}')
    require(sum(counts.values()) == 44, f'{sum(counts.values())} casts')
    require(quantise_model.conversions_between_two_quantisations(MODEL),
            'a conversion reads or writes no quantisation')


# ==========================================================================
# The boxes departing from the default rules.
# ==========================================================================
def check_the_rotary_turn_is_computed_in_bf16() -> None:
    '''The rotation box reads and returns BF16 and holds no cast, because the table of
    turns is cast to BF16 and the turn multiplies BF16 by BF16.'''
    box = box_named(quantised_whole_model.ROTATION_BOX)
    ends = {quantisation_of(array.datatype) for array in (*box.dom(), *box.cod())}
    require(ends == {BF16}, f'the rotation reads and returns {ends}')
    body = box_body(quantised_whole_model.ROTATION_BOX)
    require(not quantise_model.casts_of(body), 'the rotation holds a cast')
    tables = operations(body, dst.Rotary)
    require(tables and all(
        quantisation_of(table.output_weaves[0].datatype) == BF16 for table in tables),
        'the table of turns is not BF16')


def check_the_indexer_scores_in_fp32() -> None:
    '''The scoring box casts the indexer queries, the keys and the head weights from
    BF16 to FP32, contracts in FP32, and returns FP32 scores. The indexer box returns
    them in FP32, and the top-2048 reads them in FP32.'''
    body = box_body(quantised_whole_model.INDEXER_SCORES_BOX)
    require(cast_pairs(body) == [('BF16', 'FP32')] * 3,
            f'the scoring box casts {cast_pairs(body)}')
    for contraction in operations(body, ops.Einops):
        read = {quantisation_of(weave.datatype) for weave in contraction.input_weaves}
        require(read == {FP32}, f'a contraction of the scores reads {read}')
    for name in (quantised_whole_model.INDEXER_SCORES_BOX,
                 quantised_whole_model.INDEXER_BOX):
        returned = {quantisation_of(array.datatype) for array in box_named(name).cod()}
        require(returned == {FP32}, f'{name} returns {returned}')
    for selection in operations(box_body('Full'), dst.TopK):
        read = quantisation_of(selection.input_weaves[0].datatype)
        require(read == FP32, f'the top-2048 reads {read}')


def check_the_attention_core_is_the_fused_kernel() -> None:
    '''The core reads BF16 queries, keys and values and returns BF16. Its scores and
    softmax are FP32, and the probabilities are cast to BF16 before the product with
    the values.'''
    box = box_named(quantised_whole_model.ATTENTION_CORE_BOX)
    ends = {quantisation_of(array.datatype) for array in (*box.dom(), *box.cod())}
    require(ends == {BF16}, f'the core reads and returns {ends}')
    body = box_body(quantised_whole_model.ATTENTION_CORE_BOX)
    require(cast_pairs(body) == [('FP32', 'BF16')] * 2,
            f'the core casts {cast_pairs(body)}')
    softmax, = operations(body, ops.SoftMax)
    require(quantisation_of(softmax.output_weaves[0].datatype) == FP32,
            'the softmax is not FP32')
    products = operations(body, ops.Einops)
    read = [{quantisation_of(weave.datatype) for weave in product.input_weaves}
            for product in products]
    require(read == [{BF16}, {BF16}], f'the two products read {read}')


def check_the_feed_forward_maps_apply_their_arithmetic_in_bf16() -> None:
    '''The SiLU and the product of the two branches are BF16 in the dense MLP, the
    shared expert and the routed experts. The mixture casts the FP32 gates to BF16,
    and the sum over the eight experts and the addition of the shared expert are
    BF16.'''
    arithmetic = (ops.Arithmetic, ops.Einops, ops.AdditionOp)
    for name in (quantised_whole_model.DENSE_MLP_BOX, quantised_whole_model.MIXTURE_BOX):
        for node in operations_outside_boxes(box_body(name)):
            if not isinstance(node.operator, arithmetic) \
                    or isinstance(node.operator, ops.View):
                continue
            result = quantisation_of(node.output_weaves[0].datatype)
            require(result == BF16, f'{name} computes {node.operator} in {result}')
    casts = [(Quantization.format_name(quantisation_of(node.operator.source)),
              Quantization.format_name(quantisation_of(node.operator.target)))
             for node in operations_outside_boxes(
                 box_body(quantised_whole_model.MIXTURE_BOX))
             if isinstance(node.operator, Quantization.TypeConvert)]
    require(sorted(casts) == [('BF16', ROUNDED)] * 4 + [('FP32', 'BF16')],
            f'the mixture casts {casts}')


def check_the_router_computes_and_returns_fp32() -> None:
    '''The router casts the hidden state to FP32 in front of the FP32 router weight,
    computes the gates in FP32 and returns them in FP32.'''
    body = box_body(quantised_whole_model.ROUTER_BOX)
    require(cast_pairs(body) == [('BF16', 'FP32')],
            f'the router casts {cast_pairs(body)}')
    returned = {quantisation_of(array.datatype)
                for array in box_named(quantised_whole_model.ROUTER_BOX).cod()}
    require(returned == {FP32}, f'the router returns {returned}')


def check_the_selection_is_held_in_int32_on_the_tape() -> None:
    '''A Full layer publishing its selection casts the positions from INT64 to INT32
    before the drop. A Shared layer grabs them in INT32 and casts them to INT64 before
    the gathers, which read the positions through `.long()`.'''
    for drop in tutil.type_search(Para.Drop, MODEL):
        require(quantisation_of(drop.size.datatype) == INT32,
                f'the drop holds {quantisation_of(drop.size.datatype)}')
    for grab in tutil.type_search(Para.Grab, MODEL):
        require(quantisation_of(grab.size.datatype) == INT32,
                f'the grab holds {quantisation_of(grab.size.datatype)}')
    for name, pair in (('Full', ('INT64', 'INT32')), ('Shared', ('INT32', 'INT64'))):
        pairs = cast_pairs(box_body(name))
        require(pairs.count(pair) == 1, f'the {name} body casts {pairs}')
    for gather in operations(MODEL, ops.BlockOperator):
        if gather.operator.name.to_bodies() != quantised_whole_model.GATHER_BOX:
            continue
        positions = quantisation_of(gather.input_weaves[0].datatype)
        require(positions == INT64, f'a gather reads its positions in {positions}')


def check_the_model_reads_int64_and_returns_bf16_logits() -> None:
    '''The model reads the token identifiers in INT64 and returns the logits in BF16,
    the embedding returns BF16, and every layer reads and returns the BF16
    residual.'''
    tokens, = MODEL.dom()
    logits, = MODEL.cod()
    require(quantisation_of(tokens.datatype) == INT64 and
            quantisation_of(logits.datatype) == BF16,
            f'the model reads {tokens.datatype} and returns {logits.datatype}')
    for block in tutil.type_search(cat.Block, MODEL):
        if block.block_tag.aesthetics is None or \
                block.block_tag.aesthetics.title != text.RESIDUAL_TITLE:
            continue
        ends = {quantisation_of(array.datatype) for array in (*block.dom(), *block.cod())}
        require(ends == {BF16}, f'a residual reads and returns {ends}')


# ==========================================================================
# Stripping the quantisations.
# ==========================================================================
STRIPPED_BOXES = ('Full', 'Shared', 'Idx', 'Sco', 'Core', 'Gth', 'Rot', 'MLP', 'MoE',
                  'Gate')


def check_stripping_the_quantisations_returns_the_unquantised_model() -> None:
    '''Taking every quantisation off the quantised model leaves no quantisation and no
    conversion, and leaves the listing of `whole_model.glm53`, for the model and for
    the body of every box.'''
    require(not tuple(tutil.type_search(Quantization.Quantified, STRIPPED)),
            'the stripped model carries a quantisation')
    require(not quantise_model.conversions_of(STRIPPED),
            'the stripped model holds a conversion')
    require(ad.listing(h2m.recycle(STRIPPED)) == ad.listing(h2m.recycle(UNQUANTISED)),
            'the stripped model differs from the unquantised model')
    differing = [name for name in STRIPPED_BOXES
                 if ad.listing(h2m.recycle(box_body(name, STRIPPED)))
                 != ad.listing(h2m.recycle(box_body(name, UNQUANTISED)))]
    require(not differing, f'the bodies of {differing} differ')


# ==========================================================================
# The inspection boxes.
# ==========================================================================
def check_every_cast_and_every_weight_opens_a_box() -> None:
    '''Every cast of the model has a row, a rounding in front of an FP8 projection
    takes the row stating the rounding, and every weight has a role naming its
    quantisation.'''
    explain = quantised_whole_model.QUANTISED_OPERATOR_EXPLANATIONS[
        Quantization.TypeConvert]
    for cast in quantise_model.casts_of(MODEL):
        row = explain(cast)
        require(row is not None, f'the cast {cast.operator} has no row')
        if quantisation_of(cast.operator.target) == FP8_ROUNDED_OPERAND:
            require(row.formula == quantised_whole_model.ROUNDING_KERNEL_FORMULA,
                    'a rounding takes the row of another cast')
    roles = quantised_whole_model.quantised_operator_roles()
    missing = sorted(set(quantised_whole_model.WEIGHT_QUANTISATIONS) - set(roles))
    require(not missing, f'the weights {missing} have no role')
    for name, quantisation in quantised_whole_model.WEIGHT_QUANTISATIONS.items():
        sentence = text.WEIGHT_QUANTISATION_SENTENCE.format(
            quantisation=Quantization.describe_quantisation(quantisation))
        require(sentence in roles[name].role, f'the role of {name} omits {sentence}')
    require(set(operator_explanations.OPERATOR_EXPLANATIONS)
            <= set(quantised_whole_model.QUANTISED_OPERATOR_EXPLANATIONS),
            'the quantised tables lose a row of the model')


CHECKS: tuple[Callable[[], None], ...] = (
    check_every_weight_has_the_quantisation_of_the_checkpoint,
    check_every_wire_carries_a_quantisation,
    check_every_wire_of_the_page_carries_a_quantisation,
    check_an_fp8_projection_reads_a_rounded_operand,
    check_the_casts_of_the_model,
    check_the_rotary_turn_is_computed_in_bf16,
    check_the_indexer_scores_in_fp32,
    check_the_attention_core_is_the_fused_kernel,
    check_the_feed_forward_maps_apply_their_arithmetic_in_bf16,
    check_the_router_computes_and_returns_fp32,
    check_the_selection_is_held_in_int32_on_the_tape,
    check_the_model_reads_int64_and_returns_bf16_logits,
    check_stripping_the_quantisations_returns_the_unquantised_model,
    check_every_cast_and_every_weight_opens_a_box,
)


def report_each_check() -> int:
    '''Every check, with one line printed for each and a line of totals, returning how
    many of them failed.'''
    started = time.perf_counter()
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'ok      {check.__name__}')
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures += 1
            print(f'FAILED  {check.__name__}: {type(error).__name__}: {error}')
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} quantised GLM-5.3 checks passed '
          f'in {time.perf_counter() - started:.0f} s')
    return failures


def check_every_claim_of_the_notebook() -> None:
    '''Every check, raising when one of them fails, so that a notebook cell running
    the checks fails with it.'''
    failures = report_each_check()
    if failures:
        raise ClaimDoesNotHold(
            f'{failures} of {len(CHECKS)} claims of the notebook do not hold')


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
