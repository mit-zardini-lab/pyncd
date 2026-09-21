'''Check every claim the package makes about the integrated DeepSeek-V4.1-Flash and
about the text-only model restricted from it.

Written by Claude Fable 5.1, reasoning effort 80. The docstring was rewritten by Claude
Opus 5 (1M context), reasoning effort high, on 2026-09-20, when the claims of
`notebooks/sota/DeepSeekV41Flash.ipynb` moved into
`notebooks/sota/DeepSeekV41Flash/validate_quantised_text_only_model.py` beside this
file.

    python notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash_integrated.py

No notebook draws the integrated model, so the claims about it stand here alone, one
`check_` function per subject, in the shape of
`notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash.py`. The script prints one
line per check and the time the run took, and it exits non-zero on a failure.

Every check is structural, because `torch_compile` evaluates neither a `dst.Rotary`, a
`Quantization.TypeConvert` nor an `ops.GenericOperator`. The axes, the degrees, the
reindexings, the weaves, the slots and the wiring are compared. Two elementwise
formulas, the Engram gate and the GELU of the image projector, are evaluated at sample
values against the released lines. A uid is random per process, so no listing is diffed.

The order of the mechanisms inside an attention mode is read off the mode's hypergraph.
A wire is followed from the operation that reads it back through every producer that
has one operand, and the names of those producers are compared with the released order.

`united_explanation_tables` unites the tables of
`notebooks.sota.DeepSeekV41Flash.operator_explanations` with the rows the part modules
export. `integrated_explanations` replaces that union with its own tables once it is
written, and the whole-model display check then reads them from there.
'''
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation  # noqa: E402
import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import algebra.registries.standard_expansions as standard_expansions  # noqa: E402
import algebra.write_axis_exponents as write_axis_exponents  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import data_transfer.broadcast_occurrences as broadcast_occurrences  # noqa: E402
import data_transfer.term_json as term_json  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import graphs.data_structure.Hypergraph as hg  # noqa: E402
import graphs.processing.Hypergraph2Morphism as h2m  # noqa: E402
import para.algebra.para_sparse_expansion as para_sparse_expansion  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import para.data_structure.ParaBlockOperator as ParaBlockOperator  # noqa: E402
import para.data_structure.ParaWrap as para_wrap  # noqa: E402
import para.data_structure.inject as inject  # noqa: E402
import para.processing.show_grabbed_parameters as show_grabbed_parameters  # noqa: E402
import para.processing.tape_members as tape_members  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import term_utilities.generate_config as gc  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
import torch_compile.torch_compile as torch_compile  # noqa: E402
import websocket_transfer.send_morphism as send_morphism  # noqa: E402
from para.data_structure.ParaBlockOperator import (  # noqa: E402
    bare_box_of, slots_dropped, slots_grabbed)
from websocket_transfer.auxiliary_information import OperatorRole  # noqa: E402

import notebooks.display.axis_sizes as axis_sizes  # noqa: E402
import notebooks.display.explain_operators as explain_operators  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.tape_naming as tape_naming  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.attention_core as attention_core  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.attention_modes as attention_modes  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.candidate_pool as candidate_pool  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.divided_layer_stack as divided_layer_stack  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.dspark_draft_chain as dspark_draft_chain  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.dspark_drafter as dspark_drafter  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.engram_modules as engram_modules  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.grouped_output as grouped_output  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.gumbel_max_sampler as gumbel_max_sampler  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.integrated_attention_modes as integrated_attention_modes  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots as integrated_axes_and_slots  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.integrated_explanations as integrated_explanations  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.integrated_whole_model as integrated_whole_model  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.layer_stack as layer_stack  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.lightning_indexer as lightning_indexer  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.mhc_with_epsilons as mhc_with_epsilons  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.operator_explanations as operator_explanations  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.pinned_candidate_pool as pinned_candidate_pool  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.quantised_caches as quantised_caches  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.reference_links as reference_links  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.rotary_embedding as rotary_embedding  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.rotated_indexer as rotated_indexer  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.scaled_attention_core as scaled_attention_core  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.single_pass_mhc as single_pass_mhc  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.tempered_mixture as tempered_mixture  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.text_only_engram as text_only_engram  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.text_only_mixture as text_only_mixture  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.text_only_model as text_only_model  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.token_compressors as token_compressors  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.vision_pathway as vision_pathway  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.whole_model as whole_model  # noqa: E402
from notebooks.display.explain_operators import (  # noqa: E402
    ExplanationOfOperator, OperatorExplanation)
from notebooks.display.explain_reindexings import ReindexingExplanation  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (  # noqa: E402
    axes, axis_name, boxed, hold, node_with_box_named, node_with_operator, over)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (  # noqa: E402
    COLLAPSE, GROUP_COUNTER, GROUP_COUNTER_NAME, R, SLOT_CKVd, SLOT_CKVe, SLOT_KId,
    SLOT_POOL, SLOT_SELd, SLOT_SELe, X, B, a, b, c, d, h, i, m, n, state, x)
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (  # noqa: E402
    COMPRESSED_IDS, DRAFT_STATE, DRAFT_STATES, LOGITS, TOKEN_IDS, VOCABULARY, G, H, K,
    M, S, U, V, W, Z, Hp, L_reach, Wp)
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import (  # noqa: E402
    ENGRAM_GATE_FLOOR, MHC_EPSILON, MODALITY, NORM_EPSILON, ROUTER_EPSILON,
    ROUTER_TEMPERATURE, SLOT_IDS, SLOT_MODALITY, SLOT_STREAM_MEAN, SWIGLU_LIMIT)
from notebooks.sota.DeepSeekV41Flash.rotary_embedding import RotaryKind  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

MODEL = integrated_whole_model.v41_flash_integrated
TEXT_ONLY = text_only_model.v41_flash_text_only
RELEASED_SIZES = integrated_whole_model.RELEASED_SIZES
RUNTIME_SLOTS = {SLOT_CKVe, SLOT_SELe, SLOT_CKVd, SLOT_KId, SLOT_POOL, SLOT_SELd}
INTEGRATED_SLOTS = RUNTIME_SLOTS | {SLOT_IDS, SLOT_MODALITY, SLOT_STREAM_MEAN}
TEXT_ONLY_SLOTS = RUNTIME_SLOTS | {SLOT_IDS}
TILED = cat.WeaveMode.TILED
REUSE_COUNTER = divided_layer_stack.REUSE_COUNTER
REUSE_COUNTER_NAME = divided_layer_stack.REUSE_COUNTER_NAME
OMITTED_MECHANISM_AXES = (
    omitted_mechanisms.L, G, K, omitted_mechanisms.D, S,
    Z, U, V, omitted_mechanisms.F, M, Hp, Wp, H, W,
    omitted_mechanisms.E, omitted_mechanisms.t, omitted_mechanisms.z,
    omitted_mechanisms.zbar)


class ClaimDoesNotHold(AssertionError):
    '''A claim about the integrated model that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


# ==========================================================================
# Reading a term.
# ==========================================================================
def operations_of(term: fd.GeneralTerm) -> list[cat.Broadcasted]:
    return list(tutil.type_search(cat.Broadcasted, term))


def operators_of(term: fd.GeneralTerm) -> list[cat.Operator]:
    return [node.operator for node in operations_of(term)]


def operations_with[O: cat.Operator](
    kind: type[O], term: fd.GeneralTerm,
) -> list[cat.Broadcasted]:
    return [node for node in operations_of(term) if isinstance(node.operator, kind)]


def count_of[O: cat.Operator](kind: type[O], term: fd.GeneralTerm) -> int:
    return len(operations_with(kind, term))


def generic_operator_names(term: fd.GeneralTerm) -> set[str]:
    return {operator.name.to_bodies() for operator in operators_of(term)
            if isinstance(operator, ops.GenericOperator)}


def operator_names(kind: type[cat.Operator], term: fd.GeneralTerm) -> list[str]:
    return sorted(node.operator.name.to_bodies()
                  for node in operations_with(kind, term))


def distinct_blocks(term: fd.GeneralTerm) -> list[cat.Block]:
    by_tag = {block.block_tag.uid._id: block
              for block in tutil.type_search(cat.Block, term)}
    return list(by_tag.values())


def block_titles(term: fd.GeneralTerm) -> set[str]:
    return {block.block_tag.aesthetics.title for block in distinct_blocks(term)
            if block.block_tag.aesthetics is not None}


def sparse_axis_names(term: fd.GeneralTerm) -> set[str]:
    return {axis_name(axis)
            for axis in tutil.type_search(AffineGuards.AffineSparseAxis, term)}


def axis_bodies(term: fd.GeneralTerm) -> set[str]:
    return {axis.uid._name.body for axis in tutil.type_search(cat.RawAxis, term)
            if axis.uid._name is not None}


def arithmetic_formulas(term: fd.GeneralTerm) -> list[nm.Numeric]:
    return [node.operator.formula for node in operations_with(ops.Arithmetic, term)]


def formulas_holding(symbol: nm.Numeric, term: fd.GeneralTerm) -> list[nm.Numeric]:
    return [formula for formula in arithmetic_formulas(term)
            if symbol in tuple(tutil.type_search(type(symbol), formula))]


def holds_symbol(formula: nm.Numeric, symbol: nm.FreeNumeric) -> bool:
    return any(True for _ in tutil.search(formula, lambda term: term == symbol))


def value_at(formula: nm.Numeric, values: dict[nm.Numeric, nm.Numeric],
             inputs: torch.Tensor) -> torch.Tensor:
    '''`formula` at `inputs`, with each named constant replaced by its value.'''
    for symbol, value in values.items():
        formula = nm.substitute_subterm(formula, symbol, value)
    return torch_compile.numeric_torch(formula, inputs)


def power_of_ten(exponent: int) -> nm.Numeric:
    return nm.Power.template(nm.Integer(10), nm.Integer(exponent))


def require_pinned(references: fd.Prod[cat.CodeReference], owner: str) -> None:
    '''Every reference of `owner` is a link under the released commit with a line.'''
    require(bool(references), f'{owner} carries no reference')
    for reference in references:
        require(reference.url is not None
                and reference.url.startswith(reference_links.BASE_URL),
                f'{owner} links {reference.url}, which is not pinned')
        require(reference.line is not None, f'{owner} links {reference.label} with no line')


def require_explained(block: cat.Block) -> None:
    '''`block` carries a title, a description in sentences without the word the
    repository rules out, and pinned references with lines.'''
    aesthetics = block.block_tag.aesthetics
    require(aesthetics is not None and bool(aesthetics.title), 'a block has no title')
    require(bool(aesthetics.description) and aesthetics.description.endswith('.'),
            f'{aesthetics.title} carries no description in sentences')
    require('boundary' not in aesthetics.description.lower(),
            f'{aesthetics.title} uses a word the repository rules out')
    require_pinned(aesthetics.references, aesthetics.title)


def require_filled(block: cat.Block) -> None:
    aesthetics = block.block_tag.aesthetics
    require(aesthetics.fill_color not in (None, 'white'),
            f'{aesthetics.title} carries no fill colour')


# ==========================================================================
# Reading the wiring of a term off its hypergraph.
# ==========================================================================
def label_of(root: hg.HypergraphRoot) -> str:
    '''A short text naming the operation a root of a hypergraph wraps: a box by its
    short name and the title of its block, a weight or a view by its name, a tape seed
    by its slot, and any other operation by its class.'''
    wrapped = root.wraps
    if isinstance(wrapped, (Para.Grab, Para.Drop)):
        return f'{type(wrapped).__name__} {wrapped.tape.uid._name.to_latex()}'
    operator = wrapped.operator
    if isinstance(operator, ops.BlockOperator):
        return f'{operator.name.to_bodies()}: {operator.block.block_tag.aesthetics.title}'
    if isinstance(operator, (ops.Linear, ops.View)) and operator.name is not None:
        return f'{type(operator).__name__} {operator.name.to_bodies()}'
    return type(operator).__name__


def box_label(short_name: str, title: str) -> str:
    return f'{short_name}: {title}'


class ModeWiring:
    '''The roots of one term's hypergraph with its blocks removed, and the root that
    produces each wire.'''

    def __init__(self, mode: cat.Morphism) -> None:
        self.roots: tuple[hg.HypergraphRoot, ...] = hg.flat_subgraphs(
            hg.Multigraph.from_morphism(mode), remove_blocks=True)
        self.producer_of: dict[hg.HypergraphObject, hg.HypergraphRoot] = {
            wire: root for root in self.roots for wire in root.cod}

    def root_labelled(self, label: str) -> hg.HypergraphRoot:
        root, = (root for root in self.roots if label_of(root) == label)
        return root

    def producers_back_from(self, wire: hg.HypergraphObject) -> list[str]:
        '''The labels of the producers of `wire`, nearest first, followed back while
        each producer has one operand, and ending with the first producer that has
        several or none.'''
        labels: list[str] = []
        while wire in self.producer_of:
            root = self.producer_of[wire]
            labels.append(label_of(root))
            if len(root.dom) != 1:
                break
            wire, = root.dom
        return labels


def operator_label(wrapped: cat.Morphism) -> str:
    '''The text of the operator's name, and the class of a morphism that has none.'''
    operator = getattr(wrapped, 'operator', None)
    name = getattr(operator, 'name', None)
    if name is not None:
        return name.to_bodies()
    return type(wrapped if operator is None else operator).__name__


def roots_of(graph: hg.Hypergraph) -> Iterator[hg.HypergraphRoot]:
    '''Every leaf of `graph`, with the body of a block entered.'''
    if isinstance(graph, hg.HypergraphRoot):
        yield graph
    for subgraph in graph.subgraphs():
        if subgraph is not graph:
            yield from roots_of(subgraph)


def occurrences_of[O: cat.Operator](
    kind: type[O], term: cat.Morphism,
) -> list[cat.Broadcasted]:
    '''Every occurrence of an operator of `kind` in the graph of `term`. `type_search`
    returns two structurally equal nodes once, and the graph holds one leaf per
    occurrence.'''
    return [root.wraps for root in roots_of(hg.Multigraph.from_morphism(term))
            if isinstance(root.wraps, cat.Broadcasted)
            and isinstance(root.wraps.operator, kind)]


def feeds(term: cat.Morphism) -> set[tuple[str, str]]:
    '''Each pair of a producer and a consumer of one wire of `term`, by the labels of
    their operators. A box is one leaf, so the wires inside a box are read from the
    block of the box.'''
    roots = tuple(roots_of(hg.Multigraph.from_morphism(term)))
    producer = {wire: root for root in roots for wire in root.cod}
    return {(operator_label(producer[wire].wraps), operator_label(root.wraps))
            for root in roots for wire in root.dom if wire in producer}


# ==========================================================================
# The claims.
# ==========================================================================
MODEL_INPUTS = [['x'], ['x'], ["H'", "W'"], ['H', 'W', 'F'], ['\\Delta'], ['\\Delta']]
MODEL_OUTPUTS = [['x', 'v']] + [[]] * 10
TEXT_ONLY_INPUTS = [['x']]
TEXT_ONLY_OUTPUTS = [['x', 'v']]
TITLES_OF_THE_IMAGE_AND_THE_DRAFTS = {
    text.PATHWAY_TITLE, text.ENCODER_TITLE,
    text.PROJECTOR_TITLE, text.CELL_WRITE_TITLE,
    text.DELIMITER_WRITE_TITLE,
    text.DSPARK_TITLE, text.TAP_TITLE,
    text.BACKBONE_STATE_TITLE, text.CHAIN_TITLE,
    text.STEP_TITLE, text.HEAD_TITLE,
    text.MARKOV_TITLE, text.SAMPLER_TITLE,
    text.CONFIDENCE_TITLE,
    text.FOURTH_REINDEX_GROUP_TITLE,
    text.TAPPED_REUSE_LAYER_TITLE}
SYMBOLIC_AXES = {'x', 'H', 'W', "H'", "W'", '\\Delta'}
DERIVED_AXES = {'1', 'B', 'GK', 'P', 'V', 'b', 'z', '\\bar{d}', '\\bar{z}'}
BASE_MODES = {
    'SWA': (integrated_attention_modes.SWA, layer_stack.SWA),
    'FULLE': (integrated_attention_modes.FULLE, layer_stack.FULLE),
    'FULLD': (integrated_attention_modes.FULLD, layer_stack.FULLD),
    'REINDEX': (integrated_attention_modes.REINDEX, layer_stack.REINDEX),
    'REUSEE': (integrated_attention_modes.REUSEE, layer_stack.REUSEE),
    'REUSED': (integrated_attention_modes.REUSED, layer_stack.REUSED),
    'REUSED_IN_GROUP': (integrated_attention_modes.REUSED_IN_GROUP,
                        layer_stack.REUSED_IN_GROUP),
}
ROTATION_BOXES = (
    *rotary_embedding.ROTATE_TOKEN_LATENTS.values(),
    *rotary_embedding.ROTATE_ATTENTION_OUTPUT_BACK.values(),
    rotary_embedding.ROTATE_ENCODER_ENTRIES, rotary_embedding.ROTATE_DECODER_ENTRIES,
    rotary_embedding.ROTATE_INDEXER_QUERIES, rotary_embedding.ROTATE_ENCODER_INDEXER_KEYS,
    rotary_embedding.ROTATE_DECODER_INDEXER_KEYS)


def check_the_model_reads_six_arrays_and_returns_eleven() -> None:
    require(axes(MODEL) == (MODEL_INPUTS, MODEL_OUTPUTS), f'the model is {axes(MODEL)}')
    require(axes(MODEL)[1][0] == [axis_name(axis) for axis in dspark_draft_chain.PROBABILITIES.shape()],
            'the first result is not the probabilities of the next token')


def check_the_forty_layers() -> None:
    count = divided_layer_stack.layer_count(divided_layer_stack.divided_stack)
    require(count == divided_layer_stack.LAYERS_OF_THE_RELEASED_MODEL,
            f'the divided stack runs {count} layers')
    titles = block_titles(divided_layer_stack.divided_stack)
    for title in (text.THIRD_ENCODER_GROUP_TITLE,
                  text.FOURTH_REINDEX_GROUP_TITLE,
                  text.TAPPED_REUSE_LAYER_TITLE):
        require(title in titles, f'{title} is not a block of the divided stack')


def check_every_mode_touches_the_slots_of_the_base_mode() -> None:
    for name, (mode, base) in BASE_MODES.items():
        require(tuple(mode.dom()) == tuple(mode.cod()) == (state,),
                f'{name} does not read and return the hidden state')
        require(slots_dropped(mode) == slots_dropped(base)
                and slots_grabbed(mode) == slots_grabbed(base),
                f'{name} touches other slots than the base mode')
        require(count_of(ops.GenericOperator, mode)
                == len([node for node in operations_with(ops.GenericOperator, mode)
                        if node.operator.name.to_bodies() == quantised_caches.CEILING]),
                f'{name} holds a generic operator other than the ceiling')


def check_the_slots_of_the_whole_model() -> None:
    written = slots_dropped(MODEL)
    read = slots_grabbed(MODEL)
    require(written == INTEGRATED_SLOTS, f'the model writes {written}')
    require(read == INTEGRATED_SLOTS, f'the model reads {read}')
    constant_members = {(loop.tape, loop.index) for loop in tutil.type_search(Para.LoopGrab, MODEL)
                        if isinstance(loop.index, nm.Integer)}
    require({index for _, index in constant_members}
            >= {divided_layer_stack.THIRD_ENCODER_GROUP,
                divided_layer_stack.FOURTH_REINDEX_GROUP},
            f'the groups written alone read the members {constant_members}')


def check_the_generic_operators_that_remain() -> None:
    names = generic_operator_names(MODEL)
    explained = set(integrated_explanations.GENERIC_OPERATOR_EXPLANATIONS)
    require(names == explained, f'generic operators {names}, explained {explained}')
    print('   ', sorted(names))


def check_the_rotary_tables() -> None:
    require(count_of(dst.ComplexRotary, MODEL) == 0, 'the model holds a ComplexRotary')
    kinds = {type(node.operator) for node in operations_with(dst.Rotary, MODEL)}
    require(kinds == {dst.Rotary, dst.YarnRotary}, f'the rotary tables are {kinds}')
    conjugates = [node for node in operations_with(ops.Arithmetic, MODEL)
                  if node.operator.formula == nm.Conjugate(nm.x)]
    require(len(conjugates) > 0, 'the model turns nothing back')
    for node in conjugates:
        require(isinstance(node.output_weaves[0].datatype, dst.Complex),
                'a conjugate is applied to something other than a complex number')
    for box in ROTATION_BOXES:
        require(tuple(box.dom()) == tuple(box.cod()), 'a rotation box changes its wire')
        require(standard_expansions.expansion_for(
            node_with_operator(dst.Rotary, box.operator.block).operator) is not None,
            'a rotary table has no standard expansion')


def check_every_block_is_explained_and_referenced() -> None:
    blocks = distinct_blocks(MODEL)
    for block in blocks:
        require_explained(block)
        require_filled(block)
    print('   ', len(blocks), 'blocks')


def check_the_model_recycles_and_transports() -> None:
    recycled = h2m.recycle(MODEL)
    require(axes(recycled) == axes(MODEL), 'recycling changes the ends of the model')
    transported = send_morphism.to_morphism(MODEL)
    exported = term_json.TermJSONConverter.export_to_json(transported)
    require(len(exported) > 100_000, f'the export holds {len(exported)} characters')
    print('   ', f'{len(exported) / 1e6:.1f} MB exported')


def check_the_released_sizes_assign_every_axis() -> None:
    assigned = integrated_whole_model.released_assigned_sizes(MODEL)
    symbolic = axis_bodies(MODEL) - set(assigned) - DERIVED_AXES
    require(symbolic <= SYMBOLIC_AXES, f'unassigned axes {sorted(symbolic)}')
    for name, size in (('c', 512), ('t', 32), ('m', 5120), ('S', 5), ('e', 384)):
        require(assigned.get(name) == size, f'{name} is assigned {assigned.get(name)}')
    print('   ', len(assigned), 'sizes assigned, symbolic', sorted(symbolic))


def check_the_interactive_display() -> None:
    settings = integrated_explanations.with_explanation_tables(notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE))
    coverage = integrated_explanations.list_names_missing_a_row()
    require(coverage.is_complete(), f'the tables leave names without a row: {coverage}')
    sent, _, auxiliary = notebook_diagrams.package_auxiliary(
        notebook_diagrams.present_each_side(MODEL, settings), settings)
    in_place = [block for block in tutil.type_search(cat.Block, sent)
                if block.block_tag.aesthetics is not None
                and block.block_tag.aesthetics.drawing is cat.BlockDrawing.BODY_IN_PLACE
                and isinstance(block.body, cat.Broadcasted)]
    wrapped = {id(block.body) for block in in_place}
    for node in tutil.type_search(cat.Broadcasted, sent):
        if isinstance(node.operator, ops.GenericOperator):
            require(id(node) in wrapped,
                    f'{node.operator.name.to_latex()} opens no inspection box')
        if isinstance(node.operator, dst.Rotary):
            require(id(node) not in wrapped, 'a rotary table is wrapped and loses its expansion')
    hidden = {block.body.operator.name.to_bodies() for block in in_place
              if isinstance(block.body.operator, ops.Arithmetic)}
    require(hidden == set(integrated_explanations.ARITHMETIC_ROLES),
            f'hidden maps {sorted(hidden)}')
    numbered = broadcast_occurrences.number_broadcasts_in_import_order(sent)
    for number, node in enumerate(numbered):
        if isinstance(node.operator, (ops.Linear, ops.Normalize, dst.Rotary)):
            expansion = auxiliary['expansions'].get(str(number))
            require(expansion is not None, f'{node.operator.name.to_latex()} is not written out')
            role = integrated_explanations.OPERATOR_ROLES.get(node.operator.name.to_bodies())
            require(role is not None and expansion['description'].startswith(role.role),
                    f'{node.operator.name.to_latex()} does not say its role first')
    for record in (*auxiliary['blocks'].values(), *auxiliary['expansions'].values()):
        for reference in record['references']:
            require(reference['icon'] == 'huggingface',
                    f'{record.get("title", record.get("operator"))} links {reference}')
    print('   ', len(in_place), 'operators drawn in place,', len(auxiliary['blocks']),
          'block records,', len(auxiliary['expansions']), 'expansions')


def check_the_sparse_expansion_leaves_no_sparse_axis() -> None:
    expanded = para_sparse_expansion.expand_sparse_onto_tape(MODEL)
    require(not list(tutil.type_search(dst.SparseAxis, expanded)),
            'a sparse axis survives the expansion onto the tape')
    require(axes(expanded) == axes(MODEL), 'the expansion changes the ends of the model')


def check_the_text_only_model_reads_and_returns_one_array() -> None:
    require(axes(TEXT_ONLY) == (TEXT_ONLY_INPUTS, TEXT_ONLY_OUTPUTS),
            f'the text-only model is {axes(TEXT_ONLY)}')
    require(count_of(ops.SoftMax, TEXT_ONLY) >= 1,
            'the text-only model ends in no softmax over the vocabulary')


def check_the_text_only_model_runs_forty_layers() -> None:
    count = divided_layer_stack.layer_count(text_only_model.text_only_stack)
    require(count == divided_layer_stack.LAYERS_OF_THE_RELEASED_MODEL,
            f'the text-only stack runs {count} layers')
    titles = block_titles(text_only_model.text_only_stack)
    require(text.THIRD_ENCODER_GROUP_TITLE in titles,
            'the third encoder group is not written alone')
    for title in (text.FOURTH_REINDEX_GROUP_TITLE,
                  text.TAPPED_REUSE_LAYER_TITLE):
        require(title not in titles,
                f'{title} stands in a stack that has no DSpark to read it')


def check_the_text_only_model_holds_no_image_and_no_draft() -> None:
    require(generic_operator_names(TEXT_ONLY)
            == set(integrated_explanations.TEXT_ONLY_GENERIC_OPERATOR_EXPLANATIONS),
            f'generic operators {sorted(generic_operator_names(TEXT_ONLY))}')
    for absent in (vision_pathway.VISION_ENCODER_NAME, dspark_draft_chain.DRAFT_NAME,
                   gumbel_max_sampler.DRAW_NAME):
        require(absent not in generic_operator_names(TEXT_ONLY),
                f'{absent} stands in the text-only model')
    require(not [node for node in tutil.type_search(cat.Natural, TEXT_ONLY)
                 if node == MODALITY],
            'the text-only model reads the modality of a token')
    require(not list(tutil.type_search(inject.Inject, TEXT_ONLY)),
            'the text-only model injects at a token position')


def check_the_text_only_model_keeps_every_mechanism_of_text() -> None:
    written = slots_dropped(TEXT_ONLY)
    read = slots_grabbed(TEXT_ONLY)
    require(written == TEXT_ONLY_SLOTS, f'the text-only model writes {written}')
    require(read == TEXT_ONLY_SLOTS, f'the text-only model reads {read}')
    kinds = {type(node.operator) for node in operations_with(dst.Rotary, TEXT_ONLY)}
    require(kinds == {dst.Rotary, dst.YarnRotary},
            f'the rotary tables of the text-only model are {kinds}')
    require(count_of(Quantization.TypeConvert, TEXT_ONLY) > 0,
            'the text-only model rounds no cached array')
    titles = block_titles(TEXT_ONLY)
    for layer in omitted_mechanisms.ENGRAM_LAYERS:
        require(f'\\text{{Engram of Layer {layer}}}' in titles,
                f'the text-only model holds no Engram of layer {layer}')
    require(text_only_mixture.MIXTURE_CONFIRMATION.is_confirmed(),
            text_only_mixture.MIXTURE_CONFIRMATION.named_difference)


def names_carrying_two_scripts(term: fd.GeneralTerm, sizes: dict[str, int]) -> list[str]:
    '''The labels of `term` that would be drawn with two subscripts once the sizes are
    written onto the names, which is what a body holding its own subscript gives.'''
    sized = axis_sizes.present(term, axis_sizes.AxisSizes.SUBSCRIPT, sizes)
    return sorted(
        name.to_latex() for name in tutil.type_search(fd.DynamicName, sized)
        if name.exponent is not None and name.body is not None and '_' in name.body)


def check_no_size_is_written_onto_a_name_holding_its_own_script() -> None:
    '''A name whose body holds its own subscript takes a second one beside it once a
    size is written onto it. Two subscripts are not LaTeX, so KaTeX draws the label as
    its own source in red, colour command and all. The row count of an Engram table
    drew that way on the page until 2026-09-18. A symbol that is assigned a size
    carries its subscript as a `fd.DynamicName`, and the subscript and the size then
    join after a colon.'''
    for label, term, sizes in (
            ('integrated', MODEL, RELEASED_SIZES),
            ('text-only', TEXT_ONLY, text_only_model.RELEASED_SIZES)):
        clashing = names_carrying_two_scripts(term, sizes)
        require(not clashing, f'the {label} model draws {clashing}')


def check_the_text_only_model_is_the_integrated_model_restricted() -> None:
    '''Every block of the text-only model stands in the integrated model under the same
    title, and the blocks the integrated model holds beside them are the image pathway,
    DSpark and the two blocks that stand alone only because DSpark taps them.'''
    text_only = block_titles(TEXT_ONLY)
    integrated = block_titles(MODEL)
    require(text_only <= integrated,
            f'the text-only model holds blocks of its own: '
            f'{sorted(text_only - integrated)}')
    require(integrated - text_only == TITLES_OF_THE_IMAGE_AND_THE_DRAFTS,
            f'the blocks left out are {sorted(integrated - text_only)}')


def check_the_text_only_blocks_are_explained_and_referenced() -> None:
    blocks = distinct_blocks(TEXT_ONLY)
    for block in blocks:
        require_explained(block)
        require_filled(block)
    print('   ', len(blocks), 'blocks')


def check_the_text_only_model_recycles_and_transports() -> None:
    recycled = h2m.recycle(TEXT_ONLY)
    require(axes(recycled) == axes(TEXT_ONLY),
            'recycling changes the ends of the text-only model')
    exported = term_json.TermJSONConverter.export_to_json(
        send_morphism.to_morphism(TEXT_ONLY))
    require(len(exported) > 100_000, f'the export holds {len(exported)} characters')
    print('   ', f'{len(exported) / 1e6:.1f} MB exported')


def check_the_text_only_released_sizes_assign_every_axis() -> None:
    assigned = text_only_model.released_assigned_sizes()
    symbolic = axis_bodies(TEXT_ONLY) - set(assigned) - DERIVED_AXES
    require(symbolic <= {'x'}, f'unassigned axes {sorted(symbolic)}')
    print('   ', len(assigned), 'sizes assigned, symbolic', sorted(symbolic))


def check_the_text_only_interactive_display() -> None:
    coverage = integrated_explanations.list_text_only_names_missing_a_row()
    require(coverage.is_complete(), f'the tables leave names without a row: {coverage}')
    settings = integrated_explanations.with_text_only_explanation_tables(
        notebook_diagrams.DiagramSettings(
            mode=notebook_diagrams.DiagramMode.OFF,
            advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE))
    sent, _, auxiliary = notebook_diagrams.package_auxiliary(
        notebook_diagrams.present_each_side(TEXT_ONLY, settings), settings)
    for node in tutil.type_search(cat.Broadcasted, sent):
        if isinstance(node.operator, ops.GenericOperator):
            require(integrated_explanations.explain_text_only_generic_operator(node)
                    is not None,
                    f'{node.operator.name.to_latex()} opens no inspection box')
    print('   ', len(auxiliary['blocks']), 'block records,',
          len(auxiliary['expansions']), 'expansions')


def check_the_base_package_validators_pass() -> None:
    root = pathlib.Path(__file__).resolve().parents[3]
    for script in ('notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash.py',
                   'notebooks/sota/DeepSeekV41Flash/validate_omitted_mechanisms.py',
                   'deepseek/validate_rotary.py'):
        completed = subprocess.run(
            [sys.executable, str(root / script)], cwd=root, capture_output=True, text=True,
            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONPATH': str(root)})
        require(completed.returncode == 0, f'{script} fails:\n{completed.stdout[-2000:]}')


def engram_operator_counts(module: fd.GeneralTerm) -> dict[type[cat.Operator], int]:
    '''How many of each integer operator of the written-out hash `module` holds.'''
    return {kind: count_of(kind, module)
            for kind in (ops.FixedArray, ops.BitwiseXor, ops.Modulo, ops.Cast)}


def check_the_engram_modules_of_both_layers() -> None:
    '''Each of the two Engram modules of each model writes its hash out over the
    integer operators and holds no generic operator. Every fixed array, table and
    learned array carries the number of its layer, so the two modules share none, and
    each module addresses a table whose row count is a symbol of its own. The
    multimodal module reads the modality and the text-only module does not.'''
    first, second = omitted_mechanisms.ENGRAM_LAYERS
    token_map = f'E{omitted_mechanisms.TOKEN_MAP_NAME}'
    for label, build in (('integrated', engram_modules.engram_block),
                         ('text-only', text_only_engram.engram_block)):
        names_of_layer = {}
        for layer in omitted_mechanisms.ENGRAM_LAYERS:
            module = build(layer)
            require(not generic_operator_names(module),
                    f'the {label} Engram of layer {layer} holds a generic operator')
            require(engram_operator_counts(module) == {
                        ops.FixedArray: 2, ops.BitwiseXor: 1,
                        ops.Modulo: 1, ops.Cast: 1},
                    f'the {label} Engram of layer {layer} writes its hash out over '
                    f'{engram_operator_counts(module)}')
            rows = omitted_mechanisms.table_row_of_layer(layer)
            require(rows in set(tutil.type_search(cat.Natural, module)),
                    f'the {label} Engram of layer {layer} addresses no table of its '
                    f'own')
            names_of_layer[layer] = {
                node.operator.name.to_bodies()
                for node in (operations_with(ops.Linear, module)
                             + operations_with(ops.FixedArray, module)
                             + operations_with(ops.Embedding, module))
                if node.operator.name.to_bodies() != token_map}
            for name in names_of_layer[layer]:
                require(name.endswith(f'{{{layer}}}'),
                        f'{name} of the {label} Engram carries no layer')
        require(not names_of_layer[first] & names_of_layer[second],
                f'the two {label} Engram modules share '
                f'{sorted(names_of_layer[first] & names_of_layer[second])}')
    multimodal = set(tutil.type_search(cat.Natural, engram_modules.engram_block(first)))
    text_only = set(tutil.type_search(cat.Natural, text_only_engram.engram_block(first)))
    require(MODALITY in multimodal, 'the multimodal Engram reads no modality')
    require(MODALITY not in text_only,
            'the text-only Engram reads the modality of a token')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_model_reads_six_arrays_and_returns_eleven,
    check_the_engram_modules_of_both_layers,
    check_the_forty_layers,
    check_every_mode_touches_the_slots_of_the_base_mode,
    check_the_slots_of_the_whole_model,
    check_the_generic_operators_that_remain,
    check_the_rotary_tables,
    check_every_block_is_explained_and_referenced,
    check_the_model_recycles_and_transports,
    check_the_released_sizes_assign_every_axis,
    check_the_interactive_display,
    check_the_sparse_expansion_leaves_no_sparse_axis,
    check_the_text_only_model_reads_and_returns_one_array,
    check_the_text_only_model_runs_forty_layers,
    check_the_text_only_model_holds_no_image_and_no_draft,
    check_the_text_only_model_keeps_every_mechanism_of_text,
    check_no_size_is_written_onto_a_name_holding_its_own_script,
    check_the_text_only_model_is_the_integrated_model_restricted,
    check_the_text_only_blocks_are_explained_and_referenced,
    check_the_text_only_model_recycles_and_transports,
    check_the_text_only_released_sizes_assign_every_axis,
    check_the_text_only_interactive_display,
    check_the_base_package_validators_pass,
)


def main() -> int:
    started = time.perf_counter()
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'ok      {check.__name__}')
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures += 1
            print(f'FAILED  {check.__name__}: {type(error).__name__}: {error}')
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} integrated model checks passed '
          f'in {time.perf_counter() - started:.0f} s')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
