# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim `notebooks/website/modern/DeepSeekV41Flash.ipynb` makes about
the text-only DeepSeek-V4.1-Flash.

    python notebooks/website/modern/validate_deepseek_v41_flash.py

The notebook is public and holds the prose and the figures alone. Its claims live here,
one `check_` function per claim, in the order the notebook makes them, in the shape of
`notebooks/sota/GLM53/validate_glm53.py`. The script prints one line per check and the
time the run took, and it exits non-zero on a failure. A check of
`notebooks/sota/DeepSeekV41Flash/validate_quantised_text_only_model.py` that states a
claim of the notebook is listed in `CHECKS` as it stands.

The notebook draws every figure and both variants of its page in the CausalSlide form,
which `slide_causal_reads_backwards` returns, so the claims are checked on that form of
each model. The notebook draws the parts of the model in the reals and a part of the
quantised model, and its last cell writes a page with two variants. The last checks
compare the quantised model taken through `strip_quantisations`, which is the Python
statement of the functor a page would apply in the browser to derive its unquantised
variant, with the model in the reals. They are the evidence for drawing the unquantised
variant from a term of its own.
'''
from __future__ import annotations

import collections
import pathlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa: E402
import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa: E402
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import agent_display as ad  # noqa: E402
import caching.data_structure.Caching as Caching  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import graphs.data_structure.Hypergraph as hg  # noqa: E402
import graphs.processing.Hypergraph2Morphism as h2m  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import quantization.algebra.strip_quantisations as strip_quantisations  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
from para.data_structure.ParaBlockOperator import (  # noqa: E402
    slots_dropped, slots_grabbed)

import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.integrated_explanations as integrated_explanations  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.quantised_text_only_model as quantised_text_only_model  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.released_constants as released_constants  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.text_only_model as text_only_model  # noqa: E402
import notebooks.sota.DeepSeekV41Flash.validate_quantised_text_only_model as validate_quantised_text_only_model  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axes, axis_name  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.declared_axes import (  # noqa: E402
    COLLAPSE, GROUP_COUNTER, SLOT_CKVd, SLOT_CKVe, SLOT_KId, SLOT_POOL, SLOT_SELd,
    SLOT_SELe, B, P, X, a, b, c, w, x)
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import SLOT_IDS  # noqa: E402

BUILT_UNQUANTISED: cat.Morphism = text_only_model.v41_flash_text_only
BUILT_QUANTISED: cat.Morphism = quantised_text_only_model.v41_flash_text_only_quantised
UNQUANTISED: cat.Morphism = slide_causal_reads_backwards.slide_causal_reads_backwards(
    BUILT_UNQUANTISED)
QUANTISED: cat.Morphism = slide_causal_reads_backwards.slide_causal_reads_backwards(
    BUILT_QUANTISED)
STRIPPED: cat.Morphism = strip_quantisations.strip_quantisations(QUANTISED)
ASSIGNED_SIZES: dict[str, int] = text_only_model.released_assigned_sizes()

CEILING = '\\lceil x \\rceil'
ROUND_TRIP_BOXES = ('FP4e', 'FP4i', 'FP8w')
ATTENTION_BOXES = ('SWA', 'Full', 'Rex', 'Reu')
MIXTURE_BOX = 'MoE'

RELEASED_ATTENTION_PLAN: tuple[str, ...] = (
    ('SWA',) * 2
    + (('Full',) + ('Reu',) * 5) * 3
    + ('Full',) + ('Reu',) * 3
    + (('Rex',) + ('Reu',) * 3) * 4)
'''The attention mode of each of the forty layers, as `kv_source_layer_ids` and
`index_source_layer_ids` of the released `inference/config.json` give them.'''

RELEASED_SIZE_TABLE: dict[str, int] = {
    'v': 129280, 'm': 5120, 'n': 4, 'h': 64, 'c': 512, 'q': 1280, 'o': 1024, 'g': 8,
    'j': 8, 'w': 128, 'a': 2, 'u': 8, 'p': 2048, 'C': 16384, 's': 512, 'k': 6,
    'i': 32, 'd': 128, 'e': 384, 'f': 2304, 't': 32, 'L': 4, 'G': 3, 'K': 8, 'D': 256,
    'E': 32, 'y': 16, '\\hat{E}': 4, '\\tilde{E}': 16, '\\hat{y}': 32}
'''The integer sizes of the table of axes in the notebook.'''

WEIGHT_QUANTISATION_STEM = text.WEIGHT_QUANTISATION_SENTENCE.split('{')[0]

i_x, i_b, i_r, j_w = (nm.FreeNumeric.named(name) for name in ('i_x', 'i_b', 'i_r', 'j_w'))


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def size_of(axis: str) -> int:
    return ASSIGNED_SIZES[axis]


def released_size(axis: cat.Axis) -> int:
    '''The released size of `axis`, whose size may be a sum or a product of the sizes
    the configuration assigns.'''
    size = axis.local_size()
    return nm.evaluate_integer(size, {
        symbol: ASSIGNED_SIZES[symbol.uid._name.to_bodies()]
        for symbol in nm.free_symbols(size)})


def boxes(name: str, model: cat.Morphism = UNQUANTISED) -> tuple[cat.Broadcasted, ...]:
    return quantised_text_only_model.boxes_named(name, model)


def body_of(box: cat.Broadcasted) -> cat.Morphism:
    return box.operator.block.body


def require_sites(name: str, count: int) -> None:
    '''The model in the reals and the quantised model, each in the CausalSlide, compose
    the box `name` at `count` sites.'''
    for label, model in (('unquantised', UNQUANTISED), ('quantised', QUANTISED)):
        sites = len(boxes(name, model))
        require(sites == count, f'the {label} figure composes {name} at {sites} sites')


def first_body(name: str, model: cat.Morphism = UNQUANTISED) -> cat.Morphism:
    return body_of(boxes(name, model)[0])


def operations(term: fd.GeneralTerm) -> tuple[cat.Broadcasted, ...]:
    return tuple(tutil.type_search(cat.Broadcasted, term))


def operations_with[O: cat.Operator](
        kind: type[O], term: fd.GeneralTerm) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in operations(term) if isinstance(node.operator, kind))


def operator_name(node: cat.Morphism) -> str | None:
    '''The name of the operator of `node`, and `None` where `node` has no operator, as
    a grab or a drop of the tape has none.'''
    name = getattr(getattr(node, 'operator', None), 'name', None)
    return None if name is None else name.to_bodies()


def names_of[O: cat.Operator](kind: type[O], term: fd.GeneralTerm) -> list[str]:
    return [operator_name(node) for node in operations_with(kind, term)]


def shape_names(array: cat.Array) -> list[str]:
    return [axis_name(axis) for axis in array.shape()]


def linear_named(name: str, term: fd.GeneralTerm) -> cat.Broadcasted:
    found = [node for node in operations_with(ops.Linear, term)
             if operator_name(node) == name]
    require(len(found) >= 1, f'no Linear named {name}')
    return found[0]


def view_named(name: str, term: fd.GeneralTerm) -> cat.Broadcasted:
    found = [node for node in operations_with(ops.View, term)
             if operator_name(node) == name]
    require(len(found) >= 1, f'no view named {name}')
    return found[0]


def covariant_view_named(name: str, term: fd.GeneralTerm) -> cat.Broadcasted:
    found = [node for node in operations_with(aops.CovariantView, term)
             if operator_name(node) == name]
    require(len(found) >= 1, f'no covariant view named {name}')
    return found[0]


def blocks_titled(title: str, term: fd.GeneralTerm) -> tuple[cat.Block, ...]:
    return tuple(block for block in tutil.type_search(cat.Block, term)
                 if block.block_tag.aesthetics is not None
                 and block.block_tag.aesthetics.title == title)


def box_publishing(name: str, slot: Para.TapeSlot) -> cat.Broadcasted:
    '''The first box named `name` whose body drops `slot`.'''
    found = [box for box in boxes(name) if slot in slots_dropped(body_of(box))]
    require(len(found) >= 1, f'no box named {name} drops {slot}')
    return found[0]


def window_axis() -> AffineGuards.AffineSparseAxis:
    window = view_named('\\mathrm{Window}', first_body('SWA'))
    axis = window.cod()[0].shape()[1]
    require(isinstance(axis, AffineGuards.AffineSparseAxis),
            f'the window view returns {shape_names(window.cod()[0])}')
    return axis


def box_name_of(term: object) -> str | None:
    match term:
        case cat.Broadcasted(operator=ops.BlockOperator(name=name)) if name is not None:
            return name.to_bodies()
    wrapped = getattr(term, 'body', None)
    if isinstance(wrapped, cat.Broadcasted):
        return box_name_of(wrapped)
    return None


def sublayer_sequence(term: object) -> list[str]:
    '''The attention box or the mixture box of every sublayer `term` runs, in order,
    with every repeated block written out as many times as it repeats.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return [kind for part in parts for kind in sublayer_sequence(part)]
        case cat.Block(body=body, block_tag=block_tag):
            return sublayer_sequence(body) * block_tag.repetition._value
    name = box_name_of(term)
    return [name] if name in (*ATTENTION_BOXES, MIXTURE_BOX) else []


def graph_and_roots(term: cat.Morphism) -> tuple[hg.Hypergraph, tuple[hg.HypergraphRoot, ...]]:
    '''The hypergraph of `term` and its operations, grabs and drops outside the body of
    any box, on the wires of that one hypergraph, so that two of them compare their
    wires.'''
    graph = hg.Multigraph.from_morphism(term)
    return graph, tuple(root for root in hg.flat_subgraphs(graph, remove_blocks=True)
                        if isinstance(root, hg.HypergraphRoot))


def roots_of(term: cat.Morphism) -> tuple[hg.HypergraphRoot, ...]:
    return graph_and_roots(term)[1]


def readers_of(wire: hg.HypergraphObject, roots: tuple[hg.HypergraphRoot, ...]
               ) -> tuple[hg.HypergraphRoot, ...]:
    return tuple(root for root in roots if wire in root.dom)


def root_with[O: cat.Operator](
        kind: type[O], roots: tuple[hg.HypergraphRoot, ...]) -> hg.HypergraphRoot:
    found = [root for root in roots if isinstance(root.wraps, cat.Broadcasted)
             and isinstance(root.wraps.operator, kind)]
    require(len(found) == 1, f'{len(found)} operations of kind {kind.__name__}')
    return found[0]


def root_named(name: str, roots: tuple[hg.HypergraphRoot, ...]) -> hg.HypergraphRoot:
    found = [root for root in roots if operator_name(root.wraps) == name]
    require(len(found) == 1, f'{len(found)} operations named {name}')
    return found[0]


def top_level_operations_with[O: cat.Operator](
        kind: type[O], term: cat.Morphism) -> tuple[cat.Broadcasted, ...]:
    return tuple(root.wraps for root in roots_of(term)
                 if isinstance(root.wraps, cat.Broadcasted)
                 and isinstance(root.wraps.operator, kind))


def page_auxiliary(term: cat.Morphism, settings: notebook_diagrams.DiagramSettings
                   ) -> dict:
    presented = notebook_diagrams.present_each_side(term, settings)
    _, _, auxiliary = notebook_diagrams.package_auxiliary(presented, settings)
    return auxiliary


QUANTISED_PAGE_SETTINGS = validate_quantised_text_only_model.PAGE_SETTINGS
UNQUANTISED_PAGE_SETTINGS = integrated_explanations.with_text_only_explanation_tables(
    notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        tape=QUANTISED_PAGE_SETTINGS.tape,
        axis_sizes=QUANTISED_PAGE_SETTINGS.axis_sizes,
        assigned_sizes=ASSIGNED_SIZES,
        block_recycling=QUANTISED_PAGE_SETTINGS.block_recycling,
        sub_blocks=QUANTISED_PAGE_SETTINGS.sub_blocks,
        advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE,
        expanded_parameters=QUANTISED_PAGE_SETTINGS.expanded_parameters,
        clean_quantisation_labels=False))
'''The display fields under which the notebook writes the unquantised variant.'''

DECODE = notebook_diagrams.PageVariantGroup('decode', 'Decode')
PAGE_VARIANTS: tuple[notebook_diagrams.PageVariant, ...] = (
    notebook_diagrams.PageVariant(
        identifier='decode-quantised', group=DECODE, title='Quantised',
        detail='every array at the number format of the released inference code',
        term=QUANTISED, settings=QUANTISED_PAGE_SETTINGS),
    notebook_diagrams.PageVariant(
        identifier='decode-unquantised', group=DECODE, title='Unquantised',
        detail='the same arithmetic in the reals, keeping the rounding of the three caches',
        term=UNQUANTISED, settings=UNQUANTISED_PAGE_SETTINGS),
)
'''The two variants of the page, as the last cell of the notebook declares them.'''


def sent_term(term: cat.Morphism, settings: notebook_diagrams.DiagramSettings
              ) -> cat.Morphism:
    presented = notebook_diagrams.present_each_side(term, settings)
    sent, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
    return sent


def size_is_one_named_symbol(axis: cat.Axis) -> bool:
    size = axis.local_size()
    return isinstance(size, nm.FreeNumeric) and size.uid._name is not None


# ==========================================================================
# The model, the figures and the restriction to text.
# ==========================================================================
def check_both_variants_read_identifiers_and_return_probabilities() -> None:
    '''The model at its released quantisations and the model in the reals each read
    the identifier of every token and return one probability per token and vocabulary
    entry.'''
    for label, model in (('quantised', QUANTISED), ('unquantised', UNQUANTISED)):
        require(axes(model) == ([['x']], [['x', 'v']]),
                f'the {label} model reads and returns {axes(model)}')
        tokens, = model.dom()
        require(Quantization.holds_natural_numbers(tokens.datatype),
                f'the {label} model reads {tokens.datatype}')


def check_the_released_sizes_of_the_axes() -> None:
    '''Every integer of the table of axes is the released size of its axis, the token
    axis is sized |a| |b|, the decoder's entries are as many as the tokens, the rotated
    channels are twice the pairs, and the encoder's entries and the blocks are sized by
    free symbols no configuration binds.'''
    differing = {name: (size, ASSIGNED_SIZES.get(name))
                 for name, size in RELEASED_SIZE_TABLE.items()
                 if ASSIGNED_SIZES.get(name) != size}
    require(not differing, f'the table and the release differ at {differing}')
    require(x.local_size() == a.local_size() * b.local_size(),
            f'x is sized {x.local_size()}')
    require(B.local_size() == x.local_size(), f'B is sized {B.local_size()}')
    require('b' not in ASSIGNED_SIZES and 'P' not in ASSIGNED_SIZES,
            'the configuration binds the entries or the blocks')
    require(isinstance(P.local_size(), nm.FreeNumeric), f'P is sized {P.local_size()}')
    turned = rotation_parts()[1]
    require(released_size(turned) == 2 * size_of('t'),
            f'{released_size(turned)} channels are turned as {size_of("t")} pairs')


def check_the_entries_counted_back_range_over_the_entries() -> None:
    '''An entry counted back from a query's newest reachable entry ranges over the
    entries of its own half of the stack.'''
    for name, entries in (('Full', b), ('Full', B)):
        found = [view for box in boxes(name) for view in operations_with(
                     ops.View, body_of(box))
                 if operator_name(view) == '\\mathrm{Back}'
                 and view.dom()[0].shape()[0] == entries]
        require(found, f'no back view reads {axis_name(entries)}')
        counted_back = found[0].cod()[0].shape()[1]
        require(counted_back.local_size() == entries.local_size(),
                f'{axis_name(counted_back)} is sized {counted_back.local_size()}')


def check_every_slot_written_is_read() -> None:
    '''Every tape slot the model writes it reads.'''
    validate_quantised_text_only_model.check_every_tape_slot_written_in_the_model_is_read_in_it()


def check_the_causal_slide_keeps_the_ends_and_the_layers() -> None:
    '''The CausalSlide of each variant reads and returns what the model as built reads
    and returns, and its top level, which holds the layers, is the same, so every read
    it moves stands inside a box.'''
    for label, built, slid in (('quantised', BUILT_QUANTISED, QUANTISED),
                               ('unquantised', BUILT_UNQUANTISED, UNQUANTISED)):
        require(axes(built) == axes(slid), f'the slide changes the ends of the {label} model')
        require(ad.listing(h2m.recycle(built)) == ad.listing(h2m.recycle(slid)),
                f'the slide changes the top level of the {label} model')
        moved = [name for name in ('SWA', 'Full', 'Rex', 'Reu', 'Idx', 'Gth', 'Eng{1}')
                 if {ad.listing_without_legend(h2m.recycle(body_of(box)))
                     for box in boxes(name, built)}
                 != {ad.listing_without_legend(h2m.recycle(body_of(box)))
                     for box in boxes(name, slid)}]
        require(moved, f'the slide moves no read of the {label} model')


def check_every_box_tag_of_both_variants_holds_one_body() -> None:
    '''A figure draws the body of a box beside the first figure holding it and a page
    opens it over the box, both by the tag of its block, so every tag of either variant
    holds one body. A body the CausalSlide rebuilt at one site of a box carries a tag of
    its own.'''
    for label, model in (('quantised', QUANTISED), ('unquantised', UNQUANTISED)):
        bodies: dict[int, set[str]] = collections.defaultdict(set)
        for node in operations_with(ops.BlockOperator, model):
            bodies[node.operator.block.block_tag.uid._id].add(
                ad.listing_without_legend(h2m.recycle(node.operator.block.body)))
        shared = sorted(tag for tag, held in bodies.items() if len(held) > 1)
        require(not shared, f'{len(shared)} tags of the {label} variant hold several bodies')


def check_every_legend_row_of_both_variants_names_its_axis_in_code() -> None:
    '''Both variants draw a legend, every row of it carries the name of its axis in
    code, and a row whose size is one named symbol carries the name of that size.'''
    for variant in PAGE_VARIANTS:
        require(variant.settings.advanced_display in (
                    notebook_diagrams.AdvancedDisplay.LEGEND,
                    notebook_diagrams.AdvancedDisplay.INTERACTIVE),
                f'{variant.identifier} draws no legend')
    legends = notebook_diagrams.page_variant_legends(PAGE_VARIANTS, QUANTISED_PAGE_SETTINGS)
    for variant in PAGE_VARIANTS:
        rows = legends[variant.identifier]
        require(rows, f'{variant.identifier} carries no legend')
        axes_by_uid = {axis.uid._id: axis for axis in tutil.type_search(
            cat.Axis, sent_term(variant.term, variant.settings))}
        unnamed = sorted(row['text'] for row in rows if not row['codeName'])
        require(not unnamed, f'{variant.identifier} names no code for {unnamed}')
        unsized = sorted(row['text'] for row in rows if not row['sizeCodeName']
                         and any(size_is_one_named_symbol(axes_by_uid[uid])
                                 for uid in row['uids'] if uid in axes_by_uid))
        require(not unsized, f'{variant.identifier} names no code for the size of {unsized}')


def check_the_window_slots_hold_a_value_where_the_token_is_not_later() -> None:
    '''The window slot `j_w` of query `i_x` holds a value where `i_x - j_w` is not
    negative, and the query axis is the axis that decides it.'''
    axis = window_axis()
    require(axis.guides == (x,), f'the window slots are guided by {axis.guides}')
    require(axis.guard_form((i_x,), j_w) == nm.collect_like_terms(i_x - j_w),
            f'the window slots hold a value where {axis.guard_form((i_x,), j_w)}')


def check_the_model_holds_no_image_pathway_and_no_draft() -> None:
    '''The model holds no generic operator of the image pathway or of DSpark, reads no
    modality and writes no slot for one.'''
    generic = set(names_of(ops.GenericOperator, UNQUANTISED))
    require(generic == {CEILING}, f'the model holds the generic operators {generic}')
    require(released_constants.MODALITY not in set(
                tutil.type_search(cat.Natural, UNQUANTISED)),
            'the model reads the modality of a token')
    require(len(slots_dropped(UNQUANTISED)) == validate_quantised_text_only_model.TAPE_SLOTS,
            f'the model writes {len(slots_dropped(UNQUANTISED))} slots')


def check_the_router_adds_one_correction_bias() -> None:
    '''The router of the mixture adds one correction bias.'''
    biases = [name for name in names_of(ops.Linear, first_body('Gate'))
              if name == '\\mathrm{bias}']
    require(len(biases) == 1, f'the router adds {len(biases)} biases')


def check_the_pass_keeps_no_cache() -> None:
    '''Neither variant holds a cache carried from one call to the next.'''
    for label, model in (('quantised', QUANTISED), ('unquantised', UNQUANTISED)):
        cached = operations_with(Caching.Caching, model)
        require(not cached, f'the {label} model caches {len(cached)} arrays')


def check_the_three_caches_are_three_round_trips() -> None:
    '''The three caches of the released code stand in the model as three round trips,
    each of which rounds.'''
    for name in ROUND_TRIP_BOXES:
        require(boxes(name), f'the model holds no {name} box')
        casts = operations_with(Quantization.TypeConvert, first_body(name))
        require(casts, f'the {name} round trip rounds nothing')


# ==========================================================================
# The embedding and the four residual streams.
# ==========================================================================
def check_the_embedding_table() -> None:
    '''The identifiers are looked up in a table of 129280 rows of 5120 channels, which
    gives one hidden state per token.'''
    lookup, = (node for node in operations_with(ops.Embedding, UNQUANTISED)
               if operator_name(node) == 'E')
    require(shape_names(lookup.cod()[0]) == ['x', 'm'],
            f'the lookup returns {shape_names(lookup.cod()[0])}')
    require(size_of('v') == 129280 and size_of('m') == 5120,
            f"v is {size_of('v')} and m is {size_of('m')}")


def check_the_hidden_state_is_repeated_into_four_streams() -> None:
    '''A view repeats the hidden state into four streams.'''
    head = quantised_text_only_model.part_before_the_layers(UNQUANTISED)
    repeat = view_named('\\mathrm{Repeat}', head)
    require(axes(repeat) == ([['x', 'm']], [['x', 'n', 'm']]),
            f'the repeat reads and returns {axes(repeat)}')
    require(size_of('n') == 4, f"n is {size_of('n')}")


def check_the_first_collapse_vector_is_a_covariant_view_with_no_operands() -> None:
    '''The collapse vector of the first sublayer is a covariant view with no operands
    that writes one position of the stream axis.'''
    head = quantised_text_only_model.part_before_the_layers(UNQUANTISED)
    collapse, = operations_with(aops.CovariantView, head)
    require(not tuple(collapse.dom()), f'the collapse vector reads {axes(collapse)[0]}')
    require(tuple(collapse.operator.reindexing.dom()) == (),
            'the reindexing of the collapse vector reads a position')
    require([axis_name(axis) for axis in collapse.operator.reindexing.cod()] == ['n'],
            'the collapse vector writes no stream')
    (_, strides, shift), = collapse.operator.reindexing._cod_stride_shift
    require(strides == () and shift == nm.Integer(0),
            f'the collapse vector writes stream {shift}')


def check_the_identifiers_reach_both_engram_modules_on_the_tape() -> None:
    '''The identifiers are dropped onto the slot ids, and both Engram modules grab
    it.'''
    require(SLOT_IDS in slots_dropped(body_of(boxes('In')[0])),
            'the inputs drop no identifiers')
    for name in ('Eng{1}', 'Eng{14}'):
        require(SLOT_IDS in slots_grabbed(body_of(boxes(name)[0])),
                f'{name} grabs no identifiers')


# ==========================================================================
# The mixing coefficients of the four streams.
# ==========================================================================
def check_the_coefficients_are_a_collapse_vector_an_output_vector_and_a_combine_matrix(
) -> None:
    '''The coefficient box returns four numbers, four numbers and four by four numbers
    for each token.'''
    coefficients = boxes('Coef')[0]
    require(axes(coefficients) == ([['x', 'n', 'm']],
                                   [['x', 'n'], ['x', 'n'], ['x', 'n', 'n']]),
            f'the coefficients read and return {axes(coefficients)}')


def check_every_sublayer_hands_the_collapse_vector_to_the_next() -> None:
    '''Every sublayer reads the collapse vector and the four streams and returns the
    same two, so the collapse vector predicted by one sublayer reaches the next.'''
    sublayers = blocks_titled(text.MHC_TITLE, UNQUANTISED)
    require(len(sublayers) > 0, 'the model holds no sublayer')
    for sublayer in sublayers:
        require(tuple(sublayer.dom()) == tuple(sublayer.cod()) == (COLLAPSE, X),
                f'a sublayer reads {axes(sublayer)[0]} and returns {axes(sublayer)[1]}')


def check_the_three_coefficient_maps_and_their_activations() -> None:
    '''Three linear maps read the normalised streams of a token, and no view slices a
    result. The collapse vector takes a sigmoid plus the epsilon, and the output vector
    takes twice a sigmoid.'''
    coefficients = first_body('Coef')
    for name, returned in (('H0', ['n']), ('H1', ['n']), ('H2', ['n', 'n'])):
        linear = linear_named(name, coefficients)
        require(shape_names(linear.dom()[0]) == ['n', 'm']
                and shape_names(linear.cod()[0]) == returned,
                f'{name} reads and returns {axes(linear)}')
    require(not operations_with(ops.View, coefficients), 'a view slices a result')
    arithmetic = names_of(ops.Arithmetic, coefficients)
    for formula in ('\\sigma', 'x + \\varepsilon_{\\mathrm{hc}}', '2\\sigma'):
        require(formula in arithmetic, f'the coefficients apply {arithmetic}')


def check_the_sinkhorn_rounds() -> None:
    '''The first Sinkhorn round takes the exponential, divides the rows, adds the
    epsilon and divides the columns, and the nineteen rounds after it are one block at
    repetition nineteen dividing the rows and the columns.'''
    coefficients = first_body('Coef')
    first, = blocks_titled(text.FIRST_ROUND_TITLE, coefficients)
    require([type(node.operator) for node in operations(first)]
            == [ops.Arithmetic, ops.L1Norm, ops.Arithmetic, ops.L1Norm],
            'the first round holds other operations')
    rounds, = (block for block in tutil.type_search(cat.Block, coefficients)
               if block.block_tag.repetition != nm.Integer(1))
    require(rounds.block_tag.repetition == nm.Integer(19),
            f'the later rounds repeat {rounds.block_tag.repetition} times')
    require([type(node.operator) for node in operations(rounds)]
            == [ops.L1Norm, ops.L1Norm], 'a later round holds other operations')


def check_the_coefficients_are_computed_once_per_token() -> None:
    '''The coefficient box is broadcast over the tokens, and its body holds no token
    axis.'''
    for box in boxes('Coef'):
        require(tuple(box.degree()) == (x,),
                f'the coefficients are broadcast over {box.degree()}')
        require('x' not in {axis_name(axis) for axis in
                            tutil.type_search(cat.Axis, body_of(box))},
                'the body of the coefficients holds the token axis')


# ==========================================================================
# The sliding-window attention.
# ==========================================================================
def check_the_two_sliding_window_layers() -> None:
    '''The stack begins with two sliding-window layers, each attending over a window of
    128 slots of a 512-wide latent.'''
    window_layers = blocks_titled(text.WINDOW_LAYER_TITLE, text_only_model.text_only_stack)
    require(len(window_layers) == 2, f'the stack holds {len(window_layers)} window layers')
    require(sublayer_sequence(text_only_model.text_only_stack)[:4:2] == ['SWA', 'SWA'],
            'the first two layers are not sliding-window layers')
    require(size_of('w') == 128 and size_of('c') == 512,
            f"w is {size_of('w')} and c is {size_of('c')}")


def check_the_model_as_built_reads_the_window_after_the_latent() -> None:
    '''As the model is built, the window latent is projected from the hidden state,
    normalised, turned and rounded through the FP8 cache, and the window view reads
    the result.'''
    graph, roots = graph_and_roots(first_body('SWA', BUILT_UNQUANTISED))
    window = root_named('\\mathrm{Window}', roots)
    rounding = root_named('FP8w', roots)
    require(window.dom[0] == rounding.cod[0], 'the window reads no rounded latent')
    key_value = root_named('W^{KV}', roots)
    require(key_value.dom[0] == graph.dom[0], 'the latent reads no hidden state')


def check_the_figures_read_the_window_at_the_hidden_state() -> None:
    '''In the CausalSlide the window view reads the hidden state the layer receives,
    and the projection, the normalisation, the rotation and the FP8 round trip of the
    latent run once per token and slot.'''
    graph, roots = graph_and_roots(first_body('SWA'))
    window = root_named('\\mathrm{Window}', roots)
    require(window.dom[0] == graph.dom[0], 'the window reads no hidden state')
    key_value = root_named('W^{KV}', roots)
    require(key_value.dom[0] == window.cod[0], 'the projection reads no window')
    rounding = root_named('FP8w', roots)
    require(shape_names(rounding.wraps.cod()[0]) == ['x', 'w|x', 'c'],
            f'the round trip returns {shape_names(rounding.wraps.cod()[0])}')
    require(len(readers_of(graph.dom[0], roots)) >= 2,
            'the window read stands at no copy of the hidden state')


def check_the_window_reads_no_later_token() -> None:
    '''Slot 0 of every query is the current token, the live slots read the tokens the
    released window row names, and no slot reads a later token.'''
    sizes = {a.local_size(): 2, b.local_size(): 150, w.local_size(): 128}
    axis = window_axis()
    for query in range(300):
        live = mark_sparse_domains.live_positions(axis, (query,), sizes)
        require(live == list(range(min(128, query + 1))),
                f'query {query} reads the slots {live}')


def check_one_latent_serves_every_query_head() -> None:
    '''The window latents the attention core reads carry no head axis, and the query is
    projected through the 1280-wide low rank onto 64 heads.'''
    window = first_body('SWA')
    core = top_level_operations_with(ops.BlockOperator, window)
    core, = (node for node in core if operator_name(node) == 'Core')
    require(shape_names(core.dom()[2]) == ['x', 'w|x', 'c'],
            f'the core reads the window latents as {shape_names(core.dom()[2])}')
    down = linear_named('W^{Qa}', window)
    up = linear_named('W^{Qb}', window)
    require(shape_names(down.cod()[0]) == ['x', 'q'],
            f'the low rank is {shape_names(down.cod()[0])}')
    require(shape_names(up.cod()[0]) == ['x', 'h', 'c'],
            f'the query is {shape_names(up.cod()[0])}')
    require(size_of('h') == 64 and size_of('q') == 1280,
            f"h is {size_of('h')} and q is {size_of('q')}")


def check_the_attention_core_is_computed_once_per_head() -> None:
    '''Every attention core is one box broadcast over the heads. It scales every score
    by the inverse square root of the latent width, exponentiates, and adds the
    exponential of the sink to the denominator alone.'''
    validate_quantised_text_only_model.check_the_nine_sites_composing_the_attention_core()
    require_sites('Core', 9)
    for box in boxes('Core'):
        require(tuple(box.degree()) == (tuple(box.degree())[0],)
                and axis_name(box.degree()[0]) == 'h',
                f'a core is broadcast over {box.degree()}')
    core = first_body('Core')
    arithmetic = names_of(ops.Arithmetic, core)
    require('x / \\sqrt{\\lvert c \\rvert}' in arithmetic and 'e^{x}' in arithmetic,
            f'the core applies {arithmetic}')
    sink, _, _ = core.dom()
    require(not tuple(sink.shape()), 'the sink is not one number per head')
    denominator, = operations_with(ops.AdditionOp, core)
    require(any(not tuple(operand.shape()) for operand in denominator.dom()),
            'the sink is not added to the denominator')
    require(len(operations_with(ops.AdditionOp, core)) == 1,
            'the sink is added to more than the denominator')


def check_the_grouped_output_projection() -> None:
    '''The 64 heads are cut into 8 groups of 8, each group's heads are mapped to a
    1024-wide rank of its own by a Linear followed by a diagonal view, and one final
    weight reads the eight ranks. The projection is computed once per token.'''
    projection = boxes('Out')[0]
    require(tuple(projection.degree()) == (x,),
            f'the projection is broadcast over {projection.degree()}')
    body = body_of(projection)
    group = view_named('\\mathrm{Group}', body)
    require(shape_names(group.cod()[0]) == ['g', 'j', 'c'],
            f'the heads are cut into {shape_names(group.cod()[0])}')
    per_group = linear_named('W^{Oa}', body)
    require(shape_names(per_group.cod()[0]) == ['g', 'g', 'o'],
            f'the first map returns {shape_names(per_group.cod()[0])}')
    diagonal = view_named('\\mathrm{Diagonal}', body)
    require(shape_names(diagonal.cod()[0]) == ['g', 'o'],
            f'the diagonal returns {shape_names(diagonal.cod()[0])}')
    final = linear_named('W^{Ob}', body)
    require(shape_names(final.dom()[0]) == ['g', 'o']
            and shape_names(final.cod()[0]) == ['m'],
            f'the final weight reads and returns {axes(final)}')
    require(size_of('g') * size_of('j') == size_of('h') and size_of('o') == 1024,
            f"g is {size_of('g')}, j is {size_of('j')} and o is {size_of('o')}")
    require(size_of('j') * size_of('c') == 4096, 'a group does not hold 4096 values')


# ==========================================================================
# The rotary embedding.
# ==========================================================================
def rotation_parts() -> tuple[cat.Axis, cat.Axis]:
    cut, = operations_with(aops.DeconcatenateAxes, first_body('Rot'))
    left_alone, turned = cut.operator.parts()
    return left_alone, turned


def check_the_rotation_turns_the_last_sixty_four_channels() -> None:
    '''A rotation leaves the first 448 of the 512 channels alone and turns the last
    64.'''
    left_alone, turned = rotation_parts()
    require((released_size(left_alone), released_size(turned)) == (448, 64),
            f'the rotation cuts {released_size(left_alone)} and {released_size(turned)}')


def check_the_rotation_body_holds_five_steps() -> None:
    '''The body of a rotation deconcatenates, pairs, multiplies by the table, unpairs
    and concatenates.'''
    kinds = [type(node.operator) for node in operations(first_body('Rot'))]
    require(kinds == [aops.DeconcatenateAxes, dst.PairsAsComplex, dst.Rotary,
                      ops.Einops, dst.Decomplex, aops.ConcatenateAxes],
            f'the rotation holds {[kind.__name__ for kind in kinds]}')


def rotations_of(mode: cat.Broadcasted) -> tuple[list[str], list[str]]:
    '''The names of the rotation boxes a mode holds outside any box and inside its
    indexer.'''
    outside = [operator_name(root.wraps) for root in roots_of(body_of(mode))
               if operator_name(root.wraps) in ('Rot', 'Rot^{-1}', 'Idx')]
    indexers = [root.wraps for root in roots_of(body_of(mode))
                if operator_name(root.wraps) == 'Idx']
    inside = [operator_name(root.wraps) for indexer in indexers
              for root in roots_of(body_of(indexer))
              if operator_name(root.wraps) in ('Rot', 'Rot^{-1}')]
    return outside, inside


def check_the_arrays_a_full_layer_turns_and_turns_back() -> None:
    '''A Full layer turns five arrays, four outside its indexer and the query inside it,
    and every attention mode turns its output back once, by the conjugate of the
    table.'''
    for box in boxes('Full'):
        outside, inside = rotations_of(box)
        require(outside.count('Rot') == 4 and inside == ['Rot'],
                f'a Full layer turns {outside.count("Rot")} arrays outside its indexer '
                f'and {inside} inside it')
    for name in ATTENTION_BOXES:
        for box in boxes(name):
            outside, _ = rotations_of(box)
            require(outside.count('Rot^{-1}') == 1,
                    f'a {name} layer turns its output back {outside.count("Rot^{-1}")} times')
    for box in boxes('Rot^{-1}'):
        conjugates = [node for node in operations_with(ops.Arithmetic, body_of(box))
                      if node.operator.formula == nm.Conjugate(nm.x)]
        require(len(conjugates) == 1, 'a site turns back without the conjugate')


def check_the_window_layers_turn_at_rope_and_every_other_layer_at_yarn() -> None:
    '''Every table of the sliding-window layers is RoPE, and every table of the other
    layers is YaRN.'''
    window_tables = {type(node.operator) for node in
                     operations_with(dst.Rotary, body_of(boxes('SWA')[0]))}
    require(window_tables == {dst.Rotary}, f'the window layers turn by {window_tables}')
    require({operator_name(node) for node in
             operations_with(dst.Rotary, body_of(boxes('SWA')[0]))} == {'\\mathrm{RoPE}'},
            'a table of the window layers is not named RoPE')
    for name in ('Full', 'Rex', 'Reu'):
        for box in boxes(name):
            tables = {type(node.operator) for node in
                      operations_with(dst.Rotary, body_of(box))}
            require(tables <= {dst.YarnRotary},
                    f'a {name} layer turns by {tables}')
            require({operator_name(node) for node in operations_with(
                         dst.Rotary, body_of(box))} <= {'\\mathrm{YaRN}'},
                    f'a table of a {name} layer is not named YaRN')


def check_a_slot_turns_at_the_position_of_its_own_token() -> None:
    '''Where the window read has moved into a rotation, the rotation reads its table
    through the same window view, so the latent of each slot turns at the position of
    its own token.'''
    turned_through_the_window = []
    for box in boxes('Rot'):
        roots = roots_of(body_of(box))
        windows = [root for root in roots if operator_name(root.wraps) == '\\mathrm{Window}']
        tables = [root for root in roots if isinstance(root.wraps, cat.Broadcasted)
                  and isinstance(root.wraps.operator, dst.Rotary)]
        if windows:
            require(len(tables) == 1 and windows[0].dom[0] == tables[0].cod[0],
                    'a rotation reads its window at no table')
            turned_through_the_window.append(box)
    require(turned_through_the_window, 'no rotation reads its table through the window')


def check_the_released_rotary_constants() -> None:
    '''The rotary bases, the YaRN factor and the ends of its ramp hold their released
    values.'''
    released = {constant.symbol: constant.released_value
                for constant in released_constants.RELEASED_CONSTANTS}
    expected = {released_constants.WINDOW_ROTARY_BASE: '10000',
                released_constants.ROTARY_BASE: '160000',
                released_constants.YARN_FACTOR: '16',
                released_constants.YARN_RAMP_START: '15',
                released_constants.YARN_RAMP_END: '25'}
    differing = {symbol.to_latex(): released.get(symbol)
                 for symbol, value in expected.items() if released.get(symbol) != value}
    require(not differing, f'the released constants are {differing}')


# ==========================================================================
# The token compressors.
# ==========================================================================
def compressor_returning(entries: cat.Axis) -> cat.Broadcasted:
    found = [box for box in boxes('Comp')
             if box.cod()[0].shape()[0] == entries]
    require(found, f'no compressor returns {axis_name(entries)}')
    return found[0]


def check_the_compressors_group_two_tokens_in_the_encoder_and_one_in_the_decoder() -> None:
    '''The encoder's compressor views the tokens in groups of two, and the decoder's
    reads one token per entry. Three sites compress.'''
    validate_quantised_text_only_model.check_the_three_sites_that_compress()
    require_sites('Comp', 3)
    encoder = body_of(compressor_returning(b))
    group = view_named('\\mathrm{Group}', encoder)
    require(axes(group) == ([['x', 'm']], [['b', 'a', 'm']]),
            f'the encoder groups {axes(group)}')
    require(size_of('a') == 2, f"a is {size_of('a')}")
    decoder = body_of(compressor_returning(B))
    position = view_named('\\mathrm{Position}', decoder)
    require(axes(position) == ([['x', 'm']], [['B', 'm']]),
            f'the decoder reads {axes(position)}')


def check_the_compressor_projections_softmax_and_normalisation() -> None:
    '''The encoder's compressor projects the hidden state into a value and a logit of
    512 channels, takes a softmax over the positions of a group, contracts and
    normalises. The decoder's compressor is the projection and the normalisation.'''
    encoder = body_of(compressor_returning(b))
    for name in ('W^{C}', 'W^{Z}'):
        projection = linear_named(name, encoder)
        require(shape_names(projection.cod()[0]) == ['b', 'a', 'c'],
                f'{name} returns {shape_names(projection.cod()[0])}')
    softmax, = operations_with(ops.SoftMax, encoder)
    require([axis_name(axis) for axis in softmax.input_weaves[0].target().shape()] == ['a'],
            'the softmax does not run over the positions of a group')
    require(len(operations_with(ops.Normalize, encoder)) == 1,
            'the encoder entry is not normalised')
    decoder = body_of(compressor_returning(B))
    require([type(node.operator) for node in operations(decoder)]
            == [ops.View, ops.Linear, ops.Normalize],
            'the decoder compressor holds more than the projection and the norm')


def check_the_decoder_entries_travel_on_the_tape() -> None:
    '''The first decoder layer drops the decoder's entries, and every Reindex and Reuse
    layer of the decoder grabs them.'''
    box_publishing('Full', SLOT_CKVd)
    readers = [box for name in ('Rex', 'Reu') for box in boxes(name)
               if SLOT_CKVd in slots_grabbed(body_of(box))]
    require(len(readers) == 3,
            f'{len(readers)} decoder modes grab the entries of the decoder')


# ==========================================================================
# The rounding of the three caches.
# ==========================================================================
def check_the_scale_groups_of_the_three_caches() -> None:
    '''A compressed entry is 32 groups of 16 channels, an indexer key or query is 4
    groups of 32, and a window latent is 16 groups of 32.'''
    validate_quantised_text_only_model.check_a_compressed_entry_is_thirty_two_groups()
    require(size_of('E') * size_of('y') == size_of('c'),
            'the groups of an entry do not cover its channels')
    require(size_of('\\hat{E}') * size_of('\\hat{y}') == size_of('d'),
            'the groups of an indexer key do not cover its channels')
    require(size_of('\\tilde{E}') * size_of('\\hat{y}') == size_of('c'),
            'the groups of a window latent do not cover its channels')


def cast_targets(term: fd.GeneralTerm) -> list[str]:
    return [Quantization.describe_quantisation(Quantization.quantisation_of(
                node.operator.target)).split(',')[0]
            if Quantization.quantisation_of(node.operator.target) is not None
            else 'reals'
            for node in operations_with(Quantization.TypeConvert, term)]


def check_each_round_trip_rounds_and_returns_the_array_it_reads() -> None:
    '''Each round trip reads an array and returns one of the same axes. It views the
    channels in groups, takes the largest magnitude of each group, clamps, casts into
    its stored format and back, and writes the groups back onto the channels.'''
    for name, stored in (('FP4e', 'E2M1'), ('FP4i', 'E2M1'), ('FP8w', 'E4M3')):
        body = first_body(name)
        require(axes(body)[0] == axes(body)[1], f'{name} reads and returns {axes(body)}')
        kinds = {type(node.operator) for node in operations(body)}
        require({ops.View, ops.Maximum, aops.CovariantView} <= kinds,
                f'{name} holds {sorted(kind.__name__ for kind in kinds)}')
        require(stored in cast_targets(body) and 'reals' in cast_targets(body),
                f'{name} casts into {cast_targets(body)}')


def check_the_scale_rule_of_each_round_trip() -> None:
    '''A compressed entry clamps at 6 and rounds its scale through E4M3 with a floor of
    2^-9. An indexer key or query clamps at 6 and rounds its scale up to a power of two
    with a floor of 2^-126. A window latent clamps at 448 and rounds its scale up to a
    power of two with a floor of 10^-4.'''
    rules = {
        'FP4e': ('x / 6', '\\max(x, 2^{-9})', '\\ulcorner x \\lrcorner_{-6}^{6}'),
        'FP4i': ('x / 6', '\\max(x, 2^{-126})', '\\log_{2} x', '2^{x}',
                 '\\ulcorner x \\lrcorner_{-6}^{6}'),
        'FP8w': ('\\max(x, 10^{-4})', 'x / 448', '\\log_{2} x', '2^{x}',
                 '\\ulcorner x \\lrcorner_{-448}^{448}'),
    }
    for name, formulas in rules.items():
        arithmetic = names_of(ops.Arithmetic, first_body(name))
        missing = [formula for formula in formulas if formula not in arithmetic]
        require(not missing, f'{name} applies {arithmetic} and lacks {missing}')
    require(cast_targets(first_body('FP4e')).count('E4M3') == 1,
            'the scale of an entry is not rounded through E4M3')
    for name in ('FP4i', 'FP8w'):
        require(names_of(ops.GenericOperator, first_body(name)) == [CEILING],
                f'{name} rounds its scale up without the ceiling')


def check_the_ceiling_is_the_only_generic_operator() -> None:
    '''The ceiling is the only operator of the model with no rule for its value.'''
    validate_quantised_text_only_model.check_the_ceiling_is_the_only_operator_without_a_rule()
    generic = set(names_of(ops.GenericOperator, UNQUANTISED))
    require(generic == {CEILING}, f'the model holds the generic operators {generic}')


def check_the_casts_of_the_model_in_the_reals_stand_in_the_round_trips() -> None:
    '''The casts inside the three round trips are the only conversions of the model in
    the reals.'''
    inside = sum(len(operations_with(Quantization.TypeConvert, first_body(name)))
                 for name in ROUND_TRIP_BOXES)
    everywhere = len(quantise_model.conversions_of(UNQUANTISED))
    require(inside == everywhere == 8,
            f'the model converts {everywhere} times, {inside} of them in the round trips')


# ==========================================================================
# The lightning indexer and the Top-512 selection.
# ==========================================================================
def check_the_indexer_heads_and_its_query() -> None:
    '''The indexer runs 32 heads of 128 channels, and its query is projected from the
    1280-wide low rank of the attention query.'''
    require(size_of('i') == 32 and size_of('d') == 128,
            f"i is {size_of('i')} and d is {size_of('d')}")
    for box in boxes('Idx'):
        require(shape_names(box.dom()[0]) == ['x', 'q'],
                f'the indexer reads {shape_names(box.dom()[0])} first')
        query = linear_named('q^{I}', body_of(box))
        require(axes(query) == ([['x', 'q']], [['x', 'i', 'd']]),
                f'the indexer query is {axes(query)}')
    roots = roots_of(body_of(box_publishing('Full', SLOT_CKVe)))
    indexer = root_named('Idx', roots)
    up = root_named('W^{Qb}', roots)
    require(indexer.dom[0] == up.dom[0],
            'the indexer and the attention query read different low ranks')


def check_the_indexer_keys_are_projected_from_the_entries() -> None:
    '''The indexer keys are projected from the compressed entries at the layer that
    compresses them, once per entry in the decoder and once per entry and distance in
    the encoder.'''
    for entries, slot, read in ((b, SLOT_CKVe, ['b', 'r|b']), (B, SLOT_CKVd, ['B'])):
        keys = linear_named('k^{I}', body_of(box_publishing('Full', slot)))
        require(axes(keys) == ([[*read, 'c']], [[*read, 'd']]),
                f'the keys of {axis_name(entries)} are {axes(keys)}')


def check_the_indexer_scores() -> None:
    '''Each head scores a query against an entry, each score is rectified, and the
    rectified scores are summed under a per-token per-head weight scaled by
    (|d| |i|)^-1/2. The scoring is computed once per token.'''
    for box in boxes('Sco'):
        require(tuple(box.degree()) == (x,),
                f'the scoring is broadcast over {box.degree()}')
    scores = first_body('Sco')
    arithmetic = names_of(ops.Arithmetic, scores)
    for formula in ('x \\mathbbm{1}_{x > 0}',
                    'x / \\sqrt{\\lvert d \\rvert \\lvert i \\rvert}'):
        require(formula in arithmetic, f'the scoring applies {arithmetic}')
    weight = linear_named('w^{I}', scores)
    require(axes(weight) == ([['m']], [['i']]), f'the head weight is {axes(weight)}')
    require(axes(scores)[1] == [['r|x']], f'the scoring returns {axes(scores)[1]}')


def encoder_reach() -> tuple[AffineGuards.AffineSparseAxis, AffineGuards.AffineSparseAxis]:
    full = body_of(box_publishing('Full', SLOT_CKVe))
    idx, = (node for node in operations_with(ops.BlockOperator, full)
            if operator_name(node) == 'Idx')
    back = view_named('\\mathrm{Back}', full).cod()[0].shape()[1]
    reach = covariant_view_named('\\mathrm{Position}', body_of(idx)).cod()[0].shape()[1]
    return back, reach


@dataclass(frozen=True)
class BackRead:
    '''One back view of an attention mode: the operation writing the array it reads,
    and every operation reading that array.'''
    producer: hg.HypergraphRoot
    readers: tuple[hg.HypergraphRoot, ...]


def back_reads_by_what_they_feed(full: cat.Broadcasted) -> dict[str, BackRead]:
    '''Each back view of `full`, keyed by the name of the operation it feeds.'''
    _, roots = graph_and_roots(body_of(full))
    back_reads = {}
    for back in (root for root in roots if operator_name(root.wraps) == '\\mathrm{Back}'):
        fed, = readers_of(back.cod[0], roots)
        producer, = (root for root in roots if back.dom[0] in root.cod)
        back_reads[operator_name(fed.wraps)] = BackRead(
            producer=producer, readers=readers_of(back.dom[0], roots))
    return back_reads


def check_the_back_reads_stand_at_the_copies_that_publish() -> None:
    '''In the CausalSlide the back view of the keys leaves the indexer box and the back
    view of the entries leaves the gather box. The back view of the entries stands
    where the entries are dropped onto the tape, and so does the back view of the keys
    in the decoder. In the encoder the keys are not dropped, and their back view stands
    at the compressed entries, which the rotation of the entries reads as well.'''
    for name in ('Idx', 'Gth'):
        for box in boxes(name):
            require(not [node for node in operations(body_of(box))
                         if operator_name(node) == '\\mathrm{Back}'],
                    f'a back view stands inside the {name} box')
    decoder = back_reads_by_what_they_feed(box_publishing('Full', SLOT_POOL))
    require(set(decoder) == {'Idx', 'Gth'}, f'the decoder reads back for {sorted(decoder)}')
    for fed, back_read in decoder.items():
        require(any(isinstance(reader.wraps, Para.Drop) for reader in back_read.readers),
                f'the back view feeding {fed} stands at no copy dropped onto the tape')
    encoder = back_reads_by_what_they_feed(box_publishing('Full', SLOT_CKVe))
    require(set(encoder) == {'k^{I}', 'Gth'}, f'the encoder reads back for {sorted(encoder)}')
    require(any(isinstance(reader.wraps, Para.Drop) for reader in encoder['Gth'].readers),
            'the back view of the encoder entries stands at no copy dropped onto the tape')
    require(operator_name(encoder['k^{I}'].producer.wraps) == 'Comp',
            'the back view of the encoder keys reads no compressed entries')


def check_a_query_reaches_the_entries_the_released_code_counts() -> None:
    '''The back view holds a key where `i_b - i_r` is not negative, the merge leaves the
    slots guided by the query where `i_x - |a| i_r - (|a| - 1)` is not negative, and
    query `i_x` reaches the `floor((i_x + 1) / |a|)` newest entries.'''
    back, reach = encoder_reach()
    require(back.guard_form((i_b,), i_r) == nm.collect_like_terms(i_b - i_r),
            f'the back view holds a key where {back.guard_form((i_b,), i_r)}')
    ratio = a.local_size()
    require(reach.guides == (x,), f'the reach is guided by {reach.guides}')
    require(reach.guard_form((i_x,), i_r) == nm.collect_like_terms(
                i_x - ratio * i_r - ratio + nm.Integer(1)),
            f'a query reaches the slots where {reach.guard_form((i_x,), i_r)}')
    sizes = {a.local_size(): 2, b.local_size(): 600}
    for query in (0, 1, 2, 3, 100, 511, 512, 1022, 1023, 1024, 1025, 1199):
        live = mark_sparse_domains.live_positions(reach, (query,), sizes)
        require(live == list(range((query + 1) // 2)),
                f'query {query} reaches {len(live)} entries')


def check_the_selection_returns_distances_and_the_gather_reads_entries() -> None:
    '''The Top-512 stands outside the indexer box, returns the distances of the chosen
    entries and no score, and the gather reads one entry per distance.'''
    full = body_of(box_publishing('Full', SLOT_CKVe))
    selection, = operations_with(dst.TopK, full)
    require(shape_names(selection.dom()[0]) == ['x', 'r|x'],
            f'the selection reads {shape_names(selection.dom()[0])}')
    require(len(selection.cod()) == 1
            and Quantization.holds_natural_numbers(selection.cod()[0].datatype),
            'the selection returns more than positions')
    require(shape_names(selection.cod()[0]) == ['x', 's|x'],
            f'the selection returns {shape_names(selection.cod()[0])}')
    require(size_of('s') == 512, f"s is {size_of('s')}")
    require(not any(isinstance(node.operator, dst.TopK)
                    for node in operations(first_body('Idx'))),
            'the selection stands inside the indexer box')
    require(operations_with(dst.IndexSelect, first_body('Gth')),
            'the gather reads no entry at a position')


# ==========================================================================
# The hierarchical candidate pool.
# ==========================================================================
def check_the_pool_is_published_by_layer_twenty_and_read_by_reindex() -> None:
    '''The first decoder layer drops the pool, the Reindex layers grab it and read the
    scores of their indexer at its distances, and the pool covers eight distances of
    each of 2048 blocks.'''
    validate_quantised_text_only_model.check_the_candidate_pool_is_computed_once()
    validate_quantised_text_only_model.check_the_pool_covers_eight_distances_per_kept_block()
    require_sites('Pool', 1)
    box_publishing('Full', SLOT_POOL)
    reindex = body_of(boxes('Rex')[0])
    require(SLOT_POOL in slots_grabbed(reindex), 'the Reindex layers grab no pool')
    read, = top_level_operations_with(dst.IndexSelect, reindex)
    require([shape_names(array) for array in read.dom()] == [['x', 'C'], ['x', 'r|x']],
            f'the Reindex layer reads {[shape_names(array) for array in read.dom()]}')


def check_the_pool_body() -> None:
    '''The pool splits the distances into blocks by a view, takes the best score of each
    block, keeps the best blocks, applies the block split to each kept block number at
    every offset and lays the results along the candidate axis. It is computed once per
    token and returns positions.'''
    pool = boxes('Pool')[0]
    require(tuple(pool.degree()) == (x,), f'the pool is broadcast over {pool.degree()}')
    require(Quantization.holds_natural_numbers(pool.cod()[0].datatype),
            'the pool returns no positions')
    body = body_of(pool)
    view_named('\\mathrm{Block}', body)
    covariant_view_named('\\mathrm{Candidate}', body)
    kinds = {type(node.operator) for node in operations(body)}
    require({ops.Maximum, dst.TopK} <= kinds
            and any(type(node.operator).__name__ == 'MergedPositions'
                    for node in operations(body)),
            f'the pool holds {sorted(kind.__name__ for kind in kinds)}')


def check_the_pin_at_block_zero() -> None:
    '''Positive infinity is written at block 0 by a covariant view and added to the
    block scores, in a block of its own.'''
    body = body_of(boxes('Pool')[0])
    pin, = blocks_titled(text.PIN_TITLE, body)
    covariant_view_named('\\mathrm{Pin}', pin)
    kinds = [type(node.operator).__name__ for node in operations(pin)]
    require(kinds == ['ConstantOp', 'CovariantView', 'AdditionOp'],
            f'the pin holds {kinds}')


# ==========================================================================
# The three attention modes.
# ==========================================================================
def check_every_mode_concatenates_the_window_and_the_selection() -> None:
    '''Every mode lays the window slots and the selected slots end to end along one axis
    in front of its core, and computes its own query, window latents and output
    projection.'''
    for name in ('Full', 'Rex', 'Reu'):
        for box in boxes(name):
            body = body_of(box)
            concatenation, = top_level_operations_with(aops.ConcatenateAxes, body)
            require(axis_name(concatenation.cod()[0].shape()[1]) == 'w|x + s|x',
                    f'a {name} layer concatenates into '
                    f'{shape_names(concatenation.cod()[0])}')
            linear_named('W^{Qa}', body)
            linear_named('W^{KV}', body)
            require(any(operator_name(node) == 'Out'
                        for node in operations_with(ops.BlockOperator, body)),
                    f'a {name} layer has no output projection of its own')


def check_the_layer_plan_is_the_released_plan() -> None:
    '''Layers 2, 8, 14 and 20 are Full, layers 24, 28, 32 and 36 are Reindex, and every
    other layer from 2 upward is Reuse.'''
    plan = sublayer_sequence(text_only_model.text_only_stack)[0::2]
    require(tuple(plan) == RELEASED_ATTENTION_PLAN,
            f'the plan differs from the release at layers '
            f'{[layer for layer, (drawn, released) in enumerate(zip(plan, RELEASED_ATTENTION_PLAN)) if drawn != released]}')
    require([layer for layer, mode in enumerate(plan) if mode == 'Full'] == [2, 8, 14, 20],
            'the Full layers stand elsewhere')
    require([layer for layer, mode in enumerate(plan) if mode == 'Rex'] == [24, 28, 32, 36],
            'the Reindex layers stand elsewhere')
    validate_quantised_text_only_model.check_the_three_sites_of_the_full_mode()
    validate_quantised_text_only_model.check_the_one_reindex_mode_and_the_four_reuse_modes()
    require_sites('Full', 3)
    require_sites('Rex', 1)
    require_sites('Reu', 4)


def check_what_each_mode_computes() -> None:
    '''A Full layer compresses, indexes and selects. A Reindex layer indexes and selects
    and does not compress. A Reuse layer does none of the three.'''
    def holds(box: cat.Broadcasted) -> tuple[bool, bool, bool]:
        body = body_of(box)
        names = {operator_name(node) for node in operations_with(ops.BlockOperator, body)}
        return ('Comp' in names, 'Idx' in names, bool(operations_with(dst.TopK, body)))
    for name, expected in (('Full', (True, True, True)), ('Rex', (False, True, True)),
                           ('Reu', (False, False, False))):
        for box in boxes(name):
            require(holds(box) == expected,
                    f'a {name} layer compresses, indexes and selects as {holds(box)}')


def check_the_seven_slots_and_the_group_members() -> None:
    '''Seven slots carry the values between layers, and the encoder groups write the
    member of their iteration.'''
    slots = slots_dropped(UNQUANTISED)
    require(slots == {SLOT_CKVe, SLOT_SELe, SLOT_CKVd, SLOT_KId, SLOT_POOL, SLOT_SELd,
                      SLOT_IDS},
            f'the model writes {len(slots)} slots')
    members = {drop.index for drop in tutil.type_search(Para.LoopDrop, UNQUANTISED)
               if drop.tape == SLOT_CKVe}
    require(members == {GROUP_COUNTER, nm.Integer(2)},
            f'the encoder groups write the members {members}')


# ==========================================================================
# The mixture of experts.
# ==========================================================================
def check_every_second_sublayer_is_a_mixture() -> None:
    '''Every second sublayer is the mixture of experts.'''
    sequence = sublayer_sequence(text_only_model.text_only_stack)
    require(set(sequence[1::2]) == {MIXTURE_BOX} and MIXTURE_BOX not in sequence[0::2],
            'the mixture does not stand in every second sublayer')


def check_the_router_scores() -> None:
    '''The router scores 384 experts from the hidden state, divides the logits by the
    temperature, takes the square root of the softplus and keeps six. The released
    temperature is one.'''
    gate = first_body('Gate')
    router = linear_named('W^{R}', gate)
    require(axes(router) == ([['m']], [['e']]), f'the router is {axes(router)}')
    arithmetic = names_of(ops.Arithmetic, gate)
    for formula in ('x / \\tau', '\\sqrt{s^{+}}'):
        require(formula in arithmetic, f'the router applies {arithmetic}')
    require(size_of('e') == 384 and size_of('k') == 6,
            f"e is {size_of('e')} and k is {size_of('k')}")
    released = {constant.symbol: constant.released_value
                for constant in released_constants.RELEASED_CONSTANTS}
    require(released[released_constants.ROUTER_TEMPERATURE] == '1',
            f'the temperature is {released[released_constants.ROUTER_TEMPERATURE]}')
    validate_quantised_text_only_model.check_the_router_is_one_box_inside_the_mixture()


def check_the_bias_chooses_and_the_unbiased_scores_gate() -> None:
    '''The bias is added to one copy of the scores, which chooses the experts, and the
    gates read the unbiased scores at the chosen slots.'''
    roots = roots_of(first_body('Gate'))
    addition = root_with(ops.AdditionOp, roots)
    selection = root_with(dst.TopK, roots)
    read = root_with(dst.Select, roots)
    require(selection.dom[0] == addition.cod[0], 'the biased scores choose nothing')
    require(read.dom[1] == addition.dom[0],
            'the gates do not read the scores the bias was added to')


def check_the_gates_are_normalised_and_scaled() -> None:
    '''The six gates are divided by their sum with the router epsilon and multiplied by
    the route scale of 1.5.'''
    gate = first_body('Gate')
    require(len(operations_with(ops.L1Norm, gate)) == 1, 'the gates are not normalised')
    require(released_constants.ROUTER_EPSILON in set(
                tutil.type_search(nm.FreeNumeric, gate)),
            'the normalisation of the gates adds no epsilon')
    require('\\tfrac{3}{2}x' in names_of(ops.Arithmetic, gate),
            'the gates are not multiplied by 1.5')


def check_the_experts_are_clamped_swiglus() -> None:
    '''A routed expert is a SwiGLU of hidden width 2304 whose gate branch is limited
    from above at the limit and whose up branch is limited on both sides, the limit is
    10, and one shared expert of the same width runs at every token.'''
    mixture = first_body('MoE')
    routed, = blocks_titled(text.EXPERTS_TITLE, mixture)
    shared, = blocks_titled(text.SHARED_TITLE, mixture)
    for block, gate_projection in ((routed, 'W^{G}'), (shared, 'W^{Gs}')):
        arithmetic = names_of(ops.Arithmetic, block)
        for formula in ('\\min(x, \\lambda)', 'x \\sigma(x)',
                        '\\ulcorner x \\lrcorner_{-\\lambda}^{\\lambda}'):
            require(formula in arithmetic, f'an expert applies {arithmetic}')
        require(shape_names(linear_named(gate_projection, block).cod()[0])[-1] == 'f',
                f'{gate_projection} returns no expert width')
    require(size_of('f') == 2304, f"f is {size_of('f')}")
    released = {constant.symbol: constant.released_value
                for constant in released_constants.RELEASED_CONSTANTS}
    require(released[released_constants.SWIGLU_LIMIT] == '10',
            f'the limit is {released[released_constants.SWIGLU_LIMIT]}')


def check_the_gates_meet_the_expert_outputs_once_per_token() -> None:
    '''The down projection produces the expert axis and a diagonal keeps the expert of
    each slot, the gates meet the expert outputs in one contraction over the slots, and
    the mixture is computed once per token.'''
    mixture = first_body('MoE')
    down = linear_named('W^{D}', mixture)
    require(shape_names(down.cod()[0]) == ['k/e', 'k/e', 'm'],
            f'the down projection returns {shape_names(down.cod()[0])}')
    require(shape_names(view_named('\\mathrm{Diagonal}', mixture).cod()[0]) == ['k/e', 'm'],
            'the diagonal keeps no expert per slot')
    gathered = [node for node in operations_with(ops.Einops, mixture)
                if [shape_names(array) for array in node.dom()] == [['k/e'], ['k/e', 'm']]]
    require(len(gathered) == 1 and shape_names(gathered[0].cod()[0]) == ['m'],
            'the gates do not meet the expert outputs in one contraction')
    for box in boxes(MIXTURE_BOX):
        require(tuple(box.degree()) == (x,), f'the mixture is broadcast over {box.degree()}')


# ==========================================================================
# The Engram modules.
# ==========================================================================
def check_the_engram_sizes() -> None:
    '''The token map has 99092 results, an n-gram hash reads four identifiers in three
    orders under eight heads, a row holds 256 channels, and the tables of layers 1 and
    14 hold 384006168 and 384016682 rows.'''
    validate_quantised_text_only_model.check_the_two_engram_modules_of_the_model()
    validate_quantised_text_only_model.check_the_ngram_orders_and_the_hash_heads()
    validate_quantised_text_only_model.check_the_sizes_of_the_engram_lookup()
    require(size_of('\\hat{v}') == 99092, f"the token map has {size_of('\\hat{v}')} results")
    require(size_of('T1') == 384006168 and size_of('T14') == 384016682,
            f"the tables hold {size_of('T1')} and {size_of('T14')} rows")
    require(size_of('G') * size_of('K') == 24, 'the hash does not give 24 ranges of rows')


def check_the_hash_is_written_over_integer_operators() -> None:
    '''Each hash is written out over two fixed arrays, one exclusive or, one remainder
    and one cast, and holds no generic operator.'''
    for name in ('Hash{1}', 'Hash{14}'):
        body = first_body(name)
        counts = collections.Counter(type(node.operator) for node in operations(body))
        require((counts[ops.FixedArray], counts[ops.BitwiseXor], counts[ops.Modulo],
                 counts[ops.Cast]) == (2, 1, 1, 1),
                f'{name} holds {dict((kind.__name__, count) for kind, count in counts.items())}')
        require(not operations_with(ops.GenericOperator, body),
                f'{name} holds a generic operator')


def check_the_engram_keys_values_and_gate() -> None:
    '''The rows are projected into one key per stream and one shared value, and the
    gated value is added to every stream.'''
    for layer in (1, 14):
        body = first_body(f'Eng{{{layer}}}')
        keys = linear_named(f'W^{{K}}{{{layer}}}', body)
        values = linear_named(f'W^{{V}}{{{layer}}}', body)
        require(shape_names(keys.cod()[0]) == ['x', 'n', 'm'],
                f'the keys of layer {layer} are {shape_names(keys.cod()[0])}')
        require(shape_names(values.cod()[0]) == ['x', 'm'],
                f'the value of layer {layer} is {shape_names(values.cod()[0])}')
        require('\\mathrm{gate}' in names_of(ops.Arithmetic, body),
                f'the Engram of layer {layer} has no gate')
        require(axes(body)[0] == axes(body)[1] == [['x', 'n', 'm']],
                f'the Engram of layer {layer} reads and returns {axes(body)}')


def check_the_lookback_reads_the_identifiers_before_the_token_map() -> None:
    '''In the CausalSlide the lookback reads the identifiers grabbed from the tape, and
    the token map runs on every slot of the lookback.'''
    for layer in (1, 14):
        _, roots = graph_and_roots(first_body(f'Eng{{{layer}}}'))
        lookback = root_named('\\mathrm{Lookback}', roots)
        grab, = (root for root in roots if isinstance(root.wraps, Para.Grab))
        require(lookback.dom[0] == grab.cod[0],
                f'the lookback of layer {layer} reads no grabbed identifiers')
        token_map, = (node for node in operations_with(ops.Embedding, first_body(f'Eng{{{layer}}}'))
                      if operator_name(node) == 'E\\mathrm{cmp}')
        require(shape_names(token_map.dom()[0]) == ['x', 'L|x'],
                f'the token map of layer {layer} reads {shape_names(token_map.dom()[0])}')


def check_each_engram_module_names_its_own_layer() -> None:
    '''Every weight and fixed array of an Engram module other than the token map carries
    the number of its layer, and the two modules share none.'''
    names_of_layer = {}
    for layer in (1, 14):
        body = first_body(f'Eng{{{layer}}}')
        named = {operator_name(node) for node in operations(body)
                 if isinstance(node.operator, (ops.Linear, ops.FixedArray, ops.Embedding))
                 and operator_name(node) != 'E\\mathrm{cmp}'}
        unnumbered = sorted(name for name in named if not name.endswith(f'{{{layer}}}'))
        require(not unnumbered, f'{unnumbered} of layer {layer} carry no layer')
        names_of_layer[layer] = named
    require(not names_of_layer[1] & names_of_layer[14],
            f'the modules share {sorted(names_of_layer[1] & names_of_layer[14])}')


# ==========================================================================
# The forty layers and the output.
# ==========================================================================
def check_the_repeated_groups() -> None:
    '''The first two encoder groups are one block at repetition two, the third group is
    written alone, the four Reindex groups are one block at repetition four, and Engram
    stands between the two window layers and before the third group.'''
    validate_quantised_text_only_model.check_the_stack_holds_forty_layers()
    stack = text_only_model.text_only_stack
    for title, repetition in ((text.ENCODER_GROUP_TITLE, 2),
                              (text.THIRD_ENCODER_GROUP_TITLE, 1),
                              (text.REINDEX_GROUP_TITLE, 4)):
        block, = blocks_titled(title, stack)
        require(block.block_tag.repetition == nm.Integer(repetition),
                f'{title} repeats {block.block_tag.repetition} times')
    listing = ad.listing_without_legend(h2m.recycle(stack)).splitlines()
    engram_lines = [index for index, line in enumerate(listing) if '<Eng{' in line]
    window_lines = [index for index, line in enumerate(listing) if '<SWA>' in line]
    third_group = next(index for index, line in enumerate(listing)
                       if 'The third encoder group' in line)
    require(window_lines[0] < engram_lines[0] < window_lines[1]
            and engram_lines[1] < third_group,
            'Engram stands elsewhere in the stack')


def check_eighty_sublayers() -> None:
    '''The forty layers hold eighty sublayers, and the residual and the collapse vector
    are the only wires crossing the stack.'''
    count = len(sublayer_sequence(text_only_model.text_only_stack))
    require(count == 80, f'the stack holds {count} sublayers')
    validate_quantised_text_only_model.check_the_wires_crossing_the_layer_stack()


def check_the_output_head() -> None:
    '''The tail collapses the streams, normalises the hidden state, maps it onto the
    129280 vocabulary entries and takes a softmax.'''
    tail = quantised_text_only_model.part_after_the_layers(UNQUANTISED)
    require(axes(tail) == ([['x', 'n'], ['x', 'n', 'm']], [['x', 'v']]),
            f'the tail reads and returns {axes(tail)}')
    kinds = [type(node.operator) for node in operations(tail)]
    require(kinds == [ops.Einops, ops.Normalize, ops.Linear, ops.SoftMax],
            f'the tail holds {[kind.__name__ for kind in kinds]}')
    validate_quantised_text_only_model.check_the_size_of_the_vocabulary()


# ==========================================================================
# The quantisations of the released code.
# ==========================================================================
def check_the_released_policy() -> None:
    '''A tensor passed between modules is held in BF16, arithmetic inside a module runs
    in FP32, and the operand of a projection with an FP8 or FP4 weight is rounded to
    MXFP8.'''
    policy = quantised_text_only_model.RELEASED_POLICY
    require(policy.activations == Quantization.BF16 and policy.scalars == Quantization.FP32
            and policy.rounded_operands == Quantization.MXFP8,
            f'the policy holds {policy.activations}, {policy.scalars} and '
            f'{policy.rounded_operands}')


def check_the_block_scaled_formats() -> None:
    '''MXFP8 is E4M3 with one UE8M0 scale per 32 channels, a weight held in FP8 is E4M3
    with one UE8M0 scale per block of 32 rows by 32 channels, and the routed experts are
    MXFP4.'''
    mxfp8 = Quantization.MXFP8
    require(mxfp8.form is Quantization.Encoding.E4M3 and mxfp8.scale is not None,
            f'MXFP8 is {mxfp8}')
    require(Quantization.describe_quantisation(mxfp8).endswith(
                'with one UE8M0 scale per 32 channels'),
            Quantization.describe_quantisation(mxfp8))
    require(Quantization.describe_quantisation(quantised_text_only_model.FP8_WEIGHT)
            .endswith('with one UE8M0 scale per 32 by 32 block'),
            Quantization.describe_quantisation(quantised_text_only_model.FP8_WEIGHT))
    require(quantised_text_only_model.RELEASED_POLICY.weights_by_default
            == quantised_text_only_model.FP8_WEIGHT, 'the default weight is not FP8')
    for name in quantised_text_only_model.ROUTED_EXPERT_PROJECTIONS:
        require(quantised_text_only_model.WEIGHT_QUANTISATIONS[name] == Quantization.MXFP4,
                f'{name} is {quantised_text_only_model.WEIGHT_QUANTISATIONS[name]}')
    validate_quantised_text_only_model.check_a_block_scaled_format_differs_from_its_elements()


def check_the_integer_formats_of_the_indices() -> None:
    '''The identifiers and the token map are INT64, and the pool and the two selection
    slots carry INT32, the format the gather reads.'''
    require(quantised_text_only_model.WEIGHT_QUANTISATIONS['E\\mathrm{cmp}']
            == Quantization.INT64, 'the token map is not INT64')
    identifiers, = QUANTISED.dom()
    require(Quantization.quantisation_of(identifiers.datatype) == Quantization.INT64,
            f'the identifiers are {identifiers.datatype}')
    require(set(quantised_text_only_model.SLOT_QUANTISATIONS.values())
            == {Quantization.INT32}
            and len(quantised_text_only_model.SLOT_QUANTISATIONS) == 3,
            f'the slots carry {quantised_text_only_model.SLOT_QUANTISATIONS}')
    policies = quantised_text_only_model.BOX_POLICIES
    require(policies['Gth'].integer_operands == Quantization.INT32
            and policies['Pool'].integer_results == Quantization.INT32,
            'the gather or the pool reads another integer format')


def check_the_four_boxes_that_depart_from_the_policy() -> None:
    '''The attention kernel accumulates in FP32, the indexer scores and the pool compute
    at the quantisation they carry, and the router and the coefficients return FP32.'''
    policies = quantised_text_only_model.BOX_POLICIES
    require(policies['Core'].contractions
            is quantise_model.ContractionQuantisation.ACCUMULATED,
            'the attention kernel does not accumulate')
    for name in ('Sco', 'Pool'):
        require(policies[name].arithmetic is quantise_model.ArithmeticQuantisation.CARRIED,
                f'{name} does not compute at the quantisation it carries')
    for name in ('Gate', 'Coef'):
        require(policies[name].results == Quantization.FP32, f'{name} returns no FP32')


def check_the_quantised_figure_rounds_the_state_once_and_labels_every_wire() -> None:
    '''In the CausalSlide of the quantised model every wire carries a quantisation, the
    hidden state is rounded to MXFP8 once, the query projection and the window both read
    the rounded state, so the key-value projection of every slot reads MXFP8, and the
    rotation of a slot computes in FP32.'''
    unlabelled = quantise_model.unquantised_weaves(QUANTISED)
    require(not unlabelled, f'{len(unlabelled)} wires of the slid model carry none')
    _, roots = graph_and_roots(first_body('SWA', QUANTISED))
    rounding, = (root for root in roots if isinstance(root.wraps, cat.Broadcasted)
                 and isinstance(root.wraps.operator, Quantization.TypeConvert)
                 and shape_names(root.wraps.cod()[0]) == ['x', 'm'])
    require(Quantization.quantisation_of(rounding.wraps.operator.target)
            == Quantization.MXFP8, 'the hidden state is not rounded to MXFP8')
    window = root_named('\\mathrm{Window}', roots)
    query = root_named('W^{Qa}', roots)
    require(window.dom[0] == rounding.cod[0] and query.dom[0] == rounding.cod[0],
            'the window and the query read different states')
    key_value = root_named('W^{KV}', roots)
    require(key_value.dom[0] == window.cod[0]
            and Quantization.quantisation_of(key_value.wraps.dom()[0].datatype)
            == Quantization.MXFP8,
            'the key-value projection of a slot reads no MXFP8')
    slot_rotations = [box for box in boxes('Rot', QUANTISED)
                      if 'w|x' in shape_names(box.dom()[0])]
    require(slot_rotations and all(
                'FP32' in cast_targets(body_of(box)) for box in slot_rotations),
            'the rotation of a slot does not compute in FP32')
    slot_rotations = [box for box in boxes('Rot', QUANTISED)
                      if 'w|x' in shape_names(box.dom()[0])]
    require(slot_rotations and all(
                'FP32' in cast_targets(body_of(box)) for box in slot_rotations),
            'the rotation of a slot does not compute in FP32')


def check_the_engram_rows_reach_their_projections_with_no_cast() -> None:
    '''The n-gram table is MXFP8, and its rows reach the key and value projections with
    no cast.'''
    for layer in (1, 14):
        require(quantised_text_only_model.WEIGHT_QUANTISATIONS[
                    f'E\\mathrm{{ng}}{{{layer}}}'] == Quantization.MXFP8,
                f'the table of layer {layer} is not MXFP8')
        table, = blocks_titled(text.TABLE_TITLE, body_of(boxes(f'Eng{{{layer}}}', QUANTISED)[0]))
        require(not operations_with(Quantization.TypeConvert, table),
                f'a cast stands between the table of layer {layer} and its projections')


# ==========================================================================
# The two variants of the page.
# ==========================================================================
def check_every_cast_of_the_quantised_page_opens_a_box() -> None:
    '''Every cast of the quantised model has an explanation naming the quantisation it
    reads and the one it writes.'''
    for node in quantise_model.casts_of(QUANTISED):
        explanation = quantised_text_only_model.explain_quantised_cast(node)
        require(explanation is not None, f'a cast {node.operator} opens no box')


def weight_boxes_naming_a_quantisation(auxiliary: dict) -> int:
    return sum(WEIGHT_QUANTISATION_STEM in str(record.get('description', ''))
               for record in auxiliary['expansions'].values())


def check_the_weight_boxes_name_a_quantisation_on_the_quantised_page_alone() -> None:
    '''The box over a weight names the quantisation the released model reads it at on
    the quantised page, and no box of the unquantised page does.'''
    quantised = weight_boxes_naming_a_quantisation(
        page_auxiliary(QUANTISED, QUANTISED_PAGE_SETTINGS))
    unquantised = weight_boxes_naming_a_quantisation(
        page_auxiliary(UNQUANTISED, UNQUANTISED_PAGE_SETTINGS))
    require(quantised > 0 and unquantised == 0,
            f'{quantised} boxes of the quantised page and {unquantised} of the '
            f'unquantised page name the quantisation of a weight')


def check_both_variants_hold_the_same_blocks() -> None:
    '''The two variants hold blocks of the same titles.'''
    def titles(model: cat.Morphism) -> set[str]:
        return {block.block_tag.aesthetics.title
                for block in tutil.type_search(cat.Block, model)
                if block.block_tag.aesthetics is not None}
    require(titles(QUANTISED) == titles(UNQUANTISED),
            f'the titles differ at {sorted(titles(QUANTISED) ^ titles(UNQUANTISED))}')


def quantised_weaves(term: fd.GeneralTerm) -> set[cat.Weave]:
    return {weave for weave in tutil.type_search(cat.Weave, term)
            if Quantization.quantisation_of(weave.datatype) is not None}


def check_the_unquantised_variant_rounds_the_three_caches_alone() -> None:
    '''The unquantised variant rounds the window latents through E4M3, the keys and
    queries of the indexer through E2M1, and the compressed entries through E2M1, and no
    wire outside those three round trips carries a quantisation.'''
    formats = {name: {Quantization.quantisation_of(weave.datatype).form.value
                      for weave in quantised_weaves(first_body(name))}
               for name in ROUND_TRIP_BOXES}
    require(formats == {'FP4e': {'E2M1', 'E4M3'}, 'FP4i': {'E2M1'}, 'FP8w': {'E4M3'}},
            f'the round trips round through {formats}')
    inside = set().union(*(quantised_weaves(first_body(name)) for name in ROUND_TRIP_BOXES))
    require(quantised_weaves(UNQUANTISED) == inside,
            f'{len(quantised_weaves(UNQUANTISED) - inside)} wires outside the round trips '
            f'carry a quantisation')


def operator_counts_without_casts(model: cat.Morphism) -> collections.Counter[str]:
    return collections.Counter(
        type(node.operator).__name__ for node in operations(model)
        if not isinstance(node.operator, Quantization.TypeConvert))


def check_removing_the_quantisations_keeps_every_operation_but_the_casts() -> None:
    '''The quantised model with its quantisations removed holds the operations of the
    model in the reals, counted by kind, apart from the casts, and holds no cast.'''
    stripped = operator_counts_without_casts(STRIPPED)
    unquantised = operator_counts_without_casts(UNQUANTISED)
    require(stripped == unquantised,
            f'the counts differ at '
            f'{ {kind: (stripped[kind], unquantised[kind]) for kind in stripped | unquantised if stripped[kind] != unquantised[kind]} }')
    require(not quantise_model.conversions_of(STRIPPED), 'the stripped model converts')


def check_removing_the_quantisations_keeps_the_datatypes_of_the_wires() -> None:
    '''The wires of the stripped model carry the datatypes of the wires of the model in
    the reals, apart from the E2M1 and E4M3 channels inside the round trips.'''
    def datatypes(model: cat.Morphism) -> set[cat.Datatype]:
        return {weave.datatype for weave in tutil.type_search(cat.Weave, model)}
    in_the_reals = datatypes(UNQUANTISED)
    quantised_in_the_reals = {datatype for datatype in in_the_reals
                              if Quantization.quantisation_of(datatype) is not None}
    require({str(Quantization.quantisation_of(datatype).form.value)
             for datatype in quantised_in_the_reals} == {'E2M1', 'E4M3'},
            f'the model in the reals carries {quantised_in_the_reals}')
    require(datatypes(STRIPPED) == in_the_reals - quantised_in_the_reals,
            f'the wires differ at {datatypes(STRIPPED) ^ (in_the_reals - quantised_in_the_reals)}')


def listings_of_every_body(model: cat.Morphism) -> dict[str, set[str]]:
    listings: dict[str, set[str]] = collections.defaultdict(set)
    for node in operations_with(ops.BlockOperator, model):
        listings[operator_name(node) or ''].add(
            ad.listing_without_legend(h2m.recycle(body_of(node))))
    return listings


def check_removing_the_quantisations_changes_only_the_round_trip_bodies() -> None:
    '''The stripped model and the model in the reals have the same listing, and the same
    body in every box, apart from the three round trips.'''
    require(ad.listing(h2m.recycle(STRIPPED)) == ad.listing(h2m.recycle(UNQUANTISED)),
            'the two models have different listings')
    stripped = listings_of_every_body(STRIPPED)
    unquantised = listings_of_every_body(UNQUANTISED)
    differing = sorted(name for name in stripped.keys() | unquantised.keys()
                       if stripped.get(name) != unquantised.get(name))
    require(differing == sorted(ROUND_TRIP_BOXES), f'the bodies differ at {differing}')


def check_removing_the_quantisations_removes_the_rounding_of_the_caches() -> None:
    '''The round trips of the model in the reals cast into E2M1 or E4M3, and the round
    trips of the stripped model hold no cast.'''
    for name in ROUND_TRIP_BOXES:
        require(operations_with(Quantization.TypeConvert, first_body(name)),
                f'the {name} round trip of the model in the reals does not round')
        require(not operations_with(Quantization.TypeConvert, first_body(name, STRIPPED)),
                f'the stripped {name} round trip rounds')


# ==========================================================================
# The differences from the released code.
# ==========================================================================
def check_the_compressor_reads_the_whole_prompt() -> None:
    '''The encoder's compressor reads every token of the prompt at once.'''
    compressor = compressor_returning(b)
    require(axes(compressor)[0] == [['x', 'm']],
            f'the compressor reads {axes(compressor)[0]}')


def check_a_chosen_expert_with_a_non_positive_biased_score_has_no_gate() -> None:
    '''The gates are read at the chosen slots through the indicator of a positive biased
    score.'''
    roots = roots_of(first_body('Gate'))
    selection = root_with(dst.TopK, roots)
    indicator = root_named('\\mathbbm{1}_{x > 0}', roots)
    read = root_with(dst.Select, roots)
    require(indicator.dom[0] == selection.cod[0] and read.dom[0] == indicator.cod[0],
            'the gates are read without the indicator of a positive biased score')


def check_the_lookback_reads_the_unit_before_the_first_token() -> None:
    '''The lookback of the n-gram hash returns slots guided by the token axis, so a slot
    reaching before the first token holds the universal unit.'''
    lookback = view_named('\\mathrm{Lookback}', first_body('Eng{1}'))
    slots = lookback.cod()[0].shape()[1]
    require(isinstance(slots, AffineGuards.AffineSparseAxis) and slots.guides == (x,),
            f'the lookback returns {shape_names(lookback.cod()[0])}')


CHECKS: tuple[Callable[[], None], ...] = (
    check_both_variants_read_identifiers_and_return_probabilities,
    check_every_box_tag_of_both_variants_holds_one_body,
    check_every_slot_written_is_read,
    check_the_released_sizes_of_the_axes,
    check_the_entries_counted_back_range_over_the_entries,
    check_every_legend_row_of_both_variants_names_its_axis_in_code,
    check_the_window_slots_hold_a_value_where_the_token_is_not_later,
    check_the_causal_slide_keeps_the_ends_and_the_layers,
    check_the_model_holds_no_image_pathway_and_no_draft,
    check_the_router_adds_one_correction_bias,
    check_the_pass_keeps_no_cache,
    check_the_three_caches_are_three_round_trips,
    check_the_embedding_table,
    check_the_hidden_state_is_repeated_into_four_streams,
    check_the_first_collapse_vector_is_a_covariant_view_with_no_operands,
    check_the_identifiers_reach_both_engram_modules_on_the_tape,
    check_the_coefficients_are_a_collapse_vector_an_output_vector_and_a_combine_matrix,
    check_every_sublayer_hands_the_collapse_vector_to_the_next,
    validate_quantised_text_only_model.check_a_layer_returns_the_two_arrays_it_reads,
    check_the_three_coefficient_maps_and_their_activations,
    check_the_sinkhorn_rounds,
    check_the_coefficients_are_computed_once_per_token,
    validate_quantised_text_only_model.check_the_coefficient_prediction_is_written_once,
    check_the_two_sliding_window_layers,
    check_the_model_as_built_reads_the_window_after_the_latent,
    check_the_figures_read_the_window_at_the_hidden_state,
    check_the_window_reads_no_later_token,
    check_one_latent_serves_every_query_head,
    check_the_attention_core_is_computed_once_per_head,
    check_the_grouped_output_projection,
    check_the_rotation_turns_the_last_sixty_four_channels,
    check_the_rotation_body_holds_five_steps,
    check_the_arrays_a_full_layer_turns_and_turns_back,
    check_the_window_layers_turn_at_rope_and_every_other_layer_at_yarn,
    check_a_slot_turns_at_the_position_of_its_own_token,
    check_the_released_rotary_constants,
    check_the_compressors_group_two_tokens_in_the_encoder_and_one_in_the_decoder,
    check_the_compressor_projections_softmax_and_normalisation,
    check_the_decoder_entries_travel_on_the_tape,
    check_the_scale_groups_of_the_three_caches,
    check_each_round_trip_rounds_and_returns_the_array_it_reads,
    check_the_scale_rule_of_each_round_trip,
    check_the_ceiling_is_the_only_generic_operator,
    check_the_casts_of_the_model_in_the_reals_stand_in_the_round_trips,
    check_the_indexer_heads_and_its_query,
    check_the_indexer_keys_are_projected_from_the_entries,
    check_the_indexer_scores,
    check_a_query_reaches_the_entries_the_released_code_counts,
    check_the_back_reads_stand_at_the_copies_that_publish,
    check_the_selection_returns_distances_and_the_gather_reads_entries,
    check_the_pool_is_published_by_layer_twenty_and_read_by_reindex,
    check_the_pool_body,
    check_the_pin_at_block_zero,
    check_every_mode_concatenates_the_window_and_the_selection,
    check_the_layer_plan_is_the_released_plan,
    check_what_each_mode_computes,
    check_the_seven_slots_and_the_group_members,
    check_every_second_sublayer_is_a_mixture,
    check_the_router_scores,
    check_the_bias_chooses_and_the_unbiased_scores_gate,
    check_the_gates_are_normalised_and_scaled,
    check_the_experts_are_clamped_swiglus,
    check_the_gates_meet_the_expert_outputs_once_per_token,
    check_the_engram_sizes,
    check_the_hash_is_written_over_integer_operators,
    check_the_engram_keys_values_and_gate,
    check_the_lookback_reads_the_identifiers_before_the_token_map,
    check_each_engram_module_names_its_own_layer,
    check_the_repeated_groups,
    check_eighty_sublayers,
    check_the_output_head,
    check_the_released_policy,
    check_the_block_scaled_formats,
    check_the_integer_formats_of_the_indices,
    check_the_four_boxes_that_depart_from_the_policy,
    validate_quantised_text_only_model.check_the_casts_and_the_weights_of_the_model,
    validate_quantised_text_only_model.check_every_wire_of_the_model_carries_a_quantisation,
    validate_quantised_text_only_model.check_every_conversion_changes_the_quantisation_of_its_operand,
    validate_quantised_text_only_model.check_the_casts_counted_by_the_formats_they_read_and_write,
    check_the_engram_rows_reach_their_projections_with_no_cast,
    check_the_quantised_figure_rounds_the_state_once_and_labels_every_wire,
    validate_quantised_text_only_model.check_every_wire_of_the_page_carries_a_quantisation,
    check_every_cast_of_the_quantised_page_opens_a_box,
    check_the_weight_boxes_name_a_quantisation_on_the_quantised_page_alone,
    check_both_variants_hold_the_same_blocks,
    check_removing_the_quantisations_keeps_every_operation_but_the_casts,
    check_removing_the_quantisations_keeps_the_datatypes_of_the_wires,
    check_removing_the_quantisations_changes_only_the_round_trip_bodies,
    check_the_unquantised_variant_rounds_the_three_caches_alone,
    check_removing_the_quantisations_removes_the_rounding_of_the_caches,
    check_the_compressor_reads_the_whole_prompt,
    check_a_chosen_expert_with_a_non_positive_biased_score_has_no_gate,
    check_the_lookback_reads_the_unit_before_the_first_token,
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
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} DeepSeek-V4.1-Flash website checks '
          f'passed in {time.perf_counter() - started:.0f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
