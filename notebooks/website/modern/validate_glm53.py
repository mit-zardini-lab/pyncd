# Claude Opus 5.5 (1M context), effort 40.
'''Check every claim `notebooks/website/modern/GLM53.ipynb` makes about GLM-5.3.

    python notebooks/website/modern/validate_glm53.py

The notebook is public facing and holds the prose and the figures alone. The claims live
here, one `check_` function per claim, in the order the notebook makes them: the axes
and the ends of the model, the layer plan, the rotary embedding, the read back from
each query and the CausalSlide, the indexer, the attention, the modes, the feed-forward
maps and the output head, the quantisations of the FP8 checkpoint, the pass over new
tokens and its caches, the quantised pass, and the page of four variants. A claim
about the model or its quantised form is checked by the function of
`notebooks/sota/GLM53/validate_glm53.py` or `validate_quantised_glm53.py` that states
it. This validator shares the name of the first, and imports it by its full
module path. The script prints one line per check and the time the run took, and it
exits non-zero on a failure.

The page derives each unquantised variant in the browser from the quantised variant of
its group. `check_the_functor_gives_each_unquantised_form` applies the Python statement
of that functor, `strip_quantisations`, to each quantised form and compares the listing
of the result with the listing of the unquantised form.
`check_the_placement_is_the_narrowest_of_the_placements` derives every placement of the
caches, which takes about a minute.
'''
from __future__ import annotations

import collections
import pathlib
import sys
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards  # noqa: E402
import agent_display as ad  # noqa: E402
import caching.algebra.cache_contents as cache_contents  # noqa: E402
import caching.algebra.cost_cache_placements as cost_cache_placements  # noqa: E402
import caching.algebra.count_pass_operations as count_pass_operations  # noqa: E402
import caching.data_structure.Caching as Caching  # noqa: E402
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import deepseek.data_structure as dst  # noqa: E402
import graphs.processing.Hypergraph2Morphism as h2m  # noqa: E402
import para.data_structure.ParaWrap as para_wrap  # noqa: E402
import quantization.algebra.strip_quantisations as strip_quantisations  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
from para.data_structure.ParaBlockOperator import (  # noqa: E402
    slots_dropped, slots_grabbed)

import notebooks.caching.CachedGLM53.cache_findings as cache_findings  # noqa: E402
import notebooks.caching.CachedGLM53.cached_model as cached_model  # noqa: E402
import notebooks.caching.CachedGLM53.derive_cached_glm53 as derive_cached_glm53  # noqa: E402
import notebooks.display.explain_cached_reads as explain_cached_reads  # noqa: E402
import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.sota_figures as figures  # noqa: E402
import notebooks.sota.GLM53.assemble_page_variants as assemble_page_variants  # noqa: E402
import notebooks.sota.GLM53.declared_axes as declared_axes  # noqa: E402
import notebooks.sota.GLM53.lightning_indexer as lightning_indexer  # noqa: E402
import notebooks.sota.GLM53.operator_explanations as operator_explanations  # noqa: E402
import notebooks.sota.GLM53.quantised_whole_model as quantised_whole_model  # noqa: E402
import notebooks.sota.GLM53.slide_causal_reads as slide_causal_reads  # noqa: E402
import notebooks.sota.GLM53.validate_glm53 as validate_glm53  # noqa: E402
import notebooks.sota.GLM53.validate_quantised_glm53 as validate_quantised_glm53  # noqa: E402
import notebooks.sota.GLM53.whole_model as whole_model  # noqa: E402
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axes  # noqa: E402
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text  # noqa: E402

MODEL: cat.Morphism = whole_model.glm53
DECODE: cat.Morphism = slide_causal_reads.glm53_slid
DECODE_QUANTISED: cat.Morphism = slide_causal_reads.glm53_quantised_slid
CACHED: cat.Morphism = derive_cached_glm53.cached_glm53
CACHED_QUANTISED: cat.Morphism = derive_cached_glm53.cached_glm53_quantised
ASSIGNED_SIZES: dict[str, int] = whole_model.released_assigned_sizes()
CACHED_SIZES: dict[str, int] = derive_cached_glm53.cached_assigned_sizes()
PAGE = assemble_page_variants.page_settings(figures.DiagramMode.HTML)
VARIANTS = assemble_page_variants.page_variants(PAGE)

DISTANCES = lightning_indexer.reach
BACK_NAME = lightning_indexer.BACK_VIEW_NAME
CACHED_TOKENS = derive_cached_glm53.DERIVED.cached_tokens
BF16 = Quantization.BF16

RELEASED_AXIS_WIDTHS: dict[str, int] = {
    'm': 6144, 'h': 64, 'q': 2048, 'c': 512, 'n': 192, 't': 32, 'u': 256, 'i': 32,
    'd': 128, 's': 2048, 'e': 256, 'k': 8, 'f': 2048, 'g': 12288, 'v': 154880}
FULL_LAYER_CACHES = [64, 128, 512]
SHARED_LAYER_CACHES = [64, 512]
VALUES_CACHED_PER_TOKEN = 47616
VALUES_CACHED_ON_THE_OPERANDS_OF_THE_READS = 2558592
PLACEMENTS = 104
PASS_SIZES = ((0, 1), (32768, 1), (4096, 16))
'''Pairs of earlier and new tokens at which the derived pass and the pass written by
hand are compared.'''


class ClaimDoesNotHold(AssertionError):
    '''A claim of the notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def box_of(part: cat.Morphism) -> cat.Broadcasted:
    '''The box a part is, or stands inside a wrap of.'''
    return part.body if isinstance(part, para_wrap.ParaWrap) else part


def body_of(name: str, term: cat.Morphism) -> cat.Morphism:
    return box_of(whole_model.part_named(name, term)).operator.block


def operations_outside_boxes(term: cat.Morphism) -> tuple[cat.Broadcasted, ...]:
    return validate_quantised_glm53.operations_outside_boxes(term)


def views_named(name: str, operations: tuple[cat.Broadcasted, ...]
                ) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in operations
                 if isinstance(node.operator, ops.View) and node.operator.name is not None
                 and node.operator.name.to_bodies() == name)


def linear_maps_named(name: str, term: cat.Morphism) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in tutil.type_search(cat.Broadcasted, term)
                 if isinstance(node.operator, ops.Linear)
                 and node.operator.name.to_bodies() == name)


def shape_of(array: cat.Array) -> tuple[cat.Axis, ...]:
    return tuple(array.shape())


def recycled_listing(term: cat.Morphism) -> str:
    return ad.listing(h2m.recycle(term))


def operators_other_than_views(term: cat.Morphism) -> collections.Counter:
    '''The class and name of every written operation of `term` that is not a view,
    counted once per operation written in the expression.'''
    return collections.Counter(
        (type(node.operator).__name__,
         None if getattr(node.operator, 'name', None) is None
         else node.operator.name.to_bodies())
        for node in quantise_model.operations_of(term)
        if not isinstance(node.operator, ops.View))


def carries(term: cat.Morphism, axis: cat.Axis) -> bool:
    return any(candidate == axis
               for operation in tutil.type_search(cat.Broadcasted, term)
               for array in (*operation.dom(), *operation.cod())
               for candidate in array.shape())


def quantisation_of(datatype: cat.Datatype) -> Quantization.Quantified | None:
    return Quantization.quantisation_of(datatype)


def caches_of(term: cat.Morphism) -> tuple[cat.Broadcasted, ...]:
    return tuple(node for node in tutil.type_search(cat.Broadcasted, term)
                 if isinstance(node.operator, Caching.Caching))


def as_sent(term: cat.Morphism, settings: notebook_diagrams.DiagramSettings
            ) -> cat.Morphism:
    presented = notebook_diagrams.present_each_side(term, settings)
    sent, _, _ = notebook_diagrams.package_auxiliary(presented, settings)
    return sent


def variant(identifier: str) -> notebook_diagrams.PageVariant:
    return next(entry for entry in VARIANTS if entry.identifier == identifier)


# ==========================================================================
# Reading the figures, and the model.
# ==========================================================================
def check_the_axes_have_the_released_sizes() -> None:
    '''The axes of the table take the sizes of the configuration, the turned channels,
    the head and the unturned indexer channels are 64, 256 and 64 wide, and the
    distance axis is as long as the prompt.'''
    differing = {name: ASSIGNED_SIZES.get(name) for name, width
                 in RELEASED_AXIS_WIDTHS.items() if ASSIGNED_SIZES.get(name) != width}
    require(not differing, f'the configuration sizes {differing}')
    widths = tuple(map(validate_glm53.width_of,
                       (declared_axes.p, declared_axes.a, declared_axes.dbar)))
    require(widths == (64, 256, 64), f'p, a and d-bar are {widths} wide')
    require(declared_axes.r.local_size() == declared_axes.x.local_size(),
            'the distance axis is not as long as the prompt')


def check_two_axes_hold_a_value_at_some_positions() -> None:
    '''The distance axis `r|x` and the selection axis `s|x` each carry an affine form
    over the query axis.'''
    selection_axis = lightning_indexer.SELECTED_SLOTS
    for axis in (DISTANCES, selection_axis):
        require(isinstance(axis, AffineGuards.AffineSparseAxis)
                and tuple(axis.guides) == (declared_axes.x,),
                f'{axis.uid._name.to_bodies()} carries no form over the queries')


# ==========================================================================
# The read back from each query and the CausalSlide.
# ==========================================================================
def check_the_back_view_reads_token_i_x_minus_i_r() -> None:
    '''The view named Back reads token `i_x - i_r` for distance `i_r` of query `i_x`,
    and marks the distance axis with the form `i_x - i_r >= 0`.'''
    (axis, strides, shift), = lightning_indexer.READ_BACK._cod_stride_shift
    require(axis == declared_axes.x
            and strides == (nm.Integer(1), nm.Integer(-1)) and shift == nm.Integer(0),
            f'the view reads {axis} at strides {strides} and shift {shift}')
    require(tuple(lightning_indexer.READ_BACK._dom) == (declared_axes.x, DISTANCES)
            and DISTANCES.stride == nm.Integer(-1)
            and DISTANCES.guide_strides == (nm.Integer(1),),
            'the distance axis does not carry the form i_x - i_r >= 0')


def check_one_distance_axis_serves_the_indexer_the_keys_and_the_values() -> None:
    '''The indexer keys, the keys and the values of the attention are read back through
    one distance axis.'''
    gathers = [node for node in tutil.type_search(cat.Broadcasted, MODEL)
               if isinstance(node.operator, dst.IndexSelect)]
    distances = {shape_of(node.dom()[1])[1] for node in gathers}
    require(gathers and distances == {DISTANCES},
            f'the gathers read {len(distances)} distance axes')
    scores, = whole_model.part_named('Idx', MODEL).cod()
    require(shape_of(scores)[1] == DISTANCES, 'the indexer scores another distance axis')


def check_the_slide_reads_the_hidden_state_back_once_before_the_keys() -> None:
    '''In the CausalSlide the read back from each query stands once in each mode, on
    the hidden state at the copy feeding the queries, and the projections of the
    latent and the turned key run over the queries and the distances.'''
    for mode in ('Full', 'Shared'):
        backs = views_named(BACK_NAME, operations_outside_boxes(body_of(mode, DECODE)))
        require(len(backs) == 1, f'the {mode} body reads back {len(backs)} times')
        require(shape_of(backs[0].dom()[0]) == (declared_axes.x, declared_axes.m)
                and shape_of(backs[0].cod()[0])
                == (declared_axes.x, DISTANCES, declared_axes.m),
                f'the {mode} body reads back {backs[0].dom()[0]}')
    for name in ('W^{KVa}', 'W^{Kr}', 'W^{Kb}', 'W^{Vb}'):
        read = {shape_of(node.dom()[0])[:2] for node in linear_maps_named(name, DECODE)}
        require(read == {(declared_axes.x, DISTANCES)}, f'{name} reads {read}')
    queries = {shape_of(node.dom()[0])[:1]
               for node in linear_maps_named('W^{Qa}', DECODE)}
    require(queries == {(declared_axes.x,)}, f'W^Qa reads {queries}')


def check_the_slide_moves_the_reads_past_operations_broadcast_over_the_tokens() -> None:
    '''The CausalSlide holds the operators of the model and no other. Every projection
    the read moves past is broadcast over the queries and the distances, so it computes
    the same function at every token, and a Full layer reads back twice where it read
    back three times.'''
    require(set(operators_other_than_views(DECODE))
            == set(operators_other_than_views(MODEL)),
            'the slide adds or removes an operator other than a view')
    for name in ('W^{KVa}', 'W^{Kr}', 'W^{Kb}', 'W^{Vb}'):
        degrees = {tuple(node.degree()) for node in linear_maps_named(name, DECODE)}
        require(degrees == {(declared_axes.x, DISTANCES)},
                f'{name} is broadcast over {degrees}')
    before = views_named(BACK_NAME, tuple(tutil.type_search(
        cat.Broadcasted, body_of('Full', MODEL))))
    after = views_named(BACK_NAME, tuple(tutil.type_search(
        cat.Broadcasted, body_of('Full', DECODE))))
    require((len(before), len(after)) == (3, 2),
            f'a Full layer reads back {len(before)} and then {len(after)} times')


# ==========================================================================
# The indexer.
# ==========================================================================
def check_the_indexer_reads_back_at_its_own_copy() -> None:
    '''In the indexer the read back stands on the hidden state at the copy inside the
    indexer, and the key projection runs over the queries and the distances, while the
    weights of the heads read the hidden state at the query.'''
    body = body_of('Idx', DECODE)
    backs = views_named(BACK_NAME, operations_outside_boxes(body))
    require(len(backs) == 1 and shape_of(backs[0].dom()[0])
            == (declared_axes.x, declared_axes.m),
            f'the indexer reads back {len(backs)} times')
    keys, = linear_maps_named('k^{I}', body)
    require(shape_of(keys.dom()[0])[:2] == (declared_axes.x, DISTANCES),
            f'the indexer key projection reads {keys.dom()[0]}')
    weights, = linear_maps_named('w^{I}', body)
    require(shape_of(weights.dom()[0]) == (declared_axes.m,),
            f'the weights of the heads read {weights.dom()[0]} for one query')
    require(axes(whole_model.part_named('Idx', DECODE))
            == ([['x', 'q'], ['x', 'm']], [['x', 'r|x']]),
            'the slid indexer reads and returns other arrays')


# ==========================================================================
# The queries, the keys and the values, and the reads at the selected tokens.
# ==========================================================================
def check_the_keys_and_values_run_over_every_query_and_distance() -> None:
    '''The keys and values block of the CausalSlide reads the hidden state read back
    from every query and returns the keys and the values of every head at every query
    and distance.'''
    block = whole_model.part_titled(text.KEYS_AND_VALUES_TITLE, DECODE)
    x, h, a, u, m = (declared_axes.x, declared_axes.h, declared_axes.a,
                     declared_axes.u, declared_axes.m)
    require(tuple(map(shape_of, block.dom())) == ((x, DISTANCES, m),)
            and tuple(map(shape_of, block.cod()))
            == ((x, DISTANCES, h, a), (x, DISTANCES, h, u)),
            f'the block reads {block.dom()} and returns {block.cod()}')


def check_the_gather_reads_keys_already_read_back() -> None:
    '''A gather of the CausalSlide holds one read at the selected tokens and no view,
    because its keys arrive read back from every query.'''
    body = body_of('Gth', DECODE)
    operations = tuple(tutil.type_search(cat.Broadcasted, body))
    require([type(node.operator) for node in operations] == [dst.IndexSelect],
            f'the gather holds {[type(node.operator).__name__ for node in operations]}')
    require(shape_of(body.dom()[1])[:2] == (declared_axes.x, DISTANCES),
            f'the gather reads {body.dom()[1]}')


# ==========================================================================
# The quantised model.
# ==========================================================================
def check_the_quantised_slide_is_the_slide_quantised() -> None:
    '''Taking every quantisation off the quantised model in the CausalSlide gives the
    model in the CausalSlide.'''
    require(recycled_listing(strip_quantisations.strip_quantisations(DECODE_QUANTISED))
            == recycled_listing(DECODE),
            'the slid quantised model stripped differs from the slid model')


# ==========================================================================
# The pass over new tokens and its caches.
# ==========================================================================
def check_the_pass_reads_and_returns_the_new_tokens() -> None:
    '''The pass reads the identifiers of the new tokens and returns their logits, and no
    operation of it carries the token axis of the model.'''
    require(axes(CACHED) == ([['xnew']], [['xnew', 'v']]),
            f'the pass reads and returns {axes(CACHED)}')
    require(not carries(CACHED, declared_axes.x),
            'an operation of the pass carries the token axis of the model')


def check_the_expansions_run_over_every_cached_token() -> None:
    '''The expansions of the latent, the copy of the turned key to every head and the
    join of the key run over every cached token, and the four operators are the ones
    `expand_kv` computes.'''
    for name in derive_cached_glm53.EXPANSION_WEIGHTS:
        read = {shape_of(node.dom()[0])[0] for node in linear_maps_named(name, CACHED)}
        require(read == {CACHED_TOKENS}, f'{name} reads the axes {read}')
    kinds = sorted(type(operator).__name__
                   for operator in derive_cached_glm53.DERIVED.computed_over_the_cache)
    require(kinds == ['ConcatenateAxes', 'Linear', 'Linear', 'View'],
            f'the operators computed over the cache are {kinds}')


def check_the_caches_of_each_attention_mode() -> None:
    '''A Full layer caches the normalised latent, the turned key and, inside the
    indexer, the indexer key, and a Shared layer the first two. The 78 layers hold
    47,616 values per token.'''
    modes = {mode.mode: mode for mode in derive_cached_glm53.caches_by_mode()}
    full, shared = modes['Full'], modes['Shared']
    require(sorted(values for _, values in full.caches) == FULL_LAYER_CACHES
            and full.layers == validate_glm53.FULL_LAYERS,
            f'{full.layers} Full layers cache {full.caches}')
    require(sorted(values for _, values in shared.caches) == SHARED_LAYER_CACHES
            and shared.layers == validate_glm53.LAYERS - validate_glm53.FULL_LAYERS,
            f'{shared.layers} Shared layers cache {shared.caches}')
    require(sum(mode.values_per_token() for mode in modes.values())
            == VALUES_CACHED_PER_TOKEN, 'the layers hold another number of values')
    names = {name for name, _ in full.caches}
    require(names == {'c_{RMSNorm}', 'c_{Rot}'}, f'the caches are named {names}')
    in_the_indexer = [cache_contents.entries_per_token(cache, CACHED_SIZES)
                      for cache in cache_contents.caches_in_the_order_they_run(
                          body_of('Idx', CACHED))]
    require(in_the_indexer == [128], f'the indexer caches {in_the_indexer}')


def check_the_selection_is_not_cached() -> None:
    '''Every cache holds real numbers, and the selection still travels on the one tape
    slot within the pass.'''
    require(all(isinstance(cache.dom()[0].datatype, cat.Reals)
                for cache in caches_of(CACHED)),
            'a cache holds indices')
    require(slots_dropped(CACHED) == slots_grabbed(CACHED)
            and len(slots_dropped(CACHED)) == 1,
            'the pass does not pass the selection on one slot')


def check_the_placement_is_the_narrowest_of_the_placements() -> None:
    '''Moving each cache back past the operator it follows reaches 104 placements, the
    placement of the pass holds the fewest values per token, and the placement on the
    operands of the reads back holds 2,558,592.'''
    placements = cost_cache_placements.placements_by_sliding_caches_back(
        MODEL, declared_axes.x, derive_cached_glm53.OLD_TOKENS,
        derive_cached_glm53.NEW_TOKENS)
    require(len(placements) == PLACEMENTS, f'{len(placements)} placements')
    narrowest = cost_cache_placements.narrowest_placement(placements, CACHED_SIZES)
    require(narrowest.computed_over_the_cache
            == derive_cached_glm53.DERIVED.computed_over_the_cache,
            'the narrowest placement computes other operators over the cache')
    latest = next(placement for placement in placements
                  if not placement.computed_over_the_cache)
    widths = [cost_cache_placements.entries_per_token_of_the_caches(placement,
                                                                    CACHED_SIZES)
              for placement in (narrowest, latest)]
    require(widths == [VALUES_CACHED_PER_TOKEN, VALUES_CACHED_ON_THE_OPERANDS_OF_THE_READS],
            f'the two placements hold {widths} values per token')


def check_the_pass_written_by_hand_caches_and_computes_the_same() -> None:
    '''The pass written by hand from the reference caches arrays of the same widths in
    every layer and computes as many operations as the derived pass.'''
    derived = cache_findings.cached_entries_by_layer(CACHED, CACHED_SIZES)
    written = cache_findings.cached_entries_by_layer()
    require([(layer.mode, sorted(values for _, values in layer.entries))
             for layer in derived]
            == [(layer.mode, sorted(values for _, values in layer.entries))
                for layer in written],
            'the two passes cache other widths')
    derived_nodes = tuple(count_pass_operations.operations_of_a_pass(CACHED))
    written_nodes = tuple(count_pass_operations.operations_of_a_pass(
        cached_model.cached_glm53))
    for earlier, new in PASS_SIZES:
        counts = (
            count_pass_operations.operations_at_sizes(
                derived_nodes, {**CACHED_SIZES, 'xold': earlier, 'xnew': new}),
            count_pass_operations.operations_at_sizes(
                written_nodes, {**cached_model.CACHED_SIZES, 'P': earlier, 'x': new}))
        require(counts[0] == counts[1],
                f'after {earlier} tokens with {new} new the passes take {counts}')


# ==========================================================================
# The quantised pass over new tokens.
# ==========================================================================
def check_every_cache_holds_bf16() -> None:
    '''Every cache of the quantised pass reads and returns BF16, the quantisation of
    the states handed to it, and every wire carries a quantisation.'''
    held = {(quantisation_of(cache.dom()[0].datatype),
             quantisation_of(cache.cod()[0].datatype))
            for cache in caches_of(CACHED_QUANTISED)}
    require(held == {(BF16, BF16)}, f'the caches hold {held}')
    require(not quantise_model.unquantised_weaves(CACHED_QUANTISED),
            'a wire of the quantised pass carries no quantisation')


def check_every_cached_latent_is_rounded_after_the_cache() -> None:
    '''The rounding in front of the expansions reads the latent over every cached
    token, in each of the five attention bodies.'''
    rounded = quantised_whole_model.FP8_ROUNDED_OPERAND
    roundings = [cast for cast in quantise_model.casts_of(CACHED_QUANTISED)
                 if quantisation_of(cast.operator.target) == rounded
                 and shape_of(cast.dom()[0])[0] == CACHED_TOKENS]
    require(len(roundings) == 5
            and all(shape_of(cast.dom()[0])[1:] == (declared_axes.c,)
                    for cast in roundings),
            f'{len(roundings)} roundings read the cached axis')


def check_the_quantised_pass_stripped_is_the_pass() -> None:
    '''Taking every quantisation off the quantised pass gives the pass in the reals.'''
    require(recycled_listing(strip_quantisations.strip_quantisations(CACHED_QUANTISED))
            == recycled_listing(CACHED),
            'the stripped quantised pass differs from the pass')


# ==========================================================================
# The page.
# ==========================================================================
def check_the_page_variants_name_one_another() -> None:
    '''The page holds four variants in the groups Decode and Cached, each unquantised
    variant derived from the quantised variant of its group, and opens on the
    quantised decode variant.'''
    notebook_diagrams.check_page_variants(VARIANTS, assemble_page_variants.INITIAL_VARIANT)
    require([(entry.identifier, entry.group.identifier, entry.group.title, entry.title)
             for entry in VARIANTS]
            == [('decode-quantised', 'decode', 'Decode', 'Quantised'),
                ('decode-unquantised', 'decode', 'Decode', 'Unquantised'),
                ('cached-quantised', 'cached', 'Cached', 'Quantised'),
                ('cached-unquantised', 'cached', 'Cached', 'Unquantised')],
            'the variants are named otherwise')
    for identifier, source in (('decode-unquantised', 'decode-quantised'),
                               ('cached-unquantised', 'cached-quantised')):
        derived = variant(identifier)
        require(derived.term is None and derived.derived_from == source
                and derived.functor is notebook_diagrams.PageFunctor.DEQUANTISE,
                f'{identifier} is not derived from {source}')
    require(assemble_page_variants.INITIAL_VARIANT == 'decode-quantised'
            and assemble_page_variants.PAGE_SLUG == 'GLM53',
            'the page opens elsewhere or is written elsewhere')


def check_the_unquantised_variants_carry_the_tables_of_the_reals() -> None:
    '''Each variant in the reals carries the inspection tables of the model in the
    reals, whose roles name no quantisation, and the cached one keeps the rows of the
    reads of the token axis.'''
    for identifier in ('decode-unquantised', 'cached-unquantised'):
        settings = variant(identifier).settings
        require(settings is not None
                and settings.operator_roles == operator_explanations.OPERATOR_ROLES
                and settings.operator_explanations
                == operator_explanations.OPERATOR_EXPLANATIONS,
                f'{identifier} carries the quantised tables')
        require(settings.width == assemble_page_variants.UNQUANTISED_WIDTH,
                f'{identifier} is drawn at the width {settings.width}')
    cached_rows = set(variant('cached-unquantised').settings.reindexing_explanations)
    require(set(explain_cached_reads.CACHED_READ_EXPLANATIONS) <= cached_rows,
            'the cached variant in the reals loses the rows of the reads')


def check_the_functor_gives_each_unquantised_form() -> None:
    '''The Python statement of the functor the page applies gives the model in the
    CausalSlide from the quantised model, and the pass over new tokens from the
    quantised pass.'''
    for quantised, unquantised, label in (
            (variant('decode-quantised').term, DECODE, 'decode'),
            (variant('cached-quantised').term, CACHED, 'cached')):
        derived = notebook_diagrams.apply_page_functor(
            notebook_diagrams.PageFunctor.DEQUANTISE, quantised)
        require(recycled_listing(derived) == recycled_listing(unquantised),
                f'the functor does not give the {label} form in the reals')


def check_every_wire_of_the_quantised_variants_carries_a_quantisation() -> None:
    '''Every wire of the two quantised variants, as they are sent to the page, carries
    a quantisation, because the page keeps the quantisation label of every wire.'''
    for identifier in ('decode-quantised', 'cached-quantised'):
        entry = variant(identifier)
        require(entry.settings is not None
                and not entry.settings.clean_quantisation_labels,
                f'{identifier} cleans the quantisation labels of its wires')
        unwritten = quantise_model.unquantised_weaves(as_sent(entry.term, entry.settings))
        require(not unwritten, f'{len(unwritten)} wires of {identifier} carry none')


def check_every_legend_row_carries_a_code_name() -> None:
    '''Every variant draws its legend, every row of every legend carries the name of
    its axis in generated code, and a row whose size is one named symbol carries the
    code name of the size.'''
    by_identifier = {entry.identifier: entry for entry in VARIANTS}
    for entry in VARIANTS:
        settings = notebook_diagrams.settings_of_page_variant(entry, by_identifier, PAGE)
        require(settings.advanced_display in (figures.AdvancedDisplay.LEGEND,
                                              figures.AdvancedDisplay.INTERACTIVE),
                f'{entry.identifier} draws no legend')
    legends = notebook_diagrams.page_variant_legends(VARIANTS, PAGE)
    for identifier, rows in legends.items():
        missing = [row['text'] for row in rows if not row['codeName']]
        require(rows and not missing, f'{identifier}: the rows {missing} have no code name')
    for entry in VARIANTS:
        term = notebook_diagrams.term_of_page_variant(entry, by_identifier)
        for axis in tutil.type_search(cat.Axis, term):
            size = axis.local_size()
            if (axis.uid._name is not None and isinstance(size, nm.FreeNumeric)
                    and size.uid._name is not None):
                require(bool(size.uid._name.code_form),
                        f'the size of {axis.uid._name.to_bodies()} has no code name')


def check_every_view_of_every_variant_opens_a_box() -> None:
    '''Every named view of every variant, the read of the new tokens among them, opens
    an inspection box on the page.'''
    by_identifier = {entry.identifier: entry for entry in VARIANTS}
    for entry in VARIANTS:
        term = notebook_diagrams.term_of_page_variant(entry, by_identifier)
        settings = notebook_diagrams.settings_of_page_variant(entry, by_identifier, PAGE)
        unexplained = explain_reindexings.names_of_unexplained_views(
            as_sent(term, settings))
        require(not unexplained, f'{entry.identifier}: no box opens over {unexplained}')


CHECKS: tuple[Callable[[], None], ...] = (
    validate_glm53.check_the_model_reads_identifiers_and_returns_logits,
    check_the_axes_have_the_released_sizes,
    check_two_axes_hold_a_value_at_some_positions,
    validate_glm53.check_the_stack_holds_seventy_eight_layers,
    validate_glm53.check_the_layer_plan_is_the_released_plan,
    validate_glm53.check_every_repeated_block_returns_its_domain,
    validate_glm53.check_the_turned_channels_are_thirty_two_pairs,
    validate_glm53.check_a_full_layer_turns_four_arrays_and_a_shared_layer_two,
    check_the_back_view_reads_token_i_x_minus_i_r,
    check_the_slide_reads_the_hidden_state_back_once_before_the_keys,
    check_the_slide_moves_the_reads_past_operations_broadcast_over_the_tokens,
    check_one_distance_axis_serves_the_indexer_the_keys_and_the_values,
    validate_glm53.check_the_sizes_of_the_indexer,
    validate_glm53.check_the_indexer_reads_the_low_rank_and_the_hidden_state,
    check_the_indexer_reads_back_at_its_own_copy,
    validate_glm53.check_the_selection_hands_out_positions_alone,
    validate_glm53.check_the_sizes_of_the_attention,
    check_the_keys_and_values_run_over_every_query_and_distance,
    check_the_gather_reads_keys_already_read_back,
    validate_glm53.check_the_core_reads_the_selected_keys_and_values,
    validate_glm53.check_the_boxes_computed_once_per_index_expand_back,
    validate_glm53.check_a_full_layer_runs_one_indexer_and_a_shared_layer_none,
    validate_glm53.check_both_modes_read_and_return_the_hidden_state,
    validate_glm53.check_the_selection_is_the_one_tape_slot,
    validate_glm53.check_three_dense_layers_and_seventy_five_mixtures,
    validate_glm53.check_the_router_keeps_eight_of_two_hundred_and_fifty_six,
    validate_glm53.check_the_output_head_writes_every_vocabulary_entry,
    validate_glm53.check_every_weight_has_a_role,
    validate_glm53.check_every_named_view_and_hidden_formula_is_explained,
    validate_quantised_glm53.check_every_weight_has_the_quantisation_of_the_checkpoint,
    validate_quantised_glm53.check_every_wire_carries_a_quantisation,
    validate_quantised_glm53.check_an_fp8_projection_reads_a_rounded_operand,
    validate_quantised_glm53.check_the_casts_of_the_model,
    validate_quantised_glm53.check_the_rotary_turn_is_computed_in_bf16,
    validate_quantised_glm53.check_the_indexer_scores_in_fp32,
    validate_quantised_glm53.check_the_attention_core_is_the_fused_kernel,
    validate_quantised_glm53.check_the_feed_forward_maps_apply_their_arithmetic_in_bf16,
    validate_quantised_glm53.check_the_router_computes_and_returns_fp32,
    validate_quantised_glm53.check_the_selection_is_held_in_int32_on_the_tape,
    validate_quantised_glm53.check_the_model_reads_int64_and_returns_bf16_logits,
    validate_quantised_glm53.check_stripping_the_quantisations_returns_the_unquantised_model,
    check_the_quantised_slide_is_the_slide_quantised,
    validate_quantised_glm53.check_every_cast_and_every_weight_opens_a_box,
    check_the_pass_reads_and_returns_the_new_tokens,
    check_the_expansions_run_over_every_cached_token,
    check_the_caches_of_each_attention_mode,
    check_the_selection_is_not_cached,
    check_the_placement_is_the_narrowest_of_the_placements,
    check_the_pass_written_by_hand_caches_and_computes_the_same,
    check_every_cache_holds_bf16,
    check_every_cached_latent_is_rounded_after_the_cache,
    check_the_quantised_pass_stripped_is_the_pass,
    check_the_page_variants_name_one_another,
    check_the_unquantised_variants_carry_the_tables_of_the_reals,
    check_the_functor_gives_each_unquantised_form,
    check_every_wire_of_the_quantised_variants_carries_a_quantisation,
    check_every_legend_row_carries_a_code_name,
    check_every_view_of_every_variant_opens_a_box,
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
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} checks of the GLM-5.3 website '
          f'notebook passed in {time.perf_counter() - started:.0f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
