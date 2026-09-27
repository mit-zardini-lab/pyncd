# Claude Opus 5.5 (1M context), effort 40.
'''Check the `Caching` operator, its expansion, the counts of `cache_contents`, the pass
`derive_cached_pass` derives and the placements `cost_cache_placements` compares.

    python caching/validate_caching.py

The script prints one line per check and exits non-zero on a failure, in the shape of
`notebooks/sota/GLM53/validate_glm53.py`. Python puts the folder of the script first on
the import path, and `caching/data_structure/` would then stand in for the top-level
`data_structure/`, so the folder is replaced by the root of the repository, as
`quantization/validate_quantization.py` does. The claims about GLM-5.3 cached between
passes are in `notebooks/website/modern/validate_glm53.py`.
'''
from __future__ import annotations

import pathlib
import sys
import time

FEATURE_FOLDER = pathlib.Path(__file__).resolve().parent
sys.path = [str(FEATURE_FOLDER.parent),
            *(entry for entry in sys.path
              if pathlib.Path(entry or '.').resolve() != FEATURE_FOLDER)]

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains  # noqa: E402
import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards  # noqa: E402
import advanced_axis_dynamics.data_structure.Operators as aops  # noqa: E402
import algebra.define_by_expansion as define_by_expansion  # noqa: E402
import algebra.einops_simplification as einops_simplification  # noqa: E402
import construction_helpers as ch  # noqa: E402, F401 - the @, * and >> overloads
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.StrideCategory as sc  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import para.data_structure.ParaWrap as para_wrap  # noqa: E402
import performance_modeling.collective_cost as collective_cost  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
import utilities.utilities as util  # noqa: E402

import caching.algebra.cache_contents as cache_contents  # noqa: E402
import caching.algebra.cost_cache_placements as cost_cache_placements  # noqa: E402
import caching.algebra.derive_cached_pass as derive_cached_pass  # noqa: E402
import caching.data_structure.Caching as Caching  # noqa: E402
import caching.registries.standard_expansions as cache_expansions  # noqa: E402

x = cat.RawAxis.named('x')
P = cat.RawAxis.named('P')
c = cat.RawAxis.named('c')
h = cat.RawAxis.named('h')
CACHED_TOKENS = Caching.cached_token_axis(P, x)
LATENT_CACHE = Caching.Caching.template((x, c), CACHED_TOKENS, '\\mathrm{lat}')
HEADS_CACHE = Caching.Caching.template((h, x, c), CACHED_TOKENS, '\\mathrm{kv}',
                                       tokens_position=1)

R = cat.Reals()
sequence = cat.RawAxis.named('s')
slots = fd.DynamicName('w').capture(cat.RawAxis(_size=sequence.local_size()))
window_slots = cat.RawAxis.named('v')
keys = fd.DynamicName("s'").capture(cat.RawAxis(_size=sequence.local_size()))
m = cat.RawAxis.named('m')
k = cat.RawAxis.named('k')
STATE = cat.Array(R, (sequence, m))
SIZES = {'m': 4096, 'h': 32, 'k': 128}
MACHINE = collective_cost.MachineRates(
    matrix_operations_per_second=989e12, scalar_operations_per_second=67e12,
    memory_bytes_per_second=3.35e12, network_bytes_per_second=450e9,
    collective_latency_seconds=1e-6, memory_bytes_per_processor=80 * 2**30)


class ClaimDoesNotHold(AssertionError):
    '''A claim about the cache that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def shape_of(array: cat.Array) -> tuple[cat.Axis, ...]:
    return tuple(array.shape())


def check_a_cache_saves_this_pass_and_loads_every_cached_token() -> None:
    require(shape_of(LATENT_CACHE.dom()[0]) == (x, c)
            and shape_of(LATENT_CACHE.cod()[0]) == (CACHED_TOKENS, c),
            'a cache reads R[x, c] and returns R[P + x, c]')
    require(Caching.past_tokens_of(LATENT_CACHE) == P
            and Caching.tokens_of_this_pass(LATENT_CACHE) == x
            and shape_of(Caching.past_array_of(LATENT_CACHE)) == (P, c),
            'the past tokens are P, the tokens of the pass x, and the past array '
            'R[P, c]')
    require(CACHED_TOKENS.local_size() == P.local_size() + x.local_size(),
            'the cached axis has |P| + |x| positions')


def check_the_token_axis_may_stand_at_any_position() -> None:
    require(shape_of(HEADS_CACHE.dom()[0]) == (h, x, c)
            and shape_of(HEADS_CACHE.cod()[0]) == (h, CACHED_TOKENS, c)
            and Caching.tokens_position(HEADS_CACHE) == 1,
            'a cache of R[h, x, c] returns R[h, P + x, c]')


def check_a_cached_axis_ends_with_the_tokens_of_this_pass() -> None:
    try:
        Caching.Caching.template((x, c), Caching.cached_token_axis(x, P), 'wrong')
    except Caching.CachedAxisIsNotPastThenThisPass:
        return
    raise ClaimDoesNotHold('a cache onto x + P was accepted')


def check_the_expansion_defines_the_cache() -> None:
    definition = define_by_expansion.define_by_standard_expansion(LATENT_CACHE)
    require(isinstance(definition, cat.DefinedExpression),
            'the cache is defined by its standard expansion')
    for cache in (LATENT_CACHE, HEADS_CACHE):
        expansion = cache_expansions.load_the_past_and_append_this_pass(cache)
        require(tuple(expansion.dom()) == tuple(cache.dom())
                and tuple(expansion.cod()) == tuple(cache.cod()),
                'the expansion has the domain and the codomain of the cache')


def check_the_expansion_loads_the_past_and_appends_this_pass() -> None:
    expansion = cache_expansions.load_the_past_and_append_this_pass(LATENT_CACHE)
    grab, = tutil.type_search(Para.CacheGrab, expansion)
    drop, = tutil.type_search(Para.CacheDrop, expansion)
    require(grab.tape == drop.tape, 'the load and the append name one slot')
    require(shape_of(grab.size) == (P, c) and shape_of(drop.size) == (x, c),
            'the load reads R[P, c] and the append writes R[x, c]')
    require(Para.entry_of(grab) == Para.entry_of(drop) == Para.CacheTapeSlot(grab.tape)
            and isinstance(Para.grab_of(Para.entry_of(grab), grab.size), Para.CacheGrab)
            and isinstance(Para.drop_of(Para.entry_of(drop), drop.size), Para.CacheDrop),
            'a ParaWrap holds both seeds as one CacheTapeSlot and writes them back')


def check_the_wrapped_expansion_is_one_concatenation() -> None:
    '''The expansion wrapped by `to_para_wrap` is one wrap over the concatenation,
    which grabs the earlier entries on its first operand and keeps and drops the
    tokens of the pass on its second, and the wrap written back out is the
    expansion.'''
    expansion = cache_expansions.load_the_past_and_append_this_pass(LATENT_CACHE)
    wrapped = para_wrap.to_para_wrap(expansion)
    wrap, = tutil.type_search(para_wrap.ParaWrap, wrapped)
    slot = Para.CacheTapeSlot(tuple(tutil.type_search(Para.CacheGrab, expansion))[0].tape)
    require(isinstance(wrap.body, cat.Broadcasted)
            and isinstance(wrap.body.operator, aops.ConcatenateAxes),
            'the wrap is over the concatenation')
    require(wrap.grabs == (slot, Para.KeptAndDropped(dropped=slot))
            and wrap.drops == (None,),
            'the wrap grabs the earlier entries and keeps and drops the tokens of the pass')
    require(tuple(wrapped.dom()) == tuple(expansion.dom())
            and tuple(wrapped.cod()) == tuple(expansion.cod()),
            'the wrap reads and returns what the expansion reads and returns')
    written_back = wrap.to_base()
    require(len(tuple(tutil.type_search(Para.CacheGrab, written_back))) == 1
            and len(tuple(tutil.type_search(Para.CacheDrop, written_back))) == 1,
            'the wrap written back out loads once and appends once')


def check_a_cache_is_counted_once_per_iteration() -> None:
    body = LATENT_CACHE @ (CACHED_TOKENS >> ops.Linear.template((c,), (c,), 'W'))
    inner = cat.Block.template(body, title='inner', repetition=3)
    outer = cat.Block.template(
        (x >> ops.Linear.template((c,), (c,), 'V')) @ inner, repetition=2)
    caches = tuple(cache_contents.caches_in_the_order_they_run(outer))
    require(len(caches) == 6, f'{len(caches)} caches for a cache run 2 x 3 times')
    require(cache_contents.entries_per_token(caches[0], {'c': 512}) == 512,
            'a cache of R[x, c] keeps |c| values per token')
    require(cache_contents.entries_per_token(HEADS_CACHE, {'c': 512, 'h': 8}) == 4096,
            'a cache of R[h, x, c] keeps |h| |c| values per token')


def check_an_unbound_size_is_reported() -> None:
    try:
        cache_contents.entries_per_token(LATENT_CACHE, {})
    except cache_contents.SizeIsNotBound:
        return
    raise ClaimDoesNotHold('an unbound size was evaluated')


def projection(name: str) -> cat.BroadcastedCategory:
    return sequence >> ops.Linear.template((m,), (h, k), name)


def projections_named(term: object, *names: str) -> frozenset[cat.Operator]:
    return frozenset(
        operation.operator for operation in tutil.type_search(cat.Broadcasted, term)
        if isinstance(operation.operator, ops.Linear)
        and operation.operator.name.to_text() in names)


def read_earlier_tokens(slot_axis: cat.Axis = slots) -> cat.Broadcasted:
    '''Slot `i_w` of token `i_s` reads token `i_s - i_w`, which is the causal mask where
    the slots are as many as the tokens and a sliding window where they are fewer.'''
    back = sc.StrideMorphism(
        _dom=(sequence, slot_axis),
        _cod_stride_shift=((sequence, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
        name=fd.DynamicName('mask'))
    return mark_sparse_domains.guarded_view(
        reindexing=(back, cat.ProdObject((h, k)).identity()), name='mask')


def read_every_token_as_keys() -> cat.Broadcasted:
    renamed = sc.StrideMorphism(
        _dom=(keys,), _cod_stride_shift=((sequence, (nm.Integer(1),), nm.Integer(0)),),
        name=fd.DynamicName('='))
    return ops.View.template(reindexing=(renamed, cat.ProdObject((h, k)).identity()),
                             name='=')


def attention(read_keys: cat.Broadcasted) -> cat.BroadcastedCategory:
    '''Attention of `h` heads of width `k`, its keys and values read by `read_keys`.'''
    queries = (sequence, h, k)
    read = tuple(read_keys.cod()[0].shape())
    scores = (h, sequence, *read[1:-2])
    core = (((einops_simplification.einsum((queries, read), scores, R)
              @ ops.SoftMax.template())
             * cat.ProdObject((cat.Array(R, read),)).identity())
            @ einops_simplification.einsum((scores, read), queries, R))
    return (cat.Rearrangement((0, 0, 0), (STATE,))
            @ (projection('W^{Q}') * (projection('W^{K}') @ read_keys)
               * (projection('W^{V}') @ read_keys))
            @ core
            @ (sequence >> ops.Linear.template((h, k), (m,), 'W^{O}')))


CAUSAL_ATTENTION = attention(read_earlier_tokens())
WINDOWED_ATTENTION = attention(read_earlier_tokens(window_slots))


def residual(sublayer: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    return (cat.Rearrangement((0, 0), (STATE,))
            @ (sublayer * cat.ProdObject((STATE,)).identity())
            @ ops.AdditionOp.template())


def check_the_pass_reads_the_new_tokens_alone() -> None:
    derived = derive_cached_pass.derive_cached_pass(CAUSAL_ATTENTION, sequence, P, x)
    require(shape_of(derived.expression.dom()[0]) == (x, m)
            and shape_of(derived.expression.cod()[0]) == (x, m),
            'the pass reads the state of the new tokens and returns their results')
    caches = derived.caches()
    require(sorted(cache_contents.cache_name(cache) for cache in caches)
            == ['c_{W^{K}}', 'c_{W^{V}}'], 'the pass caches the keys and the values')
    require(all(shape_of(cache.dom()[0]) == (x, h, k)
                and shape_of(cache.cod()[0]) == (derived.cached_tokens, h, k)
                for cache in caches),
            'each cache saves R[x, h, k] and loads R[P + x, h, k]')
    require(not any(axis == sequence
                    for operation in tutil.type_search(cat.Broadcasted, derived.expression)
                    for array in (*operation.dom(), *operation.cod())
                    for axis in array.shape()),
            'no operation of the pass carries the sequence axis of the model')


def check_a_cache_is_read_at_the_earlier_tokens() -> None:
    derived = derive_cached_pass.derive_cached_pass(CAUSAL_ATTENTION, sequence, P, x)
    reads = tuple(
        move_reads_backwards.as_stride_morphism(operation.reindexings[0])
        for operation in tutil.type_search(cat.Broadcasted, derived.expression)
        if isinstance(operation.operator, ops.View)
        and shape_of(operation.dom()[0])[0] == derived.cached_tokens)
    require(len(reads) == 1,
            f'{len(reads)} distinct views read a cache, where the key and value views are one term')
    for read in reads:
        (axis, strides, shift), *_ = read._cod_stride_shift
        require(axis == derived.cached_tokens and shift == P.local_size()
                and [stride.to_latex() for stride in strides[:2]] == ['1', '-1'],
                'slot i_w of new token i_x reads the cached token |P| + i_x - i_w')


def check_the_read_of_the_new_tokens_passes_every_layer() -> None:
    layer = cat.Block.template(
        cat.Rearrangement((0, 0), (STATE,))
        @ (CAUSAL_ATTENTION * cat.ProdObject((STATE,)).identity())
        @ ops.AdditionOp.template(), title='layer', repetition=3)
    derived = derive_cached_pass.derive_cached_pass(layer, sequence, P, x)
    require(shape_of(derived.expression.dom()[0]) == (x, m),
            'three layers read the state of the new tokens alone')
    require(len(tuple(cache_contents.caches_in_the_order_they_run(derived.expression)))
            == 6, 'every layer caches its keys and its values')


def check_projections_over_the_cache_cache_the_state_once() -> None:
    derived = derive_cached_pass.derive_cached_pass(
        CAUSAL_ATTENTION, sequence, P, x,
        projections_named(CAUSAL_ATTENTION, 'W^{K}', 'W^{V}'))
    cache, = derived.caches()
    require(shape_of(cache.dom()[0]) == (x, m)
            and shape_of(cache.cod()[0]) == (derived.cached_tokens, m),
            'one cache of the state feeds both projections')
    projected = tuple(
        operation for operation in tutil.type_search(cat.Broadcasted, derived.expression)
        if isinstance(operation.operator, ops.Linear)
        and operation.operator.name.to_text() in ('W^{K}', 'W^{V}'))
    require(all(shape_of(operation.dom()[0])[0] == derived.cached_tokens
                for operation in projected),
            'the key and value projections run at every cached token')


def check_attention_without_a_causal_read_is_not_cached() -> None:
    try:
        derive_cached_pass.derive_cached_pass(
            attention(read_every_token_as_keys()), sequence, P, x)
    except derive_cached_pass.ReadsTheTokenAxisWhole:
        return
    raise ClaimDoesNotHold('attention reading every token as keys was cached')


def check_a_read_of_the_next_token_is_not_cached() -> None:
    next_token = sc.StrideMorphism(
        _dom=(sequence,),
        _cod_stride_shift=((sequence, (nm.Integer(1),), nm.Integer(1)),),
        name=fd.DynamicName('next'))
    model = ((sequence >> ops.Linear.template((m,), (m,), 'W'))
             @ ops.View.template(reindexing=(next_token, cat.ProdObject((m,)).identity()),
                                 name='next'))
    try:
        derive_cached_pass.derive_cached_pass(model, sequence, P, x)
    except derive_cached_pass.ReadsALaterToken:
        return
    raise ClaimDoesNotHold('a read of the next token was cached')


def check_the_placements_of_attention() -> None:
    placements = cost_cache_placements.placements_by_sliding_caches_back(
        CAUSAL_ATTENTION, sequence, P, x)
    require(len(placements) == 4,
            f'{len(placements)} placements of the key and value caches')
    decode = {**SIZES, 'P': 4096, 'x': 1}
    cheapest = cost_cache_placements.cheapest_placement(placements, decode, MACHINE)
    require(cheapest.computed_over_the_cache == frozenset(),
            'a decode pass is cheapest with the keys and the values cached')
    narrowest = cost_cache_placements.narrowest_placement(placements, decode)
    require(cost_cache_placements.entries_per_token_of_the_caches(narrowest, decode)
            == SIZES['m'],
            'the narrowest placement caches the state once, |m| entries per token')
    require(cost_cache_placements.first_linear_placement(placements)
            .computed_over_the_cache == frozenset(),
            'the first linear maps after the copy are the key and value projections')


def check_a_sliding_window_keeps_its_last_tokens() -> None:
    derived = derive_cached_pass.derive_cached_pass(WINDOWED_ATTENTION, sequence, P, x)
    kept, = derived.kept_tokens
    caches = derived.caches()
    require(len(caches) == 2
            and all(Caching.cached_tokens_of(cache) == kept for cache in caches),
            'the keys and the values are cached over the kept tokens and the new ones')
    require(kept.parts[0].local_size() == nm.collect_like_terms(
                nm.Addition.template(window_slots.local_size(), nm.Integer(-1))),
            'a window of |v| slots keeps |v| - 1 earlier tokens')
    reads = tuple(
        move_reads_backwards.as_stride_morphism(operation.reindexings[0])
        for operation in tutil.type_search(cat.Broadcasted, derived.expression)
        if isinstance(operation.operator, ops.View)
        and shape_of(operation.dom()[0])[0] == kept)
    require(len(reads) == 1, f'{len(reads)} distinct views read the kept tokens')
    (axis, strides, shift), *_ = reads[0]._cod_stride_shift
    require(axis == kept and shift == kept.parts[0].local_size()
            and [stride.to_latex() for stride in strides[:2]] == ['1', '-1'],
            'slot i_v of new token i_x reads position |K| + i_x - i_v of the kept tokens')


def check_a_window_moves_the_same_bytes_whatever_the_past() -> None:
    derived = derive_cached_pass.derive_cached_pass(WINDOWED_ATTENTION, sequence, P, x)
    moved = tuple(
        cost_cache_placements.cost_of_a_pass(
            derived, {**SIZES, 'v': 4096, 'P': past, 'x': 1}, MACHINE).cache_bytes
        for past in (4096, 131072))
    require(moved == (2 * 2 * SIZES['h'] * SIZES['k'] * 4096,) * 2,
            f'a window of 4,096 slots moves {moved} bytes after 4,096 and 131,072 tokens')


def check_a_hybrid_model_keeps_a_window_and_every_earlier_token() -> None:
    derived = derive_cached_pass.derive_cached_pass(
        residual(WINDOWED_ATTENTION) @ residual(CAUSAL_ATTENTION), sequence, P, x)
    kept, = derived.kept_tokens
    cached_axes = util.unique_tuple(
        Caching.cached_tokens_of(cache) for cache in derived.caches())
    require(len(derived.caches()) == 4 and len(cached_axes) == 2
            and derived.cached_tokens in cached_axes and kept in cached_axes,
            'the windowed layer caches over the kept tokens and the full layer over '
            'every earlier token')


def check_the_placements_of_a_window() -> None:
    placements = cost_cache_placements.placements_by_sliding_caches_back(
        WINDOWED_ATTENTION, sequence, P, x)
    state_cached = next(placement for placement in placements
                        if len(placement.computed_over_the_cache) == 2)
    cache, = state_cached.caches()
    require(len(placements) == 4
            and shape_of(cache.cod()[0]) == (state_cached.kept_tokens[0], m),
            'a window has four placements, and computing both projections over the '
            'cache caches the state over the kept tokens')


CHECKS = (
    check_a_cache_saves_this_pass_and_loads_every_cached_token,
    check_the_token_axis_may_stand_at_any_position,
    check_a_cached_axis_ends_with_the_tokens_of_this_pass,
    check_the_expansion_defines_the_cache,
    check_the_expansion_loads_the_past_and_appends_this_pass,
    check_the_wrapped_expansion_is_one_concatenation,
    check_a_cache_is_counted_once_per_iteration,
    check_an_unbound_size_is_reported,
    check_the_pass_reads_the_new_tokens_alone,
    check_a_cache_is_read_at_the_earlier_tokens,
    check_the_read_of_the_new_tokens_passes_every_layer,
    check_projections_over_the_cache_cache_the_state_once,
    check_attention_without_a_causal_read_is_not_cached,
    check_a_read_of_the_next_token_is_not_cached,
    check_the_placements_of_attention,
    check_a_sliding_window_keeps_its_last_tokens,
    check_a_window_moves_the_same_bytes_whatever_the_past,
    check_a_hybrid_model_keeps_a_window_and_every_earlier_token,
    check_the_placements_of_a_window,
)


def report_each_check() -> int:
    started = time.perf_counter()
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'ok      {check.__name__}')
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures += 1
            print(f'FAILED  {check.__name__}: {type(error).__name__}: {error}')
    print(f'{len(CHECKS) - failures} of {len(CHECKS)} cache checks passed in '
          f'{time.perf_counter() - started:.1f} s')
    return failures


def main() -> int:
    return 1 if report_each_check() else 0


if __name__ == '__main__':
    sys.exit(main())
