'''Checks graph passes on structural roots, block occurrences and loop state.

The independent scalar evaluator and nested-tag comparison are shared with
`graphs/validate_wiring.py`. Each check reports its result and any failure makes
the script exit nonzero.
'''
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_structure.Category as cat
import data_structure.Operators as ops
import deepseek.sparse_expansion as sparse_expansion
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as Hypergraph2Morphism
import graphs.processing.HypergraphAnalysis as HypergraphAnalysis
import graphs.processing.hypergraph_crawler as hypergraph_crawler
import graphs.processing.hypergraph_functor as hypergraph_functor
import graphs.processing.merge_duplicate_roots as merge_duplicate_roots
import graphs.processing.rewire_blocks as rewire_blocks
import graphs.validate_wiring as validate_wiring
import notebooks.display.clean_quantisation_labels as clean_quantisation_labels
import quantization.data_structure.Quantization as Quantization
import quantization.processing.conversion_insertion as conversion_insertion
import quantization.processing.quantise_model as quantise_model
import utilities.utilities as util


class RemoveBlocks(hypergraph_functor.Endofunctor):
    def apply_block_tag(self, target: cat.BlockTag) -> None:
        return None


class RenameBlocks(hypergraph_functor.Endofunctor):
    def apply_block_tag(self, target: cat.BlockTag) -> cat.BlockTag:
        return target.reconstruct(aesthetics=cat.BlockAesthetics(title='renamed'))


class WrapSeed(hypergraph_functor.Endofunctor):
    def apply_root(self, target: cat.Morphism) -> cat.Morphism:
        return cat.Block.template(target, title='wrapped')


class ReplaceReLUWithIdentity(hypergraph_functor.Endofunctor):
    def apply_root(self, target: cat.Morphism) -> cat.Morphism:
        if isinstance(target, cat.Broadcasted) and isinstance(target.operator, ops.ReLU):
            return target.dom().identity()
        return target


class ReplaceReLUWithComposedIdentity(ReplaceReLUWithIdentity):
    def apply_root(self, target: cat.Morphism) -> cat.Morphism:
        rewritten = super().apply_root(target)
        if isinstance(rewritten, cat.Rearrangement):
            return cat.Composed((rewritten, rewritten))
        return rewritten


class ReplaceReLUWithProductIdentity(ReplaceReLUWithIdentity):
    def apply_root(self, target: cat.Morphism) -> cat.Morphism:
        rewritten = super().apply_root(target)
        if isinstance(rewritten, cat.Rearrangement):
            return cat.ProductOfMorphisms((rewritten,))
        return rewritten


class ForwardGuide(hypergraph_crawler.ForwardCrawler):
    def root_processor(self, target: cat.Morphism, guide: Sequence[int],
                       ) -> tuple[cat.Morphism, Sequence[int]]:
        raise TypeError(f'structural morphism reached seed processor: {type(target)}')

    def object_processor(self, target: cat.Array, guide: int) -> tuple[cat.Array, int]:
        return target, guide


class ReverseGuide(hypergraph_crawler.ReverseCrawler):
    def root_processor(self, target: cat.Morphism, guide: Sequence[int],
                       ) -> tuple[cat.Morphism, Sequence[int]]:
        raise TypeError(f'structural morphism reached seed processor: {type(target)}')

    def object_processor(self, target: cat.Array, guide: int) -> tuple[cat.Array, int]:
        return target, guide

    def merge_guides(self, guides: Iterable[int]) -> int:
        return sum(guides)

    def generate_guide_for_zeros(self, target: cat.Array) -> int:
        return 0


class ReverseEqualGuide(ReverseGuide):
    def merge_guides(self, guides: Iterable[int]) -> int:
        return util.iallequals(guides)


class AnalysisReverseGuide(HypergraphAnalysis.ReverseCrawler):
    def seed_processor(self, target: cat.Morphism, guide: tuple[int, ...],
                       ) -> tuple[cat.Morphism, tuple[int, ...]]:
        raise TypeError(f'structural morphism reached seed processor: {type(target)}')

    def coalesce(self, guides: tuple[int, ...]) -> int:
        return sum(guides)

    def generate_zero(self, source: tuple[cat.Array, ...]) -> int:
        return 0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def block_tags(term: cat.Morphism) -> Iterator[cat.BlockTag]:
    match term:
        case cat.Block(body=body, block_tag=tag):
            yield tag
            yield from block_tags(body)
        case cat.Composed(content=content) | cat.ProductOfMorphisms(content=content):
            for child in content:
                yield from block_tags(child)


def require_same_behavior(original: cat.Morphism, rewritten: cat.Morphism) -> None:
    require(original.dom() == rewritten.dom() and original.cod() == rewritten.cod(),
            'the pass changed input or output objects')
    for sign in (1, -1):
        inputs = tuple(sign * 10 * (position + 1)
                       for position in range(len(original.dom())))
        require(validate_wiring.evaluate(original, inputs)
                == validate_wiring.evaluate(rewritten, inputs),
                'the pass changed values')
    require(validate_wiring.occurrences(original)
            == validate_wiring.occurrences(rewritten),
            'the pass changed nested block occurrences')


def check_functor(target: cat.Morphism) -> None:
    graph = hg.Multigraph.from_morphism(target)
    rebuilt = Hypergraph2Morphism.hypergraph_to_morphism(
        hypergraph_functor.Endofunctor()(graph))
    require_same_behavior(target, rebuilt)
    removed = Hypergraph2Morphism.hypergraph_to_morphism(RemoveBlocks()(graph))
    require(not tuple(block_tags(removed)), 'the block-removal hook was bypassed')
    renamed = Hypergraph2Morphism.hypergraph_to_morphism(RenameBlocks()(graph))
    tags = tuple(block_tags(renamed))
    require(len(tags) == len(tuple(block_tags(target)))
            and all(tag.aesthetics.title == 'renamed' for tag in tags),
            'the block-tag rewrite was lost')


def check_quantisation(target: cat.Morphism) -> None:
    quantised = quantise_model.quantise_morphism(
        target, quantise_model.QuantizationPolicy(results=None))
    require(len(quantised.dom()) == len(target.dom())
            and len(quantised.cod()) == len(target.cod()), 'quantisation changed arity')
    require(validate_wiring.occurrences(target)
            == validate_wiring.occurrences(quantised),
            'quantisation changed block occurrences')
    require(all(Quantization.quantisation_of(array.datatype) == Quantization.BF16
                for array in (*quantised.dom(), *quantised.cod())),
            'a wire has no BF16 quantisation')
    inputs = tuple(10 * (position + 1) for position in range(len(target.dom())))
    require(validate_wiring.evaluate(target, inputs)
            == validate_wiring.evaluate(quantised, inputs),
            'quantisation changed wiring')


def check_merge_and_recompute(target: cat.Morphism) -> None:
    graph = hg.Multigraph.from_morphism(target)
    merged = merge_duplicate_roots.merge_duplicate_roots(graph)
    rebuilt = rewire_blocks.recompute_block_boundaries(merged)
    result = Hypergraph2Morphism.hypergraph_to_morphism(rebuilt)
    require_same_behavior(target, result)
    require_same_behavior(target, sparse_expansion.expand_sparse(result))


def check_crawler(mapping: tuple[int, ...], expected_forward: tuple[int, ...],
                  expected_reverse: tuple[int, ...]) -> None:
    rearrangement = validate_wiring.wiring(2, mapping)
    root = hg.HypergraphRoot.template(
        rearrangement, cod=hg.HypergraphObject.template(rearrangement.cod()))
    forward, forward_guide = ForwardGuide().propagate(root, (10, 20))
    reverse, reverse_guide = ReverseGuide().propagate(
        root, tuple(range(1, len(mapping) + 1)))
    require(tuple(forward_guide) == expected_forward, 'incorrect forward guides')
    require(tuple(reverse_guide) == expected_reverse, 'incorrect reverse guides')
    require(forward.cod == root.cod and reverse.cod == root.cod,
            'crawler changed output wire identities')
    analysis_reverse, analysis_guide = AnalysisReverseGuide().crawl(
        root, tuple(range(1, len(mapping) + 1)))
    require(analysis_guide == expected_reverse, 'incorrect analysis crawler guides')
    require(analysis_reverse.cod == root.cod, 'analysis crawler changed output wires')


def check_aliased_loop(crawler: ForwardGuide | ReverseEqualGuide, reduce: bool) -> None:
    repeat = validate_wiring.block(validate_wiring.wiring(2, (1, 0)), 'repeat', 2)
    shared = hg.HypergraphObject(obj=validate_wiring.SCALAR)
    graph = hg.Multigraph.from_morphism(repeat, dom=(shared, shared), reduce=reduce)
    expected = (1,) * len(crawler.exit_nodes(graph))
    rebuilt, guide = crawler.propagate(graph, (1,) * len(crawler.entry_nodes(graph)))
    require(tuple(guide) == expected, 'incorrect loop guides')
    require(len(rebuilt.dom) == len(graph.dom) and len(rebuilt.cod) == len(graph.cod),
            'crawler changed outer arity')


def check_seed_expansion() -> None:
    shared = hg.HypergraphObject(obj=validate_wiring.SCALAR)
    root = hg.HypergraphRoot.template(
        validate_wiring.arithmetic(ops.AdditionOp(), 2), dom=(shared, shared))
    rewritten = WrapSeed()(root)
    require(len(rewritten.dom) == len(root.dom), 'seed rewrite reduced root arity')


def check_seed_rewritten_as_wiring(functor: ReplaceReLUWithIdentity) -> None:
    relu = validate_wiring.arithmetic(ops.ReLU(), 1)
    first = validate_wiring.block(relu, 'first')
    second = validate_wiring.block(relu, 'second')
    original = validate_wiring.compose(first, second)
    expected = validate_wiring.compose(
        first.reconstruct(body=relu.dom().identity()),
        second.reconstruct(body=relu.dom().identity()))
    rewritten = Hypergraph2Morphism.hypergraph_to_morphism(
        functor(hg.Multigraph.from_morphism(original)))
    require_same_behavior(expected, rewritten)
    loop = validate_wiring.block(
        validate_wiring.product(relu, validate_wiring.wiring(1, (0,))), 'loop', 3)
    copy = validate_wiring.wiring(1, (0, 0))
    original_loop = validate_wiring.compose(copy, loop)
    expected_loop = validate_wiring.compose(
        copy, loop.reconstruct(body=validate_wiring.wiring(2, (0, 1))))
    rewritten_loop = Hypergraph2Morphism.hypergraph_to_morphism(
        functor(hg.Multigraph.from_morphism(original_loop)))
    require_same_behavior(expected_loop, rewritten_loop)


def check_datatype_permutation() -> None:
    swap = validate_wiring.wiring(2, (1, 0))
    typed = conversion_insertion.with_datatypes(
        swap, (Quantization.BF16, Quantization.FP32),
        (Quantization.FP32, Quantization.BF16))
    require(tuple(array.datatype for array in typed.cod())
            == (Quantization.FP32, Quantization.BF16),
            'datatype rewrite ignored permutation')
    try:
        conversion_insertion.with_datatypes(
            swap, (Quantization.BF16, Quantization.FP32),
            (Quantization.BF16, Quantization.FP32))
    except conversion_insertion.ImpositionError:
        return
    raise ValueError('inconsistent output datatypes were accepted')


def check_quantisation_labels() -> None:
    datatype = Quantization.with_quantisation(cat.Reals(), Quantization.BF16)
    array = cat.Array(datatype, ())
    target = validate_wiring.block(
        cat.Rearrangement((0, 0), (array,)), 'quantised copy')
    graph = clean_quantisation_labels.strip_quantisations_where_unchanged(
        hg.Multigraph.from_morphism(target))
    for root in sparse_expansion._walk_roots(graph):
        require(tuple(wire.obj for wire in root.dom) == tuple(root.wraps.dom())
                and tuple(wire.obj for wire in root.cod) == tuple(root.wraps.cod()),
                'display root objects disagree with the wrapped morphism')
    require_same_behavior(target, Hypergraph2Morphism.hypergraph_to_morphism(graph))


def check_loop_state() -> None:
    step = validate_wiring.compose(
        validate_wiring.wiring(2, (0, 1, 1)),
        validate_wiring.product(validate_wiring.arithmetic(ops.AdditionOp(), 2),
                                validate_wiring.wiring(1, (0,))))
    loop = validate_wiring.block(step, 'accumulate', 3)
    target = validate_wiring.compose(loop, validate_wiring.wiring(2, (0,)))
    initial = hg.Multigraph.from_morphism(target)
    graph = rewire_blocks.recompute_block_boundaries(
        merge_duplicate_roots.merge_duplicate_roots(initial))
    require_same_behavior(target, Hypergraph2Morphism.hypergraph_to_morphism(graph))
    loop_graph = next(child for child in graph.subgraphs()
                      if isinstance(child, hg.HypergraphBlock))
    require(len(loop_graph.cod) == 2 and len(loop_graph.body.cod) == 2,
            'discarded final result removed iteration state')


def cases() -> Iterator[tuple[str, Callable[[], None]]]:
    swap = validate_wiring.block(validate_wiring.wiring(2, (1, 0)), 'swap')
    copy = validate_wiring.block(validate_wiring.wiring(1, (0, 0)), 'copy')
    fixtures = {
        'swap': swap,
        'copy': copy,
        'discard all': validate_wiring.block(validate_wiring.wiring(2, ()), 'discard'),
        'nested': validate_wiring.block(validate_wiring.compose(copy, swap), 'outer'),
        'repeated': validate_wiring.block(swap.body, 'repeat', 2),
        'unit': validate_wiring.block(validate_wiring.wiring(0, ()), 'unit'),
    }
    for label, target in fixtures.items():
        yield f'functor {label}', lambda target=target: check_functor(target)
        yield f'quantisation {label}', lambda target=target: check_quantisation(target)
        yield f'merge, recompute and sparse expansion {label}', (
            lambda target=target: check_merge_and_recompute(target))
    for mapping, forward, reverse in (
            ((1, 0), (20, 10), (2, 1)),
            ((0, 0), (10, 10), (3, 0)),
            ((0,), (10,), (1, 0))):
        yield f'crawler {mapping}', (
            lambda mapping=mapping, forward=forward, reverse=reverse:
            check_crawler(mapping, forward, reverse))
    for reduce in (False, True):
        for crawler in (ForwardGuide(), ReverseEqualGuide()):
            yield f'{type(crawler).__name__} alias reduction={reduce}', (
                lambda crawler=crawler, reduce=reduce:
                check_aliased_loop(crawler, reduce))
    yield 'seed expansion retains repeated input ports', check_seed_expansion
    for functor in (ReplaceReLUWithIdentity(), ReplaceReLUWithComposedIdentity(),
                    ReplaceReLUWithProductIdentity()):
        yield f'{type(functor).__name__} retains block interfaces', (
            lambda functor=functor: check_seed_rewritten_as_wiring(functor))
    yield ('datatype permutation rejects inconsistent outputs',
           check_datatype_permutation)
    yield 'display labels preserve structural copies', check_quantisation_labels
    yield 'partial discard preserves loop state', check_loop_state


def main() -> int:
    failed = 0
    total = 0
    for label, check in cases():
        total += 1
        try:
            check()
            print(f'PASS {label}', flush=True)
        except Exception as error:
            failed += 1
            print(f'FAIL {label}: {type(error).__name__}: {error}', flush=True)
    print(f'TOTAL {total} PASS {total - failed} FAIL {failed}', flush=True)
    return int(failed != 0)


if __name__ == '__main__':
    raise SystemExit(main())
