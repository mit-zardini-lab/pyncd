# Claude Opus 5.5 (1M context), reasoning effort 40.
'''What the four validators of the tutorial pages share: reading a morphism for its
operations, its shapes and its tape, checking the legend of every variant of a page,
and reporting one line per check.

A validator here holds one `check_` function per claim its notebook makes, in the order
the notebook makes them, in the shape of `notebooks/sota/GLM53/validate_glm53.py`.
`report_each_check` prints one line per check and the time the run took, and a
validator exits non-zero when a check fails.
'''
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import graphs.data_structure.Hypergraph as hg
import graphs.processing.replace_roots as replace_roots
import para.data_structure.Para as Para
import para.processing.backprop as backprop
import term_utilities.term_utilities as tutil

import notebooks.display.notebook_diagrams as notebook_diagrams
from notebooks.sota.DeepSeekV41Flash.construction_idioms import axis_name


class ClaimDoesNotHold(AssertionError):
    '''A claim of a notebook that the expression does not meet.'''


def require(holds: bool, claim: str) -> None:
    if not holds:
        raise ClaimDoesNotHold(claim)


def names(shape: Sequence[cat.Axis]) -> list[str]:
    return [axis_name(axis) for axis in shape]


def shapes(arrays: Sequence[cat.Array]) -> list[list[str]]:
    return [names(tuple(array.shape())) for array in arrays]


def ends(morphism: cat.Morphism) -> tuple[list[list[str]], list[list[str]]]:
    return shapes(tuple(morphism.dom())), shapes(tuple(morphism.cod()))


def nodes_of[O: cat.Operator](kind: type[O], term: object) -> list[cat.Broadcasted]:
    '''Every node of `term` whose operator is a `kind`, in the order of the term.'''
    return [node for node in tutil.type_search(cat.Broadcasted, term)
            if isinstance(node.operator, kind)]


def linear_named(name: str, term: object) -> cat.Broadcasted:
    node, = (node for node in nodes_of(ops.Linear, term)
             if node.operator.name.to_bodies() == name)
    return node


def targets_read(node: cat.Broadcasted) -> list[list[str]]:
    '''The axes each operand of `node` holds in its target, which are the axes the
    operation consumes from that operand.'''
    return [names(tuple(weave.select_target(array.shape())))
            for weave, array in zip(node.input_weaves, node.dom())]


def degree_names(node: cat.Broadcasted) -> list[str]:
    return names(tuple(node.degree()))


def block_titles(term: object) -> list[str]:
    return [block.block_tag.aesthetics.title for block in tutil.type_search(cat.Block, term)
            if block.block_tag.aesthetics is not None]


def slot_name(slot: Para.TapeSlot) -> str:
    return slot.uid._name.to_bodies()


def drops_of(term: object) -> list[Para.Drop]:
    return list(tutil.type_search(Para.Drop, term))


def grabs_of(term: object) -> list[Para.Grab]:
    return list(tutil.type_search(Para.Grab, term))


def saved_shapes(taped: backprop.Taped) -> list[list[str]]:
    '''The shape of every array the forward pass saves, sorted.'''
    return sorted(names(tuple(drop.size.shape())) for drop in drops_of(taped.forward))


def roots_of(term: cat.Morphism) -> tuple[hg.HypergraphRoot, ...]:
    '''Every root of the graph of `term`, at any block depth.'''
    return tuple(replace_roots.walk_roots(hg.Multigraph.from_morphism(term)))


def producer_of(wire: hg.HypergraphObject,
                roots: Sequence[hg.HypergraphRoot]) -> hg.HypergraphRoot | None:
    for root in roots:
        if wire in root.cod:
            return root
    return None


def grabbed_slot(wire: hg.HypergraphObject,
                 roots: Sequence[hg.HypergraphRoot]) -> str | None:
    '''The name of the slot a grab reads onto `wire`, or `None` where no grab
    produces it.'''
    producer = producer_of(wire, roots)
    if producer is None or not isinstance(producer.wraps, Para.Grab):
        return None
    return slot_name(producer.wraps.tape)


def slot_read_through_views(wire: hg.HypergraphObject,
                            roots: Sequence[hg.HypergraphRoot]) -> str | None:
    '''The name of the slot a grab reads the value on `wire` from, directly or
    through a chain of views, or `None` where no grab starts the chain.'''
    producer = producer_of(wire, roots)
    while (producer is not None and isinstance(producer.wraps, cat.Broadcasted)
           and isinstance(producer.wraps.operator, ops.View)):
        producer = producer_of(producer.dom[0], roots)
    if producer is None or not isinstance(producer.wraps, Para.Grab):
        return None
    return slot_name(producer.wraps.tape)


def is_exponential(root: hg.HypergraphRoot) -> bool:
    return (isinstance(root.wraps, cat.Broadcasted)
            and isinstance(root.wraps.operator, ops.Arithmetic)
            and root.wraps.operator.formula == nm.E ** nm.x)


def slots_the_exponentials_are_rebuilt_from(
        taped: backprop.Taped) -> list[list[str | None]]:
    '''For every exponential of the backward pass, the slots read by the contraction
    whose result the exponential is a pointwise image of, each read directly or through
    views.'''
    roots = roots_of(taped.backward)
    rebuilt = []
    for root in roots:
        if not is_exponential(root):
            continue
        producer = producer_of(root.dom[0], roots)
        while (producer is not None and isinstance(producer.wraps, cat.Broadcasted)
               and isinstance(producer.wraps.operator, ops.Arithmetic)
               and len(producer.dom) == 1):
            producer = producer_of(producer.dom[0], roots)
        if producer is not None and isinstance(producer.wraps, cat.Broadcasted) and (
                isinstance(producer.wraps.operator, ops.Einops)):
            rebuilt.append([slot_read_through_views(wire, roots)
                            for wire in producer.dom])
    return rebuilt


def slots_by_shape(taped: backprop.Taped) -> dict[tuple[str, ...], list[str]]:
    '''The slots the forward pass drops, keyed by the shape of the array each
    holds.'''
    slots: dict[tuple[str, ...], list[str]] = {}
    for drop in drops_of(taped.forward):
        slots.setdefault(tuple(names(tuple(drop.size.shape()))), []).append(
            slot_name(drop.tape))
    return {shape: sorted(names_of_slots) for shape, names_of_slots in slots.items()}


def is_contraction_reading(root: hg.HypergraphRoot, targets: list[list[str]]) -> bool:
    return (isinstance(root.wraps, cat.Broadcasted)
            and isinstance(root.wraps.operator, ops.Einops)
            and targets_read(root.wraps) == targets)


def check_every_legend_row_carries_code_names(
    variants: Sequence[notebook_diagrams.PageVariant],
    page: notebook_diagrams.DiagramSettings,
) -> None:
    '''Every variant of a page draws its legend, and every row of it names the axis in
    code. Every size of the tutorial models is one named symbol, so every row whose
    size is not an integer names the size in code as well.'''
    for variant in variants:
        settings = variant.settings or page
        require(settings.advanced_display is not notebook_diagrams.AdvancedDisplay.OFF,
                f'the variant {variant.identifier} draws no legend')
    for identifier, rows in notebook_diagrams.page_variant_legends(
            variants, page).items():
        require(bool(rows), f'the legend of the variant {identifier} is empty')
        unnamed = [row['text'] for row in rows if not row['codeName']]
        require(not unnamed, f'the axes {unnamed} of the variant {identifier} carry no '
                             'code name')
        unnamed_sizes = [row['text'] for row in rows
                         if not row['sizeCodeName'] and row['size'] is None]
        require(not unnamed_sizes, f'the sizes of the axes {unnamed_sizes} of the '
                                   f'variant {identifier} carry no code name')


def report_each_check(checks: Sequence[Callable[[], None]], subject: str) -> int:
    '''Every check, with one line printed for each and a line of totals, returning how
    many of them failed.'''
    started = time.perf_counter()
    failures = 0
    for check in checks:
        try:
            check()
            print(f'ok      {check.__name__}')
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures += 1
            print(f'FAILED  {check.__name__}: {type(error).__name__}: {error}')
    print(f'{len(checks) - failures} of {len(checks)} {subject} checks passed in '
          f'{time.perf_counter() - started:.0f} s')
    return failures
