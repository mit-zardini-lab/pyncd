'''
The SSA renderer. See the package docstring for why this shape.

Notation, which `listing` also prints as a legend so its output is
self-describing:

    %3               a value (a wire). The same %3 everywhere is the same wire.
    R[q, x]          an array: datatype, then its axes
    {d}              an axis the operation consumes, which is part of its target
    q                an axis it is broadcast over, which is part of its tiling
    loop N times { } a block repeated N times
'''

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import para.data_structure.Contravariant as contravariant
import para.data_structure.MultiCategory as multi_category
import para.data_structure.Para as Para
import para.data_structure.ParaWrap as pwrap
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import quantization.data_structure.Quantization as Quantization
from quantization.data_structure.Quantization import format_name
import term_utilities.term_utilities as tutils


# ==========================================================================
# Names for the things being described.
# ==========================================================================
def axis_name(axis: cat.Axis) -> str:
    '''A concatenated axis is named by its parts, as `w|x + s|x`.'''
    if isinstance(axis, AxisConcatenation.ConcatenatedAxis):
        return AxisConcatenation.PART_SEPARATOR.join(
            axis_name(part) for part in axis.parts)
    name = getattr(axis.uid, '_name', None)
    return name.to_text() if name is not None else f'?{axis.uid._id % 9973}'


def datatype_name(datatype: cat.Datatype) -> str:
    '''
    `R` for the reals, otherwise the leading letters of the class. A quantised
    datatype prints its format, as `BF16`, and names the datatype it wraps in
    brackets when that datatype is not the reals.
    '''
    match datatype:
        case Quantization.Quantified(wraps=wraps, size=size, form=form) as quantified:
            width = (format_name(quantified) if form is not None
                     else f'f{numeric_name(size)}')
            inner = datatype_name(wraps)
            return width if inner == 'R' else f'{width}({inner})'
        case cat.Reals():
            return 'R'
    return type(datatype).__qualname__[:3]


def array_type(array: cat.Array, target: frozenset[int] = frozenset()) -> str:
    '''
    `R[q, {d}]`. Positions in `target` are the ones the underlying operation
    consumes, and the rest are broadcast over.
    '''
    axes = tuple(array.shape())
    if not axes:
        return f'{datatype_name(array.datatype)}[]'
    rendered = ', '.join(
        ('{' + axis_name(axis) + '}') if index in target else axis_name(axis)
        for index, axis in enumerate(axes))
    return f'{datatype_name(array.datatype)}[{rendered}]'


def target_positions(weave: cat.Weave) -> frozenset[int]:
    '''Which positions of a weave belong to the target rather than the tiling.'''
    return frozenset(
        index for index, entry in enumerate(weave._shape)
        if not isinstance(entry, cat.WeaveMode))


def operator_name(operator: cat.Operator) -> str:
    '''The operator, plus whatever distinguishes this instance of it.'''
    kind = type(operator).__qualname__
    name = getattr(operator, 'name', None)
    body = name.to_bodies() if name is not None else ''
    if body and body.lower() != kind.lower():
        return f'{kind}<{body}>'
    return kind


def morphism_name(morphism: cat.Morphism) -> str:
    '''
    For a morphism that is not a `Broadcasted` and so has no operator of its
    own. `Para`'s `Grab` and `Drop` name their tape slot this way, which is the
    only thing that shows a forward save and a reverse load are the same slot. A
    `LoopGrab` and a `LoopDrop` add the iteration they name, as `s0[i]`.
    '''
    kind = type(morphism).__qualname__
    name = getattr(morphism, 'name', None)
    body = name.to_bodies() if name is not None else ''
    if isinstance(morphism, (Para.LoopGrab, Para.LoopDrop)):
        body = f'{body}[{numeric_name(morphism.index)}]'
    return f'{kind}<{body}>' if body else kind


def numeric_name(value: nm.Numeric) -> str:
    return value.to_latex()


# ==========================================================================
# Value numbering. A wire is a HypergraphObject; %n is per-wire and stable.
# ==========================================================================
@dataclass
class Values:
    names: dict[hg.HypergraphObject, str] = field(default_factory=dict)

    def of(self, node: hg.HypergraphObject) -> str:
        if node not in self.names:
            self.names[node] = f'%{len(self.names)}'
        return self.names[node]

    def find(self, name: str) -> hg.HypergraphObject | None:
        return next((n for n, v in self.names.items() if v == name), None)


def ordered_subgraphs(graph: hg.Hypergraph) -> tuple:
    '''
    Subgraphs in dataflow order, so that a value is declared before it is used.

    Kahn's algorithm, breaking ties by declaration order, so the result is
    deterministic. A cycle, which should not occur, degrades to declaration
    order rather than raising, since a listing is a debugging aid and refusing
    to print is the least helpful thing it could do.
    '''
    subgraphs = list(graph.subgraphs())
    produced = {
        node: subgraph for subgraph in subgraphs for node in subgraph.cod}
    available: set[hg.HypergraphObject] = set()
    remaining, emitted = list(subgraphs), []
    while remaining:
        ready = [
            subgraph for subgraph in remaining
            if all(node in available or node not in produced
                   for node in subgraph.dom)]
        if not ready:
            return tuple(emitted + remaining)
        chosen = ready[0]
        remaining.remove(chosen)
        emitted.append(chosen)
        available.update(chosen.cod)
    return tuple(emitted)


# ==========================================================================
# The body.
# ==========================================================================
def block_header(block: hg.HypergraphBlock) -> str | None:
    '''What kind of block this is, or None for a bare grouping.'''
    aesthetics = block.block_tag.aesthetics
    repetition = block.block_tag.repetition
    if repetition != nm.Integer(1):
        return f'loop {numeric_name(repetition)} times'
    # A title is the only thing a plain block carries, and a derived one -
    # `R[SoftMax]`, is named for exactly that reason. Without it the
    # block is inlined and the grouping is lost.
    if aesthetics is not None and (aesthetics.description or aesthetics.title):
        return aesthetics.description or aesthetics.title
    return None


def operation_line(root: hg.HypergraphRoot, values: Values) -> str:
    '''`%out = Op(%in[shape], ...) : type` - everything about one morphism.'''
    morphism = root.wraps
    if isinstance(morphism, pwrap.ParaWrap):
        return para_wrap_line(root, morphism, values)
    if not isinstance(morphism, cat.Broadcasted):
        outputs = ', '.join(values.of(node) for node in root.cod)
        inputs = ', '.join(values.of(node) for node in root.dom)
        if isinstance(morphism, cat.Rearrangement):
            # A rearrangement of wires, with no operation of its own.
            return f'{outputs} = rewire({inputs})'
        # Anything else, such as a tape operation, at least states what it is and
        # what it puts on the wire, since `rewire` would claim it does nothing.
        produced = ', '.join(array_type(array) for array in morphism.cod())
        line = f'{outputs} = {morphism_name(morphism)}({inputs})'
        return f'{line} : {produced}' if produced else line

    operands = ', '.join(
        f'{values.of(node)}[{", ".join(_marked(array, weave))}]'
        for node, array, weave
        in zip(root.dom, morphism.dom(), morphism.input_weaves))
    results = ', '.join(values.of(node) for node in root.cod)
    produced = ', '.join(
        array_type(array, target_positions(weave))
        for array, weave in zip(morphism.cod(), morphism.output_weaves))
    line = f'{results} = {operator_name(morphism.operator)}({operands}) : {produced}'
    # Only a named affine map is printed, such as a convolution's `+`.
    # A rearrangement that only selects degree axes is structure rather than
    # content, and are already implied by the marked shapes.
    # A named affine map is real content, such as a convolution's `+`. It may sit
    # inside a composition, so search rather than test the top level. The
    # unnamed rearrangements that merely select degree axes are structure, and
    # are already implied by the marked shapes.
    named = tuple(dict.fromkeys(
        found.name.to_bodies()
        for eta in morphism.reindexings
        for found in tutils.type_search(
            cat.StrideMorphism, eta, lambda m: getattr(m, 'name', None) is not None)))
    if named:
        line += '   via ' + ', '.join(named)
    return line


def _marked(array: cat.Array, weave: cat.Weave) -> tuple[str, ...]:
    target = target_positions(weave)
    return tuple(
        ('{' + axis_name(axis) + '}') if index in target else axis_name(axis)
        for index, axis in enumerate(array.shape()))


def body_lines(graph: hg.Hypergraph, values: Values, indent: int = 0) -> list[str]:
    pad = '  ' * indent
    match graph:
        case hg.HypergraphRoot():
            return [pad + operation_line(graph, values)]
        case hg.HypergraphBlock(body=body):
            header = block_header(graph)
            if header is None:
                return body_lines(body, values, indent)
            return [f'{pad}{header} {{',
                    *body_lines(body, values, indent + 1),
                    f'{pad}}}']
        case _:
            return [line
                    for subgraph in ordered_subgraphs(graph)
                    for line in body_lines(subgraph, values, indent)]


# ==========================================================================
# The public surface.
# ==========================================================================
type Renderable = (cat.Morphism | hg.Hypergraph
                   | multi_category.MultiCategory | cat.DefinedExpression)


def as_hypergraph(target: cat.Morphism | hg.Hypergraph) -> hg.Hypergraph:
    '''Morphism or hypergraph in, hypergraph out.'''
    if isinstance(target, hg.Hypergraph):
        return target
    return hg.Multigraph.from_morphism(target)


def _row_header(index: int, row: multi_category.MultiCategoryElement) -> str:
    direction = contravariant.covariant_or_contravariant(row)
    if direction is contravariant.CovariantOrContravariant.CONTRAVARIANT:
        return f'row {index}  contravariant, listed by its body'
    return f'row {index}  covariant'


def _each_row(
    target: multi_category.MultiCategory,
    render: Callable[[cat.Morphism], str],
) -> str:
    '''One block per row, each row rendered as its covariant body.'''
    return '\n\n'.join(
        '\n'.join([_row_header(index, row),
                   render(contravariant.covariant_body(row))])
        for index, row in enumerate(target))


def _each_side(
    target: cat.DefinedExpression,
    render: Callable[[cat.Morphism], str],
) -> str:
    '''The left-hand side, `:=` on a line of its own, and the right-hand side. Each
    side numbers its own values, so `%0` on the left and `%0` on the right are the
    same wire only where both are the domain or the codomain of the definition.'''
    return '\n'.join([
        render(target.left_hand_side), ':=', render(target.right_hand_side)])


LEGEND = (
    'legend  %n value   R[..] array   {a} axis consumed   a axis broadcast over')


def numbered(graph: hg.Hypergraph) -> tuple[Values, list[str]]:
    '''
    Number every value and render the body, in one fixed order: the domain and
    codomain first, then the operations in dataflow order. Every view goes
    through `numbered`, so `%4` means the same thing in a listing and in a
    trace.
    '''
    values = Values()
    for obj in graph.dom:
        values.of(obj)
    for obj in graph.cod:
        values.of(obj)
    return values, body_lines(graph, values)


def slot_text(entry: Para.NamedEntry, written: bool) -> str:
    '''`<s0>` for a slot, `<s0'>` for a loop variable on the side that writes it
    for the next iteration or for a value sent to other processors, `<s0*>`
    for a value received from a partner processor, and `<s0[i]>` for the member
    of a slot that the iteration `i` of a repeated block holds.'''
    name = Para.slot_of(entry).uid._name.to_bodies()
    index = Para.index_of(entry)
    if index is not None:
        name = f'{name}[{numeric_name(index)}]'
    if isinstance(entry, Para.ReductionSlot):
        mark = "'" if written else '*'
    else:
        mark = "'" if written and isinstance(entry, Para.StreamSlot) else ''
    return f'<{name}{mark}>'


def para_wrap_line(
    root: hg.HypergraphRoot, morphism, values: Values,
) -> str:
    '''
    A `ParaWrap` prints as its body, with the taped operands and results in
    place: a grabbed operand is `<s0>[shape]` where a wire would be `%n[shape]`,
    and a dropped result is `<s1>` among the outputs. So the line reads as the
    operation it wraps, and the tape is visible exactly where it touches it -
    which is what the wrap is for. A loop variable written for the next
    iteration prints as `<s1'>`.
    '''
    body = morphism.body
    inputs = iter(values.of(node) for node in root.dom)
    outputs = iter(values.of(node) for node in root.cod)
    operands = tuple(
        slot_text(slot, written=False) if slot is not None else next(inputs)
        for slot in morphism.grabs)
    results = tuple(
        slot_text(slot, written=True) if slot is not None else next(outputs)
        for slot in morphism.drops)
    if not isinstance(body, cat.Broadcasted):
        kind = ('rewire' if isinstance(body, cat.Rearrangement)
                else morphism_name(body))
        return f'{", ".join(results)} = {kind}({", ".join(operands)})'
    shown = ', '.join(
        f'{operand}[{", ".join(_marked(array, weave))}]'
        for operand, array, weave
        in zip(operands, body.dom(), body.input_weaves))
    produced = ', '.join(
        array_type(array, target_positions(weave))
        for array, weave in zip(body.cod(), body.output_weaves))
    return (f'{", ".join(results)} = {operator_name(body.operator)}({shown})'
            f' : {produced}')


def summary(target: Renderable, values: Values | None = None) -> str:
    '''The inputs, the outputs and the operation counts.

    The body is left out. A `MultiCategory` is summarised one row at a time.
    '''
    if isinstance(target, multi_category.MultiCategory):
        return _each_row(target, summary)
    if isinstance(target, cat.DefinedExpression):
        return _each_side(target, summary)
    graph = as_hypergraph(target)
    values = values if values is not None else numbered(graph)[0]

    leaves = [leaf for leaf in _all_roots(graph)]
    counts: dict[str, int] = {}
    for leaf in leaves:
        # A wrap is its body, for counting as for printing.
        morphism = (leaf.wraps.body if isinstance(leaf.wraps, pwrap.ParaWrap)
                    else leaf.wraps)
        if isinstance(morphism, cat.Broadcasted):
            key = operator_name(morphism.operator).split('<')[0]
            counts[key] = counts.get(key, 0) + 1

    lines = [
        'in    ' + (', '.join(
            f'{values.of(obj)} : {array_type(obj.obj)}' for obj in graph.dom)
            or '(none)'),
        'out   ' + (', '.join(
            f'{values.of(obj)} : {array_type(obj.obj)}' for obj in graph.cod)
            or '(none)'),
        'ops   ' + (', '.join(
            f'{count}x {name}' for name, count in sorted(counts.items()))
            or '(none)'),
    ]
    return '\n'.join(lines)


def listing(target: Renderable) -> str:
    '''
    The full SSA listing, holding everything the morphism carries, under the
    legend that names its notation.

    The listing reads forward, and two of them can be compared with `diff`.
    '''
    return '\n'.join([LEGEND, '', listing_without_legend(target)])


def listing_without_legend(target: Renderable) -> str:
    '''The listing alone, for a caller that prints many and names the notation once.'''
    if isinstance(target, multi_category.MultiCategory):
        return _each_row(target, listing_without_legend)
    if isinstance(target, cat.DefinedExpression):
        return _each_side(target, listing_without_legend)
    graph = as_hypergraph(target)
    values = Values()
    head = summary(graph, values)
    return '\n'.join([head, ''] + body_lines(graph, values))


def trace(target: Renderable, value: str) -> str:
    '''
    What produces `value`, and what consumes it, both named rather than drawn.
    '''
    graph = as_hypergraph(target)
    values, _ = numbered(graph)
    node = values.find(value)
    if node is None:
        return f'{value}: no such value'

    produced_by = [leaf for leaf in _all_roots(graph) if node in leaf.cod]
    consumed_by = [leaf for leaf in _all_roots(graph) if node in leaf.dom]
    lines = [f'{value}']
    lines += ['  from  ' + operation_line(leaf, values) for leaf in produced_by]
    if not produced_by:
        lines.append('  from  (an input of the whole expression)')
    lines += ['  to    ' + operation_line(leaf, values) for leaf in consumed_by]
    if not consumed_by:
        lines.append('  to    (an output of the whole expression)')
    return '\n'.join(lines)


def _all_roots(graph: hg.Hypergraph) -> list[hg.HypergraphRoot]:
    '''Every operation, descending through blocks, which `flat_subgraphs` does not.'''
    match graph:
        case hg.HypergraphRoot():
            return [graph]
        case hg.HypergraphBlock(body=body):
            return _all_roots(body)
        case _:
            return [root
                    for subgraph in graph.subgraphs()
                    for root in _all_roots(subgraph)]
