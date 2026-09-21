'''Re-expressing a backward chain by a value the forward pass already computed.

The FlashAttention D-trick is the motivating case. The derived attention backward
computes its softmax row statistic as

    D = <dP, P>_x        where  dP = dO . V^T

and FlashAttention computes `D = <dO, O>_v` instead. The two are equal, because a
pairing of a cotangent with a primal moves through a linear map to either end, so
`<dY W^T, X> = <dY, X W>`. The second is cheaper, because O is a `q x v` array
already in memory, where the pairing over `x` has to wait for a whole row of dP.

This module derives that rewrite, and others of the same shape, rather than
declaring them. `obsidian/07-para/Pathway Collapse.md` is the full account.

## The three facts it rests on

A backward pass is linear in its cotangents. Every tangent wire passes only
through einsums against primals, additions and reindexings. The nonlinearities,
meaning `e'`, `/z'` and a softmax's own block, touch tangent wires only by
multiplying into them, which is linear in the tangent. The tangent dataflow is
therefore a directed acyclic graph of linear maps, and any tangent value is a sum
over paths of composed linear maps applied to a seed.

A path segment of einsum links is one einsum. Inlining a chain
`t1 = einsum(dY, A); t2 = einsum(t1, B)` gives `t2 = einsum(dY, A, B)`, which is
Fubini's theorem on the nested sums. It is valid because a backward einsum has
exactly one tangent operand, since bilinearity means tangents never multiply
tangents.

Regrouping costs nothing, and a regrouped factor may already exist. In the unified
einsum, any subset G of the primal operands can be contracted first, giving
`einsum(dY, *G, *rest) = einsum(dY, M, *rest)` with `M = einsum(*G)`. If the
forward pass computed exactly `M`, with the same operand values and the same
surviving axes, the backward pass should grab it rather than recompute it. The
operands' identities pass between the two halves through the tape, because a
backward `Grab` and a forward `Drop` share a slot, and an einsum is compared as
`_pattern_key` writes it, over index variables, so the match is structural and
reads no axis by identity.

The rewrite therefore edits both halves of a `Taped`. The backward einsum is
rebuilt to read a `Grab`, and the forward pass gains a `Drop` of the matched wire
when that wire is not already taped.

The second edit is gated. A wire is recalled only when it is already taped or is a
forward output. Recalling anything else would materialise an array the forward pass
never stored, and recalling `S = Q.K^T` through a new `q x x` slot is what
FlashAttention exists to avoid.

## Three further uses of linearity

Each of the three is an instance of one fact: scaling and regrouping commute with a
multilinear chain.

A linear `Arithmetic<c·x>` is part of a chain. On the tangent path the unifier
passes through it and collects `c` onto the chain. On a primal, the match reads
through it to the wire underneath and collects `c` the same way. The constant is
reapplied once, as one `Arithmetic<c·x>` after the rewritten einsum.

A link that cannot join ends the chain rather than making the unifier decline
it. A fan-in, a nonlinearity, or a wire the chain already reads at another
shape makes the wire above it the seed. The chain below it still regroups. The
softmax needs this, because `e ⊙ ⟨dy, ·⟩ₓ` reads `e` free in the product and
contracted in the pairing, and the pairing itself is a usable seed for the
outer regrouping.

Every shape is written in index variables. A root's variables come from
`einops_simplification.index_shapes`, one per degree position and one per
contraction group, and a chain renames each link's variables into its own as
it inlines the link, so a variable a link contracts is fresh to the chain.
Nothing here reads an axis by identity, and one token axis at both positions
of the scores is two variables.

A primal made by an einsum of primals is flattened into its factors, which is
Fubini's theorem on the primal side, so a product such as `r·(-r)` offers each
factor to the match separately.

`split_inverse_squares` is the rewrite that gives those three something to act on.
`Arithmetic<c·x^{-2}>`, which is the derivative the reciprocal rule writes, becomes
`x^{-1}`, copied, with one copy scaled by `c`, multiplied back together. As a map
of z it is opaque to every rewrite here. As a product of reciprocals it is a chain
of values the forward pass already computed.

`split_off_forward_formulas` does the same for any pointwise formula that is a
function of a forward one. A backward `Arithmetic<f>` on a taped wire, where the
forward pass applies `Arithmetic<g>` to that wire and `f = h ∘ g`, becomes `g`
followed by `h`, and `migrate_drops` then reads `g` off the tape. The sigmoid's
derivative `σ(x)(1 - σ(x))` becomes `y(1 - y)` on the taped `y = σ(x)`.

On the expanded softmax the whole sequence then runs. The reciprocal migrates onto
the forward pass's tape, both copies of r regroup with e into y, and the backward
pass comes out as the closed-form softmax rule `dx = y ⊙ dy - y·⟨dy, y⟩`, with
y taped.

## What the rewrites do not do

An addition still ends a chain. `distribute_additions` moves it out of the way
first, where it can.

A loop block is a boundary. Every rewrite reaches through a block whose
repetition is 1, at any depth, and leaves a loop block as it stands, because a
root inside one computes once per iteration. The forward catalogue therefore
holds every value outside a loop, and a chain inside a block is matched and
rewritten like one at the top level.

A rewritten chain is placed one block out of the block its root sat in. The
hoisted statistic is FlashAttention's separate preprocessing kernel, which is
how it is drawn, and `rewire_blocks.recompute_block_boundaries` carries the
wires it reads out of the block and its result back in.

A wire read at one index variable twice, meaning a written diagonal, is skipped.
'''
from __future__ import annotations
from collections import Counter
import itertools
from dataclasses import dataclass
from typing import Iterable, Mapping

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import graphs.data_structure.Hypergraph as hg
import graphs.processing.merge_duplicate_roots as merge_duplicate_roots
import graphs.processing.replace_roots as replace_roots
import graphs.processing.rewire_blocks as rewire_blocks
import graphs.processing.Hypergraph2Morphism as h2m
import data_structure.Operators as ops
import term_utilities.term_utilities as tutil
import algebra.einops_simplification as es

import para.data_structure.Para as para
import para.algebra.prune_zero_cotangents as prune_zero_cotangents
import para.processing.backprop as bp


COLLAPSE_LIMIT = 64

type Shape = fd.Prod[es.IndexVariable]
type Operand = tuple[hg.HypergraphObject, Shape]
type RootShapes = tuple[tuple[Shape, ...], Shape]
type Key = tuple[tuple, frozenset]


def _no_repeats(variables: fd.Prod[es.IndexVariable]) -> bool:
    return len(set(variables)) == len(variables)


@dataclass
class _IndexedPass:
    '''One pass of a `Taped`, with its roots indexed for lookup.

    Roots are walked in construction order, which is the deterministic order
    every function below iterates in, through every block whose repetition
    is 1. The chain of blocks enclosing each root is kept, so a rewrite can
    place a root in a root's own scope or one level out of it, and
    `rewire_blocks.recompute_block_boundaries` then carries the wires. A loop
    block is left as it stands, so its roots are neither matched nor rewritten,
    and its boundary wires are indexed as objects alone.

    `shapes` writes a root's operands and result in the root's own index
    variables, per `einops_simplification.index_shapes`. A variable belongs to
    the root that runs over it. A wire read by two roots is read with each
    root's own variables, so a chain that inlines a producer renames the
    producer's variables into the consumer's, through `_Variables`, and
    nothing here compares two axes by identity.
    '''
    graph: hg.Multigraph

    def __post_init__(self) -> None:
        self.roots: list[hg.HypergraphRoot] = []
        self.chain: dict[hg.HypergraphRoot, rewire_blocks.BlockChain] = {}
        self.loops: list[hg.HypergraphBlock] = []
        self._index_subgraph(self.graph, ())
        self.producer: dict[hg.HypergraphObject, hg.HypergraphRoot] = {}
        self.objects: dict[hg.HypergraphObject, hg.HypergraphObject] = {}
        for obj in (*self.graph.dom, *self.graph.cod):
            self.objects.setdefault(obj, obj)
        for loop in self.loops:
            for obj in (*loop.dom, *loop.cod):
                self.objects.setdefault(obj, obj)
        for root in self.roots:
            for obj in (*root.dom, *root.cod):
                self.objects.setdefault(obj, obj)
            for obj in root.cod:
                self.producer[obj] = root
        self._shapes: dict[hg.HypergraphRoot, RootShapes | None] = {}

    def _index_subgraph(self, subgraph: hg.Hypergraph,
                        blocks: fd.Prod[hg.HypergraphBlock]) -> None:
        match subgraph:
            case hg.HypergraphRoot():
                self.roots.append(subgraph)
                self.chain[subgraph] = rewire_blocks.chain_of(blocks)
            case hg.HypergraphBlock(body=body):
                if rewire_blocks.is_loop(subgraph):
                    self.loops.append(subgraph)
                    return
                self._index_subgraph(body, (*blocks, subgraph))
            case hg.Multigraph():
                for inner in subgraph.subgraphs():
                    self._index_subgraph(inner, blocks)

    def consumed_wires(self) -> set[hg.HypergraphObject]:
        '''Every wire some root, loop block or the graph's codomain reads.'''
        consumed = {node for root in self.roots for node in root.dom}
        consumed |= {node for loop in self.loops for node in loop.dom}
        return consumed | set(self.graph.cod)

    def array(self, node: hg.HypergraphObject) -> cat.Array:
        return self.objects[node].obj

    def shapes(self, root: hg.HypergraphRoot) -> RootShapes | None:
        '''The operand shapes and the result shape of `root`, in variables
        of its own, or None for a root that is not a one-result `Broadcasted`
        read through selections, permutations and repeats.'''
        if root not in self._shapes:
            self._shapes[root] = self._read_shapes(root)
        return self._shapes[root]

    def _read_shapes(self, root: hg.HypergraphRoot) -> RootShapes | None:
        if not isinstance(root.wraps, cat.Broadcasted) or len(root.cod) != 1:
            return None
        try:
            inputs, output, _ = es.index_shapes(root.wraps)
        except ValueError:
            return None
        def own(shape: Shape) -> Shape:
            return tuple(es.IndexVariable((id(root), v.label), v.axis) for v in shape)
        return tuple(own(shape) for shape in inputs), own(output)

    def tangent_closure(self) -> set[hg.HypergraphObject]:
        '''Wires reachable from the pass inputs, which are the cotangents.'''
        tangent = set(self.graph.dom)
        changed = True
        while changed:
            changed = False
            for root in self.roots:
                if any(node in tangent for node in root.dom):
                    for node in root.cod:
                        if node not in tangent:
                            tangent.add(node)
                            changed = True
        return tangent


class _Variables:
    '''The index variables of one chain of einsums.

    Inlining a producer into the chain identifies the producer's result
    variables with the variables the chain reads its result at, and gives the
    producer's remaining variables, which are the ones it contracts, fresh
    names in the chain.
    '''
    def __init__(self) -> None:
        self._count = itertools.count()

    def fresh(self, axis: cat.Axis) -> es.IndexVariable:
        return es.IndexVariable(('chain', next(self._count)), axis)

    def rename(self, shape: Shape,
               mapping: dict[es.IndexVariable, es.IndexVariable]) -> Shape:
        renamed = []
        for variable in shape:
            if variable not in mapping:
                mapping[variable] = self.fresh(variable.axis)
            renamed.append(mapping[variable])
        return tuple(renamed)

    def inline(self, read_at: Shape, producer: RootShapes,
               ) -> tuple[Shape, ...]:
        '''The producer's operand shapes in the chain, given that the chain
        reads the producer's result at `read_at`.'''
        inputs, output = producer
        mapping = dict(zip(output, read_at))
        return tuple(self.rename(shape, mapping) for shape in inputs)


def _pattern_key(operands: Iterable[Operand], surviving: set[es.IndexVariable],
                 ) -> tuple[Key, dict[es.IndexVariable, int]]:
    '''A canonical form of the einsum over `operands` whose result keeps
    `surviving`, the same for any naming of the variables and any order of
    the operands, with the number each variable was given.

    Operands are ordered by wire, and two reads of one wire by which of their
    positions survive. Variables are numbered in order of first appearance
    down that list, so two einsums have one key exactly when a renaming and a
    reordering carry one onto the other.'''
    ordered = sorted(
        operands,
        key=lambda operand: (hash(operand[0]),
                             tuple(variable in surviving for variable in operand[1])))
    numbers: dict[es.IndexVariable, int] = {}
    pattern = []
    for node, shape in ordered:
        pattern.append((node, tuple(numbers.setdefault(variable, len(numbers))
                                    for variable in shape)))
    kept = frozenset(numbers[variable] for variable in surviving
                     if variable in numbers)
    return (tuple(pattern), kept), numbers


def _is_einsum_root(root: hg.Hypergraph) -> bool:
    return (isinstance(root, hg.HypergraphRoot)
            and es.is_einsum(root.wraps)
            and len(root.cod) == 1)


# ==========================================================================
# Linear arithmetic, and the inverse square split.
# ==========================================================================
def _input_free(formula: nm.Numeric) -> bool:
    return not any(True for _ in tutil.type_search(nm.FreeInput, formula))


def _formula_factor(formula: nm.Numeric, part: nm.Numeric) -> nm.Numeric | None:
    '''The constant c with `formula == c * part`, else None. Products are
    already canonical (`Multiplication.template` flattens and folds), so the
    match is literal: `part` itself, or a product with `part` as one factor
    and nothing else reading the input.'''
    if formula == part:
        return nm.Integer(1)
    match formula:
        case nm.Multiplication(content=factors):
            rest = tuple(factor for factor in factors if factor != part)
            if (len(rest) == len(factors) - 1
                    and all(_input_free(factor) for factor in rest)):
                return nm.Multiplication.template(*rest)
    return None


def linear_constant(formula: nm.Numeric) -> nm.Numeric | None:
    '''The c with `formula == c·x`.

    A chain passes through this one arithmetic, because scaling commutes with
    every einsum link.
    '''
    return _formula_factor(formula, nm.FreeInput())


def inverse_square_constant(formula: nm.Numeric) -> nm.Numeric | None:
    '''The c with `formula == c·x^{-2}`, the derivative of the reciprocal.'''
    return _formula_factor(
        formula, nm.Power(base=nm.FreeInput(), exponent=nm.Integer(-2)))


def _linear_root(root: hg.Hypergraph | None) -> nm.Numeric | None:
    '''The c of a root computing `x -> c·x`, else None.'''
    if (isinstance(root, hg.HypergraphRoot)
            and isinstance(root.wraps, cat.Broadcasted)
            and isinstance(root.wraps.operator, ops.Arithmetic)
            and len(root.dom) == 1 and len(root.cod) == 1):
        return linear_constant(root.wraps.operator.formula)
    return None


def split_inverse_squares[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''Rewrite every backward `Arithmetic<c·x^{-2}>` as a product the
    collapse can regroup: `x^{-1}`, copied, with one copy scaled by c,
    multiplied back together.

    The reciprocal rule writes `-x^{-2}` for its derivative. Written as a map
    of z, no rewrite in this module can act on it. Written as a product of
    reciprocals, every rewrite can. The reciprocal is the map the forward pass
    already applies to the same wire, so `migrate_drops` turns each copy into
    a grab of r, and the chain collapse then regroups the copies separately.
    The identity behind the rewrite is `g'(z) = γ(g(z))`, for the one map
    where γ is a product.

    The scale rides a linear `Arithmetic<c·x>`, which the unifier passes
    through. Every root outside a loop block is rewritten, like every rewrite
    here.
    '''
    graph = hg.Multigraph.from_morphism(taped.backward)
    edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {}
    for sub in _IndexedPass(graph).roots:
        constant = (
            inverse_square_constant(sub.wraps.operator.formula)
            if (isinstance(sub.wraps, cat.Broadcasted)
                and isinstance(sub.wraps.operator, ops.Arithmetic)
                and len(sub.dom) == 1 and len(sub.cod) == 1)
            else None)
        if constant is None:
            continue
        reciprocal = sub.wraps.reconstruct(
            operator=ops.Arithmetic(formula=1 / nm.x))
        copy = hg.HypergraphRoot.template(reciprocal, dom=sub.dom)
        value = tuple(reciprocal.cod())[0]
        shape = es.positional_shape(value)
        factor = copy.cod[0]
        pieces = [copy]
        if constant != nm.Integer(1):
            pieces.append(hg.HypergraphRoot.template(
                es.einsum((shape,), shape, datatype=value.datatype,
                          operator=ops.Arithmetic(formula=constant * nm.x)),
                dom=(factor,)))
        pieces.append(hg.HypergraphRoot.template(
            es.einsum((shape, shape), shape, datatype=value.datatype),
            dom=(factor, pieces[-1].cod[0]), cod=sub.cod))
        edits[sub] = tuple(pieces)
    if not edits:
        return taped
    return bp.Taped.from_passes(
        taped.forward, h2m.hypergraph_to_morphism(_rewritten(graph, edits)))


# ==========================================================================
# Pointwise formulas split at a value the forward pass holds.
# ==========================================================================
def _pointwise_formula(graph: _IndexedPass, root: hg.Hypergraph) -> nm.Numeric | None:
    '''The formula of a root that applies one `Arithmetic` to one operand and
    reads it at the positions of its result, else None.'''
    if not (isinstance(root, hg.HypergraphRoot)
            and isinstance(root.wraps, cat.Broadcasted)
            and isinstance(root.wraps.operator, ops.Arithmetic)
            and len(root.dom) == 1 and len(root.cod) == 1):
        return None
    shapes = graph.shapes(root)
    if shapes is None or shapes[0][0] != shapes[1]:
        return None
    return root.wraps.operator.formula


def _outer_function_of_forward_formula(
    composite: nm.Numeric,
    forward: fd.Prod[tuple[hg.HypergraphRoot, nm.Numeric]],
) -> tuple[hg.HypergraphRoot, nm.Numeric] | None:
    '''The first forward root whose formula `composite` is a function of, with
    that function.

    Every forward formula is tried as written before any is tried through
    `nm.expand_every_expandable`, so a match keeps the first-class numerics
    wherever the written forms allow one. When neither form matches, the
    caller leaves the backward formula as written.
    '''
    for root, formula in forward:
        outer = nm.outer_function(composite, formula)
        if outer is not None:
            return root, outer
    expanded = nm.expand_every_expandable(composite)
    for root, formula in forward:
        outer = nm.outer_function(expanded, nm.expand_every_expandable(formula))
        if outer is not None:
            return root, outer
    return None


def split_off_forward_formulas[L, M: cat.Morphism](
    taped: bp.Taped[L, M],
) -> bp.Taped[L, M]:
    '''Rewrite a backward `Arithmetic<f>` on a taped wire as the forward pass's
    `Arithmetic<g>` on that wire followed by `Arithmetic<h>`, wherever
    `f = h ∘ g`.

    The sigmoid is the case the rewrite is for. Its derivative map is
    `σ(x)(1 - σ(x))`, which is `h(σ(x))` for `h(y) = y(1 - y)`. Once the
    backward pass applies the forward map itself, `migrate_drops` grabs the
    forward value in place of the recomputation, so the backward pass reads
    `σ(x)` off the tape and computes `y(1 - y)`.

    `nm.outer_function` finds h. The formulas are compared as written first
    and through their expansions to primitives second, per
    `_outer_function_of_forward_formula`, and a backward formula that matches
    neither way keeps its first-class numerics.

    A forward value is used only when it is already taped or is a forward
    output, which is the gate `pathway_collapse` applies, because the rewrite
    adds a recomputation that `migrate_drops` then replaces with a read of the
    tape. A backward root that already applies the forward root's morphism is
    left to `migrate_drops`.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    fg, bg = _IndexedPass(forward_graph), _IndexedPass(backward_graph)
    taped_node = {root.wraps.tape: root.dom[0] for root in fg.roots
                  if isinstance(root.wraps, para.Drop)}
    kept = set(taped_node.values()) | set(forward_graph.cod)
    formulas_reading: dict[hg.HypergraphObject,
                           list[tuple[hg.HypergraphRoot, nm.Numeric]]] = {}
    for root in fg.roots:
        formula = _pointwise_formula(fg, root)
        if formula is not None and root.cod[0] in kept:
            formulas_reading.setdefault(root.dom[0], []).append((root, formula))
    grabs = {root.cod[0]: root.wraps.tape
             for root in bg.roots if isinstance(root.wraps, para.Grab)}

    edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {}
    for sub in bg.roots:
        composite = _pointwise_formula(bg, sub)
        if composite is None or sub.dom[0] not in grabs:
            continue
        candidates = tuple(
            (root, formula) for root, formula
            in formulas_reading.get(taped_node.get(grabs[sub.dom[0]]), ())
            if root.wraps != sub.wraps)
        found = (_outer_function_of_forward_formula(composite, candidates)
                 if candidates else None)
        if found is None:
            continue
        forward_root, outer = found
        if outer == nm.FreeInput():
            edits[sub] = hg.HypergraphRoot.template(
                forward_root.wraps, dom=sub.dom, cod=sub.cod)
            continue
        recompute = hg.HypergraphRoot.template(forward_root.wraps, dom=sub.dom)
        edits[sub] = (recompute, hg.HypergraphRoot.template(
            sub.wraps.reconstruct(operator=ops.Arithmetic(formula=outer)),
            dom=recompute.cod, cod=sub.cod))
    if not edits:
        return taped
    return bp.Taped.from_passes(
        taped.forward, h2m.hypergraph_to_morphism(_rewritten(backward_graph, edits)))


# ==========================================================================
# The forward catalog.
# ==========================================================================
@dataclass(frozen=True)
class Recallable:
    '''A forward wire expressible as one einsum over taped values.'''
    obj: hg.HypergraphObject          # the wire holding the value
    slot: para.TapeSlot | None        # its slot, if already taped
    is_output: bool
    out_shape: Shape                  # the value's positions, in the entry's variables
    numbers: Mapping[es.IndexVariable, int]   # those variables' numbers in the key

    def costs_no_new_memory(self) -> bool:
        '''Grabbing it needs no new memory the forward was not keeping.'''
        return self.slot is not None or self.is_output





def forward_catalog(
    graph: _IndexedPass,
) -> tuple[dict[Key, Recallable], dict[para.TapeSlot, hg.HypergraphObject]]:
    '''Every top-level forward wire that is one einsum over taped wires.

    The key is `_pattern_key` of the operand list and the surviving variables,
    which determines the einsum completely, because an einsum is a function of
    its shapes written in index variables.

    A chain of einsums unifies into one entry. A producer is inlined until the
    wire is taped, is an input, or is made by something that is not an einsum,
    so a nonlinearity stops the inlining. An entry is kept only when every one
    of its leaves is taped, because a slot is the only way the backward pass
    can name a value the forward pass computed.
    '''
    dropped: dict[hg.HypergraphObject, para.TapeSlot] = {}
    slot_to_node: dict[para.TapeSlot, hg.HypergraphObject] = {}
    for root in graph.roots:
        if isinstance(root.wraps, para.Drop):
            node = root.dom[0]
            dropped.setdefault(node, root.wraps.tape)
            slot_to_node[root.wraps.tape] = node

    outputs = set(graph.graph.cod)
    stops = set(dropped) | set(graph.graph.dom)

    def expression(node: hg.HypergraphObject, shape: Shape,
                   variables: _Variables) -> tuple[Operand, ...] | None:
        '''The wire, read at `shape`, as operands over stop wires.

        `None` where a leaf is not taped, or where anything on the way is not
        an einsum.
        '''
        if node in stops:
            if node not in dropped:
                return None
            return ((node, shape),) if _no_repeats(shape) else None
        root = graph.producer.get(node)
        return _operands(root, shape, variables) if root is not None else None

    def _operands(root: hg.Hypergraph, out_shape: Shape,
                  variables: _Variables) -> tuple[Operand, ...] | None:
        if not _is_einsum_root(root) or graph.shapes(root) is None:
            return None
        parts: tuple[Operand, ...] = ()
        for operand, shape in zip(root.dom,
                                  variables.inline(out_shape, graph.shapes(root))):
            inner = expression(operand, shape, variables)
            if inner is None:
                return None
            parts += inner
        return parts

    catalog: dict[Key, Recallable] = {}
    for root in graph.roots:
        if not _is_einsum_root(root) or graph.shapes(root) is None:
            continue
        node = root.cod[0]
        variables = _Variables()
        out_shape = variables.rename(graph.shapes(root)[1], {})
        if not _no_repeats(out_shape):
            continue
        # From the root's own operands. `expression(node)` would stop at the
        # output itself whenever it is taped, and a taped output is exactly
        # the entry worth having.
        operands = _operands(root, out_shape, variables)
        if operands is None or len(operands) < 2:
            continue
        key, numbers = _pattern_key(operands, set(out_shape))
        if any(variable not in numbers for variable in out_shape):
            continue
        entry = Recallable(graph.objects[node], dropped.get(node),
                           node in outputs, out_shape, numbers)
        # Keep the first entry for a key. The forward pass computes a value
        # once, so a second entry would name the same wire.
        catalog.setdefault(key, entry)
    return catalog, slot_to_node


# ==========================================================================
# The backward search.
# ==========================================================================
@dataclass(frozen=True)
class _TangentChain:
    '''A unified tangent chain, `constant · einsum(seed, *primals) -> out`,
    with every shape in the chain's variables.'''
    root: hg.HypergraphRoot
    seed: hg.HypergraphObject
    seed_shape: Shape
    primals: fd.Prod[Operand]
    out: hg.HypergraphObject
    out_shape: Shape
    constant: nm.Numeric


def _unify_einsum_chain(graph: _IndexedPass, tangent: set[hg.HypergraphObject],
           root: hg.HypergraphRoot) -> _TangentChain | None:
    '''Inline the tangent operand's einsum producers as far as they join.

    Every link has exactly one tangent operand, because a backward einsum is
    linear in the tangent, and every wire is free of repeated index variables.

    A link that cannot join ends the chain rather than causing a refusal. The
    wire above it becomes the seed, and the chain below it still regroups.

    A linear `Arithmetic<c·x>` on the tangent path is passed through and its
    constant is collected onto the chain. A primal made by an einsum of
    primals is flattened into its factors, so each factor is offered to the
    match separately.

    Each link's variables are renamed into the chain's as it is inlined, so a
    variable a link contracts is fresh to the chain. The chain reads each
    primal wire at one shape. A link that would read a wire the chain already
    reads, at another shape, ends the chain, because the two reads are the
    same value under two indexings, and the forward pass computed it once.
    `e ⊙ ⟨dy, e⟩ₓ` is the case: the pairing reads `e` contracted where the
    product reads it free, so the pairing is the seed of the outer regrouping.
    '''
    if not _is_einsum_root(root) or graph.shapes(root) is None:
        return None
    out = root.cod[0]
    if out not in tangent:
        return None
    variables = _Variables()
    out_shape = variables.rename(graph.shapes(root)[1], {})
    if not _no_repeats(out_shape):
        return None
    primals: list[Operand] = []
    constant: nm.Numeric = nm.Integer(1)
    read_at: dict[hg.HypergraphObject, Shape] = {}

    def reads_consistently(operands: Iterable[Operand]) -> bool:
        '''Whether every wire among `operands` is new to the chain or read at
        the shape the chain already reads it at.'''
        seen = dict(read_at)
        for node, shape in operands:
            if seen.setdefault(node, shape) != shape:
                return False
        return True

    def flattened(node: hg.HypergraphObject, shape: Shape) -> tuple[Operand, ...]:
        '''A primal as its einsum factors.

        The factors are returned when the producer is an einsum of primals
        with no repeated variable. Otherwise the primal is returned as it
        stands.
        '''
        producer = graph.producer.get(node)
        if (producer is None or not _is_einsum_root(producer)
                or graph.shapes(producer) is None):
            return ((node, shape),)
        operand_shapes = variables.inline(shape, graph.shapes(producer))
        if any(not _no_repeats(operand_shape) for operand_shape in operand_shapes):
            return ((node, shape),)
        parts = tuple(
            part for operand, operand_shape in zip(producer.dom, operand_shapes)
            for part in flattened(operand, operand_shape))
        if not reads_consistently(parts):
            return ((node, shape),)
        return parts

    frontier, frontier_shape = root, out_shape
    while True:
        operand_shapes = variables.inline(frontier_shape, graph.shapes(frontier))
        tangent_operands = [(node, shape)
                            for node, shape in zip(frontier.dom, operand_shapes)
                            if node in tangent]
        link_primals = tuple(
            part for node, shape in zip(frontier.dom, operand_shapes)
            if node not in tangent for part in flattened(node, shape))
        joins = (len(tangent_operands) == 1
                 # A primal with a repeated variable is a written diagonal,
                 # and an einsum cannot express it, so the link stands apart.
                 and all(_no_repeats(shape)
                         for node, shape in zip(frontier.dom, operand_shapes)
                         if node not in tangent)
                 and reads_consistently(link_primals))
        if not joins:
            if frontier is root:
                return None
            seed, seed_shape = frontier.cod[0], frontier_shape
            break
        primals.extend(link_primals)
        for node, shape in link_primals:
            read_at.setdefault(node, shape)
        head, head_shape = tangent_operands[0]
        # ride through any linear scaling: it commutes with every link, and a
        # pointwise map keeps every position
        producer = graph.producer.get(head)
        while (scaling := _linear_root(producer)) is not None:
            constant = constant * scaling
            head = producer.dom[0]
            producer = graph.producer.get(head)
        if (producer is None or not _is_einsum_root(producer)
                or graph.shapes(producer) is None
                or not _no_repeats(head_shape)):
            seed, seed_shape = head, head_shape
            break
        frontier, frontier_shape = producer, head_shape
    return _TangentChain(root, seed, seed_shape, tuple(primals), out, out_shape,
                         constant)


def _subsets(count: int) -> Iterable[tuple[int, ...]]:
    '''Non-trivial index subsets, largest first, in a deterministic order.'''
    from itertools import combinations
    for size in range(count, 1, -1):
        yield from combinations(range(count), size)


@dataclass(frozen=True)
class Recalled:
    '''One rewrite: what was matched, and the slot it reads through.'''
    value: hg.HypergraphObject
    slot: para.TapeSlot
    fresh: bool                      # True when the forward gained the Drop


def _see_through(
    graph: _IndexedPass, node: hg.HypergraphObject,
) -> tuple[hg.HypergraphObject, nm.Numeric]:
    '''A wire through any linear scalings that made it: the wire underneath
    and the accumulated constant.'''
    constant: nm.Numeric = nm.Integer(1)
    while (scaling := _linear_root(graph.producer.get(node))) is not None:
        constant = constant * scaling
        node = graph.producer[node].dom[0]
    return node, constant


@dataclass(frozen=True)
class _Match:
    '''The grouping of a chain's primals the forward pass has, the entry
    holding it, the constant the rewrite owes, and the recalled value's
    positions in the chain's variables.'''
    subset: tuple[int, ...]
    entry: Recallable
    scale: nm.Numeric
    grab_shape: Shape


def _match_forward_catalog(
    graph: _IndexedPass,
    chain: _TangentChain,
    grabs: dict[hg.HypergraphObject, para.TapeSlot],
    slot_to_forward: dict[para.TapeSlot, hg.HypergraphObject],
    catalog: dict[Key, Recallable],
) -> _Match | None:
    '''The largest grouping of the chain's primals the forward has, with the
    constant the rewrite owes: the chain's own and that of every linear
    scaling the grouped primals were seen through.

    A primal is a grab of a forward wire, so the group is written over forward
    wires at the chain's shapes and looked up by `_pattern_key`. The key's
    numbering pairs the entry's variables with the chain's, which is how the
    recalled value is read back at the chain's positions.'''
    as_forward: list[Operand | None] = []
    constants: list[nm.Numeric] = []
    for node, shape in chain.primals:
        base, constant = _see_through(graph, node)
        slot = grabs.get(base)
        forward_node = slot_to_forward.get(slot) if slot is not None else None
        as_forward.append((forward_node, shape) if forward_node is not None else None)
        constants.append(constant)

    seed_axes = set(chain.seed_shape)
    out_axes = set(chain.out_shape)
    for subset in _subsets(len(chain.primals)):
        if any(as_forward[i] is None for i in subset):
            continue
        kept = [i for i in range(len(chain.primals)) if i not in subset]
        outside = (seed_axes | out_axes
                   | {variable for i in kept for variable in chain.primals[i][1]})
        group_axes = {variable for i in subset for variable in as_forward[i][1]}
        key, numbers = _pattern_key(
            [as_forward[i] for i in subset], group_axes & outside)
        entry = catalog.get(key)
        if entry is not None and entry.costs_no_new_memory():
            scale = chain.constant
            for i in subset:
                scale = scale * constants[i]
            by_number = {number: variable for variable, number in numbers.items()}
            grab_shape = tuple(by_number[entry.numbers[variable]]
                               for variable in entry.out_shape)
            return _Match(subset, entry, scale, grab_shape)
    return None


# ==========================================================================
# The rewrite.
# ==========================================================================
def _hoisted_scope(
    graph: _IndexedPass, root: hg.HypergraphRoot,
    operands: Iterable[hg.HypergraphObject],
) -> rewire_blocks.BlockChain:
    '''The scope one block out of `root`'s when no operand is produced inside
    that block, and `root`'s own scope otherwise.'''
    own = graph.chain[root]
    if not own:
        return own
    for wire in operands:
        producer = graph.producer.get(wire)
        if producer is not None and graph.chain[producer][:len(own)] == own:
            return own
    return own[:-1]


def _rewritten[L, M: cat.Morphism](
    graph: hg.Multigraph[L, M],
    edits: Mapping[hg.HypergraphRoot[L, M], replace_roots.Replacement[L, M]],
    placements: Mapping[rewire_blocks.BlockChain, Iterable[hg.Hypergraph[L, M]]]
    = {},
) -> hg.Multigraph[L, M]:
    '''`graph` with `edits` applied in place, `placements` appended to their
    scopes, and every block boundary rederived.'''
    return rewire_blocks.recompute_block_boundaries(rewire_blocks.insert_roots(
        replace_roots.replace_roots(graph, edits), placements))


def pathway_collapse[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''Grab every backward chain the forward pass already computed.

    The pathway through the backward pass collapses onto the forward pass's
    value. Applied to a derived and simplified attention, the rewrite produces
    the FlashAttention D-trick. The forward pass gains `Drop(O)`, and the
    backward pass's row statistic becomes `<dO, O>`, hoisted out of the
    `R[SoftMax]` block in the way FlashAttention hoists it into a separate
    preprocessing kernel. A pair with nothing to recall is returned unchanged.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    changed = False

    for _ in range(COLLAPSE_LIMIT):
        fg, bg = _IndexedPass(forward_graph), _IndexedPass(backward_graph)
        catalog, slot_to_node = forward_catalog(fg)
        if not catalog:
            break
        grabs = {root.cod[0]: root.wraps.tape
                 for root in bg.roots if isinstance(root.wraps, para.Grab)}
        tangent = bg.tangent_closure()

        step = None
        for root in bg.roots:
            chain = _unify_einsum_chain(bg, tangent, root)
            if chain is None:
                continue
            found = _match_forward_catalog(bg, chain, grabs, slot_to_node, catalog)
            if found is not None:
                step = (chain, found)
                break
        if step is None:
            break
        chain, match = step
        subset, entry, scale = match.subset, match.entry, match.scale
        changed = True

        # -- forward: tape the matched wire, unless it already is
        if entry.slot is None:
            slot = para.new_slot()
            drop = hg.HypergraphRoot.template(
                para.Drop(tape=slot, size=entry.obj.obj),
                dom=(entry.obj,))
            forward_graph = _rewritten(
                forward_graph, {}, {fg.chain[fg.producer[entry.obj]]: (drop,)})
        else:
            slot = entry.slot

        # -- backward: the chain's last einsum, rebuilt to read the grab, and
        # placed one block out of the chain's root when every wire it reads
        # comes from outside that block, which is where FlashAttention keeps
        # its row statistic. A wire produced inside the block keeps the
        # rewrite inside it, because a root outside reading the block's
        # output and feeding its input would be a cycle. An existing grab of
        # the slot in that scope is reused: a second `Grab` of one slot would
        # be a second read the derivation never called for.
        kept = tuple(i for i in range(len(chain.primals)) if i not in subset)
        scope = _hoisted_scope(
            bg, chain.root, (chain.seed, *(chain.primals[i][0] for i in kept)))
        grab_value = entry.obj.obj
        existing = next(
            (r for r in bg.roots
             if isinstance(r.wraps, para.Grab) and r.wraps.tape == slot
             and bg.chain[r] == scope),
            None)
        grab = existing or hg.HypergraphRoot.template(
            para.Grab(tape=slot, size=grab_value))
        out_array = bg.array(chain.out)
        out_shape = chain.out_shape
        # The recalled value is read at the chain's variables, each position
        # keeping the value's own axis.
        grab_shape = tuple(
            es.IndexVariable(variable.label, axis)
            for variable, axis in zip(match.grab_shape, grab_value.shape()))
        operands: tuple[Operand, ...] = (
            (chain.seed, chain.seed_shape),
            *(chain.primals[i] for i in kept),
            (grab.cod[0], grab_shape),
        )
        replacements: list[hg.HypergraphRoot] = []

        # A factor carrying no contracted axis multiplies the result of the
        # contraction rather than its summand. It is hoisted into a second
        # einsum, so that a later round can regroup it with the factors above
        # this chain.
        contracted = ({axis for _, shape in operands for axis in shape}
                      - set(out_shape))
        inner = tuple(op for op in operands if set(op[1]) & contracted)
        outer = tuple(op for op in operands if not set(op[1]) & contracted)
        if contracted and inner and outer:
            inner_axes = {axis for _, shape in inner for axis in shape}
            stat_shape = tuple(axis for axis in out_shape
                               if axis in inner_axes)
            stat = hg.HypergraphRoot.template(
                es.einsum(tuple(shape for _, shape in inner), stat_shape,
                          datatype=out_array.datatype),
                dom=tuple(node for node, _ in inner))
            replacements.append(stat)
            operands = ((stat.cod[0], stat_shape), *outer)

        # The constant the chain collected, reapplied once at its end.
        merged = es.einsum(tuple(shape for _, shape in operands), out_shape,
                           datatype=out_array.datatype)
        if scale == nm.Integer(1):
            replacements.append(hg.HypergraphRoot.template(
                merged, dom=tuple(node for node, _ in operands),
                cod=(chain.out,)))
        else:
            product = hg.HypergraphRoot.template(
                merged, dom=tuple(node for node, _ in operands))
            replacements += [product, hg.HypergraphRoot.template(
                es.einsum((out_shape,), out_shape,
                          datatype=out_array.datatype,
                          operator=ops.Arithmetic(formula=scale * nm.x)),
                dom=product.cod, cod=(chain.out,))]

        placed = (*replacements, *(() if existing is not None else (grab,)))
        backward_graph = _rewritten(
            backward_graph, {chain.root: None}, {scope: placed})
    else:
        raise RuntimeError(f'{COLLAPSE_LIMIT} recalls without a fixed point')

    if not changed:
        return taped
    forward_graph, backward_graph = _collect_dead_roots(forward_graph, backward_graph)
    return bp.Taped.from_passes(h2m.hypergraph_to_morphism(forward_graph),
                    h2m.hypergraph_to_morphism(backward_graph))


def _scope_of_roots(
    graph: hg.Hypergraph, scope: hg.Multigraph | None = None,
) -> dict[hg.HypergraphRoot, hg.Multigraph]:
    '''The `Multigraph` each root sits in directly.'''
    match graph:
        case hg.HypergraphRoot():
            return {graph: scope}
        case hg.HypergraphBlock(body=body):
            return _scope_of_roots(body, scope)
        case hg.Multigraph():
            found: dict[hg.HypergraphRoot, hg.Multigraph] = {}
            for subgraph in graph.subgraphs():
                found |= _scope_of_roots(subgraph, graph)
            return found
    return {}


def dedup_slots[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''
    Give each wire one slot.

    `backprop` tapes a residual once per consumer. Attention drops the softmax
    output twice, once as `SoftMax`'s output residual and once as the second
    einsum's input residual, because each seed declares its own residual
    a-priori with no view of the others.

    The morphism cannot remove the duplicate, because `cat.Array` compares
    structurally and two distinct wires of one shape are equal terms. The
    graph can, because a wire there is a `HypergraphObject`.

    Two `Drop`s of one wire become the one with the lowest slot number. The
    backward pass's `Grab` of a removed slot is deleted, and its output node
    is canonicalised onto the kept grab's node. Consumers follow the node
    wherever they sit, including inside a block.

    A slot the model named, which is any slot outside the `s{n}` numbering,
    is kept ahead of every residual slot. The model's drop is then the one
    save, and where the backward pass has no grab of that slot yet, the grab
    of the removed residual slot is pointed at it rather than deleted. A
    complete `TopK` that drops its index to a slot of its own is the case:
    `deepseek.registries.derivative` declares the index as the residual, and
    the reverse pass ends up grabbing the slot the model wrote.

    Both passes are walked through their blocks. A grab whose kept twin sits
    in another block is pointed at the kept slot the same way, rather than
    deleted, because canonicalising its wire onto a grab in another block
    would leave its consumers reading a wire their block's domain does not
    carry. A slot is then saved once and grabbed once per block that reads
    it, and `dedup_roots` merges those grabs afterwards.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    saves: dict[hg.HypergraphObject, list[hg.HypergraphRoot]] = {}
    for root in replace_roots.walk_roots(forward_graph):
        if isinstance(root.wraps, para.Drop):
            saves.setdefault(root.dom[0], []).append(root)

    def rank(root: hg.HypergraphRoot) -> tuple[int, int]:
        name = root.wraps.tape.uid._name.to_bodies()
        if name.startswith('s') and name[1:].isdigit():
            return (1, int(name[1:]))
        return (0, 0)

    removed: dict[para.TapeSlot, para.TapeSlot] = {}
    dead: set[hg.HypergraphRoot] = set()
    for roots in saves.values():
        if len(roots) < 2:
            continue
        kept, *rest = sorted(roots, key=rank)
        for root in rest:
            removed[root.wraps.tape] = kept.wraps.tape
            dead.add(root)
    if not removed:
        return taped

    forward_graph = replace_roots.replace_roots(
        forward_graph, {root: None for root in dead})

    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    grabs = {root.wraps.tape: root
             for root in replace_roots.walk_roots(backward_graph)
             if isinstance(root.wraps, para.Grab)}
    scopes = _scope_of_roots(backward_graph)
    equalities = []
    edits: dict[hg.HypergraphRoot, hg.HypergraphRoot | None] = {}
    for gone, kept_slot in removed.items():
        gone_grab, kept_grab = grabs.get(gone), grabs.get(kept_slot)
        assert gone_grab is not None, f'a taped slot with no reader: {gone}'
        if kept_grab is None or scopes[gone_grab] is not scopes[kept_grab]:
            edits[gone_grab] = gone_grab.reconstruct(
                wraps=gone_grab.wraps.reconstruct(tape=kept_slot))
            continue
        edits[gone_grab] = None
        equalities.append(fd.UIDRenaming.set_canonical(
            kept_grab.cod[0], gone_grab.cod[0]))
    backward_graph = fd.Context(equalities).apply(
        replace_roots.replace_roots(backward_graph, edits))

    return bp.Taped.from_passes(h2m.hypergraph_to_morphism(forward_graph),
                    h2m.hypergraph_to_morphism(backward_graph))


def dedup_roots[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''
    Give each value one root in each pass.

    `backprop` writes one `Grab` per reader of a slot, because each rule reads
    its residual with no view of the others, and the expanded DeepSeek-V3 layer
    grabs its index in four places and its normalised tokens in five. A pair
    of roots that wrap equal morphisms on the same wires compute one value,
    and `merge_duplicate_roots` keeps one of them, in the innermost block that
    encloses every reader. A loop block is a boundary, so a value inside one is
    computed once per iteration and never shared with a root outside it.

    It runs after `dedup_slots`, because two grabs of one residual compare
    equal only once they read one slot.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    merged_forward = merge_duplicate_roots.merge_duplicate_roots(forward_graph)
    merged_backward = merge_duplicate_roots.merge_duplicate_roots(backward_graph)
    if merged_forward is forward_graph and merged_backward is backward_graph:
        return taped
    return bp.Taped.from_passes(h2m.hypergraph_to_morphism(merged_forward),
                                h2m.hypergraph_to_morphism(merged_backward))


def _collect_dead_roots(
    forward_graph: hg.Multigraph,
    backward_graph: hg.Multigraph,
) -> tuple[hg.Multigraph, hg.Multigraph]:
    '''
    Collect the roots a rewrite has orphaned.

    A rewrite orphans three things. The first is the inner links of a
    collapsed chain, when the final einsum of the chain was their only
    consumer. The second is the recomputation that a migrated drop replaced.
    The third is an earlier round's `Grab` whose consumer was itself
    rewritten.

    A root that nothing reads cannot remain, because `hypergraph_to_morphism`
    has no wire to attach it to and raises `HypergraphObject(...) is not in list`. A
    grab that nothing reads should not keep its slot alive.

    Dead roots are collected first, at any block depth, to a fixed point,
    because removing one can orphan its operands. Any `Drop` whose slot no
    remaining grab reads is collected after. The residuals `backprop` taped
    always keep their readers, so this collects only what the rewrites here
    made dead. Both graphs come back with their block boundaries rederived.
    '''
    while True:
        bg = _IndexedPass(backward_graph)
        consumed = bg.consumed_wires()
        dead = {root for root in bg.roots
                if root.cod and not any(node in consumed for node in root.cod)}
        if not dead:
            break
        backward_graph = replace_roots.replace_roots(
            backward_graph, {root: None for root in dead})

    grabbed = {root.wraps.tape for root in _IndexedPass(backward_graph).roots
               if isinstance(root.wraps, para.Grab)}
    fg = _IndexedPass(forward_graph)
    unread = {root for root in fg.roots
              if isinstance(root.wraps, para.Drop)
              and root.wraps.tape not in grabbed}
    forward_graph = replace_roots.replace_roots(
        forward_graph, {root: None for root in unread})
    return (rewire_blocks.recompute_block_boundaries(forward_graph),
            rewire_blocks.recompute_block_boundaries(backward_graph))


# ==========================================================================
# Drop migration: the tape moves through a deterministic map.
# ==========================================================================
def _single_output(root: hg.Hypergraph) -> bool:
    return (isinstance(root, hg.HypergraphRoot)
            and not isinstance(root.wraps, para.ParaMorphism)
            and len(root.cod) == 1)


def migrate_drops[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''
    Replace a backward recomputation of a forward map with a tape of the
    forward pass's result.

    The forward pass copies X to `F` and to `Drop<s>`, and the backward pass
    then applies the same `F` to `Grab<s>`. In a Cartesian category, copying
    commutes with any deterministic map, since `copy ; (F x F) = F ; copy`,
    so the drop may sit after `F` instead. The forward pass tapes `F(X)`, and
    the backward pass grabs it where it used to recompute it.

    Randomness is abstracted into generators for backpropagation, so no
    operator here breaks the premise of determinism.

    The rewrite assumes no linearity. It is the most general of the rewrites
    and runs first.

    "The same `F`" is a structural test. The backward root's morphism is `==`
    to the forward pass's, because a derivative rule rebuilds the map through
    `einops_simplification.einsum` on the same arrays, so the rebuilt
    `Arithmetic<e^{x}>` compares equal to the forward pass's. Every operand of the
    backward root is a `Grab` whose slot tapes the corresponding forward operand.

    When `F(X)` is already taped the existing slot is reused.
    `_collect_dead_roots` then removes the grab of `s` when nothing else
    consumes it, and the drop of `s` after that, which is the drop having
    moved. On the expanded softmax the reuse is what lets `s0` and `s3` share
    one tape.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    changed = False

    for _ in range(COLLAPSE_LIMIT):
        fg, bg = _IndexedPass(forward_graph), _IndexedPass(backward_graph)
        slot_to_node = {root.wraps.tape: root.dom[0] for root in fg.roots
                        if isinstance(root.wraps, para.Drop)}
        node_to_slot = {node: slot for slot, node in slot_to_node.items()}
        grabs = {root.cod[0]: root.wraps.tape
                 for root in bg.roots if isinstance(root.wraps, para.Grab)}
        forward_roots = [root for root in fg.roots if _single_output(root)]

        step = None
        for root in bg.roots:
            if not _single_output(root):
                continue
            operands = tuple(slot_to_node.get(grabs.get(node))
                             for node in root.dom)
            if any(node is None for node in operands):
                continue
            twin = next((f for f in forward_roots
                         if f.dom == operands and f.wraps == root.wraps),
                        None)
            if twin is not None:
                step = (root, twin)
                break
        if step is None:
            break
        root, twin = step
        changed = True

        # -- forward: tape F(X), unless it already is
        value = twin.cod[0]
        slot = node_to_slot.get(value)
        if slot is None:
            slot = para.new_slot()
            drop = hg.HypergraphRoot.template(
                para.Drop(tape=slot, size=fg.array(value)),
                dom=(value,))
            forward_graph = _rewritten(
                forward_graph, {}, {fg.chain[twin]: (drop,)})

        # -- backward: the recomputation becomes a grab onto the same node,
        # or the node is folded onto an existing grab's in the same scope
        existing = next(
            (r for r in bg.roots if isinstance(r.wraps, para.Grab)
             and r.wraps.tape == slot and bg.chain[r] == bg.chain[root]),
            None)
        out = root.cod[0]
        if existing is None:
            grab = hg.HypergraphRoot.template(
                para.Grab(tape=slot, size=bg.array(out)), cod=(out,))
            backward_graph = _rewritten(backward_graph, {root: grab})
        else:
            backward_graph = fd.Context([fd.UIDRenaming.set_canonical(
                existing.cod[0], out)]).apply(
                    _rewritten(backward_graph, {root: None}))
        forward_graph, backward_graph = _collect_dead_roots(
            forward_graph, backward_graph)
    else:
        raise RuntimeError(f'{COLLAPSE_LIMIT} migrations without a fixed point')

    if not changed:
        return taped
    return bp.Taped.from_passes(h2m.hypergraph_to_morphism(forward_graph),
                    h2m.hypergraph_to_morphism(backward_graph))


# ==========================================================================
# Slots an elementwise apart: save one, recompute the other.
# ==========================================================================
def _recomputable(morphism: cat.Morphism) -> bool:
    '''Whether the backward pass may re-run this pointwise map.

    An `Elementwise` qualifies, covering both the `Arithmetic` formulas and
    the named maps. A `View` does not, because it is a reindexing and performs
    no computation. A `Dropout` does not either.

    `migrate_drops` may assume determinism, because it only removes a
    recomputation the derivation had already written. This pass introduces
    one, and re-running a dropout draws a fresh mask unless its generator is a
    wire, which nothing yet makes it.
    '''
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, ops.Elementwise)
            and not isinstance(morphism.operator, (ops.View, ops.Dropout)))


def recompute_elementwise_slots[L, M: cat.Morphism](
    taped: bp.Taped[L, M],
) -> bp.Taped[L, M]:
    '''
    Reduce two slots one elementwise map apart to one slot.

    The earlier slot stays taped, the later slot's `Drop` is deleted, and the
    backward pass computes the value at the grab, writing `Grab<s> ; f` where
    it read `Grab<t>`, for a forward pass in which `t = f(s)`.

    The rewrite trades a save for a pointwise recomputation, which is
    selective activation recomputation as it is practised elsewhere. A fused
    feed-forward kernel stores the pre-activation and re-runs the ReLU in the
    backward pass rather than keeping both `q x f` arrays.

    The pass is the opposite policy to `migrate_drops`. Both resolve the same
    pattern, which is two values a deterministic map apart. `migrate_drops`
    spends memory to remove a recomputation, and this pass spends a
    recomputation to remove a slot. Which of the two is correct depends on the
    cost of tape bytes against recompute operations, so this pass is explicit
    and `collapse` does not run it.

    Graph-based like everything here, because "the same wire" is a question
    about nodes, and reaching into every block outside a loop like
    `migrate_drops`. Every backward grab of the dead slot is redirected onto
    the recomputation.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    changed = False

    for _ in range(COLLAPSE_LIMIT):
        fg, bg = _IndexedPass(forward_graph), _IndexedPass(backward_graph)
        drops = {root.dom[0]: root for root in fg.roots
                 if isinstance(root.wraps, para.Drop)}
        grabs: dict[para.TapeSlot, list[hg.HypergraphRoot]] = {}
        for root in bg.roots:
            if isinstance(root.wraps, para.Grab):
                grabs.setdefault(root.wraps.tape, []).append(root)

        step = None
        for root in fg.roots:
            if (not _recomputable(root.wraps)
                    or len(root.dom) != 1 or len(root.cod) != 1):
                continue
            source, result = root.dom[0], root.cod[0]
            if (source in drops and result in drops
                    and drops[result].wraps.tape in grabs):
                step = (drops[source].wraps.tape, drops[result], root)
                break
        if step is None:
            break
        kept_slot, dead_drop, seed = step
        changed = True

        # Forward: delete the later save. The earlier one already stands.
        forward_graph = _rewritten(forward_graph, {dead_drop: None})

        # -- backward: each grab of the dead slot becomes the earlier slot's
        # grab with the map re-applied, its readers redirected onto the
        # recomputation's fresh node.
        dead_slot = dead_drop.wraps.tape
        replaced = grabs[dead_slot]
        source_array = fg.array(seed.dom[0])
        edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {}
        equalities = []
        for old in replaced:
            grab = hg.HypergraphRoot.template(
                para.Grab(tape=kept_slot, size=source_array))
            recompute = hg.HypergraphRoot.template(
                seed.wraps, dom=grab.cod)
            edits[old] = (grab, recompute)
            equalities.append(fd.UIDRenaming.set_canonical(
                recompute.cod[0], old.cod[0]))
        backward_graph = fd.Context(equalities).apply(
            _rewritten(backward_graph, edits))

    else:
        raise RuntimeError(
            f'{COLLAPSE_LIMIT} recomputations without a fixed point')

    if not changed:
        return taped
    return bp.Taped.from_passes(h2m.hypergraph_to_morphism(forward_graph),
                    h2m.hypergraph_to_morphism(backward_graph))


# ==========================================================================
# Additions after einsums.
# ==========================================================================
def _distribute_once(graph: hg.Multigraph) -> hg.Multigraph | None:
    '''One `einsum(p, a + b)` rewritten as `einsum(p, a) + einsum(p, b)`, or
    None when no einsum reads an addition.'''
    g = _IndexedPass(graph)
    outputs = set(graph.cod)
    readers = Counter(node for root in g.roots for node in root.dom)
    for reader in g.roots:
        if not _is_einsum_root(reader):
            continue
        for position, summed in enumerate(reader.dom):
            adder = g.producer.get(summed)
            if (adder is None
                    or not isinstance(adder.wraps, cat.Broadcasted)
                    or not isinstance(adder.wraps.operator, ops.AdditionOp)
                    or len(adder.cod) != 1
                    or reader.dom.count(summed) != 1
                    or readers[summed] != 1 or summed in outputs):
                continue
            if g.shapes(reader) is None or g.shapes(adder) is None:
                continue
            in_shapes, out_shape = g.shapes(reader)
            summand_shapes = _Variables().inline(in_shapes[position], g.shapes(adder))
            elsewhere = set(out_shape) | {
                axis for i, shape in enumerate(in_shapes) if i != position
                for axis in shape}
            # A variable the sum broadcast an operand along, and which the
            # einsum then contracts with nothing else carrying it, would
            # vanish with its multiplicity. Refused rather than written as a
            # size factor.
            summed_axes = set(in_shapes[position])
            parts = []
            for operand, shape in zip(adder.dom, summand_shapes):
                if (summed_axes - set(shape)) - elsewhere:
                    break
                parts.append((operand, shape))
            else:
                datatype = g.array(reader.cod[0]).datatype
                pieces = []
                for operand, shape in parts:
                    shapes = tuple(shape if i == position else s
                                   for i, s in enumerate(in_shapes))
                    pieces.append(hg.HypergraphRoot.template(
                        es.einsum(shapes, out_shape, datatype),
                        dom=tuple(operand if i == position else node
                                   for i, node in enumerate(reader.dom))))
                total = hg.HypergraphRoot.template(
                    es.einsum((out_shape,) * len(pieces), out_shape, datatype,
                              operator=ops.AdditionOp()),
                    dom=tuple(piece.cod[0] for piece in pieces),
                    cod=(reader.cod[0],))
                return _rewritten(graph, {reader: (*pieces, total), adder: None})
    return None


def distribute_additions[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''
    Move every addition an einsum reads to after the einsum:

        einsum(p, a + b)  ->  einsum(p, a) + einsum(p, b)

    The rewrite is bilinearity, applied in the direction that makes chains longer.

    A sum is where a tangent chain ends, because `_unify_einsum_chain`
    requires one tangent operand per link. A backward pass that adds first and
    multiplies after therefore conceals its chains from the collapse.
    Distributing first exposes each summand's chain all the way back to its
    seed, and the pathway collapse is then offered the most general form.

    Factoring a common multiplicand back out is `factor_additions`' work,
    after the collapse.

    Applied to the backward only, to a fixed point. An einsum reading the sum
    twice (quadratic in it) is left alone, as is a sum whose operand was
    broadcast along an axis the einsum would contract away unaccompanied.
    A sum with another reader is left alone too. Distributing it would keep
    the sum for that reader and copy each summand into the einsum beside it,
    and `factor_additions` cannot fold a copy back while the original stands.
    On the DeepSeek-V3 layer the cotangent entering each RMSNorm is a sum the
    gain gradient reads as well, and distributing it left the backward pass
    with 89 einsums where it had 61.
    '''
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    changed = False
    for _ in range(COLLAPSE_LIMIT):
        rewritten = _distribute_once(backward_graph)
        if rewritten is None:
            break
        backward_graph, changed = rewritten, True
    else:
        raise RuntimeError(f'{COLLAPSE_LIMIT} distributions without a fixed point')
    if not changed:
        return taped
    return bp.Taped.from_passes(
        taped.forward, h2m.hypergraph_to_morphism(backward_graph))


# ==========================================================================
# Factoring additions back, once the collapse is done.
# ==========================================================================
def _sink_once(graph: hg.Multigraph) -> hg.Multigraph | None:
    '''Move one linear `Arithmetic<c·x>` from an einsum's output onto a
    strictly smaller operand, as `-(einsum(s, y)) -> einsum(-s, y)`.

    The rewrite is sound because the scale commutes with the einsum. It is
    taken only when the operand is smaller, so the pointwise work shrinks.
    The smaller operand is also what exposes the factorable form to
    `_factor_once`. Returns `None` when nothing sinks.
    '''
    g = _IndexedPass(graph)
    readers = Counter(node for root in g.roots for node in root.dom)
    outputs = set(graph.cod)
    for root in g.roots:
        if _linear_root(root) is None:
            continue
        source = root.dom[0]
        producer = g.producer.get(source)
        if (producer is None or not _is_einsum_root(producer)
                or readers[source] != 1 or source in outputs):
            continue
        if g.shapes(producer) is None:
            continue
        shapes = list(g.shapes(producer)[0])
        out_array = g.array(root.cod[0])
        out_shape = g.shapes(producer)[1]
        if not all(_no_repeats(s) for s in (*shapes, out_shape)):
            continue
        target = min(range(len(shapes)), key=lambda i: (len(shapes[i]), i))
        if len(shapes[target]) >= len(out_shape):
            continue
        scaled = hg.HypergraphRoot.template(
            es.einsum((shapes[target],), shapes[target],
                      datatype=out_array.datatype,
                      operator=root.wraps.operator),
            dom=(producer.dom[target],))
        merged = hg.HypergraphRoot.template(
            es.einsum(tuple(shapes), out_shape, datatype=out_array.datatype),
            dom=tuple(scaled.cod[0] if i == target else node
                       for i, node in enumerate(producer.dom)),
            cod=root.cod)
        return _rewritten(graph, {producer: None, root: (scaled, merged)})
    return None


def _factor_once(graph: hg.Multigraph) -> hg.Multigraph | None:
    '''Move the common factor out of one addition of einsums, as
    `einsum(p, a) + einsum(p, b) -> einsum(p, a + b)`.

    The rewrite is the inverse of `_distribute_once`, and is taken only when
    it strictly reduces the einsum count. Each residue keeps the axes it
    shares with the factor or with the output, because those axes pair up in
    the outer einsum, and pre-contracts the rest. The addition broadcasts
    residues of different shapes, which is how the scalar `-<dy, y>` meets
    `dy` at full degree. Returns `None` when nothing factors.
    '''
    g = _IndexedPass(graph)
    readers = Counter(node for root in g.roots for node in root.dom)
    outputs = set(graph.cod)
    for root in g.roots:
        wraps = root.wraps
        if (not isinstance(wraps, cat.Broadcasted)
                or not isinstance(wraps.operator, ops.AdditionOp)
                or len(root.cod) != 1):
            continue
        summands = root.dom
        if len(summands) < 2 or len(set(summands)) != len(summands):
            continue
        producers = [g.producer.get(node) for node in summands]
        if any(producer is None or not _is_einsum_root(producer)
               or readers[node] != 1 or node in outputs
               for producer, node in zip(producers, summands)):
            continue
        if g.shapes(root) is None or any(g.shapes(p) is None for p in producers):
            continue
        out_array = g.array(root.cod[0])
        adder_inputs, out_shape = g.shapes(root)
        variables = _Variables()
        operand_lists = [
            list(zip(producer.dom, variables.inline(read_at, g.shapes(producer))))
            for producer, read_at in zip(producers, adder_inputs)]
        if not all(_no_repeats(shape) for operands in operand_lists
                   for _, shape in operands) or not _no_repeats(out_shape):
            continue
        operand_lists = _align_contracted_variables(
            operand_lists, set(out_shape) | {v for s in adder_inputs for v in s})

        shared = Counter(operand_lists[0])
        for operands in operand_lists[1:]:
            shared &= Counter(operands)
        if not shared:
            continue
        # the factor, in the first summand's operand order
        take = Counter(shared)
        factor: list[Operand] = []
        for item in operand_lists[0]:
            if take[item] > 0:
                take[item] -= 1
                factor.append(item)
        residues: list[list[Operand]] = []
        for operands in operand_lists:
            take = Counter(shared)
            residue = []
            for item in operands:
                if take[item] > 0:
                    take[item] -= 1
                else:
                    residue.append(item)
            residues.append(residue)
        if any(not residue for residue in residues):
            continue

        # A residue variable the factor or the output carries pairs up in the
        # outer einsum and survives. Anything else contracts in the residue.
        allowed = list(dict.fromkeys(
            (*out_shape, *(axis for _, shape in factor for axis in shape))))
        parts: list[Operand] = []
        fresh: list[hg.HypergraphRoot] = []
        for residue in residues:
            axes = {axis for _, shape in residue for axis in shape}
            surviving = tuple(axis for axis in allowed if axis in axes)
            # A lone operand already at the surviving axes is its own
            # residue. The order does not matter, because an einsum is a
            # function of its shapes.
            if len(residue) == 1 and set(residue[0][1]) == set(surviving):
                parts.append(residue[0])
                continue
            inner = hg.HypergraphRoot.template(
                es.einsum(tuple(shape for _, shape in residue), surviving,
                          datatype=out_array.datatype),
                dom=tuple(node for node, _ in residue))
            fresh.append(inner)
            parts.append((inner.cod[0], surviving))
        if len(fresh) + 1 >= len(summands):
            continue

        sum_axes = {axis for _, shape in parts for axis in shape}
        sum_shape = tuple(axis for axis in allowed if axis in sum_axes)
        adder = hg.HypergraphRoot.template(
            es.einsum(tuple(shape for _, shape in parts), sum_shape,
                      datatype=out_array.datatype,
                      operator=ops.AdditionOp()),
            dom=tuple(node for node, _ in parts))
        outer = hg.HypergraphRoot.template(
            es.einsum((sum_shape, *(shape for _, shape in factor)), out_shape,
                      datatype=out_array.datatype),
            dom=(adder.cod[0], *(node for node, _ in factor)),
            cod=root.cod)
        edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {
            producer: None for producer in producers}
        edits[root] = (*fresh, adder, outer)
        return _rewritten(graph, edits)
    return None


def _align_contracted_variables(
    operand_lists: list[list[Operand]], shared_namespace: set[es.IndexVariable],
) -> list[list[Operand]]:
    '''Rename each summand's contracted variables onto the first summand's,
    wherever one wire is read by both at positions that agree.

    Each summand's einsum contracts variables of its own, so a factor the
    summands share reads its contracted positions under different names in
    each. Pairing the reads of one wire position by position gives the
    renaming, and a wire whose reads disagree, or that pairs a contracted
    position with a surviving one, is left as two reads.'''
    first = operand_lists[0]
    aligned = [first]
    for operands in operand_lists[1:]:
        renaming: dict[es.IndexVariable, es.IndexVariable] = {}
        used: set[int] = set()
        for node, shape in operands:
            for index, (candidate, candidate_shape) in enumerate(first):
                if (index in used or candidate != node
                        or len(candidate_shape) != len(shape)):
                    continue
                trial = dict(renaming)
                consistent = True
                for variable, target in zip(shape, candidate_shape):
                    if variable in shared_namespace or target in shared_namespace:
                        consistent = consistent and variable == target
                    elif trial.get(variable, target) != target:
                        consistent = False
                    else:
                        trial[variable] = target
                if consistent:
                    renaming, used = trial, used | {index}
                    break
        aligned.append([(node, tuple(renaming.get(v, v) for v in shape))
                        for node, shape in operands])
    return aligned


def factor_additions[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''
    Undo the distribution, rewriting every addition of einsums that share a
    common factor as one einsum of the factored addition.

    `distribute_additions` pushes sums apart so that the collapse can reach each
    chain. Once the chains have collapsed there is nothing left to reach, and
    the distributed form costs einsums. This pass factors them back, and only
    when doing so strictly reduces the einsum count.

    Linear scalings sink through their einsums onto a smaller operand first,
    which turns `-(einsum(s, y))` into `einsum(-s, y)` and exposes the common
    `y`.

    On the collapsed expanded softmax the result is the a-priori `SoftMax`
    rule's backward pass, operation for operation: negate the scalar,
    broadcast-add it to `dy`, and multiply by `y` once.

    Applied to the backward pass only.
    '''
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    changed = False
    for _ in range(COLLAPSE_LIMIT):
        rewritten = _sink_once(backward_graph) or _factor_once(backward_graph)
        if rewritten is None:
            break
        backward_graph, changed = rewritten, True
    else:
        raise RuntimeError(f'{COLLAPSE_LIMIT} factorings without a fixed point')
    if not changed:
        return taped
    return bp.Taped.from_passes(
        taped.forward, h2m.hypergraph_to_morphism(backward_graph))


def collapse[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''The rewrites, in the order in which they are sound to run.

    0. `prune_zero_cotangents` removes the zero map the rule for `Maximum`
       writes, with the chain that fed it and the slots only that chain read,
       so that a pair derived from a shifted softmax has no dead arithmetic
       before anything is matched.
    1. `split_inverse_squares` writes an inverse square as a product of
       reciprocals, so that the tape has a map the forward pass already
       applies to migrate onto.
    2. `split_off_forward_formulas` writes a backward formula that is a
       function of a forward one as the forward map followed by that
       function, which gives `migrate_drops` another such map.
    3. `migrate_drops` moves the tape through deterministic maps, and assumes
       no linearity.
    4. `distribute_additions` moves additions after einsums, which exposes
       every chain.
    5. `pathway_collapse` collapses the chains onto the forward pass.
    6. `factor_additions` factors the distribution back, once there are no
       more chains left to expose.

    The sequence is idempotent up to wire identity. A second run
    re-distributes and re-factors the same additions, so compare the two
    listings rather than the two objects. `dedup_slots` is assumed to have run
    on the input.
    '''
    return factor_additions(pathway_collapse(distribute_additions(
        migrate_drops(split_off_forward_formulas(split_inverse_squares(
            prune_zero_cotangents.prune_zero_cotangents(taped)))))))


def dedup_and_collapse[L, M: cat.Morphism](taped: bp.Taped[L, M]) -> bp.Taped[L, M]:
    '''`dedup_slots`, `dedup_roots`, `collapse`, then `dedup_roots` again.

    `collapse` builds a chain once per reader, so a factored sum two einsums
    read, such as attention's dS, comes out of it twice. The second
    `dedup_roots` merges the copies.
    '''
    return dedup_roots(collapse(dedup_roots(dedup_slots(taped))))
