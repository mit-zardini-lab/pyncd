'''
Grabbing a model's parameters: **Br** -> **Para(Br)**.

A `Linear` keeps its weight inside the operator. The expression `x @ W` has one
operand, `x`, and the `W` is a letter drawn on a glyph. A reader of the model
needs no more than that. Counting the parameters, training them or sharding
them needs more, because a functor can only transform what a morphism names in
its domain or codomain, and a weight inside an operator is named in neither.

`grab_parameters` writes them out. Each parametric `Broadcasted` gains one
input weave per parameter, holding the full weight array, and a `Grab` of a
fresh `TapeSlot` feeds it. The seed becomes the morphism `W x X -> Y` that the
Para construction means by a parametrised map:

    Linear : R[q, m] -> R[q, d]        becomes
    (Grab<W_Q> * id) ; Linear : R[m, d] x R[q, m] -> R[q, d]

The parameters occupy the first operand slots, ahead of the operands the seed
already had, and they keep the order `parameter_arrays` returns them in. A
`Linear` with a bias is therefore `(W, b, x)`. The first slot is the top wire
in a diagram, so the tape arrives above the data rather than below it, and
`ParaWrap` draws the same order along the top edge of the operator.
`derivative._parametrised_linear` and
`torch_compile.parametrised_linear_func` both read the weight off the first
slot, so the order is part of the parametrised form rather than a display
choice.

The weight array is the size of the target, which is what the operator receives
with the broadcasting excluded: `in_target x out_target` for a `Linear`, the
target itself for the gain and the bias of a `Normalize`.

A `Linear` input whose datatype is `Natural` is an index rather than a value, and
`ops.Linear` gives it the reading that it selects one weight of the many the
index ranges over. Its contribution to the weight is therefore an axis of that
many entries, minted by `selected_axis` from the datatype's size, and not the
operand's own target, which is empty. The expert `Linear` of a mixture of
experts is the case: `(Nat(n), R[m]) -> R[f]` has weight `R[n, m, f]`, holding
all `n` experts.

A `Linear` with no operands is a learned array, such as the sink logit of
DeepSeek-V4.1-Flash. Its weight is the whole of its result, so it becomes the `Grab`
alone, of a slot that carries the array's own name, and no operator is left behind.

The parameter's weave has no `TILED` entry and its reindexing deletes the whole
degree, which states that the weight is shared, so every batch index reads the
same array. The domain and codomain of the whole expression are unchanged,
because a `Grab` is composition-neutral, exactly as it is in
`backprop.forward_backward`. The result is the same morphism in `Para.Para`
form, with its parameters named.

A slot is named after the parameter, `W_Q` or `\\gamma`, rather than after a
position on the tape. Two occurrences of one seed, meaning a shared layer
reached along two paths, grab the same slot, which is what weight tying looks
like from the tape. Two layers built alike but separately have different axes,
so they are different terms and get slots of their own.

`parameter_arrays` states what is parametric, one case per operator.
`Embedding` has no case. Its table is `vocab x target`, and the vocabulary is a
`Natural` datatype rather than an axis. `selected_axis` is what a case for it
would need, since the `Linear` case now mints an axis from a `Natural`'s size
the same way, and nothing else about an `Embedding` has been settled.

`obsidian/07-para/Show Grabbed Parameters.md` describes the construction.
'''
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field

import data_structure.Term as fd  # for 'foundations'
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import utilities.utilities as util
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import graphs.processing.hypergraph_functor as functor
import graphs.processing.replace_roots as replace_roots
import para.data_structure.Para as Para
import para.processing.backprop as bp
from construction_helpers import simple_helper as sh


def selected_axis(extent: nm.Numeric) -> cat.RawAxis:
    '''An axis of size `extent`, carrying the extent's own name.

    A `Natural` input of a `Linear` selects among that many weights, per
    `ops.Linear`, so the weight needs an axis the datatype does not carry. The
    axis is minted here, and the size symbol is what ties it to the axis the
    selection was made over.
    '''
    axis = cat.RawAxis(_size=extent)
    name = extent.uid._name if isinstance(extent, nm.FreeNumeric) else None
    return axis if name is None else name.capture(axis)


def weight_datatype[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> B:
    '''The datatype of a `Linear`'s weight: that of the operand it contracts.

    A `Natural` operand is an index and carries no value, so a `Linear` whose
    first input selects takes its weight's datatype from the input after it.
    The output's datatype is the fallback, for the parameter-array case where a
    `Linear` has no inputs at all.
    '''
    for weave in target.input_weaves:
        datatype = weave.target().datatype
        if not isinstance(datatype, cat.Natural):
            return datatype
    return target.output_weaves[0].target().datatype


def weight_axes[B: cat.Datatype, A: cat.Axis](
    weave: cat.Weave[B, A],
) -> fd.Prod[A]:
    '''The axes a `Linear`'s weight carries on account of one input.

    A real input is contracted, so the weight carries that operand's target
    axes. A `Natural` input selects one weight of the many the index ranges
    over, so the weight carries an axis of that many entries and the operand's
    own target is empty. `ops.Linear` states both readings.
    '''
    target = weave.target()
    match target.datatype:
        case cat.Natural(max_value=extent):
            return (selected_axis(extent),)
    return tuple(target.shape())


def parameter_arrays[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> fd.Prod[tuple[fd.DynamicName, cat.Array[B, A]]]:
    '''The parameters of `target`'s operator, as `(name, array)` pairs.

    The arrays are built from the seed's own weaves, so each is the size of
    the target, which is what the operator receives with the broadcasting
    excluded. A `Linear`'s weight runs over every input and then the out
    target, through `weight_axes`, and its bias is the out target. A
    `Normalize`'s gain and bias are each the target itself, and it has whichever
    of the two its `gain` and `bias` fields declare, the gain first. A
    non-parametric operator returns `()`, and the transform passes it by.
    '''
    match target.operator:
        case ops.Linear(name=name, bias=False) if target.has_empty_domain():
            return ((learned_array_name(name), target.output_weaves[0].target()),)
        case ops.Linear(name=name, bias=bias):
            out_target = target.output_weaves[0].target()
            weight = cat.Array(
                weight_datatype(target),
                (*util.concat(map(weight_axes, target.input_weaves)),
                 *out_target.shape()))
            parameters = [(parameter_name('W', 'weight', name), weight)]
            if bias:
                parameters.append((parameter_name('b', 'bias', name), out_target))
            return tuple(parameters)
        case ops.Normalize(gain=gain, bias=bias):
            normalised = target.input_weaves[-1].target()
            parameters = []
            if gain:
                parameters.append(
                    (fd.DynamicName('\\gamma', code_form='gain'), normalised))
            if bias:
                parameters.append(
                    (fd.DynamicName('\\beta', code_form='bias'), normalised))
            return tuple(parameters)
    return ()


def learned_array_name(operator_name: fd.DynamicName | None) -> fd.DynamicName:
    '''The name of the slot a `Linear` with no operands is read from. Such a
    `Linear` is a learned array and its name is the name of that array, so the slot
    takes the same name, `\\mathrm{sink}`, and `W` where the operator has none.'''
    if operator_name is None:
        return fd.DynamicName('W', code_form='weight')
    return operator_name.with_code_form(operator_name.code_form_or_identifier())


def parameter_name(letter: str, code_form: str,
                   operator_name: fd.DynamicName | None) -> fd.DynamicName:
    '''`letter` subscripted by the operator's name, as `W_{Q}`, carrying the
    code form `weight_Q`: `code_form` joined with the operator name's own code
    form, or with its bodies as an identifier where it has none.'''
    return fd.DynamicName(
        letter, subscript=operator_name,
        code_form=fd.join_code_forms(
            code_form,
            operator_name.code_form_or_identifier() if operator_name is not None else None))


@dataclass
class ShowGrabbedParameters[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every parametric seed in a morphism or hypergraph, given its grabs.

    `_slots` memoises the slot per `(seed, index)`, so a seed the expression
    reaches twice, meaning a shared layer, grabs one slot. The functor is
    therefore used once per derivation, in the way `backprop` resets its slot
    counter once per derivation.
    '''
    _slots: dict[tuple[cat.Broadcasted, int], Para.TapeSlot] = field(
        default_factory=dict)

    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        if not isinstance(target, cat.Broadcasted):
            return target
        parameters = parameter_arrays(target)
        if not parameters:
            return target
        if target.has_empty_domain():
            return self._grabbed_learned_array(target, parameters)
        # The parameter is shared across the degree: its weave has no TILED
        # entry, so its reindexing is the deletion of every degree axis.
        deletion = cat.Rearrangement(mapping=(), _dom=tuple(target.degree()))
        parametrised = target.reconstruct(
            input_weaves=(
                *(cat.Weave(array.datatype, tuple(array.shape()))
                  for _, array in parameters),
                *target.input_weaves),
            reindexings=(
                *(deletion,) * len(parameters),
                *target.reindexings),
        )
        grabs = tuple(
            Para.Grab(self._slot(target, i, name), array)
            for i, (name, array) in enumerate(parameters))
        return sh.make_composed(
            sh.make_product(*grabs, target.dom().identity()),
            parametrised)

    def _grabbed_learned_array(
        self,
        target: cat.Broadcasted[B, cat.RawAxis],
        parameters: fd.Prod[tuple[fd.DynamicName, cat.Array[B, cat.RawAxis]]],
    ) -> Para.Grab:
        '''A `Linear` with no operands as the `Grab` of the array it stands for.

        The sum over its inputs is empty, so the weight is the whole of the result
        and no operator is left once the weight is read from the tape. A bias on
        such a `Linear`, or a degree it is broadcast over, would need an addition or
        a repeat after the grab, and no model writes either.
        '''
        if len(parameters) != 1 or tuple(target.degree()):
            raise NotImplementedError(
                f'{target.operator.name} has no operands, {len(parameters)} '
                f'parameters and the degree {tuple(target.degree())}, and a learned '
                'array is grabbed only as one array with no degree')
        (name, array), = parameters
        return Para.Grab(self._slot(target, 0, name), array)

    def _slot(self, seed: cat.Broadcasted, index: int,
              name: fd.DynamicName) -> Para.TapeSlot:
        key = (seed, index)
        if key not in self._slots:
            self._slots[key] = name.capture(Para.TapeSlot())
        return self._slots[key]


def grab_parameters(target):
    '''`target` with a `Grab` of a named slot feeding every parameter.'''
    return ShowGrabbedParameters()(target)


def weight_box_fed_by[B: cat.Datatype, A: cat.Axis](
    grab: Para.Grab[cat.Array[B, A]],
) -> cat.Composed[cat.Array[B, A], Para.Grab | cat.Broadcasted[B, A]]:
    '''`grab` followed by a `Linear` named after its slot, whose one operand and one
    result are the array the grab reads: `Grab<W> ; W : 1 -> [a, b]`.

    The box is the weight array `algebra.linear_expansion` writes,
    a `Linear` with no data, with its weight written out as an operand the way
    `ShowGrabbedParameters` writes the weight of every other `Linear`. It reads no
    data, so it contracts nothing and its result is its weight. `para_wrap.to_para_wrap`
    absorbs the grab onto it, and a diagram then draws a box labelled with the weight
    and a tape running down onto it.

    A `Linear` with one operand is otherwise read as a map applied to that operand,
    with its weight inside the operator. The two are told apart by the grab that
    feeds the box, so the form is written for a diagram alone, after every pass that
    reads a `Linear` has run.
    '''
    array = grab.size
    weave = cat.Weave(array.datatype, tuple(array.shape()))
    return sh.make_composed(grab, cat.Broadcasted(
        operator=ops.Linear(name=grab.tape.uid._name),
        input_weaves=(weave,),
        output_weaves=(weave,),
        reindexings=(cat.Rearrangement(mapping=(), _dom=()),)))


def weight_array_in_place_of[B: cat.Datatype, A: cat.Axis](
    grab: Para.Grab[cat.Array[B, A]],
) -> cat.Broadcasted[B, A]:
    '''A `Linear` with no operands, named after the slot of `grab`, whose one result
    is the array the grab reads: `W : 1 -> [a, b]`.

    The box is the weight array
    `linear_expansion.expand_linear_root` writes for the weight of a plain
    `Linear`. It holds its weight inside the operator and reads no tape, so a term
    whose grabs are all written this way is a morphism of **Br** again.
    '''
    array = grab.size
    return cat.Broadcasted(
        operator=ops.Linear(name=grab.tape.uid._name),
        input_weaves=(),
        output_weaves=(cat.Weave(array.datatype, tuple(array.shape())),),
        reindexings=())


def box_grabbed_weights[T: fd.GeneralTerm](target: T) -> T:
    '''`target` with every `Grab` followed by the weight box `weight_box_fed_by`
    writes for it, so a contraction that read a grabbed weight reads the result of a
    box that names the weight.'''
    return _with_every_grab_rewritten(target, weight_box_fed_by)


def write_grabs_as_weight_arrays[T: fd.GeneralTerm](target: T) -> T:
    '''`target` with every `Grab` replaced by the weight array
    `weight_array_in_place_of` writes for it, so the result reads no tape.'''
    return _with_every_grab_rewritten(target, weight_array_in_place_of)


def _with_every_grab_rewritten[T: fd.GeneralTerm](
    target: T,
    rewrite_grab: Callable[[Para.Grab], cat.Morphism],
) -> T:
    '''Each node is rewritten once, by identity, so the sharing of `target` is kept.'''
    rewritten: dict[int, object] = {}

    def write(node: object) -> object:
        if id(node) in rewritten:
            return rewritten[id(node)]
        rebuilt = (rewrite_grab(node) if isinstance(node, Para.Grab)
                   else fd.deep_reconstruct(node, write))
        rewritten[id(node)] = rebuilt
        return rebuilt

    return write(target)  # type: ignore[return-value]


def collapse_grabbed_residuals[L, M: cat.Morphism](
    taped: bp.Taped[L, M],
) -> bp.Taped[L, M]:
    '''A grabbed value is already on the tape, so it needs no residual.

    `backprop` tapes a parametrised seed's weight operand like any other
    residual, writing a `Drop<s1>` in the forward pass and a `Grab<s1>` in the
    backward pass, because a rule receives one seed and the parameter's slot sits
    on a sibling morphism. The value that drop saves came from a `Grab`, so it
    is already on the tape under the parameter's own slot. This pass deletes
    the forward `Drop` and points the backward's grab at the parameter, which
    is what makes the backward pass read the slots the forward pass reads: a
    `Transpose` grabs the weight it transposes.

    Graph-based like `pathway_collapse.dedup_slots`, and for the same reason:
    "the same wire" is a question about nodes, and a morphism has none. Run it
    before `dedup_slots`, which numbers slots by their `s{n}` names. Parameter
    slots have no such name, and this pass removes the drops that numbering
    would otherwise trip over.

    A grabbed index collapses the same way as a grabbed weight. A `Linear`
    that selects a weight declares its index operand as a residual, and where
    the model grabbed that index from a slot, the reverse pass reads the same
    slot.

    Both passes are walked through their blocks, by
    `graphs.processing.replace_roots`, because a grab inside a block is a grab.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    grabbed: dict[hg.HypergraphObject, Para.TapeSlot] = {
        root.cod[0]: root.wraps.tape
        for root in replace_roots.walk_roots(forward_graph)
        if isinstance(root.wraps, Para.Grab)}

    removed: dict[Para.TapeSlot, Para.TapeSlot] = {}
    dead: dict[hg.HypergraphRoot, None] = {}
    for root in replace_roots.walk_roots(forward_graph):
        if isinstance(root.wraps, Para.Drop) and root.dom[0] in grabbed:
            removed[root.wraps.tape] = grabbed[root.dom[0]]
            dead[root] = None
    if not removed:
        return taped

    forward_graph = replace_roots.replace_roots(forward_graph, dead)

    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    replaced = {
        root: root.reconstruct(wraps=root.wraps.reconstruct(
            tape=removed[root.wraps.tape]))
        for root in replace_roots.walk_roots(backward_graph)
        if isinstance(root.wraps, Para.Grab) and root.wraps.tape in removed}
    backward_graph = replace_roots.replace_roots(backward_graph, replaced)

    return bp.Taped.from_passes(h2m.hypergraph_to_morphism(forward_graph),
                    h2m.hypergraph_to_morphism(backward_graph))
