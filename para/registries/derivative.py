'''Converting an operation to its derivative, with one rule per operator.

A rule is the reverse derivative, written directly into the broadcasted
category rather than as a forward `Derivative` that something else then
transposes:

    R[f] : residual(f) (x) T*(cod f) ---> T*(dom f)

The signature above is the axiom of a reverse derivative category, in
`obsidian/07-para/Para Category.md`. Declaring `R` rather than `D` collapses
steps 3 and 4 of `obsidian/07-para/Derivatives.md` into one.

The reason is that the transpose of `D[f]` cannot be derived from `D[f]`
syntactically. Transposing a `SoftMax` Jacobian is linear algebra rather than a
rewrite, so it would have to be declared a-priori per operator in any case.
Declaring the composite avoids inventing a transposer that immediately needs
per-operator help.

Every rule assumes its operator is cleanly broadcast, meaning that every input
already sits in the degree and the reindexings are the degree identity.
`backprop` guarantees that by running `algebra.node_expansion.expand_to_nodes`
first, so a rule never has to account for broadcasting. Step 1 of
`obsidian/07-para/Derivatives.md` covers it.

The residual is declared a-priori rather than derived, and it is generally not the
domain:

    Einops, one operand         nothing, because a contraction is linear
    Einops, several operands    the operands, because it is multilinear
    View, which is a node       nothing, because a reindexing is linear
    Linear                      nothing. It is linear, and its transpose is the
                                same weight read the other way
    Linear, parametrised        W and x. It is multilinear once the weight is
                                an operand, which `para.show_grabbed_parameters`
                                makes it. The W residual is then a grabbed
                                value, which
                                `show_grabbed_parameters.collapse_grabbed_residuals`
                                turns back into a grab of the parameter's own
                                slot
    Linear, selecting           the index as well, because the transpose reads
                                the slab the index named and the weight
                                gradient is written onto it. The index came
                                from a grab, so the same collapse points the
                                reverse pass at the slot the model wrote
    AdditionOp                  nothing
    SoftMax                     its output, since dx = y * (dy - <dy, y>)
    L1Norm                      its input and its output, since
                                dx = (dy - <dy, y>) / sum(x), and the sum the
                                forward pass divided by is not written out
    Elementwise                 its input, for the derivative map
    Arithmetic                  its input. The derivative map is the formula's
                                derivative, written by
                                `solver.algebra.differentiate_numeric`
    Maximum                     nothing. It is the shift of a softmax, and a
                                softmax is invariant to a shift, so the
                                cotangent it passes back is the zero map
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import construction_helpers as ch
from construction_helpers import simple_helper as chsh
import term_utilities.term_utilities as tutil
import algebra.einops_simplification as einops_simplification

import para.data_structure.contraction as pcon
import para.data_structure.inject as inject
import para.algebra.tangent as tng
import para.data_structure.transpose as ptr


@dataclass(frozen=True)
class Residual:
    '''
    What the reverse pass needs off the tape, as positions in the forward
    morphism's domain and codomain. They arrive at R[f] in that order, the
    inputs first, and the incoming cotangent follows them.
    '''
    inputs: fd.Prod[int] = ()
    outputs: fd.Prod[int] = ()

    def arrays(self, target: cat.Broadcasted) -> fd.Prod[cat.Array]:
        return (*(target.dom()[i] for i in self.inputs),
                *(target.cod()[j] for j in self.outputs))


type Rule = Callable[[cat.Broadcasted], tuple[Residual, cat.BroadcastedCategory]]

RULES: dict[type[cat.Operator], Rule] = {}


def rule_for(operator: cat.Operator) -> Rule:
    '''Most specific first - `View` is a subclass of `Elementwise` and their
    rules differ, so the lookup walks up the MRO.'''
    for kind in type(operator).__mro__:
        if kind in RULES:
            return RULES[kind]
    return opaque


def register(*kinds: type[cat.Operator]):
    def decorate(rule: Rule) -> Rule:
        for kind in kinds:
            RULES[kind] = rule
        return rule
    return decorate


def _fanout(sources: fd.Prod[cat.Array],
            *selections: fd.Prod[int]) -> cat.Rearrangement:
    '''Copy and reorder wires so each part of a product gets its operands laid
    out consecutively. The copy is the comonoid, and its dual is the addition
    that `backprop` puts on a reversed `Rearrangement`.'''
    return cat.Rearrangement(
        mapping=tuple(index for selection in selections for index in selection),
        _dom=sources)


def _strided(target: cat.Broadcasted) -> bool:
    '''Whether any reindexing is more than a `Rearrangement` of degree axes.
    `type_search` is a generator, so it has to be drained rather than
    truth-tested.
    '''
    return any(
        True
        for reindexing in target.reindexings
        for _ in tutil.type_search(cat.StrideMorphism, reindexing))


# ==========================================================================
# The rules.
# ==========================================================================
@register(ops.Einops)
def einops(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    An Einops is multilinear, so its reverse is another Einops: contract the
    cotangent against every operand but the one being differentiated, over
    whatever axes that operand does not have.

        d/da  of  a[q d], b[x d] -> y[q x]   is   dy[q x], b[x d] -> da[q d]

    With one operand there is nothing to contract against and no residual: a sum
    over an axis reverses to a repeat along it, which is a `View` whose
    reindexing does not name the axis and so is a node, and a pointwise map
    reverses to itself.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    assert len(cod) == 1, 'an Einops has one output'
    dy = cod[0]
    inputs, output, _ = einops_simplification.index_shapes(target)
    residual = Residual(inputs=tuple(range(len(dom))) if len(dom) > 1 else ())
    sources = (*(dom[i] for i in residual.inputs), dy)
    cotangent = len(sources) - 1
    selections = tuple(
        (cotangent, *(j for j in range(len(residual.inputs)) if j != i))
        for i in range(len(dom)))
    parts = tuple(
        pcon.contract(
            (output, *(inputs[j] for j in range(len(dom)) if j != i)),
            inputs[i],
            datatype=dom[i].datatype)
        for i in range(len(dom)))
    return residual, chsh.make_composed(
        _fanout(sources, *selections), chsh.make_product(*parts))


@register(ops.View)
def identity(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    A node, meaning a `View` carrying nothing but a reindexing. It is linear
    and index-free, so it has no residual, and its transpose is a sum over the
    fibre
    of the reindexing: every axis the node broadcast along is contracted back.
    A repeat is a node whose reindexing names nothing but the axes it keeps,
    so this is also the rule that reverses one back into the sum it is dual to.

    The sum is free only while the reindexing is a `Rearrangement`, which
    selects, permutes and repeats degree axes and nothing else. The fibre is
    then a set of whole axes and the sum is an ordinary contraction, which the
    rewrites in `algebra/` can absorb and merge. A genuinely strided affine
    map, such as a convolution window, sends several domain axes to one
    codomain axis, so its fibre is a diagonal rather than an axis and
    `contract` cannot name it. That case is `transpose.ReindexTranspose`, an
    operator carrying the whole reindexing, still with no residual, because a
    reindexing is linear whether or not the algebra can write its fibre as a
    contraction.
    '''
    dom = tuple(target.dom())
    if _strided(target):
        return Residual(), ptr.transpose_reindexing(target)
    inputs, output, _ = einops_simplification.index_shapes(target)
    return Residual(), pcon.contract(
        (output,), inputs[0], datatype=dom[0].datatype)


@register(ptr.ReindexTranspose)
def reindex_transpose(
    target: cat.Broadcasted,
) -> tuple[Residual, cat.BroadcastedCategory]:
    '''The node a scatter-add is dual to, so a reverse pass can be reversed.'''
    return Residual(), ptr.untranspose_reindexing(target)


def _index_operands(target: cat.Broadcasted) -> fd.Prod[int]:
    '''The positions of the operands whose datatype is an index.'''
    return tuple(i for i, weave in enumerate(target.input_weaves)
                 if isinstance(weave.target().datatype, cat.Natural))


@register(ops.Linear, ptr.Transpose)
def linear(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    A `Linear` is linear, so it is its own derivative and its reverse is its
    transpose: the same weight, read the other way, which is
    `transpose.transpose` swapping the weaves. There is no residual, because
    the input is not needed to propagate a cotangent through a linear map, and
    taping it would store the input of every matrix multiply in a model for
    nothing.

    It is registered for `Transpose` as well, which makes the rule an
    involution: the reverse of a reverse is the operator itself rather than a
    stack of wrappers.

    An operand whose datatype is `Natural` is an index and selects one of the
    weights, per `ops.Linear`. The transpose then reads the same slab, so the
    index is the residual and the transpose takes it as an operand.

    Two things this does not do. A `bias` contributes nothing to the
    transpose, which is correct because `d(Wx + b)/dx = W`, but its own
    gradient is a parameter gradient, and a parameter is not an operand here,
    so no rule can produce one. Grabbing the weight as an operand is
    `show_grabbed_parameters.grab_parameters`, and the parametrised form takes
    the branch below instead. The weight of the transpose is also the weight of
    the forward pass. `Transpose` keeps the operator whole so that a later pass
    can read that off, and nothing yet does.
    '''
    indices = _index_operands(target)
    parametrised = len(target.input_weaves) > 1 and 0 not in indices
    if parametrised:
        return _parametrised_linear(target)
    if indices:
        return _selecting_transpose(target, indices)
    return Residual(), ptr.transpose(target)


def _transposed_operator(operator: cat.Operator) -> cat.Operator:
    return (operator.operator if isinstance(operator, ptr.Transpose)
            else ptr.Transpose(name=operator.name, operator=operator))


def _selecting_transpose(
    target: cat.Broadcasted, indices: fd.Prod[int],
) -> tuple[Residual, cat.BroadcastedCategory]:
    '''The transpose of an unparametrised `Linear` that selects a weight.

    `(idx, x) -> y` reverses to `(idx, dy) -> dx`, reading the slab the index
    named, so the index is the residual. A `Linear` with no real operand at
    all, which is an embedding, has nothing to propagate a cotangent to and
    reverses to the deletion of the cotangent.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    if len(dom) == len(indices):
        return Residual(), cat.Rearrangement((), (cod[0],))
    degree = target.degree()
    return Residual(inputs=indices), cat.Broadcasted(
        operator=_transposed_operator(target.operator),
        input_weaves=(*(target.input_weaves[i] for i in indices),
                      target.output_weaves[0]),
        output_weaves=(target.input_weaves[-1],),
        reindexings=(degree.identity(),) * (len(indices) + 1))


def _parametrised_linear(
    target: cat.Broadcasted,
) -> tuple[Residual, cat.BroadcastedCategory]:
    '''The parametrised form, produced by `para.show_grabbed_parameters`.

    The weight is an operand, and so is the bias where there is one, so there
    is something to differentiate with respect to. The operands are
    (W[, b][, idx], x), with the parameters first, which is the order
    `show_grabbed_parameters` builds, and an index, where the `Linear`
    selects, ahead of the data. The map is multilinear in (x, W), so each
    cotangent is the other operand contracted against the incoming one:

        dx = Transpose(W[, idx], dy)   the same weight, read the other way. The
                                transpose keeps W as an operand, so a
                                parametrised linear reverses to a parametrised
                                transpose, which is what lets the backward
                                pass grab the slot the forward pass grabbed.
                                A selecting transpose keeps the index too, so
                                it reads the slab the forward pass read.
        dW = x (x) dy           an outer product at the degree. The weight's
                                broadcast node then transposes to the sum over
                                the batch, as any node does. Where the
                                `Linear` selects, the outer product is one
                                slab of the weight, and `inject.inject_slab`
                                writes it at the position the index names
                                along the selected axis and zero elsewhere.
                                The node's sum over the degree then
                                accumulates every token's slab into the bank.
        db = dy                 one per batch index. The bias's node sums it.

    The residuals are x, W and the index, with nothing for b. The W residual
    is a taped copy of a value a `Grab` produced, which is already on the tape
    under the parameter's own slot, and
    `show_grabbed_parameters.collapse_grabbed_residuals` replaces it with a
    grab of that slot. The index residual is collapsed the same way where the
    model grabbed the index from a slot. The rule cannot do that itself,
    because the slot sits on a sibling morphism and a rule is given one seed.

    A `Linear` whose only operands are the weight and an index is an
    embedding. It has no data operand, so there is no dx, and dW is the
    cotangent itself injected onto the slab.

    Registered through `linear` for `Transpose` as well, so the involution
    holds in the parametrised form too: input 0 stays the weight, the last
    input stays the data, and transposing twice unwraps back to the operator.

    The transpose the rule writes puts its weight first for the same reason the
    forward form does, so a `Transpose` reads `(W[, idx], dy) -> dx`. The
    parts of the product are ordered to match the domain they are cotangents
    of, which is `dW`, then `db` where there is a bias, then `dx`.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    indices = _index_operands(target)
    if len(indices) > 1:
        raise NotImplementedError(
            'a Linear selecting along two indices has no reverse rule')
    real = tuple(i for i in range(1, len(dom)) if i not in indices)
    assert len(cod) == 1 and len(real) <= 2, \
        'a parametrised linear is (W[, b][, idx], x) -> y'
    weight, dy = dom[0], cod[0]
    biased = len(real) == 2
    x_at = real[-1] if real else None
    degree = target.degree()
    inputs, output, _ = einops_simplification.index_shapes(target)

    layout = ([('x', x_at)] if x_at is not None else []) + [('W', 0)] + [
        ('idx', i) for i in indices]
    residual = Residual(inputs=tuple(position for _, position in layout))
    sources = (*(dom[position] for _, position in layout), dy)
    at = {name: k for k, (name, _) in enumerate(layout)} | {'dy': len(layout)}

    selections: list[tuple[int, ...]] = []
    parts: list[cat.BroadcastedCategory] = []
    if indices:
        selections.append((at['idx'], *((at['x'],) if x_at is not None else ()),
                           at['dy']))
        parts.append(_weight_slab_gradient(target, indices[0], biased))
    else:
        selections.append((at['x'], at['dy']))
        parts.append(pcon.contract(
            (inputs[x_at], output), inputs[0], datatype=weight.datatype))
    if biased:
        selections.append((at['dy'],))
        parts.append(cat.ProdObject((dy,)).identity())
    if x_at is not None:
        selections.append((at['W'], *((at['idx'],) if indices else ()), at['dy']))
        parts.append(cat.Broadcasted(
            operator=_transposed_operator(target.operator),
            input_weaves=(target.input_weaves[0],
                          *(target.input_weaves[i] for i in indices),
                          target.output_weaves[0]),
            output_weaves=(target.input_weaves[-1],),
            reindexings=(degree.identity(),) * (2 + len(indices))))
    return residual, chsh.make_composed(
        _fanout(sources, *selections), chsh.make_product(*parts))


def _selected_position(target: cat.Broadcasted, index: int, biased: bool) -> int:
    '''The position, in the weight's target, of the axis the operand at
    `index` selects.

    `show_grabbed_parameters.parameter_arrays` lays the weight out one operand
    at a time, in operand order, and then the output target. A real operand
    contributes its target axes and an index operand contributes the one axis
    it selects, so the position is the count of axes the operands before it
    contributed.
    '''
    position = 0
    for weave in target.input_weaves[2 if biased else 1:index]:
        if isinstance(weave.target().datatype, cat.Natural):
            position += 1
        else:
            position += len(weave.target().shape())
    return position


def _weight_slab_gradient(
    target: cat.Broadcasted, index: int, biased: bool,
) -> cat.BroadcastedCategory:
    '''`(idx[, x], dy) -> dW` for a `Linear` that selects a weight.

    The outer product `x (x) dy` has the shape of one slab of the weight,
    meaning the weight's target with the selected axis left out, at the
    degree. `inject.inject_slab` writes it at the position the index names
    along that axis and zero at every other, which is the weight's whole
    shape at the degree, as the weight's node transpose expects.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    weight, dy = dom[0], cod[0]
    degree = target.degree()
    inputs, output, degree_variables = einops_simplification.index_shapes(target)
    position = _selected_position(target, index, biased)
    weight_target = tuple(target.input_weaves[0].target().shape())
    weight_variables = tuple(target.input_weaves[0].select_target(inputs[0]))
    selected = weight_target[position]
    slab_target = (*weight_target[:position], *weight_target[position + 1:])
    slab_variables = (*weight_variables[:position],
                      *weight_variables[position + 1:])
    slab = cat.Weave(weight.datatype,
                     (*(cat.WeaveMode.TILED,) * len(degree), *slab_target))
    real = tuple(i for i in range(1, len(dom)) if i != index
                 and not isinstance(dom[i].datatype, cat.Natural))
    if biased:
        real = real[1:]
    outer_product = (
        pcon.contract((inputs[real[-1]], output),
                      (*degree_variables, *slab_variables),
                      datatype=weight.datatype)
        if real else cat.ProdObject((dy,)).identity())
    return chsh.make_composed(
        chsh.make_product(cat.ProdObject((dom[index],)).identity(),
                          outer_product),
        inject.inject_slab(degree, target.input_weaves[index], slab,
                           selected, position))


@register(ops.AdditionOp)
def addition(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    Addition is linear, so the cotangent passes to every operand, summed over
    the axes that operand was broadcast along, which is what `contract` does
    when the output shape is missing an input axis.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    inputs, output, _ = einops_simplification.index_shapes(target)
    parts = tuple(
        pcon.contract((output,), shape, datatype=array.datatype)
        for shape, array in zip(inputs, dom))
    return Residual(), chsh.make_composed(
        _fanout((cod[0],), *((0,) for _ in dom)), chsh.make_product(*parts))


@register(ops.SoftMax)
def softmax(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    dx = y * (dy - <dy, y>), with the output as the residual, because the
    backward pass is unwritable from the input without recomputing the whole
    softmax.

    The contraction is over the softmax axis alone, which is the one axis in
    the operator target, so everything else stays in the degree and the chain
    below is the same at any degree.

    There is no block. A rule writes its reverse inline, and only a `Block` in
    the forward pass gets a block, titled `R[title]`, in the backward pass.
    `backprop._block` builds it.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    y, datatype = cod[0], cod[0].datatype
    inputs, output, _ = einops_simplification.index_shapes(target)
    kept = tuple(target.output_weaves[0].select_degree(output))

    return Residual(outputs=(0,)), chsh.make_composed(
            # (y, dy) -> (y, dy, dy, y)
            _fanout((y, cod[0]), (0,), (1,), (1, 0)),
            # <dy, y>, contracted over the softmax axis
            chsh.make_product(
                cat.ProdObject((y, cod[0])).identity(),
                pcon.contract((output, output), kept, datatype=datatype)),
            # negate it, so the subtraction is an ordinary AdditionOp
            chsh.make_product(
                cat.ProdObject((y, cod[0])).identity(),
                pcon.contract((kept,), kept, datatype=datatype,
                              operator=ops.Arithmetic(formula=-1 * nm.x))),
            # dy - <dy, y>, the scalar broadcast back over the softmax axis
            chsh.make_product(
                cat.ProdObject((y,)).identity(),
                pcon.contract((output, kept), inputs[0],
                              datatype=datatype, operator=ops.AdditionOp())),
            # times y
            pcon.contract((output, inputs[0]), inputs[0], datatype=datatype))


@register(ops.L1Norm)
def l1_norm(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    dx = (dy - <dy, y>) / z, where z is the sum of the input over the normalised
    axis. The input and the output are both residuals, because the forward pass
    divides by z and never writes z out, so the reverse pass sums the input
    again to recover it.

    The softmax's rule is this one with the division by z replaced by a
    multiplication by y, since a softmax normalises values that are its own
    exponentials and an L1 norm normalises whatever it is given.

    The contraction is over the normalised axis alone, which is the one axis in
    the operator target, so everything else stays in the degree and the chain
    below is the same at any degree.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    x, y, datatype = dom[0], cod[0], cod[0].datatype
    inputs, output, _ = einops_simplification.index_shapes(target)
    kept = tuple(target.output_weaves[0].select_degree(output))
    reciprocal_sum = chsh.make_composed(
        pcon.contract((inputs[0],), kept, datatype=datatype),
        pcon.contract((kept,), kept, datatype=datatype,
                      operator=ops.Arithmetic(formula=1 / nm.x)))

    return Residual(inputs=(0,), outputs=(0,)), chsh.make_composed(
            # (x, y, dy) -> (x, dy, dy, y)
            _fanout((x, y, cod[0]), (0,), (2,), (2, 1)),
            # 1/z from the input, beside dy and the pair to contract
            chsh.make_product(
                reciprocal_sum,
                cat.ProdObject((cod[0],)).identity(),
                pcon.contract((output, output), kept, datatype=datatype)),
            # negate it, so the subtraction is an ordinary AdditionOp
            chsh.make_product(
                cat.ProdObject((y, cod[0])).identity(),
                pcon.contract((kept,), kept, datatype=datatype,
                              operator=ops.Arithmetic(formula=-1 * nm.x))),
            # dy - <dy, y>, the scalar broadcast back over the normalised axis
            chsh.make_product(
                cat.ProdObject((y,)).identity(),
                pcon.contract((output, kept), inputs[0],
                              datatype=datatype, operator=ops.AdditionOp())),
            # divided by z
            pcon.contract((kept, inputs[0]), inputs[0], datatype=datatype))


@register(ops.Elementwise)
def elementwise(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    dx = dy * s'(x), the residual being the input. The derivative map is emitted
    as an `Elementwise` with a primed name, because the algebra has no reading
    of the map itself, only of the fact that its derivative is again
    pointwise.
    '''
    body = (target.operator.name.to_bodies()
            if target.operator.name is not None else '\\sigma')
    return _pointwise_reverse(
        target,
        ops.Elementwise(name=fd.DynamicName(body + "'"), operator='derivative'))


def _pointwise_reverse(
    target: cat.Broadcasted, derivative_map: cat.Operator,
) -> tuple[Residual, cat.BroadcastedCategory]:
    '''dx = dy * s'(x): the derivative map on the taped input, times the
    cotangent. Every pointwise rule shares it, and only the map differs. It is
    written inline rather than as a block, for the reason `softmax` gives.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    datatype = dom[0].datatype
    inputs, output, _ = einops_simplification.index_shapes(target)
    return Residual(inputs=(0,)), chsh.make_composed(
            chsh.make_product(
                pcon.contract((inputs[0],), output,
                              datatype=datatype, operator=derivative_map),
                cat.ProdObject((cod[0],)).identity()),
            pcon.contract((output, output), inputs[0], datatype=datatype))


@register(ops.ReLU)
def relu(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''dx = dy * 1_{x > 0}. `ops.ReLU` is the map `nm.RectifiedLinear(nm.x)`,
    so the derivative map is the `Arithmetic` the `arithmetic` rule writes for
    that formula. `nm.RectifiedLinear` expands to `x 1_{x > 0}`, and the
    product rule with the zero derivative of `nm.IsPositive` leaves the
    indicator alone.

    The generic `Elementwise` rule would emit the primed name, which states
    nothing the reverse pass can evaluate. The registry walks the MRO
    most-specific-first, so this row shadows `elementwise` for `ReLU` the way
    `View`'s shadows it.

    The value at `x = 0` is a convention. PyTorch, TensorFlow and JAX all fix
    it at 0, and 1/2 and 1 both appear in the literature. The measure-zero
    argument does not license the convention. A ReLU emits exactly 0 on every
    dead unit and the next layer receives those zeros, so `x = 0` is reached
    constantly. Nonsmooth analysis does not license it either, since the Clarke
    subdifferential of a composite need not contain what the indicator produces
    for that composite. What licenses the choice is that the reverse pass is a
    conservative field for the forward map under any of the three values, per
    `obsidian/07-para/Derivatives.md`.
    '''
    return _pointwise_reverse(
        target, ops.Arithmetic(formula=nm.RectifiedLinear(nm.x)).derivative())


@register(ops.Maximum)
def maximum(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    ds = 0 * dm, with no residual.

    A `Maximum` in this package shifts a softmax before its exponent, per
    `algebra.operator_expansion.expand_shifted_softmax`. A softmax is invariant to a
    shift of its argument, so the cotangent that reaches the maximum through
    the shift is zero in total, whatever the indicator of the argmax would
    distribute it to. The rule writes that zero directly, as a pointwise map
    whose formula is the constant 0, repeated back over the folded axis by a
    node, so the cotangent stays a wire that
    `para.algebra.prune_zero_cotangents` removes together with the chain that
    fed it.

    The rule is declared a-priori for that use. It is not the reverse
    derivative of a maximum whose value is read for its own sake, which is the
    indicator of the argmax, and this package writes no such maximum.
    '''
    inputs, output, _ = einops_simplification.index_shapes(target)
    datatype = tuple(target.dom())[0].datatype
    return Residual(), chsh.make_composed(
        pcon.contract((output,), output, datatype=datatype,
                      operator=ops.Arithmetic(formula=nm.Integer(0),
                                              name=fd.DynamicName('0'))),
        pcon.contract((output,), inputs[0], datatype=datatype))


@register(ops.Arithmetic)
def arithmetic(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    dx = dy * f'(x), and f' is written out: the formula's derivative, by
    `solver.algebra.differentiate_numeric`. The reverse of `e^{x}` multiplies
    by `e^{x}`, and the reverse of `x^{-1}` by `-x^{-2}`, an `Arithmetic` again,
    which `torch_compile` evaluates with no table of names. Residual: the
    input.
    '''
    operator = target.operator
    assert isinstance(operator, ops.Arithmetic)
    return _pointwise_reverse(target, operator.derivative())


def _box(target: cat.Broadcasted, residual: Residual) -> cat.BroadcastedCategory:
    '''A named box with the domain and codomain R[f] must have, and no
    interior.
    '''
    dom, cod = tuple(target.dom()), tuple(target.cod())
    sources = (*residual.arrays(target), *tng.obj(cod))
    results = tuple(tng.obj(dom))
    name = (target.operator.name.to_bodies()
            if target.operator.name is not None
            else type(target.operator).__qualname__)
    return cat.Broadcasted(
        operator=ops.GenericOperator(name=fd.DynamicName(f'R[{name}]')),
        input_weaves=cat.Weave.from_arrays(sources),
        output_weaves=cat.Weave.from_arrays(results),
        reindexings=(cat.ProdObject().identity(),) * len(sources))


def opaque(target: cat.Broadcasted) -> tuple[Residual, cat.BroadcastedCategory]:
    '''
    No rule registered. Emit a named box rather than raising, so a reverse pass
    is always total and the listing shows exactly which operators are still
    not declared a-priori. The residual defaults to the whole domain, which is the
    memory-worst policy and the only safe one without a declaration.
    '''
    residual = Residual(inputs=tuple(range(len(target.dom()))))
    return residual, _box(target, residual)
