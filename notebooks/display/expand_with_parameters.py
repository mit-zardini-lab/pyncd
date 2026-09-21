# Claude Fable 5.1, effort 80.
'''Writing an operator out for an inspection box with its parameters drawn.

`algebra.registries.standard_expansions` holds the rule that writes an operator out in
its primitives. A rule is given the operator as the model holds it, and a model holds
a `Linear` and a `Normalize` with the weight inside the operator, where a rule cannot
reach it. The reviewer asked on 2026-09-16 for the RMSNorm of an inspection box to be
drawn with its weights present, and for a linear map to be explained the same way.

Both functions here therefore grab the parameters first, through
`para.processing.show_grabbed_parameters.grab_parameters`, so the weight of a `Linear`
and the gain of a `Normalize` are arrays fed to the operator as operands. They then
write out every operator of the result that the registry holds a rule for. A
`Normalize` comes out as the scaling by the reciprocal root mean square followed by the
product with the gain, and a `Linear` as the `Einops` that contracts its weight,
followed by the addition of its bias where it has one, by the rule
`algebra.linear_expansion` registers. A `Linear` with no operands, which
stands for a learned array, comes out of `grab_parameters` as the grab of that array
alone. The reviewer asked on 2026-09-17 for every weight to be written out the same
way, after finding that the sink logit opened no box. A `Linear` that selects among
weights is not written out by the rule, and is drawn in the parametrised form, with
its weight as an operand.

The two functions differ in what stands where each grab stood, and `ExpandedParameters`
names the choice for `DiagramSettings.expanded_parameters`.

`expanded_with_weight_arrays` replaces each grab with the weight array
`show_grabbed_parameters.weight_array_in_place_of` writes, a `Linear` with no operands
named after the slot. A map `W : a -> b` becomes `[W : 1 -> ab] * hold(a)` followed by
the `Einops` of `ab` and `a` onto `b`, and the expansion holds no tape. The reviewer
asked for that form on 2026-09-17, and it is the default.

`expanded_with_grabbed_parameters` follows each grab with the weight box
`show_grabbed_parameters.weight_box_fed_by` writes, and presents the result under the
tape setting of the figure. The reviewer ruled on the form earlier on 2026-09-17: a map
`W : a -> b` becomes `[ParaWrap(grab(ab), W) : 1 -> ab] * hold(a)` followed by the same
`Einops`. Under `TapePresentation.ABSORBED` the grab is absorbed onto the weight box, so
a box labelled with the weight has the tape running down onto it, and the contraction
after it is an operation of its own. Without the weight box the grab was absorbed onto
the contraction, and the box over a weight drew one operation under a formula that names
two arrays.

`algebra.linear_expansion` and `algebra.operator_expansion` are imported
for their registrations, because the registry is filled by the decorators those
modules apply to their rules.
'''
from __future__ import annotations

import enum
import functools

import algebra.linear_expansion as linear_expansion  # noqa: F401
import algebra.operator_expansion as operator_expansion
import data_structure.Category as cat
import data_structure.Term as fd
import para.processing.show_grabbed_parameters as show_grabbed_parameters
import websocket_transfer.auxiliary_information as auxiliary_information

import notebooks.display.tape_presentation as tape_presentation


class ExpandedParameters(enum.Enum):
    WEIGHT_ARRAYS = 'weight_arrays'
    READ_FROM_THE_TAPE = 'read_from_the_tape'


def write_out_under(
    parameters: ExpandedParameters,
    tape: tape_presentation.TapePresentation,
) -> auxiliary_information.WriteOut:
    '''The function that writes an operator out for an inspection box under
    `parameters`. `tape` is read under `READ_FROM_THE_TAPE` alone, because an
    expansion with weight arrays holds no tape to present.'''
    match parameters:
        case ExpandedParameters.WEIGHT_ARRAYS:
            return expanded_with_weight_arrays
        case ExpandedParameters.READ_FROM_THE_TAPE:
            return functools.partial(expanded_with_grabbed_parameters, tape=tape)


def expanded_with_weight_arrays[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> fd.GeneralTerm | None:
    '''`target` with a weight array feeding each parameter of its operator, and every
    operator of the result that the registry writes out written out. `None` where
    `target` has no parameter and no rule writes it out. A `Linear` with an empty
    domain, which stands for a learned array such as the sink logit, comes out as the
    weight array of that array alone, under the name of its slot.'''
    expanded = _expanded_with_bare_grabs(target)
    if expanded is None:
        return None
    return show_grabbed_parameters.write_grabs_as_weight_arrays(expanded)


def expanded_with_grabbed_parameters[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
    tape: tape_presentation.TapePresentation,
) -> fd.GeneralTerm | None:
    '''`target` with a grab feeding each parameter of its operator through a weight
    box, and every operator of the result that the registry writes out written out,
    presented under `tape`. `None` where `target` has no parameter and no rule
    writes it out, so there is nothing to draw beyond the operator itself. A
    `Linear` with an empty domain, which stands for a learned array such as the sink
    logit, comes out as the weight box of that array alone.'''
    expanded = _expanded_with_bare_grabs(target)
    if expanded is None:
        return None
    return tape_presentation.present(
        show_grabbed_parameters.box_grabbed_weights(expanded), tape)


def _expanded_with_bare_grabs[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> fd.GeneralTerm | None:
    expanded = operator_expansion.expand_standard_operators(
        show_grabbed_parameters.grab_parameters(target))
    return None if expanded == target else expanded
