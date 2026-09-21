# Claude Fable 5.1, effort 80.
'''Choosing whether a figure carries a legend and opens inspection boxes.

tsncd draws two things beyond the diagram when asked. A legend is a table of the
term's axes, one row per named axis, with the integer the axis's size comes to in the
middle and the code form of its name on the right. An inspection box opens when the
pointer rests on a box or on an operator with a standard expansion, with its top left
corner at the pointer. For a box it shows the block's title, its description, the code
references its aesthetics hold, and its body drawn as a diagram. For an operator it
shows the formula the registry `algebra.registries.standard_expansions` holds and the
expansion drawn as a diagram. The expansion is written out with the operator's
parameters drawn, by `notebooks/display/expand_with_parameters.py`, so the gain of
an RMSNorm and the weight of a linear map each stand as a box labelled with the weight,
before the operation that reads the weight. `ExpandedParameters` chooses whether that
box is a weight array or a box that reads its array from the tape.
An operator with no expansion is
explained through a block drawn as the operator alone, which
`notebooks/display/explain_operators.py` wraps it in from a table the model supplies.
A click locks the box open, and a locked box's own boxes and operators open boxes of
their own, so a model is read down to its leaves from one figure. A click outside every
box closes them.

Everything a legend or a box shows is computed here and sent beside the term, as the
`auxiliary` field `websocket_transfer.auxiliary_information` assembles, because tsncd
does no algebra. The setting is one of three.

    OFF          Neither. The message carries no auxiliary field and no new setting,
                 so it is the message an older page reads. The default.
    LEGEND       The legend beside the figure, in every mode that draws. An INLINE
                 figure carries it in the image.
    INTERACTIVE  The legend and the inspection boxes. The boxes answer the pointer,
                 so they exist on the open page and not in a captured image.

`code_link_base` is where a reference that carries a path and no url of its own is
linked. A web base such as `https://github.com/<owner>/pyncd/blob/main/` takes
`#L<line>` after the path, and a `vscode://file/<root>/` base takes `:<line>`, which
opens the file in the editor. A reference with no url and no base is shown as text.
'''
from __future__ import annotations

import enum

import data_structure.Term as fd
import websocket_transfer.auxiliary_information as auxiliary_information
import websocket_transfer.websockets_transfer as wst

import notebooks.display.expand_with_parameters as expand_with_parameters
import notebooks.display.tape_presentation as tape_presentation


class AdvancedDisplay(enum.Enum):
    OFF = 'off'
    LEGEND = 'legend'
    INTERACTIVE = 'interactive'


def draws_legend(mode: AdvancedDisplay) -> bool | None:
    '''The `legend` display setting, or `None` to leave it out of the message.'''
    return None if mode is AdvancedDisplay.OFF else True


def opens_inspection_boxes(mode: AdvancedDisplay) -> bool | None:
    '''The `inspectionBoxes` display setting, or `None` to leave it out.'''
    return True if mode is AdvancedDisplay.INTERACTIVE else None


def auxiliary_for(
    morphism: fd.GeneralTerm,
    mode: AdvancedDisplay,
    assigned_sizes: dict[str, int] | None,
    code_link_base: str | None,
    tape: tape_presentation.TapePresentation,
    expanded_parameters: expand_with_parameters.ExpandedParameters,
    operator_references: auxiliary_information.OperatorReferences | None,
    operator_roles: auxiliary_information.OperatorRoles | None,
) -> wst.DiagramAuxiliary | None:
    '''The `auxiliary` field for `morphism` under `mode`, or `None` under `OFF`.
    `morphism` is the term the transport sends, after every display pass and after
    a hypergraph has been converted, because the expansions are keyed by the order
    of its nodes. Each expansion is written out with the operator's parameters drawn
    as `expanded_parameters` says: as weight arrays, or as weight boxes that read the
    tape, presented under `tape` as the figure is. It carries the references
    `operator_references` holds for its operator and the role `operator_roles` holds
    for its name.'''
    if mode is AdvancedDisplay.OFF:
        return None
    return auxiliary_information.auxiliary_information(
        morphism, assigned_sizes=assigned_sizes, code_link_base=code_link_base,
        write_out=expand_with_parameters.write_out_under(expanded_parameters, tape),
        operator_references=operator_references, operator_roles=operator_roles)
