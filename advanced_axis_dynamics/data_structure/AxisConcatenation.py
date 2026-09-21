'''The axis a concatenation produces, which references the axes it is made from.

Written by Claude Fable 5.1, reasoning effort high.

`aops.ConcatenateAxes` lays the positions of its parts end to end along one axis, and
`ConcatenatedAxis` is that axis. It carries its `parts` and derives everything else from
them. Its size is the sum of the parts' sizes, written on `_size` by `__post_init__` so
that the field the rest of the package reads through `local_size()` is always the sum, and
its label is the labels of the parts joined by `PART_SEPARATOR`, so the concatenation of
the window slots `w|x` and the selected slots `s|x` is labelled `w|x + s|x`.

The parts are references rather than copies of a name. A model writes the concatenation
over the axes it has in hand, and composition then replaces a part by the axis it meets,
per `construction_helpers.composition.align_axis`, so the raw window axis `w` a core is
written with becomes the guarded `w|x` the window view produces. The concatenated axis is
rewritten with its parts, because a `fd.Context` rebuilds every field, and the label
follows. A name captured on the uid at construction could not, which is why the axis has
none and `agent_display.morphism_ir.axis_name` reads the parts instead.

The concatenated axis carries no guard, whatever its parts carry. Its live positions are
the union of the parts' runs, which no single affine form states, and
`concatenation_expansion.expand_concatenations` restores each part's own form by rewriting
every consumer of the concatenation.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature.
'''
from __future__ import annotations
from dataclasses import dataclass

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc

PART_SEPARATOR = ' + '


@dataclass(frozen=True)
class ConcatenatedAxis(sc.Axis):
    '''An axis whose positions are the positions of `parts` laid end to end, in order.

    `_size` is written from the parts on construction and on every reconstruction, so a
    value passed for it is replaced by the sum of the parts' sizes.
    '''
    _size: nm.Numeric = nm.Integer(0)
    parts: fd.Prod[sc.Axis] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, '_size', nm.Addition.template(
            *(part.local_size() for part in self.parts)))
