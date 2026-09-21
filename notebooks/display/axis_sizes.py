# Claude Opus 5 (1M context), high effort. `SUBSCRIPT` by Claude Fable 5.1, effort 80.
'''Choosing where an axis's assigned size is drawn.

An expression is written with symbolic sizes, and a
`term_utilities.generate_config.NumericConfig` binds them by name, so
`config(model)` carries an integer size on every axis the configuration named.
tsncd labels the wire of such an axis with that integer, through
`StrideCategoryRenderer.AxisProcessor.size_text`, which puts the number where
the axis's name would otherwise be and leaves the model drawn in numbers.
`EXPONENT` writes each integer size into the name it belongs to instead, through
`algebra.write_axis_exponents`, so the residual width of DeepSeek-V4.1 is drawn
`m^{5120}` and the name and the number are read together. `SUBSCRIPT` writes the
same value and lowers it into the subscript, so the same axis is drawn `m_{5120}`
and an axis that already carries a subscript is drawn `q_{d:512}`. The two
differ in the `fd.ExponentPlacement` the name's settings carry and in nothing
else. An axis whose size still holds a free symbol is left as it stands, and so
is an axis of an expression no configuration has sized.

The size is then stated once. tsncd's `size_text` prints the name of an axis
whose name carries an exponent, rather than the integer, because the exponent
already says what the integer would.

A label holding more than one symbol needs the symbols themselves, which
`config(model)` has already replaced by integers. `present` therefore takes the
assignments as `assigned_sizes`, which is
`NumericConfig.assigned_integers_by_name()`, and writes an exponent onto every
axis and every named symbol of the symbolic model, so the sparse axis `k/e`
reads `|k|^{6} \\text{ of } e^{384}`, or `|k|_{6} \\text{ of } e_{384}` under
`SUBSCRIPT`. A notebook that has no assignments to hand passes none, and the
sizes are read off the sized term instead.
`obsidian/05-backends/Compound Axis Labels.md` states the rule.

The choice belongs to the display, so a notebook makes it once in its
`DiagramSettings`. `WIRE_LABEL` is the default and leaves every name as the
expression wrote it.
'''
from __future__ import annotations

import enum
from collections.abc import Mapping

import algebra.write_axis_exponents as write_axis_exponents
import data_structure.Term as fd


class AxisSizes(enum.Enum):
    WIRE_LABEL = 'wire_label'
    EXPONENT = 'exponent'
    SUBSCRIPT = 'subscript'


PLACEMENTS: dict[AxisSizes, fd.ExponentPlacement] = {
    AxisSizes.EXPONENT: fd.ExponentPlacement.SUPERSCRIPT,
    AxisSizes.SUBSCRIPT: fd.ExponentPlacement.SUBSCRIPT,
}


def present[T: fd.GeneralTerm](
    term: T, sizes: AxisSizes, assigned_sizes: Mapping[str, int] | None = None,
) -> T:
    '''`term` with its sizes drawn where `sizes` asks. `WIRE_LABEL` returns it
    as it stands. `EXPONENT` and `SUBSCRIPT` write the integer each size comes
    to under `assigned_sizes`, and read the integer off the term itself where no
    assignments are given, raised after the name or lowered into its subscript.'''
    if sizes is AxisSizes.WIRE_LABEL:
        return term
    placement = PLACEMENTS[sizes]
    if assigned_sizes is None:
        return write_axis_exponents.write_axis_size_exponents(term, placement)
    return write_axis_exponents.write_assigned_size_exponents(
        term, assigned_sizes, placement)
