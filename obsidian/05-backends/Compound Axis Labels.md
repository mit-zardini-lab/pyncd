---
tags: [layer/backends, concept]
code: algebra/write_axis_exponents.py, notebooks/display/axis_sizes.py, data_structure/Term.py, deepseek/data_structure.py
status: stable
---

# Compound Axis Labels

Written by Claude Opus 5, effort high, on 2026-09-15.

## What it is

A compound axis is an axis or a datatype whose wire label holds more than one symbol.
Four of them are drawn in this package. A guarded axis is named `w|x`, its own letter, a
bar and the letters of the axes its affine form reads, per [[Advanced Axis Dynamics]]. A
sparse axis carries an activity and an extent and is labelled `k of e`, per
[[Sparse Axes]]. A concatenated axis is labelled by the labels of its parts joined by a
plus. A `cat.Natural` datatype is labelled by its `max_value`, which is the size symbol of
the axis its indices count over, and that size may itself be a product of two symbols. The
`Natural` follows the same rule with one symbol, because its label is a symbol standing
where an integer would otherwise be drawn.

Every one of them held a bare integer where a symbol stood, once a configuration had sized
the model. The router's Top-6 over 384 experts read `6 of 384`, and a reader could not tell
which of the two numbers was the count and which the extent without knowing the model. The
requester ruled on 2026-09-15 that a label states its symbols and gives each one its size,
so the same axis reads `|k|^{6} \text{ of } e^{384}`.

## A compound label keeps every symbol and gives each one its exponent

Every symbol in a compound label keeps its letter, and the integer a configuration
assigned that symbol is drawn as an exponent on the letter. A size is never written in
place of the symbol it sizes. A guarded axis already followed the rule. The window slots
`w|x` sized at 128 read `w^{128}|x` where the size alone would read `128|x`. The rule is
carried here to every label that holds more than one symbol.

A symbol the configuration left unbound is drawn as the expression wrote it, with no
exponent, so a label can hold a sized symbol beside a symbolic one.

The exponent is drawn where the name's `fd.ExponentPlacement` puts it, per
[[Code Forms]]. Under `AxisSizes.SUBSCRIPT` every value below is lowered into the
subscript of its letter and stands outside the bars as the raised one does, so the same
labels read `m_{5120}`, `w_{128}|x`, `|k|_{6} \text{ of } e_{384}` and `|a|_{2}|b|`. The
user asked for the subscript form on 2026-09-16, and the V4.1 notebooks draw it.

## The rendering of each kind of compound label

The sizes below are the ones the last cell of `notebooks/sota/DeepSeekV41Flash.ipynb`
binds, with `b`, the encoder's compressed entries, left symbolic.

| kind | symbolic | sized, `AxisSizes.EXPONENT` | sized, `AxisSizes.SUBSCRIPT` |
|---|---|---|---|
| plain axis `m` | `m` | `m^{5120}` | `m_{5120}` |
| guarded axis `w\|x` | `w\|x` | `w^{128}\|x` | `w_{128}\|x` |
| concatenated axis | `w\|x + s\|x` | `w^{128}\|x + s^{512}\|x` | `w_{128}\|x + s_{512}\|x` |
| sparse axis `k/e` | `\|k\| \text{ of } \|e\|` | `\|k\|^{6} \text{ of } e^{384}` | `\|k\|_{6} \text{ of } e_{384}` |
| `Natural` over `e` | `\|e\|` | `\|e\|^{384}` | `\|e\|_{384}` |
| `Natural` over `\|a\|\|b\|` | `\|a\|\|b\|` | `\|a\|^{2}\|b\|` | `\|a\|_{2}\|b\|` |
| symbol with a subscript, `T` of layer 1 | `T_{1}` | `T_{1}^{384006168}` | `T_{1:384006168}` |

A symbol that carries a subscript carries it as a `fd.DynamicName`, and never as markup
inside the body. A lowered exponent joins such a subscript after a colon, which is the
`q_{Td:512}` rule above, and a body holding its own subscript takes the exponent as a
second script beside it. Two subscripts are not LaTeX, so KaTeX draws the label as its
own source in red, the colour command the renderer prepends included. The row count of
an Engram table was named with the markup in its body until 2026-09-18, because a
configuration matched a symbol by the body of its name alone and the two tables would
otherwise have shared one key. `ConfigLog.search` now reads the bodies run together
beside the body, so `T1` and `T14` address the two tables and every key that matched
before matches the same symbols.

The two halves of a sparse axis are read from two places, which is why the bars fall
differently on them. The activity half is the activity numeric itself, named as a size is
named and drawn `|k|`. The parent half is the letter after the slash of the axis's own
name, drawn `e`, and the axis's local size is the parent's size, so the exponent the pass
writes onto the name is the parent's. A sparse axis no configuration has sized has no
exponent for the parent half to carry, and the label falls back to the parent's size
symbol, `|e|`.

A product is written by juxtaposition in both repositories, so a `Natural` over `|a||b|`
with `a` assigned reads `|a|^{2}|b|`, each factor carrying its own exponent or none.

## Where the mechanism lives

| | |
|---|---|
| `algebra/write_axis_exponents.py` | `write_assigned_size_exponents(term, assigned)` writes the integer each axis's local size comes to onto that axis's name, and the integer assigned a named symbol onto the symbol's own name |
| `term_utilities/generate_config.py` | `NumericConfig.assigned_integers_by_name()` returns the assignments as `dict[str, int]`, and `ConfigLog.search` addresses a symbol by the body of its name or by its bodies run together |
| `notebooks/display/axis_sizes.py` | `present(term, sizes, assigned_sizes)` applies the pass under `AxisSizes.EXPONENT` and `AxisSizes.SUBSCRIPT`, passing the placement each names |
| `notebooks/display/notebook_diagrams.py` | `DiagramSettings.assigned_sizes`, passed to every `show_diagram` call |
| `notebooks/display/sota_figures.py` | `show(..., assigned_sizes=...)`, which stands in for the field for one call |
| `data_structure/Term.py` | `DynamicName.to_latex` draws the exponent outside the absolute bars, raised or lowered as `DynamicNameSettings.exponent_placement` says, and `without_absolute_bars` drops them |
| `deepseek/data_structure.py` | `dense_selected_axis` and `selected_axis_name` name the dense axis from the count, without the count's bars |
| `tsncd/src/data_structure/Term.ts` | the same bar rule, and `Natural.to_latex`, which is its `max_value`'s latex |
| `tsncd/src/deepseek/display_deepseek.ts` | `sparse_parent_latex` and `SparseAxisProcessor.annotation_text` |
| `tsncd/src/display/Framework/advanced_axis_dynamics/guardedAxisLabels.ts` | the exponent moved onto the letter before the bar, drawn by the letter's own `to_latex` so the placement holds |
| `tsncd/src/display/Framework/advanced_axis_dynamics/concatenatedAxisLabels.ts` | each part labelled by its own processor |

The exponent stands outside the absolute bars because the bars measure the letter. A size
symbol is the letter between bars, so `|k|` with the exponent 6 reads `|k|^{6}` and the
bars enclose `k` alone. Before 2026-09-15 the exponent stood inside them and the same name
read `|k^{6}|`, which reads as the size of something called `k^6`.

## The assignments are written by name rather than read off the sized term

`NumericConfig.__call__` replaces every assigned `FreeNumeric` with an `nm.Integer` through
an `fd.EqualityClass`, so `config(model)` holds integers where the model held symbols. A
sparse axis of the sized model carries `Integer(6)` as its activity and `Integer(384)` as
its size, and the letters are gone before any display pass runs. A label that has to state
its symbols cannot be built from a term in that form.

`write_axis_size_exponents` gets away with reading the sized term because an axis's own
name survives the substitution and only its size is read off it. The compound labels read
the symbols themselves, so `write_assigned_size_exponents` takes the symbolic model and the
assignments beside it, and the notebook draws the symbolic model. The two functions sit
side by side in `algebra/write_axis_exponents.py`, and a term already sized, with no
assignments to hand, still goes through the first.

A size is looked up by the bodies of its name, which is how `NumericConfig.assign_values`
matches in the first place. `evaluated_size_under` binds the symbols of one size by name
and hands the binding to `nm.evaluate_integer`, which looks a symbol up by the term.

## Gaps

The pass writes an exponent onto every named symbol it reaches, which includes the size
symbols standing in the strides of a reindexing. The merge of the candidate blocks is
drawn with `|u|^{8}` against its flat edge, where it read `|u|` before. Stating the size
of a stride's symbol is consistent with the rule and was not asked for.

## See also

- [[Diagram Display]] — the display settings, and the rest of what a figure is drawn from
- [[Sparse Axes]] — the axis whose label is an activity and an extent
- [[Advanced Axis Dynamics]] — the guarded axis and the concatenated axis
- [[Code Forms]] — the exponent slot on a name, and the code form beside it
- [[SOTA Model Notebooks]] — the notebooks that draw a sized model
