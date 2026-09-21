# Claude Opus 5 (1M context), effort high.
'''Choosing whether a cast is drawn as an operation or as a change of format on a wire.

`quantization/processing/quantise_model.py` puts a `Quantization.TypeConvert` named
`cast` wherever an operation requires a quantisation the value does not carry, and
tsncd draws each of them as a two-coloured chevron whose halves are as tall as the
quantisations it reads and writes. A model given the quantisations the released one
runs in holds eighty of them, so the chevrons stand along every wire of the figure.

The wire on each side of a cast is labelled with the format it carries, so a figure
drawing the chevrons states the rounding twice: once in the chevron and once in the two
labels. `THIN`, the default since 2026-09-20, draws the cast as no glyph at all, on a
box of no width whose operand and result are wired straight to each other, and the
reader sees the rounding in the label changing from `BF16` to `E4M3` along the wire.
The format the cast wrote is drawn in blue and the names of the axes are left out of
that gap, so the one mark the figure makes where a value is rounded is the new format
in colour. `DRAWN` draws the chevron, which is what a notebook whose subject is the
insertion of the conversions asks for.

The choice belongs to the display, so a notebook makes it once in its `DiagramSettings`
and `DiagramSettings.casts` carries it. The pass takes the name off every cast, and a
`TypeConvert` carrying no name is drawn by tsncd as a thin cast, in
`src/display/Framework/quantization/quantisationLabels.ts`. The quantisations stay in the
operator's `source` and `target`, so the term states the same conversion under either
presentation and a listing reads alike.

An inspection box over a cast shows `cast_explanation`: the quantisation read by the
cast, the quantisation written by it, and what the pair of formats does to a value.
The user asked on 2026-09-20 for the coloured format of a thin cast to open such a box,
and tsncd rests the pointer on that label through `ThinTypeConvertBox.region_element`,
because the cast itself is two pixels wide. `notebook_diagrams.package_auxiliary` adds
the row to the table a figure explains its operators with, so every interactive figure
holding casts carries it, and a model that writes its own row for
`Quantization.TypeConvert` keeps that row, as the quantised text-only
DeepSeek-V4.1-Flash does for a cast into a stored form of a cache.

A `TypeConvert` between two forms of a machine datatype, which is a load, a store or a
relayout, keeps its name and its glyph. `Quantization.is_cast` is the condition, and it
holds where the two sides carry different quantisations.
'''
from __future__ import annotations

import enum
from notebooks.display.display_wording import TEXT as text

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import notebooks.display.explain_operators as explain_operators
import quantization.data_structure.Quantization as Quantization


class CastPresentation(enum.Enum):
    DRAWN = 'drawn'
    THIN = 'thin'


def unname_casts[T: fd.GeneralTerm](term: T) -> T:
    '''`term` with the name taken off every `Quantization.TypeConvert` that changes
    the quantisation of what it reads. Each node is rewritten once, by identity, so
    the sharing of the term is kept.'''
    rewritten: dict[int, object] = {}

    def unname(target: object) -> object:
        if id(target) in rewritten:
            return rewritten[id(target)]
        rebuilt = fd.deep_reconstruct(target, unname)
        if (isinstance(rebuilt, Quantization.TypeConvert)
                and rebuilt.name is not None and Quantization.is_cast(rebuilt)):
            rebuilt = rebuilt.reconstruct(name=None)
        rewritten[id(target)] = rebuilt
        return rebuilt

    return unname(term)  # type: ignore[return-value]


def present[T: fd.GeneralTerm](term: T, presentation: CastPresentation) -> T:
    '''`term` with its casts presented as `presentation` asks. `DRAWN` returns it as
    it stands.'''
    if presentation is CastPresentation.DRAWN:
        return term
    return unname_casts(term)


def quantisation_latex(quantisation: Quantization.Quantified) -> str:
    '''`quantisation` as a formula names it, which is the short name
    `Quantization.format_name` gives it, drawn in typewriter where that name is one
    word and in text where it spells out a block scale.'''
    name = Quantization.format_name(quantisation)
    return rf'\text{{{name}}}' if ' ' in name else rf'\mathtt{{{name}}}'


def rounding_sentence(
    read: Quantization.Quantified, written: Quantization.Quantified,
) -> str:
    '''What the pair of quantisations does to a value. A written format of fewer bits
    holds fewer numbers and rounds, one of more bits holds every number the operand's
    format holds, and two formats of one size hold different numbers and round a value
    the written one does not hold.'''
    read_size, written_size = read.size, written.size
    if not (isinstance(read_size, nm.Integer) and isinstance(written_size, nm.Integer)):
        return text.CAST_EVERY_VALUE_ROUNDED_SENTENCE
    against = text.CAST_BITS_AGAINST_PHRASE.format(
        written_bits=written_size._value, read_bits=read_size._value)
    if written_size._value < read_size._value:
        return (text.CAST_FEWER_BITS_SENTENCE.format(against=against))
    if written_size._value > read_size._value:
        return (text.CAST_MORE_BITS_SENTENCE.format(against=against))
    return (text.CAST_SAME_BITS_SENTENCE)


def scale_sentence(quantisation: Quantization.Quantified) -> str:
    '''The sentence stating the block scale of a written format, and the empty text
    for a format whose elements denote their own value. The value denoted by an
    element of a block-scaled format is the element multiplied by the scale of its
    group, so a cast into such a format is not a rounding of each element alone.'''
    scale = quantisation.scale
    if scale is None:
        return ''
    group = (
        text.SCALE_GROUP_OF_CONSECUTIVE_VALUES_PHRASE.format(
            channels=scale.channels.to_latex())
        if scale.rows == nm.Integer(1)
        else text.SCALE_GROUP_OF_A_BLOCK_PHRASE.format(
            channels=scale.channels.to_latex(), rows=scale.rows.to_latex()))
    return text.SCALE_SENTENCE.format(form=scale.form.value, group=group)


def cast_explanation(
    target: cat.Broadcasted,
) -> explain_operators.OperatorExplanation | None:
    '''The inspection box over a cast, stating the quantisation the cast reads and
    the one it writes. A reader opens the box from the coloured format the cast wrote,
    which is the whole of a thin cast in the figure, so the box is where the two
    quantisations are named together. `None` for an operator that is not a cast,
    which leaves a load, a store and a relayout to the table of the figure.'''
    operator = target.operator
    if not isinstance(operator, Quantization.TypeConvert):
        return None
    if not Quantization.is_cast(operator):
        return None
    read = Quantization.quantisation_of(operator.source)
    written = Quantization.quantisation_of(operator.target)
    if read is None or written is None:
        return None
    return explain_operators.OperatorExplanation(
        title=r'\text{Cast}',
        formula=rf'{quantisation_latex(read)} \to {quantisation_latex(written)}',
        description=' '.join(sentence for sentence in (
            text.CAST_READS_SENTENCE.format(
                read=Quantization.format_name(read),
                written=Quantization.format_name(written)),
            rounding_sentence(read, written),
            scale_sentence(written),
            text.CAST_WIRE_LABEL_SENTENCE) if sentence))


def with_cast_explained(
    explanations: explain_operators.OperatorExplanations | None,
) -> explain_operators.OperatorExplanations:
    '''`explanations` with `cast_explanation` as the row for
    `Quantization.TypeConvert`, and `explanations` as it stands where it holds a row
    for that class already. A cast drawn thin has no glyph, so the box opened from the
    coloured format it wrote is the one place a figure names the two quantisations
    together, and a figure offers it whether or not the model wrote a table.'''
    if explanations is not None and Quantization.TypeConvert in explanations:
        return explanations
    return {**(explanations or {}), Quantization.TypeConvert: cast_explanation}
