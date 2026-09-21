'''The quantisation of a value, and the conversion between two quantisations.

Written by Claude Fable 5.1, effort 25, on 2026-09-19. The package stands on its own,
so that a model can carry a quantisation on every wire before any backend reads it.

A quantisation is the number format a value is held in, together with the size of
that format in bits. A `Quantified` datatype wraps the mathematics of a value, `Reals`
for a real number, with its quantisation. `Encoding` names the formats declared by a
model or a machine, and each is a closed case rather than a string, so a misspelt
format fails when it is written. A `TypeConvert` is the operator reading a value of one
datatype into another. Between two quantisations it is a cast, which rounds where the
target holds fewer numbers and changes nothing where it holds more.

A format with few bits holds few numbers, and a value held in one is scaled first. A
`BlockScale` states that one scale, held in its own format, is shared by a group of
consecutive values, and a `Quantified` carrying a `BlockScale` denotes the format of the
elements together with the format and the group of the scale. The value denoted by such
a wire is the element multiplied by the scale of its group. The released
DeepSeek-V4.1-Flash rounds the operand of every FP8 or FP4 projection to E4M3 with one
UE8M0 scale per 32 channels, and its GEMM multiplies the FP32 accumulator by the scale
of the operand and the scale of the weight, at `inference/kernel.py` lines 41 to 96 and
256 to 269 of the commit pinned by `notebooks/sota/DeepSeekV41Flash/reference_links.py`.
The Open Compute Project's Microscaling specification names E4M3 elements with one
E8M0 scale per 32 elements `MXFP8` and E2M1 elements with the same scale `MXFP4`, and
the released kernels say `MXFP` for the rounding of their scales to a power of two, so
`INDUSTRY_NAMES` gives those two the names a practitioner reads. Added by Claude Fable
5.1, effort 80, on 2026-09-20, when the user asked that the quantisation of a rounded
operand be accurate: a cast to bare E4M3 would saturate above 448 and flush below
2^-9, and the scale is what the rounding kernel computes and the GEMM reuses.

`quantisation_of` reads the quantisation off a datatype under whatever wrappers it
carries, `holds_real_numbers` says whether the value is a real number at all, and
`with_quantisation` imposes a quantisation on the real number held by a datatype and
leaves an index alone. A `deepseek.data_structure.Complex` of the reals is quantised by
quantising the reals wrapped by it, which is why the walk follows a `base` field beside
`wraps` and `form`. `format_name` is the short name of a quantisation, for a table key
and a count, and `describe_quantisation` is the sentence a table prints for one.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd


class Encoding(Enum):
    '''The formats a value is held in, each under its industry name. A
    floating-point format is named by the number of its exponent and mantissa bits
    where the industry names it so, `E4M3` for eight bits with four exponent bits and
    three mantissa bits, and by their common name where it has one.'''
    FP32 = 'FP32'
    FP16 = 'FP16'
    BF16 = 'BF16'
    E5M2 = 'E5M2'
    E4M3 = 'E4M3'
    E2M1 = 'E2M1'
    UE8M0 = 'UE8M0'
    INT64 = 'INT64'
    INT32 = 'INT32'


fd.register_enum(Encoding)


@dataclass(frozen=True)
class BlockScale(fd.Term):
    '''One scale, held in `form`, shared by `channels` consecutive values along the
    axis a projection contracts, and by `rows` consecutive values along the other
    axis of a weight. The value denoted by an element is the element multiplied by
    the scale of its group.'''
    form: Encoding
    channels: nm.Numeric
    rows: nm.Numeric = nm.Integer(1)


@dataclass(frozen=True)
class Quantified[B: cat.Datatype = cat.Reals](cat.Datatype):
    '''A datatype with a quantisation imposed. Wrapping `Reals` says the value
    represents a real number mathematically. `vector` is the number of values
    packed into one register word, `form` names the encoding of an element, and
    `scale` is the block scale the elements are multiplied by, or `None` for a
    format whose elements denote their own value.'''
    wraps: B
    size: nm.Numeric = nm.Integer(16)
    vector: nm.Numeric = nm.Integer(1)
    form: Encoding | None = None
    scale: BlockScale | None = None

    def registry_size(self) -> nm.Numeric:
        return self.size * self.vector


@dataclass(frozen=True)
class TypeConvert[A: cat.Datatype = Quantified, B: cat.Datatype = Quantified](cat.Operator):
    '''The operator reading a value of `source` into `target`, entry by entry.'''
    name: fd.DynamicName | None = fd.DynamicName('Convert')
    source: A = field(default_factory=Quantified)  # type: ignore
    target: B = field(default_factory=Quantified)  # type: ignore

    @classmethod
    def template(
        cls, source: A, target: B, name: fd.DynamicName | None | str
    ) -> cat.Broadcasted[A | B, Any, Self]:
        return cls.over_shape(source, target, (), name)

    @classmethod
    def over_shape[X: cat.Axis](
        cls, source: A, target: B, shape: fd.Prod[X],
        name: fd.DynamicName | None | str,
    ) -> cat.Broadcasted[A | B, X, Self]:
        '''The conversion of every element of an array of `shape`, so the
        operator is broadcast over every position.'''
        return cat.Broadcasted(
            operator=cls(
                name if name is None else fd.DynamicName.from_str(name),
                source, target),
            input_weaves=(cat.Weave(source, (cat.WeaveMode.TILED,) * len(shape)),),
            output_weaves=(cat.Weave(target, (cat.WeaveMode.TILED,) * len(shape)),),
            reindexings=(cat.ProdObject(shape).identity(),),
        )  # type: ignore


REGISTER_WORD_BITS = 32
'''The size of the word a quantisation is packed into. `Quantified.vector` is the
number of values held by one word, and a figure prints it in front of the encoding,
as `2BF16`, which the user asked for on 2026-09-20.'''

NATURAL_NUMBERS = cat.Natural(nm.FreeNumeric.named('\\mathrm{bound}'))
'''The number kind wrapped by a quantisation of an integer. `quantisation_of` returns
the quantisation of a wire with the bound of that wire replaced by this one, so the
quantisations of two index wires with two bounds compare equal.'''


def _packed(form: Encoding, size: int) -> Quantified[cat.Reals]:
    return Quantified(cat.Reals(), nm.Integer(size),
                      nm.Integer(REGISTER_WORD_BITS // size), form=form)


def _integer(form: Encoding, size: int) -> Quantified[cat.Natural]:
    return Quantified(NATURAL_NUMBERS, nm.Integer(size),
                      nm.Integer(REGISTER_WORD_BITS // size or 1), form=form)


FP32 = _packed(Encoding.FP32, 32)
FP16 = _packed(Encoding.FP16, 16)
BF16 = _packed(Encoding.BF16, 16)
E5M2 = _packed(Encoding.E5M2, 8)
E4M3 = _packed(Encoding.E4M3, 8)
E2M1 = _packed(Encoding.E2M1, 4)
UE8M0 = _packed(Encoding.UE8M0, 8)
INT64 = _integer(Encoding.INT64, 64)
INT32 = _integer(Encoding.INT32, 32)


def block_scaled(elements: Quantified[Any], scale: Quantified[Any],
                 channels: int, rows: int = 1) -> Quantified[Any]:
    '''`elements` with one scale held in `scale` per `channels` consecutive values,
    and per `rows` consecutive values along the other axis of a weight.'''
    if elements.form is None or scale.form is None:
        raise FormatHasNoEncoding(
            f'{elements} scaled by {scale} names no encoding for one of the two')
    return elements.reconstruct(scale=BlockScale(
        scale.form, nm.Integer(channels), nm.Integer(rows)))


class FormatHasNoEncoding(ValueError):
    '''A block scale asked of a quantisation that names no encoding.'''


MXFP8 = block_scaled(E4M3, UE8M0, channels=32)
MXFP4 = block_scaled(E2M1, UE8M0, channels=32)

INDUSTRY_NAMES: dict[tuple[Encoding, BlockScale], str] = {
    (Encoding.E4M3, MXFP8.scale): 'MXFP8',
    (Encoding.E2M1, MXFP4.scale): 'MXFP4',
}
'''The names given by the Open Compute Project's Microscaling specification to the
two block-scaled formats it defines with the elements the released code uses. The
tsncd mirror prints the same two names on a wire, in
`src/quantization/data_structure/Quantization.ts`.'''


def format_name(quantisation: Quantified[Any]) -> str:
    '''The short name of a quantisation: its industry name where it has one, the
    name of its encoding for a format with no scale, and the encoding with its
    scale spelled out otherwise, as `E4M3 with UE8M0 per 32x32`.'''
    if quantisation.form is None:
        return 'unquantised'
    if quantisation.scale is None:
        return quantisation.form.value
    industry = INDUSTRY_NAMES.get((quantisation.form, quantisation.scale))
    if industry is not None:
        return industry
    return (f'{quantisation.form.value} with {quantisation.scale.form.value} per '
            f'{_group_text(quantisation.scale, "x")}')


def _group_text(scale: BlockScale, joiner: str) -> str:
    channels = scale.channels.to_latex()
    rows = scale.rows.to_latex()
    return channels if rows == '1' else f'{channels}{joiner}{rows}'


def describe_quantisation(quantisation: Quantified[Any]) -> str:
    '''The sentence a table prints for a quantisation: its encoding and size, and
    the format and group of its scale where it has one, as `MXFP8: E4M3, 8 bits, with
    one UE8M0 scale per 32 channels`.'''
    form = 'unquantised' if quantisation.form is None else quantisation.form.value
    packing = ('' if quantisation.vector == nm.Integer(1)
               else f', packed {quantisation.vector.to_latex()} to a word')
    elements = f'{form}, {quantisation.size.to_latex()} bits{packing}'
    scale = quantisation.scale
    if scale is None:
        return elements
    group = (f'{scale.channels.to_latex()} channels' if scale.rows == nm.Integer(1)
             else f'{_group_text(scale, " by ")} block')
    scaled = f'{elements}, with one {scale.form.value} scale per {group}'
    industry = INDUSTRY_NAMES.get((quantisation.form, scale))
    return scaled if industry is None else f'{industry}: {scaled}'


WRAPPING_FIELDS = ('wraps', 'form', 'base')


def wrapped_datatype(datatype: cat.Datatype) -> cat.Datatype | None:
    '''The datatype one wrapper holds, and `None` for a datatype that wraps none.

    A `Quantified` holds its own in `wraps` and `deepseek.data_structure.Complex`
    holds its in `base`, and `form` is read for a wrapper that names its contents
    that way, so the three field names cover every wrapper the package declares.
    '''
    for field_name in WRAPPING_FIELDS:
        inner = getattr(datatype, field_name, None)
        if isinstance(inner, cat.Datatype):
            return inner
    return None


def quantisation_of(datatype: cat.Datatype) -> Quantified[Any] | None:
    '''The `Quantified` held by a datatype, whatever wraps it, and `None` for a
    datatype with no quantisation. The number kind it wraps is returned as the
    declared kind, `cat.Reals()` or `NATURAL_NUMBERS`, so the quantisations of two
    wires compare equal whatever bounds their indices carry.'''
    current: cat.Datatype | None = datatype
    while current is not None:
        if isinstance(current, Quantified):
            return current.reconstruct(wraps=_number_kind(current.wraps))
        current = wrapped_datatype(current)
    return None


def _number_kind(wraps: cat.Datatype) -> cat.Datatype:
    if isinstance(wraps, cat.Natural):
        return NATURAL_NUMBERS
    return wraps


def holds_real_numbers(datatype: cat.Datatype) -> bool:
    '''Whether the value `datatype` describes is a real number, under whatever
    wrappers it carries. A `cat.Natural` is an index and holds none.'''
    return _holds(datatype, cat.Reals)


def holds_natural_numbers(datatype: cat.Datatype) -> bool:
    '''Whether the value `datatype` describes is an index or a count, which is a
    `cat.Natural` under whatever wrappers it carries.'''
    return _holds(datatype, cat.Natural)


def holds_a_number(datatype: cat.Datatype) -> bool:
    return holds_real_numbers(datatype) or holds_natural_numbers(datatype)


def _holds(datatype: cat.Datatype, kind: type[cat.Datatype]) -> bool:
    current: cat.Datatype | None = datatype
    while current is not None:
        if isinstance(current, kind):
            return True
        current = wrapped_datatype(current)
    return False


def with_quantisation(datatype: cat.Datatype,
                        quantisation: Quantified[Any]) -> cat.Datatype:
    '''`datatype` with `quantisation` imposed on the number held by it, and
    `datatype` itself where it holds no number of the kind the quantisation wraps.
    A quantisation of the reals is imposed on a real number, and a quantisation of
    the naturals on an index, which keeps its own bound.

    The wrappers around the number are kept, so a
    `deepseek.data_structure.Complex` of the reals is returned as a complex of the
    quantified reals and a machine's `Stored` form keeps its memory level.
    '''
    kind = type(quantisation.wraps)
    if isinstance(datatype, Quantified):
        return (quantisation.reconstruct(wraps=datatype.wraps)
                if isinstance(datatype.wraps, kind) else datatype)
    if isinstance(datatype, kind):
        return quantisation.reconstruct(wraps=datatype)
    for field_name in WRAPPING_FIELDS:
        inner = getattr(datatype, field_name, None)
        if isinstance(inner, cat.Datatype):
            return datatype.reconstruct(
                **{field_name: with_quantisation(inner, quantisation)})
    return datatype


def is_cast(convert: TypeConvert) -> bool:
    '''Whether `convert` changes the quantisation of its operand.'''
    return quantisation_of(convert.source) != quantisation_of(convert.target)
