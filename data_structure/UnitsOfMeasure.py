'''The dimensions and units of measure the cost model states its quantities in.

Written by Claude Fable 5.1, effort 80.

Every unit here is an `nm.UnitOfMeasure`, a constant whose value is its scale in
the reference units of its dimension. The reference units of the base dimensions
are the bit, the second, the operation, the clock cycle and the joule. A derived
dimension is composed of base dimensions, so the hertz is one per second and the
watt is one joule per second, and `nm.dimensions_of` reads both down to the base
dimensions. `prefixed` scales a unit by a decimal or a binary prefix. A quantity
is a numeric multiplied by a unit, as in `3 * GIBIBYTE`, and `nm.convert_to_unit`
writes it in another unit of the same dimension.

`obsidian/01-foundations/Units of Measure.md` states the design.
'''
from __future__ import annotations
from enum import Enum

import data_structure.Numeric as nm


class DecimalPrefix(Enum):
    '''A decimal prefix, as the symbol written before the unit's symbol and the
    power of ten it scales the unit by.'''
    NANO = ('n', -9)
    MICRO = ('\\mu ', -6)
    MILLI = ('m', -3)
    KILO = ('k', 3)
    MEGA = ('M', 6)
    GIGA = ('G', 9)
    TERA = ('T', 12)
    PETA = ('P', 15)

    def symbol(self) -> str:
        return self.value[0]

    def scale(self) -> tuple[int, int]:
        exponent = self.value[1]
        return (10 ** exponent, 1) if exponent >= 0 else (1, 10 ** -exponent)


class BinaryPrefix(Enum):
    '''A binary prefix, as the symbol written before the unit's symbol and the
    power of two it scales the unit by.'''
    KIBI = ('Ki', 10)
    MEBI = ('Mi', 20)
    GIBI = ('Gi', 30)
    TEBI = ('Ti', 40)

    def symbol(self) -> str:
        return self.value[0]

    def scale(self) -> tuple[int, int]:
        return (2 ** self.value[1], 1)


def prefixed(
    unit: nm.UnitOfMeasure, prefix: DecimalPrefix | BinaryPrefix
) -> nm.UnitOfMeasure:
    '''`unit` scaled by `prefix`, named and written with the prefix in front, so
    that `prefixed(BYTE, BinaryPrefix.GIBI)` is the gibibyte, written `GiB`.'''
    numerator, denominator = prefix.scale()
    return unit.dimension.declare_unit(
        name=prefix.name.lower() + unit.name,
        symbol=prefix.symbol() + unit.symbol,
        scale_numerator=unit.scale_numerator * numerator,
        scale_denominator=unit.scale_denominator * denominator)


INFORMATION = nm.DimensionOfMeasure('information')
TIME = nm.DimensionOfMeasure('time')
OPERATIONS = nm.DimensionOfMeasure('operations')
CLOCK_CYCLES = nm.DimensionOfMeasure('clock cycles')
ENERGY = nm.DimensionOfMeasure('energy')
FREQUENCY = nm.DimensionOfMeasure('frequency', ((TIME, -1),))
POWER = nm.DimensionOfMeasure('power', ((ENERGY, 1), (TIME, -1)))

BIT = INFORMATION.declare_unit('bit', 'bit')
BYTE = INFORMATION.declare_unit('byte', 'B', 8)
KIBIBYTE = prefixed(BYTE, BinaryPrefix.KIBI)
MEBIBYTE = prefixed(BYTE, BinaryPrefix.MEBI)
GIBIBYTE = prefixed(BYTE, BinaryPrefix.GIBI)
TEBIBYTE = prefixed(BYTE, BinaryPrefix.TEBI)
KILOBYTE = prefixed(BYTE, DecimalPrefix.KILO)
MEGABYTE = prefixed(BYTE, DecimalPrefix.MEGA)
GIGABYTE = prefixed(BYTE, DecimalPrefix.GIGA)
TERABYTE = prefixed(BYTE, DecimalPrefix.TERA)

SECOND = TIME.declare_unit('second', 's')
MILLISECOND = prefixed(SECOND, DecimalPrefix.MILLI)
MICROSECOND = prefixed(SECOND, DecimalPrefix.MICRO)
NANOSECOND = prefixed(SECOND, DecimalPrefix.NANO)

OPERATION = OPERATIONS.declare_unit('operation', 'op')
CLOCK_CYCLE = CLOCK_CYCLES.declare_unit('clock cycle', 'cycle')

JOULE = ENERGY.declare_unit('joule', 'J')
HERTZ = FREQUENCY.declare_unit('hertz', 'Hz')
GIGAHERTZ = prefixed(HERTZ, DecimalPrefix.GIGA)
WATT = POWER.declare_unit('watt', 'W')
