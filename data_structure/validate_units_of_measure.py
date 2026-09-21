'''Checking the units of measure against arithmetic.

Written by Claude Fable 5.1, effort 80.

    python data_structure/validate_units_of_measure.py

Each case is a fact `obsidian/01-foundations/Units of Measure.md` states: a
prefixed unit scales its reference, a quantity evaluates in the reference units,
a conversion divides the scales, a derived unit equals the quotient it is
declared from, a product of units is one canonical term, a sum across dimensions
is refused, a unit is a constant under differentiation, and a unit survives JSON
and pickling unchanged.
'''
from __future__ import annotations
import math
import pickle
import sys
from typing import Callable

import data_structure.Numeric as nm
import data_structure.UnitsOfMeasure as units
import data_transfer.term_json as term_json
import solver.algebra.differentiate_numeric as differentiate_numeric
import solver.registries.numeric_derivative  # noqa: F401 - the derivative rows


class NoFloatForTheNumeric(ValueError):
    '''A numeric holding something the evaluation below does not read.'''


def evaluate_float(target: nm.Numeric) -> float:
    '''The number `target` denotes, with every unit taken in its reference units.'''
    match target:
        case nm.Integer(_value=whole):
            return float(whole)
        case nm.Constant() | nm.UnitOfMeasure():
            return target.to_float()
        case nm.Addition(content=terms):
            return sum(evaluate_float(term) for term in terms)
        case nm.Multiplication(content=terms):
            return math.prod(evaluate_float(term) for term in terms)
        case nm.Power(base=base, exponent=exponent):
            return evaluate_float(base) ** evaluate_float(exponent)
        case nm.Logarithm(base=base, argument=argument):
            return math.log(evaluate_float(argument), evaluate_float(base))
    raise NoFloatForTheNumeric(f'{target!r} holds no number this evaluation reads')


def refuses(build: Callable[[], object]) -> bool:
    try:
        build()
    except nm.DimensionMismatch:
        return True
    return False


def json_round_trip(target: nm.Numeric) -> nm.Numeric:
    converter = term_json.TermJSONConverter()
    return converter.reconstruct(converter.to_json(target))  # type: ignore[return-value]


def cases() -> dict[str, bool]:
    size = nm.FreeNumeric.named('n')
    bytes_per_second = units.GIGABYTE / units.SECOND
    return {
        'a prefixed unit scales the reference':
            evaluate_float(units.GIBIBYTE) == 8 * 2**30,
        'a decimal prefix and a binary prefix differ':
            (units.GIGABYTE != units.GIBIBYTE
             and evaluate_float(units.GIGABYTE) == 8 * 10**9),
        'a sub-multiple is a ratio':
            math.isclose(evaluate_float(units.MICROSECOND), 1e-6),
        'a quantity evaluates in the reference units':
            evaluate_float(3 * units.GIBIBYTE) == 3 * 8 * 2**30,
        'a conversion divides the scales':
            math.isclose(
                evaluate_float(
                    nm.convert_to_unit(4 * units.GIGABYTE, units.MEBIBYTE)),
                4e9 / 2**20),
        'a rate converts to a rate':
            math.isclose(evaluate_float(nm.convert_to_unit(
                units.GIGABYTE / units.MILLISECOND, units.TERABYTE / units.SECOND)), 1.0),
        'a watt is a joule per second':
            nm.dimensions_of(units.WATT) == nm.dimensions_of(units.JOULE / units.SECOND)
            and evaluate_float(nm.convert_to_unit(
                units.WATT, units.JOULE / units.SECOND)) == 1.0,
        'a gigahertz is one per nanosecond':
            math.isclose(
                evaluate_float(
                    nm.convert_to_unit(units.GIGAHERTZ, 1 / units.NANOSECOND)),
                1.0),
        'a product of units is one canonical term':
            units.GIBIBYTE * units.SECOND == units.SECOND * units.GIBIBYTE
            and size * units.GIBIBYTE == units.GIBIBYTE * size,
        'the unit sorts after the number and the symbols':
            (3 * size * units.GIBIBYTE).content == (nm.Integer(3), size, units.GIBIBYTE),
        'a rate reads its dimensions':
            nm.dimensions_of(bytes_per_second) == {units.INFORMATION: 1, units.TIME: -1},
        'a numeric without a unit has no dimension':
            nm.dimensions_of(3 * size) == {},
        'a sum across dimensions is refused':
            refuses(lambda: nm.dimensions_of(units.BYTE + units.SECOND)),
        'a conversion across dimensions is refused':
            refuses(lambda: nm.convert_to_unit(units.SECOND, units.BYTE)),
        'a unit is a constant under differentiation':
            differentiate_numeric.differentiate(nm.x * units.GIBIBYTE) == units.GIBIBYTE,
        'a unit survives JSON':
            json_round_trip(units.GIBIBYTE) == units.GIBIBYTE
            and json_round_trip(units.WATT) == units.WATT,
        'a unit survives pickling':
            pickle.loads(pickle.dumps(units.WATT)) == units.WATT,
        'a unit prints its symbol upright':
            units.GIBIBYTE.to_latex() == '\\mathrm{GiB}',
    }


if __name__ == '__main__':
    results = cases()
    for name, passed in results.items():
        print(f'  {"ok  " if passed else "FAIL"}  {name}')
    sys.exit(0 if all(results.values()) else 1)
