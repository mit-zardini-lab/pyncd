---
tags: [layer/foundations, concept]
code: data_structure/UnitsOfMeasure.py, data_structure/Numeric.py, data_structure/validate_units_of_measure.py
status: evolving
---

# Units of Measure

Written by Claude Fable 5.1, effort 80.

## What it is

A unit of measure is a `nm.UnitOfMeasure`, a leaf `Numeric` declared beside `Constant`.
Its value is its scale in the reference units of its dimension. The byte is 8 in bits,
the gibibyte is 8 · 2^30 in bits, and the microsecond is 1 over 10^6 in seconds. The
scale is stored as two integers in lowest terms rather than as a `Numeric`, so that
`direct_subterms` reads a unit as a leaf and a sub-multiple needs no symbolic power.

A quantity is a numeric multiplied by a unit. `3 * GIBIBYTE` is a `Multiplication` and
`GIGABYTE / SECOND` is a rate, and each evaluates to a float in the reference units of
its dimension. The numeric algebra is unchanged by the addition.
`canonical_order` sorts a unit after the integers and the symbols, so `3 * n * GIBIBYTE`
is one canonical term however it was written, and the derivative registry differentiates
a unit to zero as it does a `Constant`.

## Where it lives

`data_structure/Numeric.py` holds the two classes and the algebra on them.
`data_structure/UnitsOfMeasure.py` holds the declared dimensions, units and prefixes.

| name | what it is |
|---|---|
| `nm.DimensionOfMeasure` | the kind of quantity a unit measures: a name and a composition. A base dimension has an empty composition. A derived dimension is composed of other dimensions each raised to an integer exponent, and `base_dimensions()` reads it down to the base ones. `declare_unit(name, symbol, scale_numerator, scale_denominator)` declares a unit of it |
| `nm.UnitOfMeasure` | a unit: its dimension, a name, a symbol, and a scale. `to_float()` is the scale, and `to_latex()` writes the symbol upright, as `\mathrm{GiB}` |
| `nm.dimensions_of` | the base dimension of every unit a numeric carries, with its exponent. Bytes per second reads as information to the first power and time to the minus first. A sum whose parts differ in dimension raises `nm.DimensionMismatch` |
| `nm.convert_to_unit` | a quantity as a multiple of another unit, which is their quotient once the two are checked to share a dimension. The result is symbolic, and `numeric_value` gives the number |
| `DecimalPrefix`, `BinaryPrefix` | the prefixes, each a symbol and the power of ten or of two it scales by |
| `prefixed(unit, prefix)` | the unit scaled by the prefix, named and written with the prefix in front |

The declared dimensions are information, time, operations, clock cycles and energy,
which are base dimensions, and frequency and power, which are composed. The frequency
dimension is time to the minus first, and the power dimension is energy times time to the
minus first. The declared units are the bit, the byte with its four binary and four
decimal prefixes, the second with its milli, micro and nano prefixes, the operation, the
clock cycle, the joule, the hertz, the gigahertz and the watt.

A dimension is declared wherever a new kind of quantity is needed. Two declarations with
one name and one composition are the same dimension, because a `DimensionOfMeasure` has
no uid and compares by its fields. For the same reason a unit is the same term in a
worker process as in its parent, which a `FreeNumeric` only manages through `hash_id`.

## Reading a quantity in another unit

`nm.convert_to_unit(4 * GIGABYTE, MEBIBYTE)` is the quotient of the two, a numeric whose
value is `4e9 / 2**20`. The unit may itself be a product, as in
`convert_to_unit(GIGABYTE / MILLISECOND, TERABYTE / SECOND)`, whose value is one. A
conversion whose quotient still carries a dimension raises. The check is dimensional
analysis, applied explicitly, in keeping with the rule in [[Numerics]] that anything
beyond canonicalisation is applied explicitly rather than placed in `template`.

## Why a unit is a numeric rather than a wrapper

A `Measure` type wrapping a numeric and a unit was considered and rejected. A quantity
stays symbolic until something evaluates it, and a rate is a quotient such as bytes per
second. A wrapper would have to re-implement every operator, would need dimension
arithmetic of its own for the quotients, and would have to be unwrapped in every `match`
that reads a numeric. A unit that is a leaf numeric needs one new case in each reader.

## The reference unit of information is the bit

The bit is the reference unit of the information dimension, so a quantity stated in
bytes carries the byte as its scale and a quantity read back in bytes costs one division
by `BYTE`. The bit was chosen because it names the dimension and keeps a sub-byte element
format exact, and an element of a quantisation of [[Quantization]] may be four bits.

## Where a unit is read

Each reader of a numeric has a case for a unit. `canonical_order` keys it by dimension
and symbol. `display/display_numeric.py` prints its symbol, and
`torch_compile/torch_compile.py` fills a tensor with its scale, both of them treating it
as they treat a `Constant`. `solver/registries/numeric_derivative.py` registers it with
the constants, so it differentiates to zero. `tsncd` has no box for it, so a unit cannot
yet be drawn.

`data_structure/validate_units_of_measure.py` checks each fact above, and
`validate_repository.py` runs it.

## See also

- [[Numerics]] — the algebra a unit is a leaf of
- [[Terms]] — why a term without a uid compares by its fields
