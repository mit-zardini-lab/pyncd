from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Mapping
import math
import data_structure.Term as fd # for 'foundations'

'''
EQUALITY. Numerics compare STRUCTURALLY, like every other Term: two are equal
when they are the same construction of the same parts.

They used to compare by a `numeric_hash` taken modulo 2**16-1, with addition
summing its parts' hashes and multiplication multiplying them, so that
`2x + 2x == 4x` fell out arithmetically. It bought a little algebra and cost
correctness: it never handled division (`x/y*y != x`) or powers
(`x*x != x**2`), and being a hash modulo a small number it reported
`Integer(65535) == Integer(0)` and `Integer(65536) == Integer(1)` - silently
wrong answers, in a value used for tile counts and array sizes.

What is kept is the cheap CANONICALISATION that `template` does when a term is
built: flattening nested associatives, dropping identities, folding integer
constants, and storing commutative operands in a canonical order so that
`x + y` and `y + x` really are the same term. The canonicalisation is
normalisation rather than
algebra: it makes structural equality agree with arithmetic on the cases that
are decidable by looking.

Anything deeper, such as collecting like terms, cancelling `x/x`, or deciding whether two
symbolic expressions denote the same function, is genuine algebra and belongs
in the numeric-algebra section at the foot of this file, applied
explicitly rather than smuggled into `==`.
'''

@dataclass(frozen=True)
class Numeric(fd.Term, ABC):
    def __add__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return Addition.template(self, other)
    def __radd__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return Addition.template(other, self)
    def __mul__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return Multiplication.template(self, other)
    def __rmul__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return Multiplication.template(other, self)
    def __pow__(self, exponent: Numeric | int) -> Numeric:
        exponent = exponent if isinstance(exponent, Numeric) else Integer(exponent)
        return Power.template(self, exponent)
    def __rpow__(self, base: Numeric | int) -> Numeric:
        base = base if isinstance(base, Numeric) else Integer(base)
        return Power.template(base, self)
    def __truediv__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return division(self, other)
    def __rtruediv__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return division(other, self)
    def __sub__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return Addition.template(self, Multiplication.template(Integer(-1), other))
    def __rsub__(self, other: Numeric | int) -> Numeric:
        other = other if isinstance(other, Numeric) else Integer(other)
        return Addition.template(other, Multiplication.template(Integer(-1), self))
    def __neg__(self) -> Numeric:
        return Multiplication.template(Integer(-1), self)
    
    @abstractmethod
    def to_latex(self) -> str: ...

@dataclass(frozen=True)
class FreeNumeric(Numeric):
    uid: fd.UID[FreeNumeric] = field(default_factory=lambda: fd.UID(FreeNumeric))
    @classmethod
    def field(cls):
        return field(default_factory=lambda: FreeNumeric())
    @classmethod
    def named(cls, name: str | fd.DynamicName, force: bool = True):
        name = fd.DynamicName.from_str(name)
        return cls(fd.UID(FreeNumeric, fd.hash_id(name) if force else fd.fresh_id(), name))
    def to_latex(self) -> str:
        return self.uid.to_latex()

@dataclass(frozen=True)
class Integer(Numeric):
    _value: int = 0
    def to_latex(self) -> str:
        return str(self._value)

@dataclass(frozen=True)
class FreeInput(Numeric):
    '''
    The one variable of a formula. It has no uid, because every `FreeInput` is the
    same input, which is what makes a formula a function of one argument: an
    `ops.Arithmetic` is the elementwise map `x -> formula`, and
    `solver.algebra.differentiate_numeric.differentiate` differentiates with
    respect to it. Printed as `x`, wherever it sits in the formula.
    '''
    def to_latex(self) -> str:
        return 'x'

class ConstantSymbol(Enum):
    '''The constants a `Constant` names. Each value is the latex it prints as.'''
    EULER = 'e'
    PI = '\\pi'
    IMAGINARY_UNIT = '\\mathrm{i}'
    INFINITY = '\\infty'
fd.register_enum(ConstantSymbol)

@dataclass(frozen=True)
class Constant(Numeric):
    '''A mathematical constant, identified by its symbol. `to_float` gives its
    value.

    The default symbol is Euler's number, so that `Power(Constant(), x)` is the
    exponential and `Logarithm(Constant(), x)` the natural logarithm. The
    templates and the derivative both handle those two forms. The imaginary unit
    has no float, and `Power(Constant(), Constant(IMAGINARY_UNIT) * x)` is the
    complex exponential a rotary embedding multiplies by. Positive infinity is the
    score a selection gives an entry it has to keep, because a Top-K keeps the
    largest scores and no score is larger.
    '''
    symbol: ConstantSymbol = ConstantSymbol.EULER
    def to_latex(self) -> str:
        return self.symbol.value
    def to_float(self) -> float:
        match self.symbol:
            case ConstantSymbol.EULER:
                return math.e
            case ConstantSymbol.PI:
                return math.pi
            case ConstantSymbol.INFINITY:
                return math.inf
        raise ValueError(f'no float for {self.symbol}')

@dataclass(frozen=True)
class DimensionOfMeasure(fd.Term):
    '''The kind of quantity a `UnitOfMeasure` measures.

    A base dimension, such as time, has an empty composition. A derived dimension,
    such as power, is composed of other dimensions each raised to an integer
    exponent, and `base_dimensions` reads it down to the base ones. A dimension is
    declared wherever a new kind of quantity is needed, and two declarations with
    one name and one composition are the same dimension. The declared dimensions
    and units are in `data_structure/UnitsOfMeasure.py`.
    '''
    name: str
    composition: fd.Prod[tuple[DimensionOfMeasure, int]] = ()

    def base_dimensions(self) -> dict[DimensionOfMeasure, int]:
        if not self.composition:
            return {self: 1}
        return merge_exponents(*(
            scale_exponents(part.base_dimensions(), power)
            for part, power in self.composition))

    def declare_unit(self, name: str, symbol: str, scale_numerator: int = 1,
                     scale_denominator: int = 1) -> UnitOfMeasure:
        '''A unit of this dimension whose scale, in its reference units, is the
        ratio of the two integers, stored in lowest terms.'''
        common = math.gcd(scale_numerator, scale_denominator)
        return UnitOfMeasure(
            dimension=self, name=name, symbol=symbol,
            scale_numerator=scale_numerator // common,
            scale_denominator=scale_denominator // common)

def merge_exponents(
    *exponents: Mapping[DimensionOfMeasure, int]
) -> dict[DimensionOfMeasure, int]:
    '''The sum of the exponent maps, without the dimensions whose exponents cancel.'''
    total: dict[DimensionOfMeasure, int] = {}
    for mapping in exponents:
        for dimension, exponent in mapping.items():
            total[dimension] = total.get(dimension, 0) + exponent
    return {dimension: exponent for dimension, exponent in total.items() if exponent != 0}

def scale_exponents(
    exponents: Mapping[DimensionOfMeasure, int], factor: int
) -> dict[DimensionOfMeasure, int]:
    return {dimension: exponent * factor for dimension, exponent in exponents.items()}

def dimension_text(exponents: Mapping[DimensionOfMeasure, int]) -> str:
    if not exponents:
        return 'no dimension'
    return ' '.join(f'{dimension.name}^{exponent}'
                    for dimension, exponent in exponents.items())

@dataclass(frozen=True)
class UnitOfMeasure(Numeric):
    '''A unit of measure, such as the byte or the microsecond, as a constant whose
    value is its scale in the reference units of its dimension.

    A quantity is a numeric multiplied by a unit, so `3 * GIBIBYTE` is a
    `Multiplication` and `GIGABYTE / SECOND` is a rate, and each evaluates to a
    float in the reference units. The scale is a ratio of two integers rather
    than a `Numeric`, so that `direct_subterms` reads a unit as a leaf and a
    sub-multiple such as the microsecond needs no symbolic power.
    `dimensions_of` reads the dimensions a numeric carries and `convert_to_unit`
    writes a quantity in another unit.
    '''
    dimension: DimensionOfMeasure
    name: str
    symbol: str
    scale_numerator: int = 1
    scale_denominator: int = 1

    def __post_init__(self) -> None:
        if self.scale_numerator < 1 or self.scale_denominator < 1:
            raise ValueError(
                f'The scale of {self.name} is {self.scale_numerator} over '
                f'{self.scale_denominator}, and a scale is a ratio of positive integers')
        if math.gcd(self.scale_numerator, self.scale_denominator) != 1:
            raise ValueError(
                f'The scale of {self.name} is {self.scale_numerator} over '
                f'{self.scale_denominator}, which is not in lowest terms')

    def to_latex(self) -> str:
        return f'\\mathrm{{{self.symbol}}}'

    def to_float(self) -> float:
        return self.scale_numerator / self.scale_denominator

def canonical_order(target: Numeric) -> tuple:
    '''
    A total order on Numerics, so that the operands of a commutative operation
    can be stored one way round and `x + y` is literally the same term as
    `y + x`.

    Named symbols order by NAME, and only fall back to the uid to separate
    unnamed ones. Ordering on uid alone would still be a total order, but uids
    are random per process, so `a * b` would come out in a different order every
    run and two builds of the same expression would not be equal terms.
    '''
    match target:
        case Integer(_value=value):
            return (0, value)
        case FreeNumeric(uid=uid):
            return (1, uid._name.to_bodies() if uid._name is not None else '',
                    uid._id)
        case Constant(symbol=symbol):
            return (1, symbol.value, 0)
        case FreeInput():
            return (1, 'x', 0)
        case UnitOfMeasure(dimension=dimension, symbol=symbol):
            return (5, dimension.name, symbol)
        case Associative(content=content):
            return (2, type(target).__qualname__,
                    tuple(canonical_order(part) for part in content))
        case Power(base=base, exponent=exponent):
            return (3, canonical_order(base), canonical_order(exponent))
        case (CumulativeGaussian(argument=argument) | Sigmoid(argument=argument)
              | IsPositive(argument=argument) | RectifiedLinear(argument=argument)
              | Sign(argument=argument) | AbsoluteValue(argument=argument)
              | SquareRoot(argument=argument)):
            return (4, type(target).__qualname__, canonical_order(argument))
        case Clamp(argument=argument, lower=lower, upper=upper):
            return (4, type(target).__qualname__, canonical_order(argument),
                    canonical_order(lower), canonical_order(upper))
        case LargerOf(first=first, second=second):
            return (4, type(target).__qualname__, canonical_order(first),
                    canonical_order(second))
    return (4, type(target).__qualname__)

@dataclass(frozen=True)
class Associative(Numeric, ABC):
    content: fd.Prod[Numeric] = ()
    @classmethod
    def template(cls, *xs: Numeric) -> Numeric:
        '''
        Build the operation, canonicalised: nested instances of the same
        operation flattened, identities dropped, integer constants folded into one,
        and the rest stored in `canonical_order`. All of it is decidable by
        looking at the term. The note at the top of this file states what is
        deliberately left undone here.
        '''
        expanded = util.concat(cls.expand(x) for x in xs)
        constants = tuple(x for x in expanded if isinstance(x, Integer))
        rest = tuple(x for x in expanded if not isinstance(x, Integer))
        folded = cls.fold(tuple(x._value for x in constants))
        if folded == cls.absorbing:
            return folded
        expanded = (rest if folded == cls.identity else (folded, *rest))
        if len(expanded) == 0:
            return cls.identity
        if len(expanded) == 1:
            return expanded[0]
        return cls(content=tuple(sorted(expanded, key=canonical_order)))
    @classmethod
    def fold(cls, values: fd.Prod[int]) -> Integer:
        raise NotImplementedError
    @classmethod
    def expand(cls, target: Numeric) -> fd.Prod[Numeric]:
        match target:
            case Associative(content=xs) if isinstance(target, cls):
                return util.concat(cls.expand(x) for x in xs)
            case _ if target == cls.identity:
                return ()
            case _:
                return (target,)
    def __init_subclass__(cls, identity: Numeric, absorbing: Numeric | None = None,
                          sep: str = " ") -> None:
        cls.sep = sep
        cls.identity = identity
        # The value that swallows the operation: 0 for multiplication, nothing
        # for addition.
        cls.absorbing = absorbing
        return super().__init_subclass__()
    def to_latex(self) -> str:
        return f" {self.sep} ".join(x.to_latex() for x in self.content)

@dataclass(frozen=True)
class Addition(Associative, identity = Integer(0), sep = "+"):
    @classmethod
    def fold(cls, values: fd.Prod[int]) -> Integer:
        return Integer(sum(values))
    def to_latex(self) -> str:
        head, *tail = self.content
        return head.to_latex() + ''.join(signed_latex(x) for x in tail)

def is_negative(target: Numeric) -> bool:
    '''Whether `target` is written with a minus sign in front of it: an `Integer`
    below zero, or a product holding an odd number of negative factors.'''
    match target:
        case Integer(_value=value):
            return value < 0
        case Multiplication(content=factors):
            return sum(is_negative(factor) for factor in factors) % 2 == 1
    return False

def without_sign(target: Numeric) -> Numeric:
    '''`target` with the minus sign taken off every negative factor, so `-2 * -x`
    and `-2 * x` both become `2 * x`. A term `is_negative` reports as negative
    equals the negation of the result, and any other term equals the result.'''
    match target:
        case Integer(_value=value):
            return Integer(abs(value))
        case Multiplication(content=factors):
            return Multiplication.template(
                *(without_sign(factor) for factor in factors))
    return target

def signed_latex(target: Numeric) -> str:
    '''One term of a sum with the sign that joins it to the term before.'''
    sign = '-' if is_negative(target) else '+'
    return f' {sign} {juxtaposed_latex(without_sign(target))}'

def parenthesised_latex(target: Numeric) -> str:
    return f'({target.to_latex()})'

def product_part_latex(target: Numeric) -> str:
    '''One factor of a product, bracketed where it is a sum.'''
    return (parenthesised_latex(target) if isinstance(target, Addition)
            else target.to_latex())

def juxtaposed_latex(unsigned: Numeric) -> str:
    '''The factors of `unsigned` side by side, where `unsigned` holds no negative
    factor. `Multiplication.to_latex` cannot write them itself, because it writes
    the sign first and would be called again on the product it had unsigned.'''
    match unsigned:
        case Multiplication(content=factors):
            return ' '.join(product_part_latex(factor) for factor in factors)
    return product_part_latex(unsigned)

@dataclass(frozen=True)
class Multiplication(Associative, identity = Integer(1), absorbing = Integer(0), sep = "*"):
    @classmethod
    def fold(cls, values: fd.Prod[int]) -> Integer:
        return Integer(math.prod(values))
    def to_latex(self) -> str:
        '''A product by juxtaposition, as mathematics writes one, so a stride
        reads `3pq`. A negative product carries one minus sign, in front, so
        `-2 * -x` reads `2 x` and never `-2 -x`. `sep` is the star that
        `display.display_numeric` sets in a box of its own and
        `display.node_category` writes into a string.'''
        sign = '-' if is_negative(self) else ''
        return sign + juxtaposed_latex(without_sign(self))

@dataclass(frozen=True)
class Power(Numeric):
    base: Numeric
    exponent: Numeric
    @classmethod
    def template(cls, base: Numeric, exponent: Numeric) -> Numeric:
        if exponent == Integer(0):
            return Integer(1)
        if exponent == Integer(1):
            return base
        if base == Integer(1):
            return Integer(1)
        # Integer powers of integers evaluate, except the negative ones, which
        # are not integers and stay symbolic.
        if isinstance(base, Integer) and isinstance(exponent, Integer) \
                and exponent._value > 0:
            return Integer(base._value ** exponent._value)
        return cls(base=base, exponent=exponent)
    def to_latex(self) -> str:
        # A power of one half is written as the root it is, which needs no
        # parentheses because the radical groups its base.
        if self.exponent == Power(base=Integer(2), exponent=Integer(-1)):
            return f"\\sqrt{{{self.base.to_latex()}}}"
        # The exponent's braces group it already. A sum or a product in the base
        # would otherwise run into the caret.
        base = (parenthesised_latex(self.base) if isinstance(self.base, Associative)
                else self.base.to_latex())
        return f"{base}^{{{self.exponent.to_latex()}}}"

@dataclass(frozen=True)
class Logarithm(Numeric):
    base: Numeric
    argument: Numeric

    @classmethod
    def template(cls, base: Numeric, argument: Numeric) -> Numeric:
        # The mirror of `Power.template`: an exact integer logarithm evaluates,
        # anything else stays symbolic. log_b(1) is 0 for every base, so that
        # case does not need the base to be an integer.
        if argument == Integer(1):
            return Integer(0)
        if argument == base:
            return Integer(1)
        if (isinstance(base, Integer) and isinstance(argument, Integer)
                and base._value > 1 and argument._value > 0):
            exponent = round(math.log(argument._value, base._value))
            if base._value ** exponent == argument._value:
                return Integer(exponent)
        return cls(base=base, argument=argument)

    def to_latex(self) -> str:
        if self.base == Constant(ConstantSymbol.EULER):
            return f"\\ln({self.argument.to_latex()})"
        return f"log_{{{self.base.to_latex()}}}({self.argument.to_latex()})"

@dataclass(frozen=True)
class Conjugate(Numeric):
    '''The complex conjugate of the argument, printed `\\overline{argument}`.

    The conjugate of `e^{\\mathrm{i} y}` for a real angle `y` is
    `e^{-\\mathrm{i} y}`, so a turn clockwise through an angle is the
    conjugate of the turn counterclockwise through the same angle. The table of an
    inverse rotary embedding is written that way, in
    `deepseek.registries.standard_expansions.rotation_factors`.

    Conjugation is primitive here. It is not an `Expandable`, because every
    primitive reads a number whole and none of them takes a complex number apart
    into a real part and an imaginary part.
    '''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\overline{{{self.argument.to_latex()}}}"

@dataclass(frozen=True)
class CumulativeGaussian(Numeric):
    '''The standard normal cumulative distribution function, printed
    `\\Phi(argument)`.

    The distribution function has no closed form in the other numerics, so it is
    a numeric of its own. Its derivative does have one, and
    `standard_normal_density` writes it.
    '''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\Phi({self.argument.to_latex()})"

@dataclass(frozen=True)
class Expandable(Numeric, ABC):
    '''A numeric that the primitive numerics can also spell. Every numeric that
    is not an `Expandable` is primitive.

    An `Expandable` stays whole in a formula, so the formula and its derivative
    stay readable. `expand_to_primitives` writes the spelling of one node, and
    `expand_every_expandable` writes a whole formula in primitives. A pass
    expands two formulas only to compare them, and keeps each formula as
    written when the comparison fails.
    '''
    @abstractmethod
    def expand_to_primitives(self) -> Numeric: ...

@dataclass(frozen=True)
class IsPositive(Numeric):
    '''One where the argument is above zero and zero elsewhere, printed as the
    indicator `\\mathbbm{1}_{argument > 0}`.

    `\\mathbbm` is the `bbm` package's command, and KaTeX has none of its own.
    `tsncd` supplies it as a macro in `display/HTMLRender/katex_options.ts`,
    which every label a diagram draws goes through.
    '''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\mathbbm{{1}}_{{{self.argument.to_latex()} > 0}}"

@dataclass(frozen=True)
class Sigmoid(Expandable):
    '''The logistic function, printed `\\sigma(argument)`.

    Its expansion is `(1 + e^{-argument})^{-1}`, and the derivative of that
    spelling shares no subterm with it. `logistic_density` writes the
    derivative as `\\sigma(u) (1 - \\sigma(u))`, so the derivative of a formula
    holding a `Sigmoid` holds the same `Sigmoid`.
    '''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\sigma({self.argument.to_latex()})"
    def expand_to_primitives(self) -> Numeric:
        return division(Integer(1), Addition.template(
            Integer(1),
            Power.template(E, Multiplication.template(Integer(-1), self.argument))))

@dataclass(frozen=True)
class RectifiedLinear(Expandable):
    '''The rectified linear function, printed `\\mathrm{ReLU}(argument)`. Its
    expansion is `argument \\mathbbm{1}_{argument > 0}`.

    The class carries the function's full name because `ops.ReLU` carries the
    short one, and a `Term` class name is registered once. It has no
    derivative rule of its own. The rule for `Expandable` differentiates the
    expansion, and `IsPositive` differentiates to zero, so the result is
    `\\mathbbm{1}_{argument > 0}` times the argument's derivative.
    '''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\mathrm{{ReLU}}({self.argument.to_latex()})"
    def expand_to_primitives(self) -> Numeric:
        return Multiplication.template(self.argument, IsPositive(self.argument))

@dataclass(frozen=True)
class Clamp(Expandable):
    '''The argument where it lies between `lower` and `upper`, `lower` below
    them and `upper` above them. `lower` is assumed to be at most `upper`.

    The clamp to the unit interval is the default and prints between corner
    brackets alone, as `\\ulcorner argument \\lrcorner`. Any other pair of
    bounds is written on the closing bracket, `lower` as its subscript and
    `upper` as its superscript.

    Its expansion is
    `lower + (argument - lower) \\mathbbm{1}_{argument - lower > 0}
    - (argument - upper) \\mathbbm{1}_{argument - upper > 0}`. Below `lower`
    both indicators are zero. Between the bounds the first is one, and the sum
    is the argument. Above `upper` both are one, and the two copies of the
    argument cancel. The rule for `Expandable` differentiates the expansion to
    the difference of the two indicators times the argument's derivative, which
    is one between the bounds and zero outside them.
    '''
    argument: Numeric = FreeInput()
    lower: Numeric = Integer(0)
    upper: Numeric = Integer(1)
    def clamps_to_unit_interval(self) -> bool:
        return self.lower == Integer(0) and self.upper == Integer(1)
    def to_latex(self) -> str:
        clamped = f"\\ulcorner {self.argument.to_latex()} \\lrcorner"
        if self.clamps_to_unit_interval():
            return clamped
        return f"{clamped}_{{{self.lower.to_latex()}}}^{{{self.upper.to_latex()}}}"
    def expand_to_primitives(self) -> Numeric:
        above_lower = Addition.template(
            self.argument, Multiplication.template(Integer(-1), self.lower))
        above_upper = Addition.template(
            self.argument, Multiplication.template(Integer(-1), self.upper))
        return Addition.template(
            self.lower,
            Multiplication.template(above_lower, IsPositive(above_lower)),
            Multiplication.template(
                Integer(-1), above_upper, IsPositive(above_upper)))

@dataclass(frozen=True)
class Sign(Expandable):
    '''One above zero, minus one below zero and zero at zero, printed
    `\\operatorname{sign}(argument)`. Its expansion is
    `\\mathbbm{1}_{argument > 0} - \\mathbbm{1}_{-argument > 0}`, and both
    indicators differentiate to zero, so the sign does too.'''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\operatorname{{sign}}({self.argument.to_latex()})"
    def expand_to_primitives(self) -> Numeric:
        return Addition.template(
            IsPositive(self.argument),
            Multiplication.template(
                Integer(-1),
                IsPositive(Multiplication.template(Integer(-1), self.argument))))

@dataclass(frozen=True)
class AbsoluteValue(Expandable):
    '''The magnitude of the argument, printed `\\lvert argument \\rvert`. Its
    expansion is the argument times its sign.'''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\lvert {self.argument.to_latex()} \\rvert"
    def expand_to_primitives(self) -> Numeric:
        return Multiplication.template(self.argument, Sign(self.argument))

@dataclass(frozen=True)
class LargerOf(Expandable):
    '''The larger of `first` and `second`, printed `\\max(first, second)`. The
    class is named for what it returns because `ops.Maximum` is the reduction
    operator and a `Term` class name is registered once. Its expansion is
    `first + (second - first) \\mathbbm{1}_{second - first > 0}`, which is
    `first` where `first` is the larger and `second` elsewhere.'''
    first: Numeric = FreeInput()
    second: Numeric = Integer(0)
    def to_latex(self) -> str:
        return f"\\max({self.first.to_latex()}, {self.second.to_latex()})"
    def expand_to_primitives(self) -> Numeric:
        excess = Addition.template(
            self.second, Multiplication.template(Integer(-1), self.first))
        return Addition.template(
            self.first, Multiplication.template(excess, IsPositive(excess)))

@dataclass(frozen=True)
class SquareRoot(Expandable):
    '''The square root of the argument, printed `\\sqrt{argument}`. Its expansion
    is the power `argument^{1/2}`, which `Power.to_latex` prints as the same
    root, so the two spellings read alike and differ in the term they hold.'''
    argument: Numeric = FreeInput()
    def to_latex(self) -> str:
        return f"\\sqrt{{{self.argument.to_latex()}}}"
    def expand_to_primitives(self) -> Numeric:
        return Power.template(
            self.argument, Power(base=Integer(2), exponent=Integer(-1)))

def division(numerator: Numeric, denominator: Numeric) -> Numeric:
    return Multiplication.template(
        numerator,
        Power.template(denominator, Integer(-1))
    )

def standard_normal_density(argument: Numeric) -> Numeric:
    '''The standard normal density at `argument`, `(2 pi)^{-1/2} e^{-argument^2/2}`.'''
    return Multiplication.template(
        Power.template(
            Multiplication.template(Integer(2), Constant(ConstantSymbol.PI)),
            division(Integer(-1), Integer(2))),
        Power.template(
            Constant(ConstantSymbol.EULER),
            division(
                Multiplication.template(
                    Integer(-1), Power.template(argument, Integer(2))),
                Integer(2))))

def logistic_density(argument: Numeric) -> Numeric:
    '''The logistic density at `argument`, `\\sigma(u) (1 - \\sigma(u))`.

    The density is written through `Sigmoid` rather than through the
    exponential, so it names the value the logistic function already computed.
    '''
    sigmoid = Sigmoid(argument)
    return Multiplication.template(
        sigmoid,
        Addition.template(
            Integer(1), Multiplication.template(Integer(-1), sigmoid)))

@dataclass(frozen=True)
class Equality:
    left: Numeric
    right: Numeric

Zero = Integer(0)
# The formula vocabulary: `E ** x` is the exponential, `1 / x` the reciprocal.
x = FreeInput()
E = Constant(ConstantSymbol.EULER)
PI = Constant(ConstantSymbol.PI)

# ==========================================================================
# Numeric algebra. Every rewrite below is applied explicitly by a caller,
# and none of it is reached by `==`.
# ==========================================================================
import itertools
import utilities.utilities as util

def expand_to_addition(expr: Numeric) -> fd.Prod[Numeric]:
    match expr:
        case Addition(content=terms):
            return util.concat(expand_to_addition(term) for term in terms)
        case Multiplication(content=terms):
            subterms = [expand_to_addition(term) for term in terms]
            return tuple(
                Multiplication.template(*combination)
                for combination in itertools.product(*subterms)
            )
        case _:
            return (expr,)

def expand(expr: Numeric) -> Numeric:
    return Addition.template(*expand_to_addition(expr))

def apply_equalities(
    target: Numeric,
    *equalities: Equality
):
    assert all(isinstance(eq.left, FreeNumeric) for eq in equalities)
    for eq in equalities:
        if target == eq.left:
            return eq.right
    replacements = {
        k: (apply_equalities(v, *equalities) if isinstance(v, Numeric)
            else tuple(apply_equalities(vi, *equalities) for vi in v) if isinstance(v, tuple)
            else v)
        for k, v in target.dict().items()
    }
    return target.reconstruct(
        **replacements
    )

def direct_subterms(target: Numeric) -> fd.Prod[Numeric]:
    '''Every `Numeric` held directly by `target`, whether in a field of its own
    such as `Power.base` or in a `fd.Prod` such as `Associative.content`.
    '''
    found: list[Numeric] = []
    for value in target.dict().values():
        if isinstance(value, Numeric):
            found.append(value)
        elif isinstance(value, tuple):
            found.extend(part for part in value if isinstance(part, Numeric))
    return tuple(found)

class DimensionMismatch(ValueError):
    '''A sum of quantities of different dimensions, or a conversion between them.'''

def dimensions_of(target: Numeric) -> dict[DimensionOfMeasure, int]:
    '''The base dimension of every unit `target` carries, with its exponent, so
    that bytes per second reads as information to the first power and time to the
    minus first, and a numeric carrying no unit reads as an empty mapping. A sum
    whose parts differ in dimension raises, and so does a function applied to a
    quantity.'''
    match target:
        case UnitOfMeasure(dimension=dimension):
            return dimension.base_dimensions()
        case Multiplication(content=parts):
            return merge_exponents(*(dimensions_of(part) for part in parts))
        case Power(base=base, exponent=Integer(_value=exponent)):
            return merge_exponents(scale_exponents(dimensions_of(base), exponent))
        case Addition(content=parts):
            found = [dimensions_of(part) for part in parts]
            if any(part != found[0] for part in found):
                raise DimensionMismatch(
                    f'{target.to_latex()} adds quantities of '
                    + ' and '.join(dimension_text(part) for part in found))
            return found[0]
    carried = [part for part in direct_subterms(target) if dimensions_of(part)]
    if carried:
        raise DimensionMismatch(
            f'{target.to_latex()} applies a function to the quantities '
            + ', '.join(part.to_latex() for part in carried))
    return {}

def convert_to_unit(quantity: Numeric, unit: Numeric) -> Numeric:
    '''`quantity` as a multiple of `unit`, which is their quotient once the two are
    checked to share a dimension. `unit` may be a product of units, such as
    `GIGABYTE / SECOND`.'''
    ratio = division(quantity, unit)
    remaining = dimensions_of(ratio)
    if remaining:
        raise DimensionMismatch(
            f'{quantity.to_latex()} in units of {unit.to_latex()} leaves '
            f'{dimension_text(remaining)}')
    return ratio

def contains_free_input(target: Numeric) -> bool:
    '''Whether the formula mentions the free input at all. A formula that does
    not is a constant. A `FreeNumeric` is a size symbol rather than a variable,
    so it counts as a constant here.
    '''
    if isinstance(target, FreeInput):
        return True
    return any(contains_free_input(part) for part in direct_subterms(target))

def map_numerics_in_field[V](value: V, function: Callable[[Numeric], Numeric]) -> V:
    '''`function` applied to a field that is a `Numeric` or a `fd.Prod` of them.
    Any other field, and a `fd.Prod` nothing in which changed, is returned as
    it is.'''
    if isinstance(value, Numeric):
        return function(value)  # type: ignore[return-value]
    if isinstance(value, tuple):
        mapped = tuple(function(part) if isinstance(part, Numeric) else part
                       for part in value)
        return value if all(new is old for new, old in zip(mapped, value)) else mapped  # type: ignore[return-value]
    return value

def map_direct_subterms(
    target: Numeric, function: Callable[[Numeric], Numeric],
) -> Numeric:
    '''`target` with `function` applied to every `Numeric` it holds directly.

    The result is rebuilt through `template` wherever the class has one, so it
    is canonical. `target` itself is returned when nothing changed, which
    preserves sharing.
    '''
    fields = target.dict()
    mapped = {name: map_numerics_in_field(value, function)
              for name, value in fields.items()}
    if all(mapped[name] is fields[name] for name in fields):
        return target
    match target:
        case Associative():
            return type(target).template(*mapped['content'])
        case Power():
            return Power.template(mapped['base'], mapped['exponent'])
        case Logarithm():
            return Logarithm.template(mapped['base'], mapped['argument'])
    return target.reconstruct(**mapped)

def expand_every_expandable(target: Numeric) -> Numeric:
    '''`target` with every `Expandable` in it written in the primitive numerics.'''
    expanded = map_direct_subterms(target, expand_every_expandable)
    if isinstance(expanded, Expandable):
        return expand_every_expandable(expanded.expand_to_primitives())
    return expanded

def substitute_subterm(
    target: Numeric, subterm: Numeric, replacement: Numeric,
) -> Numeric:
    '''`target` with every occurrence of `subterm` replaced by `replacement`.

    A sum or a product also occurs wherever its operands are a sub-multiset of
    the operands of a sum or a product of the same kind, because the operation
    is associative and commutative. `x \\sigma(x)` therefore occurs in
    `2 x \\sigma(x) (1 - \\sigma(x))`.
    '''
    if target == subterm:
        return replacement
    if isinstance(subterm, Associative) and type(target) is type(subterm):
        operands, wanted = Counter(target.content), Counter(subterm.content)
        copies = min(operands[part] // count for part, count in wanted.items())
        if copies > 0:
            rest = operands - Counter(
                {part: count * copies for part, count in wanted.items()})
            return type(target).template(
                *((replacement,) * copies),
                *(substitute_subterm(part, subterm, replacement)
                  for part in rest.elements()))
    return map_direct_subterms(
        target, lambda part: substitute_subterm(part, subterm, replacement))

def outer_function(composite: Numeric, inner: Numeric) -> Numeric | None:
    '''The formula h with `composite == h(inner)`, for two formulas in the free
    input.

    Every occurrence of `inner` in `composite` is replaced by a fresh symbol.
    If the free input then appears nowhere, what remains is h, with the symbol
    written as the free input. The result is None when `inner` does not occur,
    and when `composite` also reads the free input outside `inner`.
    '''
    if inner == FreeInput() or not contains_free_input(inner):
        return None
    symbol = FreeNumeric()
    replaced = substitute_subterm(composite, inner, symbol)
    if replaced == composite or contains_free_input(replaced):
        return None
    return substitute_subterm(replaced, symbol, FreeInput())


        
def monomial_coefficients(target: Numeric) -> dict[Numeric, int]:
    '''The integer coefficient of every product of symbols in `target` written as a
    sum of products, keyed by the product with its integer factors removed, and by
    `Integer(1)` for the constant. A product whose coefficients cancel is left out.
    '''
    coefficients: dict[Numeric, int] = {}
    for term in expand_to_addition(target):
        factors = Multiplication.expand(term)
        coefficient = math.prod(
            factor._value for factor in factors if isinstance(factor, Integer))
        symbols = tuple(factor for factor in factors if not isinstance(factor, Integer))
        key = Multiplication.template(*symbols) if symbols else Integer(1)
        coefficients[key] = coefficients.get(key, 0) + coefficient
    return {key: value for key, value in coefficients.items() if value != 0}

def collect_like_terms(target: Numeric) -> Numeric:
    return Addition.template(*(
        Multiplication.template(Integer(coefficient), product)
        for product, coefficient in monomial_coefficients(target).items()))

def cancel_reciprocal_factors(target: Numeric) -> Numeric:
    '''`target` with every symbol of a product cancelled against its own
    reciprocal, so that `\\hat{v} 2^{63} \\hat{v}^{-1}` becomes `2^{63}`. A
    bare symbol counts one towards its exponent and a symbol raised to an
    integer counts that integer, and a symbol whose exponents sum to zero is
    dropped. A factor with an integer base is kept as it stands, because
    `Power.template` folds an integer power into one integer, and a bound such
    as `2^{63}` is kept as a power so that it survives the JSON transport.
    Anything other than a product is returned as it is.'''
    if not isinstance(target, Multiplication):
        return target
    kept: list[Numeric] = []
    exponents: dict[Numeric, int] = {}
    for factor in target.content:
        match factor:
            case Power(base=FreeNumeric() as base, exponent=Integer(_value=power)):
                exponents[base] = exponents.get(base, 0) + power
            case FreeNumeric():
                exponents[factor] = exponents.get(factor, 0) + 1
            case _:
                kept.append(factor)
    return Multiplication.template(*kept, *(
        Power.template(base, Integer(power))
        for base, power in exponents.items() if power != 0))

def is_product_of_symbols(target: Numeric) -> bool:
    match target:
        case Integer(_value=1) | FreeNumeric():
            return True
        case Multiplication(content=factors):
            return all(isinstance(factor, FreeNumeric) for factor in factors)
    return False

def is_nonnegative_for_positive_symbols(target: Numeric) -> bool:
    '''Whether `target` is at least zero under every assignment of a positive integer
    to each symbol. Every product of symbols is then at least one, so the condition
    is that no product carries a negative coefficient and the value with every
    symbol at one is at least zero. A form holding anything other than a product
    of symbols is reported as not provably so.
    '''
    coefficients = monomial_coefficients(target)
    if not all(is_product_of_symbols(product) for product in coefficients):
        return False
    if any(coefficient < 0 for product, coefficient in coefficients.items()
           if product != Integer(1)):
        return False
    return sum(coefficients.values()) >= 0

def is_nonpositive_for_positive_symbols(target: Numeric) -> bool:
    return is_nonnegative_for_positive_symbols(
        Multiplication.template(Integer(-1), target))

def is_positive_for_positive_symbols(target: Numeric) -> bool:
    '''Whether an integer-valued `target` is at least one under every assignment of
    a positive integer to each symbol.'''
    return is_nonnegative_for_positive_symbols(Addition.template(target, Integer(-1)))

def is_negative_for_positive_symbols(target: Numeric) -> bool:
    return is_nonpositive_for_positive_symbols(Addition.template(target, Integer(1)))

def is_zero(target: Numeric) -> bool:
    return not monomial_coefficients(target)

def free_symbols(target: Numeric) -> set[Numeric]:
    '''The `FreeNumeric`s that `target`, a sum of products of integers and symbols,
    is written in.'''
    match target:
        case FreeNumeric():
            return {target}
        case Addition(content=parts) | Multiplication(content=parts):
            return set().union(*(free_symbols(part) for part in parts))
    return set()

class NotAnAffineForm(ValueError):
    '''A numeric holding something other than integers, symbols, sums and products,
    handed to an evaluation that admits nothing else.'''

def evaluate_integer(target: Numeric, values: Mapping[Numeric, int]) -> int:
    '''The integer `target` takes with every symbol bound by `values`.'''
    match target:
        case Integer(_value=value):
            return value
        case FreeNumeric():
            return values[target]
        case Addition(content=parts):
            return sum(evaluate_integer(part, values) for part in parts)
        case Multiplication(content=parts):
            return math.prod(evaluate_integer(part, values) for part in parts)
    raise NotAnAffineForm(
        f'{target} holds something other than integers, symbols, sums and products')
