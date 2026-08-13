from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
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
built: flattening nested associatives, dropping units, folding integer
constants, and storing commutative operands in a canonical order so that
`x + y` and `y + x` really are the same term. That is normalisation, not
algebra: it makes structural equality agree with arithmetic on the cases that
are decidable by looking.

Anything deeper - collecting like terms, cancelling `x/x`, deciding whether two
symbolic expressions denote the same function - is genuine algebra and belongs
in the SOME NUMERIC ALGEBRA section at the foot of this file, applied
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
        case Associative(content=content):
            return (2, type(target).__qualname__,
                    tuple(canonical_order(part) for part in content))
        case Power(base=base, exponent=exponent):
            return (3, canonical_order(base), canonical_order(exponent))
    return (4, type(target).__qualname__)

@dataclass(frozen=True)
class Associative(Numeric, ABC):
    content: fd.Prod[Numeric] = ()
    @classmethod
    def template(cls, *xs: Numeric) -> Numeric:
        '''
        Build the operation, canonicalised: nested instances of the same
        operation flattened, units dropped, integer constants folded into one,
        and the rest stored in `canonical_order`. All of it is decidable by
        looking at the term - see the note at the top of this file for what
        deliberately is NOT done here.
        '''
        expanded = util.concat(cls.expand(x) for x in xs)
        constants = tuple(x for x in expanded if isinstance(x, Integer))
        rest = tuple(x for x in expanded if not isinstance(x, Integer))
        folded = cls.fold(tuple(x._value for x in constants))
        if folded == cls.absorbing:
            return folded
        expanded = (rest if folded == cls.unit else (folded, *rest))
        if len(expanded) == 0:
            return cls.unit
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
            case _ if target == cls.unit:
                return ()
            case _:
                return (target,)
    def __init_subclass__(cls, unit: Numeric, absorbing: Numeric | None = None,
                          sep: str = " ") -> None:
        cls.sep = sep
        cls.unit = unit
        # The value that swallows the operation: 0 for multiplication, nothing
        # for addition.
        cls.absorbing = absorbing
        return super().__init_subclass__()
    def to_latex(self) -> str:
        return f" {self.sep} ".join(x.to_latex() for x in self.content)

@dataclass(frozen=True)
class Addition(Associative, unit = Integer(0), sep = "+"):
    @classmethod
    def fold(cls, values: fd.Prod[int]) -> Integer:
        return Integer(sum(values))

@dataclass(frozen=True)
class Multiplication(Associative, unit = Integer(1), absorbing = Integer(0), sep = "*"):
    @classmethod
    def fold(cls, values: fd.Prod[int]) -> Integer:
        return Integer(math.prod(values))

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
        # Integer powers of integers evaluate, except the negative ones, which
        # are not integers and stay symbolic.
        if isinstance(base, Integer) and isinstance(exponent, Integer) \
                and exponent._value > 0:
            return Integer(base._value ** exponent._value)
        return cls(base=base, exponent=exponent)
    def to_latex(self) -> str:
        return f"{self.base.to_latex()}^{{{self.exponent.to_latex()}}}"

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
        if (isinstance(base, Integer) and isinstance(argument, Integer)
                and base._value > 1 and argument._value > 0):
            exponent = round(math.log(argument._value, base._value))
            if base._value ** exponent == argument._value:
                return Integer(exponent)
        return cls(base=base, argument=argument)

    def to_latex(self) -> str:
        return f"log_{{{self.base.to_latex()}}}({self.argument.to_latex()})"
def division(numerator: Numeric, denominator: Numeric) -> Numeric:
    return Multiplication.template(
        numerator,
        Power.template(denominator, Integer(-1))
    )

@dataclass(frozen=True)
class Equality:
    left: Numeric
    right: Numeric

Zero = Integer(0)

### SOME NUMERIC ALGEBRA
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

# def replace(
#     old: Numeric,
#     new: Numeric,
#     target: Numeric
# ):
#     if target == old:
#         return new
#     if isinstance(target, Associative) and isinstance(new, Associative) and type(target) == type(new):
#         if 
#     target = expand(target)
#     target = target.reconstruct(
#         **{k: replace(old, new, v)
#             for k, v in target.dict().items()}
#     )

        