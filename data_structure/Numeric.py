from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC
import math
import data_structure.Term as fd # for 'foundations'

HASH_MODULO = 2**16 - 1

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
    
    def numeric_hash(self) -> int:
        return hash(self)  % HASH_MODULO
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Numeric):
            return False
        return self.numeric_hash() == other.numeric_hash()

@dataclass(frozen=True, eq=False)
class FreeNumeric(Numeric):
    uid: fd.UID[FreeNumeric] = field(default_factory=lambda: fd.UID(FreeNumeric))
    @classmethod
    def field(cls):
        return field(default_factory=lambda: FreeNumeric())
    def numeric_hash(self) -> int:
        return (self.uid._id + 12435) % HASH_MODULO

@dataclass(frozen=True, eq=False)
class Integer(Numeric):
    _value: int = 0
    def numeric_hash(self) -> int:
        return self._value  % HASH_MODULO

@dataclass(frozen=True, eq=False)
class Associative(Numeric, ABC):
    content: fd.Prod[Numeric] = ()
    @classmethod
    def template(cls, *xs: Numeric) -> Numeric:
        xs = tuple(x for x in xs if x != cls.unit)
        if len(xs) == 0:
            return cls.unit
        if len(xs) == 1:
            return xs[0]
        return cls(content=xs)
    def __init_subclass__(cls, unit: Numeric, sep: str = " ") -> None:
        cls.sep = sep
        cls.unit = unit
        return super().__init_subclass__()

@dataclass(frozen=True, eq=False)
class Addition(Associative, unit = Integer(0), sep = "+"):
    def numeric_hash(self) -> int:
        return sum(x.numeric_hash() for x in self.content)  % HASH_MODULO

@dataclass(frozen=True, eq=False)
class Multiplication(Associative, unit = Integer(1), sep = "*"):
    def numeric_hash(self) -> int:
        return math.prod(x.numeric_hash() for x in self.content)  % HASH_MODULO
    
@dataclass(frozen=True, eq=False)
class Power(Numeric):
    base: Numeric
    exponent: Numeric
    @classmethod
    def template(cls, base: Numeric, exponent: Numeric) -> Numeric:
        if exponent.numeric_hash() == 0:
            return Integer(1)
        if exponent.numeric_hash() == 1:
            return base
        return cls(base=base, exponent=exponent)
    def numeric_hash(self) -> int:
        # TODO: This will not work properly as x^-y will be <1
        # return (
        #     self.base.numeric_hash() ** self.exponent.numeric_hash()
        # )  % HASH_MODULO
        return hash(self) % HASH_MODULO

def division(numerator: Numeric, denominator: Numeric) -> Numeric:
    return Multiplication.template(
        numerator,
        Power.template(denominator, Integer(-1))
    )

@dataclass(frozen=True)
class Equality:
    left: Numeric
    right: Numeric