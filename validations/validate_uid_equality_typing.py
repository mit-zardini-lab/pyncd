'''Validate UID-bearing numeric substitution and its static type contract.'''
from __future__ import annotations

import pathlib
import sys

sys.path[0] = str(pathlib.Path(__file__).resolve().parents[1])

import data_structure.Numeric as nm
import data_structure.Term as fd


def check_uid_bearing_numeric_rewrites_to_integer() -> None:
    symbol = nm.FreeNumeric.named('typing_fixture_size')
    uid_equipped: fd.UIDEquipped = symbol
    if not isinstance(uid_equipped, fd.UIDEquipped):
        raise ValueError(f'{symbol} does not satisfy UIDEquipped')

    substitution: fd.EqualityClass[nm.FreeNumeric] = fd.EqualityClass(
        _type=nm.FreeNumeric,
        bucket={symbol.uid},
        canonical=nm.Integer(128),
    )
    actual = substitution.apply((symbol, symbol))
    expected = (nm.Integer(128), nm.Integer(128))
    if actual != expected:
        raise ValueError(f'{actual} differs from {expected}')


if __name__ == '__main__':
    check_uid_bearing_numeric_rewrites_to_integer()
    print('check_uid_bearing_numeric_rewrites_to_integer: ok')
