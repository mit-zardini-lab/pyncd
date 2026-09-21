'''Checking the minus sign a sum and a product are written with.

Written by Claude Fable 5.1, effort 80.

    python data_structure/validate_numeric_signs.py

Each case is a line of `obsidian/01-foundations/Numerics.md`, under the heading
on how a numeric prints. `tsncd` asserts the same strings in
`test/numeric_signs.test.ts`.
'''
from __future__ import annotations
import sys

import data_structure.Numeric as nm


def cases() -> dict[str, bool]:
    a, b = nm.FreeNumeric.named('a'), nm.FreeNumeric.named('b')
    negative_a = nm.Multiplication(content=(nm.Integer(-1), a))
    negative_b = nm.Multiplication(content=(nm.Integer(-1), b))
    negative_two_b = nm.Multiplication(content=(nm.Integer(-2), b))
    two_negative_factors = nm.Multiplication(content=(nm.Integer(-2), negative_a))
    three_negative_factors = nm.Multiplication(
        content=(nm.Integer(-2), negative_a, negative_b))
    negated_sum = nm.Multiplication(
        content=(nm.Integer(-1), nm.Addition(content=(a, b))))
    return {
        'a sum writes a negative coefficient as a subtraction':
            nm.Addition(content=(a, negative_two_b)).to_latex() == 'a - 2 b',
        'a sum writes a negation as a subtraction':
            nm.Addition(content=(a, negative_b)).to_latex() == 'a - b',
        'a sum writes a negative integer as a subtraction':
            nm.Addition(content=(a, nm.Integer(-3))).to_latex() == 'a - 3',
        'a product of two negative factors carries no sign':
            not nm.is_negative(two_negative_factors)
            and two_negative_factors.to_latex() == '2 a',
        'a product of three negative factors carries one sign':
            nm.is_negative(three_negative_factors)
            and three_negative_factors.to_latex() == '-2 a b',
        'a negated sum is bracketed':
            negated_sum.to_latex() == '-(a + b)'
            and nm.Addition(content=(a, negated_sum)).to_latex() == 'a - (a + b)',
        'a term that is not negative is returned as it is':
            nm.without_sign(a) is a and not nm.is_negative(a),
    }


if __name__ == '__main__':
    results = cases()
    for name, passed in results.items():
        print(f'  {"ok  " if passed else "FAIL"}  {name}')
    sys.exit(0 if all(results.values()) else 1)
