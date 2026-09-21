# Claude Fable 5.1, effort 80.
'''Writing a formula in the index notation the reviewer ruled on 2026-09-17.

An index of the axis `m` is written `i_{m}`. A sum names the index it iterates and the
axis the index ranges over, `\\sum_{i_{m} \\in m}`, and an array is read at an index in
brackets, `x[i_{m}]`. `CLAUDE.md` states the rule. The functions here write the LaTeX
of each piece from the axes of an operator, so the formula an inspection box shows
over an operator names that operator's own axes.

An axis is written by the letter its name draws, without the size a display pass wrote
onto the name as an exponent. An axis with no name, such as a concatenation of two
others, takes its position among the axes it was listed with, and a letter that
occurs twice in one list is primed at its later occurrences, so a map from `m` onto
`m` reads `i_{m}` and `i_{m'}`.
'''
from __future__ import annotations

from collections.abc import Sequence

import data_structure.Category as cat
import data_structure.Term as fd


def axis_letters[A: cat.Axis](axes: Sequence[A]) -> fd.Prod[str]:
    '''The LaTeX letter of each axis of `axes`, in order, distinct within the list.'''
    letters: list[str] = []
    for position, axis in enumerate(axes):
        name = axis.uid._name
        letter = (str(position + 1) if name is None
                  else name.with_exponent(None).to_latex())
        while letter in letters:
            letter += "'"
        letters.append(letter)
    return tuple(letters)


def index_of(letter: str) -> str:
    return f'i_{{{letter}}}'


def read_at(array: str, letters: Sequence[str]) -> str:
    '''`array` read at the index of every axis in `letters`, and `array` alone where
    there is none.'''
    if not letters:
        return array
    return f'{array}[{", ".join(index_of(letter) for letter in letters)}]'


def sum_over(letters: Sequence[str]) -> str:
    '''The summation sign over the index of every axis in `letters`, and nothing
    where there is none.'''
    if not letters:
        return ''
    ranges = r',\, '.join(rf'{index_of(letter)} \in {letter}' for letter in letters)
    return rf'\sum_{{{ranges}}} '


def element_count(letters: Sequence[str]) -> str:
    '''The product of the sizes of the axes in `letters`, `|n||m|`.'''
    return ''.join(f'|{letter}|' for letter in letters)


def subscript_of(letters: Sequence[str]) -> str:
    '''The axes an operator consumes, written as the subscript of its name.'''
    return ', '.join(letters)
