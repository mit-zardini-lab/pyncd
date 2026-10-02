# Claude Opus 5.5 (1M context), effort 40.
'''Giving each index of a formula its own letter, and finding the indices a formula
holds for every position of their axes.

Every formula in the package is written in the notation of
`algebra/write_index_notation.py`, where an index of the axis `m` is `i_{m}`. The
reviewer ruled on 2026-09-27 that an inspection box writes the indices of one formula
with different letters, i, j, k, l, m and n in the order the formula first names them,
each keeping its axis as its subscript: `y[i_{x}] = \\sum_{j_{d} \\in d} ...`. A letter
the formula uses for something else is passed over. The letters passed over are the
symbols written in the formula, such as the array `k` or a bare index `i`, and the
letters in the name of an axis the formula indexes, so that no index reads `m_{m}`.
`LETTERS` continues the sequence past n for a formula with more indices.

The same ruling added a line under a formula naming each index that ranges over every
position of its axis, `\\forall i_{x} \\in x`. A clause of a formula binds an index where
it names the range of the index, as `\\sum_{j_{d} \\in d}` does, or where the index stands
inside a set, `\\{s[j_{d}] : ...\\}`, whose braces range over it. An index a clause uses
without binding it is free in that clause. The clauses are the lines of a `gathered`
block and the parts separated by `\\quad`. An index free in any clause is listed.

A description beside a formula names indices in plain text, as `i_x` or `i_x_new`, and
is written with the letters of its formula. An index of the description takes the
letter of the axis of the formula whose name, up to a `|` and without braces, is the
name the description writes.

A subscript of more than one character written without braces, `x_new`, is braced,
because KaTeX reads `x_new` as `x_n` followed by `ew`. `fd.DynamicName.to_latex` writes
a subscript chain without braces, and tsncd braces the names it draws itself.

`websocket_transfer/auxiliary_information.py` letters the formula and the description of
every block and every operator it packages.
'''
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import data_structure.Term as fd
import websocket_transfer.websockets_transfer as wst

LETTERS = 'ijklmnpqrstuvwabcdefgh'
FIRST_LETTER = LETTERS[0]

INDEX_START = re.compile(r'(?<![A-Za-z\\])i_\{')
UNBRACED_SUBSCRIPT = re.compile(r'(?<!\\)_([A-Za-z0-9]{2,})')
TEXT_GROUP = re.compile(r'\\(?:mathrm|text|operatorname|mathtt|texttt)\{[^{}]*\}')
STANDALONE_LOWERCASE = re.compile(r'(?<![A-Za-z\\])[a-z](?![A-Za-z])')
CLAUSE_SEPARATOR = re.compile(r'\\\\|\\q?quad(?![A-Za-z])')
RANGE_AFTER_INDEX = re.compile(r'\s*\\in(?![A-Za-z])')
SET_BRACE = re.compile(r'\\([{}])')
DESCRIPTION_INDEX = re.compile(r"(?<![A-Za-z\\])i_(\{[^{}]*\}|[A-Za-z0-9'_]+)")
ACCENTED_NAME = re.compile(r'\\[A-Za-z]+\{(.*)\}')


class FormulaHasTooManyIndices(ValueError):
    '''A formula indexes more axes than `LETTERS` has letters its symbols leave free.'''


class UnbalancedBraces(ValueError):
    '''A formula opens a brace it does not close.'''


@dataclass(frozen=True)
class IndexOccurrence:
    '''One index `i_{axis}` of a formula, from the offset of its `i` to the offset
    after its closing brace.'''
    start: int
    end: int
    axis: str


@dataclass(frozen=True)
class LetteredFormula:
    '''A formula with its indices lettered, the letter each axis took, the axes whose
    indices it holds for every position, in the order the formula first names them,
    and the letters no index took and no symbol of the formula uses, in order, for an
    index a range introduces.'''
    formula: str
    letters: fd.Prod[tuple[str, str]]
    free_axes: fd.Prod[str]
    spare_letters: fd.Prod[str] = ()

    def letter_of(self, axis: str) -> str:
        return dict(self.letters)[axis]

    def free_indices(self) -> list[wst.FormulaIndexRecord]:
        return [{'index': f'{self.letter_of(axis)}_{{{axis}}}', 'axis': axis}
                for axis in self.free_axes]


def closing_brace(text: str, opening: int) -> int:
    '''The offset of the brace that closes the one at `opening`.'''
    depth = 0
    for offset in range(opening, len(text)):
        if text[offset] == '{' and text[offset - 1:offset] != '\\':
            depth += 1
        elif text[offset] == '}' and text[offset - 1:offset] != '\\':
            depth -= 1
            if depth == 0:
                return offset
    raise UnbalancedBraces(f'the brace at {opening} of {text!r} is not closed')


def with_braced_subscripts(latex: str) -> str:
    return UNBRACED_SUBSCRIPT.sub(r'_{\1}', latex)


def index_occurrences(formula: str) -> fd.Prod[IndexOccurrence]:
    occurrences = []
    for match in INDEX_START.finditer(formula):
        closing = closing_brace(formula, match.end() - 1)
        occurrences.append(IndexOccurrence(
            start=match.start(), end=closing + 1,
            axis=formula[match.end():closing]))
    return tuple(occurrences)


def standalone_letters(latex: str) -> frozenset[str]:
    '''The lowercase letters `latex` writes as symbols of their own, outside the
    name of a command and outside upright text such as `\\mathrm{i}`.'''
    return frozenset(STANDALONE_LOWERCASE.findall(TEXT_GROUP.sub(' ', latex)))


def letters_taken(formula: str, occurrences: Sequence[IndexOccurrence]) -> frozenset[str]:
    '''The letters an index of `formula` may not take: every symbol of the formula
    outside its indices, and every letter of the name of an axis it indexes.'''
    outside_indices = list(formula)
    for occurrence in occurrences:
        outside_indices[occurrence.start:occurrence.end] = (
            ' ' * (occurrence.end - occurrence.start))
    taken = set(standalone_letters(''.join(outside_indices)))
    for occurrence in occurrences:
        taken |= standalone_letters(occurrence.axis)
    return frozenset(taken)


def axes_in_order_named(occurrences: Sequence[IndexOccurrence]) -> fd.Prod[str]:
    return tuple(dict.fromkeys(occurrence.axis for occurrence in occurrences))


def letters_of_axes(
    axes: Sequence[str], taken: frozenset[str], formula: str,
) -> dict[str, str]:
    free_letters = [letter for letter in LETTERS if letter not in taken]
    if len(free_letters) < len(axes):
        raise FormulaHasTooManyIndices(
            f'{len(axes)} indexed axes and {len(free_letters)} free letters in '
            f'{formula!r}')
    return dict(zip(axes, free_letters))


def clause_spans(formula: str) -> fd.Prod[tuple[int, int]]:
    '''The start and the end of each clause of `formula`.'''
    starts = [0, *(match.end() for match in CLAUSE_SEPARATOR.finditer(formula))]
    ends = [*(match.start() for match in CLAUSE_SEPARATOR.finditer(formula)),
            len(formula)]
    return tuple(zip(starts, ends))


def set_spans(formula: str) -> fd.Prod[tuple[int, int]]:
    '''The start and the end of each set `\\{...\\}` of `formula`, the outermost
    where sets nest.'''
    spans = []
    opened: list[int] = []
    for match in SET_BRACE.finditer(formula):
        if match.group(1) == '{':
            opened.append(match.start())
        elif opened:
            start = opened.pop()
            if not opened:
                spans.append((start, match.end()))
    return tuple(spans)


def binds_its_index(
    formula: str, occurrence: IndexOccurrence, sets: Sequence[tuple[int, int]],
) -> bool:
    '''Whether `occurrence` names the range of its index, `i_{d} \\in d`, or stands
    inside a set, whose braces range over it.'''
    if RANGE_AFTER_INDEX.match(formula, occurrence.end):
        return True
    return any(start < occurrence.start < end for start, end in sets)


def free_axes_of(
    formula: str, occurrences: Sequence[IndexOccurrence],
) -> fd.Prod[str]:
    '''The axes whose index some clause of `formula` uses without binding it, in the
    order the formula first names them.'''
    sets = set_spans(formula)
    free: set[str] = set()
    for start, end in clause_spans(formula):
        in_clause = [occurrence for occurrence in occurrences
                     if start <= occurrence.start < end]
        bound = {occurrence.axis for occurrence in in_clause
                 if binds_its_index(formula, occurrence, sets)}
        free |= {occurrence.axis for occurrence in in_clause
                 if occurrence.axis not in bound}
    return tuple(axis for axis in axes_in_order_named(occurrences) if axis in free)


def rewritten_with_letters(
    formula: str, occurrences: Sequence[IndexOccurrence], letters: Mapping[str, str],
) -> str:
    pieces = []
    written_up_to = 0
    for occurrence in occurrences:
        pieces.append(formula[written_up_to:occurrence.start])
        pieces.append(letters[occurrence.axis])
        written_up_to = occurrence.start + len(FIRST_LETTER)
    pieces.append(formula[written_up_to:])
    return ''.join(pieces)


def lettered_formula(formula: str) -> LetteredFormula:
    '''`formula` with its subscripts braced and each of its indices lettered, with
    the axes whose indices it holds for every position.'''
    braced = with_braced_subscripts(formula)
    occurrences = index_occurrences(braced)
    taken = letters_taken(braced, occurrences)
    letters = letters_of_axes(axes_in_order_named(occurrences), taken, braced)
    return LetteredFormula(
        formula=rewritten_with_letters(braced, occurrences, letters),
        letters=tuple(letters.items()),
        free_axes=free_axes_of(braced, occurrences),
        spare_letters=tuple(letter for letter in LETTERS
                            if letter not in taken and letter not in letters.values()))


def name_a_description_writes(axis: str) -> str:
    '''The plain name a description gives the index of `axis`: the name up to a `|`,
    without the accent over it and without braces, so the index of `w|x` is `i_w`, the
    index of `\\hat{y}` is `i_y` and the index of `x_{new}` is `i_x_new`.'''
    name = axis.split('|')[0]
    accented = ACCENTED_NAME.fullmatch(name)
    return (accented.group(1) if accented else name).replace('{', '').replace('}', '')


def lettered_description(description: str, lettered: LetteredFormula) -> str:
    '''`description` with each index it names written with the letter the formula of
    `lettered` gives the same axis. A name two axes of the formula share is left as
    it stands, because the description does not say which axis it means.'''
    letter_by_name: dict[str, str | None] = {}
    for axis, letter in lettered.letters:
        name = name_a_description_writes(axis)
        letter_by_name[name] = (
            letter if letter_by_name.get(name, letter) == letter else None)

    def letter_index(match: re.Match[str]) -> str:
        written = match.group(1)
        name = written[1:-1] if written.startswith('{') else written
        letter = letter_by_name.get(name)
        return match.group(0) if letter is None else f'{letter}_{written}'

    return DESCRIPTION_INDEX.sub(letter_index, description)
