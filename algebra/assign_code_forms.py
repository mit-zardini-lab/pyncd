'''Giving the axes of an expression their code forms by name.

Written by Claude Fable 5.1, effort 80.

An axis built by `cat.RawAxis.named('q')` or minted by `ops.Einops.template`
carries the letter it is drawn with and no code form. `assign_axis_code_forms`
gives every axis whose name reads as a key of a mapping the code form the mapping
gives it, and gives the axis's size the same code form with `_size` appended, so an
expression built from letters compiles to code in which every axis and every size
carries the name the mapping gave it. The rewrite is by name, which is a decoration, and
leaves every uid as it is, so two axes drawn `q` both read `queries` in code and
stay two axes. `axis_code_forms` reads the result back.

`obsidian/01-foundations/Code Forms.md` states the code form.
'''
from __future__ import annotations

from collections.abc import Mapping

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import term_utilities.term_utilities as tutil


def with_axis_code_form(axis: cat.Axis, code_form: str) -> cat.Axis:
    '''`axis` carrying `code_form`, its size carrying `code_form` with `_size`
    appended where the size is a named symbol.'''
    name = axis.uid._name
    if name is None:
        return axis
    renamed = name.with_code_form(code_form).capture(axis)
    size = renamed._size
    if isinstance(size, nm.FreeNumeric) and size.uid._name is not None:
        renamed = renamed.reconstruct(
            _size=size.uid._name.with_code_form(
                fd.join_code_forms(code_form, 'size')).capture(size))
    return renamed


def assign_axis_code_forms[T](term: T, code_forms: Mapping[str, str]) -> T:
    '''`term` with every axis whose name's bodies are a key of `code_forms` and
    which carries no code form given the key's value, and its size the value
    with `_size` appended. Each node is rewritten once, by identity, so the
    sharing of the term is kept.'''
    rewritten: dict[int, object] = {}

    def assign(target: object) -> object:
        if id(target) in rewritten:
            return rewritten[id(target)]
        rebuilt = fd.deep_reconstruct(target, assign)
        if isinstance(rebuilt, cat.Axis):
            name = rebuilt.uid._name
            if name is not None and name.code_form is None:
                code_form = code_forms.get(name.to_bodies())
                if code_form is not None:
                    rebuilt = with_axis_code_form(rebuilt, code_form)
        rewritten[id(target)] = rebuilt
        return rebuilt

    return assign(term)  # type: ignore[return-value]


def axis_code_forms(term: object) -> dict[str, str | None]:
    '''The code form every named axis of `term` carries, keyed by the axis's
    name as `to_bodies` reads it.'''
    forms: dict[str, str | None] = {}
    for axis in tutil.type_search(cat.Axis, term):
        name = axis.uid._name
        if name is not None:
            forms.setdefault(name.to_bodies(), name.code_form)
    return forms


def axes_without_code_form(term: object) -> list[str]:
    '''The names of the axes of `term` that carry no code form, each once.'''
    return sorted(name for name, code_form in axis_code_forms(term).items()
                  if code_form is None)
