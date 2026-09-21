'''Writing a value above each axis of an expression, by name or from its size.

Written by Claude Fable 5.1, effort 80. `write_axis_size_exponents` written by
Claude Opus 5 (1M context), high effort. `write_assigned_size_exponents` written
by Claude Opus 5, effort high.

An axis is drawn by its name, and a name carries an optional exponent drawn
above its body, per `data_structure/Term.py`. `write_axis_exponents` writes a
value from a mapping as the exponent of every axis whose name reads as a key,
so an expression whose sizes are free symbols is drawn with the sizes a
configuration gives them, `x^{4096}` for the tokens, and the sizes themselves
stay symbols. `write_axis_size_exponents` reads the value off the axis instead,
so an expression a `term_utilities.generate_config.NumericConfig` has already
sized is drawn with those sizes and needs no mapping beside it.
`write_assigned_size_exponents` takes the symbolic expression and the
assignments a configuration made, and writes an exponent onto every axis and
onto every named symbol. A label holding more than one symbol then states each
of them, so the sparse axis `k/e` reads `|k|^{6} \\text{ of } e^{384}`, the count
and the parent each carrying their own size.

Each writer takes a `placement`, which is where the value is drawn. Under
`fd.ExponentPlacement.SUPERSCRIPT`, the default, the value is raised after the
name, `m^{5120}`. Under `SUBSCRIPT` it is lowered into the subscript, `m_{5120}`,
and `notebooks/display/axis_sizes.py` chooses between the two for a figure.

`obsidian/01-foundations/Code Forms.md` states the exponent, and
`obsidian/05-backends/Compound Axis Labels.md` the labels that hold several
symbols.
'''
from __future__ import annotations

from collections.abc import Callable, Mapping

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import term_utilities.term_utilities as term_utilities

type Named = cat.Axis | nm.FreeNumeric


def write_axis_exponents[T](
    term: T, values: Mapping[str, int | str],
    placement: fd.ExponentPlacement = fd.ExponentPlacement.SUPERSCRIPT,
) -> T:
    '''`term` with every axis whose name's bodies are a key of `values` given
    the key's value as its exponent, drawn where `placement` puts it.'''
    def exponent_of(target: Named) -> int | str | None:
        bodies = name_bodies(target)
        if not isinstance(target, cat.Axis) or bodies is None:
            return None
        return values.get(bodies)

    return _write_exponent_of_each_name(term, exponent_of, placement)


def write_axis_size_exponents[T](
    term: T, placement: fd.ExponentPlacement = fd.ExponentPlacement.SUPERSCRIPT,
) -> T:
    '''`term` with every axis whose local size is an integer given that integer
    as its exponent, so a sized expression is drawn `m^{5120}`, or `m_{5120}`
    where `placement` lowers it. An axis whose size still holds a free symbol is
    left as it stands.'''
    def exponent_of(target: Named) -> int | None:
        return evaluated_local_size(target) if isinstance(target, cat.Axis) else None

    return _write_exponent_of_each_name(term, exponent_of, placement)


def write_assigned_size_exponents[T](
    term: T, assigned: Mapping[str, int],
    placement: fd.ExponentPlacement = fd.ExponentPlacement.SUPERSCRIPT,
) -> T:
    '''`term` with the integer each axis's local size comes to under `assigned`
    written as that axis's exponent, and the integer `assigned` gives a named
    symbol written as that symbol's own exponent, each drawn where `placement`
    puts it.

    A configuration assigns by name, and `assigned` is
    `term_utilities.generate_config.NumericConfig.assigned_integers_by_name()`.
    Each factor of a product is rewritten on its own, so a size written `|a||b|`
    with `a` assigned and `b` left symbolic reads `|a|^{2}|b|`.
    '''
    def exponent_of(target: Named) -> int | None:
        if isinstance(target, cat.Axis):
            return evaluated_size_under(target.local_size(), assigned)
        bodies = name_bodies(target)
        return None if bodies is None else assigned.get(bodies)

    return _write_exponent_of_each_name(term, exponent_of, placement)


def name_bodies(target: fd.UIDEquipped) -> str | None:
    '''The bodies of the name on `target`'s uid, or `None` where it has none.'''
    name = target.uid._name
    return None if name is None else name.to_bodies()


def evaluated_local_size(axis: cat.Axis) -> int | None:
    '''The integer an axis's local size comes to, or `None` where the size holds
    a free symbol or is not a sum of products of integers and symbols.

    A size assigned by a `NumericConfig` arrives as the integer it was assigned,
    and a size written as a product of two assigned sizes arrives as the product
    of two integers, because the substitution rebuilds the multiplication rather
    than templating it again. Both come to an integer here.
    '''
    try:
        return nm.evaluate_integer(axis.local_size(), {})
    except (KeyError, nm.NotAnAffineForm):
        return None


def evaluated_size_under(size: nm.Numeric, assigned: Mapping[str, int]) -> int | None:
    '''The integer `size` comes to with every symbol `assigned` names bound to its
    integer, or `None` where a symbol of the size is unbound or the size holds
    something other than integers, symbols, sums and products.'''
    try:
        return nm.evaluate_integer(size, _symbols_bound_by_name(size, assigned))
    except (KeyError, nm.NotAnAffineForm):
        return None


def _symbols_bound_by_name(
    size: nm.Numeric, assigned: Mapping[str, int],
) -> dict[nm.Numeric, int]:
    '''Each symbol of `size` whose name `assigned` binds, keyed by the symbol, in
    the form `nm.evaluate_integer` reads. A configuration assigns by name and the
    evaluation looks a symbol up by the term, so the two are joined here.'''
    return {
        symbol: assigned[bodies]
        for symbol in term_utilities.type_search(nm.FreeNumeric, size)
        if (bodies := name_bodies(symbol)) is not None and bodies in assigned
    }


def _write_exponent_of_each_name[T](
    term: T, exponent_of: Callable[[Named], int | str | None],
    placement: fd.ExponentPlacement,
) -> T:
    '''`term` with the exponent `exponent_of` gives each axis and each named
    symbol written onto that term's name and drawn where `placement` puts it, and
    a term it answers `None` for left as it stands. Each node is rewritten once,
    by identity, so the sharing of the term is kept.'''
    rewritten: dict[int, object] = {}

    def write(target: object) -> object:
        if id(target) in rewritten:
            return rewritten[id(target)]
        rebuilt = fd.deep_reconstruct(target, write)
        if (isinstance(rebuilt, (cat.Axis, nm.FreeNumeric))
                and rebuilt.uid._name is not None):
            exponent = exponent_of(rebuilt)
            if exponent is not None:
                name = rebuilt.uid._name.with_exponent(exponent)
                rebuilt = name.with_exponent_placement(placement).capture(rebuilt)
        rewritten[id(target)] = rebuilt
        return rebuilt

    return write(term)  # type: ignore[return-value]
