# Claude Fable 5.1, effort 80.
'''Numbering the `Broadcasted` nodes of a term in the order tsncd's importer builds them.

A `cat.Broadcasted` carries no uid, so nothing sent beside a term can name one of them
by identity. `TermJSONConverter.to_json` writes a term with a uid once, into the
`uid_repository`, and every later occurrence as a reference to it, and writes a term
without a uid in place at every occurrence. tsncd's `TermJSONConverter.to_term` walks
that document depth first, in the order the fields were written, follows each
reference into the repository the first time it meets it and returns the term it built
on every later one. The `Broadcasted` nodes it constructs therefore stand in one
definite order, and `number_broadcasts_in_import_order` reproduces that order from the
Python term, so that information sent beside the term can be keyed by the number.

The walk here follows the same rules as the export. A term with a uid is descended
into once, keyed by its uid, and a term without one is descended into at every
occurrence, which is how the JSON holds it. The fields are read in the order
`Term.dict` gives them, which is the order the JSON carries them and the order
`to_term` converts them, and a `Broadcasted` is numbered on entry, before its fields
are read, which is when the importer counts it.

`websocket_transfer/auxiliary_information.py` keys the expansion of each operator by
this number, and tsncd's `src/data_transfer/json.ts` records the same number on each
`Broadcasted` it constructs. `obsidian/05-backends/Advanced Display.md` states the
scheme.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Term as fd


def number_broadcasts_in_import_order(
    target: fd.GeneralTerm,
) -> tuple[cat.Broadcasted, ...]:
    '''Every `cat.Broadcasted` occurrence of `target`, in the order tsncd's importer
    constructs them, so that the occurrence at index `k` is the one tsncd numbers `k`.
    A `Broadcasted` reached along two paths that share no term with a uid appears
    twice, as it stands twice in the exported JSON.'''
    numbered: list[cat.Broadcasted] = []
    visited_uids: set[fd.IDType] = set()

    def walk(value: object) -> None:
        if isinstance(value, tuple):
            for member in value:
                walk(member)
            return
        if not isinstance(value, fd.Term):
            return
        uid = getattr(value, 'uid', None)
        if uid is not None:
            if uid._id in visited_uids:
                return
            visited_uids.add(uid._id)
        if isinstance(value, cat.Broadcasted):
            numbered.append(value)
        for field in value.dict().values():
            walk(field)

    walk(target)
    return tuple(numbered)
