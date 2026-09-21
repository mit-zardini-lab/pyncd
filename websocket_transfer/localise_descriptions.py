# Claude Fable 5.1, effort 80.
'''Localising the descriptions of a figure for the toggle on its standalone page.

A localisation is a table of wordings under the names of a wording file such as
`notebooks/sota/DeepSeekV41Flash/block_titles_and_descriptions.json`, which
`utilities/wording_json.py` reads, and `Localisation.from_wording_file` resolves a
file against the file the model was built with, so a second file holds the entries it
changes and nothing else. A page that carries two or more localisations lets the reader
switch the wording of every inspection box. The page was exported with one table, and
its `auxiliary` field holds the descriptions written from that table, with the values
filled in where a module filled a template with a layer number or a size.
`ExportedWording.localised` reads a description back as the entries that composed it
and writes it again from another table, and `embedded_localisations` collects the
descriptions that each table changes, which `websocket_transfer/standalone_page.py`
writes into the page beside the message.

Only a description is localised. A block title sets the width and the height of its
block on the page, so a page whose titles changed under the reader would be laid out
again, and the user asked on 2026-09-20 that the toggle never changes the spacing. A
formula is written from the axes of its morphism and is not prose. Prose that no
constant of the table holds, such as the role of a weight written in
`operator_explanations.py`, is kept as it was exported, and the constants around it are
still localised.

`obsidian/05-backends/Diagram Wire Format.md` states the embedded form.
'''
from __future__ import annotations

import pathlib
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import utilities.wording_json as wording_json
import websocket_transfer.websockets_transfer as wst

SENTENCE_END = '. '


class LocalisationFieldsDiffer(ValueError):
    '''A localisation writes a template with a field the exported table does not fill.'''


class DescriptionDoesNotRoundTrip(ValueError):
    '''A description read back as constants and written again from the same table
    came out differently.'''


@dataclass(frozen=True)
class Localisation:
    '''The wordings of one localisation, resolved, under the names of the wording file
    the construction modules of a model read.'''
    name: str
    table: Mapping[str, str]

    @classmethod
    def from_wording_file(
        cls, path: str | pathlib.Path,
        base: str | pathlib.Path | Sequence[str | pathlib.Path] = (),
    ) -> Localisation:
        '''The wording file at `path`, resolved against the files of `base`, which are
        the files the model and the packages drawing it were built with, so that an
        entry `path` lacks is read from them and a sentence `path` changes reaches
        every description naming it. A later base file overrides an earlier one.'''
        base_paths = [base] if isinstance(base, (str, pathlib.Path)) else list(base)
        base_wordings: dict[str, wording_json.WordingEntry] = {}
        for base_path in base_paths:
            base_wordings.update(wording_json.read_wording_file(base_path).wordings)
        file = wording_json.read_wording_file(path)
        return cls(file.name, wording_json.resolved_wordings(file, base_wordings or None))


type ParsedWording = tuple[tuple[str, str | None, str | None, str | None], ...]


def parsed_template(wording: str) -> ParsedWording | None:
    '''`wording` as `string.Formatter` parses it when every field is named by an
    identifier, and `None` when it is not a template, as a LaTeX title such as
    `\\text{Attention Core}` is not.'''
    try:
        parsed = tuple(string.Formatter().parse(wording))
    except ValueError:
        return None
    if all(field is None or field.isidentifier() for _, field, _, _ in parsed):
        return parsed
    return None


def template_fields(wording: str) -> tuple[str, ...]:
    '''The names of the fields `str.format` fills in `wording`, each once, in order.'''
    fields = [field for _, field, _, _ in parsed_template(wording) or ()
              if field is not None]
    return tuple(dict.fromkeys(fields))


def wording_pattern(wording: str) -> re.Pattern[str]:
    '''`wording` as a pattern matching the text `str.format` writes from it. A field
    matches the shortest run, empty included, that lets the literal text after it
    match, a field with no literal text after it matches to the end, and a field
    repeated in the wording matches the run its first occurrence matched. A wording
    that is not a template matches its own text.'''
    parsed = parsed_template(wording)
    if parsed is None:
        return re.compile(re.escape(wording))
    parts: list[str] = []
    seen: set[str] = set()
    for index, (literal, field, _, _) in enumerate(parsed):
        parts.append(re.escape(literal))
        if field is None:
            continue
        if field in seen:
            parts.append(f'(?P={field})')
            continue
        seen.add(field)
        is_trailing = index == len(parsed) - 1
        parts.append(f'(?P<{field}>.*)' if is_trailing else f'(?P<{field}>.*?)')
    return re.compile(''.join(parts))


@dataclass(frozen=True)
class TableEntry:
    name: str
    wording: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class DescriptionSegment:
    '''A run of a description: the text of one constant of the exported table with
    the values filled into its fields, or a run no constant matches, held as `text`
    with no `name`.'''
    text: str
    name: str | None = None
    fills: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ExportedWording:
    '''The table a page was exported with, and its constants as patterns, longest
    wording first, so that a description is read back as the largest constants that
    compose it.'''
    table: Mapping[str, str]
    entries: tuple[TableEntry, ...]

    @classmethod
    def of(cls, table: Mapping[str, str]) -> ExportedWording:
        entries = sorted(
            (TableEntry(name, wording, wording_pattern(wording))
             for name, wording in table.items()),
            key=lambda entry: (-len(entry.wording), entry.name))
        return cls(table, tuple(entries))

    def segments(
        self, description: str, start: int = 0,
        excluding: frozenset[str] = frozenset(),
    ) -> list[DescriptionSegment]:
        '''`description` from `start` as the constants that compose it, separated by
        single spaces, with any run no constant matches held up to the end of its
        sentence. A constant named in `excluding` is not matched, so that a
        description is read as the constants joined into it.'''
        if start == len(description):
            return []
        for entry in self.entries:
            if entry.name in excluding:
                continue
            match = entry.pattern.match(description, start)
            if match is None:
                continue
            found = DescriptionSegment(
                match.group(0), entry.name, tuple(match.groupdict().items()))
            if match.end() == len(description):
                return [found]
            if description[match.end()] == ' ':
                return [found, *self.segments(
                    description, match.end() + 1, excluding)]
        stop = description.find(SENTENCE_END, start)
        if stop == -1:
            return [DescriptionSegment(description[start:])]
        return [DescriptionSegment(description[start:stop + 1]),
                *self.segments(description, stop + len(SENTENCE_END), excluding)]

    def localised(self, description: str, alternative: Mapping[str, str]) -> str:
        '''`description` written again from `alternative`, with the wording of the
        exported table for a constant `alternative` lacks and for the runs no
        constant matches.'''
        segments = self.segments(description)
        written_back = self.written(segments, self.table)
        if written_back != description:
            raise DescriptionDoesNotRoundTrip(
                f'{description!r} read back as {[s.name for s in segments]} and '
                f'written again is {written_back!r}')
        return self.written(segments, alternative)

    def written(self, segments: Sequence[DescriptionSegment],
                alternative: Mapping[str, str]) -> str:
        return ' '.join(self.written_segment(segment, alternative)
                        for segment in segments)

    def written_segment(
        self, segment: DescriptionSegment, alternative: Mapping[str, str],
        excluding: frozenset[str] = frozenset(),
    ) -> str:
        '''`segment` written from `alternative`. A constant `alternative` lacks is
        read again as the constants joined into it, so that a table holding the
        sentences of a description and not the description localises it.'''
        if segment.name is None:
            return segment.text
        if segment.name not in alternative:
            return self.written_from_parts(segment, alternative, excluding)
        fills = {field: self.localised(value, alternative)
                 for field, value in segment.fills}
        wording = alternative[segment.name]
        try:
            return wording.format(**fills)
        except (KeyError, IndexError) as error:
            raise LocalisationFieldsDiffer(
                f'{segment.name} has the fields {template_fields(wording)} in a '
                f'localisation and the exported table fills {tuple(fills)}') from error

    def written_from_parts(
        self, segment: DescriptionSegment, alternative: Mapping[str, str],
        excluding: frozenset[str],
    ) -> str:
        narrower = excluding | {segment.name}
        parts = self.segments(segment.text, excluding=narrower)
        return ' '.join(self.written_segment(part, alternative, narrower)
                        for part in parts)


def localised_descriptions(
    auxiliary: wst.DiagramAuxiliary,
    exported: ExportedWording,
    alternative: Mapping[str, str],
) -> wst.LocalisedDescriptions:
    '''The descriptions of `auxiliary` that `alternative` changes, keyed as
    `auxiliary` keys them, with the nested auxiliary of an expansion localised in
    turn.'''
    blocks: dict[str, str] = {}
    for key, information in auxiliary.get('blocks', {}).items():
        description = information['description']
        if description is None:
            continue
        localised = exported.localised(description, alternative)
        if localised != description:
            blocks[key] = localised
    expansions: dict[str, wst.LocalisedExpansion] = {}
    for key, expansion in auxiliary.get('expansions', {}).items():
        changed: wst.LocalisedExpansion = {}
        localised = exported.localised(expansion['description'], alternative)
        if localised != expansion['description']:
            changed['description'] = localised
        nested = localised_descriptions(
            expansion.get('auxiliary', {}), exported, alternative)
        if nested['blocks'] or nested['expansions']:
            changed['auxiliary'] = nested
        if changed:
            expansions[key] = changed
    return {'blocks': blocks, 'expansions': expansions}


def unattributed_descriptions(
    auxiliary: wst.DiagramAuxiliary, exported: ExportedWording,
) -> list[str]:
    '''Every description of `auxiliary`, and of the nested auxiliaries of its
    expansions, in which no constant of `exported` matches, so that a localisation
    leaves it as it is.'''
    found: list[str] = []
    descriptions = [
        *(information['description'] for information in
          auxiliary.get('blocks', {}).values()),
        *(expansion['description'] for expansion in
          auxiliary.get('expansions', {}).values())]
    for description in descriptions:
        if description is None:
            continue
        if all(segment.name is None for segment in exported.segments(description)):
            found.append(description)
    for expansion in auxiliary.get('expansions', {}).values():
        found.extend(unattributed_descriptions(
            expansion.get('auxiliary', {}), exported))
    return found


def embedded_localisations(
    auxiliary: wst.DiagramAuxiliary,
    localisations: Sequence[Localisation],
) -> wst.EmbeddedLocalisations:
    '''What the page carries for its toggle: the name of the first localisation,
    which is the table `auxiliary` was written from, and for every localisation
    the descriptions it changes. The first changes none. A description written
    from a table other than the first is read back as a run no constant matches,
    and `unattributed_descriptions` lists such runs.'''
    if len(localisations) < 2:
        raise ValueError(
            f'{len(localisations)} localisations, and a toggle needs two or more')
    exported = ExportedWording.of(localisations[0].table)
    return {
        'default': localisations[0].name,
        'localisations': {
            localisation.name:
                localised_descriptions(auxiliary, exported, localisation.table)
            for localisation in localisations}}
