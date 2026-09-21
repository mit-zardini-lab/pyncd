# Claude Fable 5.1, effort 80.
'''Reading and writing a wording file, the titles and descriptions of a model as JSON.

A wording file holds a `name`, under which a figure's toggle lists it, and `wordings`,
one entry per name. An entry is a string, or a list of parts joined by single spaces.
A part is a string, or an object holding `ref`, the name of another entry, and
`fills`, the values written by `str.format` into the fields of that entry. Inside a
string, `$NAME` stands for the wording of NAME, so a sentence shared by several
descriptions is written once and each description names it. A field written as
`{layer}` is left as it is, and the module that reads the wording fills it at the call.

    {
      "name": "Current",
      "wordings": {
        "CORE_TITLE": "\\\\text{Attention Core}",
        "SOFTMAX_STEPS_SENTENCE": "The query scores against the latent at every slot, ...",
        "CORE_DESCRIPTION": ["One head's softmax over the window slots, ...",
                             "$SOFTMAX_STEPS_SENTENCE"],
        "SWA_BLOCK_DESCRIPTION": "One of the two sliding-window layers: $SLIDING_WINDOW_SUBLAYERS_CLAUSE",
        "ENTRY_ROUND_TRIP_DESCRIPTION": [
          "One compressed entry rounded the way the released cache holds it.",
          {"ref": "CHANNEL_GROUPS_SENTENCE", "fills": {"size": "16"}},
          "..."]
      }
    }

`resolved_wordings` writes every entry out as one string. A file may hold the entries
it changes and nothing else, and resolve against a `base`, the wordings of the file the
model was built with: an entry the file lacks is read from the base, and a reference
inside a base entry is read from the file where the file holds the name, so a changed
sentence reaches every description that names it.
`notebooks/sota/DeepSeekV41Flash/block_titles_and_descriptions.py` loads the file of
that model into a dataclass, and `websocket_transfer/localise_descriptions.py` reads
another file as a localisation of a page.
'''
from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

REFERENCE = re.compile(r'\$([A-Z][A-Z0-9_]*)')
REF_KEY = 'ref'
FILLS_KEY = 'fills'

type WordingPart = str | Mapping[str, object]
type WordingEntry = str | Sequence[WordingPart]


class WordingFileError(ValueError):
    '''A wording file that cannot be read as the docstring of the module states.'''


class WordingReferenceMissing(WordingFileError):
    '''An entry names a wording that neither the file nor its base holds.'''


class WordingReferenceCycle(WordingFileError):
    '''An entry names itself through a chain of references.'''


class WordingNamesDiffer(WordingFileError):
    '''The names of a wording file are not the fields of the dataclass loading it.'''


@dataclass(frozen=True)
class WordingFile:
    name: str
    path: pathlib.Path
    wordings: Mapping[str, WordingEntry]


def read_wording_file(path: str | pathlib.Path) -> WordingFile:
    read = pathlib.Path(path)
    document = json.loads(read.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or not isinstance(document.get('name'), str):
        raise WordingFileError(f'{read} holds no string under "name"')
    wordings = document.get('wordings')
    if not isinstance(wordings, dict):
        raise WordingFileError(f'{read} holds no object under "wordings"')
    for name, entry in wordings.items():
        check_entry(read, name, entry)
    return WordingFile(document['name'], read, wordings)


def check_entry(path: pathlib.Path, name: str, entry: object) -> None:
    if isinstance(entry, str):
        return
    if not isinstance(entry, list):
        raise WordingFileError(
            f'{path}: {name} is {type(entry).__name__}, and an entry is a string or '
            'a list of parts')
    for part in entry:
        if isinstance(part, str):
            continue
        if (not isinstance(part, dict) or not isinstance(part.get(REF_KEY), str)
                or not isinstance(part.get(FILLS_KEY, {}), dict)):
            raise WordingFileError(
                f'{path}: a part of {name} is {part!r}, and a part is a string or an '
                f'object with "{REF_KEY}" and "{FILLS_KEY}"')


def write_wording_file(
    path: str | pathlib.Path, name: str, wordings: Mapping[str, WordingEntry],
) -> pathlib.Path:
    written = pathlib.Path(path)
    written.parent.mkdir(parents=True, exist_ok=True)
    document = {'name': name, 'wordings': dict(wordings)}
    written.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return written


def resolved_wordings(
    file: WordingFile, base: Mapping[str, WordingEntry] | None = None,
) -> dict[str, str]:
    '''Every entry of `file`, and every entry of `base` the file lacks, written out as
    one string, with each reference replaced by the wording it names.'''
    entries: dict[str, WordingEntry] = {**(base or {}), **file.wordings}
    resolved: dict[str, str] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> str:
        if name in resolved:
            return resolved[name]
        if name in chain:
            raise WordingReferenceCycle(
                f'{file.path}: {" -> ".join((*chain, name))}')
        if name not in entries:
            raise WordingReferenceMissing(
                f'{file.path}: {chain[-1] if chain else "the file"} names {name}, '
                'which neither the file nor its base holds')
        entry = entries[name]
        parts = [entry] if isinstance(entry, str) else entry
        written = ' '.join(resolve_part(part, (*chain, name)) for part in parts)
        resolved[name] = written
        return written

    def resolve_part(part: WordingPart, chain: tuple[str, ...]) -> str:
        if isinstance(part, str):
            return REFERENCE.sub(lambda match: resolve(match.group(1), chain), part)
        fills = {field: str(value) for field, value in part.get(FILLS_KEY, {}).items()}
        return resolve(str(part[REF_KEY]), chain).format(**fills)

    for name in entries:
        resolve(name, ())
    return resolved


def load_dataclass[T](cls: type[T], path: str | pathlib.Path) -> T:
    '''The wording file at `path`, resolved, as an instance of `cls`, a frozen dataclass
    with one string field per entry. A file whose names differ from the fields is
    refused, with the names missing and the names over, so a name written in the
    file and nowhere else, or removed from the file and still read, is found at
    import.'''
    wordings = resolved_wordings(read_wording_file(path))
    names = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
    missing = sorted(names - wordings.keys())
    over = sorted(wordings.keys() - names)
    if missing or over:
        raise WordingNamesDiffer(
            f'{path} lacks the entries {missing} and holds the entries {over} that no '
            f'field of {cls.__name__} names')
    return cls(**wordings)


def table_of[T](wording: T) -> dict[str, str]:
    '''Every field of a loaded wording dataclass, by name.'''
    return dict(dataclasses.asdict(wording))  # type: ignore[call-overload]
