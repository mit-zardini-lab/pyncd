'''Reading and writing the JSON files the runner keeps between runs, under
`.cache/validations/`, which git ignores.

Written by Claude Opus 5.5 (1M context), effort 40.

A stored file is a cache, and a run can derive its content again. A file that is
missing, that is not valid JSON, or that holds something other than an object is
read as absent, and the run then derives the content again.

Two runs can finish at the same moment, as when two agents validate at once. A file is
therefore written under a name carrying the process id and then moved onto its final
name, so no run reads a half-written file. On Windows the move fails while another
process holds the final file open, and the file written by the other run is then
kept.
'''

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

STORE_FOLDER = '.cache/validations'

type JsonObject = dict[str, Any]


def stored_file_path(root: pathlib.Path, name: str) -> pathlib.Path:
    return root / STORE_FOLDER / name


def read_stored_json(path: pathlib.Path) -> JsonObject | None:
    if not path.exists():
        return None
    try:
        content = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return content if isinstance(content, dict) else None


def write_stored_json(path: pathlib.Path, content: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f'{path.name}.{os.getpid()}.partial')
    partial.write_text(json.dumps(content, sort_keys=True), encoding='utf-8')
    try:
        os.replace(partial, path)
    except PermissionError:
        partial.unlink(missing_ok=True)
