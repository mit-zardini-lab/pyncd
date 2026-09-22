# GPT-6 Astra, high reasoning effort.
'''Hashing JSON values into a repository with compact integer references.

Object field order is preserved because the TypeScript term importer constructs
dataclasses positionally. Decoding shares JSON containers, while term construction
still creates each non-UID occurrence separately. The format is documented in
`obsidian/05-backends/Diagram Wire Format.md`.
'''

from dataclasses import dataclass, field
import enum
import hashlib
import json
import math
from typing import Literal, TypedDict


type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]


class TermExportForm(enum.Enum):
    UID_REFERENCES = 'uid_references'
    COMPRESSED = 'compressed'


class NodeKind(enum.IntEnum):
    SCALAR = 0
    ARRAY = 1
    OBJECT = 2


class CompressedJSON(TypedDict):
    export_form: Literal['compressed']
    version: Literal[1]
    value_repository: list[list[JSONValue]]
    data: int


def record_digest(encoded: bytes) -> bytes:
    return hashlib.sha256(encoded).digest()


def is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


@dataclass
class JSONCompressor:
    records: list[list[JSONValue]] = field(default_factory=list)
    buckets: dict[bytes, list[tuple[bytes, int]]] = field(default_factory=dict)
    containers: dict[int, int] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)

    def reference(self, value: JSONValue) -> int:
        if is_scalar(value):
            return self.intern([NodeKind.SCALAR.value, value])
        if not isinstance(value, (list, dict)):
            raise TypeError(f'{value!r} is not a JSON value')
        identity = id(value)
        if identity in self.active:
            raise ValueError('A JSON container refers to itself')
        if identity in self.containers:
            return self.containers[identity]
        self.active.add(identity)
        try:
            record = self.container_record(value)
            reference = self.intern(record)
            self.containers[identity] = reference
            return reference
        finally:
            self.active.remove(identity)

    def container_record(
        self, value: list[JSONValue] | dict[str, JSONValue],
    ) -> list[JSONValue]:
        if isinstance(value, list):
            return [NodeKind.ARRAY.value, [self.reference(item) for item in value]]
        fields: list[JSONValue] = []
        for name, item in value.items():
            if not isinstance(name, str):
                raise TypeError(f'JSON object key {name!r} is not a string')
            fields.extend((self.reference(name), self.reference(item)))
        return [NodeKind.OBJECT.value, fields]

    def intern(self, record: list[JSONValue]) -> int:
        encoded = json.dumps(
            record, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode('utf-8')
        bucket = self.buckets.setdefault(record_digest(encoded), [])
        for previous, reference in bucket:
            if previous == encoded:
                return reference
        reference = len(self.records)
        self.records.append(record)
        bucket.append((encoded, reference))
        return reference


def compress_json(value: JSONValue) -> CompressedJSON:
    compressor = JSONCompressor()
    reference = compressor.reference(value)
    return {
        'export_form': TermExportForm.COMPRESSED.value,
        'version': 1,
        'value_repository': compressor.records,
        'data': reference,
    }


def referenced_value(reference: object, values: list[JSONValue]) -> JSONValue:
    if (not isinstance(reference, int) or isinstance(reference, bool)
            or reference < 0 or reference >= len(values)):
        raise ValueError(
            f'Invalid JSON reference {reference!r} for {len(values)} values')
    return values[reference]


def decode_record(record: object, values: list[JSONValue]) -> JSONValue:
    if not isinstance(record, list) or len(record) != 2:
        raise ValueError(f'Invalid compressed JSON record {record!r}')
    kind, payload = record
    if not isinstance(kind, int) or isinstance(kind, bool):
        raise ValueError(f'Invalid compressed JSON node kind {kind!r}')
    if kind == NodeKind.SCALAR:
        if not is_scalar(payload) or (
                isinstance(payload, float) and not math.isfinite(payload)):
            raise ValueError(f'Invalid JSON scalar {payload!r}')
        return payload
    if not isinstance(payload, list):
        raise ValueError(f'JSON references must be a list: {payload!r}')
    if kind == NodeKind.ARRAY:
        return [referenced_value(reference, values) for reference in payload]
    if kind == NodeKind.OBJECT and len(payload) % 2 == 0:
        result: dict[str, JSONValue] = {}
        for index in range(0, len(payload), 2):
            name = referenced_value(payload[index], values)
            if not isinstance(name, str) or name in result:
                raise ValueError(f'Invalid or repeated JSON field name {name!r}')
            result[name] = referenced_value(payload[index + 1], values)
        return result
    raise ValueError(f'Invalid compressed JSON node kind or fields: {record!r}')


def decompress_json(document: object) -> JSONValue:
    '''Decode references to earlier records, rejecting missing or cyclic references.

Repeated containers share storage in the result. Callers must treat the decoded
document as read-only, just as they treat the compressed repository.
'''
    if (not isinstance(document, dict)
            or document.get('export_form') != TermExportForm.COMPRESSED.value
            or isinstance(document.get('version'), bool)
            or document.get('version') != 1
            or not isinstance(document.get('value_repository'), list)):
        raise ValueError('Invalid compressed JSON envelope or unsupported version')
    values: list[JSONValue] = []
    for record in document['value_repository']:
        values.append(decode_record(record, values))
    return referenced_value(document.get('data'), values)
