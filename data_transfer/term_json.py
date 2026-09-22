from dataclasses import dataclass, field
import data_structure.Term as fd
import json
import enum
from typing import TypedDict
import data_transfer.json_compression as json_compression

TermExportForm = json_compression.TermExportForm

type JSONType = dict[str, JSONType] | list[JSONType] | str | int | float | bool | None

class JSONDataStructure(TypedDict):
    uid_repository: dict[str, JSONType]
    data: JSONType

def json_main(target) -> bool:
    return isinstance(target, (str, int, float, bool)) or target is None

def json_converter(target) -> JSONType:
    if json_main(target):
        return target
    match target:
        case type():
            return {'__registered__': 'type', 'repr': target.__qualname__}
        case enum.Enum():
            return {'__registered__': 'enum', 'type': type(target).__qualname__, 'name': target.name}
    raise ValueError(f"Cannot convert type {type(target)} to JSONType")
    

@dataclass
class TermJSONConverter:
    uid_repository: dict[fd.IDType, JSONType] = field(default_factory=dict)

    def to_json(self, data: fd.GeneralTerm) -> JSONType:
        if isinstance(data, tuple):
            return [self.to_json(member) for member in data]
        
        if not isinstance(data, fd.Term):
            return json_converter(data)

        set_id = None
        if hasattr(data, 'uid'):
            set_id = data.uid._id # type: ignore
            if set_id in self.uid_repository:
                return {'__ref__': set_id}
        
        json_dict = {
            '__type__': type(data).__qualname__,
            **{
                key: self.to_json(value)
                for key, value in data.dict().items()
            }
        }

        if set_id is not None:
            self.uid_repository[set_id] = json_dict
            return {'__ref__': set_id}
        
        return json_dict
    
    def reconstruct(self, data: JSONType) -> fd.GeneralTerm:
        match data:
            case list():
                return tuple(self.reconstruct(member) for member in data)
            case {'__ref__': int(ref_id)}:
                return self.reconstruct(self.uid_repository[ref_id])
            case {'__type__': str(type_name), **fields}:
                return fd.TermDirectory[type_name](**{
                    key: self.reconstruct(value)
                    for key, value in fields.items()
                })
            case {'__registered__': 'type', 'repr': str(type_name)}:
                return fd.TermDirectory[type_name] # type: ignore
            case {'__registered__': 'enum', 'type': str(type_name), 'name': str(member_name)}:
                enum_type = fd.EnumDirectory[type_name]
                return enum_type[member_name] # type: ignore
            case _ if json_main(data):
                return data # type: ignore
            case _:
                raise ValueError(f"Invalid JSON data for reconstruction: {data}")
            
    @classmethod
    def export(
        cls, data: fd.GeneralTerm, target_file: str, indent: None | int = 4,
        *, export_form: TermExportForm = TermExportForm.UID_REFERENCES,
    ) -> None:
        with open(target_file, 'w', encoding='utf-8') as json_file:
            json_file.write(cls.export_to_json(data, indent, export_form=export_form))

    @classmethod
    def export_document(
        cls, data: fd.GeneralTerm,
        export_form: TermExportForm = TermExportForm.UID_REFERENCES,
    ) -> JSONDataStructure | json_compression.CompressedJSON:
        converter = cls()
        json_data = converter.to_json(data)
        json_export: JSONDataStructure = {
            'uid_repository': {
                str(key): value for key, value in converter.uid_repository.items()},
            'data': json_data
        }
        if export_form is TermExportForm.COMPRESSED:
            return json_compression.compress_json(json_export)
        if export_form is TermExportForm.UID_REFERENCES:
            return json_export
        raise ValueError(f'Unsupported term export form {export_form!r}')

    @classmethod
    def export_to_json(
        cls, data: fd.GeneralTerm, indent: None | int = None,
        *, export_form: TermExportForm = TermExportForm.UID_REFERENCES,
    ) -> str:
        document = cls.export_document(data, export_form)
        separators = (',', ':') if (
            export_form is TermExportForm.COMPRESSED and indent is None) else None
        return json.dumps(document, indent=indent, separators=separators)

    @classmethod
    def import_from_json(cls, exported: str | dict) -> fd.GeneralTerm:
        document = json.loads(exported) if isinstance(exported, str) else exported
        if not isinstance(document, dict):
            raise ValueError('A term export must be a JSON object')
        if document.get('export_form') not in (
                None, TermExportForm.UID_REFERENCES.value,
                TermExportForm.COMPRESSED.value):
            raise ValueError(f'Unsupported term export form {document["export_form"]!r}')
        if document.get('export_form') == TermExportForm.COMPRESSED.value:
            document = json_compression.decompress_json(document)
        if not isinstance(document, dict) or not {
                'uid_repository', 'data'} <= document.keys():
            raise ValueError('A term export must contain uid_repository and data')
        converter = cls(uid_repository={
            int(key): value for key, value in document['uid_repository'].items()})
        return converter.reconstruct(document['data'])
