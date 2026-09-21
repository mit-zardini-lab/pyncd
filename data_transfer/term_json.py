from dataclasses import dataclass, field
import data_structure.Term as fd
import json
import enum
from typing import TypedDict

type JSONType = dict[str, JSONType] | list[JSONType] | str | int | float | bool | None

class JSONDataStructure(TypedDict):
    uid_repository: dict[fd.IDType, JSONType]
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
    def export(cls, data: fd.GeneralTerm, target_file: str, indent: None | int = 4) -> None:
        converter = cls()
        json_data = converter.to_json(data)
        json_export: JSONDataStructure = {
            'uid_repository': converter.uid_repository,
            'data': json_data
        }
        with open(target_file, 'w') as json_file:
            json.dump(json_export, json_file, indent=indent)

    @classmethod
    def export_to_json(cls, data: fd.GeneralTerm, indent: None | int = None) -> str:
        converter = cls()
        json_data = converter.to_json(data)
        json_export: JSONDataStructure = {
            'uid_repository': converter.uid_repository,
            'data': json_data
        }
        return json.dumps(json_export, indent=indent)
