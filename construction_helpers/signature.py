from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import utilities.utilities as util
from enum import Enum

type strSignature = str

class SignatureMode(Enum):
    SHORT = 'SHORT'
    LONG = 'LONG'

def double_signature(
    target: str,
    mode: SignatureMode = SignatureMode.LONG
) -> list[list[str]]:
    match mode:
        case SignatureMode.SHORT:
            return [
                [char for char in segment.strip()]
                for segment in target.strip().split(',')
            ]
        case SignatureMode.LONG:
            return [
                [name.strip() for name in segment.strip().split(' ')
                 if name.strip() != '']
                for segment in target.strip().split(',')
            ]
        
type SignatureSegment = fd.Prod[fd.Prod[int]]

def generic_signature[B: cat.Datatype = cat.Reals](
    signature: str,
    datatype: B = cat.Reals(),
    mode: SignatureMode = SignatureMode.LONG
) -> tuple[
    SignatureSegment, 
    fd.Prod[cat.Weave[B, cat.RawAxis]], 
    fd.Prod[cat.Weave[B, cat.RawAxis]], 
    fd.Prod[cat.Rearrangement[cat.RawAxis]],
    cat.ProdObject[cat.RawAxis]]:
    '''
    Symbols that appear on the right are broadcasted - degree. One that
    appears on the right only is *repeated along*: it is still degree, and the
    reindexings simply do not name it, which is what a broadcast is.
    Symbols that appear only on the left are absorbed.
    
    :param signature: Description
    :type signature: str
    :param datatype: Description
    :type datatype: B
    '''
    input_portion, output_portion = signature.split('->')

    input_segments = double_signature(
        input_portion, mode)
    output_segments = double_signature(
        output_portion, mode)
    
    input_names = {name 
                  for segment in input_segments 
                  for name in segment}
    output_names = {name 
                   for segment in output_segments 
                   for name in segment}
    
    axes_names = input_names | output_names
    broadcasted_names = output_names
    absorbed_names = input_names - broadcasted_names

    degree_names = util.iallequals(
        tuple(name for name in segment
              if name in broadcasted_names)
        for segment in output_segments
    )

    axes = {
        name: cat.RawAxis.named(name) for name in axes_names
    }

    input_weaves = tuple(
        cat.Weave(
            datatype,
            tuple(
                axes[name] if name in absorbed_names
                else cat.WeaveMode.TILED
                for name in segment
            )
        )
        for segment in input_segments
    )

    output_weaves = tuple(
        cat.Weave(
            datatype,
            (cat.WeaveMode.TILED,) * len(segment)
        )
        for segment in output_segments
    )

    degree_list = list(degree_names)
    degree = tuple(
        axes[name] for name in degree_list
    )
    reindexings = tuple(
        cat.Rearrangement(
            tuple(degree_list.index(name)
                  for name in segment 
                  if name in broadcasted_names),
            degree       
        )
        for segment in input_segments
    )

    absorbed_list = list(absorbed_names)
    segment_signature = tuple(
        tuple(
            absorbed_list.index(name)
            for name in segment
            if name in absorbed_names
        )
        for segment in input_segments
    )

    # The degree is handed back separately: with no inputs there is no
    # reindexing to carry it, and the caller puts it in `backup_degree`.
    return segment_signature, input_weaves, output_weaves, reindexings, cat.ProdObject(degree)