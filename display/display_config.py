import data_structure.Category as cat
import data_structure.Term as fd
import display.Box as Box
import utilities.utilities as util
import term_utilities.term_utilities as tutil
import utilities.justification as js
from typing import Literal, Iterable, Callable, Any

import term_utilities.generate_config as gc
import display.node_category as node_cat

def display_config(
    config_log: gc.ConfigLog[Any],
    size: int = 10
) -> Box.Box:
    terms = config_log.terms
    names = Box.Vertical((
        Box.TextBox('Name'),
        *(
            node_cat.display_uterm(term, size)
            for term in terms
        )
    ))
    types = Box.Vertical((
        Box.TextBox('Type'),
        *(
            Box.TextBox(type(term).__qualname__)
            for term in terms
        )
    ))
    bucket_infos = tuple(
        config_log.get_bucket(term)
        for term in terms
    )
    buckets = Box.Vertical((
        Box.TextBox('Bucket  '),
        *(
            Box.TextBox(
                str(bucket_info[0])
                if bucket_info is not None
                else ''
            )
            for bucket_info in bucket_infos
        )
    ))
    canonicals = Box.Vertical((
        Box.TextBox('Assignment'),
        *(
            Box.TextBox(
                str(bucket_info[1])
                if bucket_info is not None
                else ''
            )
            for bucket_info in bucket_infos
        )
    ))
    gaps = Box.Fill('|', min_width=1)
    return Box.Horizontal((names, gaps, types, gaps, buckets, gaps, canonicals))
    
gc.ConfigLog.__str__ = lambda self: str(display_config(self))