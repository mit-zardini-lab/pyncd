import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import display.Box as Box
import utilities.utilities as util
import term_utilities.term_utilities as tutil
import utilities.justification as js
from typing import Literal, Iterable, Callable

import display.Color as cl

def display_uterm(
    target: fd.UTerm,
    size: int = 2
) -> Box.Box:
    return Box.TextBox(str_uterm(target, size))

def str_uterm(
    target: fd.UTerm,
    size: int = 2
) -> str:
    text = ''
    if target.uid._name is not None:
        text = f'{target.uid._name.to_bodies()}'
    else:
        text = f'  {target.uid._id:X}'[-size:]
    return text

def display_numeric(
    target: nm.Numeric | nm.Equality,
):
    match target:
        case nm.Equality():
            return Box.Horizontal((
                display_numeric(target.left),
                Box.TextBox(' = '),
                display_numeric(target.right)
            ))
        case nm.FreeNumeric():
            return display_uterm(target, 4) # type: ignore
        case nm.Integer():
            return Box.TextBox(str(target._value))
        case nm.Associative():
            return Box.Horizontal((
                Box.TextBox('('),
                *util.join_with_none(
                    (display_numeric(elem) for elem in target.content),
                    Box.TextBox(target.sep)
                ),
                Box.TextBox(')')
            )
            )
        case nm.Power():
            return Box.Horizontal((
                display_numeric(target.base), 
                Box.TextBox('^'), 
                display_numeric(target.exponent)))
    raise ValueError(f'Unknown Numeric type: {target}')