from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload, Sequence
from enum import Enum
import utilities.utilities as util

class JustifyMode(Enum):
    LEFT = 'left'     # 'Hello      '
    RIGHT = 'right'   # '      Hello'
    CENTER = 'center' # '   Hello   '
    SPREAD = 'spread' # 'H  e l l  o'


def justify[T](
    target: Iterable[T],
    buffer: T,
    length: int | None = None,
    mode: JustifyMode = JustifyMode.LEFT,
    separator: T | None = None
):
    target = tuple(target)
    seperated_form = tuple(util.join_with_none(target, separator))
    if length is None:
        yield from seperated_form
        return
    effective_length = len(seperated_form)
    if effective_length >= length:
        yield from seperated_form[:length]
        return
    extra_space = length - effective_length
    left_space = 0
    right_space = 0
    match mode:
        case JustifyMode.SPREAD if len(seperated_form) > 1:
            yield from spread(seperated_form, buffer, length)
            return
        case JustifyMode.LEFT:
            right_space = extra_space
        case JustifyMode.RIGHT:
            left_space = extra_space
        case _:
            left_space = extra_space // 2
            right_space = extra_space - left_space
    yield from (buffer for _ in range(left_space))
    yield from seperated_form
    yield from (buffer for _ in range(right_space))

def spread[T](
    target: Sequence[T],
    buffer: T,
    length: int,
):
    extra_space = length - len(target)
    extra_space = length - len(target)
    number_gaps = len(target) - 1
    gap_size = extra_space // number_gaps
    left_gap = (extra_space % number_gaps) // 2
    right_gap = extra_space - (gap_size * number_gaps) - left_gap
    yield target[0]
    yield from (buffer for _ in range(left_gap + gap_size))
    for item in target[1:-1]:
        yield item
        yield from (buffer for _ in range(gap_size))
    yield from (buffer for _ in range(right_gap))
    yield target[-1]

def justify_str(
    target: Sequence[str],
    length: int | None = None,
    buffer: str = ' ',
    mode: JustifyMode = JustifyMode.LEFT,
    separator: str | None = None
):
    # Imported here rather than at module scope: display/__init__ imports
    # node_category, which imports this module, so a top-level import would
    # be circular.
    import display.Color as Color
    if length is not None:
        total_length = sum(len(Color.original(s)) for s in target)
        desired_buffers = length - total_length
        length = desired_buffers + len(target)
    return ''.join(
        justify(
            target,
            buffer,
            length,
            mode,
            separator
        )
    )