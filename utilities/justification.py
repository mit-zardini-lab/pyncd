'''Padding a sequence out to a fixed length.

`display` lays a diagram out by column, so every cell in a column has to occupy the
same number of screen positions. `justify` works on a sequence of items and pads it
with copies of one `buffer` item. `justify_str` is the string case, and corrects for
the ANSI colour escapes a coloured cell carries, which occupy no screen position.
'''
from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload, Sequence
from enum import Enum
import utilities.utilities as util
import display.Color as Color

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
) -> Iterator[T]:
    '''`target` padded with `buffer` items until it holds `length` items.

    `length` counts every item, including the separators. A `length` of None yields
    `target` unpadded, and a `length` shorter than `target` truncates from the right.
    `SPREAD` on a single item falls through to `CENTER`, because there is no gap to
    distribute the padding into.
    '''
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
) -> Iterator[T]:
    '''`target` with the padding shared out between its gaps, to fill `length` items.

    Every gap receives the same number of `buffer` items. The remainder is divided
    between the first gap and the last. `target` must hold at least two items, because
    a shorter sequence has no gap to divide by.
    '''
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
) -> str:
    '''The strings in `target` joined into one line `length` screen positions wide.

    `Color.original` strips the ANSI escapes before measuring, so a coloured string
    is padded to the width a reader sees rather than to the width of its bytes. Each
    `buffer` must be one screen position wide, because the width is converted into a
    count of items for `justify`. The conversion counts the strings in `target` alone,
    so passing a `separator` widens the result past `length`.
    '''
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