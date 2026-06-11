import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import display.Box as Box
from data_structure.TensorDSL import Reindex
import utilities.utilities as util
import term_utilities.term_utilities as tutil
import utilities.justification as js
from typing import Literal, Iterable, Callable

'''
Basic Infrastructure - straight rendering.

'''
def display_uterm(
    target: fd.UTerm
) -> Box.Box:
    text = ''
    if target.uid._name is not None:
        text = f'{target.uid._name.to_bodies()}  '[:2]
    else:
        text = f'  {target.uid._id:X}'[-2:]
    return Box.TextBox(text)

def display_axis(
    axis: cat.Axis
) -> Box.Box:
    if isinstance(axis._size, nm.Integer):
        return Box.TextBox(f'{axis._size._value}')
    return display_uterm(axis)

def display_long_axis(
    axis: cat.Axis,
    *sides: Literal['left', 'right'],
    extension: str = '─'
):
    return Box.Horizontal((
        *((Box.Fill(extension, min_width=2, min_height=1),) if 'left' in sides else ()),
        display_axis(axis),
        *((Box.Fill(extension, min_width=2, min_height=1),) if 'right' in sides else ()),
    ))
    
def datatype(
    target: cat.Datatype
) -> Box.Box:
    return Box.TextBox(type(target).__qualname__[:2])

def display_array[B: cat.Datatype, A: cat.Axis](
    target: cat.Array[B, A] | cat.Weave[B, A],
    display_axis: Callable[
        [A | cat.WeaveMode], 
        Box.Box],
    display_datatype: Callable[[B], Box.Box] | None = None,
) -> Box.Box:
    display_datatype = display_datatype or datatype
    return Box.Vertical((
        *(
            display_axis(axis)
            for axis in target._shape
        ),
        display_datatype(target.datatype)
    ))

def reindexed_weave[B:cat.Datatype, A:cat.Axis](
        weave: cat.Weave[B, A],
        reindexing: cat.StrideCategory[A]
) -> Box.Box:
    if tutil.is_mappable(reindexing):
        return reindexed_weave_mappable(weave, reindexing) # type: ignore
    return reindexed_weave_std(weave, reindexing)

def reindexed_weave_std[B:cat.Datatype, A:cat.Axis](
        weave: cat.Weave[B, A],
        reindexing: cat.StrideCategory[A]
) -> Box.Box:
    iterate_weave = iter(reindexing.cod())

    left = Box.Vertical((
        *(
            Box.Horizontal((
                display_axis(axis),
                Box.Fill('~', 2, 1)
            )) if isinstance(axis, cat.Axis) else
            Box.Horizontal((
                display_axis(next(iterate_weave)),
                Box.Fill('─', 2, 1),
            ))
            for axis in weave._shape
        ),
        datatype(weave.datatype)
    ))
    # TODO: improve this middle part
    name = f'{reindexing.name}   '[:3] if isinstance(reindexing, cat.StrideMorphism) else 'eta'
    middle = Box.TextBox(name)
    return Box.Horizontal((left, middle))

def reindexed_weave_mappable[B:cat.Datatype, A:cat.Axis](
    weave: cat.Weave[B, A],
    reindexing: tutil.Mappable[A]
):
    iterate_weave = iter(reindexing.cod())
    iterate_mapping = iter(tutil.get_mapping(reindexing))

    left = Box.Vertical((
        *(
            display_axis(
                axis if isinstance(axis, cat.Axis)
                else next(iterate_weave)
            )
            for axis in weave._shape
        ),
        datatype(weave.datatype)
    ))
    middle = Box.Vertical.from_iter(
        Box.Fill('~', 2, 1) if isinstance(axis, cat.Axis)
        else
        Box.Horizontal((
            Box.Fill('─', 1, 1),
            Box.TextBox(f'<{next(iterate_mapping)}'),
        )) for axis in weave._shape
    )
    return Box.Horizontal((left, middle))

def display_degree[A: cat.Axis](
        degree: cat.ProdObject[A]
) -> Box.Box:
    length = len(degree)
    return Box.Vertical.from_iter(
        Box.Horizontal((
            Box.TextBox('(' if idx == 0 else ' '),
            display_axis(axis),
            Box.TextBox(')' if idx == length - 1 else ' '),
        ),
        justify_mode=js.JustifyMode.CENTER)
        for idx, axis in enumerate(degree)
    )

def display_core(
        operator: cat.Operator
) -> Box.Box:
    return Box.TextBox(f' > {type(operator).__qualname__} > ')

def output_weave_box[B: cat.Datatype, A: cat.Axis](
    weave: cat.Weave[B, A],
    degree: cat.ProdObject[A]
) -> Box.Box:
    iterate_degree = iter(degree)
    right = Box.Vertical((
        *(
            display_long_axis(axis, 'left', extension='~')
            if isinstance(axis, cat.Axis)
            else display_long_axis(next(iterate_degree), 'left')
            for axis in weave._shape
        ),
        Box.Horizontal((
            Box.Fill(' ', min_width=2, min_height=1),
            datatype(weave.datatype),
        ))
    ))
    return right

def separated_product(
    target: Iterable[Box.Box],
    separator: Box.Box = Box.Fill('-', min_height=1)
):
    return Box.Vertical.from_iter(
        util.join_with_none(target, separator),
        justify_mode=js.JustifyMode.CENTER
    )

def display_broadcasted_join[B: cat.Datatype, A: cat.Axis](
        target: cat.Broadcasted[B, A],) -> Box.Box:
    
    return Box.Horizontal((
        separated_product(
            (reindexed_weave(weave, reindexing)
             for weave, reindexing
             in zip(target.input_weaves, target.reindexings)
             ),
             Box.Fill('-', min_height=1),
        ),
        Box.Vertical((
        Box.Vertical((
            display_degree(target.degree()),
            display_core(target.operator),
        )),), justify_mode=js.JustifyMode.CENTER),
        separated_product(
            (output_weave_box(weave, target.degree())
             for weave in target.output_weaves),
            Box.Fill('-', min_height=1
            )
        )
    ))

def rearrangement[B: cat.Datatype, A: cat.Axis](
    target: cat.Rearrangement[cat.Array[B, A]]
) -> Box.Box:

    left_labels = separated_product(
        Box.Vertical((
            *(
                Box.Horizontal((
                    display_axis(axis),
                    Box.Fill('─', 2, 1),
                ))
                for axis in array.shape()
            ),
            datatype(array.datatype)
        ))
        for array in target.dom()
    )
    right_labels = separated_product(
        Box.Horizontal(
            (Box.TextBox(f'[{idx}]'),
             Box.Vertical((
                *(
                    Box.Horizontal((
                        Box.Fill('─', 2, 1),
                        display_axis(axis),
                    ))
                    for axis in array.shape()
                ),
                datatype(array.datatype)
            )),)
        )
        for array, idx in zip(
            target.cod(),
            target.mapping
        )
    )
    return Box.Horizontal((left_labels, right_labels))


def _row_label(row: tuple, out_axes: tuple) -> str:
    """Format one Reindex row as an affine expression, e.g. 'x+dx' or '2·xo+px'."""
    const, coeffs = row
    terms = []
    if const:
        terms.append(str(const))
    for k, c in coeffs:
        ax = out_axes[k]
        name = ax.uid._name.to_bodies() if ax.uid._name is not None else f'a{k}'
        terms.append(name if c == 1 else f'{c}·{name}')
    return '+'.join(terms) if terms else '0'


def reindex(target: Reindex) -> Box.Box:
    """Render a Reindex as a labelled hexagon: in_axes — ⟨expr∣ — out_axes."""
    label = '⟨' + ', '.join(
        _row_label(row, target.out_axes) for row in target.rows
    ) + '∣'
    left_labels = separated_product(
        Box.Horizontal((display_axis(axis), Box.Fill('─', 2, 1)))
        for axis in target.in_axes
    )
    middle = Box.TextBox(label)
    right_labels = separated_product(
        Box.Horizontal((Box.Fill('─', 2, 1), display_axis(axis)))
        for axis in target.out_axes
    )
    return Box.Horizontal((left_labels, middle, right_labels))


def display_block[B: cat.Datatype, A: cat.Axis](
    target: cat.Block[
        cat.Array[B, A],
        cat.BroadcastedCategory[B, A],
    ],
):
    body_box = display_category(target.body)
    if target.aesthetics and (
        target.aesthetics.title 
        or target.aesthetics.description):
        return Box.Padded(
            Box.Vertical((
                *((Box.TextBox(f' {target.aesthetics.title} '),)
                  if target.aesthetics.title else ()),
                *((Box.TextBox(f' {target.aesthetics.description} '),)
                  if target.aesthetics.description else ()),
                  body_box
            ))
        )
    return Box.Padded(
        body_box
    )
        
        

def display_category[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A]
) -> Box.Box:
    match target:
        case cat.Broadcasted():
            return display_broadcasted_join(target)
        case cat.ThreadedComposed(content=ms) | cat.Composed(content=ms):
            boxes = (display_category(m) for m in ms)
            return Box.Horizontal.from_iter(
                util.join_with_none(
                boxes,
                Box.Fill(' ', min_height=0, min_width=1)
                )
            )
        case cat.ProductOfMorphisms(content=ms):
            return separated_product(
                (display_category(m) for m in ms),
            )
        case cat.Block():
            return display_block(target)
        case cat.Rearrangement():
            return rearrangement(target)
        case _ if isinstance(target, Reindex):
            return reindex(target)