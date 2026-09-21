'''Reading an einops signature string into a `Broadcasted`.

A signature such as `'q d, x d -> q x'` names one operand per comma-separated
segment and one result per segment on the right of the arrow. The parse assigns
each name an index, builds a `Weave` per operand, and builds one `Rearrangement`
per operand carrying that operand's reindexing from the result's degree.

`construction_helpers.signature` reads the shorter signatures the generic
operators take. `obsidian/02-categories/Construction Helpers.md` describes both,
and `obsidian/02-categories/Weaves and Degree.md` describes what a weave and a
degree are.
'''
from __future__ import annotations
import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import utilities.utilities as util


def split_array(
    target: str,
    separator_axes: str = ' ',
    verbose: bool = False,
) -> list[str]:
    axes = [
        axis.strip()
        for axis in
        target.strip().split(separator_axes)
        if (stripped := axis.strip()) != ''
    ]
    return axes


def _partition_string(
    target: str,
    separator_segments: str = ',',
    separator_axes: str = ' ',
    verbose: bool = False,
) -> tuple[list[list[str]], list[list[str]]]:
    '''The axis names of each operand, for the inputs and for the outputs.

    A signature with no arrow is read as inputs alone, and the output side comes
    back as one empty segment.
    '''
    target = target.strip()
    if '  ' in target:
        target = target.replace('  ', ' ')
    split_arrow = target.split('->')
    if len(split_arrow) == 1:
        input_portion = split_arrow[0]
        output_portion = ''
    elif len(split_arrow) == 2:
        input_portion, output_portion = split_arrow
    else:
        raise ValueError(f"Invalid signature: {target}")
    input_buckets = [
        split_array(bucket, separator_axes)
        for bucket in input_portion.strip().split(separator_segments)
    ]
    output_buckets = [
        split_array(bucket, separator_axes)
        for bucket in output_portion.strip().split(separator_segments)
    ]
    if verbose:
        print('Input buckets:', input_buckets, 'Output buckets:', output_buckets)
    return input_buckets, output_buckets

def get_axis_names(
    input_buckets: list[list[str]],
    output_buckets: list[list[str]],
    produced_as_degree: bool = False,
) -> fd.Prod[tuple[int, str]]:
    """
    Number every name in the signature.

    A negative number is a degree axis. A non-negative number is an axis of the
    operator's own target, the absorbed names numbered first and the produced
    names after them. Every output segment has to carry the whole degree, and
    the assertion below is that condition.

    A name on both sides of the arrow is degree. A name in the output alone is
    produced. For a generic operator a produced name is an axis the operator
    writes, as the output of a `Linear` is, and it occupies a target position.
    An einsum writes nothing along such a name. It repeats the value along it,
    which is a degree axis that no operand's reindexing names. `Einops.template`
    sets `produced_as_degree` for that reason, and the flag counts every output
    name as degree.
    """
    names_in_input = {a for bucket in input_buckets for a in bucket}
    names_in_output = {a for bucket in output_buckets for a in bucket}
    tiled_names = names_in_input & names_in_output
    if produced_as_degree:
        tiled_names = names_in_output
    assert all(tiled_names <= set(output_bucket) for output_bucket in output_buckets)
    input_names = names_in_input - tiled_names
    output_names = names_in_output - tiled_names
    axis_names: fd.Prod[tuple[int, str]] = (
        *((-i-1, name) for i, name in enumerate(tiled_names)),
        *((i, name) for i, name in enumerate(input_names)),
        *((i + len(input_names), name) for i, name in enumerate(output_names))
    )
    return axis_names

def signature_to_broadcast(
    signature: str,
    datatype: cat.Datatype = cat.Reals(),
    support_unit_object: bool = False,
    verbose: bool = False,
    produced_as_degree: bool = False,
) -> tuple[
    fd.Prod[fd.Prod[int]],
    fd.Prod[fd.Prod[int]],
    fd.Prod[cat.Weave[cat.Datatype, cat.RawAxis]],
    fd.Prod[cat.Weave[cat.Datatype, cat.RawAxis]],
    fd.Prod[cat.Rearrangement[cat.RawAxis]],
    cat.ProdObject[cat.RawAxis],
]:
    '''
    The parts a `Broadcasted` is built from, for one signature string.

    The six are the input indexes, the output indexes, the input weaves, the
    output weaves, the reindexings and the degree. The indexes are the numbering
    `get_axis_names` assigns, and `Einops.template` reads them to recover its
    contraction groups.

    A name made only of digits becomes an axis of that literal size rather than
    an axis named by that string. `support_unit_object` reads an empty side of
    the arrow as the unit object, in place of one operand carrying no axes.

    Every output segment has to list the degree axes in the same order, because
    `util.iallequals` takes one order for all of them and raises otherwise.
    '''
    input_buckets, output_buckets = _partition_string(signature, verbose=verbose)
    input_buckets = [] if input_buckets == [[]] and support_unit_object else input_buckets
    output_buckets = [] if output_buckets == [[]] and support_unit_object else output_buckets
    axis_names = get_axis_names(input_buckets, output_buckets, produced_as_degree)
    name_to_index = {name: index for index, name in axis_names}
    index_to_axis = {
        index: (
            cat.RawAxis.named(name)
            if not name.isdigit()
            else cat.RawAxis(_size=nm.Integer(int(name)))
        )
        for index, name in axis_names
    }
    input_indexes, output_indexes = (
        tuple(
            tuple(name_to_index[axis] for axis in segment)
            for segment in buckets
        )
        for buckets in (input_buckets, output_buckets)
    )
    degree_indexes = util.iallequals(
        [index for index in bucket if index < 0]
        for bucket in output_indexes
    )
    degree = tuple(index_to_axis[index] for index in degree_indexes)
    input_weaves, output_weaves = tuple(
        tuple(
            cat.Weave(datatype, tuple(
                index_to_axis[index]
                if index >= 0
                else cat.WeaveMode.TILED
                for index in segment
            ))
            for segment in side
        )
        for side in (input_indexes, output_indexes)
    )
    reindexings = tuple(
        cat.Rearrangement(
            tuple(degree_indexes.index(index) for index in segment if index < 0),
            _dom=degree
        )
        for segment in input_indexes
    )
    # The degree is returned separately from the reindexings. A signature with
    # no inputs has no reindexing to carry it, and the caller puts it in
    # `backup_degree` instead.
    return (input_indexes, output_indexes, input_weaves, output_weaves, reindexings,
            cat.ProdObject(degree))

def signature_to_broadcasted(
    operator: cat.Operator,
    signature: str,
    datatype: cat.Datatype = cat.Reals(),
    support_unit_object: bool = False,
) -> cat.Broadcasted[cat.Datatype, cat.RawAxis]:
    _input_indexes, _output_indexes, input_weaves, output_weaves, reindexings, degree = signature_to_broadcast(
        signature, datatype, support_unit_object=support_unit_object)
    return cat.Broadcasted(
        operator,
        input_weaves=input_weaves,
        output_weaves=output_weaves,
        reindexings=reindexings,
        backup_degree=degree if not input_weaves else None,
    )
