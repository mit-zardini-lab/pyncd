from __future__ import annotations
import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import utilities.utilities as util


### SIGNATURE OPERATIONS

def split_array(
    target: str,
    separator_axes: str = ' ',
    verbose: bool = False,
):
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
):
    # Clean the string
    target = target.strip()
    if '  ' in target:
        target = target.replace('  ', ' ')
    # Split inputs / outputs
    split_arrow = target.split('->')
    if len(split_arrow) == 1:
        input_portion = split_arrow[0]
        output_portion = ''
    elif len(split_arrow) == 2:
        input_portion, output_portion = split_arrow
    else:
        raise ValueError(f"Invalid signature: {target}")
    # Find the bucket keys
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
):
    names_in_input = {a for bucket in input_buckets for a in bucket}
    names_in_output = {a for bucket in output_buckets for a in bucket}
    # Get the unique occurences
    tiled_names = names_in_input & names_in_output
    assert all(tiled_names <= set(output_bucket) for output_bucket in output_buckets)
    input_names = names_in_input - tiled_names
    output_names = names_in_output - tiled_names
    # Make the axis naming convention
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
    verbose: bool = False
):
    input_buckets, output_buckets = _partition_string(signature, verbose=verbose)
    input_buckets = [] if input_buckets == [[]] and support_unit_object else input_buckets
    output_buckets = [] if output_buckets == [[]] and support_unit_object else output_buckets
    axis_names = get_axis_names(input_buckets, output_buckets)
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
    return (input_indexes, output_indexes, input_weaves, output_weaves, reindexings)

def signature_to_broadcasted(
    operator: cat.Operator,
    signature: str,
    datatype: cat.Datatype = cat.Reals(),
    support_unit_object: bool = False,
):
    _input_indexes, _output_indexes, input_weaves, output_weaves, reindexings = signature_to_broadcast(
        signature, datatype, support_unit_object=support_unit_object)
    return cat.Broadcasted(
        operator,
        input_weaves=input_weaves,
        output_weaves=output_weaves,
        reindexings=reindexings
    )
