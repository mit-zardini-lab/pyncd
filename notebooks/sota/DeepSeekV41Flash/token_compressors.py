'''The compressor of DeepSeek-V4.1-Flash, at the two ratios the two halves use.

Written by Claude Opus 5, effort high.

The encoder folds two consecutive tokens into one compressed entry and the decoder
folds one, so the ratio-2 form has a group view, two projections and a per-channel
softmax over the group, and the ratio-1 form degenerates to a renaming view, one
projection and the norm.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import axis_name, over, route
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, a, c, m, x
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

COMPRESSOR_COLOUR = '#B8D8CE'


def pool_tokens_into_entries[A: cat.Axis](entry_axis: A) -> cat.Block:
    '''The ratio-2 compressor: one group view, two projections, a per-channel softmax over
    the group, and one contraction.'''
    entry = axis_name(entry_axis)
    group = ops.View.template(
        reindexing=(sc.StrideMorphism(
            _dom=(entry_axis, a),
            _cod_stride_shift=((x, (a.local_size(), nm.Integer(1)), nm.Integer(0)),),
            name=fd.DynamicName('grp')),
            cat.ProdObject((m,)).identity()),
        name='grp')
    values = (over((entry_axis, a), ops.Linear.template((m,), (c,), 'W^{C}'))
              @ ops.Einops.template(f'{entry} a c -> {entry} c a'))
    gates = (over((entry_axis, a), ops.Linear.template((m,), (c,), 'W^{Z}'))
             @ ops.Einops.template(f'{entry} a c -> {entry} c a')
             @ over((entry_axis, c), ops.SoftMax.template()))
    return cat.Block.template(
        group @ route((0, 0), (cat.Array(R, (entry_axis, a, m)),))
        @ (values * gates)
        @ ops.Einops.template(f'{entry} c a, {entry} c a -> {entry} c')
        @ over((entry_axis,), ops.Normalize.template((c,))),
        title=text.COMPRESSOR_TITLE, fill_color=COMPRESSOR_COLOUR,
        description=text.POOL_TOKENS_INTO_ENTRIES_DESCRIPTION,
        references=(model_lines(429), model_lines(475)))


def project_tokens_into_entries[A: cat.Axis](entry_axis: A) -> cat.Block:
    '''The ratio-1 compressor: a stride-1 renaming view, one projection and the norm.'''
    rename = ops.View.template(
        reindexing=(sc.StrideMorphism(
            _dom=(entry_axis,),
            _cod_stride_shift=((x, (nm.Integer(1),), nm.Integer(0)),),
            name=fd.DynamicName('pos')),
            cat.ProdObject((m,)).identity()),
        name='pos')
    return cat.Block.template(
        rename @ over((entry_axis,), ops.Linear.template((m,), (c,), 'W^{C}'))
        @ over((entry_axis,), ops.Normalize.template((c,))),
        title=text.COMPRESSOR_TITLE, fill_color=COMPRESSOR_COLOUR,
        description=text.PROJECT_TOKENS_INTO_ENTRIES_DESCRIPTION,
        references=(model_lines(429), model_lines(475)))
