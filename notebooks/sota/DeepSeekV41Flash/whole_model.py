'''DeepSeek-V4.1-Flash end to end, from the token identifiers to the probabilities.

Written by Claude Opus 5, effort high.

The embedding, the expansion into the four residual streams, the initial collapse
vector, the forty layers of the layer plan, the collapse of the four streams into one
and the output head.

The initial collapse vector is a covariant view with no operands. A reindexing is a
linear map, and the row with an empty domain and the shift zero read the other way
selects position 0 of the stream axis, so read covariantly it writes its operand at
position 0 and leaves every other position holding the universal unit. The contraction
that collapses the streams ignores a unit, so the array is the one-hot vector the
reference's `make_identity_pre_mix` returns. `aops.CovariantView.template`
cannot build it, because it asks `mark_sparse_codomains.merge_groups` for a bijection
between the domain and the codomain index boxes and an injection out of the empty
product is not one, so the `cat.Broadcasted` is written directly, as the parameter
arrays of `custom_operations` are.
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, route
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, X, m, n, vocab, x
from notebooks.sota.DeepSeekV41Flash.layer_stack import (
    decoder, encoder_group, swa_block)
from notebooks.sota.DeepSeekV41Flash.single_pass_mhc import collapse_streams
from notebooks.sota.DeepSeekV41Flash.declared_axes import COLLAPSE
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

INITIAL_COLLAPSE_NAME = 'A^{0}'


def embed() -> cat.Block:
    return cat.Block.template(ops.Embedding.template(vocab, (m,)),
                              title='\\text{Embedding}', fill_color='#FCE0E1',
                              description=text.EMBED_DESCRIPTION,
                              references=(model_lines(174), model_lines(1254)))


def expand_into_streams() -> cat.Broadcasted:
    '''One hidden state copied into the four residual streams.'''
    return ops.View.template(reindexing=cat.Rearrangement((0, 2), (x, n, m)),
                             name='rep')


def initial_collapse() -> cat.Broadcasted:
    '''The coefficients the first sublayer collapses with, which are fixed rather than
    predicted because no sublayer has run yet: the one-hot vector on the first stream,
    written as the covariant reading of the row that selects the first stream.'''
    select_first_stream = sc.StrideMorphism(
        _dom=(),
        _cod_stride_shift=((n, (), nm.Integer(0)),),
        name=fd.DynamicName(INITIAL_COLLAPSE_NAME))
    return cat.Broadcasted(
        operator=aops.CovariantView(name=fd.DynamicName(INITIAL_COLLAPSE_NAME),
                                   reindexing=select_first_stream),
        output_weaves=(cat.Weave(R, (cat.WeaveMode.TILED, n)),),
        backup_degree=cat.ProdObject((x,)))


def unembed() -> cat.Block:
    return cat.Block.template(
        ops.Normalize.template() @ ops.Linear.template((m,), (vocab,))
        @ ops.SoftMax.template(),
        title='\\text{Output Probabilities}', fill_color='#DBDFEF',
        description=text.UNEMBED_DESCRIPTION,
        references=(model_lines(1269), model_lines(1012), model_lines(1291)))


v41_flash = (embed()
             @ expand_into_streams()
             @ (initial_collapse() * hold(X))
             @ swa_block
             @ encoder_group
             @ decoder
             @ collapse_streams()
             @ unembed())
