# Claude Opus 5.5 (1M context), effort 40.
'''GLM-5.3 end to end, from the token identifiers to the logits of the next token.

The model is `GlmMoeDsaForCausalLM` of the reference: the embedding, the 78 layers of
`layer_stack`, the RMS normalisation of the final hidden state and the output head. It
reads one array and returns one:

    the token identifiers   Nat(v)[x]
    the logits              R[x, v]

The reference returns the logits, and a sampler reads them. `generation_config.json`
sets the temperature to 1 and the top-p to 0.95.

The checkpoint holds a 79th layer, the multi-token prediction layer, whose weights the
reference skips when it loads the checkpoint, through
`_keys_to_ignore_on_load_unexpected`. The expression holds the 78 layers the reference
runs, and the notebook states the layer it leaves out.

`RELEASED_SIZES` gives the size of every axis in the configuration of the checkpoint,
keyed by the body of the name of its symbol, and `released_assigned_sizes` binds them on
a term through one `gc.NumericConfig`. A figure passes the mapping as `assigned_sizes`,
and the display writes each size onto the label of its axis while the term stays
symbolic.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import para.data_structure.ParaWrap as para_wrap
import term_utilities.generate_config as gc
import term_utilities.term_utilities as tutil

from notebooks.sota.DeepSeekV41Flash.construction_idioms import over
from notebooks.sota.GLM53.declared_axes import m, vocab, x
from notebooks.sota.GLM53.layer_stack import layer_stack
from notebooks.sota.GLM53.reference_links import (
    checkpoint_config_lines, modeling_lines)
from notebooks.sota.GLM53.released_constants import NORM_EPSILON
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

EMBEDDING_COLOUR = '#FCE0E1'
OUTPUT_COLOUR = '#DBDFEF'

RELEASED_SIZES: dict[str, int] = {
    'm': 6144, 'h': 64, 'q': 2048, 'c': 512, 'n': 192, 't': 32, 'u': 256,
    'i': 32, 'd': 128, 'e': 256, 'k': 8, 'f': 2048, 'g': 12288, 's': 2048,
    'v': 154880}


def embed() -> cat.Block:
    return cat.Block.template(
        ops.Embedding.template(vocab, (m,)),
        title=text.EMBEDDING_TITLE, fill_color=EMBEDDING_COLOUR,
        description=text.EMBEDDING_BLOCK_DESCRIPTION,
        references=(modeling_lines(669), modeling_lines(696, 697),
                    checkpoint_config_lines(223)))


def output_logits() -> cat.Block:
    return cat.Block.template(
        over((x,), ops.Normalize.template((m,), epsilon=NORM_EPSILON))
        @ (x >> ops.Linear.template((m,), (vocab,), 'W^{L}')),
        title=text.OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.OUTPUT_DESCRIPTION,
        references=(modeling_lines(673), modeling_lines(735), modeling_lines(753),
                    modeling_lines(799, 802), checkpoint_config_lines(217)))


glm53 = embed() @ layer_stack @ output_logits()


def released_assigned_sizes(term: fd.GeneralTerm = glm53) -> dict[str, int]:
    '''The released size of every symbol of `term` that `RELEASED_SIZES` names, keyed
    by the bodies of the symbol's name.'''
    config = gc.NumericConfig.template(term)
    config.assign_values(**RELEASED_SIZES)
    return config.assigned_integers_by_name()


class NameIsNotABoxOfTheModel(ValueError):
    '''A name asked of the model and carried by no box of it.'''


class TitleIsNotABlockOfTheModel(ValueError):
    '''A title asked of the model and carried by no block of it.'''


def is_box_named(morphism: cat.Morphism, short_name: str) -> bool:
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, ops.BlockOperator)
            and morphism.operator.name is not None
            and morphism.operator.name.to_bodies() == short_name)


def part_named(short_name: str, term: fd.GeneralTerm = glm53) -> cat.Morphism:
    '''The first box of `term` carrying `short_name`, which is how a figure asks for
    one part of the model. A box whose block holds a grab or a drop stands inside a
    `ParaWrap` naming the slot at each of its taped ports, and a figure draws the
    wrap, so a wrap is looked for first.'''
    for wrap in tutil.type_search(para_wrap.ParaWrap, term):
        if is_box_named(wrap.body, short_name):
            return wrap
    for node in tutil.type_search(cat.Broadcasted, term):
        if is_box_named(node, short_name):
            return node
    raise NameIsNotABoxOfTheModel(f'{short_name} names no box of the model')


def part_titled(title: str, term: fd.GeneralTerm = glm53) -> cat.Block:
    '''The first block of `term` carrying `title`, which is how a part the model
    composes without boxing it is asked for.'''
    for block in tutil.type_search(cat.Block, term):
        if (block.block_tag.aesthetics is not None
                and block.block_tag.aesthetics.title == title):
            return block
    raise TitleIsNotABlockOfTheModel(f'{title} titles no block of the model')
