# Claude Opus 5.5 (1M context), effort 40.
'''MiMo-V2.6-Pro end to end, from the token identifiers to the logits of the next token.

The model is the text path of `MiMoV2ForCausalLM` of the reference: the embedding, the
70 layers of `layer_stack`, the RMS normalisation of the final hidden state and the
output head. It reads one array and returns one:

    the token identifiers   Nat(v)[x]
    the logits              R[x, v]

The reference returns the logits, and a sampler reads them. The checkpoint also holds
a vision encoder and two audio encoders, whose outputs replace the embeddings of the
image, video and audio tokens of a prompt, and a drafter of five layers for speculative
decoding, whose weights the reference skips when it loads the checkpoint. The
expression holds none of them, and the notebook states each.

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
import term_utilities.generate_config as gc
import term_utilities.term_utilities as tutil

from notebooks.sota.DeepSeekV41Flash.construction_idioms import over
from notebooks.sota.MiMoV26Pro.declared_axes import m, vocab, x
from notebooks.sota.MiMoV26Pro.layer_stack import layer_stack
from notebooks.sota.MiMoV26Pro.reference_links import (
    checkpoint_config_lines, modeling_lines)
from notebooks.sota.MiMoV26Pro.released_constants import NORM_EPSILON
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text

EMBEDDING_COLOUR = '#FCE0E1'
OUTPUT_COLOUR = '#DBDFEF'

RELEASED_SIZES: dict[str, int] = {
    'm': 6144, 'h': 8, 'g': 16, 't': 32, 'n': 128, 'u': 128, 'w': 128,
    'e': 384, 'k': 8, 'f': 2048, 'd': 16384, 'v': 152576}


def embed() -> cat.Block:
    return cat.Block.template(
        ops.Embedding.template(vocab, (m,)),
        title=text.EMBEDDING_TITLE, fill_color=EMBEDDING_COLOUR,
        description=text.EMBEDDING_BLOCK_DESCRIPTION,
        references=(modeling_lines(1584), modeling_lines(1627, 1628),
                    modeling_lines(1827, 1836), checkpoint_config_lines(440)))


def output_logits() -> cat.Block:
    return cat.Block.template(
        over((x,), ops.Normalize.template((m,), epsilon=NORM_EPSILON))
        @ (x >> ops.Linear.template((m,), (vocab,), 'W^{L}')),
        title=text.OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.OUTPUT_DESCRIPTION,
        references=(modeling_lines(1595), modeling_lines(1677), modeling_lines(1704),
                    modeling_lines(1849, 1851), checkpoint_config_lines(375)))


mimo = embed() @ layer_stack @ output_logits()


def released_assigned_sizes(term: fd.GeneralTerm = mimo) -> dict[str, int]:
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


def part_named(short_name: str, term: fd.GeneralTerm = mimo) -> cat.Broadcasted:
    '''The first box of `term` carrying `short_name`, which is how a figure asks for
    one part of the model.'''
    for node in tutil.type_search(cat.Broadcasted, term):
        if is_box_named(node, short_name):
            return node
    raise NameIsNotABoxOfTheModel(f'{short_name} names no box of the model')


def part_titled(title: str, term: fd.GeneralTerm = mimo) -> cat.Block:
    '''The first block of `term` carrying `title`, which is how a part the model
    composes without boxing it is asked for.'''
    for block in tutil.type_search(cat.Block, term):
        if (block.block_tag.aesthetics is not None
                and block.block_tag.aesthetics.title == title):
            return block
    raise TitleIsNotABlockOfTheModel(f'{title} titles no block of the model')
