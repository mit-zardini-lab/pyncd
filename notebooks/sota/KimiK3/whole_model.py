# Claude Opus 5.5 (1M context), effort 40.
'''Kimi K3 end to end, from the token identifiers to the logits of the next token.

The model is the text model of `KimiK3ForConditionalGeneration` of the reference, which
is `KimiLinearForCausalLM`: the embedding, the 93 layers of `layer_stack`, the last mix
of the attention residuals, the RMS normalisation of the final hidden state and the
output head. It reads one array and returns one:

    the token identifiers   Nat(v)[x]
    the logits              R[x, v]

The vision encoder MoonViT-V2 and the projector that writes image features into the
embeddings are left out, and a prompt of text passes the embedding to the layers
unchanged.

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
from notebooks.sota.KimiK3.attention_residuals import (
    OUTPUT_INPUT_WEIGHT, mix_of, start_the_entries)
from notebooks.sota.KimiK3.declared_axes import m, vocab, x
from notebooks.sota.KimiK3.layer_stack import layer_stack
from notebooks.sota.KimiK3.reference_links import (
    checkpoint_config_lines, modeling_lines, wrapper_lines)
from notebooks.sota.KimiK3.released_constants import NORM_EPSILON
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

EMBEDDING_COLOUR = '#FCE0E1'
OUTPUT_COLOUR = '#DBDFEF'

RELEASED_SIZES: dict[str, int] = {
    'm': 7168, 'h': 96, 'q': 1536, 'c': 512, 'n': 128, 'p': 64, 'u': 128,
    'j': 96, 'd': 128, 'z': 128, 'w': 4, 'e': 896, 'k': 16, 'l': 3584, 'f': 3072,
    't': 6144, 'g': 33792, 'b': 9, 'v': 163840}


def embed() -> cat.Block:
    return cat.Block.template(
        ops.Embedding.template(vocab, (m,)) @ start_the_entries(),
        title=text.EMBEDDING_TITLE, fill_color=EMBEDDING_COLOUR,
        description=text.EMBEDDING_BLOCK_DESCRIPTION,
        references=(modeling_lines(1096, 1097), modeling_lines(1156, 1158),
                    modeling_lines(1188, 1192), wrapper_lines(1227, 1230),
                    checkpoint_config_lines(267)))


def output_logits() -> cat.Block:
    return cat.Block.template(
        mix_of(OUTPUT_INPUT_WEIGHT)
        @ over((x,), ops.Normalize.template((m,), epsilon=NORM_EPSILON))
        @ (x >> ops.Linear.template((m,), (vocab,), 'W^{L}')),
        title=text.OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.OUTPUT_DESCRIPTION,
        references=(modeling_lines(1100, 1108), modeling_lines(1215, 1219),
                    modeling_lines(1226, 1233), modeling_lines(1247, 1248),
                    modeling_lines(1298, 1301), checkpoint_config_lines(254)))


kimi_k3 = embed() @ layer_stack @ output_logits()


def released_assigned_sizes(term: fd.GeneralTerm = kimi_k3) -> dict[str, int]:
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


def part_named(short_name: str, term: fd.GeneralTerm = kimi_k3) -> cat.Morphism:
    '''The first box of `term` carrying `short_name`, which is how a figure asks for
    one part of the model.'''
    for wrap in tutil.type_search(para_wrap.ParaWrap, term):
        if is_box_named(wrap.body, short_name):
            return wrap
    for node in tutil.type_search(cat.Broadcasted, term):
        if is_box_named(node, short_name):
            return node
    raise NameIsNotABoxOfTheModel(f'{short_name} names no box of the model')


def part_titled(title: str, term: fd.GeneralTerm = kimi_k3) -> cat.Block:
    '''The first block of `term` carrying `title`, which is how a part the model
    composes without boxing it is asked for.'''
    for block in tutil.type_search(cat.Block, term):
        if (block.block_tag.aesthetics is not None
                and block.block_tag.aesthetics.title == title):
            return block
    raise TitleIsNotABlockOfTheModel(f'{title} titles no block of the model')
