# Claude Opus 5.5 (1M context), effort 40.
'''The standard expansion of a `Caching`, registered into the table of
`algebra.registries.standard_expansions`.

Import this module for its side effect, as
`import caching.registries.standard_expansions`, wherever a cache has to be written out
or shown beside its expansion.

The expansion copies the operand, appends one copy to the cache with a
`Para.CacheDrop`, reads the entries the cache holds for the earlier tokens with a
`Para.CacheGrab`, and concatenates the operand after them with
`aops.ConcatenateAxes`. The grab loads the kept earlier tokens, which are every earlier
token or the last `|K|` a sliding window keeps, and the drop stores the `|x|` tokens of
this pass, so the expansion states the memory traffic a cache costs in one pass. The
user asked on 2026-09-26 for the two bespoke seeds in place of the stream
seeds, which wrote the whole concatenation back every pass.

A `CacheGrab` reads what every `CacheDrop` of its slot appended in the earlier passes.
The loop over passes is outside the expression, and inside a model the innermost
repeated block is the loop over layers, where a stream seed would carry a value from
one layer to the next. A model therefore holds the `Caching` itself and is written out
only in a figure that defines the operator.
'''
from __future__ import annotations

import functools

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.registries.standard_expansions as standard_expansions
import algebra.write_index_notation as write_index_notation
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Term as fd
import para.data_structure.Para as Para

import caching.data_structure.Caching as Caching
from caching.registries.cache_wording import TEXT as text


@functools.cache
def slot_of_cache(cache_name: str) -> Para.TapeSlot:
    '''The tape slot a cache is written out onto, one per name of a cache in a
    process, so that the read and the write of one expansion name one slot. The
    slot is inner, per `obsidian/07-para/Outer and Inner Tape Slots.md`: a cache
    lifted over the heads, or kept over axes beside the tokens, loads and stores
    one set of entries for every index of those axes.'''
    return fd.DynamicName.from_str(cache_name).capture(Para.TapeSlot())


def names_of_the_token_axes(target: cat.Broadcasted) -> fd.Prod[str]:
    '''The LaTeX of the axis of the earlier tokens and of the axis of the tokens of
    this pass, as the cache names them, `x_old` and `x_new` in a derived pass.'''
    return write_index_notation.axis_letters(
        (Caching.past_tokens_of(target), Caching.tokens_of_this_pass(target)))


def caching_formula(target: cat.Broadcasted) -> str:
    '''The load of the earlier tokens, the result at a token of this pass, and the
    append of that token, written over the token axes of `target`.'''
    past, tokens = names_of_the_token_axes(target)
    loaded = write_index_notation.index_of(past)
    appended = write_index_notation.index_of(tokens)
    after_the_past = rf'\lvert {past} \rvert + {appended}'
    return (rf'y[{loaded}] = \mathrm{{cache}}[{loaded}], \qquad '
            rf'y[{after_the_past}] = v[{appended}], \qquad '
            rf'\mathrm{{cache}}[{after_the_past}] \leftarrow v[{appended}]')


def caching_description(target: cat.Broadcasted) -> str:
    past, tokens = names_of_the_token_axes(target)
    return text.CACHING_EXPANSION_DESCRIPTION.format(past=past, tokens=tokens)


@standard_expansions.register(
    Caching.Caching, formula=caching_formula, description=caching_description)
def load_the_past_and_append_this_pass(
    target: cat.Broadcasted,
) -> cat.BroadcastedCategory:
    '''The cache written out as a copy of the operand appended to the cache by a
    cache drop, and the operand concatenated after the earlier tokens a cache grab
    loads.'''
    saved = target.dom()[0]
    past = Caching.past_array_of(target)
    slot = slot_of_cache(target.operator.name.to_text())
    concatenation = aops.ConcatenateAxes.template(
        (tuple(past.shape()), tuple(saved.shape())), base=saved.datatype,
        concatenated=Caching.cached_tokens_of(target))
    return ((Para.CacheGrab(tape=slot, size=past)
             * cat.Rearrangement((0, 0), (saved,)))
            @ (concatenation * Para.CacheDrop(tape=slot, size=saved)))
