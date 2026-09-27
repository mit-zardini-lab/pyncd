# Claude Opus 5.5 (1M context), effort 40.
'''Deriving the pass of a generating model that reads its earlier tokens from caches.

A model over a sequence axis `x` computes one result per token. A pass of generation
appends the tokens `n` to the `P` tokens of the earlier passes and needs the results
of the new tokens alone. Every wire of the derived pass is the model's wire read through
an affine map into `x`, and the crawl carries those maps back through the model with
`move_reads_backwards.ReadCrawler`. Two kinds of map occur.

- The read of the new tokens is the stride morphism from `n` to `x` with the one row
  `i_n + |P|`.
- The read of a cached axis `K + n`, the kept earlier tokens `K` followed by the new
  tokens, is the stride morphism with the one row `i + |P| - |K|`. Where `K` is every
  earlier token the shift is zero.

Every operator a read passes is rebuilt over the axis the read returns.

A causal read, such as the mask of an attention reading position `i_x - i_w` for every
slot `i_w`, composes with the read of the new tokens into a read of the positions
`|P| + i_n - i_w`, and those positions include earlier tokens. The reach of that read is
the number of earlier tokens it reaches before the first new token, the largest value of
`-(Σ s_j i_j + c)` over its domain. A reach that grows with the past needs every earlier
token. A reach that does not, such as the `|w| - 1` of a sliding window of `|w|` slots,
needs the last `|w| - 1` earlier tokens, which is the kept axis `K`. The crawl writes the
causal read as a view on the cached axis and asks the operand for the read of that
axis. The operator producing the operand is rebuilt over `n` and followed by a
`Caching`, whose result holds the operand over the cached axis. The read of the new
tokens then continues from the operand's producer. At the copy that feeds the queries
and the keys every branch now asks for the new tokens, so the read passes the copy and
reaches the layer before.

`computed_over_the_cache` names the operators computed at every cached token rather
than at the new tokens. The read of the cached axis passes through them, and a box
among them passes it through its whole body. The `Caching` stands after the first
operator the read meets that is not named, or after the copy where a branch reading the
cached axis meets a branch reading the new tokens. An empty set places every cache on
the operand of the causal read.

A read of a later token raises, and so does an operator that reads the token axis whole
without a causal read. In both cases the entries a cache holds for an earlier token
would change when later tokens arrive, so no cache of them is correct.

A box holding tape seeds is crawled with its seeds as ports, and the grab of a slot is
required to read the new tokens its drop wrote, so that a value passed between the
layers of one pass keeps its form.

After the crawl the token axis `x` stands only on the tables that read no token, such as
a rotary table read through a view. It is replaced there by `P + n`, and its size `|x|`
by `|P| + |n|`.

`obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` states the
mathematics and the proof that the derived pass computes the model's results for the
new tokens.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import construction_helpers.simple_helper as chsh
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as ParaBlockOperator
import para.data_structure.ParaWrap as para_wrap
import term_utilities.term_utilities as tutil
import utilities.utilities as util

import caching.data_structure.Caching as Caching

type Read = move_reads_backwards.Read

NEW_TOKENS_READ_NAME = fd.DynamicName('\\mathrm{New}')
'''The name of the read of the new tokens, which a view reading a table of positions
at the new tokens carries, as the rotary table of a model is read.'''
CACHED_TOKENS_READ_NAME = fd.DynamicName('\\mathrm{Cached}')
'''The name of the read of a cached axis, the kept earlier tokens followed by the new
ones.'''
TOKEN_READ_NAMES = (NEW_TOKENS_READ_NAME, CACHED_TOKENS_READ_NAME)


class ReadsALaterToken(ValueError):
    '''A read of the token axis whose row reaches a token after the token it computes,
    or reads the token axis without the new token it computes.'''


class ReadsTheTokenAxisWhole(ValueError):
    '''An operator reading every token of an operand to compute one token, with no
    causal read bounding which tokens it reads.'''


class CachedArrayHasTwoTokenPositions(ValueError):
    '''An array read at earlier tokens that holds the token axis at two positions, or
    through two rows reaching back by different counts, which one `Caching` cannot
    state.'''


class LoopChangesTheDemand(ValueError):
    '''A repeated block whose body returns another read on its domain than the read on
    its codomain, so one iteration cannot stand for every other.'''


class TapeCarriesAnotherDemand(ValueError):
    '''A grab of a tape slot read at other tokens than the new ones, where every drop of
    a slot in a pass writes the new tokens.'''


def is_new_token_row(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
                     new_token_positions: fd.Prod[int], past_size: nm.Numeric) -> bool:
    '''Whether a row reads the token `|P| + i_n` for the new token `i_n` alone.'''
    read = move_reads_backwards.positions_read(strides)
    return (len(read) == 1 and read[0] in new_token_positions
            and strides[read[0]] == nm.Integer(1) and shift == past_size)


def offset_from_the_newest_token(shift: nm.Numeric, past_size: nm.Numeric) -> nm.Numeric:
    return nm.collect_like_terms(nm.Addition.template(
        shift, nm.Multiplication.template(nm.Integer(-1), past_size)))


def is_earlier_token_row(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
                         new_token_positions: fd.Prod[int],
                         past_size: nm.Numeric) -> bool:
    '''Whether a row reads the token `|P| + i_n - o` for the new token `i_n` and an
    offset `o` that is never negative: stride one on one new token axis, a stride of
    zero or less on every other axis it reads, and a shift of at most `|P|`.'''
    read = move_reads_backwards.positions_read(strides)
    new_tokens_read = tuple(position for position in read
                            if position in new_token_positions)
    if len(new_tokens_read) != 1 or strides[new_tokens_read[0]] != nm.Integer(1):
        return False
    other_strides = tuple(strides[position] for position in read
                          if position != new_tokens_read[0])
    offset = offset_from_the_newest_token(shift, past_size)
    return (all(isinstance(stride, nm.Integer) and stride._value <= 0
                for stride in other_strides)
            and isinstance(offset, nm.Integer) and offset._value <= 0)


def is_identity_read(read: sc.StrideMorphism) -> bool:
    '''Whether `read` reads every axis as itself, in order.'''
    return (len(read._dom) == len(read._cod_stride_shift)
            and all(move_reads_backwards.selects_one_axis(strides, shift) == position
                    and read._dom[position] == axis
                    for position, (axis, strides, shift)
                    in enumerate(read._cod_stride_shift)))


def same_demand(one: Read, other: Read) -> bool:
    return move_reads_backwards.same_read(one, other)


@dataclass
class NewTokenCrawler[B: cat.Datatype, A: cat.Axis](move_reads_backwards.ReadCrawler[B, A]):
    '''The read of the new tokens carried back through a model, with a `Caching`
    placed wherever a causal read reaches earlier tokens.'''
    tokens: A | None = None
    past_tokens: A | None = None
    new_tokens: A | None = None
    cached_tokens: AxisConcatenation.ConcatenatedAxis | None = None
    kept_tokens: dict[nm.Numeric, AxisConcatenation.ConcatenatedAxis] = field(
        default_factory=dict)
    computed_over_the_cache: frozenset[cat.Operator] = frozenset()
    inside_a_box_computed_over_the_cache: bool = False
    numbered_caches: int = 0
    operators_before_caches: list[cat.Operator | None] = field(default_factory=list)

    def token_positions(self, array: cat.Array[B, A]) -> fd.Prod[int]:
        return tuple(position for position, axis in enumerate(array.shape())
                     if axis == self.tokens)

    def read_yields_its_name(self, read: sc.StrideMorphism) -> bool:
        '''The read of the new tokens and the read of a cached axis yield their names
        to a view its caller named, so the Diagonal of a mixture keeps its name when
        the read of the new tokens passes through it.'''
        return read.name is None or read.name in TOKEN_READ_NAMES

    def array_over(self, array: cat.Array[B, A], axis: A) -> cat.Array[B, A]:
        '''`array` with `axis` in place of the token axis.'''
        return cat.Array(array.datatype, tuple(
            axis if candidate == self.tokens else candidate for candidate in array.shape()))

    def new_token_array(self, array: cat.Array[B, A]) -> cat.Array[B, A]:
        return self.array_over(array, self.new_tokens)

    def cached_axes(self) -> fd.Prod[AxisConcatenation.ConcatenatedAxis]:
        return (self.cached_tokens, *self.kept_tokens.values())

    def shift_of(self, cached: AxisConcatenation.ConcatenatedAxis) -> nm.Numeric:
        '''The position of the token axis at which position zero of `cached` stands,
        `|P| - |K|` for the kept earlier tokens `K`.'''
        return offset_from_the_newest_token(self.past_tokens.local_size(),
                                            cached.parts[0].local_size())

    def read_of_the_token_axis(self, array: cat.Array[B, A], axis: A,
                               shift: nm.Numeric, name: fd.DynamicName,
                               ) -> sc.StrideMorphism:
        '''The read of `array` at `i + shift` of `axis` on every position of the token
        axis, and at the identity on every other axis, named `name`.'''
        shape = tuple(array.shape())
        domain = tuple(axis if candidate == self.tokens else candidate
                       for candidate in shape)
        rows = tuple(
            (candidate,
             tuple(nm.Integer(1) if i == position else nm.Integer(0)
                   for i in range(len(domain))),
             shift if candidate == self.tokens else nm.Integer(0))
            for position, candidate in enumerate(shape))
        read, _ = move_reads_backwards.in_reading_order(
            sc.StrideMorphism(_dom=domain, _cod_stride_shift=rows, name=name))
        return read

    def new_token_read(self, array: cat.Array[B, A]) -> sc.StrideMorphism:
        return self.read_of_the_token_axis(
            array, self.new_tokens, self.past_tokens.local_size(), NEW_TOKENS_READ_NAME)

    def cached_read(self, array: cat.Array[B, A],
                    cached: AxisConcatenation.ConcatenatedAxis) -> sc.StrideMorphism:
        return self.read_of_the_token_axis(
            array, cached, self.shift_of(cached), CACHED_TOKENS_READ_NAME)

    def new_token_positions(self, read: sc.StrideMorphism) -> fd.Prod[int]:
        return tuple(position for position, axis in enumerate(read._dom)
                     if axis == self.new_tokens)

    def token_rows(self, read: sc.StrideMorphism
                   ) -> fd.Prod[tuple[fd.Prod[nm.Numeric], nm.Numeric]]:
        return tuple((strides, shift) for axis, strides, shift in read._cod_stride_shift
                     if axis == self.tokens)

    def reads_new_tokens_alone(self, read: sc.StrideMorphism) -> bool:
        positions = self.new_token_positions(read)
        return all(is_new_token_row(strides, shift, positions,
                                    self.past_tokens.local_size())
                   for strides, shift in self.token_rows(read))

    def cached_axis_read_by(self, read: Read) -> AxisConcatenation.ConcatenatedAxis | None:
        '''The cached axis `read` reads on every row of the token axis at unit stride
        and at the shift of that axis, and `None` where it reads another way.'''
        rows = self.token_rows(read) if read is not None else ()
        for cached in self.cached_axes():
            positions = tuple(position for position, axis in enumerate(read._dom)
                              if axis == cached) if rows else ()
            if rows and all(
                    move_reads_backwards.selects_one_axis(
                        strides, nm.Integer(0)) in positions
                    and shift == self.shift_of(cached)
                    for strides, shift in rows):
                return cached
        return None

    def reads_earlier_tokens(self, read: sc.StrideMorphism) -> bool:
        positions = self.new_token_positions(read)
        return all(is_earlier_token_row(strides, shift, positions,
                                        self.past_tokens.local_size())
                   for strides, shift in self.token_rows(read))

    def reach(self, read: sc.StrideMorphism) -> nm.Numeric:
        '''The number of earlier tokens `read` reaches before the first new token: the
        largest value of `-(Σ s_j i_j + c)` over its domain, for the strides `s_j` its
        token row reads the other axes at and its offset `c` from `|P| + i_n`.'''
        positions = self.new_token_positions(read)
        reaches = util.unique_tuple(
            nm.collect_like_terms(nm.Multiplication.template(nm.Integer(-1), nm.Addition.template(
                offset_from_the_newest_token(shift, self.past_tokens.local_size()),
                *(nm.Multiplication.template(
                    strides[position],
                    nm.Addition.template(read._dom[position].local_size(), nm.Integer(-1)))
                  for position in move_reads_backwards.positions_read(strides)
                  if position not in positions))))
            for strides, shift in self.token_rows(read))
        if len(reaches) != 1:
            raise CachedArrayHasTwoTokenPositions(
                f'the rows of {read} reach back by {[r.to_latex() for r in reaches]}')
        return reaches[0]

    def grows_with_the_past(self, reach: nm.Numeric) -> bool:
        '''Whether `reach`, with the length of the sequence written as `|P| + |n|`,
        holds `|P|`, so that every earlier token is reached.'''
        in_a_pass = with_numeric_replaced(
            reach, self.tokens.local_size(), self.cached_tokens.local_size(), {})
        return any(symbol == self.past_tokens.local_size()
                   for symbol in tutil.type_search(nm.FreeNumeric, in_a_pass))

    def kept_tokens_code_form(self) -> str | None:
        '''The code form of the kept earlier tokens: `kept` joined before the code form
        of the earlier tokens, or before that of the token axis where the earlier
        tokens carry none, so `earlier_tokens` keeps `kept_earlier_tokens`.'''
        named = next((axis.uid._name for axis in (self.past_tokens, self.tokens)
                      if axis.uid._name is not None and axis.uid._name.code_form),
                     None)
        return None if named is None else fd.join_code_forms('kept', named.code_form)

    def cached_axis_for(self, reach: nm.Numeric) -> AxisConcatenation.ConcatenatedAxis | None:
        '''Every earlier token followed by the new ones where `reach` grows with the
        past, the last `reach` earlier tokens followed by the new ones where it does
        not, and `None` where it reaches no earlier token.'''
        if reach == nm.Integer(0):
            return None
        if self.grows_with_the_past(reach):
            return self.cached_tokens
        if reach not in self.kept_tokens:
            kept = fd.DynamicName(
                'x', fd.DynamicName('kept'), code_form=self.kept_tokens_code_form(),
            ).capture(cat.RawAxis(_size=reach))
            self.kept_tokens[reach] = Caching.cached_token_axis(kept, self.new_tokens)
        return self.kept_tokens[reach]

    def rebased(self, read: sc.StrideMorphism, axis: A,
                shift: nm.Numeric) -> sc.StrideMorphism:
        '''`read` with its rows onto the token axis written onto `axis`, whose position
        zero stands at position `shift` of the token axis.'''
        return read.reconstruct(_cod_stride_shift=tuple(
            (axis, strides, offset_from_the_newest_token(row_shift, shift))
            if row_axis == self.tokens else (row_axis, strides, row_shift)
            for row_axis, strides, row_shift in read._cod_stride_shift))

    def token_part(self, read: Read, array: cat.Array[B, A]) -> Read:
        '''The read of `array` at the new tokens or at a cached axis where `read` reads
        its token axis there, and `read` itself otherwise.'''
        if read is None or not self.token_rows(read):
            return read
        if self.reads_new_tokens_alone(read):
            return self.new_token_read(array)
        cached = self.cached_axis_read_by(read)
        return read if cached is None else self.cached_read(array, cached)

    def rest_after_token_part(self, read: sc.StrideMorphism) -> sc.StrideMorphism:
        '''The read that `read` is after its token part: the axis a token row reads, at
        the identity, where `read` reads the token axis, and the rows of `read`
        elsewhere. It keeps the name of `read` unless that names a read of the token
        axis, which the rest no longer is.'''
        return read.reconstruct(
            _cod_stride_shift=tuple(
                (read._dom[move_reads_backwards.positions_read(strides)[0]], strides,
                 nm.Integer(0))
                if axis == self.tokens else (axis, strides, shift)
                for axis, strides, shift in read._cod_stride_shift),
            name=None if read.name in TOKEN_READ_NAMES else read.name)

    def stopped(self, target: cat.Broadcasted[B, A], reads: Sequence[Read]
                ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''`target` carrying the token part of every read it cannot pass whole,
        followed by a view of the rest, and stopped as the read crawl stops it where
        the token part is the whole read.'''
        token_parts = tuple(self.token_part(read, array)
                            for read, array in zip(reads, target.cod()))
        if all(move_reads_backwards.same_read(part, read)
               for part, read in zip(token_parts, reads)):
            return super().stopped(target, reads)
        rebuilt, operand_reads = self.root_processor(target, token_parts)
        views = tuple(
            cat.ProdObject((array,)).identity()
            if move_reads_backwards.same_read(part, read)
            else move_reads_backwards.view_of(array, self.rest_after_token_part(read))
            for array, part, read in zip(rebuilt.cod(), token_parts, reads))
        return (chsh.make_composed(rebuilt, chsh.make_product(*views)), operand_reads)

    def cache_of(self, array: cat.Array[B, A], name: str | fd.DynamicName,
                 cached: AxisConcatenation.ConcatenatedAxis
                 ) -> cat.Broadcasted[B, A, Caching.Caching]:
        '''The `Caching` of `array`, which holds the new tokens on one position, onto the
        cached axis `cached`.'''
        positions = tuple(position for position, axis in enumerate(array.shape())
                          if axis == self.new_tokens)
        if len(positions) != 1:
            raise CachedArrayHasTwoTokenPositions(
                f'{array} holds the new tokens at the positions {positions}')
        return Caching.Caching.template(
            tuple(array.shape()), cached, name, tokens_position=positions[0],
            base=array.datatype)

    def cache_read_through(self, array: cat.Array[B, A], name: fd.DynamicName,
                           read: sc.StrideMorphism) -> cat.BroadcastedCategory[B, A]:
        '''The `Caching` of `array`, over the new tokens, onto the cached axis `read`
        reads, followed by a view of the rest of `read` where it reads more.'''
        cached = self.cached_axis_read_by(read)
        cache = self.cache_of(array, name, cached)
        rest = self.rest_after_token_part(read)
        if is_identity_read(rest):
            return cache
        return chsh.make_composed(
            cache, move_reads_backwards.view_of(cache.cod()[0], rest))

    def next_numbered_cache_name(self) -> fd.DynamicName:
        name = fd.DynamicName.from_str(f'c_{{{self.numbered_caches}}}', lineage=False)
        self.numbered_caches += 1
        return name

    def name_of_cache_after(self, operation: cat.Broadcasted[B, A]) -> fd.DynamicName:
        '''`c` with the name of the operator the cache follows as its subscript, or a
        number where the operator has no name.'''
        self.operators_before_caches.append(operation.operator)
        if operation.operator.name is None:
            return self.next_numbered_cache_name()
        return fd.DynamicName.from_str(
            f'c_{{{operation.operator.name.to_text()}}}', lineage=False)

    def name_of_cache_after_a_copy(self) -> fd.DynamicName:
        self.operators_before_caches.append(None)
        return self.next_numbered_cache_name()

    def is_computed_over_the_cache(self, target: cat.Broadcasted[B, A]) -> bool:
        return (self.inside_a_box_computed_over_the_cache
                or target.operator in self.computed_over_the_cache
                or not target.input_weaves)

    def root_processor(self, target: cat.Broadcasted[B, A], guide: Sequence[Read]
                       ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        reads = tuple(guide)
        if not isinstance(target, cat.Broadcasted):
            return self.through_tape_seed(target, reads)
        reads_a_cached_axis = any(self.cached_axis_read_by(read) is not None
                                  for read in reads)
        if reads_a_cached_axis and not self.is_computed_over_the_cache(target):
            return self.computed_at_the_new_tokens_and_cached(target, reads)
        if (reads_a_cached_axis and isinstance(target.operator, ops.BlockOperator)
                and not self.inside_a_box_computed_over_the_cache):
            self.inside_a_box_computed_over_the_cache = True
            try:
                rebuilt, operand_reads = super().root_processor(target, reads)
            finally:
                self.inside_a_box_computed_over_the_cache = False
        else:
            rebuilt, operand_reads = super().root_processor(target, reads)
        return self.cache_where_reads_reach_earlier_tokens(target, rebuilt, operand_reads)

    def computed_at_the_new_tokens_and_cached(
            self, target: cat.Broadcasted[B, A], reads: fd.Prod[Read]
            ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''`target` computed at the new tokens and followed by a `Caching` of every
        result read at a cached axis.'''
        cached = tuple(self.cached_axis_read_by(read) for read in reads)
        new_reads = tuple(self.new_token_read(array) if axis is not None else read
                          for array, read, axis in zip(target.cod(), reads, cached))
        rebuilt, operand_reads = self.root_processor(target, new_reads)
        followers = tuple(
            self.cache_read_through(array, self.name_of_cache_after(target), read)
            if axis is not None else cat.ProdObject((array,)).identity()
            for array, read, axis in zip(rebuilt.cod(), reads, cached))
        return (chsh.make_composed(rebuilt, chsh.make_product(*followers)), operand_reads)

    def cache_where_reads_reach_earlier_tokens(
            self, target: cat.Broadcasted[B, A], rebuilt: cat.BroadcastedCategory[B, A],
            operand_reads: Sequence[Read]
            ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''`rebuilt` read from every operand whose read reaches earlier tokens through a
        view of that read onto the cached axis its reach calls for, with the operand
        read at that axis.'''
        views: list[cat.BroadcastedCategory[B, A]] = []
        reads: list[Read] = []
        for array, read in zip(target.dom(), operand_reads):
            if read is None:
                if self.token_positions(array):
                    raise ReadsTheTokenAxisWhole(
                        f'{type(target.operator).__name__} '
                        f'{target.operator.name} reads every token of {array}, and '
                        'no causal read bounds which tokens it reads')
                views.append(cat.ProdObject((array,)).identity())
                reads.append(None)
            elif (not self.token_rows(read) or self.reads_new_tokens_alone(read)
                  or self.cached_axis_read_by(read) is not None):
                views.append(cat.ProdObject(
                    (move_reads_backwards.read_domain_array(array, read),)).identity())
                reads.append(read)
            elif self.reads_earlier_tokens(read):
                cached = self.cached_axis_for(self.reach(read))
                axis, shift = ((self.new_tokens, self.past_tokens.local_size())
                               if cached is None else (cached, self.shift_of(cached)))
                views.append(move_reads_backwards.view_of(
                    self.array_over(array, axis), self.rebased(read, axis, shift)))
                reads.append(self.new_token_read(array) if cached is None
                             else self.cached_read(array, cached))
            else:
                raise ReadsALaterToken(
                    f'{type(target.operator).__name__} {target.operator.name} reads '
                    f'{array} through rows {self.token_rows(read)} of '
                    f'{read._dom}, which reach a token after |P| + i_n')
        if all(move_reads_backwards.same_read(before, after)
               for before, after in zip(operand_reads, reads)):
            return rebuilt, operand_reads
        return (chsh.make_composed(chsh.make_product(*views), rebuilt), tuple(reads))

    def tape_seed_read_at_the_new_tokens[S: Para.ParaMorphism](self, seed: S) -> S:
        '''`seed` carrying its array at the new tokens where the array holds the token
        axis.'''
        if not self.token_positions(seed.size):
            return seed
        return seed.reconstruct(size=self.new_token_array(seed.size))

    def through_tape_seed(self, seed: Para.ParaMorphism[cat.Array[B, A]],
                          reads: Sequence[Read]
                          ) -> tuple[Para.ParaMorphism[cat.Array[B, A]], Sequence[Read]]:
        '''`seed` carrying its array at the new tokens where the array holds the token
        axis, as the seeds of a wrap do. A grab is required to be read at the new
        tokens, and a drop reads its operand there.'''
        for read in reads:
            self.require_the_new_tokens_at_a_grab(seed.tape, read)
        return (self.tape_seed_read_at_the_new_tokens(seed),
                tuple(self.new_token_read(array) if self.token_positions(array) else None
                      for array in seed.dom()))

    def require_the_new_tokens_at_a_grab(self, grab: object, read: Read) -> None:
        if read is not None and not self.reads_new_tokens_alone(read):
            raise TapeCarriesAnotherDemand(
                f'the grab of {grab} is read through {read}, where the drop of the slot '
                'writes the new tokens')

    def through_para_block(self, operator: ParaBlockOperator.ParaBlockOperator,
                           target_reads: Sequence[Read]
                           ) -> tuple[ParaBlockOperator.ParaBlockOperator, Sequence[Read]]:
        '''The body of a box holding tape seeds, crawled with its seeds as ports, and
        the box rebuilt with every seed carrying its array at the new tokens.'''
        exposed = ParaBlockOperator.expose_tape_as_ports(operator.block)
        ported, body_reads = self.propagate_category(exposed.block, target_reads)
        grabs = tuple(map(self.tape_seed_read_at_the_new_tokens, exposed.grabs))
        drops = tuple(map(self.tape_seed_read_at_the_new_tokens, exposed.drops))
        for grab, read in zip(grabs, body_reads):
            self.require_the_new_tokens_at_a_grab(grab.tape, read)
        own_domain = tuple(ported.dom())[len(grabs):]
        own_codomain = tuple(ported.cod())[:len(ported.cod()) - len(drops)]
        seeded = chsh.make_composed(
            chsh.make_product(*grabs, cat.ProdObject(own_domain).identity()),
            ported.body,
            chsh.make_product(cat.ProdObject(own_codomain).identity(), *drops))
        return (operator.reconstruct(block=ported.reconstruct(body=seeded),
                                     grabs=grabs, drops=drops),
                body_reads)

    def through_box(self, target: cat.Broadcasted[B, A], reads: Sequence[Read]
                    ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        if not isinstance(target.operator, ParaBlockOperator.ParaBlockOperator):
            return super().through_box(target, reads)
        splits = self.splits_agreeing_on_the_degree(target, reads, True)
        if splits is None:
            return self.stopped(target, reads)
        operator, body_reads = self.through_para_block(
            target.operator, tuple(split.target_read for split in splits))
        carried = tuple(
            move_reads_backwards.carry_through_operand(
                splits[0].degree_read, reindexing, weave, body_read)
            for reindexing, weave, body_read
            in zip(target.reindexings, target.input_weaves, body_reads))
        return (self.rebuilt(target, splits, carried, operator),
                tuple(carry.read for carry in carried))

    def through_para_wrap(self, target: para_wrap.ParaWrap, guide: Sequence[Read]
                          ) -> tuple[para_wrap.ParaWrap, Sequence[Read]]:
        '''The body of a wrap crawled with every dropped result read at the new tokens,
        and every grabbed operand required to be read there, since the drop of a slot is
        what its grab reads.'''
        wire_reads = iter(guide)
        body_guide = tuple(
            next(wire_reads) if Para.is_kept(drop)
            else self.new_token_read(array) if self.token_positions(array) else None
            for array, drop in zip(target.body.cod(), target.drops))
        body, body_reads = self.propagate_category(target.body, body_guide)
        for grab, read in zip(target.grabs, body_reads):
            if Para.grabbed_entry(grab) is not None:
                self.require_the_new_tokens_at_a_grab(grab, read)
        return (target.reconstruct(body=body),
                tuple(read for read, grab in zip(body_reads, target.grabs)
                      if Para.is_kept(grab)))

    def propagate_category(self, target: cat.BroadcastedCategory[B, A],
                           guide: Sequence[Read]
                           ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        if isinstance(target, para_wrap.ParaWrap):
            return self.through_para_wrap(target, guide)
        if isinstance(target, cat.Block) and target.block_tag.repetition != nm.Integer(1):
            body, body_guide = self.propagate_category(target.body, guide)
            if not all(same_demand(one, other) for one, other in zip(guide, body_guide)):
                raise LoopChangesTheDemand(
                    f'the block {target.block_tag.aesthetics} returns another read on '
                    'its domain than the read on its codomain')
            return cat.Block(body=body, block_tag=target.block_tag), body_guide
        return super().propagate_category(target, guide)

    def propagate_rearrangement(self, target: cat.Rearrangement[cat.Array[B, A]],
                                guide: Sequence[Read]
                                ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''A wire whose every branch asks for one read carries it back. The branches of a
        wire that ask for one read of a cached axis, beside branches asking for another
        read, are fed by one `Caching` of the new tokens, placed after the copy and
        copied to each of them.'''
        reads = tuple(guide)
        branches_of_wire = util.Multidict(
            (wire, branch) for branch, wire in enumerate(target.mapping))
        groups: list[tuple[int, int]] = []
        representative: dict[int, int] = {}
        for branch, wire in enumerate(target.mapping):
            read = reads[branch]
            if (self.cached_axis_read_by(read) is None
                    or all(move_reads_backwards.same_read(reads[other], read)
                           for other in branches_of_wire[wire])):
                continue
            first = next((first for group_wire, first in groups
                          if group_wire == wire
                          and move_reads_backwards.same_read(reads[first], read)), None)
            if first is None:
                groups.append((wire, branch))
                first = branch
            representative[branch] = first
        if not representative:
            return super().propagate_rearrangement(target, reads)
        kept = tuple(branch for branch in range(len(target.mapping))
                     if representative.get(branch, branch) == branch)
        reduced = cat.Rearrangement(
            mapping=tuple(target.mapping[branch] for branch in kept), _dom=target._dom)
        copied, domain_reads = super().propagate_rearrangement(reduced, tuple(
            self.new_token_read(target.cod()[branch]) if branch in representative
            else reads[branch]
            for branch in kept))
        followers = chsh.make_product(*(
            self.cache_read_through(array, self.name_of_cache_after_a_copy(),
                                    reads[branch])
            if branch in representative else cat.ProdObject((array,)).identity()
            for branch, array in zip(kept, copied.cod())))
        spread = cat.Rearrangement(
            mapping=tuple(kept.index(representative.get(branch, branch))
                          for branch in range(len(target.mapping))),
            _dom=tuple(followers.cod()))
        return chsh.make_composed(copied, followers, spread), domain_reads


@dataclass(frozen=True)
class CachedPass[B: cat.Datatype, A: cat.Axis]:
    '''The pass of a model over the new tokens `n`, with every array it reads at
    earlier tokens loaded through a `Caching` onto a cached axis: every earlier token
    followed by the new ones, `cached_tokens`, or the kept earlier tokens a reach calls
    for followed by the new ones, among `kept_tokens`. It also records the read that
    reached each domain array of the model, the operators computed at every cached token,
    and the operator of the model each cache follows, or `None` for a cache after a
    copy, in the order the crawl placed them.'''
    expression: cat.BroadcastedCategory[B, A]
    cached_tokens: AxisConcatenation.ConcatenatedAxis
    kept_tokens: fd.Prod[AxisConcatenation.ConcatenatedAxis]
    domain_reads: fd.Prod[Read]
    computed_over_the_cache: frozenset[cat.Operator]
    operators_before_caches: fd.Prod[cat.Operator | None]

    def caches(self) -> fd.Prod[cat.Broadcasted[B, A, Caching.Caching]]:
        return tuple(
            operation for operation in tutil.type_search(cat.Broadcasted, self.expression)
            if isinstance(operation.operator, Caching.Caching))


def with_numeric_replaced[T: fd.GeneralTerm](target: T, numeric: nm.Numeric,
                                             replacement: nm.Numeric,
                                             memo: dict[int, tuple[object, object]]) -> T:
    '''`target` with every occurrence of `numeric` replaced by `replacement`, each
    shared part visited once.'''
    if id(target) in memo:
        return memo[id(target)][1]
    if isinstance(target, nm.Numeric):
        result = nm.substitute_subterm(target, numeric, replacement)
    elif isinstance(target, (fd.Term, tuple)):
        result = fd.deep_reconstruct(
            target, lambda part: with_numeric_replaced(part, numeric, replacement, memo))
    else:
        result = target
    memo[id(target)] = (target, result)
    return result


def with_the_token_axis_cached[B: cat.Datatype, A: cat.Axis](
        expression: cat.BroadcastedCategory[B, A], tokens: A,
        cached_tokens: AxisConcatenation.ConcatenatedAxis
        ) -> cat.BroadcastedCategory[B, A]:
    '''`expression` with the axis `tokens` replaced by `cached_tokens`, and its size by
    the size of `cached_tokens`.'''
    context = fd.Context()
    context.append_bucket(fd.EqualityClass.template(tokens, priority=0).merge(
        fd.EqualityClass.template(cached_tokens, priority=1)))
    return with_numeric_replaced(context.apply(expression), tokens.local_size(),
                                 cached_tokens.local_size(), {})


def derive_cached_pass[B: cat.Datatype, A: cat.Axis](
        model: cat.BroadcastedCategory[B, A], tokens: A, past_tokens: A, new_tokens: A,
        computed_over_the_cache: Iterable[cat.Operator] = ()) -> CachedPass[B, A]:
    '''The pass of `model` that computes its results for the new tokens `new_tokens`
    after the earlier tokens `past_tokens`, with every result read at `|P| + i_n` on
    the axis `tokens`.'''
    crawl = NewTokenCrawler[B, A](
        tokens=tokens, past_tokens=past_tokens, new_tokens=new_tokens,
        cached_tokens=Caching.cached_token_axis(past_tokens, new_tokens),
        computed_over_the_cache=frozenset(computed_over_the_cache))
    codomain_reads = tuple(
        crawl.new_token_read(array) if crawl.token_positions(array) else None
        for array in model.cod())
    expression, domain_reads = crawl.propagate_category(model, codomain_reads)
    for array, read in zip(model.dom(), domain_reads):
        if crawl.token_positions(array) and (
                read is None or (not crawl.reads_new_tokens_alone(read)
                                 and crawl.cached_axis_read_by(read) is None)):
            raise ReadsALaterToken(
                f'the model reads {array} through {read}, which is neither the read of '
                'the new tokens nor the read of a cached axis')
    if any(crawl.cached_axis_read_by(read) is not None for read in domain_reads):
        expression = chsh.make_composed(chsh.make_product(*(
            crawl.cache_read_through(crawl.new_token_array(array),
                                     crawl.name_of_cache_after_a_copy(), read)
            if crawl.cached_axis_read_by(read) is not None
            else cat.ProdObject((
                array if read is None
                else move_reads_backwards.read_domain_array(array, read),)).identity()
            for array, read in zip(model.dom(), domain_reads))), expression)
    return CachedPass(
        with_the_token_axis_cached(expression, tokens, crawl.cached_tokens),
        crawl.cached_tokens, tuple(crawl.kept_tokens.values()), tuple(domain_reads),
        crawl.computed_over_the_cache, tuple(crawl.operators_before_caches))
