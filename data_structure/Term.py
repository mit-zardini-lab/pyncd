'''Terms, identities, and the machinery that rewrites one term into another.

A term is a frozen dataclass whose fields are its parts. Every object in the package
is one. A `UTerm` carries a `UID` as well, naming a degree of freedom that a rewrite
can substitute for another.

The motivation and the mathematics are in `obsidian/01-foundations/Terms.md`,
`obsidian/01-foundations/UIDs and Names.md` and
`obsidian/01-foundations/Rewriting.md`.
'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Self,
    Type,
    TypeVar,
    Callable,
    Iterable,
    overload,
    Sequence,
    Iterator,
    Hashable,
    TypedDict,
    Protocol,
)
import hashlib
import random
import re
import math
from abc import ABC
from enum import Enum
import functools

import utilities.utilities as util
from functools import cached_property

T = TypeVar('T', covariant=True)

type Prod[T] = tuple[T, ...]
'''An immutable sequence of `T`. Every sequence held inside a `Term` has this type.'''

# Keyed by qualified class name, and read by `data_transfer/term_json.py` to rebuild
# a term from JSON.
TermDirectory: dict[str, Type[Term]] = {}
EnumDirectory: dict[str, Type[Enum]] = {}

def register_enum(cls: Type[Enum]) -> Type[Enum]:
    assert cls.__qualname__ not in EnumDirectory, f"Enum class {cls.__qualname__} already registered."
    EnumDirectory[cls.__qualname__] = cls
    return cls

def _structural_hash(self) -> int:
    '''
    The hash `@dataclass(frozen=True)` would generate, computed once per term.

    Terms form a directed acyclic graph with heavy sharing, and a structural hash
    walks that graph as a tree. A subterm reachable by n paths is hashed n times,
    so the uncached cost counts paths rather than nodes. Fusing attention made 8.3
    million calls to `hash` over a result holding 35.6 thousand distinct objects.

    Caching is sound because terms are frozen. Nothing in the package mutates one
    after construction. The cached value lives in the instance `__dict__`, outside
    `__dataclass_fields__`, so it takes no part in `__eq__` and does not change what
    a term is.
    '''
    cached = self.__dict__.get('_hash')
    if cached is None:
        cached = hash(tuple(
            getattr(self, name) for name in self.__dataclass_fields__))
        object.__setattr__(self, '_hash', cached)
    return cached

@dataclass(frozen=True)
class Term:
    '''An element of the package's formal language, holding its parts in its fields.

    Every property of a term is derivable from its fields, and nothing is evaluated
    at construction time. `obsidian/01-foundations/Terms.md` gives the construction
    rule this expresses.
    '''
    # A subclass opts out of the memoised hash by setting `_memoize_hash = False` in
    # its own body. No subclass does at present. The flag is left unannotated because
    # an annotated ClassVar still lands in __dataclass_fields__, which keys(), dict()
    # and reconstruct() all iterate over.
    _memoize_hash = True

    def __init_subclass__(cls) -> None:
        assert cls.__qualname__ not in TermDirectory, f"Term class {cls.__qualname__} already registered."
        TermDirectory[cls.__qualname__] = cls
        # __init_subclass__ runs before @dataclass runs on the subclass, and the
        # decorator leaves __hash__ alone when the class already defines one in its
        # own __dict__. Assigning here therefore installs the memoised hash on every
        # Term without editing any of the fifty or so declaration sites.
        if cls._memoize_hash:
            cls.__hash__ = _structural_hash
        return super().__init_subclass__()
    def keys(self) -> Iterable[str]:
        return (f.name for f in self.__dataclass_fields__.values())
    def dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}
    def reconstruct(self, **kwargs: Any) -> Self:
        return type(self)(**{**self.dict(), **kwargs})
    def __getstate__(self) -> dict[str, Any]:
        '''The state pickle records, which is every field and not the cached hash,
        because a hash computed in one process is wrong in another.'''
        return {key: value for key, value in self.__dict__.items() if key != '_hash'}

IDMAX = 2**31-1
type IDType = int
def fresh_id() -> IDType:
    return random.randint(1, IDMAX)
def hash_id(obj: Hashable) -> IDType:
    '''An id derived from `obj`, so that equal objects give the same id.

    `nm.FreeNumeric.named` uses it to make a symbol whose id is stable across calls
    and across processes. The id is a digest of `repr(obj)` rather than `hash(obj)`,
    because Python salts the hash of a string differently in every process, and a
    symbol named in a worker process has to be the symbol named in the parent.
    Every other id in the package comes from `fresh_id` and is random per process.
    '''
    digest = hashlib.blake2b(repr(obj).encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % IDMAX

IDENTIFIER_SEPARATOR = '_'
NON_IDENTIFIER_CHARACTERS = re.compile(r'[^0-9A-Za-z]+')


def identifier_from(text: str) -> str:
    '''`text` with every run of characters outside `[0-9A-Za-z]` written as one
    underscore and the ends trimmed, so `k/e` reads `k_e` and `:512` reads `512`.'''
    return NON_IDENTIFIER_CHARACTERS.sub(IDENTIFIER_SEPARATOR, text).strip(
        IDENTIFIER_SEPARATOR)


def join_code_forms(*parts: str | None) -> str | None:
    '''The parts that are not `None` or empty, joined by underscores, or `None`
    when there are none.'''
    kept = [part for part in parts if part]
    return IDENTIFIER_SEPARATOR.join(kept) if kept else None


class ExponentPlacement(Enum):
    '''Where the exponent of a name is drawn. `SUPERSCRIPT` raises it after the
    name, `m^{5120}`. `SUBSCRIPT` lowers it into the subscript, `m_{5120}`, after a
    colon where the name already has one, `q_{Td:512}`, and outside the absolute
    bars where the settings draw them, `|k|_{6}`.'''
    SUPERSCRIPT = 'SUPERSCRIPT'
    SUBSCRIPT = 'SUBSCRIPT'
register_enum(ExponentPlacement)


@dataclass(frozen=True)
class DynamicNameSettings(Term):
    bold: bool = False
    overline: bool = False
    absolute: bool = False
    typewriter: bool = False
    exponent_placement: ExponentPlacement = ExponentPlacement.SUPERSCRIPT
@dataclass(frozen=True)
class DynamicName(Term):
    '''A name for display: a body, an optional subscript, display settings, an
    optional code form and an optional exponent.

    A name carries no identity. It decorates the `UID` it sits on.

    The subscript is itself a `DynamicName`, so subscripts chain. `to_latex` flattens
    a chain into one subscript group, writing `x_{yz}` rather than `x_{y_{z}}`.

    The code form is the identifier the whole name stands for in generated code,
    so the query axis drawn `q` can carry `query_axis` and its size `query_axis_size`.
    `add_subscript` and `subscript_target` compose it: a name whose lineage carries
    a code form anywhere composes one for the result, each name in the lineage
    contributing its code form where it has one and its bodies as an identifier
    where it does not, and a lineage carrying none composes none. A name built
    directly with a subscript states its own code form.

    The exponent is a name drawn above the body, after the subscript, so an axis
    named `q_{d}` with the exponent `512` draws as `q_{d}^{512}`. The settings
    may lower it into the subscript instead, through `ExponentPlacement.SUBSCRIPT`,
    so the same axis draws as `q_{d:512}` and an axis `m` sized 5120 draws as
    `m_{5120}`. It is left out of `to_bodies`, so a lookup keyed by bodies is
    unchanged by it, and `to_text` writes it after a caret, or after a colon where
    it is lowered. `obsidian/01-foundations/Code Forms.md` states both slots and
    how to give a module's names them.
    '''
    body: None | str = None
    subscript: None | DynamicName = None
    settings: None | DynamicNameSettings = None
    code_form: None | str = None
    exponent: None | DynamicName = None

    def __lt__(self, other: DynamicName) -> bool:
        return (
            (self.body or '') < (other.body or '')
            or (
                other.subscript is not None
                and (self.subscript is None or self.subscript < other.subscript)
            )
        )

    def lineage(self) -> Prod[DynamicName]:
        '''This name followed by each name down its subscript chain.

        A name with no body has an empty lineage, which drops it and everything
        below it from the display.
        '''
        if self.body is None:
            return ()
        if self.subscript is None:
            return (self,)
        return (self, *self.subscript.lineage())

    def capture[S:Term](self, target: S) -> S:
        '''Return `target` with this name on its uid. `target` must be a `UTerm`.'''
        return target.reconstruct(
            uid=target.uid.reconstruct( # type: ignore
                _name=self
            )
        )

    def to_bodies(self) -> str:
        '''The bodies down the lineage, joined, with no latex, no settings and no
        exponent.'''
        return ''.join(
            b.body or ''
            for b in self.lineage()
        )

    def to_text(self) -> str:
        '''The bodies down the lineage, and the exponent's bodies after a caret
        where there is one, so a sized axis reads `qd^512` in a listing, or
        after a colon where the exponent is lowered into the subscript, `qd:512`.'''
        if self.exponent is None:
            return self.to_bodies()
        separator = ':' if self.exponent_placement() is ExponentPlacement.SUBSCRIPT else '^'
        return f'{self.to_bodies()}{separator}{self.exponent.to_bodies()}'

    def exponent_placement(self) -> ExponentPlacement:
        '''Where the exponent is drawn, which is above the name until the settings
        lower it.'''
        if self.settings is None:
            return ExponentPlacement.SUPERSCRIPT
        return self.settings.exponent_placement

    def with_exponent_placement(self, placement: ExponentPlacement) -> DynamicName:
        '''This name with its exponent drawn where `placement` puts it, and this
        name as it stands where it is drawn there already, so a name without
        settings gains none for the default placement.'''
        if placement is self.exponent_placement():
            return self
        settings = self.settings if self.settings is not None else DynamicNameSettings()
        return self.reconstruct(settings=settings.reconstruct(exponent_placement=placement))

    def to_code_form(self) -> str | None:
        return self.code_form

    def code_form_or_identifier(self) -> str:
        '''The code form, or the bodies down the lineage as an identifier where
        there is none. `add_subscript` composes code forms from it.'''
        if self.code_form is not None:
            return self.code_form
        return identifier_from(self.to_bodies())

    def with_code_form(self, code_form: str | None) -> DynamicName:
        return self.reconstruct(code_form=code_form)

    def code_form_suffixed(self, suffix: str) -> DynamicName:
        '''This name with `suffix` joined onto its code form, and this name as it
        stands where it carries no code form, so the size of `query_axis` reads
        `query_axis_size` and the size of an axis with no code form has none.'''
        if self.code_form is None:
            return self
        return self.with_code_form(join_code_forms(self.code_form, suffix))

    def with_exponent(self, exponent: DynamicName | str | int | None) -> DynamicName:
        match exponent:
            case None | DynamicName():
                return self.reconstruct(exponent=exponent)
            case _:
                return self.reconstruct(exponent=DynamicName(body=str(exponent)))

    def body_latex(self) -> str:
        body = self.body or ''
        if self.settings is None:
            return body
        if self.settings.typewriter:
            body = f'\\texttt{{{body.replace("_", chr(92) + "_")}}}'
        if self.settings.bold:
            body = f'\\bold{{{body}}}'
        if self.settings.overline:
            body = f'\\overline{{{body}}}'
        return body

    def subscript_group_latex(self) -> str:
        '''The subscript chain as the content of one group, or nothing.'''
        if self.subscript is None:
            return ''
        return ''.join(b.body_latex() for b in self.subscript.lineage())

    def exponent_group_latex(self) -> str:
        '''The exponent's lineage as the content of one group, or nothing.'''
        if self.exponent is None:
            return ''
        return ''.join(b.body_latex() for b in self.exponent.lineage())

    def exponent_latex(self) -> str:
        '''The exponent's lineage as one superscript group, as one subscript group
        where the settings lower it, or nothing.'''
        group = self.exponent_group_latex()
        if not group:
            return ''
        script = '_' if self.exponent_placement() is ExponentPlacement.SUBSCRIPT else '^'
        return f'{script}{{{group}}}'

    def to_latex(self) -> str:
        '''The name as latex: the body, the subscript chain after an underscore,
        the absolute bars where the settings ask for them, and the exponent last.

        The subscript chain is written without braces, `q_Td`, because the
        listings of the kernelization notebooks print box titles in this form and
        compare them. tsncd writes its own braces when it draws the name.

        The exponent stands outside the bars, so a size symbol assigned 6 reads
        `|k|^{6}` and the bars enclose the letter they measure. An exponent lowered
        into the subscript of a name that has one and no bars joins that subscript
        after a colon, `q_Td:512`, and otherwise stands where the raised one
        would, `m_{5120}` and `|k|_{6}`.
        `obsidian/05-backends/Compound Axis Labels.md` states the rule for every
        label holding more than one symbol.
        '''
        subscript = self.subscript_group_latex()
        exponent = self.exponent_group_latex()
        absolute = self.settings is not None and self.settings.absolute
        lowered = self.exponent_placement() is ExponentPlacement.SUBSCRIPT
        if subscript and exponent and lowered and not absolute:
            return f'{self.body_latex()}_{subscript}:{exponent}'
        latex = self.body_latex() + (f'_{subscript}' if subscript else '')
        if absolute:
            latex = f'|{latex}|'
        return latex + self.exponent_latex()

    def without_absolute_bars(self) -> DynamicName:
        '''This name with the absolute bars dropped and every other setting kept,
        so the size symbol `|k|` reads as the letter an axis of that size takes.'''
        if self.settings is None or not self.settings.absolute:
            return self
        return self.reconstruct(settings=self.settings.reconstruct(absolute=False))

    def _composed_code_form(self, other: DynamicName) -> str | None:
        '''The code form of this name with `other` hung off its subscript chain:
        `None` where neither carries one, and otherwise the two joined, each
        contributing its identifier.'''
        if self.code_form is None and other.code_form is None:
            return None
        return join_code_forms(
            self.code_form_or_identifier(), other.code_form_or_identifier())

    def subscript_target(self, target: DynamicName) -> DynamicName:
        '''Return `target` with this whole name hung off the end of its subscript chain.

        `A_B` applied to `C_D` gives `C_{D_{A_B}}`, which `to_latex` writes as
        `C_{DAB}`. Every name in the result keeps the settings it arrived with,
        and the outermost keeps its exponent and composes its code form.
        '''
        return self._hung_under(target).reconstruct(
            code_form=target._composed_code_form(self))

    def _hung_under(self, target: DynamicName) -> DynamicName:
        '''`target` with this name at the end of its subscript chain, every name
        of the chain keeping its own settings, code form and exponent.'''
        return target.reconstruct(
            subscript=(
                self
                if target.subscript is None
                else self._hung_under(target.subscript)
            ),
        )

    def subscript_capture[S:UTerm](self, target: S) -> S:
        '''Subscript this name onto `target`'s own name. An unnamed `target` is
        returned unchanged, because there is nothing to subscript onto.'''
        if (old_name := target.uid._name) is None:
            return target
        new_name = self.subscript_target(old_name)
        return new_name.capture(target)

    def add_subscript(self, other: DynamicName | str | None) -> DynamicName:
        '''Return this name with `other` hung off the end of its subscript chain.

        The argument order is the reverse of `subscript_target`. Every name above
        `other` is rebuilt without its settings. The outermost name keeps its
        exponent and composes its code form with `other`'s, and every inner name
        keeps its own.
        '''
        if not other:
            return self
        other = other if isinstance(other, DynamicName) else DynamicName(body=other)
        return self._with_subscript_added(other).reconstruct(
            settings=None,
            code_form=self._composed_code_form(other),
            exponent=self.exponent,
        )

    def _with_subscript_added(self, other: DynamicName) -> DynamicName:
        return DynamicName(
            body=self.body,
            subscript=(
                other if self.subscript is None
                else self.subscript._with_subscript_added(other)
            ),
            code_form=self.code_form,
            exponent=self.exponent,
        )

    @overload
    @classmethod
    def from_str(cls,
            name: None,
            lineage: bool = True,
            settings: None | DynamicNameSettings = None,
            code_form: str | None = None) -> None: ...
    @overload
    @classmethod
    def from_str(cls,
                 name: DynamicName | str,
                 lineage: bool = True,
                 settings: None | DynamicNameSettings = None,
                 code_form: str | None = None) -> DynamicName: ...

    @classmethod
    def from_str(cls,
            name: DynamicName | str | None,
            lineage: bool = True,
            settings: None | DynamicNameSettings = None,
            code_form: str | None = None) -> None | DynamicName:
        '''Read a name from a string.

        `None` and a `DynamicName` are returned unchanged, and `settings` and
        `code_form` are ignored for both. With `lineage=True` the string splits at
        its first underscore and the remainder becomes the subscript, so `'x_y_z'`
        reads as `x_{y_{z}}`. With `lineage=False` the whole string is one body.
        `settings` and `code_form` go on the outermost name alone.
        '''
        match name:
            case None | DynamicName():
                return name
            case str() if not lineage:
                return DynamicName(body=name, settings=settings, code_form=code_form)
            case str() if lineage:
                body, *subscript_parts = name.split('_', 1)
                subscript = cls.from_str(
                    subscript_parts[0]
                    if subscript_parts else None,
                    lineage=True,
                    settings=None)
                return DynamicName(
                    body=body,
                    subscript=subscript,
                    settings=settings,
                    code_form=code_form,
                )

@dataclass(frozen=True)
class UID[T:Term](Term):
    '''The identity of a `UTerm`, independent of the fields the term carries.

    Each `UTerm` with a uid is a degree of freedom in an expression. Substituting one
    for another is how axes are aligned, how a block is tracked across a rewrite, and
    how a kernelized axis is related to the axis it subdivides.

    Two terms with the same uid denote the same degree of freedom and can still carry
    different fields, because `reconstruct` keeps the uid and replaces the fields.
    Subordination depends on exactly that, to align an axis with its subordinated
    form. Equality and hashing stay structural, over every field including the uid.
    Comparing uids alone breaks fusion without raising.

    `_id` is random per process, so anything that iterates over a set of terms must
    sort by it to be deterministic.
    '''
    _type: Type[T]
    _id: IDType = field(default_factory=fresh_id)
    _name: DynamicName | None = None

    def __lt__(self, other: UID[T]) -> bool:
        '''Order by name, and by `_id` between two uids that are both unnamed.

        An unnamed uid sorts below a named one, so the `max` in
        `EqualityClass.from_iter` picks a named term as the canonical one.
        '''
        match self._name, other._name:
            case None, None:
                return self._id < other._id
            case None, DynamicName():
                return True
            case DynamicName(), None:
                return False
            case DynamicName(), DynamicName():
                return self._name < other._name
    
    @classmethod
    def field(cls, _type: Type[T]) -> Any:
        return field(default_factory=lambda: cls(_type))

    def to_latex(self) -> str:
        return self._name.to_latex() if self._name is not None else f'\\text{{{self._type.__qualname__}}}_{{{self._id}}}'

@dataclass(frozen=True)
class UTerm(Term):
    '''A `Term` that carries an identity as well as its fields.'''
    uid: UID[Self] = field(default_factory=lambda: UID(UTerm)) # type: ignore
    def __init_subclass__(cls) -> None:
        # @dataclass binds the default factory into the generated __init__ when it
        # runs on cls, which is after __init_subclass__. Rewriting the inherited
        # field here therefore gives each subclass a uid typed to itself, even
        # though every subclass shares this one Field object.
        cls.__dataclass_fields__['uid'].default_factory = lambda: UID(cls)
        return super().__init_subclass__()

type GeneralTerm = Term | Prod[GeneralTerm]
@overload
def deep_reconstruct[T](target: T, func: Callable[[T], T]) -> T: ...
@overload
def deep_reconstruct[T](target: Prod[T], func: Callable[[T], T]) -> Prod[T]: ...
def deep_reconstruct(target, func):
    '''
    Rebuild `target` with `func` applied to each of its parts.

    A term whose parts all come back as the same objects is returned unchanged.
    Returning the original preserves the sharing that rebuilding would destroy. A
    rewrite that rebuilds unconditionally turns the directed acyclic graph into a
    tree, and every later pass then walks the term once per path through it.
    '''
    match target:
        case Term():
            values: dict[str, Any] = {}
            changed = False
            for name in target.keys():
                old = getattr(target, name)
                new = func(old)
                changed = changed or new is not old
                values[name] = new
            return type(target)(**values) if changed else target
        case tuple():
            items = tuple(func(item) for item in target)
            return items if any(
                new is not old for new, old in zip(items, target)) else target
        case _:
            return target


class UIDEquipped(Protocol):
    uid: UID
@dataclass
class EqualityClass[T:UTerm]:
    '''A declaration that every uid in `bucket` denotes the term `canonical`.

    Applying the class to a term replaces any `UTerm` whose uid is in the bucket with
    `canonical`. Every structural change in the package is made this way. Merging two
    overlapping classes keeps the canonical term of the one with the higher
    `priority`.
    '''
    _type: Type[T]
    bucket: set[UID[T]]
    canonical: T
    priority: int = 0

    def apply[S: GeneralTerm](self, target: S, iterate: bool = True) -> S:
        '''Rewrite `target` under this class. With `iterate=False`, rewrite only the
        term itself and leave its parts alone.'''
        match target:
            case Term(uid=UID() as uid) if uid in self.bucket:
                return self.canonical  # type: ignore
        return deep_reconstruct(target, self.apply) if iterate else target
    
    @classmethod
    def from_iter(cls, target: Iterable[T], priority: int = 0) -> EqualityClass[T]:
        '''Identify every term in `target` with each other.

        The canonical term is the maximum under `UID.__lt__`, which prefers a named
        uid, so a named axis survives an identification with an unnamed one. The
        class takes its `_type` from the first term and does not check that the rest
        agree.
        '''
        target = tuple(target)
        _type = type(target[0])
        return EqualityClass(
            _type=_type,
            bucket={t.uid for t in target},
            canonical=max(
                target,
                key=lambda uterm: uterm.uid
            ),
            priority=priority
        )

    def try_merge(self, other: RewriteClass[T]) -> None | EqualityClass[T]:
        '''Merge the two classes, returning None when `other` is a `UIDRenaming` or
        their buckets are disjoint.'''
        if not isinstance(other, EqualityClass) or self.bucket.isdisjoint(other.bucket):
            return None
        canonical = max(self, other, key=canonical_rank).canonical
        return EqualityClass(
            _type=self._type,
            bucket=self.bucket.union(other.bucket),
            canonical=canonical,
            priority=max(self.priority, other.priority)
        )
    
    def merge(self, other: EqualityClass[T]) -> EqualityClass[T]:
        canonical = max(self, other, key=canonical_rank).canonical
        return EqualityClass(
            _type=self._type, # type: ignore
            bucket=self.bucket.union(other.bucket), # type: ignore
            canonical=canonical,
            priority=max(self.priority, other.priority)
        )
    
    @classmethod
    def template(cls, *targets: T, priority: int = 0) -> EqualityClass[T]:
        return cls.from_iter(targets, priority=priority)
    
    @classmethod
    def set_canonical(cls, target: T, *original: T) -> EqualityClass[T]:
        '''Identify each term in `original` with `target`, whatever their uid order.'''
        _type = type(target)
        bucket = {t.uid for t in (target, *original)}
        return EqualityClass(
            _type=_type,
            bucket=bucket,
            canonical=target
        )

def canonical_rank(eq_class: EqualityClass) -> tuple[int, int, UID | int]:
    '''The order deciding which canonical a merge of two classes keeps: the higher
    priority, then a canonical with no uid of its own, which is a size written as a
    product of symbols and says more than a symbol does, then the greater uid. Two
    canonicals without a uid are ordered by the hash of their repr.'''
    canonical = eq_class.canonical
    if isinstance(canonical, UTerm):
        return (eq_class.priority, 0, canonical.uid)
    return (eq_class.priority, 1, hash_id(repr(canonical)))


@dataclass
class UIDRenaming[T: UTerm]:
    '''A declaration that every uid in `bucket` denotes the degree of freedom
    `canonical`.

    Applying the renaming to a term whose uid is in the bucket keeps the term's
    fields and replaces its uid, where `EqualityClass.apply` replaces the whole
    term. A wire splice is the renaming case: each occurrence of a
    `graphs.data_structure.Hypergraph.HypergraphObject` keeps the object it
    carries while its neighbours name the identity just spliced in.
    '''
    _type: Type[T]
    bucket: set[UID[T]]
    canonical: UID[T]
    priority: int = 0

    def apply[S: GeneralTerm](self, target: S, iterate: bool = True) -> S:
        '''Rewrite `target` under this renaming. With `iterate=False`, rewrite only
        the term itself and leave its parts alone.'''
        match target:
            case Term(uid=UID() as uid) if uid in self.bucket and uid != self.canonical:
                return target.reconstruct(uid=self.canonical)  # type: ignore
        return deep_reconstruct(target, self.apply) if iterate else target

    @classmethod
    def set_canonical(cls, target: T | UID[T], *original: T | UID[T]) -> UIDRenaming[T]:
        '''Rename each identity in `original` to `target`'s, whatever their uid
        order. Either side may be given as a term or as a bare uid.'''
        canonical = target if isinstance(target, UID) else target.uid
        return cls(
            _type=canonical._type,
            bucket={t if isinstance(t, UID) else t.uid for t in (target, *original)},
            canonical=canonical,
        )

    def try_merge(self, other: RewriteClass[T]) -> None | UIDRenaming[T]:
        '''Merge the two renamings, returning None when `other` is an
        `EqualityClass` or their buckets are disjoint.'''
        if not isinstance(other, UIDRenaming) or self.bucket.isdisjoint(other.bucket):
            return None
        canonical = max(
            self, other,
            key=lambda renaming: (renaming.priority, renaming.canonical)
        ).canonical
        return UIDRenaming(
            _type=self._type,
            bucket=self.bucket | other.bucket,
            canonical=canonical,
            priority=max(self.priority, other.priority),
        )

type RewriteClass[T: UTerm] = EqualityClass[T] | UIDRenaming[T]

@dataclass
class Context:
    '''A list of equality classes and renamings, applied together in one traversal.

    Appending a class merges it into every class it overlaps, so the list always
    holds disjoint buckets.
    '''
    equality_classes: list[RewriteClass] = field(default_factory=list)
    _apply_memo: dict[int, Any] | None = field(
        default=None, repr=False, compare=False)

    def buckets_set(self) -> set[UID]:
        return {uid for eq_class in self.equality_classes for uid in eq_class.bucket} or set()
    def __iter__(self) -> Iterator[RewriteClass]:
        return iter(self.equality_classes)
    def apply[T: GeneralTerm](self, target: T) -> T:
        '''
        Rewrite `target` under these equality classes.

        Memoized on object identity for the duration of one top-level call.
        Terms form a DAG, and without the memo a shared subterm is rebuilt once
        per path. The result also comes back with less sharing than the
        input, so every later pass has more paths to walk. Keying on identity is
        safe because everything keyed on stays reachable from `target` for the
        whole traversal, so no id is recycled while the traversal runs.
        '''
        memo, top = self._apply_memo, self._apply_memo is None
        if top:
            memo = self._apply_memo = {}
        try:
            key = id(target)  # the ORIGINAL object, before any rewriting
            if key in memo:
                return memo[key]
            rewritten = target
            match rewritten:
                case Term(uid=UID()):
                    for eq_class in self.equality_classes:
                        rewritten = eq_class.apply(rewritten, iterate=False)
            result = memo[key] = deep_reconstruct(rewritten, self.apply)
            return result
        finally:
            if top:
                self._apply_memo = None
    def __call__[T: GeneralTerm](self, target: T) -> T:
        return self.apply(target)
    
    def append_iter[T: UTerm](self, target: Iterable[T]) -> None:
        new_eq_class = EqualityClass.from_iter(target)
        self.append_bucket(new_eq_class)

    def append_bucket[T: UTerm](self, bucket: RewriteClass[T]) -> Self:
        # Merge must ACCUMULATE: a bucket can overlap several existing classes
        # (a chain of identifications arriving out of order), and each merge
        # has to fold into the union so far, and merging each against the original
        # bucket would keep only the last union and silently drop the rest.
        new_eq_class = bucket
        to_del = []
        for i, eq_class in enumerate(self.equality_classes):
            merged = eq_class.try_merge(new_eq_class)
            if merged is not None:
                new_eq_class = merged
                to_del.append(i)
        for i in reversed(to_del):
            del self.equality_classes[i]
        self.equality_classes.append(new_eq_class)
        return self

    def append_buckets[T: UTerm](self, buckets: Iterable[RewriteClass[T]]) -> Self:
        for bucket in buckets:
            self.append_bucket(bucket)
        return self

    def append_contexts[T: UTerm](self, contexts: Iterable[Context]) -> Self:
        for context in contexts:
            if self.buckets_set().isdisjoint(context.buckets_set()):
                self.equality_classes.extend(context.equality_classes)
            self.append_buckets(context.equality_classes)
        return self
