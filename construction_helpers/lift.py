'''Broadcasting over a product of axes, `>>`.

Four liftings are defined, and `dynamic_object_lift` dispatches on the pair:

    Axes           >> (Datatype | Array)   -> Array
    StrideCategory >> (Datatype | Array)   -> Broadcasted
    StrideCategory >> Broadcasted          -> Broadcasted
    Axes           >> BroadcastedCategory  -> BroadcastedCategory

Lifting an array prepends the axes to its shape. Lifting a morphism prepends a
`WeaveMode.TILED` slot per lifted axis to every weave and composes the lift into
every reindexing, so the operator is broadcast over the new axes and reads the
same target as before.

A seed that is none of the four constructions and none of `Broadcasted` carries
its objects in a field of its own, so `OBJECT_LIFTS` holds one rule per such
class and `register_object_lift` declares one. `para.registries.object_lift`
registers the tape seeds, whose array is their `size`.

`obsidian/02-categories/Construction Helpers.md` is the full account.
'''
from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import construction_helpers.product as chp
import data_structure.Operators as ops

type ObjectLift[A: cat.Axis] = Callable[
    [cat.Morphism, cat.ProdObject[A]], cat.Morphism]

OBJECT_LIFTS: dict[type[cat.Morphism], ObjectLift] = {}


class MorphismNotLiftable(TypeError):
    '''A morphism is none of the four constructions, is no `Broadcasted`, and no
    class of it has a rule in `OBJECT_LIFTS`, so nothing states how it is read
    at an index of a product of axes.'''


def register_object_lift[A: cat.Axis](
    *kinds: type[cat.Morphism]
) -> Callable[[ObjectLift[A]], ObjectLift[A]]:
    def decorate(rule: ObjectLift[A]) -> ObjectLift[A]:
        for kind in kinds:
            OBJECT_LIFTS[kind] = rule
        return rule
    return decorate


def object_lift_for[A: cat.Axis](
    morphism: cat.Morphism
) -> ObjectLift[A] | None:
    '''The rule of the most specific registered class of `morphism`, since a
    subclass of a seed has the seed's rule unless it declares its own.'''
    for kind in type(morphism).__mro__:
        if kind in OBJECT_LIFTS:
            return OBJECT_LIFTS[kind]
    return None


def object_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductObjectTarget[A]
) -> cat.ProdObject[cat.Array[B, A]]:
    base_object = chp.object_product(base, conversion=chp.datatype_converter)
    lift_by_object = chp.object_product(lift_by)
    return cat.ProdObject.from_iter(
        cat.Array(
            datatype=segment.datatype,
            _shape=(*lift_by_object, *segment._shape)
        ) for segment in base_object
    )

def object_morphism_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductMorphismTarget[A, cat.StrideCategory[A]]
) -> cat.BroadcastedCategory[B, A]:
    '''One `View` per array, reading it through the given index map.

    `View.template` returns the array's identity where that map is an identity.
    '''
    base_object = chp.object_product(base, conversion=chp.datatype_converter)
    lift_by_morphism = chp.morphism_product(lift_by)
    return chp.morphism_product(
        tuple(ops.View.template(segment, lift_by_morphism)
        for segment in base_object)
    )

def morphism_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductMorphismTarget[cat.Array[B, A], cat.Broadcasted[B, A], B],
    lift_by: chp.ProductObjectTarget[A]
) -> cat.BroadcastedCategory[B, A]:
    '''Broadcast a whole morphism over `lift_by`, descending to its leaves.

    A `Rearrangement` is rebuilt over the lifted objects and keeps its mapping,
    because the lifted axes are prepended to every object it permutes.
    '''
    base_morphism = chp.morphism_product(base, conversion=chp.datatype_converter)
    lift_by_object = chp.object_product(lift_by)
    match base_morphism:
        case cat.Block(body=body):
            return base_morphism.reconstruct(
                body=morphism_object_lift(body, lift_by_object)
            )
        case cat.Rearrangement(mapping=mapping, _dom=_dom):
            return cat.Rearrangement(
                mapping=mapping,
                _dom=tuple(object_object_lift(base_morphism.dom(), lift_by_object))
            )
        case cat.Composed(content=ms) | cat.ProductOfMorphisms(content=ms):
            return type(base_morphism).from_iter(
                morphism_object_lift(segment, lift_by_object)
                for segment in ms
            )
        case cat.Broadcasted():
            return broadcasted_stride_lift(base_morphism, lift_by_object)
    rule = object_lift_for(base_morphism)
    if rule is None:
        raise MorphismNotLiftable(
            f'a {type(base_morphism).__name__} has no rule for being read at an '
            f'index of {tuple(lift_by_object)}')
    return rule(base_morphism, lift_by_object)

def broadcasted_stride_lift[B:cat.Datatype, A:cat.Axis](
    base: cat.Broadcasted[B, A],
    lift_by: chp.ProductMorphismTarget[A, cat.StrideCategory[A]]
) -> cat.Broadcasted[B, A]:
    lift_by_morphism = chp.morphism_product(lift_by)
    input_weaves = tuple(
        cat.Weave(
            weave.datatype,
            (cat.WeaveMode.TILED,) * len(lift_by_morphism.cod()) + weave._shape
        )
        for weave in base.input_weaves
    )
    output_weaves = tuple(
        cat.Weave(
            weave.datatype,
            (cat.WeaveMode.TILED,) * len(lift_by_morphism.cod()) + weave._shape
        )
        for weave in base.output_weaves
    )
    reindexings = tuple(
        chp.morphism_product((lift_by_morphism, reindexing))
        for reindexing in base.reindexings
    )
    # A morphism with an empty domain has no reindexing to carry the new axes.
    # They go on its backup degree instead, in front, which is where the weaves
    # put them.
    backup_degree = (
        cat.ProdObject((*lift_by_morphism.dom(), *base.degree()))
        if base.has_empty_domain() else None)
    return base.reconstruct(
        input_weaves=input_weaves,
        output_weaves=output_weaves,
        reindexings=reindexings,
        backup_degree=backup_degree,
    )

@overload
def dynamic_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductObjectTarget[A]
) -> cat.ProdObject[cat.Array[B, A]]: ...
@overload
def dynamic_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: cat.StrideCategory[A]
) -> cat.BroadcastedCategory[B, A]: ...

def dynamic_object_lift[B:cat.Datatype, A:cat.Axis]( # type: ignore
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductObjectTarget[A] | cat.StrideCategory[A]
):
    base = chp.general_product(base, conversion=chp.datatype_converter) # type: ignore
    lift_by = chp.general_product(lift_by)
    match base, lift_by:
        case cat.ProdObject(), cat.ProdObject():
            return object_object_lift(base, lift_by)
        case cat.ProdObject(), cat.Morphism():
            return object_morphism_lift(base, lift_by)
        case cat.Broadcasted(), _:
            return broadcasted_stride_lift(base, lift_by)
        case cat.Morphism(), cat.ProdObject():
            return morphism_object_lift(base, lift_by)
        case _:
            raise TypeError("Invalid types for dynamic_object_lift")

cat.Morphism.__rrshift__    = morphism_object_lift # type: ignore
cat.Broadcasted.__rrshift__ = broadcasted_stride_lift # type: ignore
cat.Datatype.__rrshift__    = dynamic_object_lift # type: ignore
cat.Array.__rrshift__       = dynamic_object_lift # type: ignore
cat.ProdObject.__rrshift__  = dynamic_object_lift # type: ignore
