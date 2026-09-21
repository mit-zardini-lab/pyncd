import construction_helpers as ch # Needed for @ auto-alignment
import construction_helpers.lift as chl
from construction_helpers import simple_helper as chsh
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import advanced_axis_dynamics.data_structure.Operators as aops
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union

@dataclass(frozen=True)
class Complex[B:cat.Datatype = cat.Reals](cat.Datatype):
    base: B = field(default_factory=cat.Reals) # type: ignore

@dataclass(frozen=True)
class ComplexRotary[B:cat.Datatype](cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('Rotary')
    frequency: nm.Numeric = nm.Integer(10_000)

    @classmethod
    def template(cls, base: B = cat.Reals()):
        return cat.Broadcasted[Complex[B], cat.RawAxis](
            operator=cls(),
            input_weaves=(),
            output_weaves=(cat.Weave(Complex(base), (cat.RawAxis(), cat.RawAxis())),),
            reindexings=()
        )
    
def rotary_linear[B:cat.Datatype](
    base: B = cat.Reals(),
    frequency: nm.Numeric = nm.Integer(10_000)
):
    x_axis = cat.RawAxis()
    m_axis = cat.RawAxis()
    h_axis = cat.RawAxis()
    d_axis = cat.RawAxis()
    linear_part = cat.Broadcasted[B | Complex[B], cat.RawAxis](
        operator=ops.Linear(fd.DynamicName('R')),
        input_weaves=(cat.Weave(base, (cat.WeaveMode.TILED, m_axis)),),
        output_weaves=(cat.Weave(Complex(base), (cat.WeaveMode.TILED, h_axis, d_axis)),),
        reindexings=(cat.ProdObject((x_axis,)).identity(),)
    )
    rotary_part = cat.Broadcasted[Complex[B], cat.RawAxis](
        operator=ComplexRotary(),
        input_weaves=(),
        output_weaves=(cat.Weave(Complex(base), (x_axis, d_axis)),),
        reindexings=()
    )
    einops_part = ops.Einops.template(
        'x d/2, x h d/2 -> x h d/2',
        datatype=Complex(base),
    )
    return linear_part, rotary_part, einops_part

@dataclass(frozen=True)
class RotaryLinear[B:cat.Datatype](cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('RotaryLinear')
    frequency: nm.Numeric = nm.Integer(10_000)


CONJUGATE_NAME = fd.DynamicName(body='\\overline{x}')


def conjugate_complex_values[B: cat.Datatype, A: cat.Axis](
    array: cat.Array[Complex[B], A],
) -> cat.Broadcasted[Complex[B], A]:
    '''Every value of `array` replaced by its complex conjugate.

    A rotary table holds `e^{i angle}`, and the conjugate of a complex number of
    length one turns a point back through its own angle, so this map after a table
    gives the factor an inverse rotary embedding multiplies by. DeepSeek-V4.1-Flash
    multiplies its attention output by the conjugated factor of the query's position,
    so that its cache holds one rotated form of every latent.
    '''
    return ops.Arithmetic.template(
        nm.Conjugate(nm.x), base=array, name=CONJUGATE_NAME)


def table_over_positions_and_pairs[B: cat.Datatype, A: cat.Axis, O: Rotary](
    operator: O,
    positions: A | None,
    pairs: A | None,
    real: B,
) -> cat.Broadcasted[Complex[B], A, O]:
    '''The `cat.Broadcasted` with no operands whose one result is `Complex(real)` over
    `(positions, pairs)`, both in the target of the output weave. A tiled operator is
    one function at every index of its degree, and the table holds a different factor
    at every position and every pair. An axis left out is a fresh `cat.RawAxis`.'''
    return cat.Broadcasted(
        operator=operator,
        input_weaves=(),
        output_weaves=(cat.Weave(Complex(real), (
            positions if positions is not None else cat.RawAxis(),
            pairs if pairs is not None else cat.RawAxis())),),
        reindexings=())


@dataclass(frozen=True)
class Rotary(cat.Operator):
    '''The table of factors a rotary embedding multiplies the channel pairs by. The
    operator has no operands, and over the positions `x` and the pairs `t` its result
    holds

        F[i_x, i_t] = e^{i stride i_x theta[i_t]},    theta[i_t] = base^{-i_t / |t|}

    A pair
    is two adjacent channels read as one complex number by `PairsAsComplex`, so `|t|`
    is half the number of rotated channels and `theta[i_t]` is the number usually
    written `base^{-2 i_t / |z|}` over `|z| = 2 |t|` channels. The rotation of an array
    of pairs is an `ops.Einops` over the `Complex` datatype against this table.

    `position_stride` is the token position one step along `x` stands for. It is one
    where `x` is the token axis. A compressed entry of DeepSeek-V4.1-Flash is rotated
    at the position of the first token of its group, so the table over the entries
    `b` of groups of `|a|` tokens has the stride `|a|` and holds the factor of token
    `|a| i_b` at index `i_b`.

    `deepseek/registries/standard_expansions.py` writes the table out with
    `ops.Arrange`, `ops.Arithmetic` and `ops.multiply_by_positions`. `YarnRotary` is
    the table with the YaRN frequencies. `ComplexRotary` is the earlier table, which
    names no axis and has no expansion, and the notebooks under
    `notebooks/base_features/` hold it.

    A table turns counterclockwise. An inverse rotary embedding multiplies by the
    conjugate of this table, which `conjugate_complex_values` writes as one
    elementwise map after it.
    '''
    name: fd.DynamicName | None = fd.DynamicName('\\mathrm{RoPE}')
    base: nm.Numeric = nm.Integer(10_000)
    position_stride: nm.Numeric = nm.Integer(1)

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        positions: A | None = None,
        pairs: A | None = None,
        *,
        base: nm.Numeric = nm.Integer(10_000),
        position_stride: nm.Numeric = nm.Integer(1),
        real: B = cat.Reals(),
        name: str | fd.DynamicName | None = None,
    ) -> cat.Broadcasted[Complex[B], A, 'Rotary']:
        '''The table over `(positions, pairs)`, per `table_over_positions_and_pairs`.
        `name` defaults to the name the class declares.'''
        return table_over_positions_and_pairs(
            cls(base=base, position_stride=position_stride,
                name=table_name(cls, name)),
            positions, pairs, real)


def table_name(
    kind: type[Rotary],
    given: str | fd.DynamicName | None,
) -> fd.DynamicName:
    '''The name of a rotary table: `given` where a caller supplies one, and the name
    the class declares otherwise.'''
    if given is not None:
        return fd.DynamicName.from_str(given)
    return kind.__dataclass_fields__['name'].default


@dataclass(frozen=True)
class YarnRotary(Rotary):
    '''A `Rotary` table whose frequencies YaRN has interpolated. Over the positions `x`
    and the pairs `t` its result holds

        F[i_x, i_t] = e^{i stride i_x theta'[i_t]}
        theta'[i_t] = base^{-i_t / |t|} (1 - r[i_t] + r[i_t] / factor)
        r[i_t] = clamp((i_t - ramp_start) / (ramp_end - ramp_start), 0, 1)

    The ramp `r` is zero up to the
    pair `ramp_start`, where the frequency is the one `Rotary` has, and one from the
    pair `ramp_end`, where the frequency is divided by `factor`. The released
    DeepSeek-V4.1-Flash computes the two ends of the ramp from its configuration, and
    they are constants of the model here.
    '''
    name: fd.DynamicName | None = fd.DynamicName('\\mathrm{YaRN}')
    factor: nm.Numeric = nm.Integer(1)
    ramp_start: nm.Numeric = nm.Integer(0)
    ramp_end: nm.Numeric = nm.Integer(1)

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        positions: A | None = None,
        pairs: A | None = None,
        *,
        base: nm.Numeric = nm.Integer(10_000),
        position_stride: nm.Numeric = nm.Integer(1),
        factor: nm.Numeric = nm.Integer(1),
        ramp_start: nm.Numeric = nm.Integer(0),
        ramp_end: nm.Numeric = nm.Integer(1),
        real: B = cat.Reals(),
        name: str | fd.DynamicName | None = None,
    ) -> cat.Broadcasted[Complex[B], A, 'YarnRotary']:
        '''The table over `(positions, pairs)`, per `table_over_positions_and_pairs`.
        `name` defaults to the name the class declares.'''
        return table_over_positions_and_pairs(
            cls(base=base, position_stride=position_stride,
                factor=factor, ramp_start=ramp_start, ramp_end=ramp_end,
                name=table_name(cls, name)),
            positions, pairs, real)

@dataclass(frozen=True)
class Decomplex(cat.Operator):
    name: fd.DynamicName = fd.DynamicName('Decomplex')

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        base: B = cat.Reals(),
        pairs: A | None = None,
        channels: A | None = None,
    ) -> cat.Broadcasted[B | Complex[B], A]:
        '''Each complex number of `pairs` written as two adjacent values of `channels`,
        the real part first. An axis left out is a fresh `cat.RawAxis`.'''
        return cat.Broadcasted(
            cls(),
            (cat.Weave(Complex(base), (pairs if pairs is not None else cat.RawAxis(),)),),
            (cat.Weave(base, (channels if channels is not None else cat.RawAxis(),)),),
            (cat.ProdObject().identity(),)
        )

    def sizing_rules[B: cat.Datatype, A: cat.Axis](self, broadcasted: cat.Broadcasted[B, A]) -> nm.Equality:
        input_dim = broadcasted.input_weaves[0].target().shape()[0]
        output_dim = broadcasted.output_weaves[0].target().shape()[0]
        return nm.Equality(
            2 * input_dim.local_size(), output_dim.local_size()
        )

@dataclass(frozen=True)
class PairsAsComplex(cat.Operator):
    '''Adjacent pairs of values read as one complex number each, the first of a pair
    as the real part, which is `torch.view_as_complex` after an `unflatten` into pairs.
    `Decomplex` is its inverse. A rotary embedding reads the rotated channels of a
    latent through it, so that the rotation of a pair is a product of two complex
    numbers.'''
    name: fd.DynamicName = fd.DynamicName('PairsAsComplex')

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        base: B = cat.Reals(),
        channels: A | None = None,
        pairs: A | None = None,
    ) -> cat.Broadcasted[B | Complex[B], A]:
        '''Every two adjacent values of `channels` read as one complex number of
        `pairs`. An axis left out is a fresh `cat.RawAxis`.'''
        return cat.Broadcasted(
            cls(),
            (cat.Weave(base, (channels if channels is not None else cat.RawAxis(),)),),
            (cat.Weave(Complex(base), (pairs if pairs is not None else cat.RawAxis(),)),),
            (cat.ProdObject().identity(),)
        )

    def sizing_rules[B: cat.Datatype, A: cat.Axis](self, broadcasted: cat.Broadcasted[B, A]) -> nm.Equality:
        channels = broadcasted.input_weaves[0].target().shape()[0]
        pairs = broadcasted.output_weaves[0].target().shape()[0]
        return nm.Equality(channels.local_size(), 2 * pairs.local_size())

@dataclass(frozen=True)
class SparseAxis(cat.Axis):
    '''An axis whose slots may be empty: `_size` positions, `activity` live.

    An operation broadcast over the axis works on `activity` values where the
    axis is `_size` wide, and the rest of the positions hold nothing. The empty
    positions are what `deepseek.sparse_expansion.expand_sparse` removes: it
    splits the axis into a dense axis of `activity` carrying the values and a
    `Natural(_size)` selector on that axis carrying the positions they came
    from.

    `obsidian/02-categories/Sparse Axes.md` carries the theory.
    '''
    activity: nm.Numeric = nm.FreeNumeric.field()

    @classmethod
    def template(
        cls,
        parent: nm.Numeric,
        activity: nm.Numeric | None = None,
        name: str | fd.DynamicName | None = None,
    ):
        # `name` is decoration, but useful decoration: an expression with two
        # selections in it draws two axes both labelled 'k/n' otherwise, and
        # the label is the only place the caller's own names can show.
        if activity is None:
            activity = fd.DynamicName('k').capture(nm.FreeNumeric())
        name = (fd.DynamicName.from_str(name) if name is not None
                else fd.DynamicName('k/n'))
        return name.capture(cls(
            _size=parent,
            activity=activity
        ))

@dataclass(frozen=True)
class Select(cat.Operator):
    '''Read a payload at the positions a selection names.

    The selection is the first operand and the payload the second. There is one
    constructor for each form of `SelectionForm` that hands out the positions.

    `template` reads a selection in the `WEIGHTS` form, with the targets
    `(k/n), (n) -> (k/n)`. The selection rides its sparse axis and the payload
    keeps its dense one, and the operator relates the two, so `n` stays dense for
    every other consumer and the `TopK` keeps its `n -> k/n` shape. An `Einops`
    cannot state the relation, because an einops signature describes a
    contraction, and a reindexing cannot state it, because a reindexing is
    affine. Pairing the two axes in a reindexing identifies them, which spreads
    the `k/n` annotation over the whole axis. The value at each live slot
    multiplies the payload read there.

    `complete` reads a selection in the `WEIGHTS_SELECT` form,
    `[R; k], [Nat(n); k], [R; n] -> [R; k]`: the surviving values, the position
    each came from, and the payload. Against a one-hot `(n, k)` matrix the same
    operator is an ordinary contraction, which is the dense reading of a gather.

    `at_positions` reads a selection in the `ONLY_SELECTION` form,
    `[Nat(n); k], [R; n] -> [R; k]`: the positions alone, and the payload. The
    selection axis `k` is in the target of the positions and of the result, and
    no value multiplies the payload.

    The constructors broadcast over nothing. A caller broadcasts the operator by
    giving the weaves tiled positions and the reindexings their degree maps, as
    for every seed morphism.
    '''
    name: fd.DynamicName = fd.DynamicName('Select')

    @classmethod
    def template[B: cat.Datatype](
        cls,
        selection_axis: SparseAxis,
        payload_axis: cat.Axis,
        base: B = cat.Reals(),
    ) -> cat.Broadcasted[B, cat.Axis]:
        return cat.Broadcasted(
            operator=cls(),
            input_weaves=(cat.Weave(base, (selection_axis,)),
                          cat.Weave(base, (payload_axis,))),
            output_weaves=(cat.Weave(base, (selection_axis,)),),
            reindexings=(cat.ProdObject().identity(),
                         cat.ProdObject().identity())
        )

    @classmethod
    def complete[B: cat.Datatype](
        cls,
        k_axis: cat.Axis,
        payload_axis: cat.Axis,
        base: B = cat.Reals(),
    ) -> cat.Broadcasted[B, cat.Axis]:
        """The expanded form: values, indices, payload -> values at those
        indices. The index operand is `Natural(n)` on the `k` axis, the same
        wire a `TopK` in the `WEIGHTS_SELECT` form hands out."""
        return cat.Broadcasted(
            operator=cls(),
            input_weaves=(cat.Weave(base, (k_axis,)),
                          cat.Weave(cat.Natural(payload_axis.local_size()),
                                    (k_axis,)),
                          cat.Weave(base, (payload_axis,))),
            output_weaves=(cat.Weave(base, (k_axis,)),),
            reindexings=(cat.ProdObject().identity(),
                         cat.ProdObject().identity(),
                         cat.ProdObject().identity())
        )

    @classmethod
    def at_positions[B: cat.Datatype](
        cls,
        selected_axis: cat.Axis,
        payload_axis: cat.Axis,
        base: B = cat.Reals(),
    ) -> cat.Broadcasted[B | cat.Natural, cat.Axis]:
        '''The payload read at the positions a `TopK` in the `ONLY_SELECTION` form
        hands out along `selected_axis`, each a `Natural` over `payload_axis`.'''
        return cat.Broadcasted(
            operator=cls(),
            input_weaves=(cat.Weave(cat.Natural(payload_axis.local_size()),
                                    (selected_axis,)),
                          cat.Weave(base, (payload_axis,))),
            output_weaves=(cat.Weave(base, (selected_axis,)),),
            reindexings=(cat.ProdObject().identity(),
                         cat.ProdObject().identity())
        )


def reads_at_positions(target: cat.Broadcasted) -> bool:
    '''Whether `target` is a `Select` reading its payload at positions held as data,
    per `Select.at_positions`.'''
    return (isinstance(target.operator, Select)
            and len(target.input_weaves) == 2
            and isinstance(target.input_weaves[0].datatype, cat.Natural))


@dataclass(frozen=True)
class IndexSelect(cat.Operator):
    """Read a payload at an index: `[Nat(n)], [R; n] -> [R]`.

    The pure gather half of an expanded `Select`, and it is **elementwise in
    the index**: the targets are a rank-0 `Natural(n)` index and the
    payload's parent axis, and everything else, the selection axis `k`
    included, is broadcast. It computes `out = payload[index]` at every point of the
    degree. The index is an operand rather than a reindexing, so the dependence on data
    sits in the datatype, where nothing demands affineness. It is the same
    posture as an expanded Linear's index operand, so the two consumers of a
    `TopK`'s index wire read it identically.

    `sparse_expansion.expand_sparse` rewrites every `Select` on a sparse axis into
    `hold(values) x IndexSelect(index, payload) ; einops('k, k -> k')`, the
    IndexSelect broadcast over the whole output array, with the held selection
    values multiplied back onto the gathered payload, which is what `Select`
    computes. A `Select` at positions, per `Select.at_positions`, computes the same
    gather with the selection axis in its target, and holds no sparse axis to
    rewrite.

    It is named after `torch.index_select`. It was briefly `Grab`, renamed because
    the Para category claims that word for its parameter-slot read.)
    """
    name: fd.DynamicName = fd.DynamicName('IndexSelect')

    @classmethod
    def template[B: cat.Datatype](
        cls,
        payload_axis: cat.Axis,
        base: B = cat.Reals(),
    ) -> cat.Broadcasted[B, cat.Axis]:
        '''The unlifted per-element gather. Broadcasting, over the selection
        axis and over anything else, is the caller's business, as it is for every seed
        morphism.'''
        return cat.Broadcasted(
            operator=cls(),
            input_weaves=(cat.Weave(cat.Natural(payload_axis.local_size()),
                                    ()),
                          cat.Weave(base, (payload_axis,))),
            output_weaves=(cat.Weave(base, ()),),
            reindexings=(cat.ProdObject().identity(),
                         cat.ProdObject().identity())
        )


class SelectionFormMismatch(Exception):
    '''A `TopK` whose outputs, or whose constructor arguments, disagree with its form.'''


@fd.register_enum
class SelectionForm(Enum):
    '''The outputs a `TopK` hands out.

    A selection over `n` keeps `k` values and the positions they came from,
    and the four forms are the four ways of handing those two things out.

    | form             | outputs                  | what it states                         |
    |------------------|--------------------------|----------------------------------------|
    | `WEIGHTS`        | `[R; k/n]`               | the values on a `SparseAxis`, with the positions implicit in the axis. The compressed form, and the default |
    | `WEIGHTS_SELECT` | `[R; k], [Nat(n); k]`    | the values on a dense `k` beside the positions. The pair `torch.topk` returns |
    | `ONLY_WEIGHTS`   | `[R; k]`                 | the values on a dense `k`, with the positions discarded |
    | `ONLY_SELECTION` | `[Nat(n); k]`            | the positions alone, for a selection whose values are not read |

    `deepseek.sparse_expansion.expand_sparse` and
    `para.algebra.para_sparse_expansion.expand_sparse_onto_tape` rewrite the
    first form into the second. The last two hold no sparse axis, so both
    passes leave them alone. `obsidian/02-categories/Sparse Axes.md` carries
    the theory.
    '''
    WEIGHTS = 'weights'
    WEIGHTS_SELECT = 'weights_select'
    ONLY_WEIGHTS = 'only_weights'
    ONLY_SELECTION = 'only_selection'


def dense_selected_axis(
    k: nm.Numeric | None,
    label: str | fd.DynamicName | None = None,
) -> cat.RawAxis:
    '''A fresh axis of size `k`, named after `k` where `k` carries a name, after
    the part of `label` before its slash where it has one, and `k` otherwise,
    which is the letter `SparseWire.template` gives the axis it mints when it
    expands a sparse axis of activity `k`.

    The absolute bars a size symbol's name carries are dropped, so a count drawn
    `|k|` sizes an axis drawn `k`.
    '''
    if k is None:
        k = fd.DynamicName('k').capture(nm.FreeNumeric())
    name = getattr(getattr(k, 'uid', None), '_name', None)
    if name is None and label is not None:
        body = fd.DynamicName.from_str(label).to_bodies()
        name = fd.DynamicName(body.split('/', 1)[0]) if body else None
    return (name or fd.DynamicName('k')).without_absolute_bars().capture(
        cat.RawAxis(_size=k))


def selected_count(selected: cat.Axis) -> nm.Numeric:
    '''How many entries the selection keeps: the activity of a sparse axis, or
    the size of a dense one.'''
    if isinstance(selected, SparseAxis):
        return selected.activity
    return selected.local_size()


def selection_output_weaves[B: cat.Datatype](
    form: SelectionForm,
    base: B,
    index: cat.Natural,
    selected: cat.Axis,
) -> fd.Prod[cat.Weave[B | cat.Natural, cat.Axis]]:
    '''The output weaves a `TopK` in `form` carries, on the axis `selected`.'''
    values = cat.Weave(base, (selected,))
    positions = cat.Weave(index, (selected,))
    match form:
        case SelectionForm.WEIGHTS | SelectionForm.ONLY_WEIGHTS:
            return (values,)
        case SelectionForm.WEIGHTS_SELECT:
            return (values, positions)
        case SelectionForm.ONLY_SELECTION:
            return (positions,)
    raise SelectionFormMismatch(f'{form} is not a SelectionForm')


@dataclass(frozen=True)
class TopK(cat.Operator):
    '''The `k` largest entries along one axis, and the positions they came from.

    `form` says which of the two the operator hands out, and how, per
    `SelectionForm`. `k` is the count kept. It is the same symbol the output
    axis carries as its activity or as its size, so one `NumericConfig`
    assignment reaches both.
    '''
    name: fd.DynamicName = fd.DynamicName('TopK')
    k: nm.Numeric = field(default_factory=nm.FreeNumeric)
    form: SelectionForm = SelectionForm.WEIGHTS

    @classmethod
    def template[B: cat.Datatype](
        cls,
        base: B = cat.Reals(),
        k: nm.Numeric | None = None,
        axis: cat.RawAxis | None = None,
        name: str | fd.DynamicName | None = None,
        form: SelectionForm = SelectionForm.WEIGHTS,
        selected_axis: cat.Axis | None = None,
        positions_of: cat.Axis | None = None,
    ) -> cat.Broadcasted[B | cat.Natural, cat.RawAxis | SparseAxis]:
        '''A selection over `axis`, broadcast over nothing, handing its outputs
        out in `form`.

        Pass `axis` so that the caller's axis is consumed. A fresh `n` minted
        here would win canonicality on composition and erase the caller's name
        from the whole term. In the `WEIGHTS` form `name` labels the sparse
        axis, and without it every selection in an expression draws as `k/n`.
        In the other three forms the outputs ride a dense axis of size `k`.
        `selected_axis` is the output axis where a caller has one, a `SparseAxis`
        in the `WEIGHTS` form and a `RawAxis` in the others, so that two
        selections can hand out one axis, as the decoder's Full and Reindex
        layers do in `notebooks/sota/DeepSeekV41Flash/attention_modes.py`.
        Otherwise the axis is minted here, named after `k` or after the letter
        of `name`.

        A selection over an `AffineSparseAxis` live on a run from its first
        position fills as many slots as that run holds, so in the dense forms
        it hands its outputs out on the axis `selected_slots` states, live on
        the same run of slots, and the slots past it hold the unit. The
        reference implementation of DeepSeek-V4.1-Flash writes `-1` into those
        slots.

        `positions_of` is the axis whose positions a second operand carries, one
        per entry of `axis`, where the entries selected over are themselves a
        selection from a wider axis held as data. The selection then runs over
        `axis` and hands its outputs out as positions of `positions_of`: on a
        `SparseAxis` over it in the `WEIGHTS` form, and as `Natural` indices
        into it in the others. The Reindex layer of the same notebook selects
        over the candidate axis and reports over the compressed positions that
        way. `deepseek.sparse_expansion.expand_topk_over_positions` composes the
        two.
        '''
        if axis is None:
            parent = fd.DynamicName('n').capture(nm.FreeNumeric())
            axis = fd.DynamicName('n').capture(cat.RawAxis(_size=parent))
        else:
            parent = axis.local_size()
        reported = parent if positions_of is None else positions_of.local_size()
        sparse = form is SelectionForm.WEIGHTS
        if selected_axis is not None and isinstance(selected_axis, SparseAxis) is not sparse:
            raise SelectionFormMismatch(
                f'the {form.name} form cannot hand its outputs out on {selected_axis}')
        if isinstance(selected_axis, SparseAxis) and selected_axis._size != reported:
            raise SelectionFormMismatch(
                f'{selected_axis} spans {selected_axis._size} positions and the '
                f'selection reports positions of {reported}')
        if selected_axis is not None:
            selected: cat.Axis = selected_axis
        elif sparse:
            selected = SparseAxis.template(reported, k, name)
        elif isinstance(axis, AffineGuards.AffineSparseAxis) and positions_of is None:
            selected = axis.selected_slots(
                dense_selected_axis(k, name).local_size(),
                dense_selected_axis(k, name).uid._name)
        else:
            selected = dense_selected_axis(k, name)
        scores = cat.Weave(base, (axis,))
        positions = (() if positions_of is None
                     else (cat.Weave(cat.Natural(reported), (axis,)),))
        return cat.Broadcasted(
            cls(k=selected_count(selected), form=form),
            (scores, *positions),
            selection_output_weaves(form, base, cat.Natural(reported), selected),
            (cat.ProdObject().identity(),) * (1 + len(positions))
        )


def selects_over_positions(target: cat.Broadcasted) -> bool:
    '''Whether `target` is a `TopK` taking the positions of the entries it selects
    over as a second operand, per `TopK.template(positions_of=)`.'''
    return (isinstance(target.operator, TopK)
            and len(target.input_weaves) == 2
            and isinstance(target.input_weaves[1].datatype, cat.Natural))


def check_form_matches_outputs(target: cat.Broadcasted) -> None:
    '''Raises `SelectionFormMismatch` where a `TopK`'s output weaves are not
    the ones its form states, which a hand-built `Broadcasted` can produce.'''
    operator = target.operator
    if not isinstance(operator, TopK):
        raise SelectionFormMismatch(f'{operator} is not a TopK')
    positions = tuple(
        isinstance(weave.datatype, cat.Natural) for weave in target.output_weaves)
    sparse = tuple(
        isinstance(axis, SparseAxis)
        for weave in target.output_weaves for axis in weave.target().shape())
    expected = {
        SelectionForm.WEIGHTS: ((False,), (True,)),
        SelectionForm.WEIGHTS_SELECT: ((False, True), (False, False)),
        SelectionForm.ONLY_WEIGHTS: ((False,), (False,)),
        SelectionForm.ONLY_SELECTION: ((True,), (False,)),
    }[operator.form]
    if (positions, sparse) != expected:
        raise SelectionFormMismatch(
            f'a TopK in the {operator.form.name} form emits '
            f'{[weave.datatype for weave in target.output_weaves]} on '
            f'{[weave.target().shape() for weave in target.output_weaves]}')


def selected_axis_name(sparse: SparseAxis) -> fd.DynamicName:
    '''The letter a dense replacement of `sparse` takes: the name of its activity
    where the activity carries one, and otherwise the part of the axis's own label
    before the slash, so that `c/B` expands onto `c`. The absolute bars a size
    symbol's name carries are dropped, so an activity drawn `|k|` names an axis
    drawn `k`.'''
    activity_name = getattr(getattr(sparse.activity, 'uid', None), '_name', None)
    if activity_name is not None:
        return activity_name.without_absolute_bars()
    label = sparse.uid._name.to_bodies() if sparse.uid._name else None
    if label and '/' in label:
        return fd.DynamicName(label.split('/', 1)[0])
    return fd.DynamicName('k')


@dataclass(frozen=True)
class MergedPositions(cat.Operator):
    '''The positions a merge writes a selected block's offsets to:
    `[Nat(P)] -> [Nat(B); u]`.

    Where an `aops.CovariantView` merges `(P, u)` into `B` by `i_B = |u| i_P + i_u`,
    this operator applies the same map to a block number arriving as data and to
    every offset. Its input is a rank-0 `Natural` over the split axis, broadcast
    over the degree, and its output carries the offset axes as targets, because the
    value is a function of the offset. It is an affine reindexing applied to index
    data, and it is what the expansion of a merged selection produces for the index
    wire of the merged axis, per `obsidian/02-categories/Sparse Expansion.md`.
    `selected_position` says which domain axis of the reindexing the arriving index
    runs over.
    '''
    reindexing: cat.StrideMorphism
    selected_position: int = 0


def merged_positions(
    reindexing: cat.StrideMorphism,
    degree: cat.ProdObject[cat.Axis],
    selected_position: int = 0,
) -> cat.Broadcasted[cat.Natural, cat.Axis]:
    '''`MergedPositions` cleanly broadcast over `degree`.'''
    dom = tuple(reindexing.dom())
    parent, = tuple(reindexing.cod())
    selected = dom[selected_position]
    offsets = tuple(axis for i, axis in enumerate(dom) if i != selected_position)
    tiled = (cat.WeaveMode.TILED,) * len(degree)
    return cat.Broadcasted(
        operator=MergedPositions(name=reindexing.name, reindexing=reindexing,
                                 selected_position=selected_position),
        input_weaves=(cat.Weave(cat.Natural(selected.local_size()), tiled),),
        output_weaves=(cat.Weave(cat.Natural(parent.local_size()), (*tiled, *offsets)),),
        reindexings=(degree.identity(),),
    )


def merge_selected_axis[B: cat.Datatype](
    reindexing: cat.StrideMorphism,
    selected: fd.Prod[cat.Axis],
    name: str | fd.DynamicName | None = None,
    base: B = cat.Reals(),
) -> cat.Broadcasted[B, cat.Axis]:
    '''The covariant view of `reindexing` with `selected` in place of its domain
    axes, merging a selection over one of them into a selection over the codomain
    axis.

    `selected` lists the axes the arriving array carries, one per domain axis of the
    reindexing and in its order, with a `SparseAxis` where a selection has been made.
    The codomain axis becomes a `SparseAxis` over the same extent whose activity is
    the product of each selected axis's activity or size, because every offset of an
    active block is active, and `name` labels it. With no `SparseAxis` in `selected`
    the result is the plain merge.
    '''
    parent, = tuple(reindexing.cod())
    if not any(isinstance(axis, SparseAxis) for axis in selected):
        return aops.CovariantView.template(reindexing, base=base, input_axes=selected)
    activity = selected_count(selected[0])
    for axis in selected[1:]:
        activity = nm.Multiplication.template(activity, selected_count(axis))
    merged = SparseAxis.template(parent.local_size(), activity, name)
    return aops.CovariantView.template(
        reindexing, base=base, input_axes=selected, output_axes=(merged,))


def merge_selected_positions(
    reindexing: cat.StrideMorphism,
    selected: cat.RawAxis,
    degree: cat.ProdObject[cat.Axis],
    merged: cat.RawAxis,
    selected_position: int = 0,
    name: str | fd.DynamicName | None = None,
) -> cat.BroadcastedCategory[cat.Natural, cat.Axis]:
    '''The positions of the codomain axis of `reindexing` that a selection over its
    domain axis at `selected_position` covers, from the positions the selection
    chose.

    The selection arrives as `Natural` indices into the split axis, one per entry
    of the dense `selected`, broadcast over `degree`. `MergedPositions` applies the
    split to each index and to every offset, and an `aops.CovariantView` lays the result
    out along `merged`, in the order of the split's strides. The caller declares
    `merged` as large as `selected` times the offsets, with a free size of its own,
    because composition cannot align an axis whose size is a product. `name` labels
    the layout and defaults to the split's own name. The result is the index
    segment `expand_merge` computes for a merged sparse axis, written as the model
    where the selection is made in the `ONLY_SELECTION` form, per
    `obsidian/02-categories/Sparse Axes.md`.
    '''
    dom = tuple(reindexing.dom())
    parent, = tuple(reindexing.cod())
    offsets = tuple(axis for i, axis in enumerate(dom) if i != selected_position)
    _, strides, _ = reindexing._cod_stride_shift[0]
    ordered_strides = (
        strides[selected_position],
        *(stride for i, stride in enumerate(strides) if i != selected_position))
    positions = merged_positions(
        reindexing, cat.ProdObject((*degree, selected)), selected_position)
    layout = aops.CovariantView.template(
        cat.StrideMorphism(
            _dom=(selected, *offsets),
            _cod_stride_shift=((merged, ordered_strides, nm.Integer(0)),),
            name=(reindexing.name if name is None
                  else fd.DynamicName.from_str(name))),
        base=cat.Natural(parent.local_size()))
    return chsh.make_composed(positions, chl.morphism_object_lift(layout, degree))
