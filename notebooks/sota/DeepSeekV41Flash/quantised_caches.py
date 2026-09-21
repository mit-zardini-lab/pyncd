'''The rounding the released DeepSeek-V4.1-Flash applies to the arrays it caches.

Written by Claude Fable 5.1, reasoning effort 80.

The released code rounds four arrays to four or eight bits for each channel: the
compressed entries, the keys and the queries of the indexer, and the window latents. It
caches every one of them but the queries. Each is rounded by a kernel called with
`inplace=True`. The kernel cuts the channels of one latent into groups, divides each
group by a scale of its own, rounds every channel to the nearest number the format
holds, and multiplies the group by its scale again. The attention and the indexer
therefore read reals that carry the rounding, and a tape slot of the integrated model
holds those reals.

A round trip is the pass of that kernel over one latent. Each is written here with
standard operators and boxed, so it draws as one named box and a caller broadcasts it
with `over(...)`.

    ENTRY_ROUND_TRIP     R[c] -> R[c]   FP4 (E2M1) in groups of 16 channels, each scale
                                        rounded through E4M3
    INDEXER_ROUND_TRIP   R[d] -> R[d]   FP4 (E2M1) in groups of 32 channels, each scale
                                        rounded up to a power of two (E8M0)
    WINDOW_ROUND_TRIP    R[c] -> R[c]   FP8 (E4M3) in groups of 32 channels, each scale
                                        rounded up to a power of two (UE8M0)

The cut of the channels into groups is affine, `i_c = |y| i_E + i_y`, so it is an
`ops.View`, and the merge back is the `aops.CovariantView` of the same reindexing,
which returns the declared channel axis itself. The rounding to a format is a
`Quantization.TypeConvert` into the format followed by one back to the reals. A
`TypeConvert` names two datatypes and holds no formula, so `explain_cast` says which
numbers each format holds. The two casts inside a round trip are into the bare `E2M1`
and `E4M3` formats, because the value they round is the channel already divided by
its scale, and the scale stands beside them in the expression rather than inside the
format. The wire in front of a projection is the other place the released code rounds,
and there the scale is carried by the format, `Quantization.MXFP8`, because the GEMM
reads the elements and the scales together.

The round trip has a purpose although its result is a real again. The released code
holds the caches in BF16, and the kernel called in place writes back the element
multiplied by its scale, so a cached number is one an FP4 or FP8 cache with the same
scales would give when read, and no scale is kept beside the cache. The attention and
the indexer then read the rounded numbers as reals.

A scale rounded up to a power of two is `2^{\\lceil \\log_2 s \\rceil}`. The package
has no numeric for a ceiling, so that one step is an `ops.GenericOperator` named
`CEILING`, between an `ops.Arithmetic` of the logarithm and an `ops.Arithmetic` of the
power. `CEILING_EXPLANATION` holds its mathematics and its released lines.

The released FP4 kernel keeps the largest magnitude of a group at or above `6 * 2**-9`
or `6 * 2**-126` and then divides it by 6. The expression divides by 6 first and keeps
the scale at or above `2^{-9}` or `2^{-126}`, which is the same function. The order is
changed because `nm.Multiplication` prints `6 * 2**-9` with its two factors side by
side, and a reader takes the two digits for the number sixty-two.

The axes of the groups are declared here. `E` is the 32 groups of an entry, as
`omitted_mechanisms` declares it, and `y` is the 16 channels of one of those groups.
`\\hat{y}` is the 32 channels of a group of the two other formats, `\\hat{E}` is the 4
groups of an indexer key or query, and `\\tilde{E}` is the 16 groups of a window latent.

`ARITHMETIC_ROLES`, `ARITHMETIC_REFERENCES`, `REINDEXING_EXPLANATIONS`, `explain_cast`
and `CEILING_EXPLANATION` are the rows the explanation tables of the integrated model
and of the omissions notebook take for the operators of this module.
'''
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.write_index_notation as write_index_notation
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import quantization.data_structure.Quantization as Quantization
from term_utilities.code_references import pinned_link

import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
import notebooks.sota.DeepSeekV41Flash.reference_links as reference_links
from notebooks.display.explain_operators import OperatorExplanation
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, c, d
from notebooks.sota.DeepSeekV41Flash.reference_links import kernel_lines, model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

E = omitted_mechanisms.E
y = cat.RawAxis.named('y', code_form='channels_per_entry_scale')
yhat = cat.RawAxis.named('\\hat{y}', code_form='channels_per_power_of_two_scale')
Ehat = cat.RawAxis.named('\\hat{E}', code_form='indexer_scale_groups')
Etilde = cat.RawAxis.named('\\tilde{E}', code_form='window_scale_groups')

FP4 = omitted_mechanisms.FP4
FP8 = Quantization.E4M3
LARGEST_E2M1 = nm.Integer(6)
LARGEST_E4M3 = nm.Integer(448)
SMALLEST_E4M3_SCALE = nm.Power.template(nm.Integer(2), nm.Integer(-9))
SMALLEST_POWER_OF_TWO_SCALE = nm.Power.template(nm.Integer(2), nm.Integer(-126))
SMALLEST_FP8_GROUP_MAXIMUM = nm.Power.template(nm.Integer(10), nm.Integer(-4))

SCALE_GROUP_SPLIT = 'qgrp'
LOGARITHM_TO_BASE_TWO = '\\log_{2} x'
CEILING = '\\lceil x \\rceil'
CAST_TO_THE_REALS = 'R'

ENTRY_BOX = 'FP4e'
INDEXER_BOX = 'FP4i'
WINDOW_BOX = 'FP8w'
FOUR_BIT_COLOUR = '#E6E1F4'
EIGHT_BIT_COLOUR = '#DDE6F2'


@dataclass(frozen=True)
class ScaleGroups[A: cat.Axis]:
    '''A channel axis cut into consecutive groups that share one scale. Channel
    `|group_channels| i_groups + i_group_channels` of `channels` is channel
    `i_group_channels` of group `i_groups`.'''
    channels: A
    groups: A
    group_channels: A

    def split(self) -> sc.StrideMorphism[A]:
        return sc.StrideMorphism(
            _dom=(self.groups, self.group_channels),
            _cod_stride_shift=((
                self.channels,
                (self.group_channels.local_size(), nm.Integer(1)),
                nm.Integer(0)),),
            name=fd.DynamicName(SCALE_GROUP_SPLIT))

    def grouped_axes(self) -> tuple[A, A]:
        return (self.groups, self.group_channels)

    def grouped_latent(self) -> cat.Array[cat.Reals, A]:
        return cat.Array(R, self.grouped_axes())

    def scales(self) -> cat.Array[cat.Reals, A]:
        return cat.Array(R, (self.groups,))


ENTRY_GROUPS = ScaleGroups(channels=c, groups=E, group_channels=y)
INDEXER_GROUPS = ScaleGroups(channels=d, groups=Ehat, group_channels=yhat)
WINDOW_GROUPS = ScaleGroups(channels=c, groups=Etilde, group_channels=yhat)


def magnitude() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.AbsoluteValue(nm.x))


def largest_of_each_group[A: cat.Axis](
    groups: ScaleGroups[A],
) -> cat.BroadcastedCategory:
    return over((groups.groups,), cat.Broadcasted(
        operator=ops.Maximum(),
        input_weaves=(cat.Weave(R, (groups.group_channels,)),),
        output_weaves=(cat.Weave(R, ()),),
        reindexings=(cat.ProdObject().identity(),)))


def divide_by(largest_stored: nm.Numeric) -> cat.Broadcasted:
    return ops.Arithmetic.template(
        nm.x / largest_stored,
        name=fd.DynamicName(f'x / {largest_stored.to_latex()}'))


def keep_at_or_above(smallest: nm.Numeric) -> cat.Broadcasted:
    '''The larger of a value and `smallest`, so a group of zeros has a scale above
    zero and the division by the scale is defined.'''
    return ops.Arithmetic.template(nm.LargerOf(nm.x, smallest))


def clamp_to(largest_stored: nm.Numeric) -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.Clamp(nm.x, -largest_stored, largest_stored))


def round_through[A: cat.Axis](
    stored: Quantization.Quantified, axes: tuple[A, ...],
) -> cat.BroadcastedCategory:
    '''Every value of an array over `axes` rounded to the nearest number `stored`
    holds and read back as a real.'''
    return (Quantization.TypeConvert.over_shape(R, stored, axes, stored.form.value)
            @ Quantization.TypeConvert.over_shape(stored, R, axes, CAST_TO_THE_REALS))


def ceiling_over[A: cat.Axis](axes: tuple[A, ...]) -> cat.Broadcasted:
    '''The smallest whole number at or above each value of an array over `axes`. The
    package has no numeric for a ceiling, so the step is a generic operator.'''
    one_value = cat.Array(R, ())
    return omitted_mechanisms.generic_operator(
        CEILING, (one_value,), (one_value,), degree=axes)


def round_up_to_a_power_of_two[A: cat.Axis](
    axes: tuple[A, ...],
) -> cat.BroadcastedCategory:
    '''Every scale `s` of an array over `axes` replaced by
    `2^{\\lceil \\log_2 s \\rceil}`, the smallest power of two at or above it, which is
    what the released `fast_round_scale` returns.'''
    logarithm = ops.Arithmetic.template(
        nm.Logarithm.template(nm.Integer(2), nm.x),
        name=fd.DynamicName(LOGARITHM_TO_BASE_TWO))
    power = ops.Arithmetic.template(nm.Power.template(nm.Integer(2), nm.x))
    return over(axes, logarithm) @ ceiling_over(axes) @ over(axes, power)


def multiply_each_group_by_its_number[A: cat.Axis](
    groups: ScaleGroups[A],
) -> cat.Broadcasted:
    '''Every channel of a group multiplied by the one number of that group. The axes
    are the declared ones, because a letter of an `ops.Einops` template mints a fresh
    axis that can replace a declared axis where the two are composed.'''
    tiled = cat.WeaveMode.TILED
    return cat.Broadcasted(
        operator=ops.Einops(name=fd.DynamicName('einops'), signature=((), ())),
        input_weaves=(cat.Weave(R, (tiled, tiled)), cat.Weave(R, (tiled,))),
        output_weaves=(cat.Weave(R, (tiled, tiled)),),
        reindexings=(route((0, 1), groups.grouped_axes()),
                     route((0,), groups.grouped_axes())))


def round_trip_of_one_latent[A: cat.Axis](
    groups: ScaleGroups[A],
    scale_of_largest_magnitude: cat.BroadcastedCategory,
    stored: Quantization.Quantified,
    largest_stored: nm.Numeric,
) -> cat.BroadcastedCategory:
    '''One latent cut into groups, each group divided by its scale, clamped to the
    range of `stored`, rounded through `stored` and multiplied by its scale again, and
    the groups merged back onto the channel axis. `scale_of_largest_magnitude` maps
    the largest magnitude of every group to the scale of that group.'''
    grouped, scales = groups.grouped_latent(), groups.scales()
    scale_of_each_group = (over(groups.grouped_axes(), magnitude())
                           @ largest_of_each_group(groups)
                           @ scale_of_largest_magnitude)
    reciprocal = ops.Arithmetic.template(nm.Integer(1) / nm.x)
    divide_and_round = ((hold(grouped) * over((groups.groups,), reciprocal))
                        @ multiply_each_group_by_its_number(groups)
                        @ over(groups.grouped_axes(), clamp_to(largest_stored))
                        @ round_through(stored, groups.grouped_axes()))
    return (ops.View.template(reindexing=(groups.split(),), name=SCALE_GROUP_SPLIT)
            @ route((0, 0), (grouped,))
            @ (hold(grouped) * scale_of_each_group)
            @ route((0, 1, 1), (grouped, scales))
            @ (divide_and_round * hold(scales))
            @ multiply_each_group_by_its_number(groups)
            @ aops.CovariantView.template(groups.split()))


def round_trip_formula[A: cat.Axis](
    groups: ScaleGroups[A],
    scale_of_largest_magnitude_latex: Callable[[str], str],
    stored: Quantization.Quantified,
    largest_stored: nm.Numeric,
) -> str:
    '''The round trip in index notation, one line each for the largest magnitude `a`
    of a group, the scale `s` of the group, which
    `scale_of_largest_magnitude_latex` writes from the LaTeX of `a` at the group, and
    the rounded latent `\\hat{v}` written from the latent `v`.'''
    _, group, channel = write_index_notation.axis_letters(
        (groups.channels, *groups.grouped_axes()))
    group_index = write_index_notation.index_of(group)
    channel_index = write_index_notation.index_of(channel)
    position = rf'\lvert {channel} \rvert\, {group_index} + {channel_index}'
    largest = (rf'a[{group_index}] = \max_{{{channel_index} \in {channel}}} '
               rf'\lvert v[{position}] \rvert')
    scale = (rf's[{group_index}] = '
             + scale_of_largest_magnitude_latex(f'a[{group_index}]'))
    limit = largest_stored.to_latex()
    rounded = (rf'\hat{{v}}[{position}] = s[{group_index}]\, '
               rf'\mathrm{{{stored.form.value}}}\big(\ulcorner v[{position}] / '
               rf's[{group_index}] \lrcorner_{{-{limit}}}^{{{limit}}}\big)')
    return rf'\begin{{gathered}} {largest} \\ {scale} \\ {rounded} \end{{gathered}}'


def entry_scale_latex(largest_magnitude: str) -> str:
    return rf'\mathrm{{E4M3}}\big(\max({largest_magnitude} / 6,\, 2^{{-9}})\big)'


def indexer_scale_latex(largest_magnitude: str) -> str:
    return (rf'2^{{\lceil \log_{{2}} \max({largest_magnitude} / 6,\, 2^{{-126}}) '
            r'\rceil}')


def window_scale_latex(largest_magnitude: str) -> str:
    return (rf'2^{{\lceil \log_{{2}} (\max({largest_magnitude},\, 10^{{-4}}) / 448) '
            r'\rceil}')


def entry_round_trip() -> cat.Broadcasted:
    '''One compressed entry through FP4 and back: groups of 16 channels, with the scale
    of each group rounded through E4M3.'''
    scale_of_largest_magnitude = (
        over((E,), divide_by(LARGEST_E2M1) @ keep_at_or_above(SMALLEST_E4M3_SCALE))
        @ round_through(FP8, (E,)))
    return boxed(cat.Block.template(
        round_trip_of_one_latent(
            ENTRY_GROUPS, scale_of_largest_magnitude, FP4, LARGEST_E2M1),
        title=text.ENTRY_TITLE, fill_color=FOUR_BIT_COLOUR,
        formula=round_trip_formula(ENTRY_GROUPS, entry_scale_latex, FP4, LARGEST_E2M1),
        description=text.ENTRY_ROUND_TRIP_DESCRIPTION,
        references=(model_lines(758, 761), kernel_lines(158),
                    kernel_lines(160, 163), kernel_lines(167, 172))),
        ENTRY_BOX)


def indexer_round_trip() -> cat.Broadcasted:
    '''One indexer key or query through FP4 and back: groups of 32 channels, with the
    scale of each group rounded up to a power of two.'''
    scale_of_largest_magnitude = (
        over((Ehat,),
             divide_by(LARGEST_E2M1) @ keep_at_or_above(SMALLEST_POWER_OF_TWO_SCALE))
        @ round_up_to_a_power_of_two((Ehat,)))
    return boxed(cat.Block.template(
        round_trip_of_one_latent(
            INDEXER_GROUPS, scale_of_largest_magnitude, FP4, LARGEST_E2M1),
        title=text.QUANTISED_INDEXER_TITLE, fill_color=FOUR_BIT_COLOUR,
        formula=round_trip_formula(
            INDEXER_GROUPS, indexer_scale_latex, FP4, LARGEST_E2M1),
        description=text.INDEXER_ROUND_TRIP_DESCRIPTION,
        references=(model_lines(546), model_lines(552), model_lines(27, 30),
                    kernel_lines(158), kernel_lines(164, 166), kernel_lines(22, 37),
                    kernel_lines(167, 172))),
        INDEXER_BOX)


def window_round_trip() -> cat.Broadcasted:
    '''One window latent through FP8 and back: groups of 32 channels, with the scale
    of each group rounded up to a power of two.'''
    scale_of_largest_magnitude = (
        over((Etilde,),
             keep_at_or_above(SMALLEST_FP8_GROUP_MAXIMUM) @ divide_by(LARGEST_E4M3))
        @ round_up_to_a_power_of_two((Etilde,)))
    return boxed(cat.Block.template(
        round_trip_of_one_latent(
            WINDOW_GROUPS, scale_of_largest_magnitude, FP8, LARGEST_E4M3),
        title=text.QUANTISED_WINDOW_TITLE, fill_color=EIGHT_BIT_COLOUR,
        formula=round_trip_formula(
            WINDOW_GROUPS, window_scale_latex, FP8, LARGEST_E4M3),
        description=text.WINDOW_ROUND_TRIP_DESCRIPTION,
        references=(model_lines(705, 707), model_lines(27, 30), kernel_lines(44, 46),
                    kernel_lines(74, 80), kernel_lines(22, 37), kernel_lines(81, 86))),
        WINDOW_BOX)


ENTRY_ROUND_TRIP = entry_round_trip()
INDEXER_ROUND_TRIP = indexer_round_trip()
WINDOW_ROUND_TRIP = window_round_trip()


def name_of(arithmetic: cat.Broadcasted) -> str:
    '''The text `notebooks.display.explain_operators.explain_named_arithmetic` keys an
    elementwise map by.'''
    return arithmetic.operator.name.to_bodies()


ARITHMETIC_ROLES: dict[str, str] = {
    name_of(magnitude()): (
        text.MAGNITUDE_ROLE),
    name_of(divide_by(LARGEST_E2M1)): (
        text.SCALE_BY_SIX_ROLE),
    name_of(divide_by(LARGEST_E4M3)): (
        text.SCALE_BY_448_ROLE),
    name_of(keep_at_or_above(SMALLEST_E4M3_SCALE)): (
        text.SCALE_FLOOR_E4M3_ROLE),
    name_of(keep_at_or_above(SMALLEST_POWER_OF_TWO_SCALE)): (
        text.SCALE_FLOOR_E8M0_ROLE),
    name_of(keep_at_or_above(SMALLEST_FP8_GROUP_MAXIMUM)): (
        text.MAGNITUDE_FLOOR_ROLE),
    LOGARITHM_TO_BASE_TWO: (
        text.LOG2_SCALE_ROLE),
    name_of(clamp_to(LARGEST_E2M1)): (
        text.CLAMP_E2M1_ROLE),
    name_of(clamp_to(LARGEST_E4M3)): (
        text.CLAMP_E4M3_ROLE),
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    name_of(magnitude()): (kernel_lines(74), kernel_lines(158)),
    name_of(divide_by(LARGEST_E2M1)): (kernel_lines(131, 132), kernel_lines(163)),
    name_of(divide_by(LARGEST_E4M3)): (kernel_lines(44, 46), kernel_lines(78)),
    name_of(keep_at_or_above(SMALLEST_E4M3_SCALE)): (kernel_lines(161, 162),),
    name_of(keep_at_or_above(SMALLEST_POWER_OF_TWO_SCALE)): (kernel_lines(165),),
    name_of(keep_at_or_above(SMALLEST_FP8_GROUP_MAXIMUM)): (kernel_lines(76),),
    LOGARITHM_TO_BASE_TWO: (kernel_lines(22, 27),),
    name_of(clamp_to(LARGEST_E2M1)): (kernel_lines(171),),
    name_of(clamp_to(LARGEST_E4M3)): (kernel_lines(85),),
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    SCALE_GROUP_SPLIT: ReindexingExplanation(
        description=text.SCALE_GROUP_SPLIT_DESCRIPTION,
        references=(kernel_lines(72), kernel_lines(156))),
}

CEILING_EXPLANATION = OperatorExplanation(
    title=r'\text{Ceiling}',
    formula=(r'\lceil \cdot \rceil : R[\,] \to R[\,], \qquad '
             r'\lceil x \rceil = \min \{ n \in \mathbb{Z} : n \geq x \}'),
    description=text.CEILING_DESCRIPTION,
    references=(kernel_lines(22, 27), kernel_lines(36, 37), kernel_lines(78),
                kernel_lines(166)))

STORED_NUMBERS: dict[Quantization.Encoding, str] = {
    Quantization.Encoding.E2M1: (text.E2M1_STORED_NUMBERS),
    Quantization.Encoding.E4M3: (text.E4M3_STORED_NUMBERS),
}

CAST_REFERENCES: dict[Quantization.Encoding, fd.Prod[cat.CodeReference]] = {
    Quantization.Encoding.E2M1: (kernel_lines(167, 172),
             pinned_link(reference_links.BASE_URL, 'inference/convert.py', 13, 15)),
    Quantization.Encoding.E4M3: (kernel_lines(81, 86), kernel_lines(163)),
}


def explain_cast(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row of an explanation table for `Quantization.TypeConvert`: a cast into one
    of the two stored forms of this module rounds, and a cast from one of them back
    to the reals changes no value. A stored form carries no scale, because the scale
    stands beside the cast in the round trip, so a cast into a block-scaled format
    such as MXFP8 is left to the row written for it, as is any other cast.'''
    operator = target.operator
    if not isinstance(operator, Quantization.TypeConvert):
        return None
    into = Quantization.quantisation_of(operator.target)
    out_of = Quantization.quantisation_of(operator.source)
    if into is not None and into.scale is None and into.form in STORED_NUMBERS:
        return OperatorExplanation(
            title=rf'\text{{Cast to {into.form.value}}}',
            formula=rf'y = \mathrm{{round}}_{{\mathrm{{{into.form.value}}}}}(x)',
            description=text.CAST_INTO_DESCRIPTION.format(
                form=into.form.value, stored_numbers=STORED_NUMBERS[into.form]),
            references=CAST_REFERENCES[into.form])
    if out_of is not None and out_of.scale is None and out_of.form in STORED_NUMBERS:
        return OperatorExplanation(
            title=rf'\text{{Cast from {out_of.form.value}}}',
            formula='y = x',
            description=text.CAST_OUT_OF_DESCRIPTION.format(form=out_of.form.value),
            references=CAST_REFERENCES[out_of.form])
    return None
