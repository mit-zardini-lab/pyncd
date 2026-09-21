# Claude Fable 5.1, effort 80.
'''Check the rotary tables `dst.Rotary` and `dst.YarnRotary` and their standard
expansions.

    python deepseek/validate_rotary.py

Every check is structural, because `torch_compile` evaluates neither an `ops.Arrange`
nor a complex exponential. The checks of the templates read the weaves and the order
of the fields, which tsncd relies on, because it constructs a term positionally. The
checks of the expansions run once per table of `TABLES`: a plain table and a YaRN
table, at the stride one and at a symbolic stride, broadcast over a degree, and with
the degree standing between the axes of the table. A table turns counterclockwise, and
`check_a_conjugated_rotation_turns_back` checks the inverse form, which is the same
table followed by `dst.conjugate_complex_values`.
`algebra.define_by_expansion.define_by_standard_expansion` raises where the expansion
and the operator disagree on a domain or a codomain, so building the definition is
the check that the rule writes at the operator's own degree.

One line is printed per check and per table, and the exit code is the number of
failures.
'''
from __future__ import annotations

# Run as `python deepseek/validate_rotary.py` (from the repository root). Python
# prepends the script's own directory, where `data_structure.py` (the deepseek one)
# would shadow the `data_structure` package, so that entry is dropped.
import os, sys
sys.path = [p for p in sys.path
            if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import subprocess
from collections.abc import Callable

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.define_by_expansion as define_by_expansion
import algebra.einops_simplification as einops_simplification
import algebra.registries.standard_expansions as standard_expansions
import algebra.write_axis_exponents as write_axis_exponents
import construction_helpers as ch  # noqa: F401 - the @ and * overloads
import construction_helpers.lift as chl
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import data_transfer.term_json as term_json
import deepseek.data_structure as dst
import deepseek.registries.standard_expansions as rotary_expansions
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.term_utilities as tutil
import websocket_transfer.auxiliary_information as auxiliary_information
import websocket_transfer.send_morphism as send_morphism

R = cat.Reals()
COMPLEX = dst.Complex(R)
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

h, x, b, a, t, c = (cat.RawAxis.named(letter) for letter in 'hxbatc')
z = fd.DynamicName('z').capture(cat.RawAxis(_size=nm.Integer(2) * t.local_size()))
zbar = fd.DynamicName('\\bar{z}').capture(
    cat.RawAxis(_size=c.local_size() - z.local_size()))

BASE = nm.FreeNumeric.named('\\beta')
FACTOR = nm.FreeNumeric.named('\\kappa')
RAMP_START = nm.FreeNumeric.named('\\mathrm{lo}')
RAMP_END = nm.FreeNumeric.named('\\mathrm{hi}')


def over[B: cat.Datatype, A: cat.Axis](
    axes: tuple[A, ...], morphism: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    return chl.morphism_object_lift(morphism, cat.ProdObject(tuple(axes)))


def hold[B: cat.Datatype, A: cat.Axis](array: cat.Array[B, A]) -> cat.Morphism:
    return cat.ProdObject((array,)).identity()


def yarn_table[A: cat.Axis](
    positions: A,
    position_stride: nm.Numeric = nm.Integer(1),
) -> cat.Broadcasted[dst.Complex[cat.Reals], A, dst.YarnRotary]:
    return dst.YarnRotary.template(
        positions, t, base=BASE, factor=FACTOR, ramp_start=RAMP_START,
        ramp_end=RAMP_END, position_stride=position_stride)


def table_with_the_degree_between_its_axes(
) -> cat.Broadcasted[dst.Complex[cat.Reals], cat.RawAxis, dst.Rotary]:
    '''A plain table whose output weave is `(x, TILED, t)` over the degree `(h,)`,
    which no template builds and a hand-written `cat.Broadcasted` may hold.'''
    return cat.Broadcasted(
        operator=dst.Rotary(base=BASE),
        input_weaves=(),
        output_weaves=(cat.Weave(COMPLEX, (x, cat.WeaveMode.TILED, t)),),
        reindexings=(),
        backup_degree=cat.ProdObject((h,)))


TABLES: dict[str, Callable[[], cat.Broadcasted]] = {
    'plain': lambda: dst.Rotary.template(x, t, base=BASE),
    'plain, stride |a|': lambda: dst.Rotary.template(
        b, t, base=BASE, position_stride=a.local_size()),
    'plain, over the heads': lambda: over((h,), dst.Rotary.template(x, t, base=BASE)),
    'plain, degree between the axes': table_with_the_degree_between_its_axes,
    'YaRN': lambda: yarn_table(x),
    'YaRN, stride |a|': lambda: yarn_table(b, position_stride=a.local_size()),
    'YaRN, over the heads': lambda: over((h,), yarn_table(x)),
}


def operators_of(term: cat.Morphism) -> tuple[cat.Operator, ...]:
    return tuple(node.operator for node in tutil.type_search(cat.Broadcasted, term))


def count_of[O: cat.Operator](kind: type[O], term: cat.Morphism) -> int:
    return sum(1 for operator in operators_of(term) if isinstance(operator, kind))


def sentences_of(description: str) -> list[str]:
    return [sentence for sentence in description.split('. ') if sentence]


def degree_leads_the_weave(table: cat.Broadcasted) -> bool:
    '''Whether every tiled position of the output weave stands ahead of every axis of
    its target, which is the layout a lift over a degree produces.'''
    is_tiled = [isinstance(entry, cat.WeaveMode)
                for entry in table.output_weaves[0]._shape]
    return is_tiled == sorted(is_tiled, reverse=True)


def check_the_template_shapes() -> None:
    '''Each template returns a morphism with no operands whose one result is
    `Complex(real)` over the positions and the pairs, both in the target of the output
    weave, and an axis left out is a fresh `cat.RawAxis`.'''
    for table in (dst.Rotary.template(x, t, base=BASE), yarn_table(x)):
        assert table.has_empty_domain() and table.reindexings == ()
        assert tuple(table.dom()) == ()
        assert tuple(table.cod()) == (cat.Array(COMPLEX, (x, t)),)
        weave, = table.output_weaves
        assert tuple(weave._shape) == (x, t), 'both axes stand in the target'
        assert tuple(table.degree()) == ()
    fresh = dst.Rotary.template()
    positions, pairs = fresh.cod()[0].shape()
    assert isinstance(positions, cat.RawAxis) and isinstance(pairs, cat.RawAxis)
    assert positions.uid != pairs.uid
    half = cat.Reals()
    assert dst.Rotary.template(x, t, real=half).cod()[0].datatype == dst.Complex(half)
    named = dst.YarnRotary.template(x, t, name='\\mathrm{RoPE}_{x}')
    assert named.operator.name == fd.DynamicName.from_str('\\mathrm{RoPE}_{x}')
    assert dst.Rotary.template(x, t).operator.name == dst.Rotary().name
    assert dst.YarnRotary.template(x, t).operator.name == dst.YarnRotary().name
    assert dst.Rotary().name.to_latex() == '\\mathrm{RoPE}'
    assert dst.YarnRotary().name.to_latex() == '\\mathrm{YaRN}'
    assert dst.YarnRotary.template(
        x, t, name='\\mathrm{Y}').operator.name.to_latex() == '\\mathrm{Y}'


def check_the_field_order() -> None:
    '''tsncd constructs a term from its fields by position, so the order of the
    fields is part of the wire format: `name`, `base` and `position_stride`, and then
    `factor`, `ramp_start` and `ramp_end` for YaRN.'''
    assert tuple(dst.Rotary().dict()) == ('name', 'base', 'position_stride')
    assert tuple(dst.YarnRotary().dict()) == (
        'name', 'base', 'position_stride', 'factor', 'ramp_start', 'ramp_end')
    assert issubclass(dst.YarnRotary, dst.Rotary)
    assert dst.Rotary().position_stride == nm.Integer(1)


def check_the_json_round_trip() -> None:
    '''The whole table survives the export and the reconstruction.'''
    table = yarn_table(b, position_stride=a.local_size())
    exported = json.loads(term_json.TermJSONConverter.export_to_json(table))
    written = exported['data']['operator']
    assert written['__type__'] == 'YarnRotary'
    converter = term_json.TermJSONConverter(uid_repository={
        int(uid): record for uid, record in exported['uid_repository'].items()})
    assert converter.reconstruct(exported['data']) == table


def check_the_rules_are_registered_by_import_alone() -> None:
    '''Both classes have a rule of their own once the registry module is imported,
    and `deepseek.data_structure` does not import it, because
    `websocket_transfer/validate_auxiliary_information.py` asserts which operators the
    registry holds when the core alone is loaded.'''
    registered = standard_expansions.registered_operators()
    assert dst.Rotary in registered and dst.YarnRotary in registered
    assert (standard_expansions.RULES[dst.Rotary].rule
            is rotary_expansions.write_rotary_table_out)
    assert (standard_expansions.RULES[dst.YarnRotary].rule
            is rotary_expansions.write_yarn_rotary_table_out)
    probe = ('import sys, deepseek.data_structure; '
             'sys.exit("deepseek.registries.standard_expansions" in sys.modules)')
    environment = {**os.environ, 'PYTHONPATH': REPOSITORY_ROOT}
    completed = subprocess.run(
        [sys.executable, '-c', probe], cwd=REPOSITORY_ROOT, env=environment)
    assert completed.returncode == 0, \
        'deepseek.data_structure imports the registry of the rotary rules'


def check_the_expansion_defines_the_table(table: cat.Broadcasted) -> None:
    '''The expansion has the domain and the codomain of the operator, holds no generic
    operator and no rotary table, and is wired so that it converts to a hypergraph
    and back.'''
    definition = define_by_expansion.define_by_standard_expansion(table)
    expansion = definition.right_hand_side
    assert tuple(expansion.dom()) == tuple(table.dom()) == ()
    assert tuple(expansion.cod()) == tuple(table.cod())
    assert not count_of(ops.GenericOperator, expansion)
    assert not count_of(dst.Rotary, expansion)
    assert not count_of(dst.ComplexRotary, expansion)
    h2m.recycle(expansion)


def check_the_operators_of_the_expansion(table: cat.Broadcasted) -> None:
    '''The positions and the pairs are each arranged once. A plain table applies one
    formula to the pair indices and YaRN three, a stride adds one formula on the
    positions, and the exponential is the last. Only a table whose degree stands
    between its axes needs a view.'''
    operator = table.operator
    expansion = standard_expansions.expand_standard(table)
    is_yarn = isinstance(operator, dst.YarnRotary)
    has_stride = operator.position_stride != nm.Integer(1)
    assert count_of(ops.Arrange, expansion) == 2, \
        'the pair indices of YaRN are arranged once and copied'
    assert count_of(ops.Arithmetic, expansion) == (
        (4 if is_yarn else 2) + (1 if has_stride else 0))
    assert count_of(ops.Einops, expansion) == (2 if is_yarn else 1)
    assert count_of(ops.View, expansion) == (0 if degree_leads_the_weave(table) else 1)
    exponential = [node for node in tutil.type_search(cat.Broadcasted, expansion)
                   if isinstance(node.operator, ops.Arithmetic)
                   and isinstance(node.output_weaves[0].datatype, dst.Complex)]
    assert len(exponential) == 1
    assert exponential[0].operator.formula == nm.Power.template(
        nm.Constant(), rotary_expansions.IMAGINARY_UNIT * nm.x)
    assert exponential[0].operator.name.to_latex() == 'e^{\\mathrm{i} x}'
    strides = [node for node in tutil.type_search(cat.Broadcasted, expansion)
               if isinstance(node.operator, ops.Arithmetic)
               and isinstance(node.output_weaves[0].datatype, cat.Natural)]
    assert len(strides) == (1 if has_stride else 0)
    for node in strides:
        assert isinstance(node.input_weaves[0].datatype, cat.Natural)
        assert node.operator.formula == operator.position_stride * nm.x


def check_the_formula_names_the_axes(table: cat.Broadcasted) -> None:
    '''The formula is one line in the index notation that reads the table at the
    index of the operator's own position axis and pair axis, and names the fields of
    the operator. The description is three plain sentences, and four for YaRN.'''
    row = standard_expansions.expansion_for(table.operator)
    formula, description = row.formula_of(table), row.description_of(table)
    operator = table.operator
    positions, pairs = table.output_weaves[0].target().shape()
    position_letter = positions.uid._name.with_exponent(None).to_latex()
    pair_letter = pairs.uid._name.with_exponent(None).to_latex()
    assert '\n' not in formula, 'the formula is one line'
    assert formula.startswith(
        f'F[i_{{{position_letter}}}, i_{{{pair_letter}}}] = e^'), formula
    assert f'|{pair_letter}|' in formula, formula
    assert rotary_expansions.symbol_latex(operator.base) in formula, formula
    assert 'e^{-\\mathrm{i}' not in formula, formula
    has_stride = operator.position_stride != nm.Integer(1)
    stride = rotary_expansions.symbol_latex(operator.position_stride)
    assert (f'{stride}\\, i_{{{position_letter}}}' in formula) == has_stride, formula
    is_yarn = isinstance(operator, dst.YarnRotary)
    if is_yarn:
        for value in (operator.factor, operator.ramp_start, operator.ramp_end):
            assert rotary_expansions.symbol_latex(value) in formula, formula
    assert ("\\theta'" in formula) == is_yarn, formula
    assert len(sentences_of(description)) == (4 if is_yarn else 3), description
    assert description.endswith('.'), description
    assert ('YaRN' in description) == is_yarn, description
    assert '\\' not in description, f'a description is plain text: {description}'


def check_the_inspection_box_is_packaged(table: cat.Broadcasted) -> None:
    '''The auxiliary information of a figure holding the table has one expansion,
    keyed by the number of the table, with the formula, the description and the
    exported expansion an inspection box shows.'''
    sent = send_morphism.to_morphism(table, recycle=False)
    auxiliary = auxiliary_information.auxiliary_information(sent, with_legend=False)
    (number, expansion), = auxiliary['expansions'].items()
    assert number == '0'
    assert expansion['operator'] == type(table.operator).__name__
    assert expansion['latex'] == table.operator.name.to_latex()
    row = standard_expansions.expansion_for(table.operator)
    assert expansion['formula'] == row.formula_of(table)
    assert expansion['description'] == row.description_of(table)
    exported = json.loads(expansion['expansion'])
    assert '"Rotary"' not in json.dumps(exported['data'])
    assert not expansion['auxiliary'].get('expansions'), \
        'the expansion holds no operator that is written out again'


def check_written_sizes_stay_out_of_the_text() -> None:
    '''A figure drawn with assigned sizes writes each size onto the name of its axis
    and of its symbol, so the stride `|a|` prints `|a|_{2}`. The formula, the
    description and the name of every `ops.Arithmetic` of the expansion name the
    symbols alone, and a base written with a subscript of its own is one body of its
    name.'''
    table = write_axis_exponents.write_assigned_size_exponents(
        dst.YarnRotary.template(
            b, t, base=nm.FreeNumeric.named('\\beta_{0}'), factor=FACTOR,
            ramp_start=RAMP_START, ramp_end=RAMP_END,
            position_stride=a.local_size()),
        {'a': 2, 'b': 8, 't': 32}, fd.ExponentPlacement.SUBSCRIPT)
    assert table.operator.position_stride.to_latex() == '|a|_{2}'
    row = standard_expansions.expansion_for(table.operator)
    names = [node.operator.name for node in tutil.type_search(
                 cat.Broadcasted, standard_expansions.expand_standard(table))
             if isinstance(node.operator, ops.Arithmetic)]
    texts = (row.formula_of(table), row.description_of(table),
             *(name.to_latex() for name in names))
    for text in texts:
        for size in ('{2}', '{8}', '{32}'):
            assert size not in text, text
    assert '|a|\\, i_{b}' in row.formula_of(table), row.formula_of(table)
    assert all(name.subscript is None for name in names), \
        'the name of a formula is one body'
    assert '\\beta_{0}^{-x / |t|}' in [name.body for name in names]


def rotate_channels(
    table: cat.Broadcasted,
) -> cat.BroadcastedCategory:
    '''The rotation of the last `|z|` channels of `R[h, x, c]` by `table`: the
    channels are deconcatenated, the rotated ones are read as complex numbers,
    multiplied by the table and written back as reals, and the two parts are
    concatenated onto `c`.

    The product is written over the declared axes. `ops.Einops.template` mints axes
    of its own, and composition keeps whichever of two aligned raw axes has the
    greater uid, which differs from one process to the next, so a comparison with
    the declared axes would pass in some runs and fail in others.'''
    shapes = ((h, x, zbar), (h, x, z))
    rotate_pairs = (
        (hold(cat.Array(COMPLEX, (h, x, t))) * table)
        @ einops_simplification.einsum(((h, x, t), (x, t)), (h, x, t), COMPLEX))
    return (aops.DeconcatenateAxes.template(shapes, concatenated=c)
            @ (hold(cat.Array(R, (h, x, zbar)))
               * (over((h, x), dst.PairsAsComplex.template(R, z, t))
                  @ rotate_pairs
                  @ over((h, x), dst.Decomplex.template(R, t, z))))
            @ aops.ConcatenateAxes.template(shapes, concatenated=c))


def check_a_rotation_keeps_the_channel_axis() -> None:
    '''A rotation written against either table maps `R[h, x, c]` to `R[h, x, c]` on
    the declared axis `c`, holds the table as its one nullary operator with a rule,
    and holds no generic operator once the table is written out.'''
    for table in (dst.Rotary.template(x, t, base=BASE), yarn_table(x)):
        rotation = rotate_channels(table)
        latent = cat.Array(R, (h, x, c))
        assert tuple(rotation.dom()) == tuple(rotation.cod()) == (latent,)
        assert count_of(dst.Rotary, rotation) == 1
        written_out = h2m.recycle(
            rotate_channels(standard_expansions.expand_standard(table)))
        assert tuple(written_out.dom()) == tuple(written_out.cod()) == (latent,)
        assert not count_of(ops.GenericOperator, written_out)
        assert not count_of(dst.Rotary, written_out)


def check_a_conjugated_rotation_turns_back() -> None:
    '''An inverse rotary embedding is the counterclockwise table followed by
    `dst.conjugate_complex_values`, which is one elementwise map over the complex
    numbers named with a conjugate bar. The conjugate of a turn of length one is the
    turn through the same angle in the other direction, so the two rotations compose
    to the identity on every pair.'''
    table = yarn_table(x)
    conjugated = table @ dst.conjugate_complex_values(table.cod()[0])
    assert tuple(conjugated.cod()) == tuple(table.cod())
    maps = [node for node in tutil.type_search(cat.Broadcasted, conjugated)
            if isinstance(node.operator, ops.Arithmetic)]
    assert len(maps) == 1
    assert maps[0].operator.formula == nm.Conjugate(nm.x)
    assert maps[0].operator.name.to_latex() == '\\overline{x}'
    assert isinstance(maps[0].output_weaves[0].datatype, dst.Complex)
    rotation = rotate_channels(conjugated)
    latent = cat.Array(R, (h, x, c))
    assert tuple(rotation.dom()) == tuple(rotation.cod()) == (latent,)
    assert count_of(dst.Rotary, rotation) == 1


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_template_shapes,
    check_the_field_order,
    check_the_json_round_trip,
    check_the_rules_are_registered_by_import_alone,
    check_written_sizes_stay_out_of_the_text,
    check_a_rotation_keeps_the_channel_axis,
    check_a_conjugated_rotation_turns_back,
)

CHECKS_OF_EVERY_TABLE: tuple[Callable[[cat.Broadcasted], None], ...] = (
    check_the_expansion_defines_the_table,
    check_the_operators_of_the_expansion,
    check_the_formula_names_the_axes,
    check_the_inspection_box_is_packaged,
)


def cases() -> dict[str, Callable[[], None]]:
    '''Every check by the line it prints: the checks that stand alone, and each check
    of a table once per table of `TABLES`.'''
    standing_alone = {check.__name__: check for check in CHECKS}
    per_table = {
        f'{check.__name__} [{label}]':
            (lambda check=check, build=build: check(build()))
        for label, build in TABLES.items() for check in CHECKS_OF_EVERY_TABLE}
    return {**standing_alone, **per_table}


def run_checks() -> int:
    '''Every case, one line each, and the number that failed.'''
    failures = 0
    for label, case in cases().items():
        try:
            case()
        except Exception as reason:
            failures += 1
            print(f'FAILED  {label}: {type(reason).__name__}: {reason}')
        else:
            print(f'ok      {label}')
    return failures


if __name__ == '__main__':
    failed = run_checks()
    total = len(cases())
    print(f'{total - failed} of {total} rotary checks passed')
    sys.exit(1 if failed else 0)
