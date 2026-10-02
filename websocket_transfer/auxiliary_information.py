# Claude Fable 5.1, effort 80. Revised by Claude Opus 5.5 (1M context), effort 40.
'''Packaging what an interactive figure shows beside the term it draws.

tsncd does no algebra, so everything an inspection box or a legend shows has to be
computed here and sent beside the term, as the `auxiliary` field of a `dataUpdate` or a
`renderRequest`. `obsidian/05-backends/Diagram Wire Format.md` states the field and
`obsidian/05-backends/Advanced Display.md` what tsncd does with it. Three things are
packaged.

The legend lists every named axis of the term with the integer its size comes to and
the code form its name carries. The size is read off the axis where a configuration has
sized the term, and evaluated under `assigned_sizes` where the term is symbolic and the
assignments are given beside it, as `notebooks.display.axis_sizes` reads them. A second
table lists every `cat.Natural` that is the datatype of an array of the term, with the
integer its bound comes to and the bound written in the code names of its symbols. The
user asked for the second table on 2026-09-27.

The block information carries, for every block of the term keyed by its tag's uid, the
title, the formula, the description and the code references its aesthetics hold. A
reference with no url of its own is linked from its path under `code_link_base`, and
left unlinked where no base is given. A reference whose url is on a host
`REFERENCE_ICONS` lists carries the name of that host's icon, which tsncd draws before
the link.

The formula and the description of every block and every operator are written with the
indices of the formula lettered i, j, k and onwards, and carry the indices the formula
holds for every position of their axes, per
`websocket_transfer/letter_formula_indices.py`. The index of a guarded axis of the block
or the operator also carries the positions of the axis that hold a value, per
`websocket_transfer/write_formula_index_ranges.py`.

The operator expansions carry, for every `Broadcasted` whose operator
`algebra.registries.standard_expansions` writes out, the expansion as its own exported
term, with the formula and the description the registry holds. A `Broadcasted` has no
uid, so each is keyed by the number tsncd's importer gives it, which
`data_transfer.broadcast_occurrences` reproduces. An expansion is packaged with its own
auxiliary information, so an operator inside one can be opened in turn. `write_out` is
the function that writes an operator out. The default applies the registry's rule to
the operator as it stands, and `notebooks/display/expand_with_parameters.py` supplies
one that grabs the operator's parameters first, so a weight inside the operator is
drawn. An operator `write_out` returns unchanged, or returns `None` for, has no entry.
`write_out` is applied to the operators of the figure alone. The operators inside an
expansion are written out by the registry's rule as they stand, because their parameters
are operands already, and a `Linear` that selects among weights, which the rule leaves
in its parametrised form, would otherwise have its weight grabbed again at every depth.
`operator_references` gives the code references an expansion carries, by operator
class, because a registry row is general and the code an operator stands for belongs
to the model that is drawn. `operator_roles` gives a sentence on what one named
operator does in that model, keyed by the text of the operator's name, so the box over
the weight `W^{Qa}` says what that weight is for before it says what a linear map is.
The formula and the description of a registry row are written from the operator where
the row holds a function, so the formula over a weight names that weight's axes.

`auxiliary_information` is called on the morphism the transport exports, after every
display pass has been applied and after a hypergraph has been converted, because the
numbering and the block tags are read off exactly what is sent.

`algebra.operator_expansion` and `caching.registries.standard_expansions` are imported
for their registrations. The registry is filled by the decorators those modules apply to
their rules, so a rule is missing until its module has been imported. The cache of
Mixtral-8x7B opened no box on 2026-09-27 because nothing its notebook ran imported the
rule of a cache, and every cache a figure holds now opens one.
'''
from __future__ import annotations

import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import algebra.operator_expansion as operator_expansion  # noqa: F401
import algebra.registries.standard_expansions as standard_expansions
import algebra.write_axis_exponents as write_axis_exponents
import caching.registries.standard_expansions as cache_expansions  # noqa: F401
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import data_transfer.broadcast_occurrences as broadcast_occurrences
import data_transfer.term_json as term_json
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.term_utilities as tutil
import websocket_transfer.letter_formula_indices as letter_formula_indices
import websocket_transfer.websockets_transfer as wst
import websocket_transfer.write_formula_index_ranges as write_formula_index_ranges


VSCODE_SCHEME = 'vscode://'

# The icon tsncd draws before a link, by the host of the link. tsncd holds the
# drawing of each icon under the same name.
REFERENCE_ICONS: dict[str, str] = {'huggingface.co': 'huggingface'}

type WriteOut = Callable[[cat.Broadcasted], fd.GeneralTerm | None]
type OperatorReferences = Mapping[type[cat.Operator], fd.Prod[cat.CodeReference]]


@dataclass(frozen=True)
class OperatorRole:
    '''What one named operator does in the model that is drawn, in a sentence or
    two, and the released lines that declare it.'''
    role: str
    references: fd.Prod[cat.CodeReference] = ()


type OperatorRoles = Mapping[str, OperatorRole]


def auxiliary_information(
    term: fd.GeneralTerm,
    *,
    assigned_sizes: Mapping[str, int] | None = None,
    code_link_base: str | None = None,
    with_legend: bool = True,
    write_out: WriteOut = standard_expansions.expand_standard,
    operator_references: OperatorReferences | None = None,
    operator_roles: OperatorRoles | None = None,
) -> wst.DiagramAuxiliary:
    '''The `auxiliary` field for `term`, holding the two tables of the legend where
    `with_legend` asks for them, the information of every block, and the expansion of
    every operator `write_out` writes out. A part with nothing in it is left out of the
    field.'''
    auxiliary: wst.DiagramAuxiliary = {}
    if with_legend:
        auxiliary['legend'] = legend_rows(term, assigned_sizes)
        naturals = natural_legend_rows(term, assigned_sizes)
        if naturals:
            auxiliary['naturals'] = naturals
    blocks = block_information(term, code_link_base)
    if blocks:
        auxiliary['blocks'] = blocks
    expansions = operator_expansions(
        term, assigned_sizes, code_link_base, write_out, operator_references or {},
        operator_roles or {})
    if expansions:
        auxiliary['expansions'] = expansions
    return auxiliary


def legend_rows(
    term: fd.GeneralTerm, assigned_sizes: Mapping[str, int] | None,
) -> list[wst.AxisLegendRow]:
    '''One row per named axis of `term`, sorted by the text of its name. Two axes
    that read the same in every column, as the window axis of each of the eight
    attention modes does, make one row, and the row carries the uids of both. An
    axis whose uid carries no name, such as a concatenation of two others, is left
    out, because its parts are listed.'''
    rows: dict[tuple[str, str, int | None, str | None, str | None], wst.AxisLegendRow] = {}
    for axis in tutil.type_search(cat.Axis, term):
        name = axis.uid._name
        if name is None:
            continue
        row: wst.AxisLegendRow = {
            'latex': name.with_exponent(None).to_latex(),
            'text': name.to_bodies(),
            'size': axis_size(axis, assigned_sizes),
            'codeName': name.code_form,
            'sizeCodeName': size_code_name(axis),
            'uids': [],
        }
        held = rows.setdefault(
            (row['latex'], row['text'], row['size'], row['codeName'], row['sizeCodeName']),
            row)
        if axis.uid._id not in held['uids']:
            held['uids'].append(axis.uid._id)
    for row in rows.values():
        row['uids'].sort()
    return sorted(rows.values(), key=lambda row: (row['text'].lower(), row['text']))


def axis_size(axis: cat.Axis, assigned_sizes: Mapping[str, int] | None) -> int | None:
    '''The integer the size of `axis` comes to, evaluated under `assigned_sizes`
    where they are given and off the axis itself where they are not, or the
    integer its name carries as an exponent, or `None`.'''
    size = write_axis_exponents.evaluated_size_under(
        axis.local_size(), assigned_sizes or {})
    if size is not None:
        return size
    name = axis.uid._name
    exponent = None if name is None or name.exponent is None else name.exponent.to_bodies()
    return int(exponent) if exponent is not None and exponent.isdigit() else None


def size_code_name(axis: cat.Axis) -> str | None:
    '''The code form of the symbol that is the size of `axis`, where the size is
    one named symbol carrying one.'''
    size = axis.local_size()
    if isinstance(size, nm.FreeNumeric) and size.uid._name is not None:
        return size.uid._name.code_form
    return None


def natural_legend_rows(
    term: fd.GeneralTerm, assigned_sizes: Mapping[str, int] | None,
) -> list[wst.NaturalLegendRow]:
    '''One row per `cat.Natural` that is the datatype of an array or a weave of
    `term`, or that the quantisation of such a datatype holds, sorted by the latex of
    its bound. Two naturals with one bound make one row.'''
    rows: dict[tuple[str, str], wst.NaturalLegendRow] = {}
    for natural in naturals_of_arrays(term):
        bound = natural.max_value
        size = natural_size(bound, assigned_sizes)
        row: wst.NaturalLegendRow = {
            'latex': letter_formula_indices.with_braced_subscripts(bound.to_latex()),
            'size': None if size is None else str(size),
            'codeName': code_expression(bound),
            'key': natural_key(bound),
        }
        rows.setdefault((row['key'], row['latex']), row)
    return sorted(rows.values(), key=lambda row: row['latex'])


def naturals_of_arrays(term: fd.GeneralTerm) -> Iterator[cat.Natural]:
    carriers = (*tutil.type_search(cat.Array, term), *tutil.type_search(cat.Weave, term))
    for carrier in carriers:
        yield from tutil.type_search(cat.Natural, carrier.datatype)


def natural_key(bound: nm.Numeric) -> str:
    '''The structure of `bound`, written as tsncd writes it for the `Natural` of a
    wire in `src/data_structure_processing/find_naturals_by_key.ts`: a symbol as `#`
    and its uid, an integer as its digits, a sum, a product and a power as `+`, `*`
    and `^` followed by the keys of their parts in brackets, and any other numeric as
    `?`.'''
    match bound:
        case nm.FreeNumeric(uid=uid):
            return f'#{uid._id}'
        case nm.Integer(_value=value):
            return str(value)
        case nm.Addition(content=parts):
            return f'+({",".join(map(natural_key, parts))})'
        case nm.Multiplication(content=parts):
            return f'*({",".join(map(natural_key, parts))})'
        case nm.Power(base=base, exponent=exponent):
            return f'^({natural_key(base)},{natural_key(exponent)})'
    return '?'


def natural_size(bound: nm.Numeric, assigned_sizes: Mapping[str, int] | None) -> int | None:
    '''The integer `bound` comes to under `assigned_sizes`, or with each of its
    symbols taken at the integer written on its name as an exponent, or `None`.'''
    size = write_axis_exponents.evaluated_size_under(bound, assigned_sizes or {})
    if size is not None:
        return size
    written = {symbol: int(exponent)
               for symbol in tutil.type_search(nm.FreeNumeric, bound)
               if (exponent := exponent_written_on(symbol)) is not None}
    try:
        value = nm.evaluate_rational(bound, written)
    except (KeyError, nm.NotAnAffineForm, ZeroDivisionError):
        return None
    return value.numerator if value.denominator == 1 else None


def exponent_written_on(symbol: nm.FreeNumeric) -> str | None:
    name = symbol.uid._name
    exponent = None if name is None or name.exponent is None else name.exponent.to_bodies()
    return exponent if exponent is not None and exponent.isdigit() else None


def code_expression(bound: nm.Numeric) -> str | None:
    '''`bound` written as Python in the code names of its symbols, with a sum inside
    a product or a power bracketed, or `None` where a symbol of it carries no code
    name.'''
    match bound:
        case nm.FreeNumeric(uid=uid):
            return None if uid._name is None else uid._name.code_form
        case nm.Integer(_value=value):
            return str(value)
        case nm.Addition(content=parts):
            written = [code_expression(part) for part in parts]
            return None if None in written else ' + '.join(written)
        case nm.Multiplication(content=parts):
            return code_product(parts)
        case nm.Power(base=base, exponent=exponent):
            written_base = code_factor(base)
            written_exponent = code_factor(exponent)
            if written_base is None or written_exponent is None:
                return None
            return f'{written_base} ** {written_exponent}'
    return None


def code_product(factors: fd.Prod[nm.Numeric]) -> str | None:
    '''The factors multiplied, with a factor raised to the power -1 divided by.'''
    multiplied = [code_factor(factor) for factor in factors if not nm.is_reciprocal(factor)]
    divisors = [code_factor(factor.base) for factor in factors if nm.is_reciprocal(factor)]
    if None in multiplied or None in divisors:
        return None
    return ' / '.join([' * '.join(multiplied) or '1', *divisors])


def code_factor(factor: nm.Numeric) -> str | None:
    written = code_expression(factor)
    if written is None or not isinstance(factor, nm.Associative):
        return written
    return f'({written})'


def block_information(
    term: fd.GeneralTerm, code_link_base: str | None,
) -> dict[str, wst.BlockInformation]:
    '''The title, the formula, the description and the code references of every
    block of `term`, keyed by the uid of the block's tag as a string, which is how
    JSON keys an object. The formula and the description are lettered.'''
    information: dict[str, wst.BlockInformation] = {}
    for block in tutil.type_search(cat.Block, term):
        aesthetics = block.block_tag.aesthetics
        key = str(block.block_tag.uid._id)
        if aesthetics is None or key in information:
            continue
        lettered = lettered_texts(aesthetics.formula, aesthetics.description, block)
        information[key] = {
            'title': aesthetics.title,
            'formula': lettered.formula,
            'description': lettered.description,
            'references': [
                reference_record(reference, code_link_base)
                for reference in (aesthetics.references or ())],
            'indices': lettered.indices,
        }
    return information


@dataclass(frozen=True)
class LetteredTexts:
    '''A formula and a description with the indices of the formula lettered, and
    the indices the formula holds for every position of their axes, each guarded one
    with the positions of its axis that hold a value.'''
    formula: str | None
    description: str | None
    indices: list[wst.FormulaIndexRecord]


def lettered_texts(
    formula: str | None, description: str | None, shown_over: fd.GeneralTerm = (),
) -> LetteredTexts:
    '''`formula` and `description` lettered, with the indices of the formula matched
    to the axes of `shown_over`, the operator or the block the formula is shown
    over.'''
    if formula is None:
        return LetteredTexts(formula=None, description=description, indices=[])
    lettered = letter_formula_indices.lettered_formula(formula)
    return LetteredTexts(
        formula=lettered.formula,
        description=(None if description is None
                     else letter_formula_indices.lettered_description(
                         description, lettered)),
        indices=write_formula_index_ranges.index_records_with_ranges(
            lettered, tuple(tutil.type_search(cat.Axis, shown_over))))


def reference_record(
    reference: cat.CodeReference, code_link_base: str | None,
) -> wst.CodeReferenceRecord:
    url = reference_url(reference, code_link_base)
    return {
        'label': reference.label,
        'url': url,
        'path': reference.path,
        'line': reference.line,
        'endLine': reference.end_line,
        'icon': reference_icon(url),
    }


def reference_icon(url: str | None) -> str | None:
    '''The name of the icon drawn before a link to `url`, or `None` where the host
    of `url` has none.'''
    if url is None:
        return None
    return REFERENCE_ICONS.get(urllib.parse.urlsplit(url).hostname or '')


def reference_url(
    reference: cat.CodeReference, code_link_base: str | None,
) -> str | None:
    '''The url of `reference`: its own where it carries one, else its path under
    `code_link_base` with its lines appended in the form the base's scheme reads,
    `:line` for a `vscode://` base and `#Lstart-Lend` for a web one.'''
    if reference.url is not None:
        return reference.url
    if reference.path is None or code_link_base is None:
        return None
    base = code_link_base if code_link_base.endswith('/') else code_link_base + '/'
    url = base + reference.path.replace('\\', '/')
    if reference.line is None:
        return url
    if code_link_base.startswith(VSCODE_SCHEME):
        return f'{url}:{reference.line}'
    if reference.end_line is None or reference.end_line == reference.line:
        return f'{url}#L{reference.line}'
    return f'{url}#L{reference.line}-L{reference.end_line}'


def operator_expansions(
    term: fd.GeneralTerm,
    assigned_sizes: Mapping[str, int] | None,
    code_link_base: str | None,
    write_out: WriteOut,
    operator_references: OperatorReferences,
    operator_roles: OperatorRoles,
) -> dict[str, wst.OperatorExpansion]:
    '''The expansion of every `Broadcasted` of `term` whose operator the registry
    holds a row for and `write_out` writes out, keyed by the number tsncd's importer
    gives that `Broadcasted`. A `Broadcasted` the term holds at several places is
    written out once.'''
    expansions: dict[str, wst.OperatorExpansion] = {}
    written: dict[cat.Broadcasted, fd.GeneralTerm | None] = {}
    for number, broadcasted in enumerate(
            broadcast_occurrences.number_broadcasts_in_import_order(term)):
        row = standard_expansions.expansion_for(broadcasted.operator)
        if row is None:
            continue
        if broadcasted not in written:
            written_out = write_out(broadcasted)
            written[broadcasted] = (
                None if written_out is None or written_out == broadcasted
                else h2m.recycle(written_out))
        expanded = written[broadcasted]
        if expanded is None:
            continue
        name = broadcasted.operator.name
        lettered = lettered_texts(
            row.formula_of(broadcasted),
            description_with_role(
                broadcasted, row.description_of(broadcasted), operator_roles),
            broadcasted)
        expansions[str(number)] = {
            'operator': type(broadcasted.operator).__name__,
            'latex': None if name is None else name.to_latex(),
            'formula': lettered.formula,
            'description': lettered.description,
            'indices': lettered.indices,
            'expansion': term_json.TermJSONConverter.export_to_json(expanded),
            'auxiliary': auxiliary_information(
                expanded, assigned_sizes=assigned_sizes,
                code_link_base=code_link_base, with_legend=False,
                operator_references=operator_references,
                operator_roles=operator_roles),
            'references': [
                reference_record(reference, code_link_base)
                for reference in (
                    *role_references(broadcasted, operator_roles),
                    *references_of(broadcasted.operator, operator_references))],
        }
    return expansions


def description_with_role(
    broadcasted: cat.Broadcasted, description: str, operator_roles: OperatorRoles,
) -> str:
    '''The role `operator_roles` gives the operator of `broadcasted` by the text of
    its name, followed by `description`, and `description` alone where the table
    holds no role for that name.'''
    role = role_of(broadcasted, operator_roles)
    return description if role is None else f'{role.role} {description}'


def role_of(
    broadcasted: cat.Broadcasted, operator_roles: OperatorRoles,
) -> OperatorRole | None:
    name = broadcasted.operator.name
    return None if name is None else operator_roles.get(name.to_bodies())


def role_references(
    broadcasted: cat.Broadcasted, operator_roles: OperatorRoles,
) -> fd.Prod[cat.CodeReference]:
    role = role_of(broadcasted, operator_roles)
    return () if role is None else role.references


def references_of(
    operator: cat.Operator, operator_references: OperatorReferences,
) -> fd.Prod[cat.CodeReference]:
    '''The references `operator_references` holds for the nearest class of
    `operator`, found along the MRO as the registry finds a rule.'''
    for kind in type(operator).__mro__:
        if kind in operator_references:
            return operator_references[kind]
    return ()
