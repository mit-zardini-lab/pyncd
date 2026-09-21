# Claude Fable 5.1, effort 80.
'''Packaging what an interactive figure shows beside the term it draws.

tsncd does no algebra, so everything an inspection box or a legend shows has to be
computed here and sent beside the term, as the `auxiliary` field of a `dataUpdate` or a
`renderRequest`. `obsidian/05-backends/Diagram Wire Format.md` states the field and
`obsidian/05-backends/Advanced Display.md` what tsncd does with it. Three things are
packaged.

The legend lists every named axis of the term with the integer its size comes to and
the code form its name carries. The size is read off the axis where a configuration has
sized the term, and evaluated under `assigned_sizes` where the term is symbolic and the
assignments are given beside it, as `notebooks.display.axis_sizes` reads them.

The block information carries, for every block of the term keyed by its tag's uid, the
title, the formula, the description and the code references its aesthetics hold. A
reference with no url of its own is linked from its path under `code_link_base`, and
left unlinked where no base is given. A reference whose url is on a host
`REFERENCE_ICONS` lists carries the name of that host's icon, which tsncd draws before
the link.

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

`algebra.operator_expansion` is imported for its registrations. The registry is filled
by the decorators that module applies to its rules, so the table is empty until it has
been imported.
'''
from __future__ import annotations

import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import algebra.operator_expansion as operator_expansion  # noqa: F401
import algebra.registries.standard_expansions as standard_expansions
import algebra.write_axis_exponents as write_axis_exponents
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import data_transfer.broadcast_occurrences as broadcast_occurrences
import data_transfer.term_json as term_json
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.term_utilities as tutil
import websocket_transfer.websockets_transfer as wst


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
    '''The `auxiliary` field for `term`, holding the legend where `with_legend` asks
    for it, the information of every block, and the expansion of every operator
    `write_out` writes out. A part with nothing in it is left out of the field.'''
    auxiliary: wst.DiagramAuxiliary = {}
    if with_legend:
        auxiliary['legend'] = legend_rows(term, assigned_sizes)
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


def block_information(
    term: fd.GeneralTerm, code_link_base: str | None,
) -> dict[str, wst.BlockInformation]:
    '''The title, the formula, the description and the code references of every
    block of `term`, keyed by the uid of the block's tag as a string, which is how
    JSON keys an object.'''
    information: dict[str, wst.BlockInformation] = {}
    for block in tutil.type_search(cat.Block, term):
        aesthetics = block.block_tag.aesthetics
        key = str(block.block_tag.uid._id)
        if aesthetics is None or key in information:
            continue
        information[key] = {
            'title': aesthetics.title,
            'formula': aesthetics.formula,
            'description': aesthetics.description,
            'references': [
                reference_record(reference, code_link_base)
                for reference in (aesthetics.references or ())],
        }
    return information


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
        expansions[str(number)] = {
            'operator': type(broadcasted.operator).__name__,
            'latex': None if name is None else name.to_latex(),
            'formula': row.formula_of(broadcasted),
            'description': description_with_role(
                broadcasted, row.description_of(broadcasted), operator_roles),
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
