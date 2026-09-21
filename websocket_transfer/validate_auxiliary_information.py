# Claude Fable 5.1, effort 80.
'''Check the auxiliary information a figure sends beside its term.

    python websocket_transfer/validate_auxiliary_information.py

One `check_` function per claim, each printing one line, and a non-zero exit when any
fails. The claims are that the registry of standard expansions holds the four
operators and writes each out in its primitives, that the numbering of the
`Broadcasted` nodes reproduces the walk tsncd's importer makes over the exported JSON,
that the legend, the block information and the expansions are packaged as
`obsidian/05-backends/Diagram Wire Format.md` states, that an inspection box draws an
operator with its parameters on the tape and each read through a box that names the
weight, that the formula and the description of a box
are written from the operator it is shown over, with the role the model gives that
operator said first, that an operator with no expansion is wrapped
in a block drawn as the operator alone, that a row of the table of explanations can
write the explanation from the operator, that an operator fed from the tape is explained
inside its wrap, that an elementwise map opens a box only where its name or a sigmoid
hides part of its formula, that a plain box fed from the tape holds the tape as a seed
inside its block, that a named reindexing is wrapped in a block
drawn as the reindexing alone with a formula written from its rows, that the messages
carry the field only when
there is one, and that the notebook setting converts the term once and keys the
expansions off what is sent.
'''
from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import sys
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import algebra.registries.standard_expansions as standard_expansions  # noqa: E402
import construction_helpers as ch  # noqa: E402,F401
import data_structure.Category as cat  # noqa: E402
import data_structure.Numeric as nm  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_structure.Term as fd  # noqa: E402
import data_transfer.broadcast_occurrences as broadcast_occurrences  # noqa: E402
import data_transfer.term_json as term_json  # noqa: E402
import para.data_structure.Para as Para  # noqa: E402
import para.data_structure.ParaBlockOperator as para_block_operator  # noqa: E402
import para.data_structure.ParaWrap as para_wrap  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import term_utilities.code_references as code_references  # noqa: E402
import term_utilities.generate_config as gc  # noqa: E402
import term_utilities.term_utilities as tutil  # noqa: E402
import websocket_transfer.auxiliary_information as auxiliary_information  # noqa: E402
import websocket_transfer.send_morphism as send_morphism  # noqa: E402
import websocket_transfer.websockets_transfer as wst  # noqa: E402

import notebooks.display.advanced_display as advanced_display  # noqa: E402
import notebooks.display.cast_presentation as cast_presentation  # noqa: E402
import notebooks.display.expand_with_parameters as expand_with_parameters  # noqa: E402
import notebooks.display.explain_operators as explain_operators  # noqa: E402
import notebooks.display.explain_reindexings as explain_reindexings  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402
import notebooks.display.tape_presentation as tape_presentation  # noqa: E402


WEB_BASE = 'https://github.com/example/pyncd/blob/main/'
RELEASED_ON_HUGGING_FACE = 'https://huggingface.co/example/model/blob/0123abcd/model.py#L7'
VSCODE_BASE = 'vscode://file/C:/repositories/pyncd/'


def small_model() -> cat.Morphism:
    '''A block used twice, so one tag stands at two places, a repeated block, and a
    softmax outside any block, sized so the legend has integers to show.'''
    x = cat.RawAxis.named('x', code_form='tokens')
    m = cat.RawAxis.named('m', code_form='hidden')
    norm = cat.Block.template(
        ops.Linear.template((m,), (m,)) @ ops.Normalize.template((m,)),
        title='\\text{Norm block}', description='A linear map then an RMSNorm.',
        references=(code_references.source_of(small_model),
                    cat.CodeReference(label='released model', url='https://example.org/model.py#L1')))
    box = ops.BlockOperator.template(norm, 'NB')
    repeated = cat.Block.template(box @ box, title='\\text{Twice}', repetition=3)
    return send_morphism.to_morphism(
        (x >> repeated) @ (x >> ops.SoftMax.template()), recycle=True)


def importer_walk_operator_names(exported: str) -> list[str]:
    '''The operator class of every `Broadcasted` record of an exported term, in the
    order tsncd's `to_term` enters them: depth first over the fields as written, a
    reference followed into the repository the first time and returned from a cache
    afterwards.'''
    document = json.loads(exported)
    repository = document['uid_repository']
    seen: set[str] = set()
    names: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for member in value:
                walk(member)
            return
        if not isinstance(value, dict):
            return
        if '__ref__' in value:
            key = str(value['__ref__'])
            if key in seen:
                return
            seen.add(key)
            walk(repository[key])
            return
        if value.get('__type__') == 'Broadcasted':
            names.append(value['operator']['__type__'])
        for field, member in value.items():
            if field != '__type__':
                walk(member)

    walk(document['data'])
    return names


def check_the_registry_writes_out_five_operators() -> None:
    registered = standard_expansions.registered_operators()
    if set(registered) != {
            ops.SoftMax, ops.L1Norm, ops.L2Norm, ops.Normalize, ops.Linear}:
        raise AssertionError(f'registered {registered}')
    m = cat.RawAxis.named('m')
    for template, expected in (
            (ops.SoftMax.template(), {'Arithmetic', 'Einops', 'Rearrangement'}),
            (ops.L1Norm.template(), {'Arithmetic', 'Einops', 'Rearrangement'}),
            (ops.L2Norm.template((m,)), {'Arithmetic', 'Einops', 'Rearrangement'}),
            (ops.Normalize.template((m,)), {'Arithmetic', 'Einops', 'Rearrangement'})):
        if not standard_expansions.has_standard_expansion(template):
            raise AssertionError(f'{type(template.operator).__name__} has no expansion')
        expanded = standard_expansions.expand_standard(template)
        found = {type(t.operator).__name__ for t in tutil.type_search(cat.Broadcasted, expanded)}
        found |= {type(t).__name__ for t in tutil.type_search(cat.Rearrangement, expanded)}
        if not expected <= found:
            raise AssertionError(f'{type(template.operator).__name__} expands to {found}')
        row = standard_expansions.expansion_for(template.operator)
        if (row is None or not row.formula_of(template)
                or not row.description_of(template)):
            raise AssertionError(f'{type(template.operator).__name__} has no formula')
    normalize = standard_expansions.expand_standard(ops.Normalize.template((m,)))
    names = {b.operator.name.to_latex() for b in tutil.type_search(cat.Broadcasted, normalize)
             if isinstance(b.operator, ops.Arithmetic)}
    if not any('epsilon' in name and '|m|' in name for name in names):
        raise AssertionError(f'the RMSNorm writes the mean and the epsilon, got {names}')
    l2 = standard_expansions.expand_standard(ops.L2Norm.template((m,)))
    names = {b.operator.name.to_latex() for b in tutil.type_search(cat.Broadcasted, l2)
             if isinstance(b.operator, ops.Arithmetic)}
    if names != {'x^{2}', 'x^{-1/2}'}:
        raise AssertionError(
            f'the L2 norm squares and takes an inverse root, got {names}')
    linear = ops.Linear.template((m,), (m,))
    if standard_expansions.expand_standard(linear) != linear:
        raise AssertionError('the rule leaves a Linear whose weight is inside the operator')


def check_the_numbering_reproduces_the_importer_walk() -> None:
    model = small_model()
    numbered = [type(b.operator).__name__
                for b in broadcast_occurrences.number_broadcasts_in_import_order(model)]
    walked = importer_walk_operator_names(term_json.TermJSONConverter.export_to_json(model))
    if numbered != walked:
        raise AssertionError(f'numbered {numbered} against the walk {walked}')
    if numbered.count('BlockOperator') != 2 or numbered.count('Normalize') != 2:
        raise AssertionError(
            f'expected the box twice and its body twice, because a block carries no '
            f'uid and is written in place at each occurrence, got {numbered}')


def check_the_auxiliary_information_is_packaged() -> None:
    model = small_model()
    config = gc.NumericConfig.template(model)
    config.assign_values(m=64)
    auxiliary = auxiliary_information.auxiliary_information(
        model, assigned_sizes=config.assigned_integers_by_name(), code_link_base=WEB_BASE)
    legend = {row['text']: row for row in auxiliary['legend']}
    if set(legend) != {'m', 'x'}:
        raise AssertionError(f'legend rows {sorted(legend)}')
    if legend['m']['size'] != 64 or legend['x']['size'] is not None:
        raise AssertionError(f'sizes {legend}')
    if legend['m']['codeName'] != 'hidden' or legend['m']['sizeCodeName'] != 'hidden_size':
        raise AssertionError(f'code names {legend["m"]}')
    for text, row in legend.items():
        named = sorted({axis.uid._id for axis in tutil.type_search(cat.Axis, model)
                        if axis.uid._name is not None
                        and axis.uid._name.to_bodies() == text})
        if row['uids'] != named:
            raise AssertionError(f'uids of {text}: {row["uids"]} against {named}')
    blocks = auxiliary['blocks']
    tags = {str(b.block_tag.uid._id) for b in tutil.type_search(cat.Block, model)}
    if set(blocks) != tags or len(blocks) != 2:
        raise AssertionError(f'block keys {sorted(blocks)} against tags {sorted(tags)}')
    norm = next(b for b in blocks.values() if b['title'] == '\\text{Norm block}')
    source, released = norm['references']
    if not source['url'].startswith(WEB_BASE + 'websocket_transfer/validate_auxiliary_information.py#L'):
        raise AssertionError(f'source url {source["url"]}')
    if released['url'] != 'https://example.org/model.py#L1':
        raise AssertionError(f'released url {released["url"]}')
    expansions = auxiliary['expansions']
    numbered = broadcast_occurrences.number_broadcasts_in_import_order(model)
    for key, expansion in expansions.items():
        if type(numbered[int(key)].operator).__name__ != expansion['operator']:
            raise AssertionError(f'key {key} names {expansion["operator"]}')
        if 'legend' in expansion['auxiliary']:
            raise AssertionError('a nested auxiliary carries no legend')
        json.loads(expansion['expansion'])
    if sorted(e['operator'] for e in expansions.values()) != ['Normalize', 'Normalize', 'SoftMax']:
        raise AssertionError(f'expansions {[e["operator"] for e in expansions.values()]}')
    if any(e['references'] for e in expansions.values()):
        raise AssertionError('an expansion carries no reference unless one is given')
    if norm['formula'] is not None or released['icon'] is not None:
        raise AssertionError(f'a block with no formula and a host with no icon: {norm}')
    referenced = auxiliary_information.auxiliary_information(
        model, operator_references={ops.Normalize: (cat.CodeReference(
            label='released norm', url=RELEASED_ON_HUGGING_FACE),)})
    icons = {r['icon'] for e in referenced['expansions'].values() for r in e['references']}
    if icons != {'huggingface'}:
        raise AssertionError(f'a link into Hugging Face carries its icon, got {icons}')
    vscode = auxiliary_information.auxiliary_information(model, code_link_base=VSCODE_BASE)
    vscode_norm = next(b for b in vscode['blocks'].values() if b['title'] == '\\text{Norm block}')
    if ':' not in vscode_norm['references'][0]['url'].rsplit('/', 1)[1]:
        raise AssertionError(f'vscode url {vscode_norm["references"][0]["url"]}')
    unlinked = auxiliary_information.auxiliary_information(model)
    unlinked_norm = next(b for b in unlinked['blocks'].values() if b['title'] == '\\text{Norm block}')
    if unlinked_norm['references'][0]['url'] is not None:
        raise AssertionError('a path with no base is unlinked')


def check_an_operator_is_written_out_with_its_parameters_on_the_tape() -> None:
    m = cat.RawAxis.named('m')
    d = cat.RawAxis.named('d')
    absorbed = tape_presentation.TapePresentation.ABSORBED
    boxed = tape_presentation.TapePresentation.BOXED
    learned_array = cat.Broadcasted(
        operator=ops.Linear(name=fd.DynamicName('sink')), input_weaves=(),
        output_weaves=(cat.Weave(cat.Reals(), (m,)),), reindexings=())
    for template, slots, written_operators in (
            (ops.Normalize.template((m,)), {'\\gamma'}, {ops.Arithmetic, ops.Einops}),
            (ops.Linear.template((m,), (d,)), {'W'}, {ops.Einops}),
            (ops.Linear.template((m,), (d,), bias=True), {'W', 'b'},
             {ops.Einops, ops.AdditionOp}),
            (learned_array, {'sink'}, set())):
        written = expand_with_parameters.expanded_with_grabbed_parameters(template, boxed)
        grabs = list(tutil.type_search(Para.Grab, written))
        if {grab.tape.uid._name.body for grab in grabs} != slots:
            raise AssertionError(f'{type(template.operator).__name__} grabs {grabs}')
        weight_boxes = [b for b in tutil.type_search(cat.Broadcasted, written)
                        if isinstance(b.operator, ops.Linear)]
        if ({box.operator.name for box in weight_boxes}
                != {grab.tape.uid._name for grab in grabs}
                or any(tuple(box.dom()) != tuple(box.cod()) or len(box.dom()) != 1
                       for box in weight_boxes)):
            raise AssertionError(
                f'{type(template.operator).__name__} reads each parameter through a '
                f'box named after its slot, and it holds {weight_boxes}')
        operators = {type(b.operator) for b in tutil.type_search(cat.Broadcasted, written)
                     if b not in weight_boxes}
        if operators != written_operators:
            raise AssertionError(
                f'{type(template.operator).__name__} is written out as {operators}')
        if written.dom() != template.dom() or written.cod() != template.cod():
            raise AssertionError(
                f'{type(template.operator).__name__} written out keeps its domain and '
                f'codomain, and it has {written.dom()} and {written.cod()}')
        wrapped = expand_with_parameters.expanded_with_grabbed_parameters(
            template, absorbed)
        wraps = list(tutil.type_search(para_wrap.ParaWrap, wrapped))
        if len(wraps) != len(slots) or any(
                not isinstance(wrap.body, cat.Broadcasted)
                or not isinstance(wrap.body.operator, ops.Linear) or wrap.dom()
                for wrap in wraps):
            raise AssertionError(
                f'{type(template.operator).__name__} absorbs each grab onto its weight '
                f'box, which then reads nothing on a wire, and it holds {wraps}')
    for parameters in expand_with_parameters.ExpandedParameters:
        interactive = notebook_diagrams.DiagramSettings(
            mode=notebook_diagrams.DiagramMode.OFF,
            advanced_display=advanced_display.AdvancedDisplay.INTERACTIVE,
            expanded_parameters=parameters)
        _, _, auxiliary = notebook_diagrams.package_auxiliary(small_model(), interactive)
        written_out = sorted(e['operator'] for e in auxiliary['expansions'].values())
        if written_out != ['Linear', 'Linear', 'Normalize', 'Normalize', 'SoftMax']:
            raise AssertionError(f'the notebook setting writes out {written_out}')


def check_an_operator_is_written_out_with_weight_arrays() -> None:
    m = cat.RawAxis.named('m')
    d = cat.RawAxis.named('d')
    learned_array = cat.Broadcasted(
        operator=ops.Linear(name=fd.DynamicName('sink')), input_weaves=(),
        output_weaves=(cat.Weave(cat.Reals(), (m,)),), reindexings=())
    for template, slots, written_operators in (
            (ops.Normalize.template((m,)), {'\\gamma'}, {ops.Arithmetic, ops.Einops}),
            (ops.Linear.template((m,), (d,)), {'W'}, {ops.Einops}),
            (ops.Linear.template((m,), (d,), bias=True), {'W', 'b'},
             {ops.Einops, ops.AdditionOp}),
            (learned_array, {'sink'}, set())):
        written = expand_with_parameters.expanded_with_weight_arrays(template)
        if tape_presentation.holds_tape_operations(written):
            raise AssertionError(
                f'{type(template.operator).__name__} written out with weight arrays '
                'reads no tape')
        weight_arrays = [b for b in tutil.type_search(cat.Broadcasted, written)
                         if isinstance(b.operator, ops.Linear)]
        if ({array.operator.name.body for array in weight_arrays} != slots
                or any(array.dom() or len(array.cod()) != 1 for array in weight_arrays)):
            raise AssertionError(
                f'{type(template.operator).__name__} reads each parameter from a '
                f'weight array named after it, and it holds {weight_arrays}')
        operators = {type(b.operator) for b in tutil.type_search(cat.Broadcasted, written)
                     if b not in weight_arrays}
        if operators != written_operators:
            raise AssertionError(
                f'{type(template.operator).__name__} is written out as {operators}')
        if written.dom() != template.dom() or written.cod() != template.cod():
            raise AssertionError(
                f'{type(template.operator).__name__} written out keeps its domain and '
                f'codomain, and it has {written.dom()} and {written.cod()}')
    if (notebook_diagrams.DiagramSettings().expanded_parameters
            is not expand_with_parameters.ExpandedParameters.WEIGHT_ARRAYS):
        raise AssertionError('an inspection box draws weight arrays unless asked otherwise')


def check_a_formula_is_written_from_the_operator_it_is_shown_over() -> None:
    m = cat.RawAxis.named('m')
    d = cat.RawAxis.named('d')
    row = standard_expansions.expansion_for(ops.Linear())
    plain = row.formula_of(ops.Linear.template((m,), (d,), 'Q'))
    if plain != 'y[i_{d}] = \\sum_{i_{m} \\in m} x[i_{m}]\\, W_\\bold{Q}[i_{m}, i_{d}]':
        raise AssertionError(f'the formula of a map with no bias: {plain}')
    biased_map = ops.Linear.template((m,), (d,), 'Q', bias=True)
    biased = row.formula_of(biased_map)
    if biased != plain + ' + b_\\bold{Q}[i_{d}]':
        raise AssertionError(f'the formula of a map with a bias: {biased}')
    if 'bias' in row.description_of(ops.Linear.template((m,), (d,), 'Q')):
        raise AssertionError('a map with no bias is described with no bias')
    if 'bias' not in row.description_of(biased_map):
        raise AssertionError('a map with a bias is described with its bias')
    onto_itself = row.formula_of(ops.Linear.template((m,), (m,), 'Q'))
    if "W_\\bold{Q}[i_{m}, i_{m'}]" not in onto_itself:
        raise AssertionError(f'a map from an axis onto itself primes one index: {onto_itself}')
    learned_array = cat.Broadcasted(
        operator=ops.Linear(name=fd.DynamicName('sink')), input_weaves=(),
        output_weaves=(cat.Weave(cat.Reals(), (m,)),), reindexings=())
    if row.formula_of(learned_array) != 'y[i_{m}] = sink[i_{m}]':
        raise AssertionError(f'the formula of a learned array: {row.formula_of(learned_array)}')
    normalize = standard_expansions.expansion_for(ops.Normalize()).formula_of(
        ops.Normalize.template((m, d)))
    if ('\\mathrm{RMSNorm}_{m, d}' not in normalize
            or '\\sum_{i_{m} \\in m,\\, i_{d} \\in d} x[i_{m}, i_{d}]^{2}' not in normalize
            or '\\frac{1}{|m||d|}' not in normalize):
        raise AssertionError(f'the formula of an RMSNorm over two axes: {normalize}')
    declared = cat.CodeReference(label='declared', url=RELEASED_ON_HUGGING_FACE)
    general = cat.CodeReference(label='linear', url=RELEASED_ON_HUGGING_FACE)
    auxiliary = auxiliary_information.auxiliary_information(
        send_morphism.to_morphism(biased_map, recycle=True),
        write_out=lambda node: expand_with_parameters.expanded_with_grabbed_parameters(
            node, tape_presentation.TapePresentation.ABSORBED),
        operator_roles={'Q': auxiliary_information.OperatorRole(
            role='The query projection.', references=(declared,))},
        operator_references={ops.Linear: (general,)})
    expansion, = auxiliary['expansions'].values()
    if not expansion['description'].startswith('The query projection. A learned linear map.'):
        raise AssertionError(f'the role comes first: {expansion["description"]}')
    if [r['label'] for r in expansion['references']] != ['declared', 'linear']:
        raise AssertionError(f'the references of the role come first: {expansion["references"]}')
    if expansion['formula'] != biased:
        raise AssertionError(f'the expansion carries the generated formula: {expansion["formula"]}')


def check_an_explained_operator_is_wrapped_in_a_block_drawn_in_place() -> None:
    model = small_model()
    explanations = {ops.SoftMax: explain_operators.OperatorExplanation(
        formula='e^{s} / \\sum e^{s}', description='A softmax.',
        references=(cat.CodeReference(label='released', url=RELEASED_ON_HUGGING_FACE),))}
    if explain_operators.present(model, None) is not model:
        raise AssertionError('no table leaves the term as it stands')
    explained = explain_operators.present(model, explanations)
    in_place = [b for b in tutil.type_search(cat.Block, explained)
                if b.block_tag.aesthetics is not None
                and b.block_tag.aesthetics.drawing is cat.BlockDrawing.BODY_IN_PLACE]
    if len(in_place) != 1 or not isinstance(in_place[0].body.operator, ops.SoftMax):
        raise AssertionError(f'{len(in_place)} blocks drawn in place')
    if explained.dom() != model.dom() or explained.cod() != model.cod():
        raise AssertionError('the wrapped term keeps its domain and its codomain')
    if any(b.block_tag.aesthetics is not None and b.block_tag.aesthetics.drawing is not None
           for b in tutil.type_search(cat.Block, model)):
        raise AssertionError('the term handed over is not edited')
    record = auxiliary_information.auxiliary_information(explained)['blocks'][
        str(in_place[0].block_tag.uid._id)]
    if record['formula'] != 'e^{s} / \\sum e^{s}' or record['title'] != 'SoftMax':
        raise AssertionError(f'the block record carries the explanation: {record}')
    if [r['icon'] for r in record['references']] != ['huggingface']:
        raise AssertionError(f'the references of the explanation: {record["references"]}')
    json.loads(term_json.TermJSONConverter.export_to_json(explained))
    legend_only = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        advanced_display=advanced_display.AdvancedDisplay.LEGEND,
        operator_explanations=explanations)
    sent, _, _ = notebook_diagrams.package_auxiliary(model, legend_only)
    if any(b.block_tag.aesthetics is not None and b.block_tag.aesthetics.drawing is not None
           for b in tutil.type_search(cat.Block, sent)):
        raise AssertionError('a figure that opens no box wraps no operator')


def check_an_explanation_is_written_from_the_operator() -> None:
    m = cat.RawAxis.named('m')
    scores = cat.Array(cat.Reals(), (m,))

    def explain_softmax(target: cat.Broadcasted) -> explain_operators.OperatorExplanation:
        return explain_operators.OperatorExplanation(
            formula=f'{len(target.dom())} operand', description='A softmax.')

    slot = Para.new_slot()
    fed_from_the_tape = Para.Grab(tape=slot, size=scores) @ ops.SoftMax.template((m,))
    presented = tape_presentation.present(
        fed_from_the_tape, tape_presentation.TapePresentation.ABSORBED)
    wrap, = tutil.type_search(para_wrap.ParaWrap, presented)
    if not isinstance(wrap.body.operator, ops.SoftMax):
        raise AssertionError('the grab is absorbed onto the softmax it feeds')
    explained = explain_operators.present(presented, {ops.SoftMax: explain_softmax})
    wrap, = tutil.type_search(para_wrap.ParaWrap, explained)
    block = wrap.body.operator.block
    if (block.block_tag.aesthetics.drawing is not cat.BlockDrawing.BODY_IN_PLACE
            or block.block_tag.aesthetics.formula != '1 operand'):
        raise AssertionError('an operator fed from the tape is explained inside its wrap')
    if wrap.dom() != presented.dom() or wrap.cod() != presented.cod():
        raise AssertionError('the wrap keeps its ports over the explaining block')
    unexplained = explain_operators.present(presented, {ops.SoftMax: lambda target: None})
    if unexplained is not presented:
        raise AssertionError('a row that returns no explanation leaves the operator alone')

    explain = explain_operators.explain_named_arithmetic({'\\sigma': 'The gate.'})
    exponential = ops.Arithmetic.template(nm.E ** nm.x)
    sigmoid = ops.Arithmetic.template(nm.Sigmoid(nm.x), name='\\sigma')
    root_of_softplus = ops.Arithmetic.template(
        nm.Power.template(nm.Logarithm.template(nm.E, nm.E ** nm.x + nm.Integer(1)),
                          nm.Integer(1) / nm.Integer(2)), name='s')
    if explain(exponential) is not None:
        raise AssertionError('a map named after its formula opens no box')
    gate = explain(sigmoid)
    if gate is None or gate.formula != 'y = \\sigma(x)':
        raise AssertionError(f'a sigmoid keeps its symbol: {gate}')
    if not gate.description.startswith('The gate. An elementwise map.'):
        raise AssertionError(f'the role of a map comes first: {gate.description}')
    hidden = explain(root_of_softplus)
    if hidden is None or hidden.formula != 'y = \\sqrt{\\ln(1 + e^{x})}':
        raise AssertionError(f'a map named more shortly than its formula: {hidden}')


def check_a_taped_box_holds_its_tape_as_a_seed() -> None:
    m = cat.RawAxis.named('m')
    d = cat.RawAxis.named('d')
    keys = cat.Array(cat.Reals(), (m, d))
    queries = cat.Array(cat.Reals(), (m, d))
    slot = Para.new_slot()
    scoring = cat.Block.template(
        ops.Einops.template('m d, m d -> m'), title='\\text{Scores}',
        description='One score per position.')
    box = ops.BlockOperator.template(scoring, 'Sco')
    model = (cat.ProdObject((queries,)).identity() * Para.Grab(tape=slot, size=keys)) @ box
    absorbed = tape_presentation.present(
        model, tape_presentation.TapePresentation.ABSORBED)
    wrap, = (w for w in tutil.type_search(para_wrap.ParaWrap, absorbed)
             if isinstance(w.body, cat.Broadcasted)
             and isinstance(w.body.operator, ops.BlockOperator))
    operator = wrap.body.operator
    if not isinstance(operator, para_block_operator.ParaBlockOperator):
        raise AssertionError('a plain box fed from the tape becomes a ParaBlockOperator')
    if wrap.grabs != (slot, None) or tuple(wrap.dom()) != (queries,):
        raise AssertionError(f'the grabbed operand leads the box: {wrap.grabs}')
    inside = {entry for inner in tutil.type_search(para_wrap.ParaWrap, operator.block)
              for entry in inner.grabs if entry is not None}
    if inside != {slot}:
        raise AssertionError('the body of the box reads the slot from the tape')
    if operator.block.block_tag.uid == scoring.block_tag.uid:
        raise AssertionError('the seeded block takes a tag of its own')
    if tape_presentation.holds_bare_seeds(operator.block):
        raise AssertionError('no seed is left bare inside the block of a rebuilt box')
    boxed = tape_presentation.present(model, tape_presentation.TapePresentation.BOXED)
    if boxed is not model:
        raise AssertionError('the boxed presentation leaves the term as it stands')
    information = auxiliary_information.block_information(absorbed, None)
    if information[str(operator.block.block_tag.uid._id)]['description'] != (
            'One score per position.'):
        raise AssertionError('the seeded block is described as the block it came from')


def check_a_named_reindexing_is_wrapped_in_a_block_drawn_in_place() -> None:
    x = cat.RawAxis.named('x')
    w = cat.RawAxis.named('w')
    m = cat.RawAxis.named('m')
    window = cat.StrideMorphism(
        _dom=(x, w),
        _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(2)),),
        name=fd.DynamicName('win'))
    formula = explain_reindexings.reindexing_formula(window)
    if formula != 'y[i_{x}, i_{w}] = x[i_{x} - i_{w} + 2]':
        raise AssertionError(f'the formula of a reindexing: {formula}')
    view = ops.View.template(
        reindexing=(window, cat.ProdObject((m,)).identity()), name='win')
    model = send_morphism.to_morphism(view @ ops.SoftMax.template(), recycle=True)
    explanations = {'win': explain_reindexings.ReindexingExplanation(
        description='A window.',
        references=(cat.CodeReference(label='released', url=RELEASED_ON_HUGGING_FACE),))}
    if explain_reindexings.present(model, None) is not model:
        raise AssertionError('no table leaves the term as it stands')
    explained = explain_reindexings.present(model, explanations)
    blocks = [b for b in tutil.type_search(cat.Block, explained)
              if isinstance(b.body, cat.StrideMorphism)]
    if len(blocks) != 1 or blocks[0].body != window:
        raise AssertionError(f'{len(blocks)} reindexings are wrapped')
    if blocks[0].block_tag.aesthetics.drawing is not cat.BlockDrawing.BODY_IN_PLACE:
        raise AssertionError('the block of a reindexing is drawn in place')
    if explained.dom() != model.dom() or explained.cod() != model.cod():
        raise AssertionError('the wrapped term keeps its domain and its codomain')
    if list(tutil.type_search(cat.Block, model)):
        raise AssertionError('the term handed over is not edited')
    numbered = broadcast_occurrences.number_broadcasts_in_import_order
    if ([type(b.operator) for b in numbered(explained)]
            != [type(b.operator) for b in numbered(model)]):
        raise AssertionError('the wrapping keeps the numbering of the operators')
    json.loads(term_json.TermJSONConverter.export_to_json(explained))
    interactive = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        advanced_display=advanced_display.AdvancedDisplay.INTERACTIVE,
        reindexing_explanations=explanations)
    sent, _, auxiliary = notebook_diagrams.package_auxiliary(model, interactive)
    wrapped, = (b for b in tutil.type_search(cat.Block, sent)
                if isinstance(b.body, cat.StrideMorphism))
    record = auxiliary['blocks'][str(wrapped.block_tag.uid._id)]
    if (record['title'], record['formula'], record['description']) != (
            'win', formula, 'A window.'):
        raise AssertionError(f'the block record carries the explanation: {record}')
    if sorted(e['operator'] for e in auxiliary['expansions'].values()) != ['SoftMax']:
        raise AssertionError('the expansions are assembled before the reindexings are wrapped')
    legend_only = dataclasses.replace(
        interactive, advanced_display=advanced_display.AdvancedDisplay.LEGEND)
    sent, _, _ = notebook_diagrams.package_auxiliary(model, legend_only)
    if list(tutil.type_search(cat.Block, sent)):
        raise AssertionError('a figure that opens no box wraps no reindexing')



def check_a_cast_is_explained_by_the_quantisations_it_reads_and_writes() -> None:
    m = cat.RawAxis.named('m')
    cast = Quantization.TypeConvert.over_shape(
        Quantization.BF16, Quantization.E4M3, (m,), 'cast')
    row = cast_presentation.cast_explanation(cast)
    if row.formula != r'\mathtt{BF16} \to \mathtt{E4M3}':
        raise AssertionError(f'the formula of a cast: {row.formula}')
    if 'BF16' not in row.description or 'E4M3' not in row.description:
        raise AssertionError(f'the description names both formats: {row.description}')
    load = Quantization.TypeConvert.over_shape(
        Quantization.BF16, Quantization.BF16, (m,), 'load')
    if cast_presentation.cast_explanation(load) is not None:
        raise AssertionError('a conversion carrying one quantisation on both sides is no cast')
    thin = cast_presentation.present(cast, cast_presentation.CastPresentation.THIN)
    interactive = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        advanced_display=advanced_display.AdvancedDisplay.INTERACTIVE)
    sent, _, auxiliary = notebook_diagrams.package_auxiliary(thin, interactive)
    wrapped, = tutil.type_search(cat.Block, sent)
    if wrapped.block_tag.aesthetics.drawing is not cat.BlockDrawing.BODY_IN_PLACE:
        raise AssertionError('the block explaining a cast is drawn in place')
    record = auxiliary['blocks'][str(wrapped.block_tag.uid._id)]
    if record['formula'] != row.formula:
        raise AssertionError(f'the block record carries the explanation: {record}')
    its_own = dataclasses.replace(interactive, operator_explanations={
        Quantization.TypeConvert: explain_operators.OperatorExplanation(
            formula='y = x', description='The table of the model is kept.')})
    _, _, auxiliary = notebook_diagrams.package_auxiliary(thin, its_own)
    if [r['description'] for r in auxiliary['blocks'].values()] != [
            'The table of the model is kept.']:
        raise AssertionError("a model's own row for a conversion is kept")
    legend_only = dataclasses.replace(
        interactive, advanced_display=advanced_display.AdvancedDisplay.LEGEND)
    sent, _, _ = notebook_diagrams.package_auxiliary(thin, legend_only)
    if list(tutil.type_search(cat.Block, sent)):
        raise AssertionError('a figure that opens no box wraps no cast')


def check_the_messages_carry_the_field_only_when_given() -> None:
    settings = send_morphism.display_settings(legend=True, inspectionBoxes=True)
    if settings != {'legend': True, 'inspectionBoxes': True}:
        raise AssertionError(f'settings {settings}')
    if send_morphism.display_settings() != {}:
        raise AssertionError('an unset flag is left out of the message')
    if send_morphism.display_settings(axisHover=wst.AxisHover.EVERYWHERE) != {
            'axisHover': 'everywhere'}:
        raise AssertionError('the axis hover setting is sent as its value')
    if send_morphism.display_settings(axisLabelFontSize=0.9) != {'axisLabelFontSize': 0.9}:
        raise AssertionError('the axis label font size is sent as it is given')
    message = {'msgType': 'dataUpdate', 'data': '{}', 'settings': {}}
    if wst.with_auxiliary(message, None) is not message:
        raise AssertionError('no auxiliary leaves the message as it stands')
    if 'auxiliary' not in wst.with_auxiliary(message, {'legend': []}):
        raise AssertionError('the auxiliary field is added')

    async def relay() -> tuple[wst.Message, wst.Message]:
        server = wst.DataServer()
        information = wst.HandlerInformation(socket=None)
        _, response = await server.process_message(
            {'msgType': 'dataUpdate', 'data': '{}', 'settings': {'legend': True},
             'auxiliary': {'legend': [{'latex': 'x', 'text': 'x', 'size': 4,
                                       'codeName': None, 'sizeCodeName': None,
                                       'uids': [1]}]}},
            information)
        _, held = await server.process_message({'msgType': 'dataRequest'}, information)
        return response, held

    response, held = asyncio.run(relay())
    if response != {'msgType': 'DataReceived'}:
        raise AssertionError(f'response {response}')
    if held.get('auxiliary', {}).get('legend', [{}])[0].get('size') != 4:
        raise AssertionError(f'the server holds the auxiliary field: {held}')


def check_the_notebook_setting_converts_once() -> None:
    model = small_model()
    off = notebook_diagrams.DiagramSettings(mode=notebook_diagrams.DiagramMode.OFF)
    term, settings, auxiliary = notebook_diagrams.package_auxiliary(model, off)
    if term is not model or auxiliary is not None or settings is not off:
        raise AssertionError('OFF sends the term as it stands and no auxiliary field')
    interactive = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF,
        advanced_display=advanced_display.AdvancedDisplay.INTERACTIVE,
        code_link_base=WEB_BASE)
    term, settings, auxiliary = notebook_diagrams.package_auxiliary(model, interactive)
    if settings.recycle or auxiliary is None:
        raise AssertionError('INTERACTIVE converts here and sends the auxiliary field')
    if set(auxiliary) != {'legend', 'blocks', 'expansions'}:
        raise AssertionError(f'auxiliary parts {sorted(auxiliary)}')
    if advanced_display.draws_legend(advanced_display.AdvancedDisplay.OFF) is not None:
        raise AssertionError('OFF leaves the legend flag out')
    if advanced_display.opens_inspection_boxes(advanced_display.AdvancedDisplay.LEGEND) is not None:
        raise AssertionError('LEGEND leaves the inspection boxes flag out')
    if advanced_display.opens_inspection_boxes(advanced_display.AdvancedDisplay.INTERACTIVE) is not True:
        raise AssertionError('INTERACTIVE sets the inspection boxes flag')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_registry_writes_out_five_operators,
    check_the_numbering_reproduces_the_importer_walk,
    check_the_auxiliary_information_is_packaged,
    check_an_operator_is_written_out_with_its_parameters_on_the_tape,
    check_an_operator_is_written_out_with_weight_arrays,
    check_a_formula_is_written_from_the_operator_it_is_shown_over,
    check_an_explained_operator_is_wrapped_in_a_block_drawn_in_place,
    check_an_explanation_is_written_from_the_operator,
    check_a_taped_box_holds_its_tape_as_a_seed,
    check_a_named_reindexing_is_wrapped_in_a_block_drawn_in_place,
    check_a_cast_is_explained_by_the_quantisations_it_reads_and_writes,
    check_the_messages_carry_the_field_only_when_given,
    check_the_notebook_setting_converts_once,
)


def main() -> int:
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'{check.__name__}: ok')
        except Exception as error:
            failures += 1
            print(f'{check.__name__}: FAILED {type(error).__name__}: {error}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
