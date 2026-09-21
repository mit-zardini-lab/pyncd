'''Merging and disentangling chained contractions.

Where one `Einops` feeds another on a raw, unkernelized expression, `merge_einops`
folds the two into one combined contraction and `disentangle_einops` splits that
contraction back into the components its segments fall into. Both rules are applied
unconditionally. `einops_cost` is the cost that justifies applying them, and it is not
consulted to choose between candidates.

Everything here reads `Einops.signature`, the weaves and the reindexings. No part of it
manipulates the `'a b, c b -> a c'` string a template is written with.

Two searches drive the same two rules. `rearrange_einops` sweeps a hypergraph once and
rebuilds it flat. `rearrange_local` offers `merge_rule` to
`algebra.merge_into_consumer`, which runs to a fixed point and leaves blocks standing.
`merge_reindexings_and_einops` alternates `reindexing_absorption.absorb`, offered to
the search that copies a node over its fan-out, with `merge_rule`, offered to the
search that folds a producer read once, until neither changes the graph. It is the
simplification the `para` notebooks apply to each pass.
'''

from __future__ import annotations
from typing import Iterable
import itertools
import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import utilities.utilities as util
import term_utilities.term_utilities as tutils
import construction_helpers.simple_helper as chsh
import algebra.merge_into_consumer as merge_into_consumer
import algebra.reindexing_absorption as reindexing_absorption


def segment_group_axes(
    operator: ops.Einops,
    weaves: fd.Prod[cat.Weave],
) -> dict[int, cat.RawAxis]:
    '''
    Maps each contraction group number in `operator.signature` to the
    RawAxis it refers to. `.signature[i]` lists, in positional order and
    skipping TILED positions, the group each non-tiled axis of
    `weaves[i]` belongs to, so zipping the signature against the
    non-TILED entries of the weave (in order) recovers the axis for each
    group, with no string parsing involved.
    '''
    axes: dict[int, cat.RawAxis] = {}
    for segment_sig, weave in zip(operator.signature, weaves):
        non_tiled = [ax for ax in weave._shape if ax is not cat.WeaveMode.TILED]
        for group, axis in zip(segment_sig, non_tiled):
            axes.setdefault(group, axis)
    return axes


def einops_cost(target: cat.Broadcasted) -> nm.Numeric:
    '''
    The size of the degree, multiplied by the size of the contracted axes.

    The degree is the shared broadcast and output shape, from
    `Broadcasted.degree()`. The contracted axes are every axis referenced
    anywhere in `.signature`. The cost justifies disentangling unconditionally
    rather than searching over candidates for the cheaper one.
    '''
    assert isinstance(target.operator, ops.Einops)
    degree_sizes = [axis.local_size() for axis in target.degree()]
    stream_sizes = [
        axis.local_size()
        for axis in segment_group_axes(target.operator, target.input_weaves).values()
    ]
    return nm.Multiplication.template(*degree_sizes, *stream_sizes)


def merge_einops(
    producer: cat.Broadcasted,
    consumer: cat.Broadcasted,
    feed_index: int,
) -> cat.Broadcasted:
    '''Fold a producer into its consumer, returning one combined `Einops`.

    The producer's single output feeds the consumer's input segment at
    `feed_index`. The function is scoped to a producer whose output is a pure
    pass-through, meaning every output position is TILED.

    The producer's output is an array over its own degree with every position
    TILED, plugged into the consumer's feed slot position for position. The
    feed slot therefore assigns each of the producer's degree axes a role.
    The axis either stays a broadcast axis in the consumer's degree, or the
    consumer contracts it into one of its groups.

    That role is computed once per axis of the producer's degree, and every
    input segment of the producer is rewritten through it. A broadcast axis,
    such as attention's query axis `q`, then threads through every segment
    alike, with no special case for a carrier segment.
    '''
    producer_op = producer.operator
    consumer_op = consumer.operator
    assert isinstance(producer_op, ops.Einops)
    assert isinstance(consumer_op, ops.Einops)
    assert len(producer.output_weaves) == 1, (
        "an Einops always has exactly one output segment")

    # A reindexing need not be a bare Rearrangement. It can be a Composed or a
    # ProductOfMorphisms of them, and reading `.mapping` off one of those
    # raises. Require every reindexing to be mappable and read it through
    # tutils.get_mapping, which collapses the whole reindexing morphism into
    # one collective Prod[int] whatever its structure.
    assert (tutils.is_mappable_broadcast(producer)
            and tutils.is_mappable_broadcast(consumer)), (
        "merge_einops needs mappable reindexings on both operands. "
        "term_utilities.is_mappable and get_mapping do the reading."
    )

    producer_out_shape = producer.output_weaves[0]._shape
    assert all(ax is cat.WeaveMode.TILED for ax in producer_out_shape), (
        "merge_einops only supports a producer whose every output position is TILED"
    )
    # The merged degree is the consumer's original degree, because the merge
    # leaves the consumer's output, and so its broadcast shape, unchanged.
    merged_degree = tuple(consumer.degree())

    feed_shape = consumer.input_weaves[feed_index]._shape
    feed_mapping = tutils.get_mapping(consumer.reindexings[feed_index])
    assert len(feed_shape) == len(producer_out_shape), (
        "the producer output and the feed slot align position for position"
    )

    # roles[k] states what the consumer does with the producer's k-th degree
    # axis, read off the feed slot, which aligns with the producer's output
    # position for position.
    #   ('degree', d)        -> it stays a broadcast axis, at merged-degree
    #                           index d
    #   ('absorb', ax, key)  -> the consumer contracts it into the group `key`,
    #                           where ax is the RawAxis the consumer names
    #                           there. A group is a position in a signature
    #                           and never an axis, so two groups over one
    #                           axis stay two groups.
    roles: list[tuple] = []
    tiled_i = 0
    feed_groups = iter(consumer_op.signature[feed_index])
    for pos in feed_shape:
        if pos is cat.WeaveMode.TILED:
            roles.append(('degree', feed_mapping[tiled_i]))
            tiled_i += 1
        else:
            roles.append(('absorb', pos, ('consumer', next(feed_groups))))

    # Rewrite each of the producer's input segments through those roles. A
    # TILED position that the producer's reindexing points at a degree axis
    # the consumer keeps as a broadcast stays TILED. One pointing at an axis
    # the consumer absorbs becomes that RawAxis, in the consumer's group. The
    # producer's own contracted positions are never TILED and stay absorbed
    # in the producer's group. The consumer's kept segments then carry over
    # unchanged, in the consumer's groups.
    datatypes: list[cat.Datatype] = []
    shapes: list[fd.Prod] = []
    mappings: list[fd.Prod[int]] = []
    group_keys: list[list[tuple]] = []
    for weave, reindex, segment in zip(
            producer.input_weaves, producer.reindexings, producer_op.signature):
        # TILED positions -> indices into the producer's degree.
        seg_mapping = tutils.get_mapping(reindex)
        own_groups = iter(segment)
        new_shape: list = []
        new_mapping: list[int] = []
        keys: list[tuple] = []
        tiled_j = 0
        for pos in weave._shape:
            if pos is not cat.WeaveMode.TILED:
                new_shape.append(pos)  # the producer's own contracted axis
                keys.append(('producer', next(own_groups)))
                continue
            role = roles[seg_mapping[tiled_j]]
            tiled_j += 1
            if role[0] == 'degree':
                new_shape.append(cat.WeaveMode.TILED)
                new_mapping.append(role[1])
            else:
                new_shape.append(role[1])
                keys.append(role[2])
        datatypes.append(weave.datatype)
        shapes.append(tuple(new_shape))
        mappings.append(tuple(new_mapping))
        group_keys.append(keys)
    for i, (weave, reindex, segment) in enumerate(
            zip(consumer.input_weaves, consumer.reindexings, consumer_op.signature)):
        if i == feed_index:
            continue
        datatypes.append(weave.datatype)
        shapes.append(weave._shape)
        mappings.append(tutils.get_mapping(reindex))
        group_keys.append([('consumer', group) for group in segment])

    # Renumber the groups from scratch, in first-seen order, rather than
    # reconciling the producer's and the consumer's original group numbers,
    # which can collide.
    group_of: dict[tuple, int] = {}
    for keys in group_keys:
        for key in keys:
            if key not in group_of:
                group_of[key] = len(group_of)
    new_signature = tuple(
        tuple(group_of[key] for key in keys) for keys in group_keys)
    new_input_weaves = tuple(
        cat.Weave(datatype=datatype, _shape=shape)
        for datatype, shape in zip(datatypes, shapes)
    )
    new_reindexings = tuple(
        cat.Rearrangement(mapping=mapping, _dom=merged_degree) for mapping in mappings
    )

    new_operator = ops.Einops(name=fd.DynamicName('einops'), signature=new_signature)
    return cat.Broadcasted(
        operator=new_operator,
        input_weaves=new_input_weaves,
        output_weaves=consumer.output_weaves,
        reindexings=new_reindexings,
    )


def disentangle_einops(
    merged: cat.Broadcasted,
) -> tuple[cat.ProdCategory, fd.Prod[int]]:
    '''
    Split the input segments of `merged` into connected components.

    Two segments are connected when they share a contraction group. Each
    component is contracted on its own, and the independent results are
    combined by a final outer-product Einops. One component, meaning that
    everything shares some group directly or transitively, leaves nothing to
    split, and `merged` is returned unchanged.

    The rule is structural and unconditional rather than a cost search.

    It also returns the result's dom in terms of the original input segment
    indices of `merged`. A value of `(2, 0, 1)` states that the result's first
    dom position is segment 2 of `merged`, and so on. The segments are
    reordered into components, so a caller that reconnects the result to the
    hypergraph nodes the inputs of `merged` came from needs the mapping.
    '''
    operator = merged.operator
    assert isinstance(operator, ops.Einops)
    n = len(operator.signature)
    merged_degree = tuple(merged.degree())

    # Union-Find over segment indices, unioning segments that share a group.
    parent = list(range(n))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    group_segments: dict[int, list[int]] = {}
    for i, seg_sig in enumerate(operator.signature):
        for g in seg_sig:
            group_segments.setdefault(g, []).append(i)
    for seg_indices in group_segments.values():
        for i in seg_indices[1:]:
            union(seg_indices[0], i)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)
    component_list = list(components.values())

    if len(component_list) <= 1:
        return merged, tuple(range(n))

    # Build each component's own sub-term, tracking which merged-degree
    # positions it actually uses (needed to wire the final join's
    # reindexings back against the *overall* degree).
    component_terms: list[cat.ProdCategory] = []
    component_degree_positions: list[fd.Prod[int]] = []
    for comp in component_list:
        degree_positions = tuple(sorted(set(
            j for i in comp for j in merged.reindexings[i].mapping
        )))
        component_degree_positions.append(degree_positions)
        comp_degree = tuple(merged_degree[j] for j in degree_positions)

        if (len(comp) == 1 and operator.signature[comp[0]] == ()
                and merged.reindexings[comp[0]].mapping == degree_positions):
            # A lone segment with nothing absorbed and no axis reordering to
            # perform is already this component's result, so it passes straight
            # through and no new node is created.
            #
            # imprint_to_degree fills its TILED slots back with the component's
            # degree axes, so a broadcast axis it carries survives, giving
            # `s[q]` rather than a bare scalar. `weave.target()` would drop
            # every TILED axis.
            #
            # The guard that the mapping equals degree_positions restricts this
            # to the case where the segment's own axis order already matches
            # the join's sorted order. Anything needing a reorder falls through
            # to the Einops branch.
            weave = merged.input_weaves[comp[0]]
            component_terms.append(
                cat.ProdObject((weave.imprint_to_degree(comp_degree),)).identity())
            continue

        group_of: dict[int, int] = {}
        for i in comp:
            for g in operator.signature[i]:
                group_of.setdefault(g, len(group_of))
        comp_signature = tuple(
            tuple(group_of[g] for g in operator.signature[i]) for i in comp)

        remap = {old: new for new, old in enumerate(degree_positions)}
        comp_reindexings = tuple(
            cat.Rearrangement(
                mapping=tuple(remap[j] for j in merged.reindexings[i].mapping),
                _dom=comp_degree,
            )
            for i in comp
        )
        comp_output_weave = cat.Weave(
            datatype=merged.output_weaves[0].datatype,
            _shape=tuple(cat.WeaveMode.TILED for _ in comp_degree),
        )
        comp_operator = ops.Einops(
            name=fd.DynamicName('einops'), signature=comp_signature)
        component_terms.append(cat.Broadcasted(
            operator=comp_operator,
            input_weaves=tuple(merged.input_weaves[i] for i in comp),
            output_weaves=(comp_output_weave,),
            reindexings=comp_reindexings,
        ))

    # Join: one segment per component, none sharing a group with any other
    # (a pure outer product) - each component already fully absorbed
    # whatever was local to it. Every join segment shares the full merged
    # degree as its reindexing _dom (all reindexings of a Broadcasted must -
    # per BroadcastedCategory.degree), and its mapping selects which of those
    # degree slots the component spans. A component covering fewer
    # axes (e.g. `s[q]` when the output degree is `q v`) simply broadcasts up
    # into the rest.
    join_datatype = merged.output_weaves[0].datatype
    join_input_weaves = tuple(
        cat.Weave(
            datatype=join_datatype,
            _shape=tuple(cat.WeaveMode.TILED for _ in positions))
        for positions in component_degree_positions
    )
    join_reindexings = tuple(
        cat.Rearrangement(mapping=positions, _dom=merged_degree)
        for positions in component_degree_positions
    )
    join_operator = ops.Einops(
        name=fd.DynamicName('einops'),
        signature=tuple(() for _ in component_list))
    join = cat.Broadcasted(
        operator=join_operator,
        input_weaves=join_input_weaves,
        output_weaves=merged.output_weaves,
        reindexings=join_reindexings,
    )

    dom_order = tuple(i for comp in component_list for i in comp)
    return chsh.make_composed(chsh.make_product(*component_terms), join), dom_order


def merge_rule(
    producer: cat.Morphism,
    consumer: cat.Morphism,
    port: int,
) -> tuple[cat.ProdCategory, fd.Prod[int]] | None:
    '''
    `merge_einops` then `disentangle_einops`, as an `algebra.merge_into_consumer`
    rule: the same two rules as `rearrange_einops`, offered one pair at a time
    to a search that someone else is driving.

    Both passes exist because they answer to different callers. `rearrange_einops`
    is what fusion runs: one sweep, and it rebuilds the graph flat, which is
    what an unkernelized expression on its way into `fuse` needs.
    `merge_into_consumer.merge_producers_into_consumers(morphism, merge_rule)` runs
    to a fixed point and leaves blocks standing, which is what a `para` pass needs,
    because a reverse pass keeps the softmax's own block, and flattening it would
    lose the one marker saying where that reverse came from.

    The refusals are the assertions of `merge_einops`, put as questions: a rule
    returns `None` where a pass may assert, because it is offered every
    adjacent pair in the graph and most of them are not two `Einops`.
    '''
    if not (isinstance(producer, cat.Broadcasted)
            and isinstance(consumer, cat.Broadcasted)):
        return None
    if not (isinstance(producer.operator, ops.Einops)
            and isinstance(consumer.operator, ops.Einops)):
        return None
    if len(producer.output_weaves) != 1 or port >= len(consumer.input_weaves):
        return None
    # A producer whose output weave has a target position does not plug into
    # the consumer's feed slot position-for-position, which is the one thing
    # `merge_einops` needs. The vault note Einops Rearrangement covers it.
    if any(entry is not cat.WeaveMode.TILED
           for entry in producer.output_weaves[0]._shape):
        return None
    if (len(producer.output_weaves[0]._shape)
            != len(consumer.input_weaves[port]._shape)):
        return None
    if not (tutils.is_mappable_broadcast(producer)
            and tutils.is_mappable_broadcast(consumer)):
        return None
    merged = merge_einops(producer, consumer, port)
    disentangled, dom_order = disentangle_einops(merged)
    # `disentangle_einops` numbers the result's operands as "the producer's,
    # then the consumer's remaining ones"; `local_rewrite` numbers them in
    # place, the producer's sitting where the consumer's feed slot was.
    return disentangled, merge_into_consumer.canonical_order(
        len(tuple(producer.dom())), len(tuple(consumer.dom())), port, dom_order)


def rearrange_local[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    '''`merge_rule` to a fixed point, on a morphism, with blocks left standing.'''
    return merge_into_consumer.merge_producers_into_consumers(target, merge_rule)


def merge_reindexings_and_einops[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    '''`reindexing_absorption.absorb` and `merge_rule` to a fixed point, on a morphism,
    with blocks left standing.

    A node is absorbed into every reader, copied over a fan-out, and a contraction
    merges only into a reader that is its sole reader. The two searches alternate
    until neither changes the graph, so a contraction that an absorption exposes
    merges in the same call, and a node that a merge exposes is absorbed in it.
    '''
    graph = hg.Multigraph.from_morphism(target)
    for _ in range(merge_into_consumer.MERGE_LIMIT):
        absorbed = merge_into_consumer.merge_producers_into_every_consumer_in_graph(
            graph, reindexing_absorption.absorb)
        merged = merge_into_consumer.merge_producers_into_consumers_in_graph(
            absorbed, merge_rule)
        if merged is graph:
            return h2m.hypergraph_to_morphism(graph)
        graph = merged
    raise RuntimeError(
        f'{merge_into_consumer.MERGE_LIMIT} rounds of absorbing and merging without '
        'reaching a fixed point')


def _is_einops_leaf(leaf: hg.Hypergraph) -> bool:
    wraps = getattr(leaf, 'wraps', None)
    return isinstance(wraps, cat.Broadcasted) and isinstance(wraps.operator, ops.Einops)


def rearrange_einops[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
) -> hg.Hypergraph[L, M]:
    '''
    One pass over a raw hypergraph: wherever one Einops's sole output feeds
    another Einops's input, `merge_einops` folds the two together and
    `disentangle_einops` splits the result, and the replacement is spliced back
    in by hypergraph node identity.

    The pass is not iterative. It runs once over the leaves present when it
    starts, so a leaf produced by disentangling is not itself re-scanned for
    further merging in the same pass. `merge_reindexings_and_einops` is the
    caller that runs the rules to a fixed point.
    '''
    leaves = list(hg.flat_subgraphs(graph, remove_blocks=True))
    consumers: util.Multidict[hg.HypergraphObject, hg.Hypergraph] = util.Multidict(
        (dom, leaf) for leaf in leaves for dom in leaf.dom
    )

    replaced: set = set()
    remaining = list(leaves)
    equalities: list[fd.EqualityClass] = []

    for producer in leaves:
        if (producer.uid in replaced
                or not _is_einops_leaf(producer)
                or len(producer.cod) != 1):
            continue
        wire = producer.cod[0]
        readers = [leaf for leaf in consumers[wire] if leaf.uid not in replaced]
        if len(readers) != 1:
            continue
        consumer = readers[0]
        if producer.uid == consumer.uid or not _is_einops_leaf(consumer):
            continue
        feed_index = next(
            i for i, dom in enumerate(consumer.dom) if dom == wire)

        merged = merge_einops(producer.wraps, consumer.wraps, feed_index)
        disentangled, dom_order = disentangle_einops(merged)

        merged_dom_nodes = tuple(producer.dom) + tuple(
            node for i, node in enumerate(consumer.dom) if i != feed_index
        )
        final_dom_nodes = tuple(merged_dom_nodes[i] for i in dom_order)

        replacement = hg.Multigraph.from_morphism(
            disentangled,
            dom=hg.HypergraphObject.template(disentangled.dom(), final_dom_nodes),
        )
        # Redirect the freshly generated output node of the replacement onto
        # the node everything downstream of the consumer already names, so that
        # nothing outside this rewrite has to change.
        equalities.append(fd.UIDRenaming.set_canonical(
            consumer.cod[0], replacement.cod[0]))

        remaining = [
            leaf for leaf in remaining
            if leaf.uid not in (producer.uid, consumer.uid)]
        remaining.extend(hg.flat_subgraphs(replacement, remove_blocks=True))
        replaced.add(producer.uid)
        replaced.add(consumer.uid)

    if not replaced:
        return graph

    rebuilt = hg.Multigraph.template(dom=graph.dom, cod=graph.cod, subgraphs=remaining)
    return fd.Context(equalities).apply(rebuilt)
