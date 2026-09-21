'''Check the sparse expansion against the DeepSeek-V3 complete form.

    python deepseek/validate_sparse.py          # assert and print listings

The fixtures are built here, so the check needs no notebook. Three of them are:

  * the compressed mixture of experts of `example_notebooks/DeepSeekV3.ipynb`
    ("The mixture of experts, with the routing in the signatures") - `expand_sparse`
    must turn it into the notebook's hand-built complete form ("The mixture of
    experts, written out"):
    a two-output TopK whose second wire is `Natural(n)` indices, that index
    entering every expert Linear as a rank-0 target operand broadcast over
    k, the diagonal spliced away, and `n` dense only at the router;

  * a Select selection, which is the shape a gather of a cache at selected
    positions takes - `expand_sparse`
    must turn the Select into hold x IndexSelect ; einops('k, k -> k'), with the
    payload's parent axis kept dense;

  * a selection over the reachable slots of the relative indexer of
    `notebooks/sota/DeepSeekV41Flash/lightning_indexer.py` - the Top-`s` over a
    guarded axis hands
    its picks out on the slots that axis states, and both expansions leave a model
    holding those axes unchanged. The guards themselves are checked in
    `advanced_axis_dynamics/validate_advanced_axis_dynamics.py`, whose fixture this
    check reads;

  * a Select at positions (the gather of
    `notebooks/sota/DeepSeekV41Flash/lightning_indexer.py`) - a `TopK` in the
    `ONLY_SELECTION`
    form hands out positions, a `Select` reads the cache at them, and both
    expansions return the gather unchanged, because it holds no sparse axis.

Structural assertions rather than a listing diff: UIDs are random per
process, so the text is not stable across runs, but the shapes are.
'''
from __future__ import annotations

# Run as `python deepseek/validate_sparse.py` (from the repo root). Python
# prepends the script's own directory, where `data_structure.py` (the
# deepseek one) would shadow the `data_structure` package - drop it.
import os, sys
sys.path = [p for p in sys.path
            if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import construction_helpers as ch  # noqa: F401 - @ auto-alignment
import construction_helpers.lift as chl
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as ds
import deepseek.sparse_expansion as dse
import graphs.processing.Hypergraph2Morphism as h2m
import para.algebra.para_sparse_expansion as para_sparse_expansion
import para.data_structure.Para as Para
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import advanced_axis_dynamics.data_structure.Operators as aops
from advanced_axis_dynamics import (
    validate_advanced_axis_dynamics as validate_advanced_axis_dynamics)
import term_utilities.generate_config as gc
import term_utilities.term_utilities as tutil


def compressed_moe():
    '''The V3 notebook's compressed MoE, verbatim (hover blocks aside).'''
    def block(target, title):
        return cat.Block(
            h2m.recycle(target),
            block_tag=cat.BlockTag(
                aesthetics=cat.BlockAesthetics(title=title, fill_color='white')))

    gate = block(
        ops.Linear.template('m', 1, 'L^g')
        @ ds.TopK.template() @ ops.SoftMax.template(),
        'Gate')
    multihead_perceptron = block(
        (0, 0) @
        ((ops.Linear.template(1, 2, 'W_0') @ ops.Elementwise.template())
         * ops.Linear.template(1, 2, 'W_1')) @
        ops.Einops.template('n f, n f -> n f') @
        ops.Linear.template(1, 2, 'W_2') @
        ops.Einops.template('k k m -> k m'),
        'MoE')
    return h2m.recycle(
        (0, 0) @ (gate * multihead_perceptron)
        @ ops.Einops.template('n, n m -> m'))


def compressed_gather():
    '''The CSA selection: score, TopK on a sparse axis, Select the cache.'''
    x = cat.RawAxis.named('x')
    b = cat.RawAxis.named('b')
    c = cat.RawAxis.named('c')
    topk = ds.TopK.template(axis=b, name='s/b')
    kb = topk.cod()[0].shape()[0]
    T = cat.WeaveMode.TILED
    select = cat.Broadcasted(
        operator=ds.Select(),
        input_weaves=(cat.Weave(cat.Reals(), (T, kb)),
                      cat.Weave(cat.Reals(), (b, T))),
        output_weaves=(cat.Weave(cat.Reals(), (T, kb, T)),),
        reindexings=(cat.Rearrangement((0,), (x, c)),
                     cat.Rearrangement((1,), (x, c))))
    lifted_topk = chl.morphism_object_lift(topk, cat.ProdObject((x,)))
    scores = cat.Array(cat.Reals(), (x, b))
    cache = cat.Array(cat.Reals(), (b, c))
    return h2m.recycle(
        cat.Rearrangement((0, 1), (scores, cache))
        @ (lifted_topk * cat.ProdObject((cache,)).identity())
        @ select)


def roots(target):
    return tuple(tutil.type_search(cat.Broadcasted, target))


def the(iterable):
    (item,) = tuple(iterable)
    return item


def check_moe() -> None:
    moe = compressed_moe()
    expanded = dse.expand_sparse(moe)

    assert not tuple(tutil.type_search(ds.SparseAxis, expanded)), \
        'sparse axes survived the MoE expansion'
    assert expanded.dom() == moe.dom() and expanded.cod() == moe.cod(), \
        'expansion changed the MoE boundary'

    topk = the(r for r in roots(expanded) if isinstance(r.operator, ds.TopK))
    assert len(topk.output_weaves) == 2, 'TopK must emit values AND indices'
    values, indices = topk.output_weaves
    assert isinstance(indices.datatype, cat.Natural), \
        'the second TopK output is the Natural(n) index wire'
    assert indices._shape == values._shape, \
        'indices ride the same k axis as the values'
    (k_axis,) = values.target().shape()

    experts = tuple(
        r for r in roots(expanded)
        if isinstance(r.operator, ops.Linear) and len(r.input_weaves) == 2)
    assert len(experts) == 3, \
        f'W_0, W_1, W_2 must each take the index; got {len(experts)}'
    for expert in experts:
        index_weave = expert.input_weaves[0]
        assert isinstance(index_weave.datatype, cat.Natural), \
            'the index operand comes FIRST, as in the V3 complete form'
        assert index_weave.target().shape() == cat.ProdObject(), \
            'the index enters as a rank-0 target, broadcast over k'
        assert k_axis in tuple(expert.degree()), \
            'the expert Linear is broadcast over k'
        assert k_axis not in tuple(the(expert.output_weaves).target().shape()), \
            'k left the weight TARGET for the degree'

    assert not any(isinstance(r.operator, ops.View) for r in roots(expanded)), \
        'the diagonal must collapse and splice out'

    # The index wire crosses both block boundaries: out of the Gate, into
    # the perceptron.
    for block in tutil.type_search(cat.Block, expanded):
        boundary = (block.dom(), block.cod())
        title = (block.block_tag.aesthetics.title
                 if block.block_tag.aesthetics else None)
        carried = any(
            isinstance(array.datatype, cat.Natural)
            for side in boundary for array in side)
        assert carried, f'the index wire must cross the {title} block'


def check_gather() -> None:
    gather = compressed_gather()
    expanded = dse.expand_sparse(gather)

    assert not tuple(tutil.type_search(ds.SparseAxis, expanded)), \
        'sparse axes survived the gather expansion'
    assert not any(isinstance(r.operator, ds.Select) for r in roots(expanded)), \
        'the Select must expand away'

    grab = the(r for r in roots(expanded) if isinstance(r.operator, ds.IndexSelect))
    index_weave, payload_weave = grab.input_weaves
    assert isinstance(index_weave.datatype, cat.Natural)
    assert index_weave.target().shape() == cat.ProdObject(), \
        'the index is a rank-0 target, like an expanded Linear operand'
    b_axis = the(payload_weave.target().shape())
    assert not isinstance(b_axis, ds.SparseAxis), \
        'the payload keeps its dense parent axis'
    assert all(isinstance(entry, cat.WeaveMode)
               for entry in the(grab.output_weaves)._shape), \
        'everything surviving is broadcast: the output weave is all TILED'
    index_reindexing = grab.reindexings[0]
    assert index_reindexing.mapping == tuple(range(len(index_reindexing.mapping))), \
        'the index reindexing is the identity prefix into the degree'

    product = the(
        r for r in roots(expanded)
        if isinstance(r.operator, ops.Einops) and r.operator.signature == ((), ()))
    assert product.input_weaves[0].target().shape() == cat.ProdObject(), \
        "einops('k, k -> k'): elementwise, everything broadcast"

    topk = the(r for r in roots(expanded) if isinstance(r.operator, ds.TopK))
    assert len(topk.output_weaves) == 2


def check_config_names() -> None:
    '''The k axis takes the activity symbol, so a NumericConfig still
    reaches every size by the names the compressed form used.'''
    expanded = dse.expand_sparse(compressed_moe())
    names = {t.uid._name.to_bodies()
             for t in gc.NumericConfig.template(expanded).terms
             if t.uid._name}
    assert {'k', 'n', 'm', 'f'} <= names, names


def check_forms() -> None:
    '''Each form hands out the outputs `SelectionForm` states, the taped
    expansion turns `WEIGHTS` into `WEIGHTS_SELECT`, and the dense forms pass
    through both expansions as the same object.'''
    x = cat.RawAxis.named('x')
    P = cat.RawAxis.named('P')
    p = nm.FreeNumeric.named('p')
    positions_of = {
        ds.SelectionForm.WEIGHTS: (False,),
        ds.SelectionForm.WEIGHTS_SELECT: (False, True),
        ds.SelectionForm.ONLY_WEIGHTS: (False,),
        ds.SelectionForm.ONLY_SELECTION: (True,),
    }
    for form, positions in positions_of.items():
        topk = ds.TopK.template(k=p, axis=P, name='p/P', form=form)
        assert topk.operator.form is form
        assert topk.operator.k == p, form
        ds.check_form_matches_outputs(topk)
        assert tuple(isinstance(array.datatype, cat.Natural)
                     for array in topk.cod()) == positions, form
        for array in topk.cod():
            axis = the(array.shape())
            if form is ds.SelectionForm.WEIGHTS:
                assert isinstance(axis, ds.SparseAxis) and axis.activity == p
            else:
                assert not isinstance(axis, ds.SparseAxis)
                assert axis.local_size() == p
                assert axis.uid._name.to_bodies() == 'p', form
        lifted = chl.morphism_object_lift(topk, cat.ProdObject((x,)))
        taped = para_sparse_expansion.expand_sparse_onto_tape(lifted)
        if form is ds.SelectionForm.WEIGHTS:
            expanded = the(r for r in roots(taped) if isinstance(r.operator, ds.TopK))
            assert expanded.operator.form is ds.SelectionForm.WEIGHTS_SELECT
            ds.check_form_matches_outputs(expanded)
        else:
            assert taped is lifted, f'the taped pass rewrote a {form.name} TopK'
            assert dse.expand_sparse(lifted) is lifted, \
                f'the wired pass rewrote a {form.name} TopK'


def merged_selection():
    '''A selection over blocks merged onto the positions, and a selection inside
    it, which is the shape of the candidate pool of
    `notebooks/sota/DeepSeekV41Flash/candidate_pool.py` and of a Reindex layer of
    `notebooks/sota/DeepSeekV41Flash/attention_modes.py`. Scores on `B` are split
    into blocks
    `(P, u)`, each block's best score is kept on `p/P`, every offset of a kept
    block is active on `c/B`, the scores are read there, a Top-`s` over `c/B`
    hands out `s/B`, and entries on `B` are gathered at it.'''
    T = cat.WeaveMode.TILED
    x = cat.RawAxis.named('x')
    B = cat.RawAxis.named('B')
    P = cat.RawAxis.named('P')
    u = cat.RawAxis.named('u')
    c = cat.RawAxis.named('c')
    p = nm.FreeNumeric.named('p')
    s = nm.FreeNumeric.named('s')
    split = cat.StrideMorphism(
        _dom=(P, u),
        _cod_stride_shift=((B, (u.local_size(), nm.Integer(1)), nm.Integer(0)),),
        name=fd.DynamicName('blk'))

    def over(axes, morphism):
        return chl.morphism_object_lift(morphism, cat.ProdObject(tuple(axes)))

    def hold(array):
        return cat.ProdObject((array,)).identity()

    block_view = ops.View.template(reindexing=(split,), name='blk')
    block_max = over((P,), cat.Broadcasted(
        operator=ops.Maximum(),
        input_weaves=(cat.Weave(cat.Reals(), (u,)),),
        output_weaves=(cat.Weave(cat.Reals(), ()),),
        reindexings=(cat.ProdObject().identity(),)))
    keep = ds.TopK.template(k=p, axis=P, name='p/P')
    pP = the(keep.cod()[0].shape())
    ones = ops.Arithmetic.template(nm.Integer(1))
    repeat = ops.View.template(reindexing=cat.Rearrangement((0,), (pP, u)), name='rep')
    merge = ds.merge_selected_axis(split, (pP, u), name='c/B')
    cB = the(merge.cod()[0].shape())
    pool = over((x,), block_view @ block_max @ keep @ ones @ repeat @ merge)
    scores = cat.Array(cat.Reals(), (x, B))
    restrict = cat.Broadcasted(
        operator=ds.Select(),
        input_weaves=(cat.Weave(cat.Reals(), (T, cB)), cat.Weave(cat.Reals(), (T, B))),
        output_weaves=(cat.Weave(cat.Reals(), (T, cB)),),
        reindexings=(cat.Rearrangement((0,), (x,)), cat.Rearrangement((0,), (x,))))
    select = over((x,), ds.TopK.template(k=s, axis=cB, name='s/B'))
    sB = tuple(select.cod()[0].shape())[1]
    entries = cat.Array(cat.Reals(), (B, c))
    gather = cat.Broadcasted(
        operator=ds.Select(),
        input_weaves=(cat.Weave(cat.Reals(), (T, sB)), cat.Weave(cat.Reals(), (B, T))),
        output_weaves=(cat.Weave(cat.Reals(), (T, sB, T)),),
        reindexings=(cat.Rearrangement((0,), (x, c)), cat.Rearrangement((1,), (x, c))))
    return h2m.recycle(
        cat.Rearrangement((0, 0, 1), (scores, entries))
        @ (pool * hold(scores) * hold(entries))
        @ (restrict * hold(entries))
        @ (select * hold(entries))
        @ gather)


def check_merged_selection() -> None:
    '''Both passes expand a merged selection and a selection over it: the merge
    becomes a reshape of the values beside the positions of the merged axis, the
    selection over the merged axis composes its positions with them, and every
    sparse axis is gone.'''
    model = merged_selection()
    assert len(tuple(tutil.type_search(ds.SparseAxis, model))) >= 3
    wired = dse.expand_sparse(model)
    taped = para_sparse_expansion.expand_sparse_onto_tape(model)
    for expanded in (wired, taped):
        assert not tuple(tutil.type_search(ds.SparseAxis, expanded)), \
            'sparse axes survived the merged selection'
        assert expanded.dom() == model.dom()
        topks = tuple(r for r in roots(expanded) if isinstance(r.operator, ds.TopK))
        assert len(topks) == 2 and all(
            r.operator.form is ds.SelectionForm.WEIGHTS_SELECT for r in topks)
        positions = the(
            r for r in roots(expanded) if isinstance(r.operator, ds.MergedPositions))
        assert isinstance(the(positions.output_weaves).datatype, cat.Natural)
        views = tuple(
            r for r in roots(expanded) if isinstance(r.operator, aops.CovariantView))
        assert len(views) == 2, 'the values and the positions are each reshaped'
        assert {type(the(v.output_weaves).datatype) for v in views} == {
            cat.Reals, cat.Natural}
        composers = tuple(
            r for r in roots(expanded)
            if isinstance(r.operator, ds.IndexSelect)
            and isinstance(r.input_weaves[1].datatype, cat.Natural))
        assert len(composers) == 1, \
            'the selection over the merged axis composes its positions once'
    dropped = {drop.tape for drop in tutil.type_search(Para.Drop, taped)}
    grabbed = {grab.tape for grab in tutil.type_search(Para.Grab, taped)}
    assert dropped == grabbed and len(dropped) == 3, (dropped, grabbed)


def positions_pool():
    '''The candidate pool of `notebooks/sota/DeepSeekV41Flash/candidate_pool.py` as
    that module has built it since 2026-09-11, with the selection over the blocks in
    the
    `ONLY_SELECTION` form. The Top-`p` over the blocks hands out block numbers
    alone, the merge applies the split to them at every offset and lays the
    result out along a dense candidate axis `C`, a Reindex layer reads its scores
    at those positions with an `IndexSelect`, its Top-`s` over `C` takes the
    positions as an operand and hands out `s/B`, and entries on `B` are gathered
    at it.'''
    T = cat.WeaveMode.TILED
    R = cat.Reals()
    x = cat.RawAxis.named('x')
    B = cat.RawAxis.named('B')
    P = cat.RawAxis.named('P')
    u = cat.RawAxis.named('u')
    c = cat.RawAxis.named('c')
    p = nm.FreeNumeric.named('p')
    s = nm.FreeNumeric.named('s')
    C = cat.RawAxis.named('C')
    split = cat.StrideMorphism(
        _dom=(P, u),
        _cod_stride_shift=((B, (u.local_size(), nm.Integer(1)), nm.Integer(0)),),
        name=fd.DynamicName('blk'))

    def over(axes, morphism):
        return chl.morphism_object_lift(morphism, cat.ProdObject(tuple(axes)))

    def hold(array):
        return cat.ProdObject((array,)).identity()

    block_view = ops.View.template(reindexing=(split,), name='blk')
    block_max = over((P,), cat.Broadcasted(
        operator=ops.Maximum(),
        input_weaves=(cat.Weave(R, (u,)),),
        output_weaves=(cat.Weave(R, ()),),
        reindexings=(cat.ProdObject().identity(),)))
    keep = ds.TopK.template(k=p, axis=P, name='p/P',
                            form=ds.SelectionForm.ONLY_SELECTION)
    p_axis = the(keep.cod()[0].shape())
    merge = ds.merge_selected_positions(
        split, p_axis, cat.ProdObject((x,)), C, name='cand')
    pool = over((x,), block_view @ block_max @ keep) @ merge
    positions = cat.Array(cat.Natural(B.local_size()), (x, C))
    scores = cat.Array(R, (x, B))
    restrict = cat.Broadcasted(
        operator=ds.IndexSelect(),
        input_weaves=(cat.Weave(cat.Natural(B.local_size()), (T, T)),
                      cat.Weave(R, (T, B))),
        output_weaves=(cat.Weave(R, (T, T)),),
        reindexings=(cat.ProdObject((x, C)).identity(),
                     cat.Rearrangement((0,), (x, C))))
    select = over((x,), ds.TopK.template(k=s, axis=C, name='s/B', positions_of=B))
    sB = tuple(select.cod()[0].shape())[1]
    entries = cat.Array(R, (B, c))
    gather = cat.Broadcasted(
        operator=ds.Select(),
        input_weaves=(cat.Weave(R, (T, sB)), cat.Weave(R, (B, T))),
        output_weaves=(cat.Weave(R, (T, sB, T)),),
        reindexings=(cat.Rearrangement((0,), (x, c)), cat.Rearrangement((1,), (x, c))))
    return h2m.recycle(
        cat.Rearrangement((0, 0, 1), (scores, entries))
        @ (pool * hold(scores) * hold(entries))
        @ cat.Rearrangement((0, 1, 0, 2), (positions, scores, entries))
        @ (restrict * hold(positions) * hold(entries))
        @ (select * hold(entries))
        @ gather)


def check_positions_pool() -> None:
    '''A selection in the `ONLY_SELECTION` form merged into a positions table
    holds no sparse axis, so the pool passes through both expansions as it is.
    The selection over the candidates that takes the table as an operand hands
    out the one sparse axis, and both passes expand it into a selection over the
    candidates composed with the table.'''
    model = positions_pool()
    assert len({axis.uid for axis in tutil.type_search(ds.SparseAxis, model)}) == 1, \
        'the pool carries positions and the Top-s over them hands out the one sparse axis'
    keep = the(r for r in roots(model)
               if isinstance(r.operator, ds.TopK)
               and r.operator.form is ds.SelectionForm.ONLY_SELECTION)
    assert isinstance(the(keep.output_weaves).datatype, cat.Natural)
    merged = the(r for r in roots(model) if isinstance(r.operator, aops.CovariantView))
    assert isinstance(the(merged.output_weaves).datatype, cat.Natural), \
        'the pool lays positions out, and carries no values'
    select = the(r for r in roots(model) if ds.selects_over_positions(r))
    assert isinstance(the(select.output_weaves).target().shape()[0], ds.SparseAxis)
    wired = dse.expand_sparse(model)
    taped = para_sparse_expansion.expand_sparse_onto_tape(model)
    for expanded in (wired, taped):
        assert not tuple(tutil.type_search(ds.SparseAxis, expanded)), \
            'sparse axes survived the positions pool'
        assert expanded.dom() == model.dom()
        topks = tuple(r for r in roots(expanded) if isinstance(r.operator, ds.TopK))
        forms = sorted(r.operator.form.name for r in topks)
        assert forms == ['ONLY_SELECTION', 'WEIGHTS_SELECT'], forms
        assert all(len(r.input_weaves) == 1 for r in topks), \
            'the positions operand leaves the expanded selection'
        assert the(r for r in roots(expanded)
                   if isinstance(r.operator, ds.MergedPositions)) is not None
        views = tuple(
            r for r in roots(expanded) if isinstance(r.operator, aops.CovariantView))
        assert len(views) == 1, 'the pool holds one layout and no values to reshape'
        composers = tuple(
            r for r in roots(expanded)
            if isinstance(r.operator, ds.IndexSelect)
            and isinstance(r.input_weaves[1].datatype, cat.Natural))
        assert len(composers) == 1, \
            'the selection over the candidates composes its positions once'
        gathers = tuple(
            r for r in roots(expanded)
            if isinstance(r.operator, ds.IndexSelect)
            and not isinstance(r.input_weaves[1].datatype, cat.Natural))
        assert len(gathers) == 2, 'the restriction and the expanded entry gather'
    dropped = {drop.tape for drop in tutil.type_search(Para.Drop, taped)}
    grabbed = {grab.tape for grab in tutil.type_search(Para.Grab, taped)}
    assert dropped == grabbed and len(dropped) == 1, (dropped, grabbed)


def block_positions_merge(
    degree: fd.Prod[cat.Axis],
) -> dict[str, cat.Axis | cat.StrideMorphism | cat.BroadcastedCategory]:
    '''The merge of the candidate pool of DeepSeek-V4.1-Flash,
    built as `notebooks/sota/DeepSeekV41Flash/candidate_pool.py` builds it, on
    axes declared here because `deepseek/` does not import the notebook package.

    The distances `r` split into `P` blocks of `u` offsets, a Top-`p` over the
    blocks hands out block numbers alone, and `merge_selected_positions` puts
    each kept block number back onto the distances at every offset, along the
    candidate axis `C`.
    '''
    reach = cat.RawAxis.named('r')
    P = cat.RawAxis.named('P')
    u = cat.RawAxis.named('u')
    C = cat.RawAxis.named('C')
    split = cat.StrideMorphism(
        _dom=(P, u),
        _cod_stride_shift=(
            (reach, (u.local_size(), nm.Integer(1)), nm.Integer(0)),),
        name=fd.DynamicName('blk'))
    keep = ds.TopK.template(k=nm.FreeNumeric.named('p'), axis=P, name='p/P',
                            form=ds.SelectionForm.ONLY_SELECTION)
    selected = the(keep.cod()[0].shape())
    return dict(
        reach=reach, P=P, u=u, C=C, split=split, selected=selected,
        merge=ds.merge_selected_positions(
            split, selected, cat.ProdObject(degree), C, name='cand'))


def check_merged_positions_weaves() -> None:
    '''The merge holds two arrays rather than one array read twice, and their
    datatypes differ.

    `MergedPositions` takes one block number, a rank-0 `Natural` counting the
    `P` blocks broadcast over the degree and over the slots the selection
    filled. It computes a `Natural` counting the `r` distances, carrying the
    offsets `u` as its target, because the position is a function of the
    offset. The covariant view after it reads a block number and an offset and
    writes the candidate axis `C`, multiplying the block number by `|u|` and
    the offset by 1.
    '''
    for degree in ((), (cat.RawAxis.named('x'),)):
        merge = block_positions_merge(degree)
        reach, P, u, C = (merge[name] for name in ('reach', 'P', 'u', 'C'))
        selected, composite = merge['selected'], merge['merge']
        blocks, distances = cat.Natural(P.local_size()), cat.Natural(
            reach.local_size())
        positions = the(r for r in roots(composite)
                        if isinstance(r.operator, ds.MergedPositions))
        operand = the(positions.input_weaves)
        assert operand.datatype == blocks, operand.datatype
        assert tuple(operand.target().shape()) == (), \
            'the block number arrives as a rank-0 Natural'
        result = the(positions.output_weaves)
        assert result.datatype == distances, result.datatype
        assert tuple(result.target().shape()) == (u,), \
            'the computed position is a function of the offset'
        view = the(r for r in roots(composite)
                   if isinstance(r.operator, aops.CovariantView))
        assert tuple(the(view.input_weaves).target().shape()) == (selected, u)
        assert tuple(the(view.output_weaves).target().shape()) == (C,)
        assert {the(view.input_weaves).datatype,
                the(view.output_weaves).datatype} == {distances}, \
            'the layout moves a position without changing what it counts'
        strides = view.operator.reindexing.strides()[0]
        assert strides == (u.local_size(), nm.Integer(1)), strides
        assert composite.dom() == cat.ProdObject(
            (cat.Array(blocks, (*degree, selected)),)), composite.dom()
        assert composite.cod() == cat.ProdObject(
            (cat.Array(distances, (*degree, C)),)), composite.cod()


def positions_gather():
    '''The selection of `notebooks/sota/DeepSeekV41Flash/lightning_indexer.py` as
    that module has written it since 2026-09-14. A Top-`s` over `b` in the `ONLY_SELECTION` form hands out
    `Nat(b)[x, s]`, and a `Select` reads the cache on `b` at those positions, with
    `s` in its target, broadcast over `x` and `c`.'''
    T = cat.WeaveMode.TILED
    R = cat.Reals()
    x = cat.RawAxis.named('x')
    b = cat.RawAxis.named('b')
    c = cat.RawAxis.named('c')
    topk = ds.TopK.template(k=nm.FreeNumeric.named('s'), axis=b,
                            form=ds.SelectionForm.ONLY_SELECTION)
    s = the(topk.cod()[0].shape())
    select = cat.Broadcasted(
        operator=ds.Select(),
        input_weaves=(cat.Weave(cat.Natural(b.local_size()), (T, s)),
                      cat.Weave(R, (b, T))),
        output_weaves=(cat.Weave(R, (T, s, T)),),
        reindexings=(cat.Rearrangement((0,), (x, c)),
                     cat.Rearrangement((1,), (x, c))))
    scores = cat.Array(R, (x, b))
    cache = cat.Array(R, (b, c))
    return h2m.recycle(
        cat.Rearrangement((0, 1), (scores, cache))
        @ (chl.morphism_object_lift(topk, cat.ProdObject((x,)))
           * cat.ProdObject((cache,)).identity())
        @ select)


def check_positions_gather() -> None:
    '''`Select.at_positions` takes the positions a `TopK` in the `ONLY_SELECTION`
    form hands out and a payload on the scored axis, and hands out the payload
    along the selection axis. A gather written that way holds no sparse axis, so
    both expansions return it as the same object.'''
    n = cat.RawAxis.named('n')
    k = cat.RawAxis.named('k')
    seed = ds.Select.at_positions(k, n)
    assert ds.reads_at_positions(seed)
    assert tuple(seed.dom()) == (cat.Array(cat.Natural(n.local_size()), (k,)),
                                 cat.Array(cat.Reals(), (n,)))
    assert tuple(seed.cod()) == (cat.Array(cat.Reals(), (k,)),)
    assert not ds.reads_at_positions(
        ds.Select.template(ds.SparseAxis.template(n.local_size()), n))
    model = positions_gather()
    assert not tuple(tutil.type_search(ds.SparseAxis, model)), \
        'a selection in the positions form carries no sparse axis'
    select = the(r for r in roots(model) if isinstance(r.operator, ds.Select))
    assert ds.reads_at_positions(select)
    positions, _ = select.input_weaves
    assert positions.target().shape() == the(select.output_weaves).target().shape(), \
        'the selection axis is the target of the positions and of the result'
    assert dse.expand_sparse(model) is model
    assert para_sparse_expansion.expand_sparse_onto_tape(model) is model


def selection_over_guarded_slots():
    '''A `TopK` over the reachable slots of the relative indexer, read by a `Select`.

    The indexer, the reads it is built from and the axes they mark are the fixture of
    `advanced_axis_dynamics/validate_advanced_axis_dynamics.py`, which owns the guards.
    What belongs here is what a selection does with a guarded axis: the Top-`s` over
    `r|x` hands its picks out on the slots `AffineSparseAxis.selected_slots` states, and
    a model built on those axes passes through both sparse expansions unchanged, because
    a guard is decided by the position and carries no index wire.
    '''
    T = cat.WeaveMode.TILED
    r = validate_advanced_axis_dynamics.guarded_reads()
    x, b, c = r['x'], r['b'], cat.RawAxis.named('c')
    back_reach, r_reach = r['back_reach'], r['r_reach']
    topk = ds.TopK.template(k=nm.FreeNumeric.named('s'), axis=r_reach,
                            form=ds.SelectionForm.ONLY_SELECTION)
    s_reach, = topk.cod()[0].shape()
    entries = cat.Array(cat.Reals(), (back_reach, c))
    select = cat.Broadcasted(
        operator=ds.Select(),
        input_weaves=(cat.Weave(cat.Natural(b.local_size()), (T, s_reach)),
                      cat.Weave(cat.Reals(), (back_reach, T))),
        output_weaves=(cat.Weave(cat.Reals(), (T, s_reach, T)),),
        reindexings=(cat.Rearrangement((0,), (x, c)),
                     cat.Rearrangement((1,), (x, c))))
    model = h2m.recycle(
        (r['indexer'] * cat.ProdObject((entries,)).identity())
        @ (chl.morphism_object_lift(topk, cat.ProdObject((x,)))
           * cat.ProdObject((entries,)).identity())
        @ select)
    return dict(r_reach=r_reach, s_reach=s_reach, model=model)


def check_selection_over_guarded_slots() -> None:
    '''A selection over an axis live on a run from its first position hands its slots
    out under the same form, and neither expansion touches a model holding such an
    axis.'''
    selection = selection_over_guarded_slots()
    i_x, i_r = (nm.FreeNumeric.named(name) for name in ('i_x', 'i_r'))
    r_reach, s_reach = selection['r_reach'], selection['s_reach']
    assert isinstance(s_reach, AffineGuards.AffineSparseAxis)
    assert s_reach.guard_form((i_x,), i_r) == r_reach.guard_form((i_x,), i_r), \
        'a selection over a prefix-live axis fills a prefix of its slots under the ' \
        'same form'
    model = selection['model']
    assert not tuple(tutil.type_search(ds.SparseAxis, model))
    assert dse.expand_sparse(model) is model
    assert para_sparse_expansion.expand_sparse_onto_tape(model) is model


if __name__ == '__main__':
    import agent_display as ad
    moe = compressed_moe()
    print('==== compressed MoE ====')
    print(ad.listing(moe))
    print('==== expanded MoE ====')
    print(ad.listing(dse.expand_sparse(moe)))
    gather = compressed_gather()
    print('==== compressed gather ====')
    print(ad.listing(gather))
    print('==== expanded gather ====')
    print(ad.listing(dse.expand_sparse(gather)))
    print('==== gather at positions ====')
    print(ad.listing(positions_gather()))
    check_moe()
    check_gather()
    check_config_names()
    check_forms()
    check_merged_selection()
    check_merged_positions_weaves()
    check_positions_pool()
    check_positions_gather()
    check_selection_over_guarded_slots()
    print('all sparse expansion checks passed')
