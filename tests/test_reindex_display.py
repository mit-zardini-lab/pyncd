from data_structure.TensorDSL import TL, real_axis, Reindex, relu

def test_reindex_has_in_axes():
    H, W, C = 6, 3, 2
    tl = TL()
    xi = real_axis('xi', H + W - 1)
    yi = real_axis('yi', H + W - 1)
    ch = real_axis('ch', C)
    x  = real_axis('x', H)
    y  = real_axis('y', H)
    dx = real_axis('dx', W)
    dy = real_axis('dy', W)
    tl.Filter.tensor(dx, dy, ch)
    tl.Image.tensor(xi, yi, ch)
    tl.Features[x, y] = relu(tl.Filter[dx, dy, ch] * tl.Image[x+dx, y+dy, ch])
    morph = tl.to_morphism()
    from data_structure.ProductCategory import ThreadedComposed
    assert isinstance(morph, ThreadedComposed)
    reindex = morph.content[0]
    assert isinstance(reindex, Reindex)
    assert len(reindex.in_axes) == 3          # xi, yi, ch
    assert {ax.uid for ax in reindex.in_axes} == {xi.uid, yi.uid, ch.uid}
    assert len(reindex.out_axes) == 5         # x, dx, y, dy, ch


def test_reindex_display_does_not_crash():
    import display as dpl
    H, W, C, P, S = 6, 3, 2, 2, 2
    H_pad = H + W - 1
    H_out = (H - P) // S + 1
    tl2 = TL()
    xi2, yi2 = real_axis('xi', H_pad), real_axis('yi', H_pad)
    x2,  y2  = real_axis('x',  H),     real_axis('y',  H)
    dx2, dy2 = real_axis('dx', W),     real_axis('dy', W)
    ch2      = real_axis('ch', C)
    px2, py2 = real_axis('px', P),     real_axis('py', P)
    xo2, yo2 = real_axis('xo', H_out), real_axis('yo', H_out)
    tl2.Filter.tensor(dx2, dy2, ch2)
    tl2.Image.tensor(xi2, yi2, ch2)
    tl2.Features[x2, y2]  = relu(tl2.Filter[dx2, dy2, ch2] * tl2.Image[x2+dx2, y2+dy2, ch2])
    tl2.Pooled[xo2, yo2]  = tl2.Features[S*xo2+px2, S*yo2+py2]
    morph = tl2.to_morphism()
    # Must not raise
    dpl.print_category(morph)  # type: ignore
