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
    assert len(reindex.out_axes) == 5         # x, dx, y, dy, ch
