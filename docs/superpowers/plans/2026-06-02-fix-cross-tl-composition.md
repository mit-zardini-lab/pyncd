# Fix Cross-TL-Instance Composition (`attn_res() @ ffn_res()`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `@` composition of morphisms built from separate `TL()` instances so that `attn_res() @ ffn_res()` and `transformer_stack(L)` work correctly.

**Architecture:** The bug is in `excess_product` in `construction_helpers/composition.py`. It uses positional slicing (BOTTOM = last N elements) to identify which elements of `right.dom()` are "extra inputs" the left doesn't provide. When the shared axis (`A`) is in the middle of `right.dom()` — not at the end — the wrong axes get identified as excess and paired incorrectly in `align_composed`, causing runtime einsum subscript size conflicts.

The fix adds a pre-processing step to `composition()`: before calling `excess_product`, check if the matched elements are already at the correct position (front); if not, prepend a `Rearrangement` to `right` that reorders its domain so matched elements come first and excess elements come last. This puts the excess at the BOTTOM (end), which is exactly what the existing `excess_product` + `morphism_product` logic expects.

**Tech Stack:** Python 3.12, PyTorch, pytest. All changes are in `construction_helpers/composition.py`, `tests/test_torch_compile.py`, and `papers/transformer_example.md`.

---

## Background: what goes wrong

For `attn.cod() = (A[q:4, m:6],)` and `ffn.dom() = (W_in[d:8, m:6], A[q:4, m:6], W_out[m:6, d:8])`:

1. `excess_product(..., BOTTOM)`: `excess_left = 1−3 = −2`; BOTTOM 2 of `ffn.dom()` = `(A, W_out)` ← **wrong** (picks up `A` itself as excess)
2. `left = morphism_product((attn, (A, W_out)))` — extends attn to pass through A and W_out
3. `new_left.cod()` = `(A_attn, A_ffn, W_out_ffn)` — three elements
4. `align_axes` pairs them with `ffn.dom()` positionally: `A_attn ↔ W_in`, `A_ffn ↔ A_ffn`, `W_out ↔ W_out`
5. **Result**: attn's `q`-axis (size 4) is unified with ffn's `d`-axis (size 8) → einsum subscript collision at runtime

**Correct excess**: `(W_in[d:8, m:6], W_out[m:6, d:8])` — elements 0 and 2, not the end.

After the fix (reorder `ffn.dom()` to `(A, W_in, W_out)` before excess_product):

1. `excess_product((A,), (A, W_in, W_out), BOTTOM)`: BOTTOM 2 = `(W_in, W_out)` ✓
2. `left = morphism_product((attn, (W_in, W_out)))` — correct
3. `new_left.cod()` = `(A_attn, W_in_ffn, W_out_ffn)`
4. `align_axes` pairs: `A_attn ↔ A_ffn`, `W_in ↔ W_in`, `W_out ↔ W_out` ✓
5. q1 (size 4) unified with q2 (size 4) ✓ — no collision

---

## File map

| File | Change |
|---|---|
| `construction_helpers/composition.py` | Add `_array_signature`, `_find_composition_reordering`; update `composition()` |
| `tests/test_torch_compile.py` | Add `test_transformer_layer_output_shape` and `test_transformer_layer_matches_single_tl_program` |
| `papers/transformer_example.md` (optional) | Update transformer helper functions to use shared axes if needed |

---

## Task 1: Add helper functions and fix `composition()`

**Files:**
- Modify: `construction_helpers/composition.py`
- Test: `tests/test_torch_compile.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_torch_compile.py  — add after the transformer tests section

def test_transformer_layer_attn_ffn_composition_output_shape():
    """attn_res() @ ffn_res() (separately constructed TL morphisms) must compile
    and run without einsum subscript conflicts."""
    import data_structure.Category as cat
    SEQ, D, H, K, DFF = 4, 6, 2, 3, 8

    def mk_attn():
        tl = TL()
        q = real_axis('q', SEQ); x = norm_axis('x', SEQ)
        m = real_axis('m', D);   h = real_axis('h', H);  k = real_axis('k', K)
        tl.Query[q, h, k]    = tl.W_Q[h, k, m] * tl.H[q, m]
        tl.Key[x, h, k]      = tl.W_K[h, k, m] * tl.H[x, m]
        tl.Value[x, h, k]    = tl.W_V[h, k, m] * tl.H[x, m]
        tl.S[h, q, x]        = softmax(tl.Query[q, h, k] * tl.Key[x, h, k],
                                        where=(x <= q))
        tl.AttnOut[q, h, k]  = tl.S[h, q, x] * tl.Value[x, h, k]
        tl.Attn[q, m]        = tl.W_O[m, h, k] * tl.AttnOut[q, h, k]
        tl.A[q, m]           = normalize(tl.Attn[q, m] + tl.H[q, m])
        return tl.to_morphism()

    def mk_ffn():
        tl = TL()
        q = real_axis('q', SEQ); m = real_axis('m', D)
        d = real_axis('d_{ff}', DFF)
        tl.F[q, d]   = relu(tl.W_in[d, m] * tl.A[q, m])
        tl.Y[q, m]   = tl.W_out[m, d] * tl.F[q, d]
        tl.Out[q, m] = normalize(tl.Y[q, m] + tl.A[q, m])
        return tl.to_morphism()

    layer = mk_attn() @ mk_ffn()
    mod = ConstructedModule.construct(layer)

    torch.manual_seed(0)
    W_Q = torch.randn(H, K, D); H0 = torch.randn(SEQ, D)
    W_K = torch.randn(H, K, D); W_V = torch.randn(H, K, D)
    W_O = torch.randn(D, H, K)
    W_in = torch.randn(DFF, D); W_out = torch.randn(D, DFF)

    out = mod(W_Q, H0, W_K, W_V, W_O, W_in, W_out)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == torch.Size([SEQ, D]), f"Expected ({SEQ},{D}), got {out.shape}"


def test_transformer_layer_composition_matches_single_tl_program():
    """attn @ ffn composition must produce numerically identical output to
    a single TL program that encodes both sub-layers."""
    SEQ, D, H, K, DFF = 4, 6, 2, 3, 8

    def mk_attn():
        tl = TL()
        q = real_axis('q', SEQ); x = norm_axis('x', SEQ)
        m = real_axis('m', D);   h = real_axis('h', H);  k = real_axis('k', K)
        tl.Query[q, h, k]    = tl.W_Q[h, k, m] * tl.H[q, m]
        tl.Key[x, h, k]      = tl.W_K[h, k, m] * tl.H[x, m]
        tl.Value[x, h, k]    = tl.W_V[h, k, m] * tl.H[x, m]
        tl.S[h, q, x]        = softmax(tl.Query[q, h, k] * tl.Key[x, h, k],
                                        where=(x <= q))
        tl.AttnOut[q, h, k]  = tl.S[h, q, x] * tl.Value[x, h, k]
        tl.Attn[q, m]        = tl.W_O[m, h, k] * tl.AttnOut[q, h, k]
        tl.A[q, m]           = normalize(tl.Attn[q, m] + tl.H[q, m])
        return tl.to_morphism()

    def mk_ffn():
        tl = TL()
        q = real_axis('q', SEQ); m = real_axis('m', D)
        d = real_axis('d_{ff}', DFF)
        tl.F[q, d]   = relu(tl.W_in[d, m] * tl.A[q, m])
        tl.Y[q, m]   = tl.W_out[m, d] * tl.F[q, d]
        tl.Out[q, m] = normalize(tl.Y[q, m] + tl.A[q, m])
        return tl.to_morphism()

    def mk_full():
        tl = TL()
        q = real_axis('q', SEQ); x = norm_axis('x', SEQ)
        m = real_axis('m', D);   h = real_axis('h', H)
        k = real_axis('k', K);   d = real_axis('d_{ff}', DFF)
        tl.Query[q, h, k]    = tl.W_Q[h, k, m] * tl.H[q, m]
        tl.Key[x, h, k]      = tl.W_K[h, k, m] * tl.H[x, m]
        tl.Value[x, h, k]    = tl.W_V[h, k, m] * tl.H[x, m]
        tl.S[h, q, x]        = softmax(tl.Query[q, h, k] * tl.Key[x, h, k],
                                        where=(x <= q))
        tl.AttnOut[q, h, k]  = tl.S[h, q, x] * tl.Value[x, h, k]
        tl.Attn[q, m]        = tl.W_O[m, h, k] * tl.AttnOut[q, h, k]
        tl.A[q, m]           = normalize(tl.Attn[q, m] + tl.H[q, m])
        tl.F[q, d]           = relu(tl.W_in[d, m] * tl.A[q, m])
        tl.Y[q, m]           = tl.W_out[m, d] * tl.F[q, d]
        tl.Out[q, m]         = normalize(tl.Y[q, m] + tl.A[q, m])
        return tl.to_morphism()

    torch.manual_seed(7)
    W_Q = torch.randn(H, K, D); H0 = torch.randn(SEQ, D)
    W_K = torch.randn(H, K, D); W_V = torch.randn(H, K, D)
    W_O = torch.randn(D, H, K)
    W_in = torch.randn(DFF, D); W_out_t = torch.randn(D, DFF)

    # Composed
    mod_comp = ConstructedModule.construct(mk_attn() @ mk_ffn())
    out_comp = mod_comp(W_Q, H0, W_K, W_V, W_O, W_in, W_out_t)
    out_comp = out_comp[0] if isinstance(out_comp, tuple) else out_comp

    # Single TL program (reference)
    mod_full = ConstructedModule.construct(mk_full())
    out_full = mod_full(W_Q, H0, W_K, W_V, W_O, W_in, W_out_t)
    out_full = out_full[0] if isinstance(out_full, tuple) else out_full

    assert torch.allclose(out_comp, out_full, atol=1e-5), (
        f"Composition output differs from single-TL reference: "
        f"max diff {(out_comp - out_full).abs().max().item():.4f}"
    )
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/williammacready/code/python/pyncd
python -m pytest tests/test_torch_compile.py::test_transformer_layer_attn_ffn_composition_output_shape tests/test_torch_compile.py::test_transformer_layer_composition_matches_single_tl_program -v
```
Expected: both FAIL — first with `RuntimeError: einsum(): subscript b has size 4 for operand 1 which does not broadcast with previously seen size 8` (or similar runtime error), second with same or assertion error.

- [ ] **Step 3: Add helpers and fix `composition()` in `construction_helpers/composition.py`**

Add two helpers immediately before the `add_excess_lift` function (around line 90), then update `composition()`:

```python
# ---------------------------------------------------------------------------
# Shape-based matching helpers for cross-TL-instance composition
# ---------------------------------------------------------------------------

def _array_signature(arr) -> tuple | None:
    """Get a comparable shape signature for an Array: (datatype_type, ((name, size),...)).

    Returns None if arr is not an Array or has no concrete axis information.
    Axes with a free (unsized) numeric contribute None for their size slot,
    which still allows partial matching on name alone.
    """
    if not isinstance(arr, cat.Array):
        return None
    axis_sigs = tuple(
        (
            ax.uid._name.body if ax.uid._name else None,
            getattr(ax._size, '_value', None),
        )
        for ax in arr.shape()
    )
    return (type(arr.datatype), axis_sigs)


def _find_composition_reordering(
    left_cod: cat.ProdObject,
    right_dom: cat.ProdObject,
) -> tuple[int, ...] | None:
    """Find a permutation of right_dom's indices that puts matched elements FIRST
    and excess elements LAST, so that excess_product with BOTTOM convention
    correctly identifies the excess.

    Returns a tuple new_order where new_order[new_pos] = orig_pos, such that
    right_dom[new_order[0..len(left_cod)-1]] are the elements that match
    left_cod by shape signature.

    Returns None if reordering is unnecessary (already correct) or impossible
    (shapes are incompatible or ambiguous).
    """
    if len(left_cod) >= len(right_dom):
        return None  # No excess right needed

    left_sigs = [_array_signature(obj) for obj in left_cod]
    if any(s is None for s in left_sigs):
        return None  # Non-Array objects — skip shape matching

    right_sigs = [_array_signature(obj) for obj in right_dom]

    # Greedily match each left element to a right element by shape signature
    matched: list[int] = []
    used: set[int] = set()
    for left_sig in left_sigs:
        for i, right_sig in enumerate(right_sigs):
            if i not in used and right_sig == left_sig:
                matched.append(i)
                used.add(i)
                break

    if len(matched) != len(left_cod):
        return None  # Could not find all matches — fall back to positional

    excess = [i for i in range(len(right_dom)) if i not in used]

    # New order: matched first, excess last
    new_order = tuple(matched + excess)

    if new_order == tuple(range(len(right_dom))):
        return None  # Already in the correct order — no reordering needed

    return new_order
```

Then update `composition()` — replace lines 135–142:

```python
def composition( # type: ignore
        left,
        right):
    if isinstance(left, tuple):
        dom_length = max(left) + 1
        dom = tuple(
            util.iallequals(
                right.dom()[cod_idx]
                for cod_idx in left
                if cod_idx == idx
            )
            for idx in range(dom_length)
        )
        left = cat.Rearrangement(
            mapping=left,
            _dom=dom
        )
    if isinstance(right, tuple):
        right = cat.Rearrangement(
            mapping=right,
            _dom = tuple(left.cod())
        )
    if isinstance(left.cod()[0], cat.Array):
        left, right = add_excess_lift(left, right)

    # When right.dom() has more elements than left.cod(), check if a shape-based
    # reordering is needed to correctly identify the excess inputs.
    # excess_product uses positional (BOTTOM) slicing, which assumes the shared
    # elements come FIRST in right.dom() and excess elements come LAST.
    # If the shared element is not at the front (e.g. ffn takes (W_in, A, W_out)
    # but attn only produces A), we prepend a Rearrangement to right that moves
    # the matched element(s) to the front.
    if len(left.cod()) < len(right.dom()):
        new_order = _find_composition_reordering(left.cod(), right.dom())
        if new_order is not None:
            # Build the inverse permutation: inv[orig_pos] = new_pos.
            # This is the mapping for the Rearrangement that takes inputs in
            # the new (reordered) order and routes them to right's original slots.
            inv_order = [0] * len(new_order)
            for new_pos, orig_pos in enumerate(new_order):
                inv_order[orig_pos] = new_pos
            reordered_dom = cat.ProdObject.from_iter(
                right.dom()[orig_pos] for orig_pos in new_order
            )
            rearrangement = cat.Rearrangement(
                mapping=tuple(inv_order),
                _dom=tuple(reordered_dom),
            )
            right = align_composed(rearrangement, right)

    excess_left, excess_right = excess_product(left.cod(), right.dom(), ExcessProductSide.BOTTOM)
    if excess_left is not None:
        left = chp.morphism_product((left, excess_left))
    elif excess_right is not None:
        right = chp.morphism_product((right, excess_right))
    return align_composed(left, right)
```

- [ ] **Step 4: Run the two new tests — expect PASS**

```bash
python -m pytest tests/test_torch_compile.py::test_transformer_layer_attn_ffn_composition_output_shape tests/test_torch_compile.py::test_transformer_layer_composition_matches_single_tl_program -v
```

- [ ] **Step 5: Run the full suite — expect no regressions**

```bash
python -m pytest tests/test_tensor_dsl.py tests/test_torch_compile.py -q
```

- [ ] **Step 6: Commit**

```bash
git add construction_helpers/composition.py tests/test_torch_compile.py
git commit -m "fix: shape-based axis matching in composition() for cross-TL-instance @ operator

excess_product uses positional (BOTTOM) slicing to identify which elements
of right.dom() are 'extra inputs' the left doesn't provide.  This fails
when the shared axis is not at the end of right.dom() — e.g. attn.cod()=(A)
and ffn.dom()=(W_in, A, W_out): BOTTOM 2 picks (A, W_out) as excess instead
of (W_in, W_out), causing wrong axis pairings and runtime einsum errors.

Fix: before excess_product, if right.dom() has excess elements, try to find
which elements match left.cod() by axis (name, size) signature.  If the
match is not already at the front, prepend a Rearrangement to right that
reorders its domain to put matched elements first, excess last — exactly the
layout excess_product expects.

This makes attn_res() @ ffn_res() work correctly when the two morphisms
come from separate TL() instances with independent axis UIDs.
"
```

---

## Task 2: Update `transformer_example.md` TL DSL code

**Background:** The `attn_res()`, `ffn_res()`, and `transformer_layer()` functions in the TL DSL section of `papers/transformer_example.md` currently use the modular pattern (`attn_res() @ ffn_res()`). Now that the composition fix is in place, verify this pattern works and update the code to use `norm_axis + where=` (already done in previous work), confirming the end-to-end example is correct.

**Files:**
- Modify: `papers/transformer_example.md` (verify only — code was already updated)

- [ ] **Step 1: Verify the transformer example runs end-to-end**

```python
# Quick smoke test: paste the TL DSL code from the paper and run it
python -c "
import sys; sys.path.insert(0, '/Users/williammacready/code/python/pyncd')
import data_structure.Category as cat
from data_structure.TensorDSL import TL, real_axis, norm_axis, softmax, normalize, relu
from torch_compile.torch_compile import ConstructedModule
import torch

SEQ, D, H, K, DFF = 4, 6, 2, 3, 8

def _m(): return real_axis('m', D)
def _h(): return real_axis('h', H)
def _k(): return real_axis('k', K)
def _d_ff(): return real_axis('d_{ff}', DFF)

def attn_res():
    tl = TL()
    q = real_axis('q', SEQ)
    x = norm_axis('x', SEQ)
    m = _m(); h = _h(); k = _k()
    tl.Query[q,h,k]  = tl.W_Q[h,k,m]*tl.H[q,m]
    tl.Key[x,h,k]    = tl.W_K[h,k,m]*tl.H[x,m]
    tl.Value[x,h,k]  = tl.W_V[h,k,m]*tl.H[x,m]
    tl.S[h,q,x]      = softmax(tl.Query[q,h,k]*tl.Key[x,h,k], where=(x<=q))
    tl.AttnOut[q,h,k]= tl.S[h,q,x]*tl.Value[x,h,k]
    tl.Attn[q,m]     = tl.W_O[m,h,k]*tl.AttnOut[q,h,k]
    tl.A[q,m]        = normalize(tl.Attn[q,m]+tl.H[q,m])
    return tl.to_morphism()

def ffn_res():
    tl = TL()
    q = real_axis('q', SEQ); m = _m(); d = _d_ff()
    tl.F[q,d]   = relu(tl.W_in[d,m]*tl.A[q,m])
    tl.Y[q,m]   = tl.W_out[m,d]*tl.F[q,d]
    tl.Out[q,m] = normalize(tl.Y[q,m]+tl.A[q,m])
    return tl.to_morphism()

layer = attn_res() @ ffn_res()
mod = ConstructedModule.construct(layer)
out = mod(torch.randn(H,K,D), torch.randn(SEQ,D), torch.randn(H,K,D),
          torch.randn(H,K,D), torch.randn(D,H,K), torch.randn(DFF,D), torch.randn(D,DFF))
out = out[0] if isinstance(out, tuple) else out
print('output shape:', out.shape)
assert out.shape == torch.Size([SEQ, D])
print('OK')
"
```

Expected: `output shape: torch.Size([4, 6])` and `OK`.

- [ ] **Step 2: If the example fails, investigate and report**

If step 1 fails for any reason (e.g. the Block-wrapping used in `transformer_layer()` introduces a different issue), report the exact error and explain whether it is covered by Task 1 or requires a separate fix.

- [ ] **Step 3: Commit if any changes were made to the doc**

```bash
git add papers/transformer_example.md
git commit -m "docs: verify transformer_example TL DSL code works with composition fix"
```

If no changes needed (example already works), skip this step.
