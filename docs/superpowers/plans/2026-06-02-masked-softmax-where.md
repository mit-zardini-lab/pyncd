# Masked Softmax `where=` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `softmax(..., where=iverson_predicate)` to the TL DSL so that Iverson predicates in softmax arguments are compiled as `-inf` masks before exponentiation rather than as multiplicative `{0,1}` factors, giving mathematically correct masked attention for any predicate.

**Architecture:** Three coordinated changes: (1) fix `EqualityClass.from_iter` in `Term.py` to accept mixed `NormAxis`/`RawAxis` types (prerequisite crash fix); (2) thread a `where_predicate` field through `SoftMax` → `_split_nonlinearity` → new `MaskedSoftMax` operator; (3) implement `ConstructedMaskedSoftMax` which materialises the Iverson, applies a compile-time-computed axis permutation/broadcast to align it with the score tensor, then does `masked_fill(-inf) + softmax`. The TL user writes `softmax(score, where=(x <= q))` with `x = norm_axis('x', SEQ)`; no other change to existing call sites.

**Tech Stack:** Python 3.12, PyTorch, einops, pytest. All changes are in `data_structure/`, `torch_compile/`, `tests/`, and `papers/`.

---

## File map

| File | Change |
|---|---|
| `data_structure/Term.py:240-250` | Fix `EqualityClass.from_iter` — accept NormAxis/RawAxis type mix |
| `data_structure/TensorDSL.py:1235-1237` | Extend `norm_axis()` to accept optional `size` |
| `data_structure/TensorDSL.py:985-989` | Remove NormAxis guard from `TL.bc_signature()` |
| `data_structure/Operators.py:91-106` | Add `where_predicate: tuple = ()` field to `SoftMax` |
| `data_structure/Operators.py` (after SoftMax) | New `MaskedSoftMax` operator class |
| `data_structure/TensorDSL.py:1268-1274` | Add `where=` parameter to `softmax()` |
| `data_structure/TensorLogic.py` (before `_split_nonlinearity`) | New `_compute_mask_alignment()` helper |
| `data_structure/TensorLogic.py:178-179` | Update `_split_nonlinearity` for `where=` → `MaskedSoftMax` path |
| `torch_compile/torch_compile.py` (after `ConstructedNorm`) | New `ConstructedMaskedSoftMax` class |
| `tests/test_tensor_dsl.py` | Unit tests for `norm_axis(size)`, `where=` syntax |
| `tests/test_torch_compile.py` | Integration tests for masked softmax + causal invariance |
| `papers/transformer_example.md` | Update TL DSL code to use `norm_axis + where=` |

---

## Task 1: Fix `EqualityClass` to accept NormAxis/RawAxis type mix

**Background:** `EqualityClass.from_iter` calls `util.iallequals(map(type, target))` which raises `ValueError: Elements are not all equal: NormAxis != RawAxis` when a `NormAxis` (from the score's LHS) and a fresh `RawAxis` (from a template) are unified in the same equality class during `@` composition. `NormAxis` is a subclass of `RawAxis`, so the fix is to use the most-general common base type rather than requiring strict type equality.

**Files:**
- Modify: `data_structure/Term.py:240-250`
- Test: `tests/test_tensor_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tensor_dsl.py  — add after existing NormAxis tests

def test_norm_axis_softmax_compiles_to_composed():
    """norm_axis() in the LHS of a softmax equation must compile to a Composed
    (einsum @ SoftMax template) rather than crashing with a type-equality error."""
    from data_structure.ProductCategory import Composed
    q, h, k = axes('q h k')
    x = norm_axis('x')
    tl = TL()
    tl.Out[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k])
    sig = tl.bc_signature()
    assert isinstance(sig, Composed), (
        f"Expected Composed(einsum, softmax); got {type(sig).__name__}"
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /Users/williammacready/code/python/pyncd
python -m pytest tests/test_tensor_dsl.py::test_norm_axis_softmax_compiles_to_composed -v
```
Expected: FAIL with `ValueError: Elements are not all equal: NormAxis != RawAxis` (or `AssertionError` because `bc_signature` returns the raw Broadcasted via the guard — either fail confirms the test is active).

- [ ] **Step 3: Fix `EqualityClass.from_iter` in `data_structure/Term.py`**

Replace lines 240–250:

```python
    @classmethod
    def from_iter(cls, target: Iterable[T]) -> EqualityClass[T]:
        target = tuple(target)
        # Determine the most-general common type.  NormAxis and NatAxis are
        # subclasses of RawAxis; allow them to coexist in one equality class
        # by widening to the common base.
        types = [type(t) for t in target]
        _type: type = types[0]
        for t in types[1:]:
            if t == _type:
                continue
            if issubclass(_type, t):
                _type = t        # t is more general; widen
            elif not issubclass(t, _type):
                raise ValueError(
                    f"Elements are not all equal: {_type.__name__} != {t.__name__}"
                )
        return EqualityClass(
            _type=_type,
            bucket={t.uid for t in target},
            canonical=max(target, key=lambda uterm: uterm.uid),
        )
```

- [ ] **Step 4: Remove the NormAxis guard from `TL.bc_signature()`**

In `data_structure/TensorDSL.py` around line 985, delete these five lines:

```python
        # NormAxis equations cannot be composed with a generic RawAxis-typed template
        # (the @-composition unification would crash because NormAxis != RawAxis).
        # Return the raw Broadcasted directly; the nonlinearity is embedded in the operator.
        if any(isinstance(ax, NormAxis) for ax in eq.lhs_indices):
            return morph
```

The method should now flow directly to `return _split_nonlinearity(...)`.

- [ ] **Step 5: Run the new test**

```bash
python -m pytest tests/test_tensor_dsl.py::test_norm_axis_softmax_compiles_to_composed -v
```
Expected: PASS.

- [ ] **Step 6: Run the full suite to confirm no regressions**

```bash
python -m pytest tests/test_tensor_dsl.py tests/test_torch_compile.py -q
```
Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add data_structure/Term.py data_structure/TensorDSL.py tests/test_tensor_dsl.py
git commit -m "fix: allow NormAxis/RawAxis type mix in EqualityClass; remove NormAxis bc_signature guard"
```

---

## Task 2: Extend `norm_axis()` to accept an optional size

**Background:** Auto-materialising an Iverson predicate with `materialise_iverson()` requires every leaf axis to have a concrete integer size. `norm_axis('x')` currently creates a free-size axis. We extend it to accept `size`, following the same pattern as `nat_axis` and `real_axis`.

**Files:**
- Modify: `data_structure/TensorDSL.py:1235-1237`
- Test: `tests/test_tensor_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
def test_norm_axis_accepts_optional_size():
    """norm_axis('x', 5) must produce a NormAxis with concrete Integer size."""
    from data_structure.Numeric import Integer
    x = norm_axis('x', 5)
    assert isinstance(x, NormAxis)
    assert x._size == Integer(5)

def test_norm_axis_no_size_unchanged():
    """norm_axis('x') without size must still work as before (free size)."""
    from data_structure.Numeric import FreeNumeric
    x = norm_axis('x')
    assert isinstance(x, NormAxis)
    assert isinstance(x._size, FreeNumeric)
```

- [ ] **Step 2: Run to confirm first test fails**

```bash
python -m pytest tests/test_tensor_dsl.py::test_norm_axis_accepts_optional_size -v
```
Expected: FAIL with `TypeError: norm_axis() takes 1 positional argument but 2 were given`.

- [ ] **Step 3: Update `norm_axis()` in `data_structure/TensorDSL.py`**

Replace lines 1235–1237:

```python
def norm_axis(name: str, size: int | None = None) -> NormAxis:
    """Return a NormAxis — marks the normalisation dimension (e.g. softmax axis).

    When size is provided, the axis carries a concrete integer size, which is
    required for auto-materialising Iverson predicates that reference this axis.
    """
    base = NormAxis.named(name)
    if size is None:
        return base
    return NormAxis(uid=base.uid, _size=nm.Integer(size))
```

- [ ] **Step 4: Run both new tests**

```bash
python -m pytest tests/test_tensor_dsl.py::test_norm_axis_accepts_optional_size tests/test_tensor_dsl.py::test_norm_axis_no_size_unchanged -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add data_structure/TensorDSL.py tests/test_tensor_dsl.py
git commit -m "feat: norm_axis() accepts optional size for Iverson materialisation"
```

---

## Task 3: Add `where_predicate` to `SoftMax` and `where=` to `softmax()`

**Background:** `SoftMax` is a frozen dataclass in `Operators.py`. We add `where_predicate: tuple = ()` to carry zero or more Iverson expressions. The `softmax()` helper in `TensorDSL.py` gains a `where=` keyword that populates this field.

**Files:**
- Modify: `data_structure/Operators.py:91-106`
- Modify: `data_structure/TensorDSL.py:1268-1274` (also add `IversonExpr` to top import)
- Test: `tests/test_tensor_dsl.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_softmax_where_stores_predicate():
    """softmax(expr, where=pred) must store the predicate in the operator."""
    q, x = axes('q x')
    tl = TL()
    pred = x <= q   # IversonBinOp('<=', x, q)
    expr = softmax(tl.A[q, x], where=pred)
    assert len(expr.operator.where_predicate) == 1
    assert isinstance(expr.operator.where_predicate[0], IversonBinOp)
    assert expr.operator.where_predicate[0].op == '<='


def test_softmax_no_where_has_empty_predicate():
    """softmax(expr) without where= must have empty where_predicate."""
    q, x = axes('q x')
    tl = TL()
    expr = softmax(tl.A[q, x])
    assert expr.operator.where_predicate == ()
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_tensor_dsl.py::test_softmax_where_stores_predicate tests/test_tensor_dsl.py::test_softmax_no_where_has_empty_predicate -v
```
Expected: FAIL (`AttributeError: where_predicate` or similar).

- [ ] **Step 3: Add `where_predicate` to `SoftMax` in `data_structure/Operators.py`**

Replace lines 91–106:

```python
@dataclass(frozen=True)
class SoftMax(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('SoftMax')
    contracted: bool = False
    where_predicate: tuple = ()   # IversonExpr objects; empty = plain softmax
    @classmethod
    def template[B:cat.Datatype=cat.Reals](
        cls,
        base: B = cat.Reals(),
    ):
        axis = cat.RawAxis()
        return cat.Broadcasted[B, cat.RawAxis](
            operator=SoftMax(name=fd.DynamicName('SoftMax')),
            input_weaves=(cat.Weave(base, (axis,)),),
            output_weaves=(cat.Weave(base, (axis,)),),
            reindexings=(cat.ProdObject().identity(),)
        )
```

- [ ] **Step 4: Add `IversonExpr` import and update `softmax()` in `data_structure/TensorDSL.py`**

At line 13, extend the import:
```python
from data_structure.TensorExpr import (
    TensorRef, IversonBinOp, IversonUnaryOp,
    IversonExpr,
    ieq, imul, iabs,
)
```

Replace lines 1268–1274:
```python
def softmax(
    expr: IndexedTensor | RHSExpression | SumExpr,
    where: IversonExpr | None = None,
) -> RHSExpression | SumExpr:
    """Wrap a score expression with a SoftMax nonlinearity.

    where: optional Iverson predicate.  Positions where the predicate is False
           are masked to -inf before exponentiation (masked softmax), so they
           receive zero attention weight regardless of the raw score.
           The normalization axis must be marked with norm_axis() in the LHS.
    """
    op = ops.SoftMax(where_predicate=((where,) if where is not None else ()))
    if isinstance(expr, SumExpr):
        return SumExpr(expr.terms, op)
    if isinstance(expr, IndexedTensor):
        expr = RHSExpression([expr], ops.Identity())
    return RHSExpression(expr.factors, op)
```

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/test_tensor_dsl.py::test_softmax_where_stores_predicate tests/test_tensor_dsl.py::test_softmax_no_where_has_empty_predicate -v
```
Expected: both PASS.

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/test_tensor_dsl.py tests/test_torch_compile.py -q
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add data_structure/Operators.py data_structure/TensorDSL.py tests/test_tensor_dsl.py
git commit -m "feat: SoftMax.where_predicate field and softmax(where=) parameter"
```

---

## Task 4: Add `_compute_mask_alignment()` helper

**Background:** The Iverson predicate materialises with axes in DFS order (e.g., `(x <= q)` gives shape `(X, Q)`). The score tensor has axes in LHS order (e.g., `(H, Q, X)`). `_compute_mask_alignment` computes, at compile time, a permutation and broadcast count so that `ConstructedMaskedSoftMax` can call `mask.permute(perm).unsqueeze(0)...` to align the mask with the score.

**Files:**
- Modify: `data_structure/TensorLogic.py` (add before `_split_nonlinearity`)
- Test: `tests/test_tensor_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_mask_alignment_causal():
    """For (x <= q) with LHS [h, q, x]:
    - Iverson DFS order: [x(pos=0), q(pos=1)], shape (X, Q)
    - Score LHS order: [h, q, x]
    - q is at iverson pos 1, lhs pos 1
    - x is at iverson pos 0, lhs pos 2
    - h is not in iverson -> n_broadcast = 1
    - perm = [1, 0]  (put iverson dim 1 first, then dim 0)
    """
    from data_structure.TensorLogic import _compute_mask_alignment
    h = real_axis('h', 2)
    q = real_axis('q', 4)
    x = real_axis('x', 4)
    pred = x <= q   # IversonBinOp('<=', x, q); DFS: [x, q]
    perm, n_broadcast = _compute_mask_alignment(pred, (h, q, x))
    assert perm == (1, 0), f"Expected (1, 0), got {perm}"
    assert n_broadcast == 1, f"Expected 1, got {n_broadcast}"


def test_compute_mask_alignment_no_broadcast():
    """For (a == b) with LHS [a, b]: no broadcast dims, perm = (0, 1) (identity)."""
    from data_structure.TensorLogic import _compute_mask_alignment
    a = real_axis('a', 3)
    b = real_axis('b', 3)
    pred = ieq(a, b)  # IversonBinOp('==', a, b); DFS: [a, b]
    perm, n_broadcast = _compute_mask_alignment(pred, (a, b))
    assert perm == (0, 1)
    assert n_broadcast == 0
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_tensor_dsl.py::test_compute_mask_alignment_causal tests/test_tensor_dsl.py::test_compute_mask_alignment_no_broadcast -v
```
Expected: FAIL (`ImportError: cannot import name '_compute_mask_alignment'`).

- [ ] **Step 3: Add `_compute_mask_alignment` to `data_structure/TensorLogic.py`**

Add the following function immediately before `_split_nonlinearity` (around line 155):

```python
def _compute_mask_alignment(
    iverson_factor: IversonBinOp | IversonUnaryOp,
    lhs_indices: tuple[sc.RawAxis, ...],
) -> tuple[tuple[int, ...], int]:
    """Compute axis permutation and broadcast count to align a materialised
    Iverson mask with the score tensor produced by an einsum.

    materialise_iverson(factor) returns a tensor whose shape matches
    _iverson_axes(factor) in DFS order.  The score tensor's shape matches
    lhs_indices in that order.  This function returns (perm, n_broadcast)
    such that:

        mask = materialise_iverson(factor)      # shape: iverson DFS axes
        mask = mask.permute(list(perm))         # reorder to lhs axis order
        for _ in range(n_broadcast):            # prepend 1-dims for broadcast
            mask = mask.unsqueeze(0)
        # mask now broadcasts correctly with the score tensor

    Axes that appear in lhs_indices but not in the Iverson contribute one
    leading broadcast dimension each.
    """
    from data_structure.TensorExpr import _iverson_axes
    iverson_axes = _iverson_axes(iverson_factor)
    uid_to_iverson_pos = {ax.uid: i for i, ax in enumerate(iverson_axes)}

    in_iverson: list[tuple[int, int]] = []   # (iverson_pos, lhs_pos)
    n_broadcast = 0
    for lhs_pos, lhs_ax in enumerate(lhs_indices):
        if lhs_ax.uid in uid_to_iverson_pos:
            in_iverson.append((uid_to_iverson_pos[lhs_ax.uid], lhs_pos))
        else:
            n_broadcast += 1

    in_iverson.sort(key=lambda t: t[1])   # sort by lhs_pos → output order
    perm = tuple(iverson_pos for iverson_pos, _ in in_iverson)
    return perm, n_broadcast
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_tensor_dsl.py::test_compute_mask_alignment_causal tests/test_tensor_dsl.py::test_compute_mask_alignment_no_broadcast -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add data_structure/TensorLogic.py tests/test_tensor_dsl.py
git commit -m "feat: _compute_mask_alignment() helper for axis-aligned masked softmax"
```

---

## Task 5: Add `MaskedSoftMax` operator

**Background:** `MaskedSoftMax` is a new operator that carries the Iverson expressions and their pre-computed axis alignments. `ConstructedMaskedSoftMax` (Task 6) uses the stored `mask_alignments` to permute and broadcast each materialised Iverson buffer before `masked_fill`.

**Files:**
- Modify: `data_structure/Operators.py` (add after the `SoftMax` class)
- Test: `tests/test_tensor_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
def test_masked_softmax_operator_template_is_broadcasted():
    """MaskedSoftMax.template() must return a Broadcasted with 1-axis weaves."""
    import data_structure.Operators as ops
    import data_structure.BroadcastedCategory as bc
    q, x = axes('q x')
    pred = x <= q
    perm, n_broadcast = (1, 0), 1  # dummy values
    template = ops.MaskedSoftMax.template(
        iverson_factors=(pred,),
        mask_alignments=((perm, n_broadcast),),
    )
    assert isinstance(template, bc.Broadcasted)
    assert len(template.input_weaves) == 1
    assert len(template.output_weaves) == 1
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest tests/test_tensor_dsl.py::test_masked_softmax_operator_template_is_broadcasted -v
```
Expected: FAIL (`AttributeError: module 'data_structure.Operators' has no attribute 'MaskedSoftMax'`).

- [ ] **Step 3: Add `MaskedSoftMax` to `data_structure/Operators.py`**

Add the following class immediately after the `SoftMax` class (after line 106, before the `Einops` class):

```python
@dataclass(frozen=True)
class MaskedSoftMax(cat.Operator):
    """Softmax with Iverson predicates applied as -inf masks before exponentiation.

    iverson_factors: tuple of IversonExpr objects (one per predicate).
    mask_alignments: tuple of (perm: tuple[int,...], n_broadcast: int) — one
        entry per factor.  ConstructedMaskedSoftMax uses these to permute and
        unsqueeze the materialised mask so it broadcasts correctly with the
        score tensor.
    """
    name: fd.DynamicName | None = fd.DynamicName('MaskedSoftMax')
    iverson_factors: tuple = ()
    mask_alignments: tuple = ()   # tuple of (perm: tuple, n_broadcast: int)

    @classmethod
    def template(
        cls,
        iverson_factors: tuple,
        mask_alignments: tuple,
        base: cat.Datatype = cat.Reals(),
    ) -> cat.Broadcasted:
        """Return a 1-axis Broadcasted for composition with the score einsum."""
        axis = cat.RawAxis()
        return cat.Broadcasted(
            operator=cls(
                iverson_factors=iverson_factors,
                mask_alignments=mask_alignments,
            ),
            input_weaves=(cat.Weave(base, (axis,)),),
            output_weaves=(cat.Weave(base, (axis,)),),
            reindexings=(cat.ProdObject().identity(),),
        )
```

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/test_tensor_dsl.py::test_masked_softmax_operator_template_is_broadcasted -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_structure/Operators.py tests/test_tensor_dsl.py
git commit -m "feat: MaskedSoftMax operator with iverson_factors and mask_alignments"
```

---

## Task 6: Update `_split_nonlinearity` for the `where=` path

**Background:** When `op` is `SoftMax` with a non-empty `where_predicate`, build the score einsum from tensor factors only (no Iverson), compute the alignment for each predicate, and return `score_br @ MaskedSoftMax.template(...)`. The existing `br @ SoftMax.template()` path is unchanged for plain softmax.

**Files:**
- Modify: `data_structure/TensorLogic.py:178-179`
- Test: `tests/test_tensor_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
def test_split_nonlinearity_with_where_produces_masked_softmax():
    """softmax(QK, where=(x<=q)) must compile to Composed(score_einsum, MaskedSoftMax)."""
    import data_structure.Operators as ops
    from data_structure.ProductCategory import Composed
    q = real_axis('q', 4)
    h = real_axis('h', 2)
    k = real_axis('k', 3)
    x = norm_axis('x', 4)
    tl = TL()
    tl.S[h, q, x] = softmax(tl.Q[q, h, k] * tl.K[x, h, k], where=(x <= q))
    sig = tl.bc_signature()
    assert isinstance(sig, Composed), f"Expected Composed, got {type(sig).__name__}"
    # The second step must be a MaskedSoftMax Broadcasted
    last = sig.content[-1]
    import data_structure.BroadcastedCategory as bc
    assert isinstance(last, bc.Broadcasted)
    assert isinstance(last.operator, ops.MaskedSoftMax)
    assert len(last.operator.iverson_factors) == 1
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest tests/test_tensor_dsl.py::test_split_nonlinearity_with_where_produces_masked_softmax -v
```
Expected: FAIL (currently produces `Composed(einsum, SoftMax)`, not `MaskedSoftMax`).

- [ ] **Step 3: Update `_split_nonlinearity` in `data_structure/TensorLogic.py`**

Replace lines 178–179 (`if isinstance(op, ops.SoftMax): return br @ ops.SoftMax.template()`):

```python
    if isinstance(op, ops.SoftMax):
        if op.where_predicate:
            # Build score einsum from tensor-only factors; Iverson factors
            # are handled separately as -inf masks before exponentiation.
            tensor_rhs = tuple(f for f in eq.rhs if isinstance(f, TensorRef))
            score_eq = TensorEquation(
                lhs_name=eq.lhs_name,
                lhs_indices=eq.lhs_indices,
                rhs=tensor_rhs,
                operator=None,
            )
            score_br = score_eq.bc_signature(
                datatype=datatype, array_datatypes=array_datatypes
            )
            alignments = tuple(
                _compute_mask_alignment(factor, eq.lhs_indices)
                for factor in op.where_predicate
            )
            return score_br @ ops.MaskedSoftMax.template(
                iverson_factors=op.where_predicate,
                mask_alignments=alignments,
            )
        return br @ ops.SoftMax.template()
```

- [ ] **Step 4: Run the test**

```bash
python -m pytest tests/test_tensor_dsl.py::test_split_nonlinearity_with_where_produces_masked_softmax -v
```
Expected: PASS.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/test_tensor_dsl.py tests/test_torch_compile.py -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add data_structure/TensorLogic.py tests/test_tensor_dsl.py
git commit -m "feat: _split_nonlinearity routes softmax(where=) to MaskedSoftMax path"
```

---

## Task 7: Add `ConstructedMaskedSoftMax` to torch_compile

**Background:** `ConstructedMaskedSoftMax` materialises each Iverson factor as a registered buffer, then in `forward()` applies the stored permutation and broadcast operations to align each mask with the score tensor before `masked_fill(-inf) + torch.softmax`.

**Files:**
- Modify: `torch_compile/torch_compile.py` (add after the `ConstructedNorm` class, line ~488)
- Test: `tests/test_torch_compile.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torch_compile.py  — add in the "normalize() semantics" section

def test_masked_softmax_excludes_future_positions():
    """softmax(QK, where=(x<=q)) must give S[h,q,x]=0 for x>q and rows summing to 1."""
    SEQ, H = 5, 2
    q_ax = real_axis('q', SEQ)
    h_ax = real_axis('h', H)
    k_ax = real_axis('k', 3)
    x_ax = norm_axis('x', SEQ)
    tl = TL()
    tl.S[h_ax, q_ax, x_ax] = softmax(
        tl.Q[q_ax, h_ax, k_ax] * tl.K[x_ax, h_ax, k_ax],
        where=(x_ax <= q_ax),
    )
    mod = ConstructedModule.construct(tl.to_morphism())

    torch.manual_seed(0)
    Q_t = torch.randn(SEQ, H, 3)
    K_t = torch.randn(SEQ, H, 3)
    S = mod(Q_t, K_t)
    S = S[0] if isinstance(S, tuple) else S    # shape (H, Q, X)

    upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert S[:, upper].abs().max() < 1e-6, (
        f"Non-zero upper-triangle: max={S[:, upper].abs().max().item()}"
    )
    assert torch.allclose(S.sum(dim=-1), torch.ones(H, SEQ), atol=1e-5), (
        f"Rows do not sum to 1: {S.sum(dim=-1)}"
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /Users/williammacready/code/python/pyncd
python -m pytest tests/test_torch_compile.py::test_masked_softmax_excludes_future_positions -v
```
Expected: FAIL (`NotImplementedError` — no constructor registered for `MaskedSoftMax`).

- [ ] **Step 3: Add `ConstructedMaskedSoftMax` to `torch_compile/torch_compile.py`**

Add the following class immediately after `ConstructedNorm` (around line 488, before the `ConstructedScan` class):

```python
class ConstructedMaskedSoftMax(
    ConstructedModule, operation_key=ops.MaskedSoftMax
):
    """Softmax with Iverson predicates applied as -inf masks before exponentiation.

    Each Iverson factor is pre-materialised as a registered buffer.  At
    forward time the buffer is permuted and unsqueezed (using the compile-time
    alignment metadata) to match the score tensor shape, then masked_fill
    sets excluded positions to -inf before torch.softmax.
    """

    def __init__(self, target: cat.Broadcasted):
        super().__init__(target)
        op = target.operator
        displacement = bcast.get_displacement(target)
        self._dim = displacement if displacement is not None else -1
        self._n_masks = len(op.iverson_factors)
        # Store alignments as plain lists (not tensors) so they survive serialisation.
        self._alignments: list[tuple[list[int], int]] = [
            (list(perm), n_broadcast)
            for perm, n_broadcast in op.mask_alignments
        ]
        for i, factor in enumerate(op.iverson_factors):
            try:
                buf = materialise_iverson(factor)
                self.register_buffer(f'_mask_{i}', buf)
            except ValueError as e:
                warnings.warn(
                    f"MaskedSoftMax factor {i} has unsized axes and cannot "
                    f"be auto-materialised; it will be skipped. ({e})",
                    stacklevel=2,
                )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        score = xs[0]
        for i, (perm, n_broadcast) in enumerate(self._alignments):
            raw = getattr(self, f'_mask_{i}', None)
            if raw is None:
                continue
            if perm:
                raw = raw.permute(perm)
            for _ in range(n_broadcast):
                raw = raw.unsqueeze(0)
            score = score.masked_fill(raw == 0, float('-inf'))
        return torch.softmax(score, dim=self._dim)
```

- [ ] **Step 4: Run the new test**

```bash
python -m pytest tests/test_torch_compile.py::test_masked_softmax_excludes_future_positions -v
```
Expected: PASS.

- [ ] **Step 5: Run the full suite**

```bash
python -m pytest tests/test_tensor_dsl.py tests/test_torch_compile.py -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add torch_compile/torch_compile.py tests/test_torch_compile.py
git commit -m "feat: ConstructedMaskedSoftMax — masked_fill(-inf) + softmax via axis-aligned Iverson buffers"
```

---

## Task 8: Causal invariance integration test

**Background:** The end-to-end test verifies that the full `attn_res` module using `norm_axis + where=` is exactly causal-invariant: changing future token inputs must not affect past query positions.

**Files:**
- Modify: `tests/test_torch_compile.py`

- [ ] **Step 1: Add the integration test**

```python
def test_attn_res_with_where_is_exactly_causally_invariant():
    """Full attn_res using norm_axis + where= must be exactly causal-invariant.

    Changing H[2:] must not affect output at positions 0 and 1.
    This test fails with the old softmax(QK) * mask approach (see commit
    history) because future keys with large scores dominate the softmax
    denominator, causing the causal values to underflow and the clamp_min
    in normalize() to corrupt the result.
    """
    SEQ, D, H, K = 4, 6, 2, 3
    q_ax = real_axis('q', SEQ)
    x_ax = norm_axis('x', SEQ)    # norm_axis marks normalization axis
    m_ax = real_axis('m', D)
    h_ax = real_axis('h', H)
    k_ax = real_axis('k', K)
    tl = TL()
    tl.Query[q_ax, h_ax, k_ax]    = tl.W_Q[h_ax, k_ax, m_ax] * tl.H[q_ax, m_ax]
    tl.Key[x_ax, h_ax, k_ax]      = tl.W_K[h_ax, k_ax, m_ax] * tl.H[x_ax, m_ax]
    tl.Value[x_ax, h_ax, k_ax]    = tl.W_V[h_ax, k_ax, m_ax] * tl.H[x_ax, m_ax]
    tl.S[h_ax, q_ax, x_ax]        = softmax(
        tl.Query[q_ax, h_ax, k_ax] * tl.Key[x_ax, h_ax, k_ax],
        where=(x_ax <= q_ax),
    )
    tl.AttnOut[q_ax, h_ax, k_ax]  = tl.S[h_ax, q_ax, x_ax] * tl.Value[x_ax, h_ax, k_ax]
    tl.Attn[q_ax, m_ax]           = tl.W_O[m_ax, h_ax, k_ax] * tl.AttnOut[q_ax, h_ax, k_ax]
    tl.A[q_ax, m_ax]              = normalize(tl.Attn[q_ax, m_ax] + tl.H[q_ax, m_ax])

    mod = ConstructedModule.construct(tl.to_morphism())
    torch.manual_seed(42)
    W_Q = torch.randn(H, K, D);  H0 = torch.randn(SEQ, D)
    W_K = torch.randn(H, K, D);  W_V = torch.randn(H, K, D)
    W_O = torch.randn(D, H, K)

    out0 = mod(W_Q, H0, W_K, W_V, W_O)
    out0 = out0[0] if isinstance(out0, tuple) else out0

    H1 = H0.clone()
    H1[2:] = torch.randn(2, D)
    out1 = mod(W_Q, H1, W_K, W_V, W_O)
    out1 = out1[0] if isinstance(out1, tuple) else out1

    assert torch.allclose(out0[:2], out1[:2], atol=1e-5), (
        f"Causal invariance violated: max diff "
        f"{(out0[:2] - out1[:2]).abs().max().item():.4f}"
    )
    assert not torch.allclose(out0[2:], out1[2:], atol=1e-5), (
        "Expected output at pos 2,3 to differ when H[2:] changed"
    )
```

- [ ] **Step 2: Run the test**

```bash
python -m pytest tests/test_torch_compile.py::test_attn_res_with_where_is_exactly_causally_invariant -v
```
Expected: PASS.

- [ ] **Step 3: Run the full suite one final time**

```bash
python -m pytest tests/test_tensor_dsl.py tests/test_torch_compile.py -q
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_torch_compile.py
git commit -m "test: causal invariance integration test using norm_axis + where="
```

---

## Task 9: Update `transformer_example.md`

**Background:** The TL DSL code in Section 2 should use the correct `norm_axis + where=` form for the masked softmax. The equations (4) in the math section is already correct. The `causal_softmax` import is replaced with the standard `softmax + where=`.

**Files:**
- Modify: `papers/transformer_example.md`

- [ ] **Step 1: Update the import line in the TL DSL code block**

Find:
```python
from data_structure.TensorDSL import TL, real_axis, causal_softmax, normalize, relu
```
Replace with:
```python
from data_structure.TensorDSL import TL, real_axis, norm_axis, softmax, normalize, relu
```

- [ ] **Step 2: Update `attn_res()` in the TL DSL code block**

Find:
```python
    q = real_axis('q', SEQ)
    x = real_axis('x', SEQ)   # key/value token position
    m = _m(); h = _h(); k = _k()

    # Q/K/V projections (H threaded to all three)
    tl.Query[q, h, k]   = tl.W_Q[h, k, m] * tl.H[q, m]
    tl.Key[x, h, k]     = tl.W_K[h, k, m] * tl.H[x, m]
    tl.Value[x, h, k]   = tl.W_V[h, k, m] * tl.H[x, m]

    # Causal masked softmax — mask applied before exponentiation (contract k)
    tl.S[h, q, x]       = causal_softmax(tl.Query[q, h, k] * tl.Key[x, h, k])
```
Replace with:
```python
    q = real_axis('q', SEQ)
    x = norm_axis('x', SEQ)   # key/value token position; norm_axis marks softmax axis
    m = _m(); h = _h(); k = _k()

    # Q/K/V projections (H threaded to all three)
    tl.Query[q, h, k]   = tl.W_Q[h, k, m] * tl.H[q, m]
    tl.Key[x, h, k]     = tl.W_K[h, k, m] * tl.H[x, m]
    tl.Value[x, h, k]   = tl.W_V[h, k, m] * tl.H[x, m]

    # Causal masked softmax: where= applies -inf mask before exponentiation
    tl.S[h, q, x]       = softmax(tl.Query[q, h, k] * tl.Key[x, h, k], where=(x <= q))
```

- [ ] **Step 3: Verify the transformer tests still pass**

```bash
python -m pytest tests/test_torch_compile.py -k "transformer" -v
```
Expected: all transformer tests pass (they use the old `_mk_attn_res` helper which still uses the old syntax — that is fine; the old syntax continues to work, it just lacks exact causal invariance).

- [ ] **Step 4: Commit**

```bash
git add papers/transformer_example.md
git commit -m "docs: transformer_example.md uses norm_axis + softmax(where=) for causal attention"
```
