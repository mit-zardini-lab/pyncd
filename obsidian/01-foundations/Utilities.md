---
tags: [layer/foundations, reference]
code: utilities/utilities.py, term_utilities/term_utilities.py
status: stable
---

# Utilities

Small shared helpers. They are worth knowing because using the wrong one is how
non-determinism and quadratic traversals get in.

## `utilities/utilities.py`

| name | what it does |
|---|---|
| `iallequals(xs, fallback=RAISE)` | assert every element equal and return it. Used constantly to check that, e.g., every reindexing of a `Broadcasted` has the same domain |
| `unique_iterable` / `unique_tuple(xs, unique_test=…)` | order-preserving dedup, optionally keyed |
| `intersection`, `difference`, `concat`, `unique_concat` | order-preserving set ops on iterables |
| `predicate_partition` | split into the entries that match and the entries that do not |
| `deconcatenate` | a flat mapping back into per-segment index pairs |
| `Multidict[K, V]` | one-to-many, and backed by a set. The determinism warning below applies |
| `Prod[T]` | re-export of the tuple alias |

> [!warning] `util.Multidict` is set-backed
> and `UID._id` is random per process, so anything iterating a set of terms must **sort**
> before it produces output or makes a decision. `sorted(uids, key=lambda uid: uid._id)`
> in `para/algebra/tie_tapes.py` is the idiom. See [[UIDs and Names]].

## `term_utilities/term_utilities.py`

Traversal and inspection over terms.

| name | what it does |
|---|---|
| `search(target, predicate)` | find subterms. **Memoised** — it is one of the three DAG-aware traversals ([[Terms]]) |
| `type_search(T, target, predicate)` | the common case: all subterms of a type |
| `search_repeat` | repeated search to a fixpoint |
| `identify_category(target)` | is this in **St**, in **Br**, or unclear |
| `is_mappable`, `is_mappable_broadcast`, `get_mapping` | collapse a reindexing that is really a permutation — a `Composed` or `ProductOfMorphisms` of `Rearrangement`s — into one `Prod[int]`. [[Einops Rearrangement]] depends on this |
| `is_identity` | is this reindexing the identity |
| `flat_axes` | every axis under a term |

`term_utilities/generate_config.py` produces the display configuration consumed by
[[Diagram Display]].

## `utilities/justification.py`

Text layout for [[Diagram Display]]: `justify` with LEFT / RIGHT / CENTER / SPREAD
modes. Imports `display.Color`, so it sits on the display side of the fence.

## Conventions this enforces

- **`fd.Prod[T]` (a tuple alias), never a list**, for anything inside a `Term`.
- **Use `util.Multidict` for a one-to-many mapping**, in place of a dict of lists.
- Write a new traversal over terms so that it visits each node once, or reuse `search`.

## See also

- [[Terms]] — why traversal cost follows the number of paths rather than the number of nodes
- [[Invariants]]
