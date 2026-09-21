---
tags: [layer/foundations, concept]
code: data_structure/Term.py
status: stable
---

# UIDs and Names

## What it is

A `UID` is a term's **identity**, independent of its structure. Two `UTerm`s with the
same `uid` are "the same in every way" as far as the package is concerned. Each `UTerm`
with a UID acts as a **degree of freedom** in an expression: substituting one UTerm for
another is how axes are aligned, how blocks are tracked across a rewrite, and how a wire
spliced into a graph takes the identity of the wire it replaces.

A `DynamicName` is the *decoration*: a body, an optional subscript (itself a
`DynamicName`), and aesthetic settings (bold, overline, absolute bars). Names exist so
diagrams and listings read well. They carry no identity.

## Where it lives

`data_structure/Term.py`:

| name | what it is |
|---|---|
| `UID[T]` | `_type`, `_id` (random per process), `_name` |
| `fresh_id()` | a new random id in `[1, 2**31-1]` |
| `hash_id(obj)` | an id *derived* from a value, as a digest of its `repr`, so it is stable across calls and across processes |
| `Term.__getstate__` | the state pickle records: every field and not the cached hash, which is wrong in another process |
| `DynamicName` | body / subscript / settings / code form / exponent, with `to_bodies()`, `to_text()`, `to_latex()`, `to_code_form()`, `add_subscript()`, `capture()`. The code form and the exponent are optional and are stated in [[Code Forms]] |
| `DynamicNameSettings` | bold, overline, absolute, typewriter |

## The rules

> [!warning] An axis is identified by its UID, and its name exists for display
> `cat.RawAxis.named('q')` called twice gives two different axes, with different UIDs,
> and even different size symbols, which merely both print as `|q|`. Never treat equal
> names as the same axis. Composition lines axes up **positionally**, through
> [[Construction Helpers]], rather than by name.

> [!warning] The one exception: `nm.FreeNumeric.named(x)`
> derives its id from the name via `fd.hash_id`, and so *is* stable across calls. The two
> `named` constructors deliberately differ. The difference is easy to miss, and it is
> deliberately relied on, because a symbol named twice has to be one term rather than two
> that merely print alike.

> [!warning] A term crosses a process boundary without its hash, and a named symbol keeps its id
> Python salts the hash of a string differently in every process, so the structural hash
> a term caches is wrong once the term is unpickled elsewhere. `Term.__getstate__` leaves
> the cached hash out, and the other process computes its own. `hash_id` once took
> `hash(obj)`, so `FreeNumeric.named('MatrixRate')` was a different symbol in every
> process. Since 2026-09-09 it digests
> `repr(obj)` instead, so a worker process that names a symbol names the symbol its
> parent process named.

> [!warning] `reconstruct()` keeps the UID and changes the fields
> So a UID does **not** imply structural equality: two terms with the same UID can
> differ. A functor rewrites the fields of a term and leaves its uid alone, which is how
> a pass keeps the wiring while it changes what each wire carries. Do not change hashing
> or equality to compare UIDs alone. `HypergraphObject` is the one deliberate exception, because its
> uid is a wire, per [[Hypergraphs]].

## Determinism

`UID._id` is **random per process**. Combined with the set-backed `util.Multidict`, this
means anything that iterates a set of terms has to sort: by `uid._id` for stability
within a run, and by name where the ordering affects the work as well as the output.

Both failure modes have happened:

- Unsorted iteration reordered results between runs. `sorted(uids, key=lambda uid: uid._id)`
  in `para/algebra/tie_tapes.py` is the fix, and `deepseek/sparse_expansion.py` sorts its
  wires the same way.
- Sorting by the random `uid._id` where the order decides which rewrite runs first made a
  derivation time swing between identical runs. Sort on a name where the order affects the
  work rather than the output alone.

## Naming conventions in practice

- Display a name with `axis.uid._name.to_bodies()`, which may return `None`.
- Subscripts chain, so `DynamicName` renders `{x}_{{y}{z}}` rather than `x_{y_z}`.

## See also

- [[Terms]] — what a UTerm is
- [[Rewriting]] — how substituting one UID for another actually happens
- [[Invariants]]
