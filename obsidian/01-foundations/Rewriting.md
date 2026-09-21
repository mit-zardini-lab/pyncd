---
tags: [layer/foundations, concept]
code: data_structure/Term.py
status: stable
---

# Rewriting

## What it is

Every structural change in `pyncd`, whether it aligns axes or splices an operation into
a graph, is done by rewriting identity:
declaring that a set of UIDs all denote one canonical term, then applying that
declaration across a structure.

Three objects do it:

- **`EqualityClass`** — a bucket of UIDs, a canonical term, and a priority. Applying it
  to a term replaces any `UTerm` whose UID is in the bucket with the canonical one.
- **`UIDRenaming`** — a bucket of UIDs and a canonical UID. Applying it keeps a matching
  term's fields and replaces its uid alone. A wire splice is the renaming case, because
  each occurrence of a `HypergraphObject` must keep the object it carries.
- **`Context`** — a list of equality classes and renamings, applied together in one
  traversal.

## Where it lives

`data_structure/Term.py`, bottom half.

| constructor | use |
|---|---|
| `EqualityClass.from_iter(terms)` / `.template(*terms)` | "these are all the same"; the canonical is the `max` by UID |
| `EqualityClass.set_canonical(new, *old)` | "the old terms become this specific new one" |
| `UIDRenaming.set_canonical(new, *old)` | "the old identities become this one, fields untouched" — the form used for every wire splice |
| `Context([...]).apply(target)` | run the rewrite |
| `Context.append_bucket` / `append_buckets` / `append_contexts` | accumulate classes, merging overlaps |

`Context.__call__` is `apply`, so a context is usable as a function.

## The mathematics

An `EqualityClass` is a quotient: it identifies a set of degrees of freedom and picks a
representative. A `UIDRenaming` is the same quotient on identities alone, leaving each
term's fields where they are. A `Context` is a *set* of such quotients, which must be
closed under overlap, which is what `try_merge` is for. A renaming merges with a renaming
and a class with a class, and the two kinds never share a bucket, because a UID's equality
includes its `_type`. The canonical is chosen by `(priority, canonical.uid)`, so a caller
can force a particular representative to win by raising its priority.

Applied to a whole expression, this is exactly substitution in the sense the categorical
presentation needs: a morphism is a term, so substituting an axis inside it produces the
morphism over the substituted axis.

## The rules

> [!warning] Merging must accumulate
> A bucket can overlap several existing classes, when a chain of identifications arrives out
> of order, and each merge has to fold into the union so far. Merging each candidate
> against the *original* bucket keeps only the last union and silently drops the rest.
> `append_bucket` does it correctly, and it is commented at the site because it was a bug.

> [!warning] `Context.apply` is memoised on object identity, per top-level call
> Terms form a directed acyclic graph, per [[Terms]]. Without the memo a shared subterm is rebuilt once per
> path, and the result comes back with less sharing than the input, so every
> later pass has more paths to walk. Keying on `id()` is safe because everything keyed on
> stays reachable from `target` for the whole traversal.

> [!warning] When splicing into a hypergraph, rewrite the terminal alone and leave the region
> Rewriting a whole region wires consumers to the pre-splice value and bypasses the node
> just inserted. [[Leaf Splicing]] states the rule and `graphs/processing/leaf_splicing.py`
> is the code.
>
> ```python
> fd.Context([fd.UIDRenaming.set_canonical(new, old)]).apply(subgraph)
> ```

## Where it is used

- [[Construction Helpers]] — `@` aligns the axes of two morphisms by identifying them.
- [[Hypergraphs]] — every splice is `UIDRenaming.set_canonical(new, old)` on a
  `HypergraphObject`. A wire is the object's uid, so renaming it *is* rewiring.

## See also

- [[Terms]], [[UIDs and Names]]
- [[Hypergraphs]] — the structure most rewrites happen in
- [[Functors]] — the other way to transform a structure, when the change is uniform
