---
tags: [layer/foundations, concept]
code: data_structure/Term.py
status: stable
---

# Terms

## What it is

**Everything in `pyncd` is a `Term`.** A term is an element of the package's formal
language: a frozen dataclass whose fields are its parts. Abstractly it is a construction
rule

$$\gamma_k : \prod_i x_i \rightarrow T_k$$

so that every property of $T_k$ is derivable from the parts $\prod_i x_i$. Nothing is
evaluated at construction time, and evaluation is left to a consumer, such as
[[Torch Compile]] or [[Diagram Display]].

`UTerm` is a `Term` that additionally carries a [[UIDs and Names|UID]], which is an identity
independent of its structure. Axes, block tags and hypergraph nodes are
`UTerm`s. Weaves, arrays and morphisms mostly are not.

## Where it lives

`data_structure/Term.py`. The file is about 470 lines and reads top to bottom. It is the
only file every other module depends on.

| name | what it is |
|---|---|
| `Term` | the base frozen dataclass |
| `UTerm` | a `Term` with a `uid` field |
| `Prod[T]` | `tuple[T, ...]` — the immutable sequence used **everywhere** inside terms |
| `TermDirectory` | qualname to class, populated by `__init_subclass__`; how JSON round-trips |
| `deep_reconstruct` | rebuild a term with a function applied to each part |
| `EqualityClass`, `Context` | the rewriting machinery, covered by [[Rewriting]] |

`keys()`, `dict()` and `reconstruct(**kwargs)` are the reflective interface: they iterate
`__dataclass_fields__`, which is why a `ClassVar` annotation is a trap (below).

## The structure it forms

Terms form an immutable directed acyclic graph with heavy sharing. A subterm reachable by
many paths is one object, referenced many times. Every property of the system that
matters for performance follows from this:

- A naive traversal walks paths rather than nodes. Fusing attention once made 8.3 million `hash`
  calls over a result holding 35k distinct objects.
- Three places are memoised because of it: `Term.__hash__`, `Context.apply`,
  `term_utilities.search`. Write any new traversal so that it visits each node once.
- `deep_reconstruct` returns the original object when nothing changed, which is what
  preserves sharing. A rewrite that rebuilds unconditionally turns the DAG into a tree
  and makes every later pass slower.

## The rules

> [!warning] A `ClassVar` annotation still lands in `__dataclass_fields__`
> which `keys()`, `dict()` and `reconstruct()` all iterate over, so an annotated class flag
> gets passed to `__init__` and blows up. Class-level flags on terms must be
> **unannotated**: `_memoize_hash = True`.

> [!warning] `@dataclass(frozen=True)` leaves `__hash__` alone if the class already has
> one in its `__dict__`
> That is how `Term.__init_subclass__` installs the memoised structural hash on every
> subclass without touching the ~50 declaration sites: `__init_subclass__` runs *before*
> the decorator.

- **Hash caching is sound because terms are frozen.** The cache lives in the instance
  `__dict__`, outside `__dataclass_fields__`, so it takes no part in `__eq__`.
- **Subclasses opt out** of the memoised hash by setting `_memoize_hash = False`
  ([[Numerics]] does, because its equality goes elsewhere).
- **`reconstruct()` keeps the UID and changes the fields**, so UID identity is *not*
  structural identity. [[UIDs and Names]] covers it, and [[Functors]] is the pass that
  relies on it.

Python 3.13+ generics (`def f[T](...)`) and structural `match` are used throughout the
package, and this file is where the style is set.

## See also

- [[UIDs and Names]] — identity, and why names are decoration
- [[Rewriting]] — `EqualityClass` and `Context`, the substitution mechanism
- [[Numerics]] — the one subtree with its own equality story
- [[Invariants]] — the consolidated list of traps
