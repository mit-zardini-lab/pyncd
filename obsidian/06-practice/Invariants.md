---
tags: [layer/practice, reference]
status: stable
---

# Invariants

**Each of the facts below cost an experiment to establish.** They are collected here, and
each also lives in the note for the thing it constrains.

## Terms and identity

**Terms form an immutable directed acyclic graph with heavy sharing.**
A naive traversal walks paths rather than nodes, so a subterm reachable fifty ways is visited
fifty times. Fusing attention once made 8.3 million calls to `hash` over a result holding 35
thousand distinct objects. Three places account for the total and all three are memoised:
`Term.__hash__`, `Context.apply` and `term_utilities.search`. Write any new traversal so that
it visits each node once. → [[Terms]]

**`deep_reconstruct` returns the original when nothing changed.**
Returning the original preserves the sharing. A rewrite that rebuilds unconditionally
produces a tree from a directed acyclic graph and makes every later pass slower. → [[Terms]]

**`reconstruct()` keeps the UID and changes the fields.**
UID identity is therefore not structural identity, and two terms with the same UID can
differ. A functor pass relies on exactly that: it rewrites what each wire carries and
leaves the wiring alone. Do not change hashing or equality to compare UIDs alone.
`HypergraphObject` is the one deliberate exception, because its uid is a wire. →
[[UIDs and Names]], [[Hypergraphs]]

**A repeat is a `View` and never an operator.**
A degree axis that no operand's reindexing names is one the output is repeated along. The
dual of a sum is therefore a `View` that drops the axis, `einsum` has no produced case,
`Einops.template('q -> q v')` is legal, and nothing may introduce a copying operator, because
`reindexing_absorption.absorb` would not know to fold it. `para` carried one, called
`Broadcast`, with a rule of its own, for a day before the reading was seen. →
[[Expression Simplification]], [[Weaves and Degree]]

**An axis is identified by its UID, and its name exists for display.**
Calling `cat.RawAxis.named('q')` twice gives two different axes, with different UIDs and
different size symbols that both print as `|q|`. Never treat two equal names as the same axis.
Composition lines axes up by position. The one exception is `nm.FreeNumeric.named(x)`, which
derives its id from the name through `fd.hash_id` and is stable across calls and across
processes. The two `named` constructors differ deliberately. → [[UIDs and Names]]

**A `ClassVar` annotation still appears in `__dataclass_fields__`**, which `Term.keys()`,
`dict()` and `reconstruct()` all iterate over, so an annotated class flag is passed to
`__init__` and raises. Declare a class-level flag on a `Term` without an annotation, as
`_memoize_hash = True`. → [[Terms]]

**`@dataclass(frozen=True)` leaves `__hash__` alone when the class already defines one** in
its `__dict__`. `Term.__init_subclass__` uses that to install the memoised hash on every
subclass without editing the fifty or so declaration sites, because `__init_subclass__` runs
before the decorator. → [[Terms]]

## Numerics

**Numerics compare structurally, and `template` canonicalises.**
Flattening, dropping units, folding integer constants and sorting commutative operands is
normalisation. Anything deeper, such as collecting like terms or cancelling `x/x`, is genuine
algebra, so it goes in the numeric-algebra section at the foot of `Numeric.py` and is applied
explicitly. Do not put it in `==`. Equality once went through a hash modulo `2**16-1`, which
made `2x + 2x == 4x` true and also made `Integer(65535) == Integer(0)` true. → [[Numerics]]

## Weaves

**Every weave of a `Broadcasted` has exactly `len(degree())` TILED positions.**
`imprint_to_degree` consumes the degree lazily and stops when the slots run out, so a weave
one slot short still builds a plausible array out of a prefix of the degree. The object is
right and the alignment is wrong. Everything that pairs degree positions with weave positions
then indexes past the end, and in [[Diagram Display|tsncd]] that surfaces as
`Cannot read properties of undefined (reading 'anchors')` out of `link_weaves`. An axis in
the degree is TILED, even where the same axis is an operator target elsewhere. →
[[Weaves and Degree]]

**`Broadcasted.backup_degree` is a `ProdObject` on a morphism whose domain is empty and
`None` on every other morphism.** `has_empty_domain()` and `backup_degree is not None`
therefore report the same condition, and `degree()` reads `backup_degree` whenever it is
set. A morphism with an empty domain that is not broadcast carries the empty `ProdObject`
rather than `None`, which `__post_init__` fills in. Giving a morphism with inputs a
`backup_degree` raises there, so a rewrite that adds an input to a morphism with an empty
domain has to clear the field. →
[[Broadcasted Category]]

**A `Broadcasted` is manipulated by position.** An axis identity is not a reliable
indicator of structure. It records that two positions have the same size and that
composition carries an index from one to the other along a wire, and it does not record
that an operation reads two positions with one index variable. A weave is a sequence of positions, a
reindexing maps output degree positions to input degree positions, and an `Einops`
signature lists a group per non-tiled position, so those three state the morphism and the
axis at a position supplies its size and its name. Deduplicating a shape, testing an axis
for membership in a shape, keying a dictionary by an axis or calling `shape.index(axis)`
identifies an axis with an index variable, and reads one token axis at two positions of
the scores as a diagonal. `einops_simplification.einsum` takes shapes of `IndexVariable`s, which `index_shapes` reads
off a morphism, and the derivative rules, `operator_expansion`, `pathway_collapse` and
`merge_einops` write their shapes in them. A target position of an operator that states no
grouping is related to the other side's target by its axis, which [[Open Gaps]] records.
→ [[Expression Simplification]], [[Open Gaps]]

## Hypergraphs

**A wire is a `HypergraphObject`, and its uid is its identity.**
Equality and hashing read the uid alone, so two objects with the same uid are the same wire
even when their `.obj` differs, which happens constantly, because a functor pass rewrites
`.obj` while the wiring stands still. Every splice works by renaming wire identity with
`fd.UIDRenaming.set_canonical(new, old)` inside an `fd.Context`, which keeps each
occurrence's `.obj` where an `EqualityClass` would overwrite it. → [[Hypergraphs]]

**A guide crosses into a block by wire rather than by position.**
`HypergraphBlock.template` deduplicates the block's `dom` and the body's `dom` is not
deduplicated, so a positional zip of anything per-wire across the two is shifted by one after
a duplicated wire. `Crawler.propagate_graph` realigns with `realign_guide`, and anything else
that pairs a block's domain or codomain with its body's has to key by wire. → [[Crawlers]]

**A block carries meaning as well as appearance.**
A `BlockTag` with `repetition != 1` denotes a loop, which carries a running value from one
iteration to the next. An `aesthetics` field says how the block is drawn and says nothing
about the semantics of its body. → [[Product Categories]]

**A loop block bounds the merging of duplicate roots.**
`merge_duplicate_roots` merges two roots that wrap equal morphisms on the same wires, and
keys the grouping on the loop blocks enclosing each root. A root inside a loop computes once
per iteration, so it is never merged with a root outside the loop, and two roots inside one
loop merge within its body. A block with `repetition == 1` is transparent, and the kept root
is placed in the innermost block enclosing every reader, with the block domains recomputed
so its wire reaches them. → [[Functors]]

**A root outside a block may not read the block's output and feed its input.**
The block and the root then read each other, no order of the scope carries the pair, and
`hypergraph_to_morphism` drops the cycle rather than raising. A rewrite that hoists a root
out of a block checks that every wire the root reads is produced outside the block first.
`pathway_collapse` and `merge_duplicate_roots` both do. → [[Functors]]

**`flat_subgraphs(g, remove_blocks=True)` does not remove every block.**
It descends through a block only when `repetition == 1`. A loop block comes back as a
`HypergraphBlock` with no `.wraps`, so a walk that assumes it has reached a leaf will raise.
Recurse into `.body` directly. → [[Hypergraphs]]

**`HypergraphAnalysis` answers over one level of direct children.**
Its indices are declared `cached_property`, and a fresh instance is constructed for each
query, so nothing is shared between instances. For a deep walk, use `flat_subgraphs` with a
producer and consumer lookup by wire. → [[Hypergraph Analysis]]

**When splicing a node into a graph, rewrite the terminal alone and leave its consumers
alone.** Rewriting the whole region wires the consumers to the value from before the splice
and bypasses the node just inserted. → [[Rewriting]], [[Leaf Splicing]]

**`ValueError: HypergraphObject(...) is not in list`** from `hypergraph_to_morphism` means
the wiring is wrong and the conversion is correct. Some wire is required on the right of a
branch and nothing on the left produces it. → [[Hypergraph to Morphism]]

**Merging equality classes has to accumulate.** A bucket can overlap several existing classes,
and each merge has to fold into the union so far. → [[Rewriting]]

## Splicing

**The reverse crawler hands upstream the guide it reads off the rebuilt morphism.**
`BuiltReverseCrawler.root_processor` rebuilds the morphism and then reads a fresh guide off
its domain, rather than passing on the guide that arrived from downstream. →
[[Crawlers]]

**Call `rescope` after any splice that changes a container's domain or codomain.** A block
that gains an operation reading a wire from outside has gained that wire on its domain, and
nothing else works that out. → [[Leaf Splicing]]

## Determinism

**`util.Multidict` is backed by a set and `UID._id` is random per process**, so any code that
iterates over a set of terms must sort by `uid._id` first, as `para/algebra/tie_tapes.py`
and `deepseek/sparse_expansion.py` do. Without the sort, results reorder between runs. →
[[Utilities]]

**Sorting by `uid._id` is not always enough.** Where the order affects the work rather than
the output, sort by a name. A search whose order of candidates came from the random ids
took between 30 s and 290 s across identical runs until it did.

**A term is pickled without its cached hash, and a named symbol has the same id in every
process.** Python salts the hash of a string per process, so a cached structural hash is
wrong once a term is unpickled elsewhere, and `Term.__getstate__` leaves it out.
`fd.hash_id` digests `repr(obj)` rather than taking `hash(obj)`, so `FreeNumeric.named`
gives the same term in a worker process as in the parent. Before the change a worker's
`MatrixRate` differed from the parent's. → [[UIDs and Names]]

## What must not import what

**`display` imports nothing above it.** Every layer above the backends may import
`display`, so an import in the other direction forms a cycle. The rule holds for
`agent_display`, `torch_compile`, `data_transfer` and `websocket_transfer` as well. →
[[Diagram Display]]

**Write `import construction_helpers as ch` for its side effects**, even when the name is
unused, because the `@`, `*` and `>>` overloads do not exist without it. →
[[Construction Helpers]]

## Ordering constraints

**[[Linear Expansion]] runs after [[Einops Rearrangement]]**, or the normaliser folds a
projection into the score and produces a three-operand contraction, from which the two
contractions can no longer be told apart.

**A quantisation is written onto every wire before a cast is inserted**, because a cast is
written where two neighbouring wires disagree and nothing disagrees until both carry a
quantisation. → [[Quantization]]

## Conventions

- Use `fd.Prod[T]`, which is an alias for a tuple, for anything held inside a `Term`. Never a
  list.
- Use `util.Multidict` for a one-to-many mapping, and `util.unique_tuple`, `util.iallequals`
  and `util.concat` for the operations that go with it.
- Read a name for display with `axis.uid._name.to_bodies()`, which may return `None`.
- Python 3.13 generics, written `def f[T](...)`, and structural `match` are used throughout,
  so the package requires Python 3.13 or newer. `environment_versions.txt` records the
  versions it is developed against.

## See also

- [[Validation]] — how to check that a change preserved all of the above
- [[Vault Conventions]] — where a newly discovered rule should be written down
