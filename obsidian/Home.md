---
tags: [moc]
status: stable
---

# The pyncd Vault

Written by Claude Opus 5 (1M context), effort high.

`pyncd` writes a deep learning model as an algebraic expression and then derives things
from it: neural circuit diagrams, PyTorch modules, a backward pass, and the quantisation
every value of a released model is held in.

The vault is the reading order and the reasoning behind the code. The code is the authority
on what happens. These notes are the authority on why it is shaped that way, what rule it
follows, and what is still missing.

> [!important] Before changing anything
> Read [[Vault Conventions]] and [[Invariants]]. An agent starting a substantial piece of
> work reads [[Agent Log Protocol]] first, because a log is owed at the end.

## The one-paragraph version

An expression is a morphism in a product category, per [[Product Categories]]. Its objects
are arrays and its leaves are single broadcast operations, per [[Broadcasted Category]], and
the broadcasting is made explicit by weaves and affine reindexings, per
[[Weaves and Degree]] and [[Stride Category]]. Because a reindexing is affine, the
dependency between two axes is integer linear algebra rather than symbolic execution, so
which positions of one array a position of another reads is answered by composing matrices.
An expression is rewritten in the hypergraph form of [[Hypergraphs]], where a pass replaces
one operation by several, per [[Leaf Splicing]], or rewrites every wire at once, per
[[Functors]]. Differentiating a model is one such rewrite, in the category of
[[Para Category]], and so is writing the quantisation of every value onto its wire, per
[[Quantization]].

## Start here

| the task | the note |
|---|---|
| the layout of the repository | [[Repository Map]] |
| building an expression | [[Operators]], then [[Construction Helpers]] |
| writing a model the way this package writes one | [[Representing Models]] |
| reading an expression back | [[Agent Display]] |
| drawing an expression | [[Diagram Display]] |
| differentiating a model | [[Training]], then [[Backpropagation]] |
| quantising a model | [[Quantization]] |
| following one expression from form to form | [[Forms of an Expression]] |
| finding what is unfinished | [[Open Gaps]] |
| finding what the package does not represent | [[Design Space]] |
| finding the facts that cost an experiment | [[Invariants]] |
| checking that nothing broke | [[Validation]] |

## The layers

```
01 foundations   Term, UID, Numeric, rewriting. Depends on nothing else.
02 categories    St (axes and strides), Br (arrays and weaves), operators, builders,
                 and the rewrites any feature may use.
03 hypergraphs   The rewriting form, both directions, analysis, functors, splicing.
04 quantization  The number format a value is held in, and the pass that writes one
                 onto every wire.
05 backends      agent_display, display, torch_compile, websocket_transfer.
06 practice      How to work here: the map, the invariants, validation, the gaps.
07 para          Training as a second morphism: the tape and the reverse functor.
```

Each folder's notes link downward to the code and sideways to each other. Nothing in layers
01 and 02 refers to a backend, and nothing in `display` imports anything above it, per
[[Invariants]].

## Maps of content

- **Foundations** — [[Terms]], [[UIDs and Names]], [[Code Forms]], [[Numerics]], [[Units of Measure]], [[Rewriting]], [[Utilities]]
- **Categories** — [[Product Categories]], [[Stride Category]], [[Broadcasted Category]], [[Weaves and Degree]], [[Operators]], [[Sparse Axes]], [[Sparse Expansion]], [[The Universal Unit]], [[Padding and Masks as Sparse Axes]], [[Advanced Axis Dynamics]], [[Expression Simplification]], [[Einops Rearrangement]], [[Linear Expansion]], [[Discovering Broadcasts]], [[Construction Helpers]]
- **Hypergraphs** — [[Hypergraphs]], [[Hypergraph to Morphism]], [[Hypergraph Analysis]], [[Functors]], [[Crawlers]], [[Leaf Splicing]]
- **Quantization** — [[Quantization]], [[Stripping Quantisations]]
- **Backends** — [[Agent Display]], [[Diagram Display]], [[Advanced Display]], [[Diagram Themes]], [[Compound Axis Labels]], [[Diagram Wire Format]], [[Terms Mirrored in tsncd]], [[Torch Compile]]
- **Practice** — [[Repository Map]], [[Invariants]], [[Validation]], [[Notebooks]], [[Representing Models]], [[SOTA Model Notebooks]], [[Yoneda and Cartesian Tricks]], [[Debugging the Jupyter Restart Button]], [[Open Gaps]], [[Design Space]], [[Code Style]], [[Forms of an Expression]]
- **Para** — [[Para Category]], [[Training]], [[Derivatives]], [[Backpropagation]], [[Pathway Collapse|pathway collapse]], [[Recomputing the Exponent in the Backward Pass]], [[Selection and the Reverse Pass]], [[Para Wrap]], [[Para Block Operator]], [[Show Grabbed Parameters]], [[Training Mixture of Experts Gates]], [[DeepSeek-V3 Backward Pass]]
- **Meta** — [[Vault Conventions]], [[Agent Log Protocol]], [[Agent Log Index]]
