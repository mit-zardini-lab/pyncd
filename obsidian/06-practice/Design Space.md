---
tags: [layer/practice, index]
status: evolving
---

# Design Space

Written by Claude Fable 5.1, effort 80. Rewritten by Claude Opus 5 (1M context), effort
high.

## What it is

The design space is the class of operations the package's rewrites are written for. This
note lists what lies outside it. The user asked for the list on 2026-09-13, so that a model
needing an operation outside the space is recorded and the rules are not assumed to apply
to it. An operation outside the design space can still be written as an expression, because
an operator of [[Broadcasted Category]] is any named map, and the expression can be drawn
and differentiated. The rules for dividing a fold and for combining its partial results do
not apply to it. A model that needs such an operation is entered in the table at the end of
this note with the form it needs.

## The operations inside

Every operation is a morphism of [[Broadcasted Category]], meaning an operator
broadcast over a degree with affine reindexings.

Every fold over an axis has an accumulator, which is the operator combining two partial
results of that fold into one. `algebra/registries/accumulator.py` registers them, and two
are registered: the contraction with `+`, whose starting value is `0`, and the maximum with
`\max`, whose starting value is `-\infty`. An operator with no row reads every position of
the axis to write every position and has no partial result at all, which is why a
`ops.SoftMax`, a `ops.L1Norm`, a `ops.L2Norm` and a `ops.Normalize` have none.

An accumulator is pointwise over the partial result, and its starting value is one value
broadcast over the whole of it, written with `ops.ConstantOp` and `backup_degree`. The
universal unit of [[The Universal Unit]] is that one value.

A data-dependent selection with static bounds, meaning a choice of `k` of `L` positions
made at runtime, is inside. It is written on a `SparseAxis` or as index wires, per
[[Sparse Axes]] and [[Sparse Expansion]]. [[Open Gaps]] item 1 records what the reading
still lacks.

## The operations outside

| outside | what it is | why the rules do not apply | where a model needs it |
|---|---|---|---|
| an if statement | a computation whose operations depend on a runtime value | every rewrite reads the operations off the expression before any value exists. A runtime value may choose positions, which is a reindexing and is inside, and may not choose operations. No operator in `data_structure/Operators.py` is a conditional | no model in the repository is written with one |
| a non-associative operation over an axis | an operation whose result over a run of positions cannot be assembled from its results over the parts of that run | the parts have to arrive in order and no partial result can be computed elsewhere, so nothing can be registered for it in the accumulator registry | none in the repository. A recurrence whose step is non-linear, such as a gated RNN cell, is the example |
| an accumulator that does not commute | an accumulator that exists and gives a different answer for each order of its two operands | a fold run in pieces combines the pieces in an order nothing fixes, so a result that depends on the order is not the result of the fold | the inter-chunk recurrence of GDN and lightning attention, a matrix-valued scan, per [[Open Gaps]] |
| an accumulator that is not pointwise | an accumulator that reads several positions of a partial result at once | its starting value is a structured array, such as the identity matrix of the chunk scan, where the registry writes one value broadcast over the whole | the same chunk recurrence |

## Recording a model outside the design space

When a model needs an operation in the table, add its name to the last column with a
link to the model's note or notebook, and say which form it needs. When a model needs
an operation the table does not list, add a row. The expression is still built in Br,
and the note or notebook says which of its operations the rules do not cover.

## See also

- [[Open Gaps]] — the ranked worklist, which the last two rows of the table feed
- [[The Universal Unit]] — the one value a fold's starting value is
- [[Sparse Axes]] and [[Sparse Expansion]] — the data-dependent selection that is inside
