---
tags: [layer/backends, tool]
code: torch_compile/
status: evolving
---

# Torch Compile

## What it is

Turns an algebraic expression into a runnable `torch.nn.Module`. The module is the package's
**executable semantics**: whatever a morphism means in the algebra, the module is what it means
numerically.

Requires PyTorch and [`einops`](https://einops.rocks/).

## How it works

`ConstructedModule` is the base. Subclasses register against an operator with
`operation_key=`, so the dispatch is by operator type:

| construct | module |
|---|---|
| `Composed` | `ConstructedComposed` |
| `ProductOfMorphisms` | `ConstructedProduct` |
| `Rearrangement` | `ConstructedRearrangement` |
| `Block` | `ConstructedBlock` |
| `Einops` | `ConstructedEinops` |
| `Linear` | `ConstructedLinear` |
| `Embedding` | `ConstructedEmbedding` |
| `Normalize` | `ConstructedNorm` |
| an elementwise function | `Lambda` |

`generate_einops_signature` turns a [[Operators|signature]] plus its weaves back into an
`einops` string, which is the inverse of the parsing in [[Construction Helpers]].
`torch_utilities.py` has `Weights` and `Multilinear`.

## Broadcasting

`bcast.py` determines **how** a `Broadcasted` is run.

| predicate | meaning |
|---|---|
| `is_semantically_broadcastable` | the reindexing selects, permutes and repeats degree positions and reads none twice — Torch can do it with a permute and a reshape |
| `no_copying` | the mapping does not duplicate |
| `weave_displacement`, `get_displacement` | how far the target sits inside the weave |
| `vmappable`, `find_vmap_loc`, `broadcast_vmap` | otherwise, fall back to `torch.vmap` |
| `broadcast_to_degree`, `unsqueeze_guide` | permute an operand so its dims read the degree in order, then build the `(p, 1, r)` reshape the mapping implies, with `mapping[k]` the degree position dim `k` reads |
| `expand_repeats` | before an einsum, expand a dim of size 1 to the size the same name has in another operand, because a `View` node compiles to a broadcastable view whose repeated axes have size 1 |

`AdditionOp` compiles n-ary, because the reverse of a wire copied three ways is a sum of
three cotangents. The three rows above were corrected on 2026-09-05, when self-attention
from one copied input first reached the validator: `unsqueeze_guide` had indexed the
operand's shape by degree position, which holds only for a prefix selection.

So the same `Broadcasted` runs as a cheap view when its reindexing is simple and as a
`vmap` when it is not.

## Rules and gaps

- `Multilinear.forward` contracts with `tensordot` over the trailing axes. The explicit
  per-axis version was left commented out and **unverified** (`PublicCodeTODOs.md`).
- This layer lowers an expression as it is written. It runs each operation in turn
  through the PyTorch function registered for it, so what it checks is the algebra rather
  than any schedule.

## See also

- [[Operators]] — the registry this is one of
- [[Weaves and Degree]] — what `bcast` is interpreting
- [[Open Gaps]]
