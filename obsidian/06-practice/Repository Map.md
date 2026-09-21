---
tags: [layer/practice, reference]
status: stable
---

# Repository Map

Written by Claude Opus 5 (1M context), effort high.

Repository root: the parent of this vault. Every path below is relative to it.

## Live packages

| path | what it is | note |
|---|---|---|
| `data_structure/` | `Term`, **St**, **Br**. Knows nothing about graphs | [[Terms]], [[Stride Category]], [[Broadcasted Category]] |
| `construction_helpers/` | the `@ * >>` overloads, signature parsing, lifting | [[Construction Helpers]] |
| `algebra/` | rewrites belonging to no feature in particular: `node_expansion` factors a `Broadcasted`'s reindexings out of its operator, `reindexing_absorption` and `einops_simplification` put an expanded expression back together, `merge_into_consumer` is the search they share, `einops_rearrange` merges one contraction into the next and disentangles the result, `linear_expansion` splits a `Linear` into a weight array and a contraction, `operator_expansion` writes a softmax and a normalisation out in primitives, and `discovering_broadcasts` writes a repeated body as one block broadcast over a degree and confirms it by expanding it again | [[Derivatives]], [[Einops Rearrangement]], [[Linear Expansion]], [[Expression Simplification]], [[Discovering Broadcasts]] |
| `graphs/data_structure/` | the hypergraph itself | [[Hypergraphs]] |
| `graphs/processing/` | both directions of conversion, analysis, the generic `Crawler` and `Functor`, and `leaf_splicing`, which walks every leaf, replaces one in place and rebuilds the containers around it | [[Hypergraphs]], [[Crawlers]], [[Functors]], [[Leaf Splicing]] |
| `quantization/` | the number format and the size in bits a value is held in, the conversion operator between two of them, and the pass that writes a quantisation onto every wire of a model | [[Quantization]], [[Stripping Quantisations]] |
| `agent_display/` | the SSA listing. **Read this one** | [[Agent Display]] |
| `display/` | 2D ASCII neural circuit diagrams | [[Diagram Display]] |
| `data_transfer/`, `websocket_transfer/` | JSON encoding and the `tsncd` bridge | [[Diagram Display]], [[Diagram Wire Format]] |
| `torch_compile/` | expression to `nn.Module` | [[Torch Compile]] |
| `term_utilities/`, `utilities/` | traversal, dedup, layout | [[Utilities]] |
| `solver/` | numeric solving and the derivative of a numeric | [[Numerics]] |
| `deepseek/` | DeepSeek-specific data structures, and the rewrite that writes a selection out | [[Sparse Axes]], [[Sparse Expansion]] |
| `advanced_axis_dynamics/` | the axis a read outside an axis leaves, the affine form that says which of its positions hold a value, and the concatenation of two axes carrying two forms. `data_structure/` holds the axis and the two covariant operators, `algebra/` derives a form and restores it, and `registries/` holds the folds and the reverse derivatives. Nothing below `deepseek/` imports it | [[Advanced Axis Dynamics]], [[Padding and Masks as Sparse Axes]] |
| `para/` | the tape, and `forward_backward`: one morphism to a forward pass and a backward one. Split into `data_structure/`, `algebra/`, `registries/` and `processing/`. `data_structure/ParaBlockOperator.py` boxes a body that reads and writes a tape and lists its seeds on the operator, and `registries/object_lift.py` is how a seed is broadcast | [[Training]], [[Derivatives]], [[Backpropagation]], [[Para Block Operator]] |
| `notebooks/` | one folder per feature, plus the shared display helpers | [[Notebooks]] |
| `example_notebooks/` | the short notebooks a new reader starts with | [[Notebooks]] |
| `validations/` | the registry of every validation in the repository, an import graph over its source files, the selection of the targets a modified file reaches, and the runner that executes them at once as subprocesses | [[Validation]] |
| `_guide/` | the figures and the poster | |

## Dead or misleading

| path | why |
|---|---|
| `data_structure/Reversed.py` | a single `# TODO` |
| `construction_helpers/factorize.py` | entirely commented out |

## Long-form notes

The repository keeps no long design notes beside the code. They are all in the vault, so
there is one place to look for the mathematics. [[Diagram Wire Format]] is the longest of
them, and it is mirrored at `tsncd/PROTOCOL.md` in the TypeScript repository.

Still at the repository root, because they are about the repository rather than the
mathematics:

| file | what it is |
|---|---|
| `README.md` | what the package is, and how to run it |
| `CLAUDE.md` | how to work here |
| `AGENTS.md` | the instruction an agent reads first |
| `StyleGuide.md` | where the writing rules are |
| `PublicCodeTODOs.md` | `TODO` comments lifted out of the source before release |
| `validate_repository.py` | imports, validators and vault links, in one command. Its `validators` check is the runner of `validations/` applied to the discovered `validate_*.py` scripts |
| `run_server.py` | the relay a browser diagram goes through |

## Where to start reading code

Read from the bottom up. Four files cover most of the package:

1. `data_structure/Term.py`
2. `data_structure/StrideCategory.py`, which is the axis and the affine map alone since
   2026-09-15
3. `data_structure/BroadcastedCategory.py`
4. `graphs/data_structure/Hypergraph.py`

Then `algebra/__init__.py`, which lists its modules and what each holds.

## See also

- [[Home]] — the layer diagram
- [[Notebooks]] — which notebooks run, and what each shows
- [[Invariants]] — the facts that cost an experiment to establish
