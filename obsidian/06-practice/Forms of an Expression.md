---
tags: [layer/practice, moc]
status: evolving
---

# Forms of an Expression

Written by Claude Opus 5 (1M context), effort high.

## What it is

One expression passes through several forms on its way to a diagram, a training pair, a
quantised model or a cached pass, and each pass reads one form and writes the next. The notes listed below
each carry a Mermaid graph whose boxes are the forms and whose edges are the passes. This
note indexes those graphs and draws how the pipelines feed one another.

## The pipelines

```mermaid
flowchart TD
    M["Morphism in Br"]
    M -->|"Multigraph.from_morphism and hypergraph_to_morphism"| G["Multigraph<br>the form every rewrite runs on"]
    M -->|"expand_to_nodes, ExpandLinear, expand_softmax, expand_sparse_onto_tape or grab_parameters"| X["Expanded morphisms"]
    X -->|"absorb and merge_einops"| M
    M -->|"quantise_model"| Q["Quantised model<br>a quantisation on every wire and a cast where two disagree"]
    Q -->|"strip_quantisations"| M
    M -->|"grab_parameters and forward_backward"| T["Taped forward and backward passes"]
    T -->|"dedup_and_collapse"| TC["Collapsed Taped"]
    TC -->|"recompute_elementwise_slots, or recompute_contraction_slots then store_operands_of_views"| TR["Taped with recomputed slots"]
    M -->|"slide_causal_reads_backwards"| S["CausalSlide<br>each causal read at the copy it masks"]
    M -->|"derive_cached_pass"| C["Cached pass<br>the model over the new tokens, with a Caching wherever a causal read reaches an earlier token"]
    C -->|"with_linear_maps_absorbed"| CA["Cached pass with the maps over its caches absorbed into the queries"]
```

| pipeline | the note holding its graph |
|---|---|
| the expansions, and the form each writes out | *The expansions*, below |
| a morphism and its hypergraph | [[Hypergraphs]] |
| the expansions of an operator and the rewrites that undo them | [[Expression Simplification]] |
| the three forms of a selection | [[Sparse Axes]] |
| the forms of a padded or masked axis | [[Padding and Masks as Sparse Axes]] |
| the three forms of a weight | [[Linear Expansion]] |
| a quantised model, and the model under it | [[Quantization]], [[Stripping Quantisations]] |
| training, from a morphism to a checked pair | [[Training]] |
| the collapse of a backward pass | [[Pathway Collapse]] |
| the CausalSlide, a cached pass, its placements and the absorbed order | [[Deriving Caches by Dragging the New Tokens]] |

## The expansions

Every expansion writes out something an operator or a wire held implicitly. The implicit
form is the one a model is written in and can be worked in. The full form is the complete
statement, and where it is in Para the tape is the only coupling between its pieces. A
figure shows the implicit form or, for a form in Para, the wrapped form. The tied form is
reached on request and is not worked in.

```mermaid
flowchart LR
    I["Implicit morphism in Br<br>weights inside operators, one wire per selection, one glyph per softmax"]
    I -->|"expand_to_nodes"| N["Reindexings factored out<br>a View per operand and a core with the degree identity"]
    I -->|"ExpandLinear and expand_parametrised_linears"| L["Weight as an array<br>a 0-input Linear contracted by an Einops"]
    I -->|"expand_softmaxes, expand_shifted_softmaxes and expand_normalizes"| P["Operators in their primitives<br>Arithmetic maps, copies, contractions and products"]
    I -->|"grab_parameters"| G["Full form, in Para<br>each parameter an operand grabbed from a slot named after it"]
    I -->|"expand_sparse_onto_tape, the default for a selection"| T["Full form, in Para<br>each index dropped by its TopK and grabbed by each consumer"]
    I -->|"expand_sparse, depending on nothing in para"| X["Tied form<br>the index on a wire routed through every block it crosses"]
    T -->|"tie_tapes, on request"| X
    G -->|"to_para_wrap, for display"| W["ParaWrap form<br>tapes drawn on the operations"]
    T -->|"to_para_wrap, for display"| W
    I -->|"BlockOperator.expand"| O["Box opened<br>the block's body lifted over the box's degree"]
    I -->|"expand_concatenations"| J["Concatenation written out<br>each consumer of the concatenated positions rewritten into the consumers of the parts"]
    I -->|"load_the_past_and_append_this_pass"| K["Cache written out, in Para<br>a CacheGrab of the earlier tokens, a concatenation and a CacheDrop of the new ones"]
```

| expansion | what it writes out | module |
|---|---|---|
| reindexings out of a `Broadcasted` | a `View` per operand and a core | `algebra/node_expansion.py`, per [[Expression Simplification]] |
| a `Linear`'s weight | a 0-input array contracted by an `Einops` | `algebra/linear_expansion.py`, per [[Linear Expansion]] |
| a `SoftMax` or a `Normalize` | its primitives | `algebra/operator_expansion.py`, per [[Expression Simplification]] |
| the parameters an operator reads | a grabbed operand per parameter | `para/processing/show_grabbed_parameters.py`, per [[Show Grabbed Parameters]] |
| a selection, onto the tape | a dropped and grabbed index per selection | `para/algebra/para_sparse_expansion.py`, per [[Sparse Axes]] |
| a selection, onto a wire | the same index as a wire | `deepseek/sparse_expansion.py`, per [[Sparse Expansion]] |
| a box | the block's body over the box's degree | `data_structure/Operators.py` |
| a concatenation of two axes | the consumers of the parts | `advanced_axis_dynamics/algebra/concatenation_expansion.py`, per [[Advanced Axis Dynamics]] |
| a `Caching` | the load of the earlier tokens, the concatenation of the new ones after them, and the append of the new ones | `caching/registries/standard_expansions.py`, per [[Caching Between Passes]] |

## See also

- [[Home]] — the layer diagram
- [[Repository Map]] — where each pipeline's code sits
