---
tags: [layer/caching, reference]
code: performance_modeling/morphism_work.py, performance_modeling/registries/operation_work.py, performance_modeling/collective_cost.py, performance_modeling/registries/machine_rates.py
status: stable
written: Claude Opus 5.5 (1M context), effort 40, on 2026-09-27.
---

# Operation Counts and Machine Rates

## The public package holds the operation count and the machine rates

A placement of the caches of a pass is compared with another by the operations the pass
performs and by the bytes its caches move, each at the rate a machine performs it, per
[[Deriving Caches by Dragging the New Tokens]]. `performance_modeling/` holds the two
things read by that comparison: the operation count of a broadcast morphism, and the rates
of a machine. The public copy of the package holds five files and nothing else.

| file | what it holds |
|---|---|
| `performance_modeling/morphism_work.py` | `read_symbolic_work`, the index sizes and the operations of one `cat.Broadcasted` with every size left as the numeric its axis carries, `read_morphism_work`, the same with every size bound, and `numeric_value`, which evaluates a numeric at given bindings |
| `performance_modeling/registries/operation_work.py` | the rule of each operator, which gives its operations per element and the role of every index read by it |
| `performance_modeling/collective_cost.py` | `MachineRates`, the rates for costing a pass |
| `performance_modeling/registries/machine_rates.py` | `H100_SXM5`, the published figures of an H100 SXM5 with the source of each |
| `performance_modeling/__init__.py`, `registries/__init__.py` | the package markers |

## The operation count of a morphism

`read_symbolic_work` reads the index variables of a morphism through
`einops_simplification.index_shapes`, one per degree position and one per contraction
group, per [[Expression Simplification]]. It looks up the rule registered for the
operator's class, or for the nearest class above it, and raises `UnsupportedWork` where
none is registered. The rule returns a role for each index, the indices carried by a
weight inside the operator, and the operations per element. `IndexRole.REDUCE` marks an
index summed or folded over by the operation. The result is a `SymbolicWork`, whose `elements` is
the product of the sizes of every index and whose `operations` is the elements times
the operations per element. Every size is a numeric, so the count is written before any
size is bound, and `bound` or `read_morphism_work` binds it.

The rules count a multiply and an add as two operations.

| operator | operations per element | elements |
|---|---|---|
| `Einops` | the number of operands less one, and one more where an index is summed, so 2 for a contraction of two operands | every index read or written by the operation |
| `Linear` | 2 | the degree, the features written and the channels summed |
| `Maximum`, `Product` | 1 | every index read |
| `AdditionOp` | the number of operands less one | every index of the result |
| `Arithmetic` | 1 | every index of the result |
| `Elementwise` | an estimate by the name of the function: 1 for `relu`, 4 for `sigmoid`, `exp`, `sqrt` and `tanh`, 8 for `gelu` | every index of the result |
| `View` | 0 | every index of the result |
| `SoftMax` | 8 | every index of the result |
| `Normalize`, `LayerNorm` | 7, and one more for each operand beside the array normalised | every index of the result |

A contraction of $q$ and $x$ over $d$ therefore performs $2 \lvert q \rvert \lvert x \rvert \lvert d \rvert$ operations, and a linear map from $m$ to $o$ applied to every token of $x$ performs $2 \lvert x \rvert \lvert m \rvert \lvert o \rvert$. The rule also names a resource, `matrix` for a `Linear` and for a contraction of two operands with a sum and `scalar` for every other operation, and nothing public reads it.

Three readers use the count. `caching.algebra.count_pass_operations` walks a pass,
multiplies in the repetition of every block and the degree of every box around each
operation, and counts every `Linear` and every contraction of two operands.
`advanced_axis_dynamics.algebra.absorb_linear_maps` compares the orders of a chain of
two contractions by the same count, per [[Advanced Axis Dynamics]].
`caching.algebra.cache_contents` binds the size of a cached axis with `numeric_value`.

## The rates of a machine

`collective_cost.MachineRates` holds the matrix and scalar arithmetic rates in
operations per second, the memory and network bandwidths in bytes per second, the
latency of one step of a collective, the memory of one GPU, and the bytes of one
element, two by default. It refuses a rate that is not finite and positive, a negative
latency, and a memory or element size below one byte.
`caching.algebra.cost_cache_placements.cost_of_a_pass` reads three of the fields: the
operations at `matrix_operations_per_second`, and the bytes the caches move at
`memory_bytes_per_second`, counted in elements of `bytes_per_element` bytes.

`machine_rates.H100_SXM5` holds the figures of the whole part, each beside a
`RateSource` naming where it was read.

| rate | H100 SXM5 | source |
|---|---|---|
| matrix operations per second | 989 TFLOP/s | the H100 SXM5 datasheet, dense BF16 tensor-core throughput |
| scalar operations per second | 67 TFLOP/s | the H100 SXM5 datasheet, FP32 throughput of the CUDA cores |
| memory bytes per second | 3.35 TB/s | HBM3 bandwidth, Table 1 of the FlashAttention-3 paper |
| network bytes per second | 450 GB/s | fourth-generation NVLink, 900 GB/s in both directions together |
| latency of one step of a collective | 1 µs | assumed |
| memory of one GPU | 80 GB | HBM3 capacity, Table 1 of the FlashAttention-3 paper |

`notebooks/classic/cached_mixtral_8x7b.py` builds its `MACHINE` from these figures at two
bytes per cached value, and costs every placement of the caches of one layer at one new
token after 32,767 earlier ones. `validate_mixtral_8x7b.py` checks the rates read by the module.

## Gaps

- `count_pass_operations` counts linear maps and contractions alone, so a normalisation
  or a rotation recomputed over a cache costs nothing in the comparison of placements,
  although `read_symbolic_work` counts both.
- The scalar rate, the network bandwidth and the latency are carried and read by nothing
  public. A pass is costed on one GPU, and the code does not estimate how long GPUs
  would spend exchanging results.
- The operations of an `Elementwise` are estimates by name, and a function missing from
  the table raises `UnsupportedWork`.

## See also

- [[Deriving Caches by Dragging the New Tokens]] — the comparison of placements that reads the count and the rates
- [[Caching Between Passes]] — the caches whose bytes are counted
- [[Advanced Axis Dynamics]] — the choice of order of a chain of contractions
- [[Expression Simplification]] — the index variables the count is read through
