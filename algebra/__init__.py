'''General rewrites of an expression, usable by any feature.

The rewrites here change how an expression is presented without changing what it
computes, and more than one feature needs them.

  node_expansion          Factors a Broadcasted's reindexings out of the operator
                          and into standalone View morphisms. The core it leaves
                          behind carries the degree identity as its own reindexing.
  reindexing_absorption   The move back. A node folds into each reader's reindexing
                          for that operand, copied over a fan-out, so a repeat stands
                          as late as possible on every branch.
  einops_simplification   An einsum as a function of shapes alone, and so two
                          einsums in a row as one, when the single einsum is not a
                          worse algorithm.
  merge_into_consumer     The search that both of the rewrites above are phrased
                          against. One search finds a morphism read exactly once, by
                          exactly one other morphism. The other finds a morphism read
                          by any number, and copies it into each of them.
  einops_rearrange        Where one Einops feeds another, the two folded into one
                          contraction, and that contraction split back into the
                          components its segments fall into.
  linear_expansion        A Linear split into a 0-input weight array and an Einops
                          that contracts it against the real input, so that the
                          output axes are wired in through a reindexing.
  operator_expansion      An operator written out in its primitives. A SoftMax
                          becomes exp ; copy ; sum ; 1/z ; scale.
  discovering_broadcasts  A repeated operation written as one block broadcast over a
                          degree, proved by expanding the statement back out.
  registries/accumulator  The operator that folds two partial results of a fold into
                          one, asked of a stream loop by `para` and of a
                          concatenation by `advanced_axis_dynamics`.
  registries/standard_expansions
                          The rules of operator_expansion collated by operator class,
                          each with the formula it writes out, which the display reads
                          to offer the expansion of every such operator in a figure.
  write_index_notation    The pieces of a formula in index notation, `i_{m}`,
                          `\\sum_{i_{m} \\in m}` and `x[i_{m}]`, written from the axes of
                          an operator, which the rows of the registry above use to
                          write the formula an inspection box shows.
  define_by_expansion     An operator paired with its standard expansion in a
                          `cat.DefinedExpression`, which a figure draws as the
                          operator, `:=` and the expansion.

Expansion and absorption are not each other's inverse in practice, which is why both
are here. A derivation expands in order to differentiate, because a reindexing and an
operator have duals of completely different kinds. The expression the derivation
writes afterwards has nodes the original never had, sitting against consumers they
were never factored out of.

The package sits on `data_structure`, `construction_helpers`, `term_utilities` and
`graphs`, so that every feature above them can depend on it.

`linear_expansion` is the one module that imports `para`, for
`para.processing.write_linear_formula`, which writes the formula an inspection box
shows for an expanded weight. `linear_expansion` registers the standard expansion of
a `Linear` that the display reads, and `einops_rearrange` is half of the pass
`merge_reindexings_and_einops` that `validate_simplification` and the `para` notebooks
run.

`graphs` became a dependency when `merge_into_consumer` arrived. "Read once, and by that
one" is a question about wires, and a morphism has no wires. A morphism has positions
in a product, a `Rearrangement` away from meaning anything. `graphs` sits on
`data_structure` alone, so the dependency adds nothing to the package's own reach.
'''
