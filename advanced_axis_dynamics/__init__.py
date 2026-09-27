'''Axes whose live positions are decided by an affine form of the position.

Written by Claude Opus 5 (1M context), reasoning effort medium.

A codomain index of a `sc.StrideMorphism` outside `[0, size)` of its axis names no
position, and a read there yields the universal unit. That rule belongs to the stride
category itself. What follows from it does not: an array read through such a morphism
carries an axis some of whose positions hold the unit, which positions those are is an
affine form of the position, and the form has to be derived, carried through further
reads and folds, and restored when a consumer needs the parts of an axis separately.
This folder holds that derivation.

Nothing in `data_structure/`, `graphs/` or `algebra/` imports it, so a change here
reaches only `deepseek/`, `caching/`, two modules of `para/`, and the notebooks that
read at a negative stride or derive a cache. A validation of anything else need not
be run for a change here. It imports
`algebra.discovering_broadcasts` and `algebra.registries.accumulator` itself, which sit
below it and reach nothing of it.

    data_structure/
      AffineGuards   the axis carrying an affine form, the ends it empties, and
                     the reach of an affine row over a domain box
      Operators      the operators that read a reindexing covariantly:
                     `CovariantView`, one reindexing, and `ConcatenateAxes`, one
                     per input with their images filling one axis, and
                     `DeconcatenateAxes`, which cuts one axis into the parts that
                     fill it
      AxisConcatenation
                     the axis a concatenation produces, which references its parts,
                     is sized by the sum of their sizes and is labelled by them

    algebra/
      mark_sparse_domains     the domain axes a row reading outside its axis marks,
                              the pull-back through a further read, the form a fold
                              leaves, and `guarded_view`
      mark_sparse_codomains   the codomain axes a merge read covariantly does not
                              fill, and the groups that make the read a function
      concatenation_expansion `F(Concat(x, y)) = B_F(F(x), F(y))`, as a rewrite of
                              every consumer of the concatenated wire, and the same
                              rewrite through a block, which is written out first so
                              that the concatenation and its consumers stand in one
                              scope
      disentangle_reindexings the independent maps a reindexing holds, as the product
                              of one factor per connected component of its rows
      drag_index_backwards    one index of one axis of a result carried back to the
                              inputs by a reverse crawl, `[F; x](input)[t_x] =
                              F(input[t_x])`, every position it reaches pinned at
                              it, carried through the body of every box, a view
                              written where it stops, and `pin_guards`, which
                              afterwards substitutes the index into every guard
                              whose guide it passed through
      move_reads_backwards    the read of a result moved back towards the inputs by
                              the same crawl, composed into every view it meets,
                              carried through every operator broadcast over the axes
                              it reads and through the body of every box, and turned
                              into the values of the index at an arrangement, so
                              that the index morphism `t_x` and the causal mask
                              become one view in front of the key and value
                              projections and the rotary table of GLM-5.3 becomes
                              the turns of the one query
      slide_causal_reads_backwards
                              the same crawl started at every causal read, which
                              moves it back to the copy it masks and gives the
                              CausalSlide, the standard form of a displayed
                              expression
      absorb_linear_maps      a chain of two contractions rewritten into the order
                              of its operands that costs the fewest operations at
                              bound sizes, which absorbs the up-projections of
                              multi-head latent attention into its queries

    registries/
      part_combination  the operator folding the partial results over the parts,
                        which is the accumulator `algebra.registries.accumulator`
                        declares for the fold
      derivative        the reverse derivative of a merge, of a concatenation and
                        of a deconcatenation
      standard_expansions
                        a deconcatenation written out as a copy and one view per
                        part

`validate_advanced_axis_dynamics.py` checks the forms the reads of
DeepSeek-V4.1-Flash derive, their live positions at concrete sizes, and the
concatenation expansion against the two-branch attention core written by hand.
`validate_covariant_broadcast.py` checks the merge that broadcasts its input over the
domain axes it does not carry, against the entries the reference's `compress_lens`
admits at the compression ratios one to four, and the split of a reindexing into the
independent maps it holds.

The mathematics is in `obsidian/02-categories/Advanced Axis Dynamics.md`, with the
three reads that produce a guard in
`obsidian/02-categories/Padding and Masks as Sparse Axes.md`.
'''

from advanced_axis_dynamics.algebra import (
    concatenation_expansion as concatenation_expansion)
from advanced_axis_dynamics.algebra import (
    drag_index_backwards as drag_index_backwards)
from advanced_axis_dynamics.algebra import (
    move_reads_backwards as move_reads_backwards)
from advanced_axis_dynamics.algebra import (
    mark_sparse_codomains as mark_sparse_codomains)
from advanced_axis_dynamics.algebra import mark_sparse_domains as mark_sparse_domains
from advanced_axis_dynamics.data_structure import AffineGuards as AffineGuards
from advanced_axis_dynamics.data_structure import AxisConcatenation as AxisConcatenation
from advanced_axis_dynamics.data_structure import Operators as Operators
