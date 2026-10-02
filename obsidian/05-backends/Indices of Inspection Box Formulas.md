---
tags: [layer/backends, tool]
code: websocket_transfer/letter_formula_indices.py, websocket_transfer/write_formula_index_ranges.py, advanced_axis_dynamics/algebra/write_guard_ranges.py, websocket_transfer/auxiliary_information.py, algebra/write_index_notation.py
status: stable
---

# Indices of Inspection Box Formulas

Written by Claude Opus 5.5 (1M context), effort 40, on 2026-09-28.

An inspection box shows a formula over the block or the operator it opens on, per [[Advanced Display]]. The user ruled on the indices of those formulas on 2026-09-27 and 2026-09-28. The indices of one formula take different letters, the formula is followed by a line naming the range of each index it holds for every position of the index's axis, and the range of an index over a guarded axis names the positions that hold a value. This note states the rules, how the package writes them, the choices made and the cases measured on the nine website pages.

## The indices of one formula take the letters i, j, k in turn

Every formula in the package is written with each index of the axis `m` as `i_{m}`, per `algebra/write_index_notation.py` and `CLAUDE.md`. More than thirty functions write formulas from operators, and about a hundred formulas are written by hand, so the lettering is applied once, where the formula is packaged for tsncd. `websocket_transfer/letter_formula_indices.py` finds every `i_{…}` of the formula, with balanced braces, and gives the axes the letters i, j, k, l, m, n, p, q and onwards in the order the formula first names them. The axis stays as the subscript, so the formula of a linear map reads $y[i_{o}] = \sum_{j_{m} \in m} x[j_{m}]\, W_{Q}[j_{m}, i_{o}]$.

A letter is passed over where the formula already uses it. The formula of GLM-5.3's indexer projection writes an axis named `i`, so its indices begin at j. A formula holding the array `k`, or a bare index `i` as in $\sum_{i \in S}$, passes over that letter too, and so does a letter of the name of an axis the formula indexes, so no index reads $m_{m}$. The user asked for the letters in the order i, j, k, l, m, n, and the passing-over rule was added to keep an index apart from the symbols of its formula.

A description beside the formula names indices in plain text, as `i_x`, and is written with the same letters. An index of the description takes the letter of the axis whose name, up to a `|` and without braces or an accent, is the name the description writes, so the mask of the tutorial reads "Reads the state of token i_x - j_w at slot j_w of token i_x". A description that names an index no axis of its formula carries is left as written, and a localised page keeps a relettered sentence as exported, because no constant of its wording matches it.

A subscript of more than one character written without braces, as in `x_new`, is braced, because KaTeX reads `x_new` as $x_{n}$ followed by `ew`.

## A line under the formula names the free indices

An index is free where some clause of the formula uses it without naming its range. The clauses are the lines of a `gathered` block and the parts separated by `\quad`. A clause binds an index where it writes `i_{d} \in d`, as a sum or a maximum does, or where the index stands inside a set `\{…\}`. The quantisation formula of DeepSeek-V4.1-Flash binds $i_{y}$ under a maximum on its first line and uses it freely on its third, so $j_{y}$ is listed. The top-k selections, written as sets, list nothing.

The free indices travel as the `indices` field of the record, per [[Diagram Wire Format]], each as the formula writes the index and its axis. tsncd's `src/advanced_display/freeIndexLine.ts` draws the line $\forall i_{x} \in x,\; j_{d} \in d$ under the formula, each clause a KaTeX span of its own. Resting the pointer on a clause, or tapping it, shows "The axis $x$ carries a set of indexes, in $[0, |x|)$." The user worded the tooltip, and the axis was named in it because a line holds several clauses.

## A guarded index ranges over the positions that hold a value

An axis marked by a read outside its axis is an `AffineGuards.AffineSparseAxis`, per [[Advanced Axis Dynamics]]. Its position $j$, taken at the positions $i$ of its guides, holds a value where $0 \le \sum g\, i + s\, j + c < E$, and the universal unit elsewhere. The user asked on 2026-09-28 that the line write such an index over the positions that hold a value, $j_{r|x} \in [0, i_{x}]$ for a causal read and $j_{w|x} \in [0, \min(i_{x}, |w| - 1)]$ for a window.

`advanced_axis_dynamics/algebra/write_guard_ranges.py` solves the form for $j$. Where the stride $s$ is 1 or -1, `guarded_interval` writes the interval. Each end is the larger or the smaller of a bound of the axis, $0$ or $|w| - 1$, and a bound of the form. A bound is left out where the other one is at least as tight at every position of the guides, which the corner values `lowest_value` and `highest_value` of `AffineGuards` decide over sizes taken as positive integers. The causal read of the tutorial has $|w| = |x|$, so its bound $i_{x}$ never passes $|w| - 1$ and the interval is $[0, i_{x}]$. The window of DeepSeek-V4.1-Flash has a size of its own, and both bounds stay.

A stride of any other size makes each bound the floor of a quotient: the blocks of the candidate pool reach $j_{P|x} \le \lfloor i_{x} / |u| \rfloor$, and the compressed entries $j_{r|x} \le \lfloor (i_{x} + 1)/|a| \rfloor - 1$. The user chose on 2026-09-28 to state the condition in that case rather than the floors. The line then writes $i_{P|x} \in P|x$, and the tooltip carries the condition.

`guarded_condition` writes the condition for every guarded index, with the term of the index alone between the bounds of the form, and the tooltip adds "The position $j_{w|x}$ holds a value where $j_{w|x} \le i_{x}$, and the universal unit elsewhere." A form that reaches both ends of its axis bounds the index on both sides.

`websocket_transfer/write_formula_index_ranges.py` matches each free index to its axis by the LaTeX of the axis name among the axes of the block or the operator the formula is shown over. Two guarded axes of one name with different forms leave the index without a range, because the formula does not say which one it means. DeepSeek-V4.1-Flash holds two axes named `r|x`, one compressed and one not, and two named `s|x`. Each range names only indices already on the line. A guide the formula names is written with its letter, and a guide the formula does not name takes the next spare letter and is placed first, so the block split of the candidate pool reads $\forall k_{x} \in x,\; i_{P|x} \in P|x,\; j_{u|x,P} \in [0, \min(k_{x} - |u|\, i_{P|x}, |u| - 1)]$. The fields are `range` and `condition` of the index record.

## The guarded indices of the website pages

The table was read on 2026-10-01 off the pages written in this repository, with each size written as a subscript where the page assigns it. The plain attention of the tutorial has no mask and no guarded index.

| page | axis | line | condition on the hover |
|---|---|---|---|
| three tutorial pages, Mixtral-8x7B, DeepSeek-V3 | `w\|x` of the causal mask | $[0, i_{x}]$ | $j_{w \vert x} \le i_{x}$ |
| Attention Is All You Need | `w\|y` of the causal mask of the decoder | $[0, i_{y}]$ | $j_{w \vert y} \le i_{y}$ |
| GLM-5.3 | `r\|x` of the keys read back by the indexer | $[0, i_{x}]$ | $j_{r \vert x} \le i_{x}$ |
| DeepSeek-V4.1-Flash | `w\|x` of the window | $[0, \min(i_{x}, \lvert w \rvert_{128} - 1)]$ | $j_{w \vert x} \le i_{x}$ |
| DeepSeek-V4.1-Flash | `L\|x` of the Engram lookback | $[0, \min(i_{x}, \lvert L \rvert_{4} - 1)]$ | $j_{L \vert x} \le i_{x}$ |
| DeepSeek-V4.1-Flash | `L'\|G` of the n-gram prefix | $[\lvert G \rvert_{3} - i_{G} - 1, \lvert L \rvert_{4} - 1]$ | $\lvert G \rvert_{3} - i_{G} - 1 \le j_{L' \vert G}$ |
| DeepSeek-V4.1-Flash | `GK'\|GK` of the primes before a pair | $[\lvert G \rvert_{3} \lvert K \rvert_{8} - i_{GK}, \lvert G \rvert_{3} \lvert K \rvert_{8} - 1]$ | $\lvert G \rvert_{3} \lvert K \rvert_{8} - i_{GK} \le j_{GK' \vert GK}$ |
| DeepSeek-V4.1-Flash | `r\|b` and `r\|B` of the count back from an entry | $[0, i_{b}]$ | $j_{r \vert b} \le i_{b}$ |
| DeepSeek-V4.1-Flash | `P\|x` of the block split | the axis $P \vert x$ | $\lvert u \rvert_{8}\, i_{P \vert x} \le k_{x}$ |
| DeepSeek-V4.1-Flash | `u\|x,P` of the block split | $[0, \min(k_{x} - \lvert u \rvert_{8}\, i_{P \vert x}, \lvert u \rvert_{8} - 1)]$ | $j_{u \vert x,P} \le k_{x} - \lvert u \rvert_{8}\, i_{P \vert x}$ |

## Choices and results not taken

Writing the letters into each formula at its source was considered and not done. The functions that write a formula from an operator do not see the other indices of the formula they are part of, and about a hundred hand-written strings would have to be kept in step.

A first rule for a set counted an index bound only where it headed the set, $\{i_{r} : \ldots\}$. The top-k formulas write $\{p[i_{s}]\} = \{q[i_{C}] : \ldots\}$, so every index inside the braces is bound.

The floors of a strided guard were written out and set aside for the condition at the user's choice. The floors are correct, and a reader has to work each one back to the inequality it came from.

## Open choices

The free indices are read off the text of the formula. A formula that binds an index in a way the two rules do not see, such as a product $\prod$ written without `\in`, lists it as free. None of the formulas of the nine pages does.

A guide that the formula binds rather than holds free, where a free guarded index depends on a summed one, would be placed on the line as free. No formula of the pages does this.

## See also

- [[Advanced Display]] — the boxes and the legend
- [[Diagram Wire Format]] — the `indices` field and its `range` and `condition`
- [[Advanced Axis Dynamics]] — the guarded axis and its form
- [[Naturals in the Legend]] — the second table of the legend, added the same day as the lettering
