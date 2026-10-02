---
tags: [layer/backends, tool]
code: websocket_transfer/auxiliary_information.py, websocket_transfer/websockets_transfer.py
status: stable
---

# Naturals in the Legend

Written by Claude Opus 5.5 (1M context), effort 40, on 2026-09-28.

The legend of a figure lists its axes, per [[Advanced Display]]. The user asked on 2026-09-27 for a second table listing the naturals that appear in arrays, clickable as the rows of axes are and each with a code name. A `cat.Natural` is the datatype of an array of positions, and its field `max_value` is the bound its values stay below. The token identifiers of a language model are `Natural(|v|)`, where `|v|` is the size of the vocabulary. A natural is drawn as a wire of its own below the axes of its array, labelled with its bound.

## Each row states the bound, its size and its code name

`auxiliary_information.natural_legend_rows` writes one row for each natural that is the datatype of an array or a weave of the term, or that the quantisation of such a datatype holds, as an `INT64` quantisation of the token identifiers holds `Natural(|v|)`. Two naturals with one bound make one row. A row carries the LaTeX of the bound, the integer it comes to, the bound written in the code names of its symbols, and a key. The size is evaluated under the sizes the page assigns, or at the size written on the name of each symbol. It travels as a string of digits, because the bound of a 64-bit integer, $2^{63}$, is larger than the largest integer a JavaScript number holds exactly. The code name is Python in the code names of the symbols, with a sum inside a product bracketed and a reciprocal written as a division, as in `new_tokens_size + earlier_tokens_size` and `compression_ratio_size * entries_size`. A bound holding a symbol with no code name has none.

## A row names its natural by the structure of the bound

A natural has no uid, so a row cannot name its wires the way a row of axes names its axes. `auxiliary_information.natural_key` writes the structure of the bound: a symbol as `#` followed by its uid, an integer as its digits, and a sum, a product and a power as `+`, `*` and `^` followed by the keys of their parts in brackets. Any other numeric is written `?`. tsncd's `src/data_structure_processing/find_naturals_by_key.ts` writes the same key for the natural of every wire. It sits outside `advanced_display/`, because `display/Framework` needs the key to register the halo of a wire and may not import from `advanced_display/`.

## A row halos and locks its wires

Resting the pointer on a row halos every wire whose datatype is a natural with that key, together with the label of the wire, under the highlight token `natural:<key>`. A click locks the halo, and the lock reaches the diagrams inside the inspection boxes, as the lock of an axis row does. The token also reaches the arrow of a natural array in the two arrow forms, and a tape carrying a natural. The label in the first column is the one tsncd writes on the wire, so it reads as the figure reads. Both printers write a product with a reciprocal factor after a slash, so the bound of the multipliers of Engram reads $2^{63} / \hat{v}_{99092}$ in the table and on its wire.

## Naturals measured on the website pages

The rows were read on 2026-09-27 off the pages as packaged.

| page | naturals |
|---|---|
| Mixtral-8x7B | $\lvert v \rvert_{32000}$, the token identifiers |
| GLM-5.3 | $\overline{v}_{154880}$, and $\lvert x_{new} \rvert + \lvert x_{old} \rvert$ in the cached pass |
| DeepSeek-V4.1-Flash | twelve, among them $2^{63}$, $2^{63} / \hat{v}_{99092}$, the prime bounds of Engram and the block positions $\lvert P \rvert$ |

## Open choices

A bound holding a numeric other than a symbol, an integer, a sum, a product or a power has the key `?`, and two such naturals share a key. No page holds one.

Inside the N-gram Hash box of DeepSeek-V4.1-Flash the wire of the natural $2^{63} / \hat{v}_{99092}$ runs through the subscript of its label. The page written before this work shows the same overlap under the old spelling of the bound, because a datatype label claims no room of its own on the wire.

## See also

- [[Advanced Display]] — the legend and its first table
- [[Diagram Wire Format]] — the `naturals` field
- [[Indices of Inspection Box Formulas]] — the other addition of the same day
