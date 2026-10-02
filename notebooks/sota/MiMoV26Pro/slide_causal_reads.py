# Claude Opus 5.5 (1M context), effort 40.
'''MiMo-V2.6-Pro in the CausalSlide, the form a figure of the model is drawn in.

A full attention layer reads its keys and its values back from every query through the
view named Back, at token `i_x - i_r` for distance `i_r`, and a sliding window layer
through the view named Window, at token `i_x - i_w` for slot `i_w`. An operation
broadcast over the tokens computes the same function at every token, so a read can
stand before it without changing a value of the expression.
`slide_causal_reads_backwards` moves every read back until it meets a copy whose other
branches read the same array at the query itself. In each layer the read of the keys
and the read of the values are one read, so the two meet at the copy of the normalised
hidden state that feeds the queries, the keys and the values, and stop there. The key
and value projections, the value scale and the rotation of the keys then run over the
queries and the slots, and the rotation reads its table of turns back through the same
view.
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa: E501

import notebooks.sota.MiMoV26Pro.whole_model as whole_model

mimo_slid = slide_causal_reads_backwards.slide_causal_reads_backwards(whole_model.mimo)
