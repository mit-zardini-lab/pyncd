# Claude Opus 5.5 (1M context), effort 40.
'''Kimi K3 in the CausalSlide, the form a figure of the model is drawn in.

Kimi K3 reads earlier tokens in two places. The latent attention reads the keys and the
values of every head at token `i_x - i_r` at distance `i_r` of each query `i_x`, and the
three causal convolutions of the delta attention read token `i_x - i_w` at tap `i_w`.
An operation broadcast over the tokens computes the same function at every token, so a
read can stand before it without changing a value of the expression.
`slide_causal_reads_backwards` moves every read back until it meets a copy whose other
branches read the same array at the token itself. In a latent layer the read of the
keys and the read of the values meet at the copy of the latent and pass it, and stop at
the copy of the hidden state that also feeds the queries and the output gate. In a delta
layer the three reads of the taps meet at the copy of the hidden state that feeds the
decays, the strengths and the gate, and the three projections then run over the tokens
and the taps.

The loop of the delta rule reads each token at the counter, which is no causal read. The
slide receives no read from the results of the loop and passes it whole.
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa: E501

import notebooks.sota.KimiK3.whole_model as whole_model

kimi_k3_slid = slide_causal_reads_backwards.slide_causal_reads_backwards(
    whole_model.kimi_k3)
