# Claude Opus 5.5 (1M context), effort 40.
'''GLM-5.3 and its quantised form in the CausalSlide, the form a figure of the model is
drawn in.

A causal read of GLM-5.3 reads token `i_x - i_r` at distance `i_r` of each query `i_x`.
The indexer reads its keys back that way, and each gather reads the keys or the values
of every head back that way before it picks the selected tokens. An operation
broadcast over the tokens computes the same function at every token, so a read can
stand before it without changing a value of the expression.
`slide_causal_reads_backwards` moves every read back until it meets a copy whose other
branches read the same array at the query itself. In a Full or a Shared layer the read
of the keys and the read of the values meet at the copy of the latent and pass it,
because both read the one distance axis of `lightning_indexer.READ_BACK`, and then
stop at the copy of the hidden state that also feeds the queries. The key and value
projections then run over the queries and the distances. In the indexer the read
stops at the copy of the hidden state inside the indexer, because the weights of the
heads read the hidden state at the query.

The quantised form is the quantised model slid the same way. A cast is broadcast over
the tokens, so the read passes the casts, and taking the quantisations off the slid
quantised model gives the slid model.
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards  # noqa: E501

import notebooks.sota.GLM53.quantised_whole_model as quantised_whole_model
import notebooks.sota.GLM53.whole_model as whole_model

glm53_slid = slide_causal_reads_backwards.slide_causal_reads_backwards(
    whole_model.glm53)

glm53_quantised_slid = slide_causal_reads_backwards.slide_causal_reads_backwards(
    quantised_whole_model.glm53_quantised)
