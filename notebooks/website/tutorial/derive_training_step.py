# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Deriving the training step of a tutorial model: the forward pass, which saves to a tape
the values the backward pass reads, and the backward pass, which returns the gradient of
every input and every weight.

`derive_training_step` applies the following passes to a model, in order.
`grab_parameters` writes every weight a `Linear` holds as a grab of a slot named after
the weight. `expand_softmaxes` writes each
softmax as an exponential, a sum, a reciprocal and a product, and
`merge_reindexings_and_einops` moves the reciprocal past the contraction against the
values, so that the probabilities are never built. `forward_backward` derives both
passes, `collapse_grabbed_residuals` points the backward pass at the slots of the
weights, and `dedup_and_collapse` removes the values the backward pass can read from the
forward pass instead of the tape. The backward pass then computes the row statistic of
FlashAttention from the output.

The collapsed tape still holds the exponential of the scores, one number for every pair
of a query and a key. `recompute_contraction_slots` deletes that slot and rebuilds the
exponential in the backward pass from the queries and the keys, which are on the tape
already. `store_operands_of_views` moves every slot holding the result of a view, such as
the keys read through a causal mask at every slot of every token, to the array the view
reads, and replays the view in the backward pass. `dedup_roots` then merges the grabs and
views the two rewrites repeat. The tape that is left holds what FlashAttention stores,
which is the queries, the keys, the values, the output and one statistic for each query,
and the input of every learned matrix, which the gradient of the matrix reads.

`backprop.forward_backward` carries every `cat.Block` into the forward pass and gives the
backward pass one block titled `R[...]` for it, and no later pass removes a block, so the
training step keeps every block the model was written with. A softmax is one operator
rather than a block, and `merge_reindexings_and_einops` moves its reciprocal past the
contraction that follows it, so no block could hold the written-out softmax once the
contractions are merged. `obsidian/07-para/Training.md` states the forms a trained model
passes through, and `obsidian/07-para/Recomputing the Exponent in the Backward Pass.md`
the two rewrites that shrink the tape.
'''
from __future__ import annotations

from dataclasses import dataclass

import algebra.einops_rearrange as einops_rearrange
import algebra.operator_expansion as operator_expansion
import data_structure.Category as cat
import para.algebra.pathway_collapse as pathway_collapse
import para.algebra.recompute_contraction_slots as recompute_contraction_slots
import para.algebra.store_operands_of_views as store_operands_of_views
import para.data_structure.MultiCategory as multi_category
import para.data_structure.ParaWrap as para_wrap
import para.processing.backprop as backprop
import para.processing.show_grabbed_parameters as show_grabbed_parameters


@dataclass(frozen=True)
class TrainingStep[L, M: cat.Morphism]:
    '''The forms a model passes through on the way to its training step.

    `parametrised` reads every weight from a named slot. `prepared` has every softmax
    written out and the contractions merged. `derived` holds both passes as
    `forward_backward` derives them, and `collapsed` holds them after the transfers
    between the passes are simplified. `recomputed` is the training step, with every
    slot a contraction of taped values rebuilds recomputed in the backward pass and
    every result of a view stored as the array the view reads.'''
    parametrised: cat.ProdCategory[L, M]
    prepared: cat.ProdCategory[L, M]
    derived: backprop.Taped[L, M]
    collapsed: backprop.Taped[L, M]
    recomputed: backprop.Taped[L, M]

    def forward_over_backward(self) -> multi_category.MultiCategory[L, M]:
        '''The training step's passes stacked, with every grab and every drop drawn on
        the operation it touches.'''
        return para_wrap.to_para_wrap_rows(self.recomputed)


def merge_each_pass[L, M: cat.Morphism](
    taped: backprop.Taped[L, M],
) -> backprop.Taped[L, M]:
    return backprop.Taped.from_passes(
        einops_rearrange.merge_reindexings_and_einops(taped.forward),
        einops_rearrange.merge_reindexings_and_einops(taped.backward))


def shrink_the_tape[L, M: cat.Morphism](
    collapsed: backprop.Taped[L, M],
) -> backprop.Taped[L, M]:
    return pathway_collapse.dedup_roots(store_operands_of_views.store_operands_of_views(
        recompute_contraction_slots.recompute_contraction_slots(collapsed)))


def derive_training_step[L, M: cat.Morphism](
    model: cat.ProdCategory[L, M],
) -> TrainingStep[L, M]:
    parametrised = show_grabbed_parameters.grab_parameters(model)
    prepared = einops_rearrange.merge_reindexings_and_einops(
        operator_expansion.expand_softmaxes(parametrised))
    derived = merge_each_pass(show_grabbed_parameters.collapse_grabbed_residuals(
        backprop.forward_backward(prepared)))
    collapsed = pathway_collapse.dedup_and_collapse(derived)
    return TrainingStep(
        parametrised=parametrised, prepared=prepared, derived=derived,
        collapsed=collapsed, recomputed=shrink_the_tape(collapsed))
