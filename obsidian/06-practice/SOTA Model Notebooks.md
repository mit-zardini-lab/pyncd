---
tags: [layer/practice, reference]
code: notebooks/sota/, notebooks/sota/GLM53/, notebooks/website/modern/
status: stable
---

# The SOTA Model Notebooks

Written by Claude Opus 5 (1M context), effort high.

Two packages under `notebooks/sota/` each assemble a frontier architecture as a single
morphism in **Br**, drawn at the level a careful reader of the release would recognise.
DeepSeek-V4.1-Flash is drawn by `notebooks/sota/DeepSeekV41Flash.ipynb` and by its page
of the lab website, and GLM-5.3 by its page alone, per [[Website Notebooks]].

| notebook | model | the mechanism it exists to draw |
|---|---|---|
| `DeepSeekV41Flash.ipynb` | DeepSeek-V4.1-Flash, 10 September 2026, MIT. 552B backbone plus 196B Engram, 8B active in prefill and 16B in decode, 1M context, restricted to text | **CED and CSA2, drawn at the quantisations of the released code**: the decoder's global key-value entries are one projection of the final hidden state of the encoder, and every layer from 2 upward reads a 128-token sliding window beside its selected entries in one of three modes, Full, Reindex and Reuse, the last of which runs no indexer at all. The hierarchical sparse indexer keeps the best 2048 blocks of 8 compressed positions, so a Reindex layer scores 16,384 positions however long the context is. The entries, the indexer keys, the candidate pool and the selection travel between layers on tape slots, one per field of the shared runtime of the released code. The core reads the window slots and the selected slots as one concatenated slot axis, which is the spelling of the released code. Causality is drawn rather than masked: the window axis, the entry axes, the block axes and the selection axes are `AffineGuards.AffineSparseAxis`es carrying the affine forms computed by the masks of the released code, per [[Padding and Masks as Sparse Axes]] and [[Advanced Axis Dynamics]]. Every figure is a part of `quantised_text_only_model.v41_flash_text_only_quantised`, so every wire carries the quantisation of the released code, and one figure takes every quantisation back off, per [[Quantization]] and [[Stripping Quantisations]] |
| `../website/modern/GLM53.ipynb` | GLM-5.3, first commit on Hugging Face 27 August 2026, licence `glm-5.3`. The base model of GLM-5.2 after further post-training. 753B parameters in the checkpoint, 78 layers, 1M context | **The architecture from its reference implementation**: every size read from the `config.json` of the checkpoint, and every mechanism written from `modeling_glm_moe_dsa.py` of `transformers` at one pinned commit. Multi-head latent attention reads the 2048 tokens kept by a lightning indexer through a gather, the core is one box computed once per query and head, and a Full layer drops its selection onto the tape slot `sel` for the three Shared layers after it to grab. The model is the package `notebooks/sota/GLM53/`, and the claims are `validate_glm53.py` and `validate_quantised_glm53.py` in the package and `notebooks/website/modern/validate_glm53.py` beside the notebook, per the section below |

## The model lives in one package beside the notebook

The notebook keeps its prose and its figures and imports the model from
`notebooks/sota/DeepSeekV41Flash/`, so that several agents can work on the pieces at
once. The package holds two kinds of module, and its `__init__.py` lists every one of
them and what it holds.

| kind of module | what it holds |
|---|---|
| a mechanism module | one part of the architecture: the attention core and its modes, the lightning indexer, the candidate pool, the mixture of experts, the layer stack, the rotary tables, the caches, the Engram hash, the DSpark draft chain and the vision pathway. `whole_model.py` composes them |
| a wiring module | the tape slots, the boxes and the layer plan that write the mechanisms into one expression. `text_only_model.py` is the model restricted to text, and `quantised_text_only_model.py` gives that model the quantisations of the released code |

The claims live in a validator rather than in the notebook's cells, which is the
exception to the rule that a notebook is its feature's test, ruled on 2026-09-15.
`validate_quantised_text_only_model.py` holds every claim the notebook makes,
`quantization/validate_quantization.py` holds the claims about the pass that
quantises the model, and the three other validators of the package hold the claims
about the parts the model is built from. Each runs one `check_` function per claim,
prints one line per check and exits non-zero on a failure, so the notebook reads as
an account of the model and the evidence for a sentence of it is in one file.

| validator | what it asserts |
|---|---|
| `DeepSeekV41Flash/validate_quantised_text_only_model.py` | every claim of `notebooks/sota/DeepSeekV41Flash.ipynb`, in forty-six checks: the ends of the model, the released size of every axis a figure is drawn at, the number of sites each mechanism is composed at, the 190 casts counted by the pair of formats each reads and writes, and the model with every quantisation taken back off it |
| `DeepSeekV41Flash/validate_deepseek_v41_flash.py` | the shape of each part: the axes each operation consumes, the modes of the attention core, the forms the selections carry and the wiring of the layer stack |
| `DeepSeekV41Flash/validate_omitted_mechanisms.py` | each mechanism the base model leaves out, written with the standard operators as far as they go and one `ops.GenericOperator` where the standard set cannot state the operation |
| `DeepSeekV41Flash/validate_deepseek_v41_flash_integrated.py` | the integrated model and the text-only model, including that every block of the text-only model stands in the integrated model under the same title |

## The account of DeepSeek-V4.1-Flash

Since 2026-09-20 the notebook is one account of the model for a reader who has seen
nothing else. Forty-seven cells run in the reading order of the forward pass. The
setup cell is followed by a cell running
`validate_quantised_text_only_model.check_every_claim_of_the_notebook`, which prints
one line per claim, so a reader sees each claim confirmed where the notebook
asserted it until 2026-09-20. Four further cells explain how a diagram is read, give
the axes, the affine forms and the released constants, and say what the restriction
to text removes. Each section after them draws
one part of the model and explains four things about it: what the part computes, why the
released code computes it that way, how the part is written as a morphism, and where the
quantisation of a value matters. The parts are the embedding and the four residual
streams, the manifold-constrained hyper-connections, the sliding-window attention and the
attention core, the rotary embedding and its YaRN scaling, the token compressors, the
three quantised caches, the lightning indexer and its Top-512, the hierarchical candidate
pool, the Full, Reindex and Reuse modes, the mixture of experts and its routing, the
Engram modules, the forty layers and the output probabilities.

Every figure is `quantised_text_only_model.v41_flash_text_only_quantised` or a part of
it, so every wire carries the quantisation of the released code and every rounding is
read off the label of the wire after it. Quantisation is not the subject of the notebook,
and each section says in a sentence or two where a format matters to its own part: the
cast into MXFP8 in front of a projection whose weight is FP8 or FP4, the FP4 and FP8
round trips of the three caches, and the INT32 positions handed to the attention kernel.
`quantised_text_only_model.part_named` returns a box of the quantised model,
`part_titled` a block of it, and `part_before_the_layers` and `part_after_the_layers` the
head and the tail, which are the four ways a figure asks for a part.

Three short sections near the end give the policy table, the weight table and the counts
of the 190 casts, with the counts asserted in the cell beside them. [[Quantization]]
states the pass and the policy. Two figures send the quantised model to the open page and
write it to `outputs/pages/DeepSeekV41Flash.html`, both with an inspection box over every
block and every operator. One cell then applies `strip_quantisations` to the quantised
model and draws the whole model with every quantisation taken back off, so that a figure
carrying the formats can be read against the same figure carrying none.
[[Stripping Quantisations]] states the functor and the one part of the model that comes
back differently. The notebook closes with the axes of the model across a deployment, the
facts its figures leave unstated, and the reference table, which cites the released
inference code at one commit with a line range per mechanism, one row per mechanism.

`notebooks/website/modern/DeepSeekV41Flash.ipynb` is the account of the same text-only
model for the lab website, per [[Website Notebooks]]. It draws the model part by part in
the CausalSlide form, holds no checks, and writes
`notebooks/website/output/modern/DeepSeekV41Flash/index.html`, one page with the variants
Quantised and Unquantised in the group Decode. The unquantised variant is
`text_only_model.v41_flash_text_only` embedded as a term of its own, because the
dequantisation functor would delete the roundings of the three caches, which the released
code performs. Every figure and both variants are drawn with the quantisation labels kept
on every wire, and `notebooks/website/modern/validate_deepseek_v41_flash.py` beside the
notebook holds every claim it makes.

## The text-only model and its quantisations

The text-only model reads the token identifiers and returns the probabilities of the next
token, so it holds the algorithm a prompt of text goes through and nothing else. The image
pathway, the modality of a token and DSpark are taken out, and every mechanism that acts on
text is kept and is stated as the integrated model states it. Three things follow from there
being no image. The model writes one tape slot of its own, for the identifiers, where the
integrated model writes three. The router reaches the first of its two correction biases at
every token, so `text_only_mixture.py` gives it a bias of one learned number per expert. The
gate of Engram has no factor for the modality, in `text_only_engram.py`, and the hash of both
models reads the identifiers alone. The layer count is unchanged at forty, and the four
Reindex groups are one repeated block.

`quantise_model` then writes a `Quantization.Quantified` onto every wire of
`text_only_model.v41_flash_text_only` and a `Quantization.TypeConvert` named `cast`
wherever an operation requires another quantisation, descending into the body of every box.
A tensor between two modules or kernels is BF16, arithmetic inside a module upcasting on
entry and inside every kernel is FP32, an activation is rounded only in front of a
projection whose weight is FP8 or FP4, to MXFP8, which is E4M3 with one UE8M0 scale per 32
channels, and everything else promotes as PyTorch does. An index carries INT64, and the
positions handed to the attention kernel INT32. The model holds 190 cast operations, and
every quantisation and every weight cites the released line read for it. The policy is
`quantised_text_only_model.RELEASED_POLICY` and the rule followed by each operator is
`quantization/registries/operator_quantisations.py`, per [[Quantization]].

Every figure of the notebook draws its casts thin, the default since 2026-09-20, so each
change of quantisation is read from the label on the wire changing. The FP4 round trip is
drawn a second time with `CastPresentation.DRAWN`, which gives each cast its chevron, and
the stripped model of [[Stripping Quantisations]] is drawn beside the quantised one, which
is the same figure with the casts and the labels gone. The last figures send the quantised
model to the open page and write
`outputs/pages/DeepSeekV41Flash.html`, both with an inspection box over
every block and every operator.

## The account of GLM-5.3

`notebooks/sota/GLM53/` writes GLM-5.3 from its reference implementation. Z.ai ships no
inference code with the checkpoint, and the model card names the class
`GlmMoeDsaForCausalLM`, which the `glm_moe_dsa` model of `transformers` implements. The
package reads `modeling_glm_moe_dsa.py` and `configuration_glm_moe_dsa.py` at the commit
`7cd73d9df0c1` of `huggingface/transformers` and the `config.json` of the checkpoint at
the commit `aca966e4e027` of `zai-org/GLM-5.3`, all read on 2026-09-23, and
`reference_links.py` pins every link to those commits. The package takes the form of the
V4.1 package: one module per mechanism, the title and the description of every block in
`block_titles_and_descriptions.json`, the tables of the inspection boxes in
`operator_explanations.py`, and the claims in `validate_glm53.py`.

- Every size is read from the configuration of the checkpoint, so the configuration binds
  every axis but the tokens.
- The selection of an IndexShare group is the tape slot `sel`, per [[Representing Models]].
  Layer 2 and the first layer of each of the eighteen repeated groups drop it, and the
  three Shared layers after each grab it, the repeated group carrying the counter `l`.
- The reference cuts the results of `q_b_proj`, `kv_a_proj_with_mqa` and `kv_b_proj` with
  `torch.split`, so each part is a weight of its own, and the unturned and turned parts of a
  query or a key are joined onto the axis `a` by `aops.ConcatenateAxes`.
- The reference expands the key and the value of every token and masks the tokens outside
  the selection. The expression reads the selected tokens with a `dst.IndexSelect` after a
  view reading back from each query, so the distance axis `r|x` and the selection axis
  `s|x` carry the causal form stated by the mask of the reference. The read back is marked
  once, as `lightning_indexer.READ_BACK`, so the CausalSlide passes it to the copy of the
  latent, per [[Advanced Axis Dynamics]].
- The multi-token prediction layer of the checkpoint is left out, because the reference
  skips its weights when it loads the checkpoint.

The expression was compared by value with the reference on 2026-09-26. On random weights
at small sizes, with the released plan of 78 layers, the two agree at the logits to a
relative difference of 9.3e-7 or less, which is the float32 arithmetic inside the
reference. The router marks the chosen experts with `[x > 0]` on the biased scores, which
gives the reference's gates only where every chosen expert's biased score is positive.
The correction biases of the checkpoint are at least 3.69 in every layer, so the
condition holds for the released weights.

`quantised_whole_model.py` quantises the model as `transformers` runs the FP8 checkpoint
`zai-org/GLM-5.3`, beside which Z.ai publishes `zai-org/GLM-5.3-BF16`. It states the
quantisations as a policy whose every row cites its lines in `modeling_glm_moe_dsa.py`, in
the FP8 integration of `transformers` at the same commit, in the two hub kernels that
integration loads, and in the configuration and shard headers of the checkpoint, all read
on 2026-09-26 and checked again on 2026-09-27. The projections are E4M3 with an FP32 scale
per 128 by 128 block, and the operand of each is rounded to E4M3 with an FP32 scale per 128
channels. The environment the policy describes is PyTorch 2.10.0 in its wheel for CUDA
12.8. `validate_quantised_glm53.py` holds the claims about the policy, including that
stripping the quantisations returns `whole_model.glm53`, and [[Quantization]] states the
policy.

`notebooks/website/modern/GLM53.ipynb` is the public notebook of the model. It draws every
figure in the CausalSlide, the quantisations, and the pass over new tokens derived from
the model by `notebooks/caching/CachedGLM53/derive_cached_glm53.py`, per
[[Caching Between Passes]], and writes the four variants into
`notebooks/website/output/modern/GLM53/index.html`. `notebooks/website/modern/validate_glm53.py`
beside it holds its claims.

## The wording of a figure is a file

`notebooks/sota/DeepSeekV41Flash/block_titles_and_descriptions.json` holds every block
title and every block description of both packages, as a wording file in the form stated by
`utilities/wording_json.py`: one entry per name, a sentence shared by several descriptions
as one entry named by the others as `$NAME`, and a description composed by a module from a
layer number or a size as an entry with a field in braces, filled at the call with
`.format`. The module beside the file declares the dataclass `BlockTitlesAndDescriptions`,
one typed field per entry, loads the file into `TEXT` at import, and refuses a file whose
names differ from the fields. Each construction module imports `TEXT` as `text` and names a
field of it where the string used to stand.

The user asked on 2026-09-18 for one file they could edit to change the wording of a
figure, on 2026-09-20 for the shared sentences to be written once, and the same day for the
wording to be a JSON file that is edited, copied and translated on its own, with the
dataclass reading it. A formula stays beside the algebra that builds it, because a formula
is written from the axes of its morphism. The entries are grouped by mechanism in the order
the model is built, and the fields follow the same order.

The same day the user asked for an HTML page that switches between wordings, and
`websocket_transfer/localise_descriptions.py` reads every description of a page back as the
entries that composed it and writes it again from a second wording file, which holds the
entries it changes and resolves against the first, per *A page that carries its
localisations* in [[Diagram Wire Format]]. Two further wording files hold the prose of the
packages that draw a model: `algebra/registries/expansion_wording.json` for the standard
expansions, the linear maps and the rotary tables, and
`notebooks/display/display_wording.json` for the casts and the elementwise maps, each
loaded by a dataclass beside it. On the quantised text-only page no description is left
outside the three files.

The axes of `declared_axes` and the wiring idioms of `construction_idioms` are imported by
name, because an expression written out of `declared_axes.x` and `construction_idioms.over`
cannot be read against the mathematics it states, and every other module is imported under
its own name and called through it.

## Where the sizes arrive

No size appears anywhere in the construction, and it is a rule rather than a habit. Every
axis carries a free symbol, and `quantised_text_only_model.released_assigned_sizes()`
builds a `NumericConfig` over the model, assigns the released sizes by name and returns
the integers. A figure passes them as `assigned_sizes`, so the display writes each size
onto the label of its axis while the term stays symbolic. The second cell of the notebook
tabulates them, one row per axis with its size and what the axis indexes, and the third
tabulates every named constant of the released configuration with its value and the line
that sets it, from `released_constants.RELEASED_CONSTANTS`.

The selection counts take deliberate effort, because `ds.TopK` accepts an `nm.Integer` and
hides a number inside the expression. Write `k = nm.FreeNumeric.named('k')` instead, and
the top-k joins everything else in the configuration.

Keep the configuration's names to single-letter bodies. `DynamicName.from_str` splits on
`_` and `NumericConfig.assign_values` matches on the body, so `k_e` and `k_x` collide
silently into one entry.

Thirty-three symbols are bound, because the released configuration is public, and they
include the vocabulary at 129,280 and the row count of each Engram table. The token axis
`x` stays symbolic, and so do the four axes the views relate to it by their strides: `b`,
the compressed entries of the encoder, `B`, those of the decoder, `r`, an entry counted
back from a query, and `P`, the blocks of the candidate pool. The counter `l` of the two
repeated groups stays symbolic as well.

GLM-5.3 binds all fifteen of `m h q c n t u i d e k f g s v` from the `config.json` of
the checkpoint, with `p`, `a`, `\bar{d}` and `r` written as sums and products of them,
and leaves `x` and the counter `l` of the repeated IndexShare group symbolic.

A number that is not sourced stays a symbol. Guessing one puts an invented figure into a
diagram whose whole discipline is that its figures are sourced.

## Sources

DeepSeek-V4.1-Flash was released on 2026-09-10, and its sources are its
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), its
[release note](https://api-docs.deepseek.com/news/news260910/), sections 2 and 3.2 of the
technical report shipped in the model repository, and the reference forward pass shipped
beside it, all read on the day of release. `reference_links.py` in the base package holds
every link, pinned to the commit the line was read at, and an inspection box over an
operation lists the lines it was written from.

GLM-5.3's sources, read on 2026-09-23, are its
[model card](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/README.md)
and [configuration](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json)
at the commit `aca966e4e027`, the `glm_moe_dsa` model of `transformers` at
[commit `7cd73d9df0c1`](https://github.com/huggingface/transformers/tree/7cd73d9df0c14b151c684b708a9f27d8d0349dfe/src/transformers/models/glm_moe_dsa),
the parameter count of the checkpoint reported by the Hugging Face API, and the
[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3), which has no commit to pin and
was updated on 2026-09-18.

Corroboration is not confirmation. The notebook's header states which of its numbers are
disclosed, which are assumptions from the model's lineage, and which stay symbolic.

## See also

- [[Representing Models]] — the rules a model is written to
- [[Quantization]] — the pass that writes a quantisation onto every wire
- [[Stripping Quantisations]] — the functor that takes them off again
- [[Advanced Display]] — the legend and the inspection boxes an interactive figure carries
- [[Notebooks]] — the other notebooks, and how to run one
- [[Website Notebooks]] — the pages of the lab website that draw both models
- [[Caching Between Passes]] — the cached pass of GLM-5.3
