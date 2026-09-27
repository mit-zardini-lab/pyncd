---
tags: [layer/practice, reference]
code: notebooks/website/, notebooks/website/website_output.py, notebooks/website/rewrite_website_pages.py, notebooks/website/tutorial/derive_training_step.py, notebooks/classic/, notebooks/sota/GLM53/, notebooks/caching/CachedGLM53/
status: evolving
written: Claude Opus 5.5 (1M context), effort 40, on 2026-09-27.
---

# Website Notebooks

## One notebook, one validator and one page for each model

Each model shown on the diagrams page of the lab website has one notebook under
`notebooks/website/`, one validator beside it, and one interactive page. A notebook holds
the prose and the figures and makes no checks. Every fact stated by its prose is checked
by the validator beside it, one `check_` function per claim, in the order of the
notebook. [[Validation]] discovers each `validate_*.py` and each notebook under
`notebooks/website/` as a target of its own, so the following command runs every one of
them.

```bash
python validations/run_validations.py --name website
```

The nine notebooks stand in three folders. `tutorial/` holds attention, attention with
weights and a residual connection, multi-head attention and grouped-query attention.
`classic/` holds the transformer of *Attention Is All You Need*, Mixtral-8x7B and
DeepSeek-V3. `modern/` holds GLM-5.3 and DeepSeek-V4.1-Flash. Every figure and every page
draws its model in the CausalSlide form, in which each causal read stands at the copy
that feeds the keys and the values, per [[Yoneda and Cartesian Tricks]].

`notebooks/website/WebsiteDisplay.md` beside the notebooks lists every notebook with its
validator, its page and its variants.

## The files of the website notebooks

| path | what it holds |
|---|---|
| `notebooks/website/tutorial/` | the four tutorial notebooks and their validators, and the modules imported by them: `express_attention.py` builds the four expressions, `derive_training_step.py` derives the training step, `tutorial_pages.py` holds the variants of each page, `tutorial_wording.json` the titles and the descriptions, and `check_gradients_in_torch.py` compares a training step with `torch.autograd` |
| `notebooks/website/classic/`, `notebooks/website/modern/` | the five model notebooks and their validators |
| `notebooks/classic/` | the classic models: a module building each model, a module quantising it, a module deriving its cached pass, a module holding its page variants, a wording file of its block titles and descriptions, `shared_mechanisms.py`, which holds the causal read, the rotation of channel pairs, the broadcast of a box once per head and the residual connection shared by the three models, and `reference_links.py` |
| `notebooks/sota/GLM53/` | GLM-5.3 from its reference implementation, one module per mechanism, with `validate_glm53.py` and `validate_quantised_glm53.py` |
| `notebooks/caching/CachedGLM53/` | the cached pass of GLM-5.3, written by hand and derived, per [[Caching Between Passes]] |
| `notebooks/sota/DeepSeekV41Flash/` | the DeepSeek-V4.1-Flash package of [[SOTA Model Notebooks]] |
| `notebooks/website/output/` | the pages, one folder per page |
| `notebooks/website/website_output.py` | the folders holding the pages |
| `notebooks/website/rewrite_website_pages.py` | the script that writes every page again |

## A page carries every variant of its model

The last cell of each notebook writes its page through
`notebook_diagrams.show_page_variants`, per [[Diagram Display]]. A page carries every
variant of its model in one file, and a selector above the figure switches between them.
The selector copies the model selector of the diagram viewer of the website. It stays on
the page when the website hides the form and theme buttons of tsncd, because the website
has no selector of its own for the variants. The variants share one compressed table of
values, so a subterm or a description held by two variants is stored once, per
[[Diagram Wire Format]].

A `PageVariant` belongs to a `PageVariantGroup`, and carries a title and a detail
written under the title in the selector. The pages use three groups.

| group | variants | what each shows |
|---|---|---|
| Pass | Forward, Training | the model as it is written, and the forward pass saving to the tape over the backward pass derived from it, per [[Backpropagation]] |
| Decode | Quantised, Unquantised | the whole model over every token with no cache, in the quantisations of the released code and in the reals |
| Cached | Quantised, Unquantised | one pass over the new tokens reading the earlier tokens from caches, derived by `caching/algebra/derive_cached_pass.py`, per [[Deriving Caches by Dragging the New Tokens]] |

A page opens on `decode-quantised` where it has that variant. Each variant carries a wrap
width of its own, chosen so that every block of the variant is drawn whole, per
[[Diagram Display]].

## An unquantised variant is derived in the browser or embedded

A derived variant carries no term. It names its source variant and the `PageFunctor`
applied by tsncd to the term of the source. `PageFunctor.DEQUANTISE` is the one
functor. tsncd applies it while the page shows "Applying Dequantization Functor...". The
functor removes every quantisation, turns every cast that no longer converts into an
identity, and then removes identities from the leaves upwards, so that a composition of
identities and a block around an identity become the identity.
`quantization/algebra/strip_quantisations.py` states the same functor in Python, with
`algebra/remove_identities.py`, and each validator checks that it turns the quantised
model into the unquantised one, per [[Stripping Quantisations]].

An embedded variant carries its own term. A page embeds a variant where the functor
would not give the model in the reals. The unquantised DeepSeek-V4.1-Flash is embedded. The
released code rounds its cached latents, the entries of its candidate pool, and the keys
and queries of its indexer through FP8 and FP4 in place, so the model written in the
reals keeps those roundings, in the blocks titled *FP8 Round Trip of a Window Latent*,
*FP4 Round Trip in the Indexer* and *FP4 Round Trip of an Entry*. The functor would turn
every one of those casts into an identity, and the blocks would round nothing. Embedding
the model in the reals adds about 0.5 MB to the page.

A derived variant whose settings change the text of the inspection boxes, such as the
roles of the weights or the tables of explanations, carries an auxiliary of its own, so
the unquantised variant of a model states what the weights of the model in the reals
are. The validators of the transformer of *Attention Is All You Need*, DeepSeek-V3 and
GLM-5.3 check that the unquantised variants carry the tables of the reals.

## The pages are written into one folder per page

`website_output.py` declares the folders under `notebooks/website/output/`, which mirror
the folders of the notebooks. Each notebook passes the folder of its group as
`page_directory` in the settings of its page. A page is written as `index.html` in a
folder named after the model, so the address shared by a reader ends in the name of the
folder, as `.../classic/Mixtral8x7B/` does. A page named `<Model>.html` stands beside
each folder and sends a reader to the folder. It keeps the query and the hash of the
address, so `Mixtral8x7B.html?variant=cached-quantised` opens `Mixtral8x7B/` on the same
variant. [[Diagram Display]] gives the redirect in full.

```
notebooks/website/output/
  tutorial/   Attention/index.html                        Attention.html
              AttentionWithWeightsAndResidual/index.html  AttentionWithWeightsAndResidual.html
              GroupedQueryAttention/index.html            GroupedQueryAttention.html
              MultiHeadAttention/index.html               MultiHeadAttention.html
  classic/    AttentionIsAllYouNeed/index.html            AttentionIsAllYouNeed.html
              DeepSeekV3/index.html                       DeepSeekV3.html
              Mixtral8x7B/index.html                      Mixtral8x7B.html
  modern/     DeepSeekV41Flash/index.html                 DeepSeekV41Flash.html
              GLM53/index.html                            GLM53.html
```

A page is one HTML file that opens with no server and no network. It holds the tsncd
bundle, the terms of every variant in one compressed table, and the selector. The
website reads each page as `diagrams/<slug>/figure.html`, and its build plugin reads the
opening form and theme of a page from the `settings` written at the top of the
`tsncd-variants` element. `websocket_transfer/standalone_page.py` writes those settings
in the form expected by the patterns of the plugin.

## The tutorial models are written with symbolic sizes

The tutorial pages bind no released sizes, and every size is a free symbol. Every model
after the first is causal. The training variant is the forward pass and the backward pass
written out, with every block of the forward pass kept and a reverse block `R[...]` for
each. Both training steps are compared with `torch.autograd` in float64 for every input
and every weight.

The tape of each training step holds what FlashAttention stores. The backward pass
rebuilds the exponentials of the scores from the saved queries and keys, through
`para/algebra/recompute_contraction_slots.py`, so no saved array holds a number for
every pair of a query and a key. The weighted model saves its keys and values at
`[x, d]` and reads them through the mask again in the backward pass, through
`para/algebra/store_operands_of_views.py`. Its training step is derived from the model
as built, with the mask after the projections, and its forward variant draws the
CausalSlide. [[Recomputing the Exponent in the Backward Pass]] states both rewrites.

| notebook | validator | page | what it shows |
|---|---|---|---|
| `tutorial/Attention.ipynb` | `tutorial/validate_attention.py` | `tutorial/Attention/`, Forward and Training | scaled dot-product attention over queries `q` and keys `x`, the softmax written out with its division moved past the weighted sum, and the training step, whose forward pass saves the five arrays stored by FlashAttention and whose backward pass rebuilds the exponentials of the scores from the queries and the keys and computes the row statistic of FlashAttention from the output |
| `tutorial/AttentionWithWeightsAndResidual.ipynb` | `tutorial/validate_attention_with_weights_and_residual.py` | `tutorial/AttentionWithWeightsAndResidual/`, Forward and Training | causal self-attention with $W^{Q}$, $W^{K}$, $W^{V}$, $W^{O}$ and a residual connection, the model as built beside the CausalSlide of its figures, the weights grabbed from slots named after them, and the training step derived from the model as built, which keeps the three blocks, gives each an `R[...]` block, saves six arrays with the keys and the values at `[x, d]`, and reads them through the mask again in the backward pass |
| `tutorial/MultiHeadAttention.ipynb` | `tutorial/validate_multi_head_attention.py` | `tutorial/MultiHeadAttention/`, Forward | causal multi-head attention, with every operation of the scaled dot-product attention broadcast over the heads, checked in PyTorch against the sum over the heads |
| `tutorial/GroupedQueryAttention.ipynb` | `tutorial/validate_grouped_query_attention.py` | `tutorial/GroupedQueryAttention/`, Forward | causal grouped-query attention with `h'` groups of `g` query heads as Mixtral-8x7B writes it, checked in PyTorch against the sum over the groups and against the released pairing that repeats the keys and the values |

`check_gradients_in_torch.py` runs a training step in PyTorch, with rules for the causal
mask, its transpose, and a sum or a softmax over a guarded slot axis.

## The classic models carry a decode form and a cached pass

Each classic page has the four variants of the groups Decode and Cached. Each cached pass
is derived by `caching/algebra/derive_cached_pass.py`, and each page draws the placement
of the caches used by the released code. Every unquantised variant is derived. Every mask is
a guarded slot axis read at `i_x - i_w`, per [[Padding and Masks as Sparse Axes]], and no
model holds a mask operator.

| notebook | validator | modules in `notebooks/classic/` | what it shows |
|---|---|---|---|
| `classic/AttentionIsAllYouNeed.ipynb` | `classic/validate_attention_is_all_you_need.py` | `attention_is_all_you_need.py`, `quantised_attention_is_all_you_need.py`, `cached_attention_is_all_you_need.py`, `attention_is_all_you_need_page_variants.py` | the encoder-decoder transformer of Vaswani et al. (2017) at its base size, checked against the paper by section and against tensor2tensor. It is held in FP32 on every real value and INT32 on the token identifiers with no cast, as tensor2tensor runs it. Its cached pass is derived from the decoder alone, and its caches hold the keys and the values after `W^{K}` and `W^{V}`, as tensor2tensor's do. The encoder and the projections of the encoded input are left in every pass, and the notebook says so |
| `classic/Mixtral8x7B.ipynb` | `classic/validate_mixtral_8x7b.py` | `mixtral_8x7b.py`, `quantised_mixtral_8x7b.py`, `cached_mixtral_8x7b.py`, `mixtral_8x7b_page_variants.py` | Mixtral-8x7B, checked against mistral-inference and the released `config.json`. It is held in BF16 and computes in FP32 where mistral-inference upcasts. Its cached pass caches the turned keys and the values, and the notebook tabulates the six placements of the caches of one layer with their cost on an H100, per [[Operation Counts and Machine Rates]] |
| `classic/DeepSeekV3.ipynb` | `classic/validate_deepseek_v3.py` | `deepseek_v3.py`, `released_deepseek_v3.py`, `quantised_deepseek_v3.py`, `cached_deepseek_v3.py`, `deepseek_v3_page_variants.py` | the released DeepSeek-V3 at the sizes of `config_671B.json`, whose FP8 weights the released code dequantises to BF16 before every product. Its cached pass caches the normalised latent and the turned key with the up-projections absorbed into the queries, as the absorb mode of the released code does |

## The modern models follow their released inference code

| notebook | validator | package | variants |
|---|---|---|---|
| `modern/GLM53.ipynb` | `modern/validate_glm53.py` | `notebooks/sota/GLM53/`, with `slide_causal_reads.py` and `assemble_page_variants.py`, and `notebooks/caching/CachedGLM53/derive_cached_glm53.py` | Decode and Cached, each quantised and unquantised, the unquantised variants derived |
| `modern/DeepSeekV41Flash.ipynb` | `modern/validate_deepseek_v41_flash.py` | `notebooks/sota/DeepSeekV41Flash/` | Decode, quantised and unquantised, the unquantised variant embedded |

The GLM-5.3 notebook draws the model in the CausalSlide, with the read back from each
query standing once at the copy of the hidden state in every attention mode and once
inside the indexer. It draws the quantisations of the FP8 checkpoint as `transformers`
runs it, and the pass over new tokens derived from the model, whose caches hold the
normalised latent, the turned key and the indexer key, 47,616 values per token, in BF16
as the dynamic cache of `transformers` holds them. The DeepSeek-V4.1-Flash notebook draws
the text-only model of [[SOTA Model Notebooks]] part by part in the CausalSlide form with
its quantisations, and its validator compares the quantised model taken through
`strip_quantisations` with the model in the reals. Its page holds the two variants of
the group Decode and no cached pass.

## Every axis in a legend carries a code name

Each validator checks that every row of the legend of every variant carries the code
name of its axis, and the code name of its size where the size is one named symbol, per
[[Code Forms]]. The classic and modern validators read the legends through
`notebook_diagrams.page_variant_legends`, which returns them as the page carries them
without computing the expansions of the inspection boxes. The validators of the classic
models and of GLM-5.3 also check that every named view of every variant opens an
inspection box, per [[Advanced Display]].

## Rewriting a page after a change

A page holds the tsncd bundle it was written with, so every page is written again after
`npm run build` in the tsncd checkout. `rewrite_website_pages.py` runs each website
notebook with every cell under its declared mode, which writes the page and leaves the
notebook file as it is, or runs only the notebooks named on its command line.

```bash
python notebooks/website/rewrite_website_pages.py
python notebooks/website/rewrite_website_pages.py notebooks/website/classic/Mixtral8x7B.ipynb
```

It prints how many pages were rewritten and names each notebook that failed, and exits
non-zero when one did. `websocket_transfer.headless.find_dist` finds the bundle, per
[[Diagram Display]].

## Gaps

- A wrap width that leaves every block whole lies in a window a few tens of pixels wide,
  and the window moves whenever a block of the figure changes width. Each variant's width
  was found by writing the page at a sweep of widths and counting the fills of its
  blocks, and a change to a model can leave a block split across two rows.
- The cached pass of the transformer of *Attention Is All You Need* runs the encoder and
  the projections of the encoded input in every step, per
  [[Deriving Caches by Dragging the New Tokens]] and [[Open Gaps]].
- No cached pass of DeepSeek-V4.1-Flash has been derived, so its page has no Cached
  group.

## See also

- [[Notebooks]] — every other notebook, and how to run one
- [[SOTA Model Notebooks]] — the DeepSeek-V4.1-Flash and GLM-5.3 packages
- [[Caching Between Passes]] — the cached passes drawn by the pages
- [[Diagram Display]] — `show_page_variants`, the page folders and the redirect
- [[Diagram Wire Format]] — the embedded form of a page of variants
- [[Stripping Quantisations]] — the dequantisation of a derived variant
