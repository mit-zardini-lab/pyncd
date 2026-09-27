# The Notebooks Published on the Website

*Written by Claude Opus 5.5 (1M context) at effort 40, 2026-09-27.*

Each model shown on the lab's diagrams page has one public notebook here, one validator
beside it, and one interactive page. A notebook holds the prose and the figures and makes
no checks. Every fact its prose states is checked by the validator beside it, one `check_`
function per claim, in the order the notebook states them. `validations/` discovers each
`validate_*.py` and each notebook under `notebooks/website/` as a target of its own, so
`python validations/run_validations.py --name website` runs all of them.

The last cell of each notebook writes its page through
`notebook_diagrams.show_page_variants`. A page carries every variant of its model
in one file, and a selector above the figure switches between them. The selector copies the
model selector of the website's diagram viewer. It stays on the page when the website hides
tsncd's own form and theme buttons, because the website has no selector of its own for the
variants. The variants share one compressed table of values, so a subterm or a description
that two variants hold is stored once.

An unquantised variant is either derived in the browser or embedded. A derived variant is
the quantised variant with the dequantisation functor applied to it by tsncd while the page
shows "Applying Dequantization Functor...". The functor removes every quantisation, turns
every cast that no longer converts into an identity, and then removes identities from the
leaves upwards, so that a composition of identities and a block around an identity become
the identity. `quantization/algebra/strip_quantisations.py` states the same functor in
Python, and each validator checks that it turns the quantised model into the unquantised
one. An embedded variant carries its own term, which is done where the functor would not
give the model in the reals.

## The pages are written into `output/`, one folder per page

The folders under `output/`, beside this file, mirror the folders of the notebooks, and
`website_output.py` declares them. Each notebook passes the folder of its group as
`page_directory` in the settings of its page. A page is written as `index.html` in a
folder named after the model, so the address a reader shares ends in the name of the
folder, as `.../classic/Mixtral8x7B/` does. A page named `<Model>.html` stands beside
each folder and sends a reader to the folder. It keeps the query and the hash of the
address, so `Mixtral8x7B.html?variant=cached-quantised` opens `Mixtral8x7B/` on the same
variant. `obsidian/05-backends/Diagram Display.md` gives the redirect in full.

```
output/
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

## The tutorial models are written with symbolic sizes

The tutorial pages bind no released sizes. Their supporting modules sit beside the
notebooks in `notebooks/website/tutorial/`. `express_attention.py` builds the four
expressions, `derive_training_step.py` derives the training step, `tutorial_pages.py`
holds the page variants, `tutorial_wording.json` the titles and descriptions, and
`check_gradients_in_torch.py` compares the training steps with `torch.autograd`.

| notebook | validator | page under `output/` | variants |
|---|---|---|---|
| `tutorial/Attention.ipynb` | `tutorial/validate_attention.py` | `tutorial/Attention/` | Pass: Forward, Training |
| `tutorial/AttentionWithWeightsAndResidual.ipynb` | `tutorial/validate_attention_with_weights_and_residual.py` | `tutorial/AttentionWithWeightsAndResidual/` | Pass: Forward, Training |
| `tutorial/MultiHeadAttention.ipynb` | `tutorial/validate_multi_head_attention.py` | `tutorial/MultiHeadAttention/` | Forward only |
| `tutorial/GroupedQueryAttention.ipynb` | `tutorial/validate_grouped_query_attention.py` | `tutorial/GroupedQueryAttention/` | Forward only |

The training variant is the forward pass and the backward pass written out, with every
block of the forward pass kept and a reverse block `R[...]` for each. Both training steps
are compared with `torch.autograd` in float64 for every input and every weight.

The tape of each training step holds what FlashAttention stores. The backward pass
rebuilds the exponentials of the scores from the saved queries and keys, through
`para/algebra/recompute_contraction_slots.py`, so no saved array holds a number for every
pair of a query and a key. The weighted model saves its keys and values at `[x, d]` and
reads them through the mask again in the backward pass, through
`para/algebra/store_operands_of_views.py`. Its training step is derived from the model as
built, with the mask after the projections, and its forward variant draws the
CausalSlide.

## The classic models carry a decode form and a cached pass

Each classic page has four variants in two groups. Decode is the whole model over every
token with no cache, and Cached is one pass over the new tokens that reads the earlier
tokens from caches. Each is drawn quantised, in the formats the released code holds its
values in, and unquantised. The cached passes are derived by
`caching/algebra/derive_cached_pass.py`, and each page draws the placement of the caches
the released code uses. The supporting modules are in `notebooks/classic/`.

| notebook | validator | page under `output/` | unquantised variants | supporting modules |
|---|---|---|---|---|
| `classic/AttentionIsAllYouNeed.ipynb` | `classic/validate_attention_is_all_you_need.py` | `classic/AttentionIsAllYouNeed/` | derived | `quantised_attention_is_all_you_need.py`, `cached_attention_is_all_you_need.py`, `attention_is_all_you_need_page_variants.py` |
| `classic/Mixtral8x7B.ipynb` | `classic/validate_mixtral_8x7b.py` | `classic/Mixtral8x7B/` | derived | `quantised_mixtral_8x7b.py`, `cached_mixtral_8x7b.py`, `mixtral_8x7b_page_variants.py` |
| `classic/DeepSeekV3.ipynb` | `classic/validate_deepseek_v3.py` | `classic/DeepSeekV3/` | derived | `released_deepseek_v3.py`, `quantised_deepseek_v3.py`, `cached_deepseek_v3.py`, `deepseek_v3_page_variants.py` |

Attention Is All You Need is held in FP32 throughout, as tensor2tensor runs it, and its
cached pass is derived from the decoder alone. Mixtral-8x7B is held in BF16 and computes in
FP32 where mistral-inference upcasts. The DeepSeek-V3 page draws the released 671B
checkpoint, whose FP8 weights the released code dequantises to BF16 before every product,
and its cached pass holds the normalised latent and the turned key, as the absorb mode of
the released code does.

## The modern models follow their released inference code

| notebook | validator | page under `output/` | variants | supporting modules |
|---|---|---|---|---|
| `modern/GLM53.ipynb` | `modern/validate_glm53.py` | `modern/GLM53/` | Decode and Cached, each quantised and unquantised; unquantised derived | `notebooks/sota/GLM53/`, with `slide_causal_reads.py` and `assemble_page_variants.py`, and `notebooks/caching/CachedGLM53/derive_cached_glm53.py` |
| `modern/DeepSeekV41Flash.ipynb` | `modern/validate_deepseek_v41_flash.py` | `modern/DeepSeekV41Flash/` | Decode, quantised and unquantised; unquantised embedded | `notebooks/sota/DeepSeekV41Flash/` |

The unquantised DeepSeek-V4.1-Flash is embedded. The released code rounds its cached
latents, the entries of its candidate pool, and the keys and queries of its indexer
through FP8 and FP4 in place, so the model written in the reals keeps those roundings, in
the blocks titled *FP8 Round Trip of a Window Latent*, *FP4 Round Trip in the Indexer*
and *FP4 Round Trip of an Entry*. The functor would turn every one of those casts into an
identity, and the blocks would round nothing. Embedding the model in the reals adds about
0.5 MB to the page.

GLM-5.3 and DeepSeek-V4.1-Flash are drawn in the CausalSlide form, in which each causal
read stands at the copy that feeds the keys and the values, as the classic and tutorial
models are.

## Every axis in a legend carries a code name

Each validator checks, through `notebook_diagrams.page_variant_legends`, that every row of
the legend of every variant carries the code name of its axis, and the code name of its
size where the size is one named symbol.

## Rewriting a page after a change

A page holds the tsncd bundle it was written with. After `npm run build` in
`../tsncd`, each page is rewritten by running its notebook with every cell under
the mode it declares, which writes the page and leaves the notebook file as it is.
`rewrite_website_pages.py` beside this file does that for every website notebook, or for
the notebooks it is given:

```bash
python notebooks/website/rewrite_website_pages.py
python notebooks/website/rewrite_website_pages.py notebooks/website/classic/Mixtral8x7B.ipynb
```

The website reads each page as `diagrams/<slug>/figure.html`. Its build plugin reads the
form and the theme a page opens in from the `settings` written at the top of the
`tsncd-variants` element, which `websocket_transfer/standalone_page.py` keeps in the form
the plugin's patterns expect.
