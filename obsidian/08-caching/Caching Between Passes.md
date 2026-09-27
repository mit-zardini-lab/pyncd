---
tags: [layer/caching, concept]
code: caching/data_structure/Caching.py, caching/registries/standard_expansions.py, caching/registries/cache_wording.py, caching/algebra/cache_contents.py, caching/validate_caching.py, notebooks/caching/CachedGLM53/, notebooks/display/explain_cached_reads.py
status: partly implemented
written: Claude Opus 5.5 (1M context), effort 40, on 2026-09-27, from the private note of 2026-09-25.
---

# Caching Between Passes

## A cache keeps the arrays of the earlier passes

A model generating text runs one pass for every token appended to the text. The first pass reads
the prompt, and every later pass reads the tokens appended since the pass before. An
attention reads the keys of every earlier token, and those keys were computed by the
passes that read those tokens. A cache keeps them, so that no pass computes them a
second time.

In this vault the word cache names the arrays kept by a pass for the passes after it, and
nothing else.

[[Deriving Caches by Dragging the New Tokens]] derives the caches of a pass from the
uncached model, and states where a cache can stand and what each placement costs. This
note states the operator used by the derived passes, its expansion, and the caches of
the public models.

## The files of the caching package

| file | what it holds |
|---|---|
| `caching/data_structure/Caching.py` | the operator `Caching`, `cached_token_axis`, and the functions reading the token position, the cached axis and the earlier tokens off a cache |
| `caching/registries/standard_expansions.py` | the standard expansion of a `Caching`, on a tape slot named after the cache |
| `caching/registries/cache_wording.py`, `cache_wording.json` | the formula and the sentence shown by an inspection box over a cache |
| `caching/algebra/cache_contents.py` | the caches of a term in the order they run, grouped by box, and the values each keeps per token |
| `notebooks/caching/CachedGLM53/` | one pass of GLM-5.3 written by hand from its reference, with every layer reading its keys from caches |
| `notebooks/display/explain_cached_reads.py` | the rows of the inspection boxes over the two reads of the token axis written as views by a derived pass |
| `caching/validate_caching.py` | the checks of the operator, its expansion, the counts, the derivation and the placements, one line per check |

`tsncd` mirrors the operator in `src/caching/data_structure/Caching.ts` and draws it as an
upright cylinder carrying the name of the cache, per [[Terms Mirrored in tsncd]].

## The operator saves the tokens of a pass and loads every cached token

`caching.data_structure.Caching.Caching` is the cache as one operator. The tokens of a
pass are an axis `x`, and the tokens cached by the earlier passes are an axis `P`. Every
token held by the cache after the pass stands on the axis `P + x`, an
`AxisConcatenation.ConcatenatedAxis` whose positions are those of `P` followed by those
of `x`, per [[Advanced Axis Dynamics]]. The operand of a `Caching` is the array computed
by the pass for its own tokens, and the pass saves it. The result is the same array over
`P + x`, and the pass loads it. Token `i_x` of the pass therefore stands at
position `|P| + i_x` of the result. Every other axis is the degree, because a cache keeps
each channel of a token apart from the other channels. For an operand `v` over channels
`c`, the result `y` and the array `cache` held between passes are

$$y[i_{P}, i_{c}] = \mathrm{cache}[i_{P}, i_{c}], \qquad y[\lvert P \rvert + i_{x}, i_{c}] = v[i_{x}, i_{c}], \qquad \mathrm{cache} \leftarrow y.$$

The operator is `update` of a `DynamicLayer` in the cache of `transformers`, which
concatenates the states handed to it onto its own states and returns the whole. The
user stated the design, an operator whose domain is saved and whose codomain is loaded,
on 2026-09-25. `Caching.template` builds a cache from the saved shape, the cached axis
and the position of the token axis, and raises `CachedAxisIsNotPastThenThisPass` for a
cached axis whose last part is not the token axis of the operand.

The first part of the cached axis may also be the last `|K|` earlier tokens alone, and a
sliding window of `|K| + 1` slots needs no more. The result then stands on `K + x`, and token
`i_x` of the pass stands at position `|K| + i_x` of it. A position of `K` before the
first token of the sequence holds the universal unit, per [[The Universal Unit]].
[[Deriving Caches by Dragging the New Tokens]] derives which of the two a causal read
needs from how far back it reads.

A `Caching` returns its values at the quantisation of its operand. Its rule in
`quantization/registries/operator_quantisations.py` is the rule of a view and a
concatenation, which perform no arithmetic, so a cache held in another format is a cast
in front of the cache, per [[Quantization]].

## The expansion loads the earlier tokens and appends the new ones on the tape

`caching/registries/standard_expansions.py` writes the operator out with two seeds of
[[Para Category]], on a tape slot named after the cache. A `Para.CacheGrab` loads the
array held by the cache for the earlier tokens, over `P`. An `aops.ConcatenateAxes` lays
the operand after it along `P + x`, and a `Para.CacheDrop` appends a copy of the operand
to the slot. The grab loads `|P|` tokens and the drop stores `|x|`, so the expansion
states the memory traffic of the cache in one pass. A `ParaWrap` holds both seeds as a
`Para.CacheTapeSlot`, which is where a schedule of the loads and the appends would be
written. The inspection box over a `Caching` on a page opens this expansion, with the
formula and the sentence of `cache_wording.json`.

The slot of a cache is inner, per [[Outer and Inner Tape Slots]]. A `Caching` lifted over
the heads, or kept over axes beside the tokens, therefore loads and appends one set of
entries for every index of those axes, and the inspection box over a `Caching` broadcast
over the heads draws the cache of one head. The user asked for the two seeds on
2026-09-26. The expansion before that day read the earlier tokens with a `StreamGrab` and
wrote the whole concatenation back with a `StreamDrop`, which stored every token again
in every pass.

Inside a model the innermost repeated block is the loop over the layers. A stream seed
there carries a value from one layer to the next, and a cache has to carry the value
written by the same layer in the pass before. The loop over passes is outside the
expression, and the operator names the value that crosses it without naming that loop. A
model therefore holds the `Caching`, and only a figure that defines the operator writes
the expansion. A `Caching` inside a repeated block stands for one cache per iteration, as
a weight inside it stands for one weight per iteration.

The array held by a cache grows by `|x|` tokens every pass, so the slot holds an array
over `P` when the pass begins and over `P + x` when it ends. The next pass is the same
expression with `|P|` replaced by `|P| + |x|`. A static cache, the `StaticIndexedLayer`
of `transformers`, holds one buffer of the longest sequence and writes each pass into it
in place, and no operator states it.

## Counting the caches of a term

`caching.algebra.cache_contents.caches_in_the_order_they_run` walks a term in the order
its parts compose and writes every repeated block out as many times as it repeats. A
cache inside a block of repetition 3 inside a block of repetition 18 therefore appears 54
times, once for every array held by the reference per layer. A block whose repetition is a
symbol raises `RepetitionIsNotAnInteger`. `caches_by_outermost_box` groups the same
sequence by the outermost box holding each cache, which is the attention sublayer of a
transformer layer. `entries_per_token` multiplies the sizes of every axis of the operand
but the token axis, with every symbol bound by the bodies of its name, and raises
`SizeIsNotBound` for a symbol not named by the sizes.

`caching/validate_caching.py` checks that a cache saves the tokens of the pass and loads
every cached token, that the token axis may stand at any position of the operand, that
the expansion defines the operator and wraps into one concatenation, that a cache inside
a repeated block is counted once per iteration, and that an unbound size is reported.

## A mask reads the cache at some placements and not at others

The user asked on 2026-09-26 whether a mask always follows a cache, so that the mask
could be drawn as part of the cache. It does not, and the mask stays an operation of its
own.

- The pass derived with every cache on the operand of a causal read reads each cache
  with the causal view alone. The cached passes of the transformer of *Attention Is All
  You Need* and of Mixtral-8x7B stand there.
- The other placements put operations between the cache and the mask. Of the 26
  placements of DeepSeek-V3's attention, the one with nothing computed over the cache is
  read by the mask alone. Each of the others is read by an operator on the path from the
  copy of the hidden state to the mask, such as `W^{UK}`, `W^{UV}`, the RMSNorm, the
  rotary box, a concatenation or a repeat, and the mask comes after it.
- The cached GLM-5.3 reads its latent cache with `W^{Kb}` and `W^{Vb}` and its turned key
  with a repeat over the heads. Only the cache of its indexer is read by a mask.

## The caches of the public models

Each classic and modern page of [[Website Notebooks]] draws a cached pass beside the pass
over every token, and the validator beside each notebook checks the caches.

| model | caches of one layer | values per token | checked by |
|---|---|---|---|
| the transformer of *Attention Is All You Need* | the keys and the values of the masked self-attention, directly after `W^{K}` and `W^{V}`, as tensor2tensor caches them | 6,144 over the six decoder layers | `notebooks/website/classic/validate_attention_is_all_you_need.py` |
| Mixtral-8x7B | the keys after the rotary embedding and the values, over the eight key-value heads, as mistral-inference caches them | 2,048 per layer, 128 KiB per token in BF16 over the 32 layers | `notebooks/website/classic/validate_mixtral_8x7b.py` |
| DeepSeek-V3 | the normalised latent (512) and the turned key (64), as the absorb mode of the released code caches them | 576 per layer, 35,136 over the 61 layers | `notebooks/website/classic/validate_deepseek_v3.py` |
| GLM-5.3 | the normalised latent (512) and the turned key (64) in every layer, and the indexer key (128) in the 21 layers that run an indexer | 47,616 over the 78 layers | `notebooks/website/modern/validate_glm53.py` |

Every cache of a quantised pass holds BF16, because each reference hands its cache states
in BF16. The transformer of *Attention Is All You Need* runs in FP32 throughout, so its
caches hold FP32.

## GLM-5.3 cached between passes by hand

`notebooks/caching/CachedGLM53/` writes one pass of GLM-5.3 with every layer reading its
keys from caches, from the reference implementation followed by `notebooks/sota/GLM53/`.
Every part left unchanged by the cache is imported from `notebooks/sota/GLM53/`. The
reference caches the normalised latent of 512 channels and the turned key of 64 channels
of every token in every layer, and the indexer key of 128 channels in the 21 layers that
run an indexer. The pass holds three kinds of `Caching`, `lat`, `rot` and `idx`, 177 in
all, and keeps 47,616 values per token. Three things change besides the caches
themselves.

- The rotary table is written over `P + x` and read at `|P| + i_x`, because the reference
  adds the count of cached tokens to every position of the pass.
- A query reads the cache back at `|P| + i_x - i_r`, so the distance axis has
  `|P| + |x|` positions and holds a value where `|P| + i_x - i_r >= 0`.
- The top-2048 selects over the cache.

The selection is not cached. It passes from a Full layer to the Shared layers of its
group on the tape slot `sel` within one pass, and the next pass selects again.

`cache_findings.operations_of_a_pass` reads the operations of every linear map and
contraction of the attention off the expression, symbolic in `|P|` and `|x|`. The
reference expands every cached latent by `W^{Kb}` and `W^{Vb}` in every pass, which is
`2 |c| |h| (|n| + |u|) (|P| + |x|)` operations per layer. A decode pass at a context of
1,048,576 tokens then spends 2.4 PFLOP on the expansion, more than a hundred times the
rest of the attention. Expanding only the 2,048 latents selected by each query would take
`2 |c| |h| (|n| + |u|) |x| |s|`, 4.69 TFLOP. The two orders take the same operations at
`|P| + |x| = |x| |s|`. A prefill with an empty cache takes fewer operations in the order
of the reference, 18.8 TFLOP against 38.4 PFLOP for 8,192 tokens, so the order with fewer
operations depends on the pass.

The reference caches the narrowest array of the key-value path, 576 values per token and
layer, against 6,144 in the hidden state and 32,768 once expanded. One sequence of
1,048,576 tokens holds 49.9 GB at one byte per value and 99.9 GB at two.

The pass drawn by the public page is derived from the uncached model rather than written by
hand, by `notebooks/caching/CachedGLM53/derive_cached_glm53.py`, and
`validate_glm53.py` checks that the pass written by hand caches arrays of the same widths
in every layer and takes as many operations as the derived pass.

## Gaps

- The order with the gather before the expansion is counted from the expression and not
  written as one. The gather is an `IndexSelect` at a position held as data, so the read
  crawl, which moves an affine read, does not move it past the expansion. The order that
  multiplies every query by `W^{Kb}` and every result of the core by `W^{Vb}` is derived
  on DeepSeek-V3's attention, per [[Deriving Caches by Dragging the New Tokens]], and has
  not been applied to GLM-5.3. [[Open Gaps]] records both.
- A `Caching` states the dynamic cache alone. No operator states a static buffer of the
  longest sequence written in place.
- A value computed once per sequence and loaded by every later pass has no operator,
  because a `Caching` appends the tokens of each pass to what it holds. [[Open Gaps]]
  records the consequence for the transformer of *Attention Is All You Need*.

## See also

- [[Deriving Caches by Dragging the New Tokens]] — the derivation of a cached pass, and the placements
- [[Operation Counts and Machine Rates]] — the operation count and the machine rates read by the cost of a placement
- [[Para Category]] — the seeds of the expansion
- [[Outer and Inner Tape Slots]] — why the slot of a cache is inner
- [[Advanced Axis Dynamics]] — the concatenated axis `P + x`
- [[Website Notebooks]] — the pages that draw each cached pass
