---
tags: [layer/practice, reference]
code: notebooks/, example_notebooks/
status: evolving
---

# Notebooks

Written by Claude Opus 5 (1M context), effort high.

Notebooks are how this project explains itself. The narrative lives in them, and the
modules are extracted afterwards. A notebook demonstrates a feature and serves as its
test, so everything it claims is asserted in text and holds with the diagrams turned off.

There are two folders. `example_notebooks/` holds the short introductions a new reader
starts with. `notebooks/` holds the long ones, one folder per feature, each opening with
a title cell naming the feature it introduces.

## `example_notebooks/`

Each of these builds one construction and draws it, in a handful of cells. They are the
shortest path from an empty file to a diagram. Each draws through
`notebooks/display/notebook_diagrams.py` and is a target of the registry [[Validation]]
describes, so `validations/run_validations.py` runs it with the diagrams off beside the
notebooks under `notebooks/`.

| notebook | what it builds |
|---|---|
| `Attention.ipynb` | attention as three operators composed with `@`: the score contraction, the softmax and the value contraction |
| `AttentionPoster.ipynb` | the same expression drawn at the proportions the poster in `_guide/` uses |
| `Convolution.ipynb` | a convolution as an affine reindexing followed by a contraction, which is why the package needs no convolution operator |
| `Transformer.ipynb` | a decoder-style transformer: masked multi-head attention and a feed-forward layer, boxed into blocks and repeated |
| `AttentionIsAllYouNeed.ipynb` | the architecture of Vaswani et al. (2017), with both towers and the cross-attention between them |
| `AlgebraicTransformer.ipynb` | the same stack written with the blocks and the fill colours a figure uses |
| `DeepSeekV3.ipynb` | DeepSeek-V3, with its mixture of experts written as a `TopK` and the expert projections that read it |
| `Results.ipynb` | the four things the package does with an expression: automatic axis alignment, generating a configuration and applying it, compiling to a PyTorch module, and converting to a hypergraph and back |

## `notebooks/base_features/`

`BuildingAModel.ipynb` states each rule of construction beside the code that follows it
and asserts what it claims. It is where a correction on how a model is expressed is
recorded, per [[Representing Models]], and it is the first thing to read before writing
an expression.

## `notebooks/sota/`

`DeepSeekV41Flash.ipynb` is one account of DeepSeek-V4.1-Flash restricted to a prompt of
text. Forty-six cells run in the reading order of the forward pass, each section drawing
one part of the model and explaining it against the released code, and every figure is the
quantised model or a part of it, so each wire carries the quantisation the released code
holds its array in. One cell near the end takes every quantisation back off and draws the
model again. Its model lives in the package beside it, `DeepSeekV41Flash/`, and its claims
live in `quantization/validate_quantization.py` and in the three `validate_*.py` scripts
of that package rather than in the notebook, which is the exception [[Validation]]
describes. [[SOTA Model Notebooks]] states how the package is laid out and what each
validator asserts.

## Running them

- The kernel must be rooted at the **repository root**. `.vscode/settings.json` sets
  `jupyter.notebookFileRoot`. Launching `jupyter lab` from inside the folder will not
  resolve the imports, and `notebooks/fix_notebook_dir.py` does the same job outside VS
  Code.
- **Diagrams need a browser**: the headless renderer, or `python run_server.py` plus a
  `tsncd` page open. `tsncd` carries the same server under `npm run server`, and only one
  of the two may hold port 8765. Every notebook under `notebooks/` and under
  `example_notebooks/` draws through `notebooks/display/notebook_diagrams.py`. A notebook builds one `DiagramSettings` at
  the top of its setup cell and passes it to every call, so the mode is changed in one
  place. Work in `BROWSER`, at about 15 ms, and use `INLINE` for the run that gets
  committed. An `INLINE` diagram takes 166 ms for the transformer figure once the headless
  browser is warm, and about 1.7 s for the first one, which starts it. See
  [[Diagram Display]].
- An agent executing a notebook in the background runs it through
  `notebooks/execute_notebook.py`, which sets `PYNCD_DIAGRAMS=listing` for the kernel, and
  `show_diagram` then prints the [[Agent Display]] listing in place of every diagram.
- Nearly all of a diagram's cost is the browser laying it out. Building an expression is
  tens of milliseconds, and a whole-model construction takes seconds.
- **A notebook cell never prints an [[Agent Display]] listing.** The user ruled on
  2026-09-16 that a listing is a diagram mode for the agent that executes the notebook
  and is not useful to a person. A cell calls `show_diagram`, and `DiagramMode.LISTING`
  or `execute_notebook.py` prints the listing in its place.
- **An inline equation and its sentence go on one markdown line**, or the equation
  stands alone on its own line. A hard wrap that starts a line with `$` leaves the LaTeX
  source unrendered in the notebook, which the user found in
  `notebooks/sota/DeepSeekV41Flash.ipynb` on 2026-09-16. The rule is in
  `CLAUDE.md` with the rejected wrap beside its replacement.

```bash
python notebooks/execute_notebook.py notebooks/base_features/BuildingAModel.ipynb
python notebooks/execute_notebook.py notebooks/sota/DeepSeekV41Flash.ipynb --diagrams off
```

`--output <notebook path>` saves the outputs after every cell succeeds, and combined with
`--diagrams inline` it embeds the rendered diagrams. The output path may be the input
path. Omitting `--output` leaves the notebook file untouched, so the mode a person set at
the top of the setup cell survives the run.

## Supporting modules

`notebooks/display/` holds the code the notebooks import. `notebook_diagrams.py` is the
mechanism, and everything beside it is a presentation applied to a term just before it is
drawn.

| module | what it does |
|---|---|
| `notebook_diagrams.py` | `DiagramSettings`, `DiagramMode` and `show_diagram`, which is the one route to a figure |
| `notebook_listings.py` | the listing a figure is replaced by under `DiagramMode.LISTING` |
| `remember_drawn_blocks.py` | which bodies a figure draws, given what the figures before it drew |
| `block_recycling.py` | whether a block's body is recycled through a hypergraph before it is drawn |
| `tape_presentation.py`, `tape_naming.py` | how a tape slot is drawn and how it is labelled |
| `loop_initializers.py` | whether the initializer of a loop variable is drawn |
| `clean_quantisation_labels.py` | labelling an array only where its quantisation changes |
| `cast_presentation.py` | whether a cast is drawn as a chevron or as no glyph at all |
| `axis_sizes.py` | where the size of a sized axis is written |
| `advanced_display.py`, `explain_operators.py`, `explain_reindexings.py` | the legend and the inspection boxes of an interactive figure, per [[Advanced Display]] |
| `expand_with_parameters.py` | what stands for a weight in the expansion an inspection box draws |
| `attention_figures.py`, `sota_figures.py` | the figures a model notebook repeats |
| `display_wording.py`, `display_wording.json` | the sentences the display writes, held once |

`notebooks/execute_notebook.py` runs a notebook with its kernel at the repository root, and
`notebooks/fix_notebook_dir.py` does the same for a kernel started elsewhere.

## See also

- [[Repository Map]] — what is live and what is dead
- [[Validation]] — the checks a notebook is one of
- [[SOTA Model Notebooks]] — the one whole-model notebook and its packages
