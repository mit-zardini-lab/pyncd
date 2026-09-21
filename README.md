# pyncd

```python
import construction_helpers as ch # Needed for algebraic manipulation
import data_structure.Category as cat
import data_structure.Operators as ops


qk_matmul = ops.Einops.template('q h d, x h d -> h q x')
softmax = ops.SoftMax.template()
mask = ops.WeightedTriangularLower.template()
sv_matmul = ops.Einops.template('h q x, x h d -> q h d')
attention_core = qk_matmul @ softmax @ mask @ sv_matmul
```
 ![The Pythonic algebraic expression above generates an aligned attention algorithm.](_guide/figures/alignment.png)

## Description

This is a package for formally expressing deep learning models based on [Neural Circuit Diagrams](https://openreview.net/forum?id=RyZB4qXEgt), [FlashAttention on a Napkin](https://openreview.net/forum?id=pF2ukh7HxA), [Spherical Attention](https://arxiv.org/abs/2505.09326) and a [GPU Mode presentation](https://www.youtube.com/watch?v=hAoY2bpRIKg). The main goal of this package is to provide a simple and intuitive way to define and visualize deep learning models, while also allowing for formal reasoning about their properties. In `data_structure`, you will find a high-level implementation of the structural aspects of deep learning models.

The other folders provide utilities. These are;

 - `construction_helpers`: Allows models to be defined via operator overloading, `@` (for sequential composition), `*` (for parallel "products") and `>>` (for batch lifting). When using `@`, axes are automatically aligned.
 - `data_transfer` and `websocket_transfer`: These packages provide JSON encoding and communication over WebSockets, integrating with the [`tsncd`](https://github.com/mit-zardini-lab/tsncd) package for displaying diagrams.
 - `torch_compile`: This package allows algebraic descriptions to be converted into PyTorch modules.
 - `display`: This package allows for textual display of algebraic expressions.
 - `graphs`: This package implements the mathematical process of [converting morphisms in a symmetric monoidal category into hypergraphs](https://arxiv.org/pdf/2305.08768), which opens up flexible algebraic manipulation in the future.
 - `algebra`: General-purpose rewrites of an expression, such as factoring an operator's reindexings out of it and putting an expanded expression back together.
 - `solver`: Symbolic algebra over the `Numeric` terms that give axes their sizes, including the derivative of a numeric.
 - `quantization`: The number format and the size in bits a value is held in, the conversion between two of them, and the pass that writes a quantisation onto every wire of a model.
 - `advanced_axis_dynamics`: A read that falls outside an axis leaves an axis only some of whose positions hold a value. This folder derives the affine form saying which positions those are, and carries it through further reads.
 - `deepseek`: The operators DeepSeek's models need, including a complex datatype with rotary position embeddings and the top-k gate behind a mixture of experts.
 - `para`: The backward pass, derived from the forward one rather than declared.
 - `term_utilities`: Reading structure back out of a term, the code references a block carries, and the configuration that gives a model's free numerics concrete sizes.
 - `utilities`: The collection types the terms are built from, and the wording files a figure's descriptions are read from.
 - `validations`: Every validator and notebook as a target, the import graph that says which targets a modified file reaches, and the runner that executes them at once.
 - `agent_display`: An expression as an SSA listing, which is easier to read as text than the diagram is.
 - `notebooks`: One folder per feature. Each notebook demonstrates a feature and asserts what it claims, so it doubles as a test.
 - `example_notebooks`: Short introductory notebooks, each one runnable, and `minimum_working_example.py`.
 - `obsidian`: An Obsidian vault documenting the package, one note per module or concept.

These utilities build on the core data structure. They feed into a "web" of tools that allow for algebraic manipulation, diagrammatic visualization, and execution of deep learning models. For instance, we can compose from algebraic constructs to the Python data structure, to PyTorch or diagrammatic visualizations. Given the underlying mathematical structure of the data structure, we can imitate mathematical transforms such as product categories to hypergraphs.

*The modularized tools in this package generate a "web" of features which integrate into each other.*
![alt text](_guide/figures/the_web_tsncd.png)

 ## The Structure
 We implement mathematical expressions with `Term`s. We keep everything in a high-level structure, and leave evaluation to specific tools such as the Torch compiler or TypeScript diagramming. Deep learning models and their components are morphisms in the `BroadcastedCategory`. (Product) Categories are compositional structures that allow for components called **morphisms** to be sequentially **composed** and placed into parallel **products**, forming new morphisms. Composition is anchored by **objects**. We also have a special morphism to **rearrange** objects in a product.
 
 The structure of a deep learning model consists of these constructed terms ultimately referencing seed morphisms. In the case of the `BroadcastedCategory`, representing deep learning models, the seed morphisms are single, broadcasted operations called `Broadcasted`, and the objects are `Array`s.

 The broadcasting semantics of this package are defined through `Weave`s and the `StrideCategory`. Weaves indicate which axes are tiled or form part of the "target" operation. The output along indexes of tiled axes are defined relative to indexes along the inputs by passing through a morphism of the `StrideCategory`, which corresponds to an affine transform. This relationship may be direct, for example, the `i, j` output index may correspond to the `i, j` input index, or they may take a stride manipulation. These are always affine. This allows more complex patterns to be enforced. The `i, j` output index may correspond to the `j, i` input index, enforcing a transpose. The `i` output index may correspond to the `i, i` input index, giving a diagonalization. Or, the `x, k` output index may correspond to the `x + k` input index, giving a convolution.
 
  *Here, the p0, p1 indexes of the output correspond to the reindexed locations along the input.*
 ![The p0, p1 indexes of the output correspond to the reindexed locations along the input.]( _guide/figures/broadcast_weave.png)

# Setup
 - ***This package requires Python 3.13.***

 - For diagrammatic visualization, the [`tsncd`](https://github.com/mit-zardini-lab/tsncd) package is required. We run `python run_server.py` (requires `pip install websockets`) in an independent terminal. `tsncd` carries the same relay under `npm run server`, which needs no Python package and answers the same messages. Either will do. Only one of them may hold port 8765. When the server is running, the browser connects on refresh. As in `example_notebooks/Transformer.ipynb`, we connect from Python, updating the data in the server. The wire format between the two packages is documented in
[`obsidian/05-backends/Diagram Wire Format.md`](obsidian/05-backends/Diagram%20Wire%20Format.md).

# Diagrams in the notebook

`send_morphism` puts a diagram in the browser window, and nothing of it survives
into the notebook. To get the picture back inline, use `capture_morphism`, which
renders exactly the same way and hands the image back. The image is saved with
the notebook and survives export:

```python
from websocket_transfer.capture import capture_morphism

await capture_morphism(attention)                        # inline PNG
await capture_morphism(attention, save_to='fig.svg')     # and on disk
```

By default the capture also lands on screen, so the diagram you are looking at
is the one you get. The capture includes this call's display settings. Pass
`disturb_display=False` to render off-screen instead and leave the browser
exactly as it was:

```python
await capture_morphism(variant, disturb_display=False)
```

Either way a tsncd page must be open, since that page is what draws. There is
no way around this: the diagram's geometry is CSS layout plus measured text, so
it takes a real browser engine. `jsdom` and friends report every box as zero.

To rebuild figures without one, `websocket_transfer/headless.py` drives its own
browser. The rebuild does not depend on which tab happened to be focused:

```python
from websocket_transfer.headless import save_figures

await save_figures({
    'attention':   attention,
    'convolution': convolution,
}, width=1400)
# -> ./outputs/attention.pdf, ./outputs/convolution.pdf
```

Each key is a name rather than a path. Each becomes a file under the target
directory, which defaults to `./outputs`. The default directory is beside the
notebook that made them. Pass a directory as the second argument to put them
elsewhere. PDF is the default because it is what a paper wants: vector,
embedded text, and *smaller* than the equivalent PNG. A name may carry its own
extension (`'fig1.png'`) to override the format for that one figure.

`width` is the wrap width in px, and so sets the figure's proportions rather
than its scale. 750 (the default) gives a 1.43 aspect ratio on the transformer,
1400 gives 3.29. Screens want the former; a figure spanning a paper's text block
usually wants the latter.

Headless capture needs `pip install -r requirements-headless.txt`,
`playwright install chromium`, and a built tsncd bundle (`npm run build`).

 - The `example_notebooks/minimum_working_example.py` both hosts a server and has command line inputs for various components which are sent to the server for visualization.

 - For compiling deep learning models, [PyTorch](https://pytorch.org/) and [Einops](https://einops.rocks/) (`pip install einops`) are required.

# Displaying a model

 - We write the model as an expression. `notebooks/base_features/BuildingAModel.ipynb`
states each rule of construction beside the code that follows it, and asserts what it
claims.

 - We draw the expression through `notebooks/display/notebook_diagrams.py`. A notebook
builds one `DiagramSettings` at the top of its setup cell and passes it to every
`show_diagram` call, so the mode is set in one place. There are six modes. `INLINE`
renders the figure and embeds it in the cell, so it is saved with the notebook, and it
draws in a browser the notebook starts itself or in the open page. `BROWSER` sends the
figure to the open page and embeds nothing, so it needs a relay on port 8765, from
`python run_server.py` or `tsncd`'s `npm run server`, and a page open from `npm run dev`.
`HTML` writes the figure as one page that needs nothing, since it opens from the disk
with no server and no network. `DUMP` writes the term to JSON. `LISTING` prints the
`agent_display` listing and draws nothing. `OFF` skips the diagram.

 - We quantise the model with `quantization/processing/quantise_model.py`, which writes a
quantisation onto every wire and a conversion wherever an operation requires another
quantisation. `quantization/algebra/strip_quantisations.py` takes the quantisations off
again and returns the model in the reals.

 - We read the expression as text with `agent_display`, which renders it as an SSA
listing, and we compile it to a PyTorch module with `torch_compile`.

 - `notebooks/sota/DeepSeekV41Flash.ipynb` is the worked example, and the model it builds
lives beside it in `notebooks/sota/DeepSeekV41Flash/`. We run it as a listing with
`python notebooks/execute_notebook.py notebooks/sota/DeepSeekV41Flash.ipynb`, which needs
no browser. For the figures, we open it in VS Code with the kernel started at the
repository root, which `.vscode/settings.json` arranges.

 - We check the repository with `python validate_repository.py` and
`python validations/run_validations.py --all`.
