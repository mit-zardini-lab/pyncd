# Claude Opus 5.5 (1M context), effort 40.
'''MiMo-V2.6-Pro as a morphism in Br, one module per mechanism of the reference
implementation and one per piece of wiring that joins the mechanisms into a model.

Xiaomi ships the reference implementation in the checkpoint, as the remote code
`modeling_mimo_v2.py` that `transformers` loads through `auto_map`, and
`reference_links` pins every link to it. `notebooks/website/modern/MiMoV26Pro.ipynb`
draws the model part by part in the reading order of the forward pass, explains each
part against the reference, derives the pass over new tokens, and writes the model and
the pass into one interactive page. The notebook holds the prose and the figures, and
`notebooks/website/modern/validate_mimo_v26_pro.py` holds the claims it makes.

    block_titles_and_descriptions
                            every block title, block description and inspection-box
                            sentence of this package, held in
                            `block_titles_and_descriptions.json` and loaded into the
                            dataclass `TEXT`, imported as `text`
    reference_links         the pinned links into the remote code and the
                            configuration of the checkpoint, and into the files of
                            `transformers` 5.3.0 that decide what a sliding window
                            reads and keeps
    released_constants      the two rotary bases, the two epsilons and the value
                            scale, with the value each has in the reference
    declared_axes           every structural axis and the arrays carried by the wires
    rotary_embedding        the box turning the first 64 channels of a query head or a
                            key head in the layout of `rotate_half`, one for each table
                            of turns
    grouped_query_attention the queries, the keys and the values, the views reading
                            the earlier tokens, the two attention cores and the output
                            projection
    attention_modes         full attention and sliding window attention, each one box
    feed_forward            the SwiGLU map, and the dense MLP of layer 0 as one box
                            computed once per token
    mixture_of_experts      the router box, the 384 routed experts and the combine, as
                            one box computed once per token
    layer_stack             the pre-norm residual, the 70 layers, and the count of the
                            layers of a term
    whole_model             the embedding, the layer stack and the output logits, the
                            released sizes, and the lookups by which a figure asks for
                            a part
    slide_causal_reads      the model in the CausalSlide, the form a figure is drawn in
    derive_cached_mimo_v26_pro
                            the pass over new tokens, derived from the model by
                            `caching/algebra/derive_cached_pass.py`
    operator_explanations   the tables that fill the inspection box over an operator,
                            a weight or a view
    assemble_page_variants  the two forms the interactive page switches between
'''
