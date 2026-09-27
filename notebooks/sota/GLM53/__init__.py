# Claude Opus 5.5 (1M context), effort 40.
'''GLM-5.3 as a morphism in Br, one module per mechanism of the reference implementation
and one per piece of wiring that joins the mechanisms into a model.

Z.ai ships no inference code with the checkpoint, and its model card names the class
`GlmMoeDsaForCausalLM` of the Hugging Face `transformers` library. The class is the
reference implementation here, read at one commit, and `reference_links` pins every
link to it. `notebooks/website/modern/GLM53.ipynb` draws the model part by part in the
reading order of the forward pass, explains each part against the reference, and
writes the model, its quantised form and the pass over new tokens to one page with an
inspection box over every block and every operator. The notebook holds the prose and
the figures. `validate_glm53.py` and `validate_quantised_glm53.py` hold the claims
made about the model and its quantised form, and
`notebooks/website/modern/validate_glm53.py` runs them for that notebook, per the
ruling of 2026-09-15 on a SOTA notebook whose model lives in a package beside it.

`notebooks/sota/GLM53/` reads every size from the configuration of the checkpoint and
carries the shared selection on a tape slot, per
`obsidian/06-practice/Representing Models.md`.

    block_titles_and_descriptions
                            every block title, block description and inspection-box
                            sentence of this package, held in
                            `block_titles_and_descriptions.json` and loaded into the
                            dataclass `TEXT`, imported as `text`
    reference_links         the pinned links into `modeling_glm_moe_dsa.py`, into
                            `configuration_glm_moe_dsa.py` and into the `config.json`
                            of the checkpoint
    released_constants      the rotary base, the three epsilons and the route scale,
                            with the value each has in the reference
    declared_axes           every structural axis, the two selection counts, the arrays
                            carried by the wires and the tape slot of the shared
                            selection
    rotary_embedding        the box turning the 64 turned channels of a query head or a
                            key, and the box turning the first 64 channels of an
                            indexer vector
    lightning_indexer       the indexer queries and keys, the keys read back from every
                            query, the scoring box computed once per query and the
                            top-2048
    multi_latent_attention  the queries through their low rank, the keys and values
                            expanded from the latent, the reads at the selected tokens,
                            the core computed once per query and head, and the output
                            projection
    attention_modes         the Full mode, which runs its own indexer, and the Shared
                            mode, which grabs the selection of the Full layer of its
                            group
    feed_forward            the SwiGLU map, and the dense MLP of layers 0 to 2 as one
                            box computed once per token
    mixture_of_experts      the router box, the 256 routed experts, the shared expert
                            and the combine, as one box computed once per token
    layer_stack             the pre-norm residual, the 78 layers and the IndexShare
                            groups, and the count of the layers of a term
    whole_model             the embedding, the layer stack and the output logits, the
                            released sizes, and the lookups by which a figure asks for
                            a part
    operator_explanations   the tables that fill the inspection box over an operator,
                            a weight or a view
    quantised_whole_model   the model with the quantisations of the FP8 checkpoint as
                            `transformers` runs it, the policy with the line of every
                            entry, the tables the notebook shows and the rows of the
                            inspection boxes over a cast and a weight
    validate_glm53          one `check_*` function per claim about the model
    validate_quantised_glm53
                            one `check_*` function per claim about the quantised
                            model
'''
