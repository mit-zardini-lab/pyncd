'''DeepSeek-V4.1-Flash as a morphism in Br, one module per mechanism of the released
architecture and one per piece of wiring that joins the mechanisms into a model.

Written by Claude Opus 5 and by Claude Fable 5.1. The two packages this one was made of
were merged by Claude Opus 5 (1M context), effort high, on 2026-09-20.

A mechanism module states one part of the released model on its own, with the axes it
reads and the operators it applies. A wiring module states the tape slots, the boxes and
the layer plan that put the mechanisms together, and three models come out of that wiring.
`integrated_whole_model` composes `v41_flash_integrated`, which holds every mechanism and
reads a prompt of text and images. `text_only_model` composes `v41_flash_text_only`, which
is the integrated model without the image pathway and without DSpark.
`quantised_text_only_model` writes the quantisation the released model runs in onto every
wire of the text-only model, and states beside it the same model with every quantisation
stripped off again.

`notebooks/sota/DeepSeekV41Flash.ipynb` is the account of the model. It draws the
quantised text-only model part by part in the reading order of the forward pass, explains
each part against the released code, and ends with the stripped model beside it. It holds
the prose and the figures, and the four `validate_*.py` of this package hold every claim
made about the three models, which is the rule for a model written as a package beside
its notebook. `validate_quantised_text_only_model` holds the claims of the notebook
itself, and the notebook runs it in the cell after its setup cell.

The rules for writing the model are in `obsidian/06-practice/Representing Models.md`,
and `obsidian/06-practice/SOTA Model Notebooks.md` gives the rationale for it.

    block_titles_and_descriptions
                               every block title and every block description of this
                               package, held in
                               `block_titles_and_descriptions.json` and loaded into
                               the dataclass `TEXT`, imported as `text`, so that the
                               wording of a figure is changed in one file that holds
                               no Python
    released_constants         every named constant the mechanisms read, with the value
                               each has in the released configuration
    reference_links            the pinned links into the released code
    operator_explanations      the tables an inspection box is filled from, for the
                               base model and, joined with the mechanism modules'
                               tables, for the omissions notebook
    integrated_explanations    the same tables for the integrated model, united with
                               the tables of the base model
    declared_axes              every structural axis, the free symbols for the
                               selection counts, the arrays the wires carry and the
                               six tape slots of the shared attention runtime
    integrated_axes_and_slots  the further tape slots the wiring reads and writes,
                               added to the declared axes of the model
    construction_idioms        hold, route and over, the two block-operator
                               constructors, and the readers the assertions use
    custom_operations          the parameter array and the pointwise maps, each an
                               Arithmetic over a formula
    rotary_embedding           the seven arrays the released model turns, each one
                               box holding the table of turns of its positions, with
                               the definitions of that table, of the YaRN frequencies
                               and of the YaRN ramp
    token_compressors          the compressor at ratio 2 and at ratio 1
    attention_core             the sliding window view, the sink logit, and one
                               softmax over the window slots, the selected slots and
                               the sink, as one box computed once per head reading
                               the two kinds of slot concatenated into one axis
    scaled_attention_core      the same core with the scale the released kernel puts
                               on every score, between the contraction and the
                               exponential
    grouped_output             the head split, the block-diagonal map into eight
                               ranks and the weight over all of them, as one box
                               computed once per query
    lightning_indexer          the indexer's relative reads, its Top-512 and the
                               gather the selection drives
    rotated_indexer            the same indexer with the turns of its keys and its
                               queries, the FP4 round trip and the scale on the
                               weight of every indexer head
    candidate_pool             the block split, the Top-2048 over the blocks and the
                               pool of distances each Reindex layer scores inside,
                               as one box
    pinned_candidate_pool      the same pool with the block of the newest entries
                               pinned by a score of positive infinity
    quantised_caches           the FP4 and FP8 round trips of the cached arrays, each
                               boxed, with the ceiling as their one generic operator
    attention_modes            the seven modes, each grabbing what it reads from a
                               slot and dropping what it publishes onto one, at the
                               iteration of its group where it stands in one
    integrated_attention_modes the same seven modes with the rotations, the round
                               trips, the pinned pool and the score scales of the
                               mechanism modules above
    mixture_of_experts         the router, the routed experts, the shared expert and
                               the combine
    clamped_mixture_of_experts the same mixture with the two limits of the released
                               expert, the router's temperature, the correction bias
                               the token's modality selects and the epsilon of the
                               gate normalisation
    tempered_mixture           the mixture of `clamped_mixture_of_experts` computed
                               once per token, fed the modality of every token from
                               the tape
    text_only_mixture          the same mixture with one correction bias, for a model
                               that reads text
    single_pass_mhc            the three sets of mixing coefficients, the collapse of
                               the four streams and the write back
    mhc_with_epsilons          the same coefficients with the epsilons the released
                               kernel adds and the Sinkhorn rounds in its order
    gumbel_max_sampler         the draw of one token from the probabilities of one
                               position, with the exponential draw as its one generic
                               operator
    dspark_draft_chain         the drafter of five tokens, from the backbone state and
                               the accepted token, with the draft trunk as its one
                               generic operator
    write_at_token_positions   the write of an array of values into the residual at
                               token positions that arrive as data, over two
                               `inject.Inject` and a product
    vision_pathway             the vision encoder, the pixel unshuffle, the projector
                               and the writes of the cells and the span delimiters
    engram_modules             Engram of layers 1 and 14 on the tape, with the gate of
                               an image token set to zero
    text_only_engram           Engram with no modality, for a model that reads text
    dspark_drafter             the tap that writes a layer's stream mean onto the
                               tape, the grab that reads the three means back, and
                               the DSpark box the model ends in
    layer_stack                the boxed modes and the repetition blocks that state
                               the forty-layer plan
    divided_layer_stack        the same forty layers, divided where Engram and DSpark
                               stand between two layers
    whole_model                the embedding, the initial collapse vector, the layer
                               stack and the output head
    integrated_whole_model     the composition `v41_flash_integrated` and the released
                               sizes
    text_only_model            the composition `v41_flash_text_only`, which is the
                               integrated model without the image pathway and without
                               DSpark
    quantised_text_only_model  the text-only model with the quantisations the released
                               model runs in, written onto every wire by
                               `quantization.processing.quantise_model`, drawn by
                               `notebooks/sota/DeepSeekV41Flash.ipynb`, checked by
                               `validate_quantised_text_only_model.py` beside it and
                               by `quantization/validate_quantization.py`
    omitted_mechanisms         every mechanism the base model leaves out, stated with
                               the standard operators as far as they go and one
                               generic operator at the operation they cannot state,
                               with the claims about each in
                               `validate_omitted_mechanisms.py`
    validate_deepseek_v41_flash
                               one `check_*` function per part of the model the
                               mechanism modules build
    validate_omitted_mechanisms
                               one `check_*` function per mechanism of
                               `omitted_mechanisms`
    validate_deepseek_v41_flash_integrated
                               one `check_*` function per claim about the integrated
                               model and the text-only model
    validate_quantised_text_only_model
                               one `check_*` function per claim of
                               `notebooks/sota/DeepSeekV41Flash.ipynb`, which that
                               notebook runs in the cell after its setup cell so that
                               a reader sees one line confirming each claim
'''
