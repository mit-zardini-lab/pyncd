# Claude Opus 5.5 (1M context), effort 40.
'''One pass of GLM-5.3 with every layer reading its keys from caches, as a morphism in
Br.

The uncached model is `notebooks/sota/GLM53/`, which reads every token of the sequence
in one pass. The package here writes the pass the reference runs once a cache holds the
tokens of the earlier passes: it reads the tokens `x` of this pass, saves the arrays
the reference caches through one `caching.data_structure.Caching.Caching` each, and reads
every cached token on the axis `P + x`. Every part the cache leaves unchanged is
imported from `notebooks/sota/GLM53/`. `notebooks/website/modern/validate_glm53.py`
checks that this pass caches arrays of the same widths as the pass derived by
`derive_cached_glm53` and computes as many operations.

    wording                 every block description of this package, held in
                            `wording.json`
    reference_links         the pinned links into `cache_utils.py` and into the lines
                            of `modeling_glm_moe_dsa.py` that write to the cache
    cached_axes             the past tokens `P`, the cached tokens `P + x` and the
                            distances back over the cache
    rotation_at_this_pass   the two rotation boxes, reading the table of turns at the
                            positions `|P| + i_x`
    cached_indexer          the indexer key cache `idx`, the indexer reading it back
                            from every query, and the top-2048 over the cache
    cached_attention        the queries of the pass, the caches `lat` and `rot`, the
                            keys and the values expanded from them, the gathers, the
                            core and the output projection
    cached_attention_modes  the Full and the Shared modes, cached
    cached_model            the 78 layers and the whole pass
    cache_findings          the values cached per token and layer, the values on the
                            key-value path, and the operations of a pass by part
'''
