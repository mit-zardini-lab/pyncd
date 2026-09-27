# caching - the arrays a pass keeps for the passes that follow it, and the operator an
# array passes through to be kept. Written by Claude Opus 5.5 (1M context), effort 40,
# on 2026-09-25.
#
#   data_structure/Caching           `Caching`, whose operand is the array a pass saves
#                                    for its own tokens and whose result is the array
#                                    over every cached token, on the axis `P + x`
#   registries/standard_expansions   `Caching` written out as a `CacheGrab` of the
#                                    earlier tokens, a concatenation, and a `CacheDrop`
#                                    appending the tokens of the pass
#   algebra/cache_contents           the caches of a term, how many times each runs in
#                                    the loops around it, and how many entries each
#                                    keeps per token
#   algebra/derive_cached_pass       the pass of an uncached model over its new tokens,
#                                    with a `Caching` wherever a causal read demands an
#                                    earlier token, added 2026-09-26
#   algebra/cost_cache_placements    every placement of the caches of a pass, and the
#                                    bytes, operations and seconds of each
#   algebra/absorb_into_the_queries  a derived pass with the linear maps over its caches
#                                    contracted against the queries instead
#   algebra/count_pass_operations    the operations of every linear map and contraction
#                                    of a pass
#   validate_caching                 the checks
#
# The word "cache" here names the arrays a model keeps between the passes of
# generation. It does not name the cache memory of a GPU.
# `obsidian/08-caching/Caching Between Passes.md` states the package, and
# `notebooks/caching/CachedGLM53/` applies it to GLM-5.3.
# `obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` states the
# derivation, and `notebooks/classic/cached_mixtral_8x7b.py` applies it to
# Mixtral-8x7B.
