# quantization - the quantisation of a value, which is the number format it is held
# in with the size of that format in bits, the conversion between two quantisations,
# and the passes writing a quantisation onto every wire of an expression. Written by Claude Fable 5.1, effort 25, and extended by Claude Opus 5
# (1M context), effort high, on 2026-09-19.
#
#   data_structure/Quantization        `Encoding`, `Quantified`, `TypeConvert`, the
#                                      declared formats, and reading a quantisation off
#                                      a datatype or imposing one on it
#   processing/conversion_insertion    reading a hypergraph wire by wire, rewriting
#                                      the datatype of every wire, and inserting a
#                                      `TypeConvert` beside the producer of a wire or
#                                      in front of the operations reading it
#   processing/quantise_model          a quantisation on every wire of a model by
#                                      dataflow rules and a policy, with a cast
#                                      wherever an operator requires another
#   registries/operator_quantisations  the rule followed by each operator class for
#                                      the quantisation of its results and operands
#   validate_quantization              the checks
#
# `obsidian/04-quantization/Quantization.md` states the package.
