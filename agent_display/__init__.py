'''
Rendering morphisms for a reader that consumes text as a token stream.

`display` renders a neural circuit diagram, drawing wires as lines and operations as
boxes, and stacking and aligning axes. A morphism read one line at a time loses most of
what that picture carries, for four reasons.

  * Vertical alignment carries the wiring. Which output feeds which input is stated by
    two things sitting on the same row, and a line-by-line read loses the row.
  * Width changes the output. Padding, truncation and wrapping follow the terminal, so
    the same morphism does not render the same way twice.
  * Nothing carries a name. The `qTa` on one row and the `qTa` on another may or may not
    be the same wire, and only position says which, and only to an eye.
  * Every axis is redrawn on every segment it passes through.

This package renders the same object as a static-single-assignment listing, in the
manner of LLVM or MLIR. Every value is named and typed, every operation names its
operands, a declaration comes before its uses, and nesting is indentation. Compilers
print that form, and it appears throughout what language models have read. Five
properties of it matter here.

  * One line carries one fact. Horizontal position encodes nothing.
  * Identity is written out. `%4` is `%4` everywhere, so a claim about the wiring is
    checked by matching strings rather than by tracing a picture.
  * Values are declared before they are used, so one pass builds the graph.
  * Every value carries its type, and the broadcast structure with it. `{d}` marks an
    axis the operation consumes, and a bare axis is one it is broadcast over.
  * The rendering is deterministic. The same morphism gives byte-identical text, so two
    versions can be compared with `diff`.

Entry points, each of which accepts a morphism or a hypergraph:

    ad.listing(target)     the full SSA listing, under its legend
    ad.listing_without_legend(target)
                           the same listing, for a caller that prints many and
                           names the notation once
    ad.summary(target)     the inputs, the outputs and the operation counts, with
                           no body
    ad.trace(target, '%4') what produces a value, and what consumes it
'''

from agent_display import morphism_ir

listing = morphism_ir.listing
listing_without_legend = morphism_ir.listing_without_legend
summary = morphism_ir.summary
trace = morphism_ir.trace
