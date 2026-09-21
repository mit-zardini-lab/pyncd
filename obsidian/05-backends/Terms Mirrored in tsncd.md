---
tags: [layer/backends, reference]
code: data_transfer/term_json.py, websocket_transfer/headless.py
status: stable
---

# Terms Mirrored in tsncd

Written by Claude Opus 5 (1M context), effort 80.

`tsncd` draws the diagrams of this package. It is a TypeScript repository checked out
beside this one, in the parent directory of `pyncd`, as `../tsncd`.
`websocket_transfer/headless.py` finds its built bundle at `tsncd/dist` beside the
checkout, unless `TSNCD_DIST` names another. `npm run build` in `tsncd` writes the
bundle. A notebook kernel holds the headless
renderer until the kernel exits, so a rebuild reaches a notebook after the kernel restarts
or after `notebook_diagrams.close_renderer()`.

## A term reaches tsncd only if tsncd mirrors its class

`data_transfer/term_json.py` writes every `Term` as its class name, under `__type__`, and
its dataclass fields in the order the class declares them. It writes an enum as its type
name and its member name. `src/data_transfer/json.ts` in tsncd looks each class name up in
`TermDirectory` and each enum type up in `EnumDirectory`, and builds the TypeScript object
by passing the fields to its constructor in the order they arrive.

A class with no mirror in tsncd stops the whole transport. The import raises
`Term type not found in TermDirectory: <name>`, and nothing of the diagram is drawn.
`show_diagram` prints `(no diagram: Error: ...)` and carries on. The class need not be a
morphism. The `MemoryLevel` inside a `Stored` datatype stopped every diagram of
a notebook drawing it until it was mirrored on 2026-09-13. An enum with no mirror fails in the
same import with a `TypeError`. The listing of [[Agent Display]] needs no mirror, so
`DiagramMode.LISTING` still works on a term tsncd cannot import.

## Only the data_structure folders are mirrored

The terms an expression holds are declared in the `data_structure` folders of the features,
and only those are mirrored. A `Hypergraph` is converted to a morphism before it is sent,
per [[Hypergraph to Morphism]], so `graphs/data_structure/` has no mirror. Each
`data_structure` folder of `pyncd` has a folder of the same path under `src/` in tsncd, and
each module in it a TypeScript file of the same name.

| `pyncd` | `tsncd` |
|---|---|
| `data_structure/` | `src/data_structure/` |
| `advanced_axis_dynamics/data_structure/` | `src/advanced_axis_dynamics/data_structure/` |
| `para/data_structure/` | `src/para/data_structure/` |
| `quantization/data_structure/` | `src/quantization/data_structure/` |
| `deepseek/data_structure.py` | `src/deepseek/data_structure.ts` |

A module may be mirrored in part. The units of measure in `data_structure/Numeric.py`
have no mirror yet, so a numeric carrying a unit cannot be drawn. `solver/data_structure/`
has no folder in tsncd. The
behaviour that draws a mirrored class sits outside these folders, in the registries under
`src/display/`.

## Mirroring a class

1. Write the class in the tsncd file that mirrors its module, decorated
   `@fd.register_term`, extending the mirror of its Python base class.
2. Give the constructor one `readonly` parameter per dataclass field, in the order Python
   declares them, with a subclass repeating its parent's fields first. The fields arrive by
   position, so a reordered field produces a term with two members exchanged and raises
   nothing.
3. Mirror an enum as a TypeScript `enum` whose first member is `type = '<Name>'`, and
   register it with `fd.register_enum`.
4. Make sure the module is evaluated. TypeScript drops an import whose names are used only
   in annotations, so a module that other modules name only as a type never registers its
   classes. Give it an `establish()` function and call it from `src/index.ts`, as
   `Quantization.ts` and `AxisConcatenation.ts` do.
5. Register how the class draws, where the default is wrong: `opsRegistry` for an
   operator's box, `datatypesRegistry` for a datatype's anchor, `axesRegistry` for an axis
   and `blocksRegistry` for block aesthetics. Each registry matches the exact class name, so
   a subclass needs its own registration.

## What a mirrored class needs beyond its constructor

A change that added two seeds, a slot entry, a numeric and a block aesthetics at once
needed a different second step after each class was mirrored.

| kind | pyncd side | tsncd side beyond the class |
|---|---|---|
| an operator with a glyph of its own, `L1Norm` | declare the `cat.Operator` subclass in `data_structure/Operators.py` with a `template` building its weaves, and register it wherever `ops.SoftMax` is registered: the derivative rule, the standard expansion and `torch_compile` | mirror the class in `src/data_structure/Operators.ts` with its fields in the Python order, and register a box on the exact class through `opsRegistry` in `additionalOperationBoxes.ts`. An operator with no box draws as an empty gap with overlapping labels and raises nothing. Where the new glyph is a variation of an existing one, extract the shared geometry into a function both boxes call, as `draw_normalisation_triangle` is shared by the softmax and the L1 norm. Check the glyph in dark mode as well: a mark running the full width of the box sits on the same line as the wires it joins, takes their colour, and reads as a wire passing through |
| a seed, `ReductionGrab` and `ReductionDrop` | subclass `Grab` or `Drop` in `para/data_structure/Para.py`; give it an entry class beside `StreamSlot` and extend `entry_of`, `slot_of`, `grab_of` and `drop_of`; give the entry a mark in `agent_display.morphism_ir.slot_text`; add the seed to the `isinstance` checks of `tape_members` and `tie_tapes` that single out the stream seeds | mirror the entry class and extend `entry_of` and `slot_of` in `src/para/data_structure/Para.ts`; give the entry its mark in `ParaWrapDisplay.labels`. `ParaWrap.ts` and `ParaCategoryRenderer.ts` test `instanceof Grab` and `Drop`, so a subclass draws as its parent without a registration |
| a seed with a field of its own, `LoopGrab` and `LoopDrop` | declare the new field after `tape` and `size`, so the two the parent declares keep their positions, and carry it on the entry class as well; read it through a function beside `slot_of`, which is `index_of` | repeat the parent's `tape` and `size` as `readonly` constructor parameters before the new one and pass them to `super`, because reconstruction from JSON is positional; write the field into the label in `ParaWrapDisplay`, which for an index is `entry_latex` putting it in a subscript under the braced slot name, since a name carrying a subscript of its own and a second subscript in a row is not LaTeX |
| an operator that draws as another does, `ParaBlockOperator` | subclass `ops.BlockOperator` in `para/data_structure/ParaBlockOperator.py`, adding the fields the algebra reads | mirror the class in `src/para/data_structure/ParaBlockOperator.ts`, repeating `name` and `block` before the new fields, and register the parent's box for it by call in `ParaCategoryRenderer.ts`, as `Zero` takes `GenericOperatorBox`. The registry matches the exact class name, so the subclass draws nothing without that line. Name the class in `PARA_OPERATORS` as well, or webpack drops the module |
| an axis whose label is derived from other axes, `ConcatenatedAxis` | subclass `sc.Axis` in `advanced_axis_dynamics/data_structure/AxisConcatenation.py`, with `parts` as its one field of its own and a `__post_init__` writing `_size` as the sum of the parts' sizes; put no name on the uid, and label the axis in `agent_display.morphism_ir.axis_name` by joining the parts' labels with `PART_SEPARATOR` | mirror the class in `src/advanced_axis_dynamics/data_structure/AxisConcatenation.ts` with the constructor `(uid, _size, parts)` and an `establish()` called from `index.ts`; register a processor on the exact class through `axesRegistry`, in `display/Framework/advanced_axis_dynamics/concatenatedAxisLabels.ts`, whose `annotation_text` labels each part through the registry's processor for the part's class and joins the labels with the mirrored `PART_SEPARATOR`, so a guarded part reads inside the concatenation as it reads on its own wire; `AxisProcessor.label_width()` is zero by default and this processor reports the width of its label, which `AxisAnchor.label_width()` takes the maximum of, because a concatenated label is wider than the gap a short wire leaves for it. The three-way rule for one axis's label, the name with its exponent, else the integer size, else the name, is the exported `axis_size_text` in `StrideCategoryRenderer.ts`, which the default `size_text()` calls |
| a place in a codebase a block stands for, `CodeReference` | declare the class in `data_structure/ProductCategory.py` with `label`, `url`, `path`, `line` and `end_line`, and carry a tuple of them as the trailing `references` field of `BlockAesthetics`, so an older bundle receiving four fields ignores the fourth | mirror the class in `src/data_structure/ProductCategory.ts` and add the trailing constructor parameter to `BlockAesthetics`; nothing in the renderer core reads it, and `src/advanced_display/` lists the references in an inspection box, per [[Advanced Display]] |
| a formula a block shows, and whether the operator holding the block is drawn as a box or as the block's body in place, `BlockAesthetics.formula` and `BlockAesthetics.drawing` | add the two fields after `references`, in that order, and declare `BlockDrawing` with `BOX` and `BODY_IN_PLACE` beside the class, registered with `fd.register_enum`. `drawing` is `None` on every ordinary block and reads as `BOX`, so an ordinary term exports no enum value and a bundle built before the enum existed still reads it. tsncd's importer raises on an enum it has no registration for, so a default of `BlockDrawing.BOX` would have stopped every diagram holding a block on an older bundle | mirror the two trailing constructor parameters and the enum in `src/data_structure/ProductCategory.ts`, registered as `BlockDrawing`. `BroadcastedCategoryRenderer.broadcasted_box` builds the box of the body for a `BlockOperator` whose block says `BODY_IN_PLACE` and registers the hover region for the wrapper, per [[Advanced Display]] |
| a definition, `DefinedExpression` | declare the `fd.Term` in `data_structure/ProductCategory.py` with `left_hand_side` and `right_hand_side`, export it from `cat`, and give every pass that reads a morphism a case for it: `send_morphism.to_morphism` converts each side, `agent_display` lists each side with `:=` between them, and `notebook_diagrams.present_each_side` applies each presentation to each side | mirror the class in `src/data_structure/ProductCategory.ts` with the constructor `(left_hand_side, right_hand_side)` and re-export it from `Category.ts`. It is a `Term` and no `Morphism`, so it is drawn at the top level: `render_figure` in `src/display/diagramRenderTarget.ts` sends it to `render_definition`, which renders each side through `render_with_subblock`, so each side keeps the bodies of its own `BlockOperator`s, wraps each side at half of `width` less the glyph, and sets `\coloneqq` at 1.6em between them in the theme's text colour. The legend and the inspection boxes reach both sides, because the first walks the fields of the term and the second reads the regions the render filled |
| an operator that is the mirror image of another, `DeconcatenateAxes` and `PairsAsComplex` | declare the operator beside the one it mirrors, with the same fields in the same order, and register its reverse rule and its standard expansion in the feature's `registries/` | mirror the class beside its counterpart and share the geometry: `JunctionOfPartsBox` in `covariantOperatorBoxes.ts` draws the junction circle of a concatenation on the result side and of a deconcatenation on the operand side. Since 2026-09-17 the operation box takes no room and links nothing of its own: the `BroadcastedBox` that holds it links its own anchors of the parts to its own anchor of the whole axis, through `OperationBox.link_holder_anchors`, and the circle is drawn on that anchor. The circle is drawn on the broadcast layer, because the junction of a concatenation stands on the right column, which paints its wire after the core. The box clears `annotate_in_gap` on the degree anchors of its holder, so the axis that is cut or filled and its parts are the only axes it names, and `Anchor.gap_label_inset` starts the name of the filled axis clear of the circle. Beside it, and `draw_pairing_chevron` in `display_deepseek.ts` draws the chevron of `Decomplex` and, pointing the other way with `\mathbb{C}`, of `PairsAsComplex` |
| the read of one position, a row with an empty domain and a shift | build an `ops.View` of `StrideMorphism(_dom=(), _cod_stride_shift=((axis, (), index),))`, with `index` a free numeric, and name the `View` by the LaTeX of the index, because `Elementwise.template` names an unnamed one `\sigma` | nothing to add. `StrideMorphismBox` already draws a shift-only row as a pentagon on the wire of the axis. The wire ends at the pentagon's flat edge, the shift's LaTeX is written against its point, and since 2026-09-17 the pentagon is as wide as that label needs and no wider |
| a member of a mirrored enum, `ConstantSymbol.IMAGINARY_UNIT` and, since 2026-09-17, `ConstantSymbol.INFINITY` | add the member to the enum in `data_structure/Numeric.py`, with the LaTeX the constant prints as its value, `'\\infty'` | add a member with the same key and the same value to the enum in `src/data_structure/Numeric.ts`, because the JSON writes an enum member by its name and the numeric prints the value |
| an operator class and a subclass of it that share one glyph, `Rotary` and `YarnRotary` | declare `Rotary(cat.Operator)` in `deepseek/data_structure.py` with `name`, `base` and `position_stride`, in that order, and give every field a default, so that `YarnRotary(Rotary)` may declare `factor`, `ramp_start` and `ramp_end` after them. A dataclass lists the fields of its base first, so the JSON of a `YarnRotary` lists the three inherited fields and then its own three. Register the standard expansion of each class in `deepseek/registries/standard_expansions.py`, which nothing in the core imports, per [[Operators]]. `deepseek/validate_rotary.py` asserts the field order | mirror `Rotary` in `src/deepseek/data_structure.ts` with the three constructor parameters in the Python order, and mirror `YarnRotary` by repeating those three before its own three and passing them to `super`. Register the circle of `ComplexRotaryBox` for each class by call in `display_deepseek.ts`, as `bb.opsRegistry.registerClass(ds.Rotary)(ComplexRotaryBox)`, because the registry matches the exact class name and a subclass with no registration of its own draws as an empty gap. Since 2026-09-18 every table writes its own name in a strip above the circle, at `ROTARY_NAME_FONT_SIZE`, so `\mathrm{RoPE}` and `\mathrm{YaRN}` are told apart by reading the name. A box centres its core on the mid-line of its wires, so the room for the name is reserved above and below the circle and the circle is drawn in a child element translated down by half of it, which keeps the circle on that line. `region_element` returns that child, so the pointer still answers on the glyph alone. The `RotationDirection` enum was removed the same day, and an inverse rotation is the plain table followed by an elementwise conjugate. The inspection box over the circle needs nothing further: every `BroadcastedBox` registers a hover region, and the expansion arrives in `auxiliary.expansions` under the number of the table, per [[Advanced Display]] |
| a field added to an operator that is already mirrored, the `gain`, the `bias` and the `epsilon` of a `Normalize` and the `epsilon` of an `L1Norm` | add the field to the dataclass in `data_structure/Operators.py`, after the fields that are there, and give it a default, because a term is constructed positionally. Read it in the standard expansion and in `torch_compile` | add the constructor parameter to the mirror in `src/data_structure/Operators.ts`, in the same position and with the same default, and nothing else, because a field a box does not read changes no glyph. A box that reads one draws from it: `NormalizeBox` strokes its outline and its root tick heavier where `gain` is true, and keeps the radius, so the footprint is the same and no layout moves |
| an operator with no operands that names an index, `Arrange` | declare the operator in `data_structure/Operators.py` with the one field `name`, and put the axis in the target of the output weave | mirror the class in `src/data_structure/Operators.ts` and register a box in `additionalOperationBoxes.ts`. `ArrangeBox` draws one arc joining the axis wire to the `Natural` wire, the cup of an `Einops` mirrored, and no label. Since the third review of 2026-09-17 the arc is stroked with the `wire_attributes()` of the datatype anchor and carries the triangle `draw_datatype_direction_triangle` draws, centred on the half-way point `draw_cup_between_anchors` returns |
| a numeric with one argument, `Conjugate` | declare the class in `data_structure/Numeric.py` with `argument` alone, print it with `to_latex`, and register its derivative in `solver/registries/numeric_derivative.py` | mirror the class in `src/data_structure/Numeric.ts` in the same place in the file, with the one constructor parameter and the same `to_latex`. `NumericRenderer.ts` needs no case, because a numeric that prints itself is rendered through `to_latex` |
| a normalisation that may consume several axes, `L2Norm` | declare the operator in `data_structure/Operators.py` with the one field `name`, build it with `sized`, and register its standard expansion beside that of a `Normalize` | mirror the class in `src/data_structure/Operators.ts` beside `L1Norm` and register a `GlyphBox` for it in `additionalOperationBoxes.ts`. `L2NormBox` draws the `draw_normalisation_triangle` of a `SoftMax` in its glyph square and the `draw_root_tick` of a `NormalizeBox` inside the triangle, so the three normalisations read as one family. It is a `GlyphBox` rather than an `OperationBox` because the operator may consume more than one axis, and a shape stretched to a box carrying four wires is a different shape in every figure. The bite is on the top-right corner, opposite the corner a `LinearBox` and a `NormalizeBox` take, because this operator carries no learned weight |
| a numeric with bounds, `Clamp` | subclass `nm.Expandable` with `argument`, `lower` and `upper` in that order, and add a case to `canonical_order` | mirror the class with the three fields in the same order and the same `to_latex`, and add `clamp_string` to `numeric_string` in `NumericRenderer.ts` |
| an operator whose label is a long formula, `ConstantOp` | pass the formula as `value` and its LaTeX as `name`, so the listing shows it | `ConstantBox` sizes its core from the estimated width of its value's LaTeX, where it had a fixed core of 40 by 30 that wrapped a long formula over the wire |
| a term added as a trailing optional field, `scale: BlockScale | None` on `Quantified` on 2026-09-20 | declare the new class as a `Term` beside the class holding it, give the field a default of `None` and put it last, so that a JSON export made before the field existed still constructs | mirror the class in the same file with `@fd.register_term`, add the constructor parameter last with a `null` default, and export it from that file beside the others. `BlockScale` carries `form`, `channels` and `rows`, and `Quantified.width_latex` prints `\mathtt{MXFP8}` and `\mathtt{MXFP4}` for the two block-scaled formats the Microscaling specification names, from a two-row table mirroring Python's `INDUSTRY_NAMES`, and the element encoding with the scale in small type otherwise. The packing prefix prints the count against the format, `2\mathtt{BF16}` since the same day |
| an enum replacing a string field, `Encoding` on `Quantified` | declare the `Enum` beside the class it is a field of, register it with `fd.register_enum`, and give each member the text the display prints, `E4M3` and `BF16`, so that a misspelt encoding fails where it is written. `Encoding`, `Quantified` and `TypeConvert` live in `quantization/data_structure/Quantization.py`, and the JSON carries a class by its `__qualname__` alone, so the module a class stands in does not reach the transport | mirror the enum in `src/quantization/data_structure/Quantization.ts` with `type = 'Encoding'` as its first member, each member key the Python member name and each value the text `Quantified.width_latex` writes into `\mathtt{}`, and put the two classes there beside it. The new file needs an `establish()` called from `src/index.ts`, per the fourth step above |

The user ruled on 2026-09-13 that the tsncd side of an investigation is written by an
Opus agent launched for it, while the investigation goes on in pyncd. The agent is given
the Python files that define the terms, the positional rule below, a fixture, and the
checks, and it commits nothing and touches no server, because a relay page may be open.
The tsncd JSON reader constructs a term positionally, so a class whose Python fields are
reordered draws with two members exchanged and raises nothing, per the rule above. A
fixture is the check that catches a mismatch before a notebook does:
`data_transfer.term_json.TermJSONConverter.export_to_json` writes a morphism that
carries the new terms to a file, and after `npm run build` the headless renderer
draws it through `HeadlessRenderer.render_json` and `capture_rendered`, which is what
the tsncd bundle was built before the notebook was executed. A parse failure is an exception from `render_json`. A page open on the relay
server keeps the bundle it loaded, so a rebuild reaches it after the tab is reloaded,
and `notebook_diagrams.close_renderer()` or a kernel restart reaches the headless
renderer. The label a loop draws sits in a fixed slot sixty pixels wide beside the
bracket, so a repetition is written short.

## See also

- [[Diagram Wire Format]] — the messages the two repositories exchange
- [[Diagram Display]] — how a notebook draws, and the modes that need no tsncd
- [[Quantization]] — the datatypes a wire label is written from
