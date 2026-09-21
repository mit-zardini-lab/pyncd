---
tags: [layer/foundations, concept]
code: data_structure/Term.py, algebra/assign_code_forms.py, algebra/write_axis_exponents.py, para/processing/tape_members.py, notebooks/display/tape_naming.py
status: evolving
---

# Code Forms

Written by Claude Fable 5.1, effort 80, on 2026-09-13.

## What it is

A `DynamicName` carries two optional slots beside its body, its subscript and its
settings, per [[UIDs and Names]].

The **code form** is the identifier the whole name stands for in generated code. The
query axis is drawn `q` and carries the code form `queries`. Its size is drawn `|q|` and
carries `queries_size`. A tape slot is drawn `s_7` and carries `slot_7`,
a weight drawn `W_{G}` carries `weight_G`, and its gradient drawn `dW_{G}` carries
`grad_weight_G`.

The **exponent** is a name drawn above the body, after the subscript. An axis of a model
drawn at a published configuration is drawn `m^{7168}`. The exponent is left out of
`to_bodies`, so a lookup keyed by bodies is unchanged by it, and `to_text` writes it after
a caret, `m^7168`, which is what a listing prints.

The **exponent placement** is where the exponent is drawn, and is a field of
`DynamicNameSettings`. `fd.ExponentPlacement.SUPERSCRIPT`, the default, raises it after
the name. `SUBSCRIPT` lowers it into the subscript, so the same axis is drawn `m_{7168}`,
a name that already has a subscript joins the value to it after a colon, and a size symbol
keeps its bars outside the value, `|k|_{6}`. `to_text` writes a lowered value after a
colon. `to_bodies` is unchanged by
the placement, so a legend or a lookup keyed by bodies reads the same under either. The
user asked on 2026-09-16 for the sizes of a model's axes to be drawn as subscripts, with
the choice a setting, and `AxisSizes.SUBSCRIPT` in [[Diagram Display]] is that setting.

Both slots are `None` unless something sets them. A name that carries neither is drawn,
listed, compared and serialised as it was before the slots existed, so every existing
expression, notebook and stored signature is unchanged.

## Where it lives

| | |
|---|---|
| `data_structure/Term.py` | `DynamicName.code_form`, `DynamicName.exponent`, `DynamicNameSettings.typewriter`, `DynamicNameSettings.exponent_placement`, `ExponentPlacement`, `to_code_form`, `to_text`, `with_code_form`, `code_form_suffixed`, `with_exponent`, `exponent_placement`, `with_exponent_placement`, `identifier_from`, `join_code_forms` |
| `data_structure/StrideCategory.py` | `Axis.named(name, code_form=...)`, which gives the size `code_form` with `_size` appended |
| `data_structure/ProductCategory.py` | `Block.template(..., index_name=...)`, which names a block's tag, and a repeated block's tag name is the index of its loop |
| `algebra/assign_code_forms.py` | `assign_axis_code_forms(term, {'q': 'queries'})` for an expression built from letters, and `axis_code_forms`, `axes_without_code_form` to read them back |
| `algebra/write_axis_exponents.py` | `write_axis_exponents(term, {'m': 7168})`, and `write_axis_size_exponents(term)`, which reads the value off each axis's own size, each taking the `placement` the value is drawn at |
| `notebooks/display/axis_sizes.py` | `AxisSizes`, with `EXPONENT` and `SUBSCRIPT` for the two placements, and `DiagramSettings.axis_sizes` |
| `para/data_structure/Para.py` | `new_slot` and `slot_named`, `s7` with `slot_7` |
| `para/processing/show_grabbed_parameters.py` | `parameter_name`, `weight_G`, `bias_G`, `gain` |
| `para/processing/backprop.py` | `gradient_slot`, `grad_weight_G` |
| `para/data_structure/inject.py` | `selection_slot`, `index_k_e` |
| `para/processing/tape_members.py` | `tape_touches`, `tape_members`, `rename_tape_members`, `LoopIndex`, `TapeMember` |
| `notebooks/display/tape_naming.py` | `TapeNaming`, `DiagramSettings.tape_naming`, `members_frame` |
| `tsncd/src/data_structure/Term.ts` | the same five fields, in the same order, the `ExponentPlacement` enum and the fifth field of `DynamicNameSettings`, and `to_latex` drawing the exponent where the placement puts it and the typewriter setting |

`notebooks/display/tape_naming.py` chooses which of the three a tape member is drawn
under, so a figure of a taped pair names every member consistently.

## The composition rule

A name's code form is the identifier of the whole name, subscript chain included.
`add_subscript` and `subscript_target` compose it. When neither the name nor the name
hung under it carries a code form, the result carries none. When either does, the result
carries the two joined by an underscore, each side contributing its code form where it
has one and its bodies as an identifier where it does not, through
`code_form_or_identifier`. `identifier_from` writes every run of characters outside
`[0-9A-Za-z]` as one underscore, so `k/e` reads `k_e` and `:512` reads `512`. A name
built directly with a subscript, as `DynamicName('W', subscript=name)`, states its own
code form, and `join_code_forms` is how the sites that do so build it.

The exponent belongs to the outermost name. `add_subscript` keeps it on the result, and
`with_exponent` sets it from a name, a string or an integer.

## The tape and its loops

A tape slot has one name, and a loop repeats every operation inside it. A plain `Grab`
or `Drop` inside a loop touches a different member of the tape on every iteration,
because each iteration's residual, weight or index is an array of its own. A `StreamGrab`
or `StreamDrop` touches the loop variable the loop carries from one iteration to the next,
per [[Para Category]], which is one member for the whole loop.

`tape_members.tape_touches` walks a morphism with the loops and the blocks around each
grab and drop, and `tape_members` groups the touches by slot and index path into
`TapeMember`s. A member's `indexed_code_form` is the slot's code form followed by one
bracketed loop index per loop that selects a member, `weight_G[layer]`, and its
`indexed_latex` is the slot's name as drawn followed by the same indices, `W_{G}[l]`. A
loop variable's member is indexed by the loops outside its own loop alone, and
`carried_by` names the loop it is carried across.

A loop's index is the name of its `BlockTag`. `cat.Block.template(..., repetition=L,
index_name=fd.DynamicName('l', code_form='layer'))` sets it, and `backprop._block`
carries it onto the reversed block, so the backward loop reads the members the forward
loop wrote under the same index. A loop whose tag has no name is indexed `i`, `j`, `k`,
then `i2`, by its nesting depth.

`notebooks/display/tape_naming.py` is the display of it. `TapeNaming.INDEXED` rebuilds
every grab and drop with the indexed label as its slot's body, so a diagram of a
repeated layer draws `W_{G}[l]`, and `TapeNaming.CODE_FORM` rebuilds them with the
indexed code form in typewriter, `weight_G[layer]`, which is the form a listing reads
best. `NAMED` is the default and leaves the labels as the term names them. The choice is
a field of `DiagramSettings` and a parameter of `notebook_listings.print_listing`, and
`members_frame` tabulates the members.

## Giving a module's names code forms

Read the sites in the table above and follow the nearest one.

1. **An axis a model declares.** Pass `code_form` to `cat.RawAxis.named`, as
   a model's own axis declarations do. The size takes the code form with `_size`
   appended.
2. **An axis minted from a letter.** Apply `assign_axis_code_forms(term, mapping)` to
   the finished expression. An axis that already carries a code form keeps it. Check the
   result with `axes_without_code_form`.
3. **A name a pass derives from another name.** Compose it. Where the pass hangs one
   name under another, `add_subscript` composes the code form and nothing more is
   needed. Where the pass builds a `DynamicName` with a subscript directly, or writes a
   prefix into the body as `gradient_slot` writes `d`, give it
   `code_form=fd.join_code_forms(prefix, source.code_form_or_identifier())`, and pass
   `None` where the source carries no code form if the pass has to leave a name without
   one unchanged.
4. **A name a pass mints.** State the code form beside the body, as `slot_named` does.
5. **A value written into a name for display.** Put it in the exponent with
   `with_exponent`, as `write_assigned_values` does under `SizePlacement.EXPONENT` and
   `write_axis_size_exponents` does under `AxisSizes.EXPONENT`,
   and leave `to_bodies` to key any lookup. Where the value is to be drawn below the
   name, keep it in the exponent and lower it with `with_exponent_placement`, as
   `write_axis_size_exponents` does under `AxisSizes.SUBSCRIPT`, so `to_bodies` stays
   unchanged.
6. **A loop a model declares.** Name its tag through `index_name`, so the members inside
   it are indexed by a word.
7. **A term crossing to `tsncd`.** Nothing. The fields are serialised by position, and
   `Term.ts` declares them in the same order. A new field on `DynamicName` goes at the
   end on both sides, and the bundle is rebuilt with `npx webpack --mode production` in
   the `tsncd` checkout.

Read a code form with `to_code_form()`, and never parse it back into its parts. A code
form is an identifier for generated code, and the name's own fields carry the structure.

## Gaps

- The backward pass of a repeated block reads the tape in the order the forward pass
  wrote it. A loop's tape has to be read in reverse, and [[Open Gaps]] records it.
- `torch_compile` does not yet read code forms. A generated module names its parameters
  after `to_bodies`, and the code form is the name it should use.
- The `INDEXED` label is the name's LaTeX followed by its indices, so a listing under
  `INDEXED` prints `<W_\bold{G}[l]>` for a bold weight. `CODE_FORM` is the listing form.

## See also

- [[UIDs and Names]] — the name a code form decorates
- [[Para Category]] — the grab, the drop, the stream seeds and the indexed seeds
- [[Para Wrap]] — how a slot is drawn on the operation it touches
- [[Diagram Display]] — the settings a notebook draws with
- [[DeepSeek-V3 Backward Pass]] — the layer the tape members are read from
