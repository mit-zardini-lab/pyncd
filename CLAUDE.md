# Working in this repository

Written by Claude Opus 5 (1M context), effort high.

`pyncd` formalises deep learning models as algebraic expressions and derives diagrams,
PyTorch code, a backward pass and a quantised model from them.

`README.md` describes what the package is for. This file describes how to work in it.
Read the writing rules below before you write any code or prose here.

When creating a file, can you put your model version and effort at the top.
This is so I can track the writing style of different agents.

# How to write here

These rules apply to code, comments, docstrings, notebook cells, vault notes, and agent
responses. The aim is to have direct, unbiased, prose that is not quippy and is clear
for a human reader.

**When a reviewer corrects the writing in this repository, add the correction to this
section.** Name the construction being ruled on, using the ordinary grammatical or
rhetorical term where one exists, and give one example of the rejected form beside its
replacement.

**When a reviewer corrects how a model is expressed, add the correction to
`obsidian/06-practice/Representing Models.md`.** That note is the consolidated record of
every ruling on how a model is written as a morphism, grouped by subject, each with the
rejected form beside its replacement. Put the correction in the section it belongs to.
When the rule has a code form, add it to
`notebooks/base_features/BuildingAModel.ipynb` in the same session, in the section it
belongs to, with an assertion.

***Code should be self-commenting with descriptive naming. EVERY COMMENT IS A FAILURE.***
***Avoid mannered prose.***

## Prose

Write plain English. Use the words a working practitioner in this field would use, and
use them in ordinary sentences.

**Avoid overusing jargon or compressing away the explanation.** Explain an operation
in ordinary language before naming it. Include the steps needed to understand why the
result follows. The user's examples and preferences for Astra are in
[AGENTS.md](AGENTS.md).

> Rejected: *Communication costs do not yet guide placement selection.*
> Replacement: *The code does not yet estimate how long GPUs would spend exchanging
> results. It therefore cannot use that estimate to choose how to divide the calculation.*

**Avoid rhetorical signposting in stand-alone sentences.** Do not announce that an
example matters or praise the explanation. Begin with the technical fact and explain
it in connected prose. Avoid mannered prose.

> Rejected: *The MLP example in the notebook shows why that distinction matters.*
> Replacement: *In Megatron's MLP, each GPU computes its own set of intermediate
> features. Each GPU can apply the activation to those features locally because
> their complete values are already available there.*

**Give every demonstrative an explicit antecedent.** A sentence beginning *This is*,
*That is*, *This means* or *That makes* points back at a whole preceding sentence and
leaves the reader to work out which part of it is meant. Repeat the noun instead.

> Rejected: *That is enough to read a model and not enough to do anything with the
> parameters.*
> Replacement: *A letter is enough to read a model and not enough to do anything with
> the parameters.*

**Put one idea in one sentence.** Do not join fragments with commas, semicolons, hyphens
or dashes. Do not interrupt a sentence with a parenthetical clause set off by dashes.
Two plain sentences are better than one sentence that has been extended twice.

**A colon is allowed when the phrase after it is in apposition to the phrase before it
and names the means.** The colon is then explanatory rather than decorative.

> Accepted: *Reading a Morphism: the SSA Listing*, because the listing is how a morphism
> is read.
> Rejected: *Parameters, exhibited*, where the trailing past participle is an ornament
> and names no method.

**Do not use antithesis to state a rule.** The *X, not Y* figure has been contaminated by
overuse and is now a fatal cliche. It leads to unclear, unreadable text.

> Rejected: *Name the role, not the position.*
> Replacement: *Name a parameter after what the function does with it.*

**Do not write a verbless heading or rule.** A rule with no finite verb is a slogan.
Write the instruction as a sentence in the imperative.

> Rejected: *No commented-out code.*
> Replacement: *Delete commented-out code. The previous version is in the git history.*

**Write "declared a-priori" where a fact was entered into a registry before the algebra
ran.** On its own, "declared" does not say by what or in what sense. The longer phrase
says that the knowledge base is built by hand, and marks the contrast with a fact the
algebra derives.

> Rejected: *The residual is declared, not derived.*
> Replacement: *The residual is declared a-priori, not derived.*

The ordinary English sense is untouched. An SSA value is still declared before it is
used, and a colour is still declared on a level.

**Define a technical term the first time you use it.** A common English word becomes
jargon as soon as it names a specific technical thing. The word "boundary" has been used
in this repository for a block's domain and codomain, and for the division between the
forward and the backward pass. Use the specific name, or define the word where you first
use it. The reviewer has rejected "boundary" twice, the second time in a module written
after this rule was recorded, so do not use the word at all.

> Rejected: *the boundary of a block*, for the wires a block reads and the wires it
> supplies.
> Replacement: `dom()` and `cod()`, which is what the code calls them, and the sentence
> that defines them, "a block's domain is the wires it reads and its codomain is the
> wires it supplies".

**Use one word for one thing, and do not introduce a synonym beside an established
term.** A reader who meets two words for one thing looks for a difference that is not
there. "Slot" names a place on the tape that a `Grab` reads and a `Drop` writes, and it
is qualified where the kind matters: a stream slot, a loop slot, a reduction slot. A
module that added "register", "cell" and "store" beside it would be rejected, and
"block" already names a `Block` in the algebra.

> Rejected: *the register the drop writes*, `CellEntry`, `store_naming`.
> Replacement: *the slot the drop writes*, `SlotEntry`, `slot_naming`.

**Do not write a phrase whose meaning has to be worked out.** A plain sentence that
states the fact is better than a compressed one the reader has to unpack, even when the
compressed one is shorter. This rules out the aphorism and the epigram. A hook is not
memorable, and it makes the sentence take longer to read.

**Name a heading after its subject, and do not write it as a free relative clause.** A
heading beginning *Where*, *What* or *How* is a headless relative used as a noun phrase.
It names no subject, so the reader learns what the section holds only by reading the
section. The construction also invites a metaphor to fill the gap it leaves, and the
metaphor is usually one the subject cannot support. A decision does not land anywhere.

> Rejected: *Where the decision lands differently.*
> Replacement: *Prefill and decode weight these differently.*

**Write a heading that can be read on its own.** A heading is read out of order, in a
contents list or by a reader scanning for the section they want, so nothing outside it is
on the screen. Two things therefore may not appear in one. An anaphor, meaning *this*,
*that*, *the other* or *it*, has its antecedent in a section the reader may not have
reached. A term the surrounding prose defined is unavailable for the same reason, even
where the term is the repository's own. Name the subject in full, and give the reader the
structure of the document before they have read it.

> Rejected: *What this spelling still needs.*
> Replacement: *The expanded selection needs a scatter.*

**Do not open a paragraph with a run-in label.** A bold phrase followed by a full stop
at the head of a paragraph is a heading in disguise, and a headless relative clause used
that way, *What the derivation already had*, is the free relative the rule above rejects.
The phrase hangs, because it names a topic and states nothing about it, and the reader
has to read the paragraph to learn what the label meant. Write the fact as the first
sentence, with its subject and its verb, and say what the things are. This rule applies
to an agent's response to the user as much as to a note.

> Rejected: ***What the derivation already had.*** *Each selection carries its index as a
> free numeric.*
> Replacement: *The derivations already had a free numeric for every selection count,
> named `k` for the count of experts a token routes to.*

**Do not write a trade-off as a purchase.** *Buys*, *pays for*, *pays the cost* and
*the price of* are a commercial metaphor, and the reviewer rejected it on 2026-09-20.
Name what is spent and what is gained as two facts.

> Rejected: *The reference code pays the BF16 cost so that its result matches that
> deployment.*
> Replacement: *The reference code stores and computes in BF16, which takes more memory
> and time than FP4, and its result matches that deployment.*

**Do not give a component an action it cannot perform.** Personification, metonymy and
synecdoche are all rejected. An interconnect transfers data and does not refuse anything.
An arithmetic unit takes a number of cycles and does not price.

> Rejected: *the wrap machinery never asked which operator it was dressing.*
> Replacement: *the wrap machinery does not test which operator it is applied to.*

**Do not modify a noun with a relative clause in which the noun is the object of the
clause's verb.** The construction puts the verb after the noun with the subject between
them, and the reader holds the noun until the verb arrives. Write the past participle
with *by*, or write a second sentence. The reviewer rejected the construction on
2026-09-20. A subject relative clause, *the operations reading it*, was not ruled on.

> Rejected: *carries the widths the released code declares*, *the wire the cast writes*,
> *the width a box returns*.
> Replacement: *carries the quantisations declared by the released code*, *the wire
> written by the cast*, *the quantisation returned by a box*.

**Do not write a pseudo-cleft.** The construction *X is what Y does*, and its mirror
*What Y does is X*, moves the real subject into the complement and leaves the empty verb
*is* carrying the sentence. The reader meets the fact only after the frame is built.
Write the subject and its verb in the ordinary order. The reviewer ruled on this on
2026-09-20.

> Rejected: *returning reals is what the released code does*.
> Replacement: *the released code returns reals*.

**Write "quantisation" for the number format a value is held in together with the size
of that format in bits, and do not write "width" for it.** "Width" is a common English
word, and the size in bits is only part of a quantisation, since E4M3 and E5M2 are both
eight bits wide. The package had written "width" beside "quantification" and
"precision" for the one thing. The reviewer rejected "width" on 2026-09-20, and the
package now says "quantisation" in prose and in its names, as `quantise_model` already
did.

> Rejected: *a width on every wire*, `narrow_operands`, `ArithmeticWidth`.
> Replacement: *a quantisation on every wire*, `rounded_operands`,
> `ArithmeticQuantisation`.

**State facts.** Do not narrate, do not build towards a conclusion, and do not address
the reader as a participant in a story.

**Do not evaluate what you are describing.** Write what a function does. Do not write
that it is elegant, clever, unfortunate or surprising. An adjective of that kind records
the writer's opinion on the day it was written, and a later reader cannot check it
against the code.

> Rejected: *the elegant trick here is that the reindexing already carries the degree.*
> Replacement: *the reindexing already carries the degree.*

**Do not use capital letters for emphasis.** If a sentence needs capitals to carry its
point, write a clearer sentence.

**Write an equation on one line.** The rule applies to every markdown file, in the vault
and in a notebook cell. A hard line break inside a sentence that holds an inline
equation, so that a line begins with `$`, stops the notebook renderer from typesetting
the equation, and the reader sees the LaTeX source. Write the whole sentence on one line,
however long the line becomes, or put the equation alone on a line of its own. The
reviewer ruled on this on 2026-09-16.

> Rejected:
> ```
> a distance $i_r$ for query
> $i_x$ names the entry
> $\lfloor (i_x + 1) / \lvert a \rvert \rfloor - 1 - i_r$ of the reference.
> ```
> Replacement:
> ```
> a distance $i_r$ for query $i_x$ names the entry $\lfloor (i_x + 1) / \lvert a \rvert \rfloor - 1 - i_r$ of the reference.
> ```

**Write a sum with the index it iterates and the axis the index ranges over, and read an
array at an index in brackets.** An index of the axis `m` is written `i_m`, and
`i_m \in m` says that it ranges over the positions of `m`. A sum written over the axis
alone does not say what is summed at each position, and a subscript on an array is
already used for the name of a weight, as in `W_Q`. The reviewer ruled on the notation on
2026-09-17, for the formulas an inspection box shows.

> Rejected: `\sum_{m} x^{2}`, and `(j, s_{j})` for an entry of a top-k selection.
> Replacement: `\sum_{i_{m} \in m} x[i_{m}]^{2}`, and `(j, s[j])`.

## Names

A reader should understand a name without opening the file it is defined in.

**Name a function after the action it performs and the thing it performs it on.**
`collapse_grabbed_residuals`, `migrate_drops` and `to_para_wrap` all do this.

**You may name a title, heading or notebook with a gerund and its object.** The verb must be the
plain one that describes the action, chosen for accuracy rather than for interest. A
title the reader has to interpret is wrong however well it reads.

> Rejected: *Putting weights on the boundary of an expression.* The verb phrase is
> vague, and "boundary" is an undefined term.
> Replacement: *Exposing Parameters.*

**Do not name a module, function or variable with a single common English word.** One
word names a subject and leaves the reader to guess the action. The file `parameters.py`
could introduce parameters, backpropagate them or count them. It displays them, so it is
called `show_grabbed_parameters.py`. The file `validate.py` became
`validate_simplification.py` for the same reason.

**Name a parameter after what the function does with it.** In
`merge_einops(producer, consumer)` the names say which contraction feeds which. In
`merge_einops(a, b, feed_index)` they say only that there are two of them, so the
docstring has to explain the order.

**`producer` and `consumer` name the two sides of a wire, and nothing else.** A wire is
a `HypergraphObject`. Its producer is the single root carrying it in its `cod`. Its
consumers are the roots carrying it in their `dom`. The pair is used throughout `algebra`,
`graphs`, `para` and `deepseek`, and reads clearly because it is consistent.

Consistency is the whole of its value, so do not use either word for anything else. Two
other meanings were removed to keep it:

- The packages that turn an expression into a diagram, PyTorch code or JSON were the
  vault's consumers layer. They are **backends**, and the layer is `05-backends`.
- A rewrite in `algebra/` that no single feature owns belongs to no one **feature** in
  particular, not to no consumer.

`producer` and `consumer` warpgroups, in the FlashAttention-3 sense, are a third meaning
and stay, because they are that paper's own term. Write them with the noun attached, as
"producer warpgroup", never bare.

**Name a boolean after the condition it reports.** `is_outermost` states a condition.
`top` states a place, and the reader has to find the definition to learn what is being
asked about it.

**Write the condition in the package's own vocabulary rather than in a term of art
borrowed from another field.** A borrowed term has to be learned before the name can be
read, and it competes with the word the package already uses for the same thing. Every
`Morphism` here declares `dom()`, so a reader already has "domain". "Nullary" is the arity
term from logic and gives them a second word for the same fact.

> Rejected: `is_nullary()`.
> Replacement: `has_empty_domain()`.

**Write "covariant" and "contravariant" for the direction a categorical construction
reads, and reserve "forward" and "backward" for the two training passes.** "Forward" and
"backward" are unambiguous when the subject is a training step, which is a deep learning
concept the reader already has. Using the same pair for the direction of a functor makes
the reader decide which of the two is meant at every occurrence. The categorical pair says
which one it is.

> Rejected: `ForwardOrReverse`, with `forward_or_reverse()` and `forward_body()`.
> Replacement: `CovariantOrContravariant`, with `covariant_or_contravariant()` and
> `covariant_body()`.

The paper uses "covariant" and "contravariant" a second time, for whether a construction
rule stores its builders or a root term stores its properties. The two senses sit at
different levels and the storage sense appears only in that one aside, so both stay.

**Do not select between two behaviours with a boolean parameter.** Write two functions
and name each after what it does. A boolean parameter is two functions sharing one name,
and the reader has to know what the name of the flag means before they can read the call.

> Rejected: `draw(pnorm, width=450, wrap=False)`.
> Replacement: `show_with_grab_boxes(pnorm, width=450)`, beside `show_to_para_wrap`.

**Use a single letter only for a loop counter.** `i`, `j`, `k` and `n` are acceptable. A
single letter standing for a datatype or an axis is not.

**`axis` names a `cat.Axis`.** Call a matplotlib `Axes` `panel`, and call a tensor
dimension number `dim`, as PyTorch does.

**Alias a module by its own name, except for the core.** Follow the Google import
convention: `import package.module as module`, and call `module.function(...)`. The alias
is the module's own basename, so a reader who has not seen the import still knows what
`einops_rearrange.merge_rule` is.

The core packages are the exception, because their aliases appear in every file and a
reader learns them once: `fd`, `cat`, `hg`, `ch`, `chsh` and `util`, for `data_structure`,
`graphs`, `construction_helpers` and `utilities`. Everything else is written out.

> Rejected: `import algebra.reindexing_absorption as ra`, then `ra.absorb(...)`.
> Replacement: `import algebra.reindexing_absorption as reindexing_absorption`, then
> `reindexing_absorption.absorb(...)`.

A short function name may stay when every call site qualifies it, as in
`tangent.array(...)`, because the qualifier supplies the missing context.

## Comments and docstrings

**Make the code state its own meaning first.** Naming a thing well removes the need to
describe it. Reach for a better name before reaching for a comment, and delete a comment
that a rename has made redundant.

A declaration whose name and type already say what it is needs nothing beside it:

```python
# Notebook args
DIAGRAMS = notebook_diagrams.DiagramSettings(
    mode=notebook_diagrams.DiagramMode.INLINE, width=900)
SIMPLIFY_DISPLAY = True
```

The modes do not need listing here. They are listed once, in the module that defines
them.

A function whose name states what it does needs no docstring:

> Rejected: `async def draw(term, wrap=True)`, with six lines explaining that it wraps
> each grab onto the morphism it touches before drawing.
> Replacement: `async def show_to_para_wrap(term)`, with no docstring.

**Do not wrap a function only to rename it.** A local `def simplify(m)` whose body is one
call to `merge_into_consumer.merge_producers_into_consumers(m, ...)` hides the name that says what
happens. Call the function, or give the wrapper a name that states what the particular
combination of arguments does.

Then write only what the code cannot state.

Write a docstring when it says something the signature does not. If the parameter names
and the return type already describe the function, write no docstring.

Write a docstring as a statement of what the function does, which is the descriptive
form Google asks for, and never as an instruction to the reader.

> Rejected: `'''Use sum reduction and equal contiguous output shards.'''`
> Replacement: `'''The shards each rank holds after summing `partials` and splitting
> the sum along `axis` into equal contiguous pieces.'''`

Delete a comment that repeats the line below it.

When a comment names an algorithm, extract that algorithm into a function with the same
name and delete the comment.

Delete commented-out code. The previous version is in the git history.

A function longer than about fifty lines usually contains a sequence of steps. Give each
step a name by extracting it into its own function.

Record a failed experiment in a log under `obsidian/00-meta/logs/`. Do not record it in
the docstring of the function that survived it.

**Read a function before you rewrite its docstring.** A docstring carries the claims the
signature cannot: a precondition, a limit, a reason. Rewriting the prose without reading
the body keeps a wrong claim and can sharpen it into a wronger one.

> `_sink_once` picks the operand with the fewest axes. Its docstring said "smaller", and
> a reader takes that to mean fewer elements. An operand of rank 2 at 4096 by 4096 has
> fewer axes and far more elements than one of rank 3 at 8 by 8 by 8.

## Where the mathematics goes

Write the general mathematics of a feature in `obsidian/`. A note states the motivation,
the mathematics and the shape of the implementation.

A module describes what its own functions do, and links to the note for the theory:

```python
'''Merging and disentangling chained contractions.

The rules are in `obsidian/02-categories/Einops Rearrangement.md`.
'''
```

Do not restate a note inside a module, and do not restate a module inside a note. Two
copies of an explanation start to disagree as soon as one of them is edited.

## Sourcing data from the internet

Cite the exact file a fact was read from, with a link pinned to a commit and the line
numbers the fact sits at, and state the date it was read. Prefer the primary repository.
When the primary repository has no GitHub mirror, say so and link the repository that
holds the code. `notebooks/sota/DeepSeekV41Flash/reference_links.py` cites the released
`inference/model.py` at one commit with a line range per mechanism, and is the pattern.

## Typing

Annotate every function and class, both the arguments and the return type. The
categorical structure makes some annotations long, and the Python 3.13 generic syntax
carries them:

```python
def fresh_subscripted_uid[T: fd.Term, S: fd.Term = T](target: T) -> S: ...
```

Write the accurate type even when it is long. Where a type cannot be written, write a
comment saying why, rather than leaving the annotation off. *It will make the user very happy*
*if you can add types to functions that do not have them, especially if there is a*
*category-theoretic aspect to this.*

## Code

Follow the [Google Python style guide](https://google.github.io/styleguide/pyguide.html)
where this section is silent. This section diverges from Google in several places, and
the departures are listed at its end. Where the two disagree, this file wins. Read two
neighbouring modules before writing a new one, and match them.

Write in a functional style. Return a new value rather than mutating an argument. Never
assign to a parameter. *Avoid this if the functional code is a performance bottleneck.*

Put per-operator rules in the feature's `registries/` folder, as a dictionary extended by
a decorator, so that supporting a new operator does not require editing the algebra.

Raise a typed exception carrying a message. Do not use a bare `assert`, and do not use
`0/0`.

**Do not read an environment variable.** Pass an argument instead. When a whole file needs
a mode, declare a variable at the top of the file and pass it to the functions that use
it. Two variables are read, and no others. `TSNCD_DIST`, read in
`websocket_transfer/headless.py`, names a built tsncd bundle. `PYNCD_DIAGRAMS`, read in
`notebooks/display/notebook_diagrams.py`, lets an agent executing a notebook in the
background replace its diagrams with listings without editing the notebook.

**Do not configure a module by assigning to its globals.** Writing
`other_module.SETTINGS.mode = ...` is an environment variable in a different form: it is
invisible at the call site, and every later call reads it without saying so. Declare a
settings object and pass it.

> Rejected: `DIAGRAMS = 'inline'` followed by
> `nd.SETTINGS.mode = nd.DiagramMode(DIAGRAMS)`, which holds one setting in two variables
> and mutates another module.
> Replacement: `DIAGRAMS = nd.DiagramSettings(mode=nd.DiagramMode.INLINE)`, passed as
> `nd.show_diagram(term, settings=DIAGRAMS)`.

**Name a closed set of cases with an `Enum`.** A string literal standing for one case is
a boolean parameter with more values. The reader has to find every comparison to learn
what the values are, and a misspelt value fails silently. `cat.WeaveMode` and
`Quantization.Encoding` are the pattern.

> Rejected: `kind: str`, compared with `kind != "local"` in six places.
> Replacement: `class TransferKind(Enum)`, compared with
> `kind is not TransferKind.LOCAL`.

**Test a type with `isinstance`.** `type(x) is int` rejects a subclass and reads as a
workaround for something.

**Return a dataclass rather than a tuple when a function returns more than two things.**
A four-tuple names none of its fields, and every caller has to unpack it in the right
order. `_Match` in `pathway_collapse` replaced exactly such a tuple.

**Construct a dataclass with keyword arguments once it has more than three fields.**
`Quantified(2, 16, Encoding.BF16, None)` cannot be read without the class definition
open.

**Import at the top of the module.** An import inside a function hides a dependency from
a reader of the module header. If the import is there to break a cycle, the function
belongs in a module that can import both sides.

**Write an error message that states the fact and includes the value that violated it.**
The message is read by someone who did not write the check, so it is not an instruction
to them.

> Rejected: `"Supply one quantisation for every operand"`.
> Replacement: `f"{len(quantisations)} quantisations for {len(operands)} operands"`.

**A validator is a script that prints one line per case and exits non-zero on a
failure.** `validate_repository.py` runs it and reads the exit code. Two shapes exist:
`validate_simplification` and `validate_backward` expose `cases(torch) -> dict[str, bool]`,
and `validate_sparse` runs `check_*` functions. Follow the one nearest to what is being
checked.

The departures from Google are:

- Lines may run to 88 characters rather than 80, because the generic signatures used here
  wrap badly at 80.
- The fixed module aliases stay, rather than the `import x.y.z` form, because they are
  consistent across every file in the package.
- `Args:` and `Returns:` sections are not required. A named signature with a one-line
  docstring is preferred. Use the sections when a function has a contract the signature
  cannot express.
- Docstrings are written with `'''` rather than `"""`. Every module in the package uses
  the single-quoted form, so a new module does too.
- `from dataclasses import dataclass` is written rather than `import dataclasses`, so a
  declaration reads `@dataclass(frozen=True)`. Google prefers importing the module. The
  package settled on the symbol before the guide was written, and every module uses it.

## Folders

Each top-level folder holds one feature and has matching notes in `obsidian/`. Within a
feature, use whichever of these four subfolders apply:

| folder | contents |
|---|---|
| `data_structure/` | the types that appear in expressions, mirrored in `tsncd` |
| `registries/` | per-operator implementations of a general rule, extended by a decorator |
| `algebra/` | the algebraic manipulations the feature needs |
| `processing/` | the passes that apply the algebra, and the pipeline that drives them |

General-purpose manipulations go in the top-level `algebra/` rather than a feature's own.
A feature that needs two of the four subfolders has two. `validate_<feature>.py` sits at
the feature root.

## Notebooks

Notebooks live under `notebooks/`, one folder per feature, and `example_notebooks/` holds
the short introductory notebooks a new reader starts with. A notebook demonstrates a
feature and serves as its test, so everything it claims is asserted in text and holds
with the diagrams turned off. A SOTA notebook whose model lives in a package beside it
is the exception, ruled on 2026-09-15: its claims live in a `validate_*.py` in that
package, which imports the modules and is discovered by `validations/`, and the notebook
keeps the prose and the figures. `notebooks/sota/DeepSeekV41Flash/` is the pattern.

These rules apply to the prose inside a code cell as much as to a markdown cell. A
comment or a docstring in a notebook is subject to the same rules as one in a module.

**A diagram is delivered in the notebook's INLINE output.** The user reads diagrams by
opening the notebook. Do not write PNG files into the repository to show a diagram, and
do not add a script whose purpose is to write them.

> Rejected: rendering the figures of a notebook into a `*_figures/` folder so they can be
> viewed outside the notebook.
> Replacement: executing the notebook with `DiagramMode.INLINE` and saving its outputs.

**Do not print an `agent_display` listing in a notebook.** A listing is the form an agent
reads a morphism in, and a person reads the diagram. The listing is therefore a diagram
mode: an agent executing a notebook sets `DiagramMode.LISTING`, or runs
`notebooks/execute_notebook.py`, and `show_diagram` prints the listing in place of every
diagram. A cell that calls `ad.listing` shows the person a form that is not useful to
them, whatever mode the notebook declares. Call `show_diagram` and let the mode decide.
The reviewer ruled on this on 2026-09-16.

> Rejected: `print(ad.listing(core_box.operator.block))` beside the diagram of the same
> block.
> Replacement: `show_diagram(core_box.operator.block, settings=DIAGRAMS)` alone.

**Draw a derived expression as the derivation returned it.** A diagram of a derivation
is the graph the pipeline produced, passed to `show_diagram`. Do not assemble a figure
from separately derived pieces, and do not draw a box or a title the derivation did not
produce. Put what a title would have said in the caption.

> Rejected: a display-side graph that wrapped each derived piece in a titled block inside
> an outer block.
> Replacement: `show_diagram(model.derived.graph)`, with the derivation named in the
> caption.

The first cell is a markdown title, following the naming rule above: a gerund and its
object, as in "Deriving the Backward Pass" and "Quantising a Released Model". Below it,
put one or two sentences describing what the notebook shows.

Do not restate the display module's documentation in a notebook. A notebook declares its
settings and calls `show_diagram`. How a diagram reaches the screen, and what has to be
installed for the headless renderer, belong in
`notebooks/display/notebook_diagrams.py`.

Draw every diagram through `notebooks/display/notebook_diagrams.py`. Do not write capture,
fallback or dump logic in a notebook. Those are the mechanism, and the mechanism lives in
one place.

When a notebook needs a display case the shared folder does not cover, decide which of
the two it is. A case that will recur belongs in `notebooks/display/`, as
`attention_figures.py` and `sota_figures.py` do. A case that belongs to one notebook
alone is written in that notebook.

A `.py` file may sit in a notebook folder to hold code the notebook imports. Name it
after its action, following the rule for modules. The assertions a notebook makes stay in
the notebook, so that the notebook remains the feature's test, except for a SOTA notebook
with a package, whose assertions live in the package's validator as the paragraph above
states.

Set the diagram mode in a variable at the top of the setup cell. Do not read it from the
environment. `notebook_diagrams.show_diagram` applies the `PYNCD_DIAGRAMS` override on its
own, so a notebook never has to.

A notebook and the vault note for the same feature link to each other.

# The vault, `obsidian/`

`obsidian/` is an Obsidian vault documenting this codebase, with one note per module or
concept, in folders numbered by dependency layer. It is the long form of this file. Read
it before substantial work and update it in the same session.

- `obsidian/Home.md` has the layer diagram, a table of where to start, and a map of every
  note. Begin there.
- `obsidian/06-practice/Open Gaps.md` is the ranked worklist. It records what is
  unfinished, why it is hard, and for some items what has already been tried.
- `obsidian/06-practice/Invariants.md` collects the invariants listed below and links to
  the notes explaining each.
- `obsidian/00-meta/Vault Conventions.md` states what belongs in a note.

Every long-form design note is in the vault. The longest of them,
`obsidian/05-backends/Diagram Wire Format.md`, is mirrored at `tsncd/PROTOCOL.md` in the
TypeScript repository, so edit both together.

## Substantial work requires a log

`obsidian/00-meta/Agent Log Protocol.md` states the full protocol. In summary:

Write a log in `obsidian/00-meta/logs/YYYY-MM-DD <Short Title>.md`, with one section per
row of the table in the protocol, and add it to `obsidian/00-meta/Agent Log Index.md`. A log is required
when the work spanned more than a couple of files or sessions, changed or established a
rule, produced a negative result, opened or closed an item in *Open Gaps*, or answered a
research question. A typo requires a commit message and no log.

The log records what git cannot: what was attempted, what was learned, and what did not
work. The *Negative results* section is the most valuable one. If nothing was abandoned,
write that nothing was abandoned rather than leaving the section empty.

Link every feature you touched to its note with a wiki link. The backlinks of a note then
give the history of that part of the system, including the attempts that failed.

Correct any note that has drifted from the code, in the same session. A note that
contradicts the code is worse than no note.

Write the log before you report completion. If you are running out of room, write the log
first.

# Where to start reading

To write an expression, read `notebooks/base_features/BuildingAModel.ipynb` first.
It states each rule of construction beside the code that follows it and asserts what it
claims. `obsidian/06-practice/Representing Models.md` holds every ruling on how a model
is expressed, and is where a correction is recorded.

Read the package from the bottom up. Four files cover most of it.

1. `data_structure/Term.py` defines `Term`, `UTerm`, `UID` and `DynamicName`, and the
   rewriting machinery `deep_reconstruct`, `EqualityClass` and `Context`. Every object in
   the package is a `Term`.
2. `data_structure/StrideCategory.py` defines axes and affine reindexings, the category
   **St**.
3. `data_structure/BroadcastedCategory.py` defines arrays, weaves and `Broadcasted`, the
   category **Br**. The division of a weave into tiled positions and a target is how
   broadcasting is represented.
4. `graphs/data_structure/Hypergraph.py` defines the graph form that expressions are
   rewritten in.

Then read `algebra/` for the rewrites any feature may use. Its `__init__.py` lists every
module and its contents.

# Folder structure

```
data_structure/       Term, St, Br. Depends on nothing else in the package.
construction_helpers/ The operator overloads that expressions are built with.
algebra/              General rewrites of expressions, usable by any feature,
                      including the contraction merge of einops_rearrange and the
                      weight split of linear_expansion.
graphs/               The hypergraph form, both directions of conversion, and the
                      leaf walk and splice of leaf_splicing.
solver/               Solving a numeric, and the derivative of a numeric.
para/                 The backward pass: derivatives, tapes, pathway collapse.
quantization/         The number format and the size in bits a value is held in, the
                      datatype conversion operator, and the pass that writes a
                      quantisation onto every wire of a model.

deepseek/             Sparse axes and the sparse expansion.
advanced_axis_dynamics/
                      The axis a read outside an axis leaves, the affine form that
                      says which of its positions hold a value, and the
                      concatenation of two axes carrying two forms. Only deepseek/,
                      two para modules and the V4.1 notebook import it.
validations/          Every validator and notebook as a target, the import graph
                      that says which targets a modified file reaches, and the
                      runner that executes the selected targets at once.
agent_display/        A morphism as an SSA listing. Read this rather than display/.
display/, torch_compile/, websocket_transfer/, data_transfer/
                      Backends. display/ imports nothing above it.
notebooks/            One folder per feature, plus the shared display helpers.
example_notebooks/    The short introductory notebooks.
obsidian/             The vault: a note per module, mirroring these layers.
```

# The algebraic structure

An expression is a morphism in a product category, defined in
`data_structure/ProductCategory.py`. It is built from sequential composition `Composed`,
parallel composition `ProductOfMorphisms`, a `Rearrangement` that permutes, copies and
deletes wires, and a `Block` that groups. The leaves are seed morphisms. A
`cat.DefinedExpression` pairs two such morphisms with one domain and one codomain, and a
diagram draws them with `:=` between them. It states that an operator equals its
expansion, or that an operator whose value depends on an index equals a formula in that
index, per `obsidian/06-practice/Representing Models.md`. Two categories matter.

**St** is the axis-stride category. Its objects are axes carrying a size. Its morphisms
are affine index maps. A `StrideMorphism` carries `_dom` and `_cod_stride_shift`, so each
codomain index is `Σ strideᵢ·domᵢ + shift`. A `Rearrangement` is the case that only
permutes, copies and deletes. Affineness makes the dependency between two axes integer
linear algebra rather than symbolic execution. A codomain index outside `[0, size)` of
its axis reads the universal unit, always, and no guard mechanism exists beside that
rule. `advanced_axis_dynamics/` derives which positions of a read array are empty.
`mark_sparse_domains.guarded_view` marks the last domain axis of every row that reads
outside its axis as an `AffineGuards.AffineSparseAxis`, which carries the affine form
that says which positions hold a value. `ops.View.template` composes the same rows and
marks nothing, so a model asks for the form by the builder it calls. A condition no
single read states is written as a relative read followed by a merge, an
`aops.CovariantView` whose codomain `mark_sparse_codomains.mark_sparse_codomain` marks
and whose broadcast axes it re-guides. Do not write it as a row onto an axis that a
`Rearrangement` deletes: **St** is Cartesian, so a copy, a map on the copy and a
deletion compose to the identity and the algebra may erase the row. Two axes carrying
two forms are concatenated, with `aops.ConcatenateAxes`, and
`concatenation_expansion.expand_concatenations` rewrites every consumer that treats the
concatenated positions one at a time into the consumers of the parts. An axis is cut
into the parts that fill it with `aops.DeconcatenateAxes`, which is the reverse
derivative of a concatenation and is written out as a copy followed by one `View` per
part. `obsidian/02-categories/Advanced Axis Dynamics.md` states the feature and
`obsidian/02-categories/Padding and Masks as Sparse Axes.md` the reads that produce a
form.

**Br** is the array-broadcasted category. Its objects are arrays `[datatype, shape]`. Its
seed morphism is `Broadcasted`, which carries an `operator`, `input_weaves`,
`output_weaves` and `reindexings`.

A weave, `Weave._shape`, is a shape whose entries are either an axis or
`WeaveMode.TILED`. The operation is broadcast over the TILED positions. The remaining
positions are the target, which is the array the underlying operation receives.
`weave.target()` extracts the target, and `select_target` and `select_degree` project a
shape onto those positions.

The degree is the tiling common to every weave of a morphism, returned by
`broadcasted.degree()`. A reindexing maps output degree indices to input degree indices,
which is how one operator expresses transposes, diagonals, repetitions and convolution
windows.

`construction_helpers` supplies the operator overloads that expressions are built with.
`@` composes sequentially and aligns axes automatically, `*` forms a parallel product,
and `>>` lifts over a batch axis. Import the module for its side effects even when you do
not call it, as `import construction_helpers as ch`. A contraction is normally written
`ops.Einops.template('q d, x d -> q x')`. `notebooks/base_features/BuildingAModel.ipynb`
walks through the operators, the three composition operators, axes, sizes, blocks and
compilation, with a test of each rule.

# Hypergraphs

`graphs/data_structure/Hypergraph.py` defines the form that expressions are rewritten in.
Morphisms are the form they are read in. Convert with
`hg.Multigraph.from_morphism(m)` and `h2m.hypergraph_to_morphism(g)`.
`h2m.recycle` normalises a hand-built morphism by converting in both directions.

There are three kinds of graph:

| kind | contents |
|---|---|
| `HypergraphRoot` | one morphism, in `.wraps` |
| `HypergraphBlock` | a body and a `BlockTag`, carrying `repetition` and `aesthetics` |
| `Multigraph` | a set of subgraphs with a domain and a codomain |

# Invariants

Each of the facts below cost an experiment to establish. Read them before changing the
code they describe.

### Term sharing

Terms form an immutable directed acyclic graph with heavy sharing. A traversal that
follows paths rather than nodes visits a subterm reachable fifty ways fifty times. One
rewrite of attention produced 8.3 million calls to `hash` over a result holding 35
thousand distinct objects. Three functions accounted for the total and all three are now
memoised: `Term.__hash__`, `Context.apply` and `term_utilities.search`. Write any new
traversal so that it visits each node once.

### UID identity

`reconstruct()` keeps the UID and replaces the fields, so two terms with the same UID can
differ. A functor pass depends on exactly that, because it rewrites what each wire carries
and leaves the wiring alone. Do not change hashing or equality to compare UIDs alone.
`HypergraphObject` is the one deliberate exception, because its uid is a wire, and the
wire identity invariant below states what its equality means.

### Axis identity

An axis is identified by its UID. Its name exists for display. Calling
`cat.RawAxis.named('q')` twice produces two different axes, with different UIDs and
different size symbols that both print as `|q|`. Never treat two equal names as the same
axis. Composition aligns axes by position, through `construction_helpers`, and never by
name.

`nm.FreeNumeric.named(x)` is the exception. It derives its id from the name through
`fd.hash_id`, so it is stable across calls. The two `named` constructors differ
deliberately, and the difference is easy to miss.

### Manipulating a `Broadcasted` by position

An axis identity is not a reliable indicator of the structure of an expression or of a
rewrite. It records that two positions have the same size and that composition carries
an index from one to the other along a wire, and `@` assigns it by joining UIDs position
by position. It does not record that an operation reads two positions with one index
variable. Read in isolation, the structural and algebraic properties of an expression
and of a transformation are determined by the position of an axis within the degree and
the target of a `Broadcasted`. A weave is a sequence of positions, each tiled or holding
a target axis. A reindexing maps output degree positions to input degree positions. An
`Einops` signature lists a contraction group per non-tiled position. Those three state
the morphism, and the axis at a position supplies its size and its name.

Read a morphism through `select_degree`, `select_target`, `target_idx`, a reindexing's
`mapping` and `einops_rearrange.segment_group_axes`, and build one by writing weaves,
reindexings and signatures. Do not deduplicate a shape, test an axis for membership in a
shape, key a dictionary by an axis, or look a position up with `shape.index(axis)`. Each
of those identifies an axis with an index variable, and an expression whose one token
axis stands at two positions, which is self-attention written from one copied input, is
read there as a diagonal.

`einops_simplification.einsum` builds a morphism from shapes of `IndexVariable`s, which
`index_shapes` reads off a morphism, one per degree position and one per contraction
group, and every rule in `para/registries/derivative.py`, `operator_expansion`,
`pathway_collapse` and `einops_rearrange.merge_einops` writes its shapes in them. A target
position of an operator that states no grouping, meaning a `Linear`, a `SoftMax` or a
`Maximum`, is related to the other side's target by its axis, because the operator
declares nothing else, and `obsidian/06-practice/Open Gaps.md` lists that fallback with
the one construction that still relates a new operand by axis, in `sparse_expansion`.

### Preserving sharing

`deep_reconstruct` returns the original object when nothing changed, which preserves
sharing. A rewrite that rebuilds unconditionally converts the directed acyclic graph into
a tree and slows every later pass.

### Class flags on Terms

A `ClassVar` annotation still appears in `__dataclass_fields__`, which `Term.keys()`,
`dict()` and `reconstruct()` all iterate over. An annotated class flag is therefore passed
to `__init__` and raises. Declare a class-level flag on a `Term` without an annotation, as
in `_memoize_hash = True`.

### The memoised hash

`@dataclass(frozen=True)` leaves `__hash__` alone when the class already defines one in
its `__dict__`. `Term.__init_subclass__` uses that to install the memoised hash on every
subclass without editing the fifty or so declaration sites, because `__init_subclass__`
runs before the decorator.

### Numeric equality

Numerics compare structurally, and `template` canonicalises. `Addition.template` and
`Multiplication.template` flatten nested operations, drop units, fold integer constants
and sort operands into `canonical_order`. As a result `x + y` equals `y + x`, and `1 + -1`
equals `0`.

Anything deeper, such as collecting like terms or cancelling `x/x`, is algebra. Put it in
the numeric algebra section of `Numeric.py` and apply it explicitly. Do not put it in
`==`. Equality once went through a `numeric_hash` modulo 2**16-1, which made
`2x + 2x == 4x` true and also made `Integer(65535) == Integer(0)` true.

### Wire identity in a hypergraph

A wire is a `HypergraphObject`, and its uid is its identity. Equality and hashing read
the uid alone, so two objects with the same uid are the same wire even when their `.obj`
differs. The two differ constantly, because a functor pass rewrites `.obj` while the
wiring stays fixed.

Every splice in this package renames wire identity with
`fd.UIDRenaming.set_canonical(new, old)` inside an `fd.Context`. A `UIDRenaming` keeps
each occurrence's fields and replaces its uid, where an `EqualityClass` replaces the
whole term and would overwrite each occurrence's `.obj` with the canonical's.

### Blocks carry meaning

A `BlockTag` with `repetition != 1` denotes a loop, which carries a running value from one
iteration to the next. An `aesthetics` field says how the block is drawn and says nothing
about the semantics of its body.

### Flattening a graph

`flat_subgraphs(g, remove_blocks=True)` does not remove every block. It descends through a
block only when `repetition == 1`. A loop block is returned as a `HypergraphBlock` with no
`.wraps`, so a walk that assumes it has reached a leaf will raise. Recurse into `.body`
yourself, or use `leaf_splicing.all_leaves`.

### Connectivity queries

`HypergraphAnalysis` answers connectivity questions through `left_subgraphs`,
`right_subgraphs`, `nodes_left` and `nodes_right`. `HypergraphSpecialTag.LEFT` and
`HypergraphSpecialTag.RIGHT` denote the world outside the graph. Its indices are declared
`cached_property`, but a fresh instance is constructed for each query, so nothing is
shared between instances.

### The reverse crawler's guide

In `graphs.processing.hypergraph_crawler.BuiltReverseCrawler`, the guide handed upstream
is read off the domain of the rebuilt morphism, through `_object_to_guide`, rather than
being the guide that arrived from downstream. A rewrite that changes what a wire carries
therefore hands the changed form upstream without doing anything further.

### Splicing a node

When splicing a node into a graph, rewrite the terminal alone and leave its consumers
alone. Rewriting the whole region connects the consumers to the value from before the
splice and bypasses the node just inserted. `graphs/processing/leaf_splicing.py` holds the
walk and the splice, and `rescope` rebuilds the domain and codomain of every container
around the change.

### Determinism

`util.Multidict` is backed by a set, and `UID._id` is random per process. Any code that
iterates over a set of terms must sort them by `uid._id` first, as
`para/algebra/tie_tapes.py` and `deepseek/sparse_expansion.py` do. Without the sort,
results reorder between runs.

### Pickling a term

A term is sent to a worker process by pickling. Python salts the hash of a string
differently in every process, so the structural hash a term caches in its `__dict__` is
wrong once the term is unpickled elsewhere, and `Term.__getstate__` leaves it out.
`fd.hash_id` digests `repr(obj)` rather than taking `hash(obj)`, so a named symbol has
the same id in every process. Before that change `nm.FreeNumeric.named('MatrixRate')` was
a different term in a worker and in the parent.

### The display import cycle

`display` imports nothing above it. Every layer above the backends may import `display`,
so an import in the other direction forms a cycle.

### A wiring diagnostic

When `hypergraph_to_morphism` raises `ValueError: HypergraphObject(...) is not in list`,
the wiring is wrong and the conversion is correct. Some wire is required on the right of a
branch and nothing on the left produces it. Two real bugs in a splicing pass were found
this way.

# Conventions

- Use `fd.Prod[T]`, which is an alias for a tuple, for anything held inside a `Term`.
  Never a list.
- Use `util.Multidict` for a one-to-many mapping, and `util.unique_tuple`,
  `util.iallequals` and `util.concat` for the operations that go with it.
- Read a name for display with `axis.uid._name.to_bodies()`, which may return `None`.
- Python 3.13 generics, written `def f[T](...)`, and structural `match` are used
  throughout.

# Running the checks

`validations/` finds every validation in the repository and runs the selected ones at
once as subprocesses. A validator is any `validate_*.py` at a feature root, found by
that name rather than read from a list, so a new feature's validator runs as soon as
the file exists. A notebook is a validation too, because it asserts what it claims with
the diagrams off. The checks of `validate_repository.py` are the third kind.

```bash
# the targets that import a file git reports as modified, and every target when the
# working tree is clean
python validations/run_validations.py
python validations/run_validations.py --all              # every target
python validations/run_validations.py --kind validator   # the validate_*.py scripts
python validations/run_validations.py --name Attention   # by name, matched loosely
python validations/run_validations.py --since main       # what changed since a ref
python validations/run_validations.py --list             # targets and dependencies
python validations/run_validations.py --dependencies-of BuildingAModel

# every module imports, every notebook's imports and calls resolve, the
# validators pass, and every link and path in the vault resolves
python validate_repository.py
python validate_repository.py imports        # or one check at a time
```

There are thirty-four targets: four repository checks, twenty validators and ten
notebooks. Only code that depends on a modified feature needs validating, which is the
reason the features are kept in separate folders, and the default run selects exactly
that. `--include-excluded` runs any target carrying a reason to be left out, and
`--list` gives the reason beside each.

Run `validate_repository.py` after moving or renaming anything.

The validators are `data_structure/validate_numeric_signs.py`,
`data_structure/validate_units_of_measure.py`,
`algebra/validate_discovering_broadcasts.py`, `algebra/validate_simplification.py`,
`para/validate_backward.py`, which checks derived gradients against `torch.autograd`,
`para/validate_loop_seeds.py`, `para/validate_para_block_operator.py`,
`deepseek/validate_rotary.py`, `deepseek/validate_sparse.py`,
`advanced_axis_dynamics/validate_advanced_axis_dynamics.py`,
`advanced_axis_dynamics/validate_covariant_broadcast.py`,
`quantization/validate_quantization.py`, `utilities/validate_wording_json.py`, the three
under `websocket_transfer/`, and the three in the packages under `notebooks/sota/`.
`python validations/run_validations.py --list --kind validator` lists every one of them,
because they are discovered rather than named here.

# Reading a morphism

Use `agent_display` rather than `display`. `display` draws a neural circuit diagram in two
dimensions using ASCII and ANSI colour. A human reads it well. Read as a stream of tokens
it is close to unreadable, because vertical alignment carries the wiring, and that does
not survive being read one line at a time.

`agent_display` renders the same object as an SSA listing, in the form a compiler prints
its intermediate representation:

```python
import agent_display as ad
print(ad.listing(morphism_or_hypergraph))   # the full listing
print(ad.summary(target))                   # inputs, outputs, operation counts
print(ad.trace(target, '%4'))               # what produces %4, and what consumes it
```

```
%4 = Einops(%0[q, {d}], %1[x, {d}]) : R[q, x]
%5 = SoftMax(%4[q, {x}]) : R[q, {x}]
%3 = Einops(%5[q, {x}], %2[{x}, d]) : R[q, d]
```

A name of the form `%n` denotes a wire and denotes the same wire everywhere. An axis in
braces is consumed by the operation. A bare axis is one the operation is broadcast over.
Blocks nest by indentation, and every value is declared before it is used. The listing is
deterministic, so two versions can be compared with `diff`. Fully expanded attention is 32
lines, against 98 for the diagram.

# Notebooks and diagrams

Notebooks are under `notebooks/`, one folder per feature, and the short introductions are
under `example_notebooks/`. Their kernels must start at the repository root.
`.vscode/settings.json` sets `jupyter.notebookFileRoot` for that purpose. Starting
`jupyter lab` inside a notebook folder will not resolve the imports. Outside VS Code,
`notebooks/fix_notebook_dir.py` performs the same job.

Diagrams are drawn by the headless renderer, or by a
[tsncd](https://github.com/mit-zardini-lab/tsncd) page, which requires an open browser tab
and a server on port 8765. Either `python run_server.py` here or `npm run server` in
`tsncd` provides that server, and only one of them may hold the port. You cannot see the
diagrams, so use `agent_display` instead.

`tsncd` is checked out beside this repository, as `../tsncd`, and draws a term only
if it mirrors every class the term holds. A class with no TypeScript mirror stops the whole
diagram from transporting, wherever in the term it is nested. Only the classes of the
`data_structure` folders are mirrored, each folder by the folder of the same path under
tsncd's `src/`. `obsidian/05-backends/Terms Mirrored in tsncd.md` gives the table and how
to add a mirror.

When an investigation adds a term that a diagram has to draw, launch an Opus agent to
implement the tsncd side, as the user ruled on 2026-09-13, while the investigation goes
on here. Give the agent the Python files that define the terms, the rule that tsncd
constructs a term positionally in the order of the Python dataclass fields, a JSON
fixture of a morphism holding the terms, exported with
`data_transfer.term_json.TermJSONConverter.export_to_json`, and the checks to run:
`npm run typecheck`, `npm run build`, and the fixture rendered through
`websocket_transfer.headless.HeadlessRenderer.render_json` and looked at. The agent
works in `../tsncd`, commits nothing and touches no server, because a relay
page may be open on port 8765. Run the notebook once the agent reports.

Every notebook under `notebooks/` draws through `notebooks/display/notebook_diagrams.py`
and sets its mode in a variable at the top of the setup cell:

| mode | cost | effect |
|---|---|---|
| `INLINE` | 166 ms warm, 1.7 s for the first | captures the image and embeds it in the cell |
| `BROWSER` | about 15 milliseconds | sends it to the open page and embeds nothing |
| `HTML` | under a second, with no browser | writes the figure to `outputs/pages/<slug>.html`, one file that opens with no server and no network and whose inspection boxes answer the pointer |
| `DUMP` | fast | writes the term to JSON, for a term `tsncd` cannot draw |
| `LISTING` | tens of milliseconds | prints the `agent_display` listing and draws nothing |
| `OFF` | none | skips the diagram |

Almost all of a diagram's cost is the browser laying it out. Building the expression takes
tens of milliseconds. Work in `BROWSER` and use `INLINE` for the run you commit. Everything
a notebook asserts holds with the diagrams off, because the diagrams illustrate rather than
demonstrate.

When you execute a notebook yourself, run it through `notebooks/execute_notebook.py`:

```bash
python notebooks/execute_notebook.py notebooks/base_features/BuildingAModel.ipynb
```

It starts the kernel at the repository root with `PYNCD_DIAGRAMS=listing` in its
environment, so `show_diagram` prints the listing in place of every diagram whatever mode
the notebook declares, and it prints each cell's text output as the cell completes. The
notebook file is not written, so the mode a person set at the top of the setup cell
survives. `--diagrams off` is the fastest run. Every notebook under `notebooks/` and
under `example_notebooks/` draws through `show_diagram`, so the override reaches all of
them.

# Known problems

- `torch_compile` has no module for `dst.Rotary`, `ops.Arrange`,
  `Quantization.TypeConvert`, a complete `TopK`, `Inject`, or a `Linear` with an index
  operand, so the checks that cover those constructions are structural rather than
  numerical. `obsidian/06-practice/Open Gaps.md` item 2 states what each rule would need.
- `ops.Normalize` always carries a gain, and the normalisation of the four streams in the
  mixing coefficients of DeepSeek-V4.1-Flash has none in the released code. The inspection
  box over that RMSNorm therefore draws a gain the released model does not hold. An
  RMSNorm with no gain needs a field on the operator or an operator of its own.
