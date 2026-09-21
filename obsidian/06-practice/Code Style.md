---
tags: [layer/practice, meta]
code: CLAUDE.md
status: evolving
---

# Code Style

The rules are in `CLAUDE.md`, under *How to write here*, and `StyleGuide.md` points at
them. This note records the survey the rules came from, the measurements behind them, and
what has and has not been applied.

## What the survey found, 2026-08-22

The package is not under-commented. It is under-structured. The prose is good and mostly
worth keeping. Three things were wrong with it.

It lived in the wrong container, as a header comment or inline narration rather than a
docstring.

It mixed three kinds of content in one block: the rule, the reason for the rule, and the
record of the experiment that established it.

In several places it narrated code that should have been named instead.

| measure | 2026-08-22 | Google |
|---|---|---|
| modules with a docstring | 25 of 124 | required, §3.8.2 |
| modules documented by a `#` header instead | 48 | should be a docstring |
| public functions with a docstring | 313 of 1157 | required unless trivial, §3.8.3 |
| files using an `Args:` or `Returns:` section | 1 | the standard form, §3.8.3 |
| lines of commented-out code | 125, across 20 files | delete, since git holds the previous version |
| functions over 55 lines | 30, the longest being 131 | prefer small and focused, §3.18 |
| highest comment-to-code ratio | 0.82, in a hand-transcribed numerical check | |

## What the register survey found, 2026-08-23

The second survey measured the writing itself rather than its container. It counted three
constructions the rules now forbid: a dash or spaced hyphen joining fragments, a
semicolon, and a word in capitals used for emphasis.

| body of prose | words | violations at the start |
|---|---|---|
| notebook markdown | 41,358 | 1,486 |
| comments and docstrings | 49,115 | 1,588 |
| vault concept notes | 55,957 | 2,540 |
| vault logs | 45,295 | 2,666 |
| research summaries | 18,000 | 1,763 |

The count is a detector rather than a measure. It finds the mechanical constructions and
does not find an aphorism, a personification or a title that has to be interpreted. Those
were found by reading.

## What has been applied

The register rules are written down in `CLAUDE.md` and are being applied file by file.
Ten notebooks and six modules were rewritten first, chosen by violation count.

The naming rules are applied. `validate.py` became `validate_simplification.py`,
the module that shadowed the standard library `json` became `term_json.py`, `parameters.py` became
`show_grabbed_parameters.py`, the `top` booleans became `is_outermost`, and
`pathway_collapse`'s single-word internals became `_unify_einsum_chain`,
`_match_forward_catalog`, `_collect_dead_roots`, `_catalog_key`, `_IndexedPass` and
`_TangentChain`.

The folder rules are applied. Each top-level folder is one feature, with
`data_structure/`, `registries/`, `algebra/` and `processing/` used where they apply.

The mathematics rule is applied. Every long-form design note is in this vault.

## What is outstanding

Twenty-eight modules still open with a comment header of five or more lines. The longest
are `deepseek/sparse_expansion.py` and `para/algebra/pathway_collapse.py`.

Roughly 1,150 function docstrings have not been read.

425 functions are not fully annotated, 297 of them missing only a return type. No type
checker is configured, so the typing rule is stated and not enforced.

Commented-out code has not been deleted, and `assert` and `0/0` have not been converted
to typed exceptions.

Five registries exist with five different spellings, where the rules call for one
decorator.

## Not adopted from Google

The 80-column limit is not adopted. This code carries type parameters such as
`def fresh_subscripted_uid[T: fd.Term, S: fd.Term = T](...)`, and 80 columns forces
wrapping that costs more than it saves. The limit here is 88.

The rule to write `import x.y.z` and never `from x import y` is not adopted. The package
has a fixed short alias per module, being `fd`, `cat`, `hg` and `chsh`, which is more
consistent than the Google form and is documented in `CLAUDE.md`.

Requiring `Args:` and `Returns:` sections everywhere is not adopted. A named signature
with a one-line docstring is preferred, and the sections are used where a function has a
contract the signature cannot express.

## Names

The recurring problem is a name that is accurate and states nothing. `both` names both of
what? `validate.py` validates what?

The test is whether a reader understands the name without opening the file it is
defined in. A name that makes sense only from inside its own module costs the reader a
jump.

`para/validate_backward.py` holds the worked example. The function was
`both`, was renamed to `show_both` on 2026-08-21, and was renamed again to
`show_both_directions`. Each of the two renames left a name that made the reader map a
word onto the forward pass and the backward pass. It is now `show_forward_and_backward`,
beside `show_forward_and_backward_as_derived`, and the `simplified=None` parameter that
chose between the two presentations is gone.

### Forward and backward name the training passes, and nothing else

`para/data_structure/Contravariant.py` carried a single common English word as its name
until 2026-09-01, and its
`ForwardOrReverse` enum became `CovariantOrContravariant`. The reviewer's statement is
that "forward" and "backward" are clear when the subject is a training step, which is
particular to deep learning models, and that the categorical pair is what says the generic
transformation is meant instead. Both words appear in one repository, so the name has to
choose which sense is in play.

The paper uses the same two words for whether a construction rule stores its builders or a
root term stores its properties, in `mid_sections/1ConstructionRules1.tex`. That sense is
about the representation rather than about a morphism, and it appears in one aside, so the
two do not compete.

### A term of art from another field is a second word for a fact the package already names

`Broadcasted.has_empty_domain` was `is_nullary` for a day, on 2026-09-01. "Nullary" is the
arity term from logic, and a reader has to know it before they can read the condition.
Every `Morphism` in the package declares `dom()`, so "domain" is a word the reader already
has, and `has_empty_domain()` is exactly `len(self.dom()) == 0`.

The prose about `Broadcasted.backup_degree` moved with the name, in
[[Broadcasted Category]], [[Weaves and Degree]], [[Construction Helpers]], [[Invariants]]
and [[Open Gaps]]. "Nullary" stays where the subject is an operator's arity rather than a
morphism's domain, as in `ConstantOp` and `Zero`. The two are different objects, and the
arity of an operator is the plain reading of the word.

## See also

- [[Invariants]] — the facts that cost an experiment to establish
- [[Repository Map]] — what is live and what is dead
