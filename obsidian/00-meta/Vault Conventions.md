---
tags: [meta]
status: stable
---

# Vault Conventions

This vault lives at `obsidian/` inside the `pyncd` repository. It is a companion to the code
rather than a copy of it.

## What belongs here, and what stays in the code

| goes in the vault | stays in the code |
|---|---|
| why a rule is the rule | what the rule does, line by line |
| the mathematics a module implements | the implementation |
| how modules relate, and in what order they run | call signatures |
| what is unfinished, and why it is hard | `TODO` comments, of which there are almost none, per `PublicCodeTODOs.md` |
| what an agent did and what it touched | commit messages |

If a note starts restating a function body, delete the restatement and link the file. The
code changes and a paraphrase of it rots silently. A reason does not.

Long-form mathematics belongs here rather than beside the code. A module explains what its
own functions do and links to the note explaining why. The design notes that used to sit in
the tree are now notes like any other, of which [[Diagram Wire Format]] is the longest.

## Note anatomy

Every note carries frontmatter:

```yaml
---
tags: [layer/hypergraphs, concept]
code: graphs/processing/leaf_splicing.py
status: stable | evolving | speculative
---
```

- `code:` names the file or files the note is about, repo-relative. One or several.
- `status:` is `stable` when the subject is settled and changes are rare, `evolving` when it
  is actively being worked on, so expect drift, and `speculative` when the design is not
  implemented, or only partly.

Then, in this order where each applies:

1. **What it is** — one paragraph, plain.
2. **Where it lives** — the files, and the entry points.
3. **The mathematics** — the correspondence to the formal object.
4. **The rules** — what is invariant, what is forbidden, and what a reader will not expect.
5. **Gaps** — what is not done, linked to [[Open Gaps]].
6. **See also** — links.

A note that owns a pipeline carries a Mermaid graph whose boxes are the forms the
expression passes through and whose edges name the passes between them.
[[Forms of an Expression]] indexes those graphs. A wiki link may not appear inside a
graph label, because `validate_repository.py` reads every link outside backticks, so the
links go in a table under the graph.

## Paths and links

- **A code path is repo-relative and written in backticks**, as
  `graphs/processing/leaf_splicing.py`. The repository root is the vault's parent
  directory. A line number is appended after a colon, as `data_structure/Term.py:37`.
- **A note link is a wiki link**, as `[[Leaf Splicing]]`. Note names are unique
  across the vault, so no folder path is needed.
- A link to a note not yet written is allowed and useful, because it marks work.
  [[Open Gaps]] collects the ones that matter.

## Naming

Folders are numbered by layer, from `01-foundations` to `07-para`, so that the file tree
reads in dependency order. A note is Title Case with spaces and is named after the concept
rather than the file, so it is `[[Leaf Splicing]]` rather than `leaf_splicing.py`. One
concept goes in one note.

## Vocabulary

Use the code's words exactly, and do not invent a synonym:

**term**, **UID**, **morphism**, **object**, **weave**, **target**, **degree**,
**reindexing**, **operator**, **block**, **wire**, **producer**, **consumer**, **slot**,
**grab**, **drop**, **tape**, **accumulator**, **quantisation** and **cast**.

Each has a note. Needing a new word is a signal that the concept is new, so write it a note.

## See also

- [[Agent Log Protocol]] — what a substantial piece of work owes at the end
- [[Repository Map]] — the code layout this vault mirrors
- [[Invariants]] — the facts that each cost an experiment to establish
- [[Code Style]] — the writing rules, and the survey they came from
