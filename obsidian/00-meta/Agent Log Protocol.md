---
tags: [meta, protocol]
status: stable
---

# Agent Log Protocol

Anyone, human or agent, who does substantial work on this repository writes a log entry in
`obsidian/00-meta/logs/` and links it from [[Agent Log Index]].

A log is neither a diary nor a changelog. Git records what changed. The log records what was
attempted, what was learned, which parts of the system were touched, and what the next person
should not have to rediscover, including the things that did not work, which git does not
record.

## When a log is owed

Write one if any of the following is true.

- The work spanned more than a couple of files, or more than one session.
- It changed, or established, a rule, meaning something in [[Invariants]] or in the algebra.
- It produced a negative result, meaning an approach that looked right and was not.
- It closed, opened, or moved an item in [[Open Gaps]].
- It was a research task, of the form "work out how X relates to Y" or "make the algebra
  derive Z".

Ordinary maintenance, meaning a typo, a rename, or one bug fixed in one place, needs a commit
message rather than a log.

## How to write one

1. Write `00-meta/logs/YYYY-MM-DD <Short Title>.md`, with one section per row of the
   table below. Use the date the work started, and append the end date in the frontmatter
   if it ran longer.
2. Fill it in. Be specific and be honest, because a log that reports only successes is worth
   very little to the next agent.
3. Link every feature you touched with a wiki link to its note. The links are what make the
   vault navigable in reverse: open a note, look at its backlinks, and every session that
   has ever been inside it is there.
4. If a note you touched has drifted from the code, update the note in the same session. A
   stale note is worse than a missing one.
5. Add the entry to [[Agent Log Index]], newest first.

## What each section is for

| section | what goes in it |
|---|---|
| **Task** | what was requested, in the requester's words where possible |
| **Outcome** | one paragraph. Did it work, and what is now true that was not? |
| **Features touched** | a wiki link to the note for every module or concept worked with, each with a word on how it was touched: read, extended, rewritten, or broken and fixed |
| **Code changed** | repo-relative paths, grouped, with one line each |
| **What was learned** | the reusable facts. Anything here that is a rule also goes into [[Invariants]] or the relevant note. The log is where it is recorded rather than where it lives |
| **Negative results** | what was tried and abandoned, and why, ranked by how likely the next person is to try it again |
| **Verification** | what was run, and what it printed. `python validate_repository.py` at minimum for anything touching the algebra, per [[Validation]] |
| **Gaps** | opened, closed or moved, with links to [[Open Gaps]] |
| **Follow-ups** | the moves that come next, ranked |

## Long-running research

For work that runs over days and generates its own artefacts, meaning scripts, notebooks and
measurements, keep the detailed account beside the artefacts and make the vault log entry a
short summary linking to it. Do not fork the account into two places, because the two will
then disagree.

## For agents specifically

- Write the log before reporting completion. If you run out of room, the log is the thing that
  has to survive.
- Prefer linking to inventing. If a concept already has a note, link it, and create a note only
  when the concept is genuinely new.
- If the task was research and the answer was that the thing cannot be done yet, that is the
  outcome. Record the shape of the obstacle in [[Open Gaps]] and state what would have to be
  true to remove it.

## See also

- [[Agent Log Index]] — every log, newest first
- [[Vault Conventions]] — note anatomy and link style
- [[Open Gaps]] — the standing worklist that logs feed into
