# CLAUDE.md — leanncd

## New worktree / fresh checkout: don't cold-build Mathlib

This project depends on Mathlib (thousands of files). A `lake build` from a
checkout that has never been built compiles Mathlib from source, which can
take several hours — even though `.lake/packages` (the source fetch) may
already be present, `.lake/build` (the compiled `.olean`s) starts empty.

Before running `lake build` in a new worktree or fresh clone, check whether
another checkout of this repo already has a fully-built `.lake` directory,
and copy it over instead of rebuilding:

```bash
# 1. Find candidate checkouts (main repo, other worktrees)
git worktree list

# 2. Check if a candidate is actually fully built (compare to the target's
#    lakefile.toml / lake-manifest.json — a build from a different mathlib
#    rev won't help and lake will partially recompile it anyway, which is
#    still much faster than a full cold build):
find <candidate>/leanncd/.lake/packages/mathlib/.lake/build/lib -name "*.olean" | wc -l
find <candidate>/leanncd/Mathlib -name "*.lean" | wc -l   # rough total to compare against

# 3. If the candidate's count is at or near the total, sync its .lake into
#    the new worktree before building:
rsync -a "<candidate>/leanncd/.lake/" "./leanncd/.lake/"
```

**Use plain `rsync -a`, not `--info=progress2` or other GNU-rsync-only flags.**
macOS ships `openrsync` (protocol 29) by default, not GNU rsync — passing a
flag it doesn't recognize causes it to print usage and silently copy nothing,
while still exiting 0. Check `rsync --version` if unsure; don't trust a
"completed, exit 0" report alone — verify the `.olean` count in the
destination afterward.

After syncing, run `lake build` normally — it will only need to compile the
handful of `LeanNCD`-specific modules (and whatever the new worktree's edits
touch), typically well under a minute instead of hours.
