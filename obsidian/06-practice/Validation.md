---
tags: [layer/practice, reference]
code: validations/run_validations.py, validations/validation_targets.py, validate_repository.py
status: stable
---

# Validation

Written by Claude Opus 5 (1M context), effort high.

Every check in this repository runs the real pipeline over a fixed set of expressions and
either compares the result against an independently known answer or asserts a structural
fact about it. A validator prints one line per case and exits non-zero on a failure, and
`validations/` finds every one of them and runs the selected ones at once.

## Running everything

```bash
python validate_repository.py
python validations/run_validations.py --all
```

The first runs five checks of its own. The second runs every target of the registry,
which includes those five, every `validate_*.py` script and every notebook.

## The five repository checks

`validate_repository.py` takes the name of one check, or no argument for all of them.

| check | what it establishes |
|---|---|
| `imports` | every module in the package imports cleanly |
| `notebooks` | every repository module a notebook imports still exists |
| `calls` | every module-qualified call in a notebook names a function that exists and takes the keyword arguments given |
| `validators` | the `validate_*.py` scripts pass |
| `vault` | every wiki link, `code:` path and file mention in `obsidian/` resolves |

A notebook is never imported, so nothing else notices when a rename breaks one. The
`calls` check found `sk._rescope(..., top=True)` after that parameter was renamed to
`is_outermost`. The `validators` check is the runner of `validations/` applied to the
scripts it discovered, so the two never disagree about which scripts exist.

Run `validate_repository.py` after moving or renaming anything.

## The simplification check

```bash
python algebra/validate_simplification.py
```

It compiles both forms with [[Torch Compile]] and runs them on the same random inputs, so
that [[Expression Simplification]] is checked against arithmetic rather than against a
listing. There are three cases with an independently known answer: absorbing a node undoes
`expand_to_nodes`, a diagonal followed by a sum is a contraction, and a chain of contractions
is left alone. It needs `torch`, which `requirements.txt` does not list, and without `torch`
the script reports the fact and exits 0.

## The backward check

```bash
python para/validate_backward.py
```

It is stronger than the simplification check, because it checks the derivation itself against
`torch.autograd`. Each expression goes through `forward_backward`,
[[Expression Simplification|simplification]] and, where there is anything to recall,
[[Pathway Collapse|recall]]. Both passes are compiled through `detape`, where a `Drop`
becomes an extra output and a `Grab` an extra input, in slot order, and the results of the
backward pass are compared with `autograd.grad` of the forward on random tensors.

It covers the matmul, the softmax, the expanded softmax with real semantics registered for
its named elementwise maps, where the `torch.relu` default for `Elementwise` is a
placeholder, attention, the two recalled attentions, including the transport over two
rounds through an output projection, and self-attention from one input copied three ways
with its four projections parametrised.

The shifted cases write the softmax out with `operator_expansion.expand_shifted_softmaxes`
first and compare the derived pair against the unshifted expression, because the shift
changes no value. The shifted softmax and shifted attention are each checked as derived and
after `dedup_and_collapse`. The collapsed backward pass is checked to hold no
`Arithmetic<0>`, which is the zero map the rule for `Maximum` writes and
`prune_zero_cotangents` removes, and the collapsed forward pass of the shifted softmax is
checked to keep its `Maximum`.

## The indexed tape seed check

```bash
python para/validate_loop_seeds.py
```

It checks the seeds of a slot indexed by the iteration of a repeated block, per
[[Para Category]]. Both fixtures are a block of four iterations whose body drops a computed
value onto the member the block's counter names. The first grabs at that counter, so each
iteration reads what it wrote, and the second grabs at `N - 1 - i`, so an iteration reads
what another wrote, which is the expansion path of a UNet reading a skip. The checks are
structural, because a UID is random per process. `type_search` finds both seeds and no
stream seed, `Para.entry_of` and `Para.grab_of` round-trip a seed through its entry,
`to_para_wrap` leaves no bare seed and writes a `Para.LoopSlot` carrying the index onto the
operation each seed touches, `tie_tapes` refuses both seeds, `tape_members` reports one
member per iteration, and `agent_display.listing` prints the index beside the slot's name.

## The sparse checks

```bash
python deepseek/validate_sparse.py
python advanced_axis_dynamics/validate_advanced_axis_dynamics.py
```

The first checks that the two presentations of a selection expand to the same expression,
that the wired pass and the taped pass of [[Sparse Expansion]] agree line for line, and
that an expansion leaves the domain and codomain of the model it rewrote unchanged. The
second checks the affine forms of [[Padding and Masks as Sparse Axes]]: the live set a form
gives at concrete sizes equals the set the reference admits, an unshifted view marks
nothing, and a concatenation of two axes carrying two forms reads as the two forms side by
side.

## The quantisation check

```bash
python quantization/validate_quantization.py
```

It checks the pass of [[Quantization]] against the quantised text-only
DeepSeek-V4.1-Flash: that every wire carries a quantisation, that a cast stands wherever
two neighbouring operations disagree, that the weights named by the file's own table carry
the quantisation the table gives, and that the counts of casts by kind are what the
released code implies. Eight further cases check the functor of
[[Stripping Quantisations]], from the model holding no quantisation after stripping to a
conversion that is not between two quantisations being kept.

## The model package checks

A SOTA notebook whose model lives in a package beside it keeps its claims in a
`validate_*.py` in that package rather than in the notebook, so the notebook keeps the
prose and the figures. There are six:

```bash
python notebooks/sota/DeepSeekV41Flash/validate_quantised_text_only_model.py
python notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash.py
python notebooks/sota/DeepSeekV41Flash/validate_omitted_mechanisms.py
python notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash_integrated.py
python notebooks/sota/GLM53/validate_glm53.py
python notebooks/sota/GLM53/validate_quantised_glm53.py
```

The first of them holds every claim `notebooks/sota/DeepSeekV41Flash.ipynb` makes, in forty-six checks, and the
notebook runs it in the cell after its setup cell so that a reader sees one line
confirming each claim. The two of `notebooks/sota/GLM53/` hold the claims about the
model and about its quantisations. [[SOTA Model Notebooks]] states what each asserts.

## The website checks

Each notebook under `notebooks/website/` holds prose and figures and no checks, and a
validator beside it holds one `check_` function per claim, in the order the notebook
states them. There are eleven, one per page, and the following command runs them with
the notebooks.

```bash
python validations/run_validations.py --name website
```

The four tutorial validators compare each training step with `torch.autograd` in float64
through `notebooks/website/tutorial/check_gradients_in_torch.py`. The seven model
validators check the model against its released code, its quantisations and the model
in the reals returned by stripping them where the page carries them, its cached pass
where the page has one, and that every legend row of every variant of the page carries
a code name.
[[Website Notebooks]] lists every notebook with its validator.

## The caching check

```bash
python caching/validate_caching.py
```

It checks the `Caching` operator and its expansion, the counts of `cache_contents`, the
pass `derive_cached_pass` derives from causal attention, the refusal of attention with no
causal read and of a read of the next token, the four placements of causal attention and
their costs, and the kept axis of a sliding window, which moves the same bytes whatever
the number of earlier tokens, per [[Deriving Caches by Dragging the New Tokens]]. The
checks are structural, because `torch_compile` has no rule for a `Caching`.

## The figure checks

```bash
python websocket_transfer/validate_auxiliary_information.py
python websocket_transfer/validate_standalone_page.py
python websocket_transfer/validate_page_variants.py
python websocket_transfer/validate_localise_descriptions.py
python utilities/validate_wording_json.py
```

The first checks the `auxiliary` field of a message, meaning the legend, the inspection
boxes and the expansions an interactive figure carries, per [[Advanced Display]]. The
second checks that a standalone page holds the bundle, the fonts and the message, and
opens with no server. The third checks a page carrying several variants of one model.
It checks the one compressed repository of their messages, the settings read by the
build plugin of the lab website, a derived variant naming its source and its functor,
the refusal of inconsistent variants, and the folder and the redirect written for the
page, per [[Diagram Display]]. The fourth checks that a page carries each wording of its
descriptions and switches between them. The fifth checks that every `$NAME` reference
in a wording file resolves.

## Running the validations in parallel

`validations/` is the package that runs the validations. It holds a registry of every
validation in the repository, an import graph over the repository's source files, and a
runner that executes the selected validations at once as subprocesses.

```bash
python validations/run_validations.py algebra/x.py caching/  # what these paths reach
python validations/run_validations.py --plan algebra/x.py    # the selection, not run
python validations/run_validations.py                  # what git reports as modified
python validations/run_validations.py --all            # every target
python validations/run_validations.py --kind validator # the validate_*.py scripts
python validations/run_validations.py --name Attention # by name, matched loosely
python validations/run_validations.py --since main     # what changed since a ref
python validations/run_validations.py --list           # targets and dependencies
python validations/run_validations.py --dependencies-of BuildingAModel
python validations/run_validations.py --reach-by-folder # how far a change reaches
```

A validation the registry knows about is a target, and on 2026-10-01 there were
sixty-one of them: four repository checks, thirty-six validators and twenty-one
notebooks. A validator is a
`validate_*.py` script, found by that name at a feature root or under a feature, so a new
feature's validator is run as soon as the file exists and no list has to be edited. A
notebook is every `.ipynb` under one of the two folders `NOTEBOOK_FOLDERS` names,
`notebooks/` and `example_notebooks/`, run through `notebooks/execute_notebook.py` with
`--diagrams off`, and it is a validation because a notebook asserts what it claims and its
assertions hold with the diagrams turned off. A repository check is one of the checks of
`validate_repository.py`, which register themselves with the registry through the
`repository_check` decorator.

A target may carry a reason it is left out of a default run, and `--list` prints the reason
beside the name. `--include-excluded` runs it anyway. No target carries one at present.

### Only the targets that depend on a modified file are run

Each feature sits in its own folder, so a validator or a notebook imports the feature
it demonstrates and the features that feature is built on, and nothing else. A
modification to one folder therefore leaves most targets untouched. Making that
selection possible is one of the reasons for keeping the code modular.

`validations/module_import_graph.py` parses every `.py` file and every notebook with
`ast` and records the repository files each one imports. A dotted name resolves to a
module file or to a package `__init__.py`, and every existing prefix of the name
resolves as well, because importing `a.b.c` executes the `__init__.py` of `a` and
of `a.b` too.
`ImportGraph.files_reached_by` closes the relation transitively, visiting each node
once, so an import cycle terminates.

A file also depends on the data files it reads. A string literal outside a docstring
that ends in `.json` or `.ipynb`, and that names an indexed file read relative to the
folder of the file or to one of its ancestors, is recorded as a data file read by the
file. A module imported by a file and no longer present is recorded too, so a deleted
or renamed module selects every file still importing it. The parse of each file is
stored in `.cache/validations/`, which git ignores, and a run parses again only the
files changed since the last run, so the selection takes under a second.

`validations/list_modified_files.py` reads the modified paths from
`git status --porcelain --untracked-files=all` and from `git diff --name-only <ref>`.
Both print a repository-relative path with forward slashes, which is the form the
graph indexes, so a reported path is looked up without conversion. The
`--untracked-files=all` argument matters because the default collapses a wholly
untracked folder to the folder's own path, and a folder path matches no indexed file.

`validations/select_affected_targets.py` selects a target when a modified file is
among the files its entry point reaches. A notebook's dependencies include
`notebooks/execute_notebook.py` and everything that script imports, because a change
there changes how every notebook runs. Three repository checks have no entry point
whose imports say what they read, so each carries a `DependencyRule` instead:
`imports` reads every module, `notebooks` and `calls` read every module and every
notebook, and `vault` reads every note and tests every path a note mentions. When a run
is selected by modified files, the runner passes them to the `imports`, `notebooks` and
`calls` checks, and each then checks only the files whose own dependencies include one
of them. Windows refuses a command line longer than 32,767 characters, so when the
modified files would take more than `MODIFIED_FILES_ARGUMENT_CHARACTERS` of it the check
is passed nothing and checks every file.

Paths given as arguments replace what git reports, so that an agent working beside
other sessions validates its own change alone. A path may be relative to the repository
root or absolute, with either kind of slash, and a folder stands for every file in it.
The runner lists each modified file reached by no validator or notebook, which only the
repository checks cover. `validations/validate_target_selection.py` checks the
selection on a repository built in a temporary folder.

### The runner is a thread per subprocess

Every target is already a process. A validator is a script, and a notebook runs under
a Jupyter kernel that `execute_notebook.py` starts. A thread per target is therefore
enough, because each thread spends its time waiting on a subprocess rather than
running Python, and it releases the interpreter lock while it waits.
`concurrent.futures.ThreadPoolExecutor` runs `--jobs` of them at once, eight by
default.

The runner prints a line per target as it finishes, carrying the status, the wall time
and the name, and prints the last twelve lines of output for each failure once the run
ends. It exits non-zero when anything failed. A `--timeout` kills the subprocess, and
on Windows it does not kill the Jupyter kernel started under it, so a timed-out
notebook can leave a kernel behind.

### What a run costs

The runner records the duration of every target that passed in `.cache/validations/`,
through `validations/order_targets_by_duration.py`, and the next run starts the longest
targets first, so the length of a run is set by the total work over its jobs. `--plan`
prints the selection in that order with the wall time it expects. A target that has not
run on the machine is ordered by its cost class, which `MEASURED_COSTS` in
`validations/validation_targets.py` gives for every target measured at more than ten
seconds. A run of every public target on 2026-10-01 took 130 seconds of wall time over
1,020 seconds of target time.

## What none of this covers

- Nothing checks a figure against a picture. A figure check asserts the structure of the
  message and the page, and a person reads the diagram.

## See also

- [[Invariants]] — what the checks are protecting
- [[Notebooks]] — which notebooks run, and what each shows
- [[Website Notebooks]] — the nine notebooks whose claims stand in a validator beside each
- [[Agent Display]] — deterministic, and what to compare when a listing moves
