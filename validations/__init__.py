'''Running the repository's validations in parallel, and only the ones a change reaches.

Written by Claude Opus 5 (1M context), effort high.

Every feature keeps its own `validate_*.py` beside the code it checks, and every
notebook asserts what it claims with the diagrams off. A validation therefore
exercises one feature and the features it imports, and a change to one feature
leaves most of them untouched. This package finds the validations, works out which
of them import a modified file, and runs those as concurrent subprocesses.

    validation_targets       every validation, where it is located and what it costs
    module_import_graph      which repository files each `.py` file and notebook imports
    list_modified_files      the modified paths, read from git
    select_affected_targets  the targets that import a modified file
    run_validation_targets   the concurrent runner over subprocesses
    run_validations          the command line

`obsidian/06-practice/Validation.md` describes the validators themselves and the
dependency selection this package performs.
'''
