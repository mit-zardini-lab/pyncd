# Claude Opus 5.5 (1M context), effort 40.
'''Rewriting the page of each website notebook with the tsncd bundle built last.

    python notebooks/website/rewrite_website_pages.py
    python notebooks/website/rewrite_website_pages.py notebooks/website/classic/Mixtral8x7B.ipynb

A page holds the bundle it was written with, so every page is written again after
`npm run build` in the tsncd checkout. Each notebook runs with every cell under the
mode its settings declare, so the page cell writes its page into the folder that
`notebooks/website/website_output.py` names for the group of the notebook, and the
notebook file is not written. With no argument every notebook in the folder of each
group is run: `notebooks/website/tutorial/`, `classic/` and `modern/`.
'''
from __future__ import annotations

import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import notebooks.execute_notebook as execute_notebook  # noqa: E402
import notebooks.website.website_output as website_output  # noqa: E402

EVERY_CELL_UNDER_ITS_OWN_MODE = ''
SECONDS_ALLOWED_PER_CELL = 1800


def website_notebooks() -> list[pathlib.Path]:
    notebook_folders = [REPOSITORY_ROOT / website_output.notebook_folder_of(page_folder)
                        for page_folder in website_output.PAGE_FOLDERS]
    return sorted(notebook for folder in notebook_folders
                  for notebook in folder.glob('*.ipynb'))


def rewrite_pages(notebooks: list[pathlib.Path]) -> list[pathlib.Path]:
    '''Run each of `notebooks` under its own modes, and return those that failed.'''
    kernel = execute_notebook.kernelspec_for(sys.executable)
    failed = []
    for notebook in notebooks:
        print(f'=== {notebook.relative_to(REPOSITORY_ROOT)}', flush=True)
        if execute_notebook.execute(
                notebook, EVERY_CELL_UNDER_ITS_OWN_MODE, kernel,
                SECONDS_ALLOWED_PER_CELL) != 0:
            failed.append(notebook)
    return failed


def main(argv: list[str]) -> int:
    notebooks = ([pathlib.Path(argument).resolve() for argument in argv]
                 or website_notebooks())
    failed = rewrite_pages(notebooks)
    print(f'{len(notebooks) - len(failed)} of {len(notebooks)} pages rewritten')
    for notebook in failed:
        print(f'failed: {notebook.relative_to(REPOSITORY_ROOT)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
