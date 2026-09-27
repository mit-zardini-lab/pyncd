# Claude Opus 5.5 (1M context), effort 40.
'''The folders the pages of the website notebooks are written into.

Each group of website notebooks writes its pages into the folder of the same name
under `notebooks/website/output/`, so the folders of pages mirror the folders of
notebooks. The page of `notebooks/website/classic/Mixtral8x7B.ipynb` is
`notebooks/website/output/classic/Mixtral8x7B/index.html`, and
`notebooks/website/output/classic/Mixtral8x7B.html` redirects to its folder. A
notebook passes the folder of its group as `page_directory` in the settings of its
page. The paths are relative to the root of the repository, where the kernel of
every notebook starts, as the default `page_directory` of
`notebook_diagrams.DiagramSettings` is.
'''
import pathlib

WEBSITE_NOTEBOOKS = pathlib.Path('notebooks/website')
WEBSITE_OUTPUT = WEBSITE_NOTEBOOKS / 'output'
TUTORIAL_PAGES = WEBSITE_OUTPUT / 'tutorial'
CLASSIC_PAGES = WEBSITE_OUTPUT / 'classic'
MODERN_PAGES = WEBSITE_OUTPUT / 'modern'
PAGE_FOLDERS = (TUTORIAL_PAGES, CLASSIC_PAGES, MODERN_PAGES)


def notebook_folder_of(page_folder: pathlib.Path) -> pathlib.Path:
    '''The folder of the notebooks whose pages are written into `page_folder`.'''
    return WEBSITE_NOTEBOOKS / page_folder.relative_to(WEBSITE_OUTPUT)
