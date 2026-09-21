'''Printing a morphism as an SSA listing from a notebook.

`agent_display.listing` prints a legend before the listing. The legend is the
same on every cell, so a notebook that prints many listings prints it many
times. `print_listing` calls `agent_display.listing_without_legend` instead.

    import notebooks.display.notebook_listings as notebook_listings

    notebook_listings.print_listing(morphism, 'forward')

`obsidian/05-backends/Agent Display.md` describes what the listing shows.
'''

from __future__ import annotations

import agent_display as agent_display
import graphs.processing.Hypergraph2Morphism as h2m
import data_structure.Category as cat
import notebooks.display.tape_naming as tape_naming
import notebooks.display.tape_presentation as tape_presentation

TapePresentation = tape_presentation.TapePresentation
TapeNaming = tape_naming.TapeNaming


def listing_without_legend(
    target, recycle: bool = True,
    tape: TapePresentation = TapePresentation.BOXED,
    naming: TapeNaming = TapeNaming.NAMED,
) -> str:
    '''The SSA listing of `target`, in the tape presentation asked for, its
    tape members labelled as `naming` asks.

    `recycle=False` lists a hand-built morphism as it stands, rather than
    normalising it by converting to a graph and back. `to_para_wrap` converts
    through the graph on its own, so an absorbed listing is recycled either way.
    '''
    if recycle and tape is TapePresentation.BOXED:
        target = h2m.recycle(target)
    return agent_display.listing_without_legend(
        tape_naming.present(tape_presentation.present(target, tape), naming))


def print_listing(target, title: str | None = None, recycle: bool = True,
                  tape: TapePresentation = TapePresentation.BOXED,
                  naming: TapeNaming = TapeNaming.NAMED) -> None:
    '''Print `target` as an SSA listing, without the legend.'''
    if title:
        print(f'--- {title} ---')
    body = listing_without_legend(target, recycle, tape, naming).split('\n')
    print('\n'.join(line for line in body if line.strip()))
    print()


def operation_counts(target: cat.Morphism) -> str:
    '''The operation-count line of `agent_display.summary`, on its own.'''
    for line in agent_display.summary(target).splitlines():
        if line.strip().startswith('ops'):
            return line.strip()
    return ''
