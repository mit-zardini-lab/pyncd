import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import display.Box as Box
import utilities.utilities as util
import term_utilities.term_utilities as tutil
import utilities.justification as js
from typing import Literal, Iterable, Callable

import display.Color as cl
import display.node_category as dcat

import graphs.data_structure.Hypergraph as hg

def display_graph(target: hg.Hypergraph, offset: int = 2) -> Box.Box:
    header = Box.Horizontal(
        (*(dcat.display_uterm(dom) for dom in target.dom),
         Box.TextBox(' => '),
         *(dcat.display_uterm(cod) for cod in target.cod)
         ),
         justify_mode=js.JustifyMode.LEFT
    )
    match target:
        case hg.HypergraphRoot():
            core = dcat.display_category(target.wraps)
        case hg.HypergraphBlock():
            core = Box.Padded(
                display_graph(target.body, offset=offset)
            )
        case hg.Multigraph():
            core = Box.Horizontal(
                (Box.Fill('|', min_width=offset),
                 Box.Vertical.from_iter(
                     display_graph(subgraph, offset=offset)
                     for subgraph in target.subgraphs()
                 )
                )
            )
        case _:
            raise NotImplementedError(f'Unknown graph type: {type(target)}')
    return Box.Vertical((header, core))

