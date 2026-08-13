import data_structure.Category as category
import display.Box as Box
import display.node_category as cat
import display.display_numeric as nm
import display.display_graph as graph
import graphs.Hypergraph as hg


__all__ = [
    'Box',
    'cat',
    'nm',
    'graph'
]

def print_category(target: category.BroadcastedCategory[category.Datatype, category.Axis]) -> None:
    text = cat.display_category(target).render()
    print(text)

def print_graph(target: hg.Hypergraph) -> None:

    match target:
        case hg.HypergraphRoot():
            print_category(target.wraps)
        case hg.HypergraphBlock():
            print('--- START BLOCK ---')
            print_graph(target.body)
            print('---  END BLOCK  ---')
        case hg.AuxiliaryGraph():
            for subgraph in target.subgraphs():
                print('##############')
                print_graph(subgraph)
        