'''Exposing a pass's tape as ordinary inputs and outputs.

A pass with tape operations is not a function. A `Grab` reads a side channel and
a `Drop` writes one, so the pass cannot be compiled, kernelized or costed as a
morphism on its own. The tape is positional state rather than control flow, and
neither pass writes a slot twice, so a `Drop` is an extra output and a `Grab` an
extra input, in slot order. `detape` rewrites the graph that way and hands back
the slot names beside the morphism, so a caller can carry the residuals from the
forward pass's extra outputs to the backward pass's extra inputs by name, which
is what a training loop does with its saved activations.

`para/validate_backward.py` compiles both passes this way.
'''
from __future__ import annotations

import data_structure.Category as cat
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import para.data_structure.Para as para
import term_utilities.term_utilities as tutil


def slot_order(root: hg.HypergraphRoot) -> tuple[int, int, str]:
    '''Residual slots `s{n}` by number, then parameter and gradient slots by
    name, which `show_grabbed_parameters` names after the parameter.'''
    name = root.wraps.tape.uid._name.to_bodies()
    if name.startswith('s') and name[1:].isdigit():
        return (0, int(name[1:]), '')
    return (1, 0, name)


def detape[L, M: cat.Morphism](
    morphism: cat.ProdCategory[L, M],
) -> tuple[cat.Morphism, list[tuple[str, bool]]]:
    '''The pass as a pure function, with each `Drop` a trailing output and each
    `Grab` a trailing input, ordered by slot. The tape comes back as
    `(slot name, is_grab)` pairs, in that order.'''
    graph = hg.Multigraph.from_morphism(morphism)
    tape = sorted(
        (sub for sub in graph._subgraphs
         if isinstance(sub, hg.HypergraphRoot)
         and isinstance(sub.wraps, para.ParaMorphism)),
        key=slot_order)
    kept = tuple(sub for sub in graph._subgraphs if sub not in tape)
    dom, cod = graph.dom, graph.cod
    for root in tape:
        match root.wraps:
            case para.Drop():
                cod = (*cod, root.dom[0])
            case para.Grab():
                dom = (*dom, root.cod[0])
    result = h2m.hypergraph_to_morphism(hg.Multigraph.template(dom, cod, kept))
    if any(True for _ in tutil.type_search(para.ParaMorphism, result)):
        raise ValueError(
            'tape operations below the top level, which detape cannot reach')
    return result, [(root.wraps.tape.uid._name.to_bodies(),
                     isinstance(root.wraps, para.Grab)) for root in tape]
