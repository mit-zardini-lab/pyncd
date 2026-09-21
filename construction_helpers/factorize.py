'''No implementation.

Factoring a composition back into a product of independent branches is done by
converting the morphism to a hypergraph and back, with
`graphs.processing.Hypergraph2Morphism.recycle`.
'''
from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import construction_helpers.product as chp
import construction_helpers.lift as chl
import utilities.utilities as util
from enum import Enum
