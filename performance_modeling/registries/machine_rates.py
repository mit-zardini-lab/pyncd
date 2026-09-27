'''The published rates of a GPU, each with its source.

Written by Claude Fable 5.1, effort 80.

A notebook reads the figures of a GPU from here rather than writing them by hand, and
builds from them the `collective_cost.MachineRates` used to cost a pass, as
`notebooks/classic/cached_mixtral_8x7b.py` does for an H100:

    H100 = machine_rates.H100_SXM5
    MACHINE = collective_cost.MachineRates(
        matrix_operations_per_second=H100.matrix_operations_per_second, ...)

Every rate is the whole part's, and `sources` states where each figure was read.
'''

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RateSource:
    '''Where one figure of a `GPU` comes from.'''
    rate: str
    whole_gpu: str
    source: str


@dataclasses.dataclass(frozen=True)
class GPU:
    '''One GPU part: its arithmetic rates, its memory and network bandwidths, the
    latency of one step of a collective, its memory, the bytes of one element, and the
    source of each figure.'''
    name: str
    matrix_operations_per_second: float
    scalar_operations_per_second: float
    memory_bytes_per_second: float
    network_bytes_per_second: float
    collective_latency_seconds: float
    memory_bytes_per_processor: float
    bytes_per_element: int
    sources: tuple[RateSource, ...]


H100_SXM5 = GPU(
    name='H100 SXM5',
    matrix_operations_per_second=989e12,
    scalar_operations_per_second=67e12,
    memory_bytes_per_second=3.35e12,
    network_bytes_per_second=450e9,
    collective_latency_seconds=1e-6,
    memory_bytes_per_processor=80 * 10**9,
    bytes_per_element=2,
    sources=(
        RateSource('matrix operations per second', '989 TFLOPS',
                   'H100 SXM5 datasheet, BF16 dense tensor-core throughput'),
        RateSource('scalar operations per second', '67 TFLOPS',
                   'H100 SXM5 datasheet, FP32 throughput of the CUDA cores'),
        RateSource('memory bytes per second', '3.35 TB/s',
                   'HBM3 bandwidth, FlashAttention-3 Table 1'),
        RateSource('network bytes per second', '450 GB/s',
                   'fourth-generation NVLink, 900 GB/s in both directions together'),
        RateSource('collective latency', '1 us per ring step', 'assumed'),
        RateSource('memory per processor', '80 GB',
                   'HBM3 capacity, FlashAttention-3 Table 1')))
