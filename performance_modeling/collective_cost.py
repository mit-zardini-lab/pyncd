"""The rates of one machine, used to cost a pass.

`MachineRates` holds the matrix and scalar arithmetic rates, the memory and network
bandwidths, the latency of one step of a collective, the memory of one GPU and the
bytes of one element. `caching.algebra.cost_cache_placements` reads the matrix rate,
the memory bandwidth and the bytes per element to cost a pass, and
`registries/machine_rates.py` holds the published figures of an H100.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class MachineRates:
    matrix_operations_per_second: float
    scalar_operations_per_second: float
    memory_bytes_per_second: float
    network_bytes_per_second: float
    collective_latency_seconds: float
    memory_bytes_per_processor: int
    bytes_per_element: int = 2

    def __post_init__(self) -> None:
        rates = (self.matrix_operations_per_second, self.scalar_operations_per_second,
                 self.memory_bytes_per_second, self.network_bytes_per_second)
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("Machine rates must be finite and positive")
        latency = self.collective_latency_seconds
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"The latency must be finite and nonnegative, not {latency}")
        if self.memory_bytes_per_processor < 1 or self.bytes_per_element < 1:
            raise ValueError(
                f"{self.memory_bytes_per_processor} bytes per processor of "
                f"{self.bytes_per_element} bytes per element")
