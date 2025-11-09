from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None


@dataclass
class EnergySample:
    timestamp: float
    power_watts: float


class EnergyMeter:
    def __init__(self):
        self.enabled = False
        self.handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.enabled = True
            except Exception:
                self.enabled = False

    def read_power_watts(self) -> Optional[float]:
        if not self.enabled:
            return None
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # type: ignore
            return mw / 1000.0
        except Exception:
            return None

    def integrate_energy_joules(self, fn, warmup: int = 1, iters: int = 1) -> float:
        # Simple rectangular integration over polling intervals
        if not self.enabled:
            # Run anyway to keep behavior consistent
            for _ in range(warmup):
                fn()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            t1 = time.perf_counter()
            # Unknown energy; return 0 as placeholder
            return 0.0

        for _ in range(warmup):
            fn()
        energy_j = 0.0
        last_t = time.perf_counter()
        for _ in range(iters):
            # sample before
            p0 = self.read_power_watts() or 0.0
            fn()
            t1 = time.perf_counter()
            p1 = self.read_power_watts() or 0.0
            dt = max(0.0, t1 - last_t)
            p = 0.5 * (p0 + p1)
            energy_j += p * dt
            last_t = t1
        return energy_j
