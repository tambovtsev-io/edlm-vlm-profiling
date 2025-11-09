from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FlopsEstimate:
    total_tflops: float
    note: str


class FlopsEstimator:
    def __init__(self) -> None:
        pass

    def estimate(
        self, num_tokens: int, image_resolution: int, model_param_count_b: float
    ) -> FlopsEstimate:
        # Very rough estimate: transformer FLOPs ~ 6 * Nparams * NToks (AdaPT heuristic)
        # Vision encoder cost approximated as quadratic with patches (not exact)
        transformer_tflops = 6.0 * model_param_count_b * 1e12 * num_tokens / 1e12
        vision_cost_tflops = 0.05 * (image_resolution / 224) ** 2
        total = transformer_tflops + vision_cost_tflops
        return FlopsEstimate(
            total_tflops=total,
            note="Heuristic estimate; use kernel profilers for accuracy",
        )
