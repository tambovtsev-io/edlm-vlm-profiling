from __future__ import annotations

import random
from typing import Dict


def build_prompt(example: Dict, target_length: int) -> str:
    base = ""
    if example.get("dataset") == "scienceqa":
        q = example.get("question", "")
        hint = example.get("hint", "")
        choices = example.get("choices")
        if choices:
            choices_text = " ".join(f"({i}) {c}" for i, c in enumerate(choices))
        else:
            choices_text = ""
        base = f"Question: {q}\n{choices_text}\nHint: {hint}"
    elif example.get("dataset") == "textvqa":
        q = example.get("question", "")
        base = f"Read the image text and answer concisely. Question: {q}"
    elif example.get("dataset") == "coco":
        base = "Describe the image in detail."
    else:
        base = "Answer based on the image and text."

    # Pad with filler tokens to approximate prompt length in tokens (~0.75 chars/token heuristic)
    approx_chars = int(max(0, target_length * 4 - len(base)))
    filler = (" context." * 1000)[:approx_chars]
    return (base + filler).strip()
