from __future__ import annotations

import os
from typing import Dict, List

from datasets import load_dataset


def load_textvqa_subset(name: str, split: str = "validation") -> List[Dict]:
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        ds = load_dataset(name, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(name, split=split)

    records: List[Dict] = []
    for item in ds:
        question = item.get("question", "")
        image = item.get("image", None)
        answers = item.get("answers", [])
        records.append(
            {
                "image": image,
                "question": question,
                "answers": answers,
                "dataset": "textvqa",
            }
        )
    return records
