from __future__ import annotations

import os
from typing import Dict, Iterable, List

from datasets import load_dataset


def load_scienceqa_subset(name: str, split: str = "ScienceQA-IMG") -> List[Dict]:
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        ds = load_dataset(name, split, cache_dir=cache_dir)
    else:
        ds = load_dataset(name, split)

    records: List[Dict] = []
    for ds_type in ds:
        for item in ds[ds_type]:
            # Variants in lmms-lab/ScienceQA
            question = item.get("question", "")
            choices = item.get("choices", None)
            image = item.get("image", None)
            answer = item.get("answer", "")
            records.append(
                {
                    "image": image,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "dataset": "scienceqa",
                }
            )
    return records
