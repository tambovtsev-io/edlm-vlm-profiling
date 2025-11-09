from __future__ import annotations

import os
from typing import Dict, List

from datasets import load_dataset


def load_coco_subset(name: str, split: str = "val") -> List[Dict]:
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        ds = load_dataset(name, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(name, split=split)

    records: List[Dict] = []
    for item in ds:
        image = item.get("image", None)
        question = item.get("question", "")
        answer = item.get("answer", "")
        records.append(
            {
                "image": image,
                "question": question,
                "answer": answer,
                "dataset": "coco",
            }
        )
    return records
