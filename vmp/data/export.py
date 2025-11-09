from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

from PIL import Image

ImageLike = Union[None, Image.Image, str]


def _save_image_to_jpeg(image_obj: ImageLike, output_path: Path) -> Optional[str]:
    try:
        if image_obj is None:
            return None
        pil_image: Image.Image
        if isinstance(image_obj, Image.Image):
            pil_image = image_obj
        elif isinstance(image_obj, str):
            pil_image = Image.open(image_obj)
        else:
            to_pil = getattr(image_obj, "to_pil", None)
            if callable(to_pil):
                pil_image = cast(Image.Image, to_pil())
            else:
                return None
        pil_image.convert("RGB").save(output_path, format="JPEG", quality=92)
        return str(output_path)
    except Exception:
        return None


def export_records(
    dataset_key: str, records: List[Dict], base_dir: Union[str, Path]
) -> Path:
    base_path = Path(base_dir)
    dataset_dir = base_path / dataset_key
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for index, example in enumerate(records):
            relative_image_path: Optional[str] = None
            output_image_path = images_dir / f"{index:06d}.jpg"
            saved_path = _save_image_to_jpeg(
                example.get("image", None), output_image_path
            )
            if saved_path is not None:
                relative_image_path = str(Path("images") / output_image_path.name)

            row: Dict = {
                "dataset": example.get("dataset", dataset_key),
                "id": index,
                "image": relative_image_path,
            }
            if dataset_key == "scienceqa":
                row.update(
                    {
                        "question": example.get("question", ""),
                        "choices": example.get("choices", []),
                        "hint": example.get("hint", ""),
                        "answer": example.get("answer", ""),
                    }
                )
            elif dataset_key == "textvqa":
                row.update(
                    {
                        "question": example.get("question", ""),
                        "answers": example.get("answers", []),
                    }
                )
            elif dataset_key == "coco":
                row.update({"captions": example.get("captions", [])})

            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return manifest_path
