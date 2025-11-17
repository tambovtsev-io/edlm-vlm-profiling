import base64
import io
import re
from typing import List, Optional

import cv2
import evaluate
import numpy as np
import pandas as pd
from PIL import Image


def resize_image(
    image: np.ndarray, dims: tuple[int, int], interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """Ресайз изображения
    Args:
        image (np.ndarray): Изображение
        dims (tuple[int, int]): Размер для ресайза в формате (w, h)
        interpolation (int, cv2.INTER_CUBIC): Индекс для метода интерполяции из opencv
    Returns:
        np.ndarray: Изображение
    """
    resized_image = cv2.resize(
        image,
        dsize=(dims[1], dims[0]),
        interpolation=interpolation,
    )
    return resized_image


def image_to_data_url(image: np.ndarray) -> str:
    """Преобразование изображения в data URL
    Args:
        image (np.ndarray): Изображение в виде numpy массива (H, W, C)
    Returns:
        str: data URL
    """
    pil_img = Image.fromarray(image)
    fmt = pil_img.format or "JPEG"
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


_NUM_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}

_punct_re1 = re.compile(r"[\"#$%&()*+/:;<=>@\[\\\]^_`{|}~]")
# точки и запятые удаляем, если они НЕ между цифрами (оставляем десятичные числа вида 3.14, 1,000)
_punct_dot_comma = re.compile(r"(?<!\d)[\.,]|[\.,](?!\d)")


def _normalize(ans: Optional[str]) -> str:
    """Приближённая нормализация как в VQA: lower, удаление пунктуации/артиклей, маппинг чисел."""
    if not ans:
        return ""
    s = ans.lower().strip()

    # базовая пунктуация
    s = _punct_re1.sub(" ", s)
    s = _punct_dot_comma.sub(" ", s)

    # дефисы -> пробел, апострофы убираем
    s = s.replace("-", " ").replace("'", "")

    # токены: убираем артикли, слова-числа -> цифры
    toks = []
    for t in s.split():
        if t in _ARTICLES:
            continue
        toks.append(_NUM_MAP.get(t, t))

    # финал: один пробел, обрезка краёв
    return " ".join(toks).strip()


def vqa_accuracy(llm_answer: str, answers: List[str]) -> float:
    """
    Acc = min(n/3, 1), где n — число точных совпадений с нормализованными референсами.
    """
    refs = [_normalize(a) for a in answers if a is not None]
    if not refs:
        return 0.0
    pred = _normalize(llm_answer)
    n = sum(1 for a in refs if a == pred)
    return min(1.0, n / 3.0)


def vqa_mean_accuracy(df: pd.DataFrame, col_llm_answer: str, col_answers: str) -> float:
    """
    Средняя точность VQA для датасета
    """
    return df.apply(
        lambda r: vqa_accuracy(r[col_llm_answer], r[col_answers]), axis=1
    ).mean()


def rougeL_coco(df: pd.DataFrame, col_llm_answer: str, col_answers: str) -> float:
    """
    ROUGE-L для COCO
    """
    rouge = evaluate.load("rouge")
    predictions = df[col_llm_answer].tolist()
    references = df[col_answers].tolist()
    results = rouge.compute(predictions=predictions, references=references)
    return results["rougeL"]


def accuracy_scienceqa(
    df: pd.DataFrame,
    col_llm_answer: str,
    col_answers: str,
) -> float:
    """
    Точность для ScienceQA при условии, что предсказания модели — это 0-based индексы без лишних слов.
    Возвращает accuracy.
    """
    pred = df[col_llm_answer]

    gold = df[col_answers]
    gold = gold.apply(lambda x: 0 if isinstance(x, list) and len(x) == 0 else x)
    gold = gold.astype(int) + 1
    gold = gold.astype(str)

    choices = df["choices"]
    total = 0
    for p, g, c in zip(pred, gold, choices):
        gold_text = c[int(g) - 1]
        if str(g) in str(p) or str(gold_text) in str(p):
            total += 1

    return total / len(pred)


def compute_metrics(
    dataset_name: str, df: pd.DataFrame, col_llm_answer: str, col_answers: str
) -> float:
    """
    Вычисление метрик для датасета
    """
    if dataset_name == "textvqa":
        return vqa_mean_accuracy(df, col_llm_answer, col_answers)
    elif dataset_name == "coco":
        return rougeL_coco(df, col_llm_answer, col_answers)
    elif dataset_name == "scienceqa":
        return accuracy_scienceqa(df, col_llm_answer, col_answers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
