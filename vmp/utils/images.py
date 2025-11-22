import base64
import io

import cv2
import numpy as np
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
