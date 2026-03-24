from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class ColorRange:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]
    label: str


def get_default_ranges() -> Dict[str, ColorRange]:
    # HSV ranges tuned for common indoor lighting.
    return {
        "red1": ColorRange((0, 120, 70), (10, 255, 255), "Red"),
        "red2": ColorRange((170, 120, 70), (180, 255, 255), "Red"),
        "green": ColorRange((36, 60, 60), (85, 255, 255), "Green"),
        "blue": ColorRange((90, 60, 60), (130, 255, 255), "Blue"),
        "yellow": ColorRange((15, 80, 80), (35, 255, 255), "Yellow"),
    }


def build_color_masks(hsv_frame: np.ndarray) -> Dict[str, np.ndarray]:
    ranges = get_default_ranges()
    masks: Dict[str, np.ndarray] = {}

    for key, color_range in ranges.items():
        lower = np.array(color_range.lower, dtype=np.uint8)
        upper = np.array(color_range.upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower, upper)
        masks[key] = mask

    # Combine red ranges into one mask for detection.
    if "red1" in masks and "red2" in masks:
        masks["red"] = cv2.bitwise_or(masks["red1"], masks["red2"])

    return masks


def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask
