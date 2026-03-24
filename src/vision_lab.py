from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from colors import build_color_masks, cleanup_mask, get_default_ranges
from tracker import SimpleTracker, TrackerConfig, draw_tracks


@dataclass
class AppConfig:
    camera_index: int = 0
    width: int = 960
    height: int = 540
    blur: int = 5
    min_area: int = 1200
    record: bool = False
    output_path: str = "output.mp4"


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="VisionLab: color + shape tracker")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=960, help="Frame width")
    parser.add_argument("--height", type=int, default=540, help="Frame height")
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size")
    parser.add_argument("--min-area", type=int, default=1200, help="Minimum contour area")
    parser.add_argument("--record", action="store_true", help="Record output to mp4")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video file")
    args = parser.parse_args()
    return AppConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        blur=max(3, args.blur | 1),
        min_area=args.min_area,
        record=args.record,
        output_path=args.output,
    )


def detect_shapes(mask: np.ndarray, min_area: int) -> List[Tuple[Tuple[int, int], str, np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Tuple[Tuple[int, int], str, np.ndarray]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        shape = "Circle"

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            shape = "Square" if 0.85 <= ratio <= 1.15 else "Rectangle"
        elif len(approx) > 6:
            shape = "Circle"

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        detections.append(((cx, cy), shape, cnt))

    return detections


def draw_detection(frame: np.ndarray, contour: np.ndarray, label: str, color: Tuple[int, int, int]) -> None:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    config = parse_args()

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try --camera 1 or 2.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    tracker = SimpleTracker(TrackerConfig())
    ranges = get_default_ranges()

    video_writer = None
    if config.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(config.output_path, fourcc, 30.0, (config.width, config.height))

    prev_time = time.time()
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (config.width, config.height))
        blurred = cv2.GaussianBlur(frame, (config.blur, config.blur), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        masks = build_color_masks(hsv)
        detections: List[Tuple[Tuple[int, int], str]] = []

        for key, mask in masks.items():
            if key not in ranges and key != "red":
                continue

            cleaned = cleanup_mask(mask)
            label = "Red" if key == "red" else ranges[key].label
            shape_detections = detect_shapes(cleaned, config.min_area)

            for center, shape, contour in shape_detections:
                detections.append((center, label))
                draw_detection(frame, contour, f"{label} {shape}", (0, 255, 255))

        tracks = tracker.update(detections, frame_index)
        draw_tracks(frame, tracks)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        frame_index += 1

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Keys: [q] quit  [r] toggle record",
            (12, config.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("VisionLab - Color + Shape Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    config.output_path, fourcc, 30.0, (config.width, config.height)
                )
            else:
                video_writer.release()
                video_writer = None

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
