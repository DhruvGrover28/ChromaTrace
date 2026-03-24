from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    center: Tuple[int, int]
    last_seen: int
    color_label: str
    trail: List[Tuple[int, int]]


@dataclass
class TrackerConfig:
    max_distance: float = 60.0
    max_missed_frames: int = 12
    trail_length: int = 24


def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class SimpleTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Tuple[Tuple[int, int], str]], frame_index: int) -> Dict[int, Track]:
        updated_tracks: Dict[int, Track] = {}
        used_ids = set()

        for center, color_label in detections:
            best_id = None
            best_distance = self.config.max_distance

            for track_id, track in self.tracks.items():
                if track_id in used_ids:
                    continue
                dist = _distance(center, track.center)
                if dist < best_distance:
                    best_distance = dist
                    best_id = track_id

            if best_id is None:
                track_id = self.next_id
                self.next_id += 1
                updated_tracks[track_id] = Track(
                    track_id=track_id,
                    center=center,
                    last_seen=frame_index,
                    color_label=color_label,
                    trail=[center],
                )
            else:
                track = self.tracks[best_id]
                track.center = center
                track.last_seen = frame_index
                track.color_label = color_label
                track.trail.append(center)
                track.trail = track.trail[-self.config.trail_length :]
                updated_tracks[best_id] = track
                used_ids.add(best_id)

        for track_id, track in self.tracks.items():
            if track_id in updated_tracks:
                continue
            if frame_index - track.last_seen <= self.config.max_missed_frames:
                updated_tracks[track_id] = track

        self.tracks = updated_tracks
        return self.tracks


def draw_tracks(frame: np.ndarray, tracks: Dict[int, Track]) -> None:
    for track in tracks.values():
        for idx in range(1, len(track.trail)):
            cv2.line(frame, track.trail[idx - 1], track.trail[idx], (60, 170, 255), 2)

        cx, cy = track.center
        cv2.circle(frame, (cx, cy), 8, (0, 255, 200), -1)
        cv2.putText(
            frame,
            f"ID {track.track_id} - {track.color_label}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
