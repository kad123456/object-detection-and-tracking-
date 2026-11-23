"""Adapters for Ultralytics' built-in ByteTrack and BoT-SORT trackers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml


@dataclass(frozen=True)
class TrackObservation:
    """One tracked object in a frame, using pixel-space XYXY coordinates."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        """Return the box center in pixel coordinates."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class ObjectTracker:
    """Track per-frame detections with Ultralytics ByteTrack or BoT-SORT."""

    def __init__(self, name: str, class_names: dict[int, str]) -> None:
        normalized = name.casefold()
        if normalized not in {"bytetrack", "botsort"}:
            raise ValueError("tracker must be 'bytetrack' or 'botsort'.")

        config_path = check_yaml(f"{normalized}.yaml")
        settings = YAML.load(config_path)
        args = argparse.Namespace(**settings)
        tracker_type = BYTETracker if normalized == "bytetrack" else BOTSORT
        # These built-in tracker classes read tracker policy from the YAML Namespace.
        # Their public constructor currently accepts only that Namespace.
        self.tracker = tracker_type(args=args)
        self.name = normalized
        self.class_names = class_names

    def update(self, result: Any, frame: np.ndarray) -> list[TrackObservation]:
        """Update tracker state from a YOLO Results object and current frame."""
        boxes = result.boxes
        if boxes is None:
            return []

        tracks = self.tracker.update(boxes.cpu().numpy(), frame)
        if tracks is None or len(tracks) == 0:
            return []

        observations: list[TrackObservation] = []
        for row in tracks:
            x1, y1, x2, y2, track_id, confidence, class_id = row[:7]
            class_id_int = int(class_id)
            observations.append(
                TrackObservation(
                    track_id=int(track_id),
                    class_id=class_id_int,
                    class_name=self.class_names.get(class_id_int, str(class_id_int)),
                    confidence=float(confidence),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return observations
