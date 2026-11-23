"""Ultralytics YOLO inference wrapper with automatic device selection."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from ultralytics import YOLO


class Detector:
    """Load a YOLO model and run inference on individual video frames."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self.model_path = model_path
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        self.model = YOLO(model_path)
        self.names: dict[int, str] = {
            int(class_id): str(name) for class_id, name in self.model.names.items()
        }

    def resolve_classes(self, classes: Sequence[str] | None) -> list[int] | None:
        """Convert class names or numeric IDs to YOLO class IDs."""
        if not classes:
            return None

        names_by_lower = {name.casefold(): class_id for class_id, name in self.names.items()}
        resolved: list[int] = []
        for value in classes:
            token = str(value).strip()
            if not token:
                continue
            if token.isdecimal():
                class_id = int(token)
                if class_id not in self.names:
                    raise ValueError(f"Class ID {class_id} is not present in model {self.model_path!r}.")
            else:
                try:
                    class_id = names_by_lower[token.casefold()]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown class {token!r}. Model classes include: {', '.join(self.names.values())}."
                    ) from exc
            if class_id not in resolved:
                resolved.append(class_id)
        return resolved or None

    def predict(
        self,
        frame: Any,
        confidence: float,
        image_size: int,
        classes: Sequence[int] | None = None,
    ) -> Any:
        """Run detector inference on one BGR frame and return Ultralytics Results."""
        options = {
            "source": frame,
            "conf": confidence,
            "imgsz": image_size,
            "classes": list(classes) if classes is not None else None,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            # Ultralytics replaced its deprecated ``half`` argument with quantize.
            options["quantize"] = 16
        result = self.model.predict(**options)
        return result[0]
